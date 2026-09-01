"""
TDX quote parsing, crypto operations, and server helper functions.
"""

import asyncio
import secrets
import base64
import json
import tempfile
from datetime import datetime
from typing import Dict, List, Optional
from sqlalchemy import select, update
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.sql import func
from sqlalchemy.ext.asyncio import AsyncSession
from urllib.parse import unquote
from aiohttp import ClientResponse
from cryptography.fernet import Fernet
from fastapi import Depends, Header, HTTPException, Request, status
from api.database import get_db_session
from api.metagraph import get_miner_by_hotkey
from loguru import logger
import time
from dcap_qvl import get_collateral, verify, Quote, PHALA_PCCS_URL
from api.config import settings, TeeMeasurementConfig
from cryptography import x509
from cryptography.x509 import Certificate
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import padding, ec
from cryptography.hazmat.backends import default_backend
from cryptography.exceptions import InvalidSignature
from api.server.exceptions import (
    AttestationError,
    GpuEvidenceError,
    InvalidClientCertError,
    InvalidGpuEvidenceError,
    InvalidQuoteError,
    InvalidSignatureError,
    InvalidTdxConfiguration,
    MeasurementMismatchError,
    NoClientCertError,
    NoServerCertError,
    NonceError,
    UnauthorizedError,
)
from api.server.quote import TdxQuote, TdxVerificationResult, resolve_tdx_tcb_status
import hashlib

from api.server.schemas import (
    HostProfileRecord,
    Server,
    VmCacheConfig,
    LuksVolumeRotation,
    HotkeyAuth,
    HostProfile,
)
from api.constants import (
    HostProfileStatus,
    MIN_ROOT_ROTATION_VERSION,
    ATTESTATION_PROXY_AUTH_HEADER,
    CVM_PROXY_AUTH_HEADER,
    REGISTRY_PROXY_AUTH_HEADER,
    HOTKEY_HEADER,
    SIGNATURE_HEADER,
    NONCE_HEADER,
    RC_ATTESTATION_PURPOSE,
)
from api.util import nonce_is_valid, semcomp, verify_request_signature


def generate_nonce() -> str:
    """Generate a cryptographically secure nonce."""
    return secrets.token_hex(32)


def get_nonce_expiry_seconds(minutes: int = 10) -> int:
    """Get expiry time for a nonce in seconds."""
    return minutes * 60


def _proxy_provenance(request: Request) -> tuple[bool, bool]:
    """Return ``(via_cvm_proxy, via_attestation_proxy)`` for the request.

    Each flag is a constant-time match of the proxy's injected header against its configured
    secret; an unset secret never matches. Two proxies front attestation during the 1.3.x ->
    1.4.0 migration -- the cvm proxy (full-mTLS 1.4.0 VMs) and the attestation proxy (legacy
    1.3.x VMs) -- and callers key their policy off which one stamped the request.
    """

    def _matches(secret: Optional[str], header: str) -> bool:
        if not secret:
            return False
        provided = request.headers.get(header, "")
        return secrets.compare_digest(provided.encode(), secret.encode())

    return (
        _matches(settings.cvm_proxy_secret, CVM_PROXY_AUTH_HEADER),
        _matches(settings.attestation_proxy_secret, ATTESTATION_PROXY_AUTH_HEADER),
    )


def require_mtls_proxy():
    """
    FastAPI dependency for endpoints BOTH VM generations reach through a proxy: the legacy 1.3.x
    attestation proxy or the 1.4.0 cvm proxy.

    Accepts either proxy's secret. Fails closed only when NEITHER is configured (503) -- so a
    deploy that predates provisioning doesn't 503 the fleet -- and 403s a request carrying no
    valid secret. The spoofable Host check was dropped: the secrets are the provenance signal,
    so the proxy DNS names are pure infra config.
    """

    async def _check(request: Request):
        via_cvm, via_att = _proxy_provenance(request)
        if not settings.cvm_proxy_secret and not settings.attestation_proxy_secret:
            logger.error(
                "attestation endpoint rejected: no proxy secret is configured; refusing to "
                f"serve unguarded path={request.url.path}"
            )
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Attestation proxy secret is not configured.",
            )
        if not (via_cvm or via_att):
            logger.warning(
                f"attestation endpoint rejected: proxy secret mismatch path={request.url.path}"
            )
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Request did not arrive via a trusted attestation proxy.",
            )

    return _check


def require_attestation_proxy():
    """
    FastAPI dependency for the 1.3.x-only endpoints (luks/attest): the request must arrive via the
    attestation proxy. The mirror image of ``require_cvm_proxy``.
    """

    async def _check(request: Request):
        _, via_att = _proxy_provenance(request)
        if not settings.attestation_proxy_secret:
            logger.error(
                "attestation-proxy endpoint rejected: ATTESTATION_PROXY_SECRET is not configured "
                f"path={request.url.path}"
            )
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Attestation proxy secret is not configured.",
            )
        if not via_att:
            logger.warning(
                f"attestation-proxy endpoint rejected: not via the attestation proxy "
                f"path={request.url.path}"
            )
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=(
                    "This endpoint serves only legacy VMs, via the attestation proxy. Newer VMs "
                    "provision through POST /servers/{vm_name}/provision."
                ),
            )

    return _check


def require_cvm_proxy():
    """
    FastAPI dependency for full-mTLS 1.4.0 endpoints: the request must arrive via the cvm proxy
    (cvm.chutes.ai), which injects ``X-Cvm-Proxy-Auth`` = CVM_PROXY_SECRET.

    Used by the 1.4.0-only endpoints (provision, provision/confirm). nonce and luks/confirm use
    gate_legacy_attestation, which upgrades to this behaviour per-VM via the version gate.
    Fails closed: 503 if CVM_PROXY_SECRET is unconfigured, 403 if the request is not via the cvm
    proxy.
    """

    async def _check(request: Request):
        via_cvm, _ = _proxy_provenance(request)
        if not settings.cvm_proxy_secret:
            logger.error(
                f"cvm endpoint rejected: CVM_PROXY_SECRET is not configured path={request.url.path}"
            )
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="CVM proxy secret is not configured.",
            )
        if not via_cvm:
            logger.warning(f"cvm endpoint rejected: not via the cvm proxy path={request.url.path}")
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="This endpoint is only accessible via the CVM mTLS proxy.",
            )

    return _check


def gate_legacy_attestation():
    """
    FastAPI dependency for the transitional attestation endpoints (nonce, luks/confirm) that
    1.3.x VMs reach on api.chutes.ai (VALIDATOR_BASE_URL), where no proxy can inject a secret.

    Not a blind allow -- it forces already-upgraded VMs onto the cvm proxy so they can't be
    serviced on the insecure legacy path:

      * request carries the cvm secret (came via the cvm proxy) -> allow (1.4.0, provenance
        proven); no DB lookup.
      * else resolve the VM by caller IP: if it is a known server attested at
        >= ``tee_mtls_min_version``, reject (403) -- a VM this new must use cvm.chutes.ai.
      * else (older / unknown / first boot) -> allow (legacy 1.3.x path).

    Tightens automatically as the fleet upgrades; setting ``tee_mtls_min_version`` to "0.0.0" is
    the kill switch that closes the legacy path entirely (mirrors the registry auth gate).
    """

    async def _check(request: Request, db: AsyncSession = Depends(get_db_session)):
        via_cvm, _ = _proxy_provenance(request)
        if via_cvm:
            return
        # Always set by the HTTP middleware (X-Resolved-IP or the socket peer); read directly.
        client_ip = request.state.client_ip
        result = await db.execute(select(Server).where(Server.ip == client_ip))
        server = result.scalar_one_or_none()
        if (
            server is not None
            and server.version
            and semcomp(server.version, settings.tee_mtls_min_version) >= 0
        ):
            logger.warning(
                f"legacy attestation rejected: server {server.name} at {client_ip} is on "
                f"{server.version} (>= {settings.tee_mtls_min_version}); must use the cvm proxy "
                f"path={request.url.path}"
            )
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="This VM must use the CVM mTLS proxy (cvm.chutes.ai).",
            )

    return _check


def extract_client_cert_hash():
    async def _extract_request_client_cert(request: Request):
        try:
            cert = _get_client_certificate(request)
            return get_public_key_hash(cert)
        except NoClientCertError:
            raise
        except Exception as e:
            # This runs as a FastAPI dependency (before the route body), so a route-level
            # try/except cannot map it -- convert to HTTPException here, at the dependency
            # boundary, exactly like the sibling nonce dependencies. Keep the raw parse
            # error in the log only; return a safe message to the client.
            logger.error(f"Boot attestation failed, no client cert provided:\n{e}")
            raise NoClientCertError(detail=str(e))

    return _extract_request_client_cert


def extract_server_cert_hash(response: ClientResponse):
    try:
        cert = _get_server_certificate(response)
        cert_hash = get_public_key_hash(cert)

        return cert_hash
    except Exception as e:
        logger.error(f"Exception trying to extract cert hash from server cert:\n{e}")
        raise NoServerCertError(detail=str(e))


def _get_server_certificate(response: ClientResponse) -> bytes:
    """
    Extract client certificate from Uvicorn request.
    Simplified for FastAPI-to-FastAPI communication.
    """
    # Get the server certificate from the connection
    # The transport contains the SSL object with peer certificate info
    transport = response.connection.transport
    ssl_object = transport.get_extra_info("ssl_object")

    if ssl_object is None:
        raise ValueError("No SSL connection established")

    # Get the peer certificate in DER format
    cert_der = ssl_object.getpeercert(binary_form=True)

    if cert_der is None:
        raise ValueError("No peer certificate available")

    # Load the DER certificate
    cert = x509.load_der_x509_certificate(cert_der, default_backend())

    return cert


def get_public_key_hash(cert: Certificate) -> str:
    """
    Compute SHA-256 hash of certificate's public key in DER format.
    This matches the bash snippet's logic:
    openssl x509 -pubkey -noout | openssl pkey -pubin -outform der | sha256sum
    """
    # Extract the public key
    public_key = cert.public_key()

    # Serialize public key to DER format (matching openssl pkey -outform der)
    public_key_der = public_key.public_bytes(
        encoding=serialization.Encoding.DER,
        format=serialization.PublicFormat.SubjectPublicKeyInfo,
    )

    # Compute SHA-256 hash
    hash_digest = hashlib.sha256(public_key_der).hexdigest()

    return hash_digest


def validate_user_nonce(nonce: str) -> str:
    """
    Validate that a user-provided nonce is exactly 64 hex characters (32 bytes).

    Args:
        nonce: Nonce string to validate

    Returns:
        Validated nonce string

    Raises:
        NonceError: If nonce is not exactly 64 hex characters
    """
    if not nonce:
        raise NonceError("Nonce is required")

    if len(nonce) != 64:
        raise NonceError(f"Nonce must be exactly 64 hex characters (32 bytes), got {len(nonce)}")

    try:
        # Validate it's valid hex
        int(nonce, 16)
    except ValueError:
        raise NonceError("Nonce must be a valid hexadecimal string")

    return nonce


def cert_to_base64_der(cert: Certificate) -> str:
    """
    Convert a Certificate object to base64-encoded DER format.

    Args:
        cert: Certificate object to convert

    Returns:
        Base64-encoded DER certificate string
    """
    cert_der = cert.public_bytes(serialization.Encoding.DER)
    cert_base64 = base64.b64encode(cert_der).decode("utf-8")
    return cert_base64


def require_proxy_secret(expected_secret: Optional[str], header_name: str):
    """
    Build a FastAPI dependency asserting a request arrived via the proxy that injects
    ``header_name`` with ``expected_secret``.

    When ``expected_secret`` is falsy the guard is a no-op (the check is disabled),
    matching the "optional hardening" rollout posture: the proxy injects an empty header
    and the API does not enforce it until a secret is provisioned on both sides.
    """

    async def _dep(request: Request):
        if not expected_secret:
            return
        provided = request.headers.get(header_name, "")
        if not secrets.compare_digest(provided.encode(), expected_secret.encode()):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Request did not arrive via the expected proxy.",
            )

    return _dep


def require_registry_proxy_secret():
    """
    FastAPI dependency guarding /registry/auth: when REGISTRY_PROXY_SECRET is configured,
    the request must carry the registry proxy's auth header.  Single point that binds the
    setting to its header — see require_proxy_secret for the no-op-when-unset semantics.
    """
    return require_proxy_secret(settings.registry_proxy_secret, REGISTRY_PROXY_AUTH_HEADER)


def _parse_client_cert_header(request: Request) -> Optional[Certificate]:
    """
    Parse the nginx-injected ``X-Client-Cert`` header into a typed ``Certificate``.

    Pure extraction with no trust logic: returns ``None`` when the header is absent or
    empty (a legacy client that presented no mTLS cert), and raises ``HTTPException(400)``
    when a cert is present but cannot be parsed.  Trust that the request actually came
    through the mTLS proxy is enforced separately (``require_attestation_proxy`` /
    ``require_proxy_secret``), so this helper never inspects proxy secrets.
    """
    cert_header = request.headers.get("X-Client-Cert")
    if not cert_header:
        return None
    try:
        # nginx URL-escapes the PEM in $ssl_client_escaped_cert.
        return x509.load_pem_x509_certificate(unquote(cert_header).encode(), default_backend())
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Malformed client certificate.",
        ) from e


def verify_hotkey_auth(auth: HotkeyAuth) -> HotkeyAuth:
    """Verify the caller's sr25519 signature, returning the same auth once it is proven.

    Called once, at the edge -- never from a service. Raises UnauthorizedError (401) on anything
    but success, so a returned value always carries a proven ``miner_hotkey``; callers that have
    no proof to check pass None onward instead of calling this. Does not check nonce freshness --
    each caller's nonce is validated by its endpoint's nonce dependency or by get_current_user.
    """
    if not auth.miner_hotkey or not auth.signature or not auth.nonce:
        logger.warning("Hotkey auth rejected: incomplete hotkey/signature/nonce material.")
        raise UnauthorizedError()
    if not auth.purpose and not auth.body_sha256:
        # Only reachable if the body-hashing middleware did not run for this request.
        logger.warning("Hotkey auth rejected: no request body hash available to verify.")
        raise UnauthorizedError()
    try:
        verify_request_signature(
            auth.miner_hotkey,
            auth.signature,
            auth.nonce,
            payload_hash=auth.body_sha256,
            purpose=auth.purpose,
        )
    except HTTPException as e:
        logger.warning(
            f"Hotkey auth rejected for {auth.miner_hotkey[:12]}...: {e.detail}"
        )
        raise UnauthorizedError() from e

    logger.info(f"Hotkey auth verified for {auth.miner_hotkey[:12]}...")
    return auth


def authenticate_proxy_evidence(
    hotkey: Optional[str],
    nonce: Optional[str],
    signature: Optional[str],
    *,
    host: str,
) -> Optional[HotkeyAuth]:
    """Authenticate the attestation proxy's miner-hotkey proof on an evidence RESPONSE.

    Shared by the two evidence-fetch flows (server registration and chute admission): the proxy
    stamps every response it serves, so both read the same three headers. Returns None when the
    proxy sent no proof (older image, or no seed configured) -- the same "nothing was offered"
    signal ``extract_hotkey_auth`` gives.

    Unlike a request, this proof carries no endpoint nonce dependency to bound replay, so the
    freshness check lives here -- the nonce is the proxy's own timestamp on a response we fetched.
    """
    if not hotkey:
        return None

    if not nonce_is_valid(nonce):
        logger.warning(
            f"Evidence rejected for {host}: the proxy's hotkey proof carries a stale or "
            "malformed nonce."
        )
        raise UnauthorizedError(
            "The attestation proxy's hotkey signature is stale. Check the VM's clock."
        )

    return verify_hotkey_auth(
        HotkeyAuth(
            miner_hotkey=hotkey,
            signature=signature,
            nonce=nonce,
            purpose=RC_ATTESTATION_PURPOSE,
        )
    )


async def _claimed_miner_hotkey(request: Request, header_hotkey: Optional[str]) -> Optional[str]:
    """The hotkey this request claims to act as: the header, else the JSON body's miner_hotkey.

    The body fallback exists for pre-1.4.0 boot attestation, whose initramfs sends the hotkey only
    in the body -- the X-Chutes-Hotkey header on that route arrived with the 1.4.0 signer. Reading
    the body here is safe: the body-hashing middleware has already consumed and cached it.
    TODO: drop the fallback once tee_minimum_boot_version is 1.4.0 and every caller sends the header.
    """
    if header_hotkey:
        return header_hotkey
    if request.method not in ("POST", "PUT", "PATCH"):
        return None
    try:
        body = await request.json()
    except Exception:
        return None
    return body.get("miner_hotkey") if isinstance(body, dict) else None


def deny_blacklisted_miner():
    """Dependency refusing miners that are blacklisted or not registered on the subnet.

    Identifies the caller by X-Chutes-Hotkey, falling back to the body's ``miner_hotkey``. This is
    a claim, not a proof -- pair it with hotkey auth where the identity must be proven. It gates
    on membership, so a caller asserting someone else's hotkey only borrows their standing, never
    their secrets.

    The two refusals are distinct: a hotkey that is not registered on the subnet is 403; one that
    is registered but blacklisted is 401, as is presenting no hotkey at all.
    """

    async def _dep(
        request: Request,
        db: AsyncSession = Depends(get_db_session),
        hotkey: Optional[str] = Header(None, alias=HOTKEY_HEADER),
    ) -> None:
        miner_hotkey = await _claimed_miner_hotkey(request, hotkey)
        if not miner_hotkey:
            logger.warning(f"Rejected: no miner hotkey presented path={request.url.path}")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail=(
                    "No miner hotkey was presented. Send X-Chutes-Hotkey (or miner_hotkey in the "
                    "request body)."
                ),
            )
        node = await get_miner_by_hotkey(miner_hotkey, db)
        if not node:
            logger.warning(
                f"Rejected {miner_hotkey[:12]}... path={request.url.path}: not registered on "
                f"subnet {settings.netuid}"
            )
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Your hotkey is not registered on {settings.netuid}",
            )

        if node.blacklist_reason:
            logger.warning(
                f"MINERBLACKLIST: hotkey={miner_hotkey} path={request.url.path} "
                f"reason={node.blacklist_reason}"
            )
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail=f"Your hotkey has been blacklisted: {node.blacklist_reason}",
            )

    return _dep


def _hotkey_auth_from_request(request: Request, purpose: Optional[str]) -> HotkeyAuth:
    """The caller's auth material, read off the request. Nothing is verified here."""
    return HotkeyAuth(
        miner_hotkey=request.headers.get(HOTKEY_HEADER),
        signature=request.headers.get(SIGNATURE_HEADER),
        nonce=request.headers.get(NONCE_HEADER),
        body_sha256=getattr(request.state, "body_sha256", None),
        purpose=purpose,
    )


def extract_hotkey_auth(*, purpose: Optional[str] = None):
    """Dependency yielding the caller's proven ``HotkeyAuth``, or None when none was offered.

    For routes both VM generations reach. An unsigned request is legitimate from an image whose
    measured initramfs predates the sr25519 signer, so absence is not decided here -- it is left
    to the handler, which knows the attested version once the quote verifies. A signature that IS
    offered is always verified, so a caller can never downgrade itself by sending a bad one.

    Use ``require_hotkey_auth`` instead on routes only 1.4.0+ VMs reach. ``purpose`` must match the
    endpoint's ``get_current_user`` purpose so the reconstructed signing message is identical.
    """

    async def _dep(request: Request) -> Optional[HotkeyAuth]:
        auth = _hotkey_auth_from_request(request, purpose)
        return verify_hotkey_auth(auth) if auth.offered else None

    return _dep


def require_hotkey_auth(*, purpose: Optional[str] = None):
    """Dependency yielding the caller's proven ``HotkeyAuth``, 401ing if none was offered.

    For routes only a signing VM can reach (the /provision pair, behind require_cvm_proxy). The
    requirement is stated here rather than re-derived from the attested version downstream, so it
    holds before the handler runs and cannot be forgotten by one. Handlers that need nothing from
    the result still declare it -- as a bare dependency -- to keep the check at the door.
    """

    async def _dep(request: Request) -> HotkeyAuth:
        auth = _hotkey_auth_from_request(request, purpose)
        if not auth.offered:
            raise UnauthorizedError()
        return verify_hotkey_auth(auth)

    return _dep


def extract_client_cert():
    """
    FastAPI dependency returning the mTLS client ``Certificate``, or raising
    ``NoClientCertError`` when none was presented.  For mTLS-required endpoints.
    """

    async def _dep(request: Request) -> Certificate:
        return _get_client_certificate(request)

    return _dep


def extract_optional_client_cert():
    """
    FastAPI dependency returning the mTLS client ``Certificate`` when one is presented,
    or ``None`` when it is absent — for dual-auth endpoints (e.g. the registry) that
    accept either an mTLS cert or a legacy credential.  A present-but-malformed cert
    still raises ``HTTPException(400)`` via ``_parse_client_cert_header``.
    """

    async def _dep(request: Request) -> Optional[Certificate]:
        return _parse_client_cert_header(request)

    return _dep


def _get_client_certificate(request: Request) -> Certificate:
    """
    Extract the required mTLS client certificate from the nginx-injected X-Client-Cert
    header, returning a typed ``Certificate`` or raising ``NoClientCertError``.

    When a proxy secret is configured the request must carry a matching proxy header (from
    either the attestation or the cvm proxy) before we trust anything in ``X-Client-Cert``.
    This is a second line of defence against header-injection attacks: even if
    ``require_attestation_proxy`` is somehow absent from a future endpoint the cert header
    cannot be forged without also knowing a proxy secret.
    """
    if settings.attestation_proxy_secret or settings.cvm_proxy_secret:
        via_cvm, via_att = _proxy_provenance(request)
        if not (via_cvm or via_att):
            raise NoClientCertError(
                detail="X-Client-Cert header rejected: request did not arrive via a trusted mTLS proxy."
            )

    try:
        cert = _parse_client_cert_header(request)
    except HTTPException as e:
        # Present-but-malformed cert: preserve the "no valid client cert" contract that
        # callers (and extract_client_cert_hash) already expect from this helper.
        raise NoClientCertError(detail="No client certificate provided") from e
    if cert is None:
        raise NoClientCertError(detail="No client certificate provided")
    return cert


def extract_nonce(quote: TdxQuote):
    """Extract nonce from quote report_data. Raises InvalidQuoteError if report_data is missing."""
    if quote.report_data is None:
        raise InvalidQuoteError(
            "Quote has no report data; nonce cannot be extracted. The quote may be malformed."
        )
    return quote.report_data[:64].lower()


def extract_cert_hash(quote: TdxQuote):
    return quote.report_data[64:128].lower()


def extract_report_data(quote: TdxQuote):
    # Extract nonce from report_data (first printable ASCII portion)
    nonce = extract_nonce(quote)
    cert_hash = extract_cert_hash(quote)

    return nonce, cert_hash


async def verify_quote_signature(quote: TdxQuote) -> TdxVerificationResult:
    """
    Verify the cryptographic signature of a TDX quote using dcap-qvl.

    Args:
        quote_bytes: Raw TDX quote bytes
        verify_collateral: Whether to verify against Intel's collateral (requires PCCS)

    Returns:
        True if signature is valid, False otherwise
    """

    logger.info("Verifying TDX quote signature using dcap-qvl")

    try:
        # Fetch collateral once so the module-identity fallback can reuse it.
        collateral = await get_collateral(PHALA_PCCS_URL, quote.raw_bytes)
        try:
            verified_report = verify(quote.raw_bytes, collateral, int(time.time()))
            result = TdxVerificationResult.from_report(verified_report)
        except ValueError as e:
            # dcap-qvl can't match a TCB level for newer TDX module generations
            # (see resolve_tdx_tcb_status). This is raised only after all crypto
            # checks pass, so re-resolve the TCB verdict; anything else fails closed.
            if "No matching TCB level found" not in str(e):
                raise
            result = _resolve_tdx_tcb_via_module_identity(quote, collateral, e)

        if result.is_valid:
            logger.success("TDX quote signature verification successful")
        else:
            error_msg = f"Verification status: {result.status}"
            if result.advisory_ids:
                error_msg += f"; advisory_ids: {result.advisory_ids}"
            logger.error(f"TDX quote signature verification failed: {error_msg}")
            raise InvalidSignatureError("TDX quote signature verification failed")

        return result
    except AttestationError:
        # Already a structured, fail-closed attestation error; don't mask it.
        raise
    except Exception as e:
        logger.error(f"Unexpected error during quote verification: {e}")
        raise InvalidQuoteError("Unable to parse provided quote for verification.")


def _resolve_tdx_tcb_via_module_identity(
    quote: TdxQuote, collateral, original_error: Exception
) -> TdxVerificationResult:
    """
    Re-resolve a TDX quote's TCB verdict via Intel's module-identity algorithm
    when dcap-qvl could not match a platform TCB level.

    ``verify`` already validated every signature (and the Intel-signed
    ``collateral.tcb_info``) before raising, so here we only recompute the TCB
    status; measurements come from dcap-qvl's parse of the verified TD report.
    Fails closed on anything unexpected.
    """
    parsed = Quote.parse(quote.raw_bytes)
    if not parsed.is_tdx():
        # SGX quotes never use TDX module identity; the original failure stands.
        raise original_error
    report = parsed.report
    pck = parsed.pck_extension()

    tcb_info = json.loads(collateral.tcb_info)
    status, advisory_ids = resolve_tdx_tcb_status(
        tcb_info=tcb_info,
        tee_tcb_svn=list(report.tee_tcb_svn),
        sgx_tcb_components=list(pck.cpu_svn),
        pce_svn=pck.pce_svn,
        mr_signer_seam=report.mr_signer_seam,
        seam_attributes=report.seam_attributes,
    )
    logger.info(
        "Resolved TDX TCB via module identity fallback: "
        f"status={status}, tee_tcb_svn={list(report.tee_tcb_svn)[:2]}..."
    )
    return TdxVerificationResult.from_fields(
        mr_td=report.mr_td,
        rt_mr0=report.rt_mr0,
        rt_mr1=report.rt_mr1,
        rt_mr2=report.rt_mr2,
        rt_mr3=report.rt_mr3,
        report_data=report.report_data,
        td_attributes=report.td_attributes,
        status=status,
        advisory_ids=advisory_ids,
    )


def get_latest_measurement_version() -> str:
    """Return the highest semver version string across all accepted TEE measurement configs."""
    versions = [m.version for m in settings.tee_measurements if m.version]
    if not versions:
        return "0.0.0"
    latest = versions[0]
    for v in versions[1:]:
        if semcomp(v, latest) > 0:
            latest = v
    return latest


def get_matching_measurement_config(quote: TdxQuote) -> TeeMeasurementConfig:
    """
    Find the measurement config that matches the quote by full MRTD + RTMRs.

    Multiple configs may share the same RTMR0 (e.g. old and new VM versions);
    matching is by full MRTD and all RTMRs from the quote.

    Returns:
        The matching TeeMeasurementConfig

    Raises:
        MeasurementMismatchError: If no config matches
    """
    for config in settings.tee_measurements:
        if quote.matches_measurement(config):
            return config

    logger.info(
        f"No measurement config matched quote (MRTD + RTMRs)\n{quote.mrtd=}\n{quote.rtmrs=}"
    )
    raise MeasurementMismatchError()


def _require_nonempty_hotkey_allowlist(config: TeeMeasurementConfig) -> None:
    """The rc allowlist (``authorized_hotkeys``) must be non-empty.
    _load_tee_measurements already drops such rc entries; defense in depth so an empty allowlist is
    never treated as "allow all"."""
    if not config.authorized_hotkeys:
        logger.error(
            f"rc measurement '{config.name}' v{config.version} rejected: authorized_hotkeys "
            "allowlist is empty (config-loading bug)."
        )
        raise MeasurementMismatchError()



def authorize_rc_measurement(config: TeeMeasurementConfig, auth: Optional["HotkeyAuth"]) -> None:
    """Central release-candidate (rc) authorization check, invoked from ``verify_quote`` so it
    covers every trust-granting flow at the single point where the measurement (hence version) is
    known.

    No-op for published (non-rc) measurements: they lock down identical guest software for
    everyone, so operator identity is irrelevant and nothing changes for existing VMs.

    For an rc match, fail closed: the caller must have PROVEN possession of a hotkey in
    ``authorized_hotkeys``. ``auth`` is None unless it came back from ``verify_hotkey_auth`` at the
    edge, so an unproven caller arrives here with nothing and is refused.

    Any failure surfaces a bare ``MeasurementMismatchError`` whose default message is the generic
    no-match text -- deliberately identical to what a caller whose quote matches nothing sees, so
    the response never reveals that a measurement is rc-gated or who is allowed. The specific reason
    is logged server-side only.
    """
    if not config.rc:
        return  # published measurement: the gate is a no-op

    _require_nonempty_hotkey_allowlist(config)

    if auth is None:
        # Never leak that this measurement is rc-gated: the caller sees the generic no-match text.
        logger.warning(
            f"rc measurement '{config.name}' v{config.version} rejected: caller proved no hotkey."
        )
        raise MeasurementMismatchError()


    if auth.miner_hotkey not in config.authorized_hotkeys:
        logger.warning(
            f"rc measurement '{config.name}' v{config.version} rejected: hotkey "
            f"{auth.miner_hotkey[:12]}... is not in the authorized_hotkeys allowlist."
        )
        raise MeasurementMismatchError()

    logger.info(
        f"rc measurement '{config.name}' v{config.version} authorized via hotkey "
        f"{auth.miner_hotkey[:12]}..."
    )


def verify_measurements(quote: TdxQuote) -> TeeMeasurementConfig:
    """
    Verify quote measurements against allowed measurement values.

    Finds the matching config by full MRTD + RTMRs (multiple configs may share RTMR0).

    Args:
        quote: Parsed TDX quote

    Returns:
        The matched TeeMeasurementConfig (so callers -- e.g. verify_quote's rc gate -- can reuse
        it without a second lookup).

    Raises:
        MeasurementMismatchError: If any measurements don't match
    """
    measurement_config = get_matching_measurement_config(quote)
    expected_rtmrs = (
        measurement_config.boot_rtmrs
        if quote.quote_type == "boot"
        else measurement_config.runtime_rtmrs
    )

    logger.info(
        f"Verifying quote for measurement config '{measurement_config.name}' "
        f"(version={measurement_config.version}, RTMR0: {quote.rtmr0.upper()[:16]}...)"
    )
    _verify_measurements(quote, expected_rtmrs, measurement_config.name, measurement_config.mrtd)
    return measurement_config


def verify_result(quote: TdxQuote, result: TdxVerificationResult) -> bool:
    """
    Ensure the parsed quote matches the DCAP verification result.

    Compares quote.mrtd and quote.rtmrs to result.mrtd and result.rtmrs.
    Has nothing to do with measurement config; only validates that our parsing
    matches what DCAP verified.

    Raises:
        MeasurementMismatchError: If quote and result measurements differ
    """
    logger.info("Verifying quote matches DCAP verification result.")
    return _verify_measurements(quote, result.rtmrs, "DCAP result", result.mrtd)


def _verify_measurements(
    quote: TdxQuote,
    expected_rtmrs: Dict[str, str],
    measurement_name: str,
    expected_mrtd: str,
) -> bool:
    """
    Compare quote measurements to expected mrtd and rtmrs.

    Used both to compare quote to config (verify_measurements) and quote to DCAP result (verify_result).
    """
    try:
        mismatches = []

        if quote.mrtd.upper() != expected_mrtd.upper():
            error_msg = (
                f"MRTD mismatch for measurement config '{measurement_name}': "
                f"expected {expected_mrtd[:16]}..., got {quote.mrtd[:16]}..."
            )
            logger.error(error_msg)
            mismatches.append(error_msg)

        for rtmr_name, expected_value in expected_rtmrs.items():
            actual_value = quote.rtmrs.get(rtmr_name.lower()) or quote.rtmrs.get(rtmr_name)
            if not actual_value:
                error_msg = f"Quote missing expected RTMR[{rtmr_name}]"
                logger.error(error_msg)
                mismatches.append(error_msg)
            elif actual_value.upper() != expected_value.upper():
                error_msg = (
                    f"RTMR {rtmr_name} mismatch for measurement config '{measurement_name}': "
                )
                logger.error(f"{error_msg} expected {expected_value}..., got {actual_value}...")
                mismatches.append(error_msg)

        if mismatches:
            logger.error(f"Measurement verification failed: {'; '.join(mismatches)}")
            raise MeasurementMismatchError()

        logger.info(
            f"Measurements verified successfully for measurement config '{measurement_name}'"
        )
        return True

    except MeasurementMismatchError:
        raise
    except Exception as e:
        logger.error(f"Unexpected error during measurement verification: {e}", exc_info=True)
        # Re-raise as AttestationError for unexpected exceptions
        raise AttestationError("Measurement verification failed due to an unexpected error.")


def generate_cache_passphrase() -> str:
    """
    Generate a new cryptographically secure passphrase for cache volume encryption.

    Returns:
        128-character hex passphrase
    """
    return secrets.token_hex(64)


def _get_fernet() -> Fernet:
    """Get Fernet cipher for encrypting/decrypting cache passphrases.

    Returns:
        Fernet cipher instance

    Raises:
        InvalidTdxConfiguration: If encryption key is not configured
    """
    fernet = settings.fernet_key
    if not fernet:
        logger.error("No passphrase encryption key configured")
        raise InvalidTdxConfiguration(
            "PASSPHRASE_ENCRYPTION_KEY environment variable must be set. "
            "Generate a valid key with: python -c 'from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())'"
        )
    return fernet


def encrypt_passphrase(passphrase: str) -> str:
    """Encrypt a cache passphrase for storage.

    Args:
        passphrase: Plain text passphrase

    Returns:
        Encrypted passphrase (base64 encoded)
    """
    fernet = _get_fernet()
    encrypted = fernet.encrypt(passphrase.encode())
    return encrypted.decode()


def decrypt_passphrase(encrypted_passphrase: str) -> str:
    """Decrypt a stored cache passphrase.

    Args:
        encrypted_passphrase: Encrypted passphrase (base64 encoded)

    Returns:
        Plain text passphrase
    """
    fernet = _get_fernet()
    decrypted = fernet.decrypt(encrypted_passphrase.encode())
    return decrypted.decode()


async def _get_vm_cache_config(
    db: AsyncSession, miner_hotkey: str, vm_name: str
) -> Optional[VmCacheConfig]:
    """Get VmCacheConfig row if it exists."""
    result = await db.execute(
        select(VmCacheConfig).where(
            VmCacheConfig.miner_hotkey == miner_hotkey,
            VmCacheConfig.vm_name == vm_name,
        )
    )
    return result.scalar_one_or_none()


async def _create_vm_cache_config(
    db: AsyncSession, miner_hotkey: str, vm_name: str
) -> VmCacheConfig:
    """Create and persist a new VmCacheConfig row."""
    vm_config = VmCacheConfig(
        miner_hotkey=miner_hotkey,
        vm_name=vm_name,
        volume_passphrases={},
        last_boot_at=func.now(),
    )
    db.add(vm_config)
    await db.flush()
    return vm_config


async def delete_luks_passphrases_for_server(
    db: AsyncSession, miner_hotkey: str, server_name: str
) -> None:
    """Remove all LUKS passphrases for a VM (e.g. when server is deleted)."""
    result = await db.execute(
        select(VmCacheConfig).where(
            VmCacheConfig.miner_hotkey == miner_hotkey,
            VmCacheConfig.vm_name == server_name,
        )
    )
    vm_config = result.scalar_one_or_none()
    if vm_config:
        await db.delete(vm_config)
        await db.commit()
        logger.info(f"Deleted LUKS config for VM {server_name} (miner: {miner_hotkey})")


async def generate_confirm_nonce(miner_hotkey: str, vm_name: str) -> str:
    """Generate a confirm nonce and store in Redis with 5-minute TTL."""
    nonce = secrets.token_hex(32)
    redis_key = f"confirm:{miner_hotkey}:{vm_name}"
    await settings.redis_client.setex(redis_key, 300, nonce)
    logger.info(f"Generated confirm nonce for VM {vm_name} (miner: {miner_hotkey})")
    return nonce


async def generate_luks_quote_nonce(miner_hotkey: str, vm_name: str) -> str:
    """Generate a LUKS quote nonce and store in Redis with 10-minute TTL."""
    nonce = secrets.token_hex(32)
    redis_key = f"luks_quote_nonce:{miner_hotkey}:{vm_name}"
    await settings.redis_client.setex(redis_key, 600, nonce)
    logger.info(f"Generated LUKS quote nonce for VM {vm_name} (miner: {miner_hotkey})")
    return nonce


async def rotate_luks_passphrases(
    db: AsyncSession,
    miner_hotkey: str,
    vm_name: str,
    volume_names: List[str],
) -> tuple[Dict[str, LuksVolumeRotation], "VmCacheConfig"]:
    """
    Rotate LUKS passphrases for the given volumes.

    For each volume:
    - Reads the current passphrase from DB (None if first boot)
    - Discards any stale pending passphrase from a prior unconfirmed rotation
    - Generates a new passphrase stored as pending_{vol} in volume_passphrases

    Returns a tuple of (volume_data, vm_config) where volume_data maps each
    volume name to a LuksVolumeRotation and vm_config is the updated ORM object
    (after commit + refresh).
    """
    vm_config = await _get_vm_cache_config(db, miner_hotkey, vm_name)
    if vm_config is None:
        vm_config = await _create_vm_cache_config(db, miner_hotkey, vm_name)

    stored: Dict[str, str] = dict(vm_config.volume_passphrases or {})
    result: Dict[str, LuksVolumeRotation] = {}

    for vol in volume_names:
        current_enc = stored.get(vol)
        current = decrypt_passphrase(current_enc) if current_enc else None

        # Discard any stale pending from a prior unconfirmed rotation
        stored.pop(f"pending_{vol}", None)

        new_passphrase = generate_cache_passphrase()
        encrypted_new = encrypt_passphrase(new_passphrase)
        stored[f"pending_{vol}"] = encrypted_new

        if current is None:
            # WORKAROUND: remove once setup_storage sets STORAGE_KEY_ADDED=1
            # after luksFormat (so confirm sends rotated=true on first boot).
            # Until then the VM confirms with rotated=false, which discards
            # the pending key — this duplicate write ensures the passphrase
            # used to format the volume survives the discard.
            stored[vol] = encrypted_new

        result[vol] = LuksVolumeRotation(current=current, next=new_passphrase)

    vm_config.volume_passphrases = stored
    vm_config.last_boot_at = func.now()
    await db.commit()
    await db.refresh(vm_config)
    logger.info(f"LUKS rotation for VM {vm_name}: volumes={volume_names}")
    return result, vm_config


def get_default_root_passphrase(image_version: Optional[str]) -> str:
    """Return the build-time default root LUKS passphrase for the given image version.

    Each VM image bakes in a version-specific root-volume passphrase, so the passphrase
    returned by /boot/attestation is keyed by the attested measurement version. Sourced from
    the version-keyed LUKS_PASSPHRASES secret (settings.luks_passphrases), never the database:
    build-time defaults are identical for every VM on a given image version, so they stay in
    the mounted secret (blast radius unchanged) and are kept out of the DB. Only per-VM
    *rotated* passphrases are persisted (encrypted) in vm_cache_configs.volume_passphrases.

    Raises:
        InvalidTdxConfiguration: If image_version is None or no passphrase is configured for it.
    """
    passphrase = settings.luks_passphrases.get(image_version) if image_version else None
    if not passphrase:
        logger.error(f"No LUKS passphrase configured for version {image_version}")
        raise InvalidTdxConfiguration(f"No LUKS passphrase configured for version {image_version}")
    return passphrase


async def get_root_passphrase_for_boot(
    db: AsyncSession,
    miner_hotkey: str,
    vm_name: str,
    first_boot: bool,
    measurement_version: str,
) -> tuple[str, Optional[str], Optional[str]]:
    """Resolve the current root passphrase and optionally stage the next rotation.

    Returns a tuple of (key, root_next, root_confirm_nonce):
    - key: passphrase the VM should use to unlock the root volume right now.
    - root_next: new passphrase for the VM to rotate into (None for pre-1.4.0 VMs).
    - root_confirm_nonce: single-use confirm nonce (None for pre-1.4.0 VMs).

    The measurement_version is used both to gate rotation capability and as the
    image version key for looking up the build-time default passphrase.
    """
    vm_config = await _get_vm_cache_config(db, miner_hotkey, vm_name)
    if vm_config is None:
        vm_config = await _create_vm_cache_config(db, miner_hotkey, vm_name)

    stored: Dict[str, str] = dict(vm_config.volume_passphrases or {})

    if first_boot:
        stored.pop("root", None)
        stored.pop("pending_root", None)
        key = get_default_root_passphrase(measurement_version)
    else:
        current_enc = stored.get("root")
        if current_enc:
            key = decrypt_passphrase(current_enc)
        else:
            key = get_default_root_passphrase(measurement_version)

    root_next: Optional[str] = None
    root_confirm_nonce: Optional[str] = None

    if semcomp(measurement_version, MIN_ROOT_ROTATION_VERSION) >= 0:
        # Discard any stale pending from a prior unconfirmed rotation.
        stored.pop("pending_root", None)
        new_passphrase = generate_cache_passphrase()
        stored["pending_root"] = encrypt_passphrase(new_passphrase)
        root_next = new_passphrase
        root_confirm_nonce = await generate_confirm_nonce(miner_hotkey, vm_name)

    vm_config.volume_passphrases = stored
    vm_config.last_boot_at = func.now()
    await db.commit()
    await db.refresh(vm_config)

    logger.info(
        f"Root passphrase resolved for VM {vm_name} (miner: {miner_hotkey}, "
        f"first_boot={first_boot}, rotation={'yes' if root_next else 'no'})"
    )
    return key, root_next, root_confirm_nonce


async def _track_server(
    db: AsyncSession,
    server_id: str,
    name: str,
    host: str,
    miner_hotkey: str,
    is_tee: bool = False,
):
    # Add server and nodes to DB (server_id provided by client)
    server = Server(
        server_id=server_id,
        name=name,
        ip=host,
        miner_hotkey=miner_hotkey,
        is_tee=is_tee,
    )

    db.add(server)
    await db.commit()
    await db.refresh(server)

    return server


async def verify_quote(
    quote: TdxQuote,
    expected_nonce: str,
    expected_cert_hash: str,
    *,
    auth: Optional["HotkeyAuth"] = None,
) -> tuple[TdxVerificationResult, TeeMeasurementConfig]:
    """Verify a TDX quote end to end: nonce + cert-hash binding, DCAP signature, quote/DCAP
    consistency, measurement match, and -- centrally, for every trust-granting flow -- the
    release-candidate authorization check.

    ``auth`` gates access to release-candidate (``rc: true``) measurements: for an rc match the
    caller must hold a proven hotkey that is on the measurement's allowlist (see
    ``authorize_rc_measurement``). It is a no-op for published measurements, so callers that can
    never present a proof may omit it -- doing so simply makes rc measurements unreachable for
    them, which is the safe direction.
    This is the one place the rc check lives, so it cannot be bypassed by any endpoint that
    verifies a quote.

    Returns ``(report, measurement_config)`` -- the measurement the quote MATCHED, so callers never
    re-look it up (settings.tee_measurements re-reads the ConfigMap on every access).
    """
    nonce, cert_hash = extract_report_data(quote)

    if nonce != expected_nonce:
        logger.info(f"Nonce error:  {nonce} =/= {expected_nonce}")
        raise NonceError("Quote nonce does not match expected nonce.")

    if cert_hash != expected_cert_hash:
        raise InvalidClientCertError()

    result = await verify_quote_signature(quote)
    verify_result(quote, result)
    measurement_config = verify_measurements(quote)

    # Central rc gate: verify_measurements has matched the config (so we know the version/rc flag);
    # restrict rc measurements to authorized hotkeys with proof of possession. No-op for published.
    authorize_rc_measurement(measurement_config, auth)

    return result, measurement_config


def verify_leaf_cert_signed_by_ca(leaf: Certificate, ca: Certificate) -> None:
    """
    Verify that the ``leaf`` certificate was issued and signed by the ``ca``.

    Both are already-parsed ``Certificate`` objects: ``leaf`` is the mTLS client cert extracted by
    the ``extract_optional_client_cert`` / ``extract_client_cert`` dependency; ``ca`` is the VM's
    registered root CA (``server.vm_root_ca_certificate``).

    Raises InvalidClientCertError (403) on any verification failure so callers need not
    catch specific crypto exceptions.  Self-signed leaf certs (issuer == subject)
    are rejected even if the signature could technically verify.
    """
    # Reject self-signed leaf certs.
    if leaf.subject == leaf.issuer:
        raise InvalidClientCertError(detail="Self-signed leaf cert not allowed.")

    # Leaf issuer must match CA subject.
    if leaf.issuer != ca.subject:
        raise InvalidClientCertError(detail="Leaf cert issuer does not match CA cert subject.")

    # Verify leaf signature against CA public key.
    ca_pubkey = ca.public_key()
    try:
        if isinstance(ca_pubkey, ec.EllipticCurvePublicKey):
            ca_pubkey.verify(
                leaf.signature,
                leaf.tbs_certificate_bytes,
                ec.ECDSA(leaf.signature_hash_algorithm),
            )
        else:
            ca_pubkey.verify(
                leaf.signature,
                leaf.tbs_certificate_bytes,
                padding.PKCS1v15(),
                leaf.signature_hash_algorithm,
            )
    except InvalidSignature:
        raise InvalidClientCertError(detail="Leaf cert signature verification failed.")
    except Exception as e:
        logger.warning(f"verify_leaf_cert_signed_by_ca unexpected error: {e}")
        raise InvalidClientCertError(detail="Leaf cert signature verification failed.")


def verify_server_cert(client_cert: Optional[Certificate], server: Server) -> None:
    """
    Verify a VM's presented mTLS client cert against the root CA it registered.

    The shared core of "authenticate a post-provision VM by its client cert": a VM records its
    root CA via POST /provision, then presents a leaf signed by that CA on subsequent mTLS
    calls. This verifies the leaf against ``server.vm_root_ca_cert``, regardless of how the
    server was resolved (by source IP for the registry, by ``(hotkey, vm_name)`` for
    ``require_server_mtls``).

    Raises NoClientCertError (403) if no client cert is presented or the VM has no CA on file;
    verify_leaf_cert_signed_by_ca raises InvalidClientCertError (403) if the leaf fails to verify.
    """
    ca = server.vm_root_ca_certificate
    if client_cert is None or ca is None:
        raise NoClientCertError(
            detail="VM must present an mTLS leaf certificate signed by its registered CA."
        )
    verify_leaf_cert_signed_by_ca(client_cert, ca)


async def verify_gpu_evidence(evidence: list[Dict[str, str]], expected_nonce: str) -> None:
    try:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json") as fp:
            json.dump(evidence, fp)
            fp.flush()

            verify_gpus_cmd = [
                "chutes-nvattest",
                "--nonce",
                expected_nonce,
                "--evidence",
                fp.name,
            ]

            # Capture the verifier's output (stderr merged into stdout) so the actual
            # failure reason is logged rather than discarded. communicate() drains the
            # pipe and waits for exit.
            process = await asyncio.create_subprocess_exec(
                *verify_gpus_cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
            )
            stdout, _ = await process.communicate()
            output = stdout.decode(errors="replace").strip() if stdout else ""

    except FileNotFoundError as e:
        logger.error(f"Failed to verify GPU evidence.  chutes-nvattest command not found?:\n{e}")
        raise GpuEvidenceError()
    except Exception as e:
        logger.error(f"Unexepected exception encoutnered verifying GPU evidence:\n{e}")
        raise GpuEvidenceError()

    # Raise outside the try so a failed verification surfaces as InvalidGpuEvidenceError
    # (with the verifier output) instead of being swallowed by the except above. The raw
    # verifier output is logged here (server-side only), never returned in the client message.
    if process.returncode != 0:
        logger.error(
            f"GPU evidence verification failed (chutes-nvattest exit={process.returncode}):\n{output}"
        )
        raise InvalidGpuEvidenceError()

    logger.info("GPU evidence verified successfully." + (f"\n{output}" if output else ""))


async def store_host_profile(
    db: AsyncSession,
    profile: HostProfile,
    hotkey: str,
) -> tuple[str, bool]:
    """
    Record a submitted host profile, first-write-wins.

    A host class already on file -- pending or measured -- is left untouched; the row is keyed by
    fingerprint, so ON CONFLICT DO NOTHING gives idempotency without a read-then-write race.

    Returns (fingerprint, created); created=False means that host class was already known.
    """
    fingerprint = profile.fingerprint
    result = await db.execute(
        pg_insert(HostProfileRecord)
        .values(
            fingerprint=fingerprint,
            profile=profile.model_dump(by_alias=True),
            miner_hotkey=hotkey,
        )
        .on_conflict_do_nothing(index_elements=["fingerprint"])
        .returning(HostProfileRecord.fingerprint)
    )
    created = result.scalar_one_or_none() is not None
    if not created:
        logger.info(f"Host profile {fingerprint} already known, ignoring {hotkey=}")
        return fingerprint, False

    await db.commit()
    logger.success(
        f"Stored host profile {fingerprint} from {hotkey=}: "
        f"{profile.gpu.count}x{'/'.join(sorted(set(profile.gpu.pci_device_ids)))}"
    )
    return fingerprint, True


def measurements_for_fingerprint(fingerprint: str) -> list[dict]:
    """
    Every published measurement carrying this fingerprint, as ``{version, rc}`` pairs.

    Surfaced on GET /servers/tdx/host_profiles so a reader can see which VM images are covered for
    a host class. rc is a property of the version, not the topology, so both rc and non-rc entries
    are listed -- a reader wanting the rc image and one wanting the release both see theirs.

    No version floor: only ``1.4.0+`` measurements carry a fingerprint at all (older ones never
    join), so the listed versions are already the full covered set. Independent of the profile
    row's ``measured_at``, which can lag a fresh publish; the empty list is a real answer (nothing
    published for the class right now). Deduplicated and ordered for a stable response.
    """
    seen: dict[tuple[str, bool], dict] = {}
    for measurement in settings.tee_measurements:
        if measurement.fingerprint != fingerprint:
            continue
        seen.setdefault(
            (measurement.version, bool(measurement.rc)),
            {"version": measurement.version, "rc": bool(measurement.rc)},
        )
    return [seen[key] for key in sorted(seen)]


async def host_profile_state(
    db: AsyncSession, fingerprint: str
) -> "tuple[bool, Optional[datetime]]":
    """``(on_file, measured_at)`` for a host class: whether a row exists and, if so, when it was
    first measured (None while still pending). One query backing the retention status.
    """
    row = (
        await db.execute(
            select(HostProfileRecord.measured_at).where(
                HostProfileRecord.fingerprint == fingerprint
            )
        )
    ).one_or_none()
    if row is None:
        return False, None
    return True, row[0]


async def host_profile_is_known(db: AsyncSession, fingerprint: str) -> bool:
    """Whether a profile for this host class is on file, pending or measured."""
    on_file, _ = await host_profile_state(db, fingerprint)
    return on_file


def _host_profile_status(on_file: bool, measured_at) -> HostProfileStatus:
    """The monotonic retention status, from the profile row alone.

    ACCEPTED once ``measured_at`` is stamped -- the reconciler sets it (rc-inclusive) the moment any
    measurement carries the fingerprint, and never clears it, so the class stays accepted for good.
    PENDING when a row exists but is not yet reconciled, UNKNOWN when even the row is absent. The
    config is not consulted here: measured_at is the single retention marker, so the table and the
    generator cannot drift into disagreeing about what has been measured.
    """
    if measured_at is not None:
        return HostProfileStatus.ACCEPTED
    if on_file:
        return HostProfileStatus.PENDING
    return HostProfileStatus.UNKNOWN


async def resolve_host_profile_status(
    db: AsyncSession,
    profile: HostProfile,
    hotkey: str,
) -> "tuple[str, HostProfileStatus, bool]":
    """
    Store a submitted host class and return its retention ``status``.

    ``status`` is the monotonic retention lifecycle (unknown -> pending -> accepted) read from the
    profile row plus ``measured_at``; a submission is always stored, so it never returns UNKNOWN --
    the class is at least PENDING once recorded. Submission answers only which of the three the class
    is in; whether a specific image can boot is POST /servers/tdx/preflight, and the per-version
    ``{version, rc}`` list lives on GET -- neither is decided here.

    The profile is stored even when a measurement already covers the fingerprint, if we hold no
    profile for it: a fingerprint cannot be inverted back to its topology inputs, so an accepted
    host class with no stored profile cannot have its RTMR0 regenerated after a firmware change.
    store_host_profile no-ops when the row already exists.

    Returns (fingerprint, status, stored).
    """
    fingerprint = profile.fingerprint
    _, stored = await store_host_profile(db=db, profile=profile, hotkey=hotkey)

    on_file, measured_at = await host_profile_state(db, fingerprint)
    status = _host_profile_status(on_file, measured_at)
    return fingerprint, status, stored


async def list_pending_profiles(db: AsyncSession) -> list[HostProfileRecord]:
    """Host classes awaiting measurement generation, oldest first."""
    result = await db.execute(
        select(HostProfileRecord)
        .where(HostProfileRecord.measured_at.is_(None))
        .order_by(HostProfileRecord.created_at)
    )
    return list(result.scalars().all())


async def list_host_profile_records(
    db: AsyncSession,
    include_pending: bool = False,
) -> list[dict]:
    """
    Host profiles for publication, as {fingerprint, measured, measurements, profile}.

    Measured only by default: that is the set a third party can actually verify, by joining
    ``fingerprint`` to a published measurement. A pending row is an unverified claim, so nobody
    gets one without asking.

    ``include_pending`` adds them, for the measurement generator -- a profile becomes measured only
    once measurements are generated for it, and generation has to fetch it first, so the queue has
    to be reachable somehow.

    ``measurements`` is the per-fingerprint ``{version, rc}`` list from the config, so a reader sees
    exactly which VM images are covered for the class (empty for a pending row). ``profile`` is
    returned as stored: the machine-identifying fields are dropped at submission (HostProfile
    declares them ``exclude=True``), so the column holds only host-class data. ``miner_hotkey`` is
    a column this query never selects.
    """
    query = select(
        HostProfileRecord.fingerprint,
        HostProfileRecord.profile,
        HostProfileRecord.measured_at,
    ).order_by(HostProfileRecord.fingerprint)
    if not include_pending:
        query = query.where(HostProfileRecord.measured_at.isnot(None))

    result = await db.execute(query)
    return [
        {
            "fingerprint": fingerprint,
            "measured": measured_at is not None,
            "measurements": measurements_for_fingerprint(fingerprint),
            "profile": profile,
        }
        for fingerprint, profile, measured_at in result.all()
    ]


async def reconcile_host_profiles(db: AsyncSession) -> list[str]:
    """
    Mark every pending profile measured whose fingerprint now appears in the measurement config.

    The config is the source of truth: a fingerprint published there means the topology has been
    generated, so its profile moves into the retained set. One atomic UPDATE -- unlike the
    copy-then-delete this replaces, there is no window where a crash leaves it half-moved. rc
    counts here: an rc entry means the topology WAS generated, even though it cannot launch yet.
    """
    published = {m.fingerprint for m in settings.tee_measurements if m.fingerprint}
    if not published:
        return []

    result = await db.execute(
        update(HostProfileRecord)
        .where(
            HostProfileRecord.fingerprint.in_(published),
            HostProfileRecord.measured_at.is_(None),
        )
        .values(measured_at=func.now())
        .returning(HostProfileRecord.fingerprint)
    )
    promoted = sorted(result.scalars().all())
    if promoted:
        await db.commit()
        logger.success(f"Reconciled {len(promoted)} host profile(s) as measured")
    return promoted
