"""
TDX quote parsing, crypto operations, and server helper functions.
"""

import asyncio
import secrets
import base64
import json
import tempfile
from typing import Dict, List, Optional
from sqlalchemy import select
from sqlalchemy.sql import func
from sqlalchemy.ext.asyncio import AsyncSession
from urllib.parse import unquote
from aiohttp import ClientResponse
from cryptography.fernet import Fernet
from fastapi import HTTPException, Request, status
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
)
from api.server.quote import TdxQuote, TdxVerificationResult, resolve_tdx_tcb_status
import hashlib

from api.server.schemas import Server, VmCacheConfig, LuksVolumeRotation, RootPassphraseDefault
from api.constants import (
    MIN_ROOT_ROTATION_VERSION,
    MTLS_PROXY_AUTH_HEADER,
    REGISTRY_PROXY_AUTH_HEADER,
)
from api.util import semcomp


def generate_nonce() -> str:
    """Generate a cryptographically secure nonce."""
    return secrets.token_hex(32)


def get_nonce_expiry_seconds(minutes: int = 10) -> int:
    """Get expiry time for a nonce in seconds."""
    return minutes * 60


def require_mtls_proxy_secret():
    """
    FastAPI dependency asserting an attestation request arrived via the mTLS nginx proxy.

    The mTLS proxy injects ``X-Mtls-Proxy-Auth`` (= ``MTLS_PROXY_SECRET``); every other proxy
    (including api.chutes.ai) must strip it. A missing or wrong value means the request bypassed
    the mTLS proxy and may carry an attacker-injected ``X-Client-Cert``. The former Host-header
    check was dropped: a Host header is trivially spoofable, so the shared secret is the real
    (and only) proxy-provenance guard -- which also means the attestation domain can change
    (e.g. cvm.chutes.ai) as pure DNS/infra config with nothing hardcoded here.

    This is the SOLE proxy-provenance guard on the attestation endpoints -- several of which
    (e.g. GET /nonce) have no other auth -- so it FAILS CLOSED: when ``MTLS_PROXY_SECRET`` is
    not configured it rejects every request rather than silently running unguarded.
    """

    async def _check(request: Request):
        if not settings.mtls_proxy_secret:
            logger.error(
                "mTLS endpoint rejected: MTLS_PROXY_SECRET is not configured; refusing to serve "
                f"the attestation endpoint unguarded path={request.url.path}"
            )
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="mTLS proxy secret is not configured.",
            )
        proxy_auth = request.headers.get(MTLS_PROXY_AUTH_HEADER, "")
        if not secrets.compare_digest(proxy_auth.encode(), settings.mtls_proxy_secret.encode()):
            logger.warning(f"mTLS endpoint rejected: proxy secret mismatch path={request.url.path}")
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Request did not arrive via the mTLS attestation proxy.",
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
        encoding=serialization.Encoding.DER, format=serialization.PublicFormat.SubjectPublicKeyInfo
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
    through the mTLS proxy is enforced separately (``require_mtls_proxy_secret`` /
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

    When ``MTLS_PROXY_SECRET`` is configured the ``X-Mtls-Proxy-Auth`` header must
    match before we trust anything in ``X-Client-Cert``.  This is a second line of
    defence against header-injection attacks: even if ``require_mtls_proxy_secret`` is
    somehow absent from a future endpoint the cert header cannot be forged without
    also knowing the proxy secret.
    """
    if settings.mtls_proxy_secret:
        proxy_auth = request.headers.get(MTLS_PROXY_AUTH_HEADER, "")
        if not secrets.compare_digest(proxy_auth.encode(), settings.mtls_proxy_secret.encode()):
            raise NoClientCertError(
                detail="X-Client-Cert header rejected: request did not arrive via the mTLS proxy."
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
    raise MeasurementMismatchError(
        "Quote does not match expected measurements. Ensure you are running a supported VM."
    )


def verify_measurements(quote: TdxQuote) -> bool:
    """
    Verify quote measurements against allowed measurement values.

    Finds the matching config by full MRTD + RTMRs (multiple configs may share RTMR0).

    Args:
        quote: Parsed TDX quote

    Returns:
        True if all measurements match

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
    return _verify_measurements(
        quote, expected_rtmrs, measurement_config.name, measurement_config.mrtd
    )


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


def get_luks_passphrase(version: str) -> str:
    """
    Get the root-volume LUKS passphrase for the given measurement version.

    Each VM image bakes in a version-specific passphrase, so the passphrase returned by
    /boot/attestation is keyed by the attested measurement version. There is no fallback.

    Args:
        version: Attested measurement version (e.g. "1.3.0").

    Returns:
        LUKS passphrase string for that version.

    Raises:
        InvalidTdxConfiguration: If no passphrase is configured for the version.
    """
    passphrase = settings.luks_passphrases.get(version)
    if not passphrase:
        logger.error(f"No LUKS passphrase configured for version {version}")
        raise InvalidTdxConfiguration(f"No LUKS passphrase configured for version {version}")

    return passphrase


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


async def get_default_root_passphrase(db: AsyncSession, image_version: Optional[str]) -> str:
    """Return the build-time default root passphrase for the given image version.

    Looks up root_passphrase_defaults by image_version; falls back to
    settings.luks_passphrase if no record exists or image_version is None.
    """
    if image_version:
        row = await db.get(RootPassphraseDefault, image_version)
        if row is not None:
            return decrypt_passphrase(row.encrypted_passphrase)
    passphrase = settings.luks_passphrase
    if not passphrase:
        raise InvalidTdxConfiguration("Missing LUKS passphrase configuration")
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
        key = await get_default_root_passphrase(db, measurement_version)
    else:
        current_enc = stored.get("root")
        if current_enc:
            key = decrypt_passphrase(current_enc)
        else:
            key = await get_default_root_passphrase(db, measurement_version)

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
    quote: TdxQuote, expected_nonce: str, expected_cert_hash: str
) -> TdxVerificationResult:
    nonce, cert_hash = extract_report_data(quote)

    if nonce != expected_nonce:
        logger.info(f"Nonce error:  {nonce} =/= {expected_nonce}")
        raise NonceError("Quote nonce does not match expected nonce.")

    if cert_hash != expected_cert_hash:
        raise InvalidClientCertError()

    result = await verify_quote_signature(quote)
    verify_result(quote, result)
    verify_measurements(quote)

    return result


def verify_leaf_cert_signed_by_ca(leaf: Certificate, ca_cert_pem: str) -> None:
    """
    Verify that the ``leaf`` certificate was issued and signed by the CA in ca_cert_pem.

    ``leaf`` is an already-parsed ``Certificate`` (the mTLS client cert extracted by the
    ``extract_optional_client_cert`` / ``extract_client_cert`` dependency); ``ca_cert_pem`` is the
    stored per-VM CA PEM (a DB blob), parsed here.

    Raises InvalidClientCertError (403) on any verification failure so callers need not
    catch specific crypto exceptions.  Self-signed leaf certs (issuer == subject)
    are rejected even if the signature could technically verify.
    """
    try:
        ca = x509.load_pem_x509_certificate(ca_cert_pem.encode(), default_backend())
    except Exception as e:
        raise InvalidClientCertError(detail="Malformed CA cert PEM.") from e

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
                leaf.signature, leaf.tbs_certificate_bytes, ec.ECDSA(leaf.signature_hash_algorithm)
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
    if client_cert is None or not server.vm_root_ca_cert:
        raise NoClientCertError(
            detail="VM must present an mTLS leaf certificate signed by its registered CA."
        )
    verify_leaf_cert_signed_by_ca(client_cert, server.vm_root_ca_cert)


async def verify_gpu_evidence(evidence: list[Dict[str, str]], expected_nonce: str) -> None:
    try:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json") as fp:
            json.dump(evidence, fp)
            fp.flush()

            verify_gpus_cmd = ["chutes-nvattest", "--nonce", expected_nonce, "--evidence", fp.name]

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
