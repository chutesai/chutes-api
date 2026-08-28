from contextlib import asynccontextmanager
import base64
from dataclasses import dataclass
import hashlib
import json
import ssl
import time
from typing import Any, Dict, Optional, Tuple
from urllib.parse import urljoin

import aiohttp
from bittensor_wallet.keypair import Keypair
from loguru import logger
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from cryptography.x509 import Certificate
from api.constants import (
    ATTESTATION_SIGNATURE_HEADER,
    HOTKEY_HEADER,
    NONCE_HEADER,
    SIGNATURE_HEADER,
    MIN_VM_AUTH_KEY_VERSION,
)
from api.server.exceptions import GetEvidenceError
from api.server.quote import RuntimeTdxQuote, TdxQuote
from api.server.schemas import Server, VmAuthKey, VmBootRecord
from api.server.util import _get_server_certificate, decrypt_passphrase
from api.config import settings
from api.util import semcomp


@dataclass
class ChuteEvidenceResponse:
    """Structured response from TeeServerClient.get_chute_evidence().

    quote, gpu_evidence, and cert are always present.
    signature and attested_body are set only when the attestation proxy returns
    an X-Signature header (proxy >= 0.2.0); both are None on older proxies.
    """

    quote: TdxQuote
    gpu_evidence: Dict[str, Any]
    cert: Certificate
    signature: Optional[str] = None
    attested_body: Optional[str] = None


class TeeServerClient:
    def __init__(self, server: Server, keypair: Keypair):
        self.server = server
        self._url = f"https://{server.ip}:30443"
        self._keypair = keypair

    @classmethod
    async def create(cls, db: AsyncSession, server: Server) -> "TeeServerClient":
        """Resolve the signing keypair for validator->VM calls.

        >= MIN_VM_AUTH_KEY_VERSION VMs trust only their per-VM ephemeral key (vm_auth_keys); older
        VMs trust the validator key. The deciding version is the current-boot version: server.version
        if set, else the latest VmBootRecord's measurement_version (server.version is None during
        registration, and stale after a downgrade).
        """
        result = await db.execute(
            select(VmAuthKey).where(
                VmAuthKey.miner_hotkey == server.miner_hotkey,
                VmAuthKey.vm_name == server.name,
            )
        )
        vm_auth_key = result.scalar_one_or_none()

        # server.version is unset during registration; fall back to the current-boot version.
        version = server.version
        if not version:
            boot_version_result = await db.execute(
                select(VmBootRecord.measurement_version)
                .where(
                    VmBootRecord.miner_hotkey == server.miner_hotkey,
                    VmBootRecord.vm_name == server.name,
                    VmBootRecord.measurement_version.isnot(None),
                )
                .order_by(VmBootRecord.created_at.desc())
                .limit(1)
            )
            version = boot_version_result.scalar_one_or_none()

        use_ephemeral = bool(version) and semcomp(version, MIN_VM_AUTH_KEY_VERSION) >= 0
        if use_ephemeral:
            if vm_auth_key is None:
                # >= 1.4.0 must have a row; a missing one is a real inconsistency, not a legacy VM.
                raise GetEvidenceError(
                    f"No vm_auth_key row for {server.name} (miner: {server.miner_hotkey}) at "
                    f"version {version} (>= {MIN_VM_AUTH_KEY_VERSION}); cannot sign VM calls"
                )
            seed_hex = decrypt_passphrase(vm_auth_key.auth_seed)
            keypair = Keypair.create_from_seed(seed_hex)
            logger.debug(
                f"Loaded per-VM auth keypair for {server.name} "
                f"(miner: {server.miner_hotkey}): {keypair.ss58_address}"
            )
        else:
            # Legacy VM (or unknown version): validator key. A stale ephemeral row is ignored.
            keypair = settings.validator_keypair
            logger.debug(
                f"Using validator keypair for {server.name} (miner: {server.miner_hotkey}); "
                f"resolved version={version!r} (legacy signer path)"
            )

        return cls(server=server, keypair=keypair)

    def _sign_request(
        self, payload: Dict[str, Any] | str | None = None, purpose: str | None = None
    ):
        """Generate a signed request from validator to attestation proxy.

        Uses the per-VM ephemeral keypair if available, otherwise the validator's
        global keypair (legacy VMs). The signing protocol is unchanged; only the
        key identity differs.
        """
        ss58 = self._keypair.ss58_address
        nonce = str(int(time.time()))
        headers = {
            HOTKEY_HEADER: ss58,
            NONCE_HEADER: nonce,
        }

        payload_string = None
        if payload is not None:
            if isinstance(payload, dict):
                headers["Content-Type"] = "application/json"
                payload_string = json.dumps(payload)
            else:
                payload_string = str(payload)
            payload_hash = hashlib.sha256(payload_string.encode()).hexdigest()
        else:
            payload_hash = purpose or ""

        # Sign: ss58:nonce:payload_hash
        signature_string = f"{ss58}:{nonce}:{payload_hash}"
        logger.info(f"Signature string: {signature_string}")
        signature = self._keypair.sign(signature_string.encode()).hex()

        logger.info(f"Signing: {ss58=} {nonce=} {payload_hash=} {purpose=} {signature=}")
        headers[SIGNATURE_HEADER] = signature

        return headers, payload_string

    @asynccontextmanager
    async def _attestation_session(self):
        """Creates an aiohttp session configured for the attestation service.

        For VMs that have registered their attestation CA (vm_root_ca_cert is set),
        the stored CA cert is used as the trust anchor so the proxy's server cert is
        verified against it.  For pre-migration VMs (vm_root_ca_cert is None),
        SSL verification is skipped and certificate authenticity is instead checked
        out-of-band via the TDX quote's REPORTDATA hash.
        """
        ssl_context = ssl.create_default_context()
        ssl_context.check_hostname = False

        if self.server.vm_root_ca_cert:
            ssl_context.verify_mode = ssl.CERT_REQUIRED
            ssl_context.load_verify_locations(cadata=self.server.vm_root_ca_cert)
        else:
            ssl_context.verify_mode = ssl.CERT_NONE

        connector = aiohttp.TCPConnector(ssl=ssl_context)

        async with aiohttp.ClientSession(connector=connector, raise_for_status=True) as session:
            yield session

    async def get_server_evidence(self, nonce: str) -> Tuple[TdxQuote, Dict[str, str], Certificate]:
        try:
            url = urljoin(self._url, "server/attest")
            headers, _ = self._sign_request(purpose="attest")
            async with self._attestation_session() as session:
                async with session.get(
                    url,
                    headers=headers,
                    params={"nonce": nonce},
                ) as resp:
                    cert = _get_server_certificate(resp)
                    data = await resp.json()
                    quote = RuntimeTdxQuote.from_base64(data["tdx_quote"])
                    gpu_evidence = json.loads(data["nvtrust_evidence"])
                    return quote, gpu_evidence, cert
        except Exception as exc:
            logger.error(f"Failed to get attestation evidence from {self._url}: {exc}")
            raise GetEvidenceError(f"Failed to get evidence for attestation: {str(exc)}")

    async def get_chute_evidence(
        self, deployment_id: str, nonce: Optional[str] = None
    ) -> ChuteEvidenceResponse:
        """Get attestation evidence for a specific chute deployment.

        Two flows:
        - Verification (claim_tee_launch_config): call with no nonce. Hits chute's
          verify endpoint; chute uses its stored nonce to prove it is the same instance.
        - Third-party runtime evidence: call with nonce=caller_nonce. Hits chute's
          evidence endpoint with ?nonce=...; chute returns evidence bound to that nonce.

        Args:
            deployment_id: The chute deployment ID (or instance identifier for the chute service).
            nonce: Optional. If set, request goes to evidence endpoint with this nonce as query param.

        Returns:
            ChuteEvidenceResponse with quote, gpu_evidence, cert always populated.
            signature and attested_body are set when the proxy returns X-Signature (>= 0.2.0),
            otherwise both are None.
        """
        try:
            target_endpoint = "evidence" if nonce else "verify"
            url = urljoin(self._url, f"service/chute-service-{deployment_id}/{target_endpoint}")
            headers, _ = self._sign_request(purpose="attest")
            params = {"nonce": nonce} if nonce else None
            async with self._attestation_session() as session:
                async with session.get(url, headers=headers, params=params) as resp:
                    cert = _get_server_certificate(resp)
                    signature = resp.headers.get(ATTESTATION_SIGNATURE_HEADER)
                    raw_body = await resp.read()
                    data = json.loads(raw_body)
                    attested_body_b64 = (
                        base64.b64encode(raw_body).decode("ascii") if signature else None
                    )
                    quote = RuntimeTdxQuote.from_base64(data["evidence"]["tdx_quote"])
                    gpu_evidence = json.loads(data["evidence"]["nvtrust_evidence"])
                    return ChuteEvidenceResponse(
                        quote=quote,
                        gpu_evidence=gpu_evidence,
                        cert=cert,
                        signature=signature,
                        attested_body=attested_body_b64,
                    )
        except Exception as exc:
            logger.error(f"Failed to get chute evidence from {self._url}: {exc}")
            raise GetEvidenceError()
