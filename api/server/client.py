from contextlib import asynccontextmanager
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

from api.constants import HOTKEY_HEADER, NONCE_HEADER, SIGNATURE_HEADER
from api.server.exceptions import GetEvidenceError
from api.server.quote import RuntimeTdxQuote, TdxQuote
from api.server.schemas import Server, VmAuthKey
from api.server.util import _get_server_certificate, decrypt_passphrase
from api.config import settings


class TeeServerClient:
    def __init__(self, server: Server, keypair: Keypair):
        self.server = server
        self._url = f"https://{server.ip}:30443"
        self._keypair = keypair

    @classmethod
    async def create(cls, db: AsyncSession, server: Server) -> "TeeServerClient":
        """Async factory that resolves the per-VM auth keypair from the database.

        Looks up the vm_auth_keys row for this server. If one exists, decrypts
        the seed and reconstructs the ephemeral keypair. If no row exists (legacy
        VM that hasn't booted with new firmware yet), falls back to the validator's
        global keypair for backward compatibility.

        A DB read + keypair reconstruction costs ~1-5ms total, which is negligible
        compared to the TDX quote verification and GPU evidence checks that always
        surround these calls. Caching is deliberately omitted: these calls are
        infrequent (boot/registration-time only) and multi-pod deployments make
        an in-process cache ineffective anyway.
        """
        result = await db.execute(
            select(VmAuthKey).where(
                VmAuthKey.miner_hotkey == server.miner_hotkey,
                VmAuthKey.vm_name == server.name,
            )
        )
        vm_auth_key = result.scalar_one_or_none()

        if vm_auth_key is not None:
            seed_hex = decrypt_passphrase(vm_auth_key.auth_seed)
            keypair = Keypair.create_from_seed(seed_hex)
            logger.debug(
                f"Loaded per-VM auth keypair for {server.name} "
                f"(miner: {server.miner_hotkey}): {keypair.ss58_address}"
            )
        else:
            keypair = settings.validator_keypair
            logger.debug(
                f"No per-VM auth key found for {server.name} "
                f"(miner: {server.miner_hotkey}); using validator keypair (legacy VM)"
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

        SSL verification is disabled because certificate authenticity is verified
        through TDX quotes, which include a hash of the service's public key.
        """
        ssl_context = ssl.create_default_context()
        ssl_context.check_hostname = False
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
    ) -> Tuple[TdxQuote, Dict[str, str], Certificate]:
        """Get attestation evidence for a specific chute deployment.

        Two flows:
        - Verification (claim_tee_launch_config): call with no nonce. Hits chute's
          verify endpoint; chute uses its stored nonce to prove it is the same instance.
        - Third-party runtime evidence: call with nonce=caller_nonce. Hits chute's
          evidence endpoint with ?nonce=...; chute returns evidence bound to that nonce.
        """
        try:
            target_endpoint = "evidence" if nonce else "verify"
            url = urljoin(self._url, f"service/chute-service-{deployment_id}/{target_endpoint}")
            headers, _ = self._sign_request(purpose="attest")
            params = {"nonce": nonce} if nonce else None
            async with self._attestation_session() as session:
                async with session.get(url, headers=headers, params=params) as resp:
                    cert = _get_server_certificate(resp)
                    data = await resp.json()
                    quote = RuntimeTdxQuote.from_base64(data["evidence"]["tdx_quote"])
                    gpu_evidence = json.loads(data["evidence"]["nvtrust_evidence"])
                    return quote, gpu_evidence, cert
        except Exception as exc:
            logger.error(f"Failed to get chute evidence from {self._url}: {exc}")
            raise GetEvidenceError()
