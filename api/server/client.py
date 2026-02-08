from contextlib import asynccontextmanager
import hashlib
import json
import ssl
import time
from typing import Any, Dict, Optional, Tuple
from urllib.parse import urljoin

import aiohttp
from loguru import logger
from cryptography.x509 import Certificate
from api.constants import HOTKEY_HEADER, NONCE_HEADER, SIGNATURE_HEADER
from api.server.exceptions import GetEvidenceError
from api.server.quote import RuntimeTdxQuote, TdxQuote
from api.server.schemas import Server
from api.server.util import _get_server_certificate
from api.config import settings


class TeeServerClient:
    def __init__(self, server: Server):
        self.server = server
        self._url = f"https://{server.ip}:30443"

    def _sign_request(
        self, payload: Dict[str, Any] | str | None = None, purpose: str | None = None
    ):
        """
        Generate a signed request from validator to attestation proxy.
        """
        nonce = str(int(time.time()))
        headers = {
            HOTKEY_HEADER: settings.validator_ss58,
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

        # Sign: validator:nonce:payload_hash
        signature_string = f"{settings.validator_ss58}:{nonce}:{payload_hash}"
        logger.info(f"Signature string: {signature_string}")
        signature = settings.validator_keypair.sign(signature_string.encode()).hex()

        logger.info(
            f"Signing: {settings.validator_ss58=} {nonce=} {payload_hash=} {purpose=} {signature=}"
        )
        headers[SIGNATURE_HEADER] = signature

        return headers, payload_string

    @asynccontextmanager
    async def _attestation_session(self):
        """
        Creates an aiohttp session configured for the attestation service.

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
                    params={
                        "nonce": nonce,
                    },
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

        Args:
            deployment_id: The chute deployment ID (or instance identifier for the chute service).
            nonce: Optional. If set, request goes to evidence endpoint with this nonce as query param.

        Returns:
            Tuple of (quote, gpu_evidence, cert). Callers can hash cert if needed.
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
