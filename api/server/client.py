

from contextlib import asynccontextmanager
import json
import ssl
from typing import Dict, Tuple
from urllib.parse import urljoin

import aiohttp
from loguru import logger
from api.server.exceptions import GetEvidenceError
from api.server.quote import RuntimeTdxQuote, TdxQuote
from api.server.schemas import Server
from api.server.util import extract_server_cert_hash



class TeeServerClient:

    def __init__(self, server: Server):
        self.server = server
        self._url = f"https://{server.ip}:30443"

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
        
        async with aiohttp.ClientSession(
            connector=connector,
            raise_for_status=True
        ) as session:
            yield session


    async def get_evidence(self, nonce: str) -> Tuple[TdxQuote, Dict[str, str], str]:
        
        try:
            url = urljoin(self._url, "server/attest")
            async with self._attestation_session() as session:
                async with session.get(url, params={
                    "nonce": nonce
                }) as resp:
                    expected_cert_hash = extract_server_cert_hash(resp)
                    data = await resp.json()
                    quote = RuntimeTdxQuote.from_base64(data["tdx_quote"])
                    gpu_evidence = json.loads(data["nvtrust_evidence"])

                    return quote, gpu_evidence, expected_cert_hash
        except Exception as exc:
            logger.error(f"Failed to get attestation evidence from {self._url}: {exc}")
            raise GetEvidenceError()