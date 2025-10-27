

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

    async def get_evidence(self, nonce: str) -> Tuple[TdxQuote, Dict[str, str], str]:
        
        try:
            url = urljoin(self._url, "attest")
            async with aiohttp.ClientSession() as session:
                async with session.get(url, params={
                    "nonce": nonce
                }) as resp:
                    expected_cert_hash = extract_server_cert_hash(resp)
                    data = await resp.json()
                    quote = RuntimeTdxQuote.from_base64(data["tdx_quote"])
                    gpu_evidence = data["nvtrust_evidence"]

                    return quote, gpu_evidence, expected_cert_hash
        except Exception as exc:
            logger.error(f"Failed to get attestation evidence from [{self.server.name}] {self._url}: {exc}")
            raise GetEvidenceError()