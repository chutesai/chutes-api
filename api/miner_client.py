"""
Helper to send requests to miners.
"""

import aiohttp
import hashlib
import time
import orjson as json
from loguru import logger
from typing import Any, Dict
from api.config import settings
from api.constants import MINER_HEADER, VALIDATOR_HEADER, NONCE_HEADER, SIGNATURE_HEADER


def get_signing_message(
    miner_ss58: str,
    nonce: str,
    payload_str: str | bytes | None,
    purpose: str | None = None,
    payload_hash: str | None = None,
) -> str:
    """
    Get the signing message for a request to a miner.
    """
    if payload_str:
        if isinstance(payload_str, str):
            payload_str = payload_str.encode()
        return f"{settings.validator_ss58}:{miner_ss58}:{nonce}:{hashlib.sha256(payload_str).hexdigest()}"
    elif purpose:
        return f"{settings.validator_ss58}:{miner_ss58}:{nonce}:{purpose}"
    elif payload_hash:
        return f"{settings.validator_ss58}:{miner_ss58}:{nonce}:{payload_hash}"
    else:
        raise ValueError("Either payload_str or purpose must be provided")


def sign_request(miner_ss58: str, payload: Dict[str, Any] | str | None = None, purpose: str = None):
    """
    Generate a signed request (for miner requests to validators).
    """
    nonce = str(int(time.time()))
    headers = {
        VALIDATOR_HEADER: settings.validator_ss58,
        MINER_HEADER: miner_ss58,
        NONCE_HEADER: nonce,
    }
    signature_string = None
    if payload is not None:
        if isinstance(payload, dict):
            headers["Content-Type"] = "application/json"
            payload_string = json.dumps(payload)
        else:
            payload_string = payload
        signature_string = get_signing_message(
            miner_ss58,
            nonce,
            payload_str=payload_string,
            purpose=None,
        )
    else:
        signature_string = get_signing_message(miner_ss58, nonce, payload_str=None, purpose=purpose)
    logger.debug(f"Signing message for miner request: {signature_string}")
    headers[SIGNATURE_HEADER] = settings.validator_keypair.sign(signature_string.encode()).hex()
    return headers, payload_string


async def post(miner_ss58: str, url: str, payload: Dict[str, Any], **kwargs):
    """
    Perform a post request to a miner.
    """
    async with aiohttp.ClientSession() as session:
        headers = kwargs.pop("headers", {})
        new_headers, payload_data = sign_request(miner_ss58, payload=payload)
        headers.update(new_headers)
        async with session.post(url, data=payload_data, headers=headers, **kwargs) as response:
            yield response


async def get(miner_ss58: str, url: str, purpose: str, **kwargs):
    """
    Perform a get request to a miner.
    """
    async with aiohttp.ClientSession() as session:
        headers = kwargs.pop("headers", {})
        new_headers, payload_data = sign_request(miner_ss58, purpose=purpose)
        headers.update(headers)
        async with session.get(url, headers=headers, **kwargs) as response:
            yield response
