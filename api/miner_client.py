"""
Helper to send requests to miners.
"""

import aiohttp
import hashlib
import time
import orjson as json
from contextlib import asynccontextmanager
from loguru import logger
from typing import Any, Dict
from api.metasync import get_miner_by_hotkey
from api.database import get_session
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
        return f"{miner_ss58}:{settings.validator_ss58}:{nonce}:{hashlib.sha256(payload_str).hexdigest()}"
    elif purpose:
        return f"{miner_ss58}:{settings.validator_ss58}:{nonce}:{purpose}"
    elif payload_hash:
        return f"{miner_ss58}:{settings.validator_ss58}:{nonce}:{payload_hash}"
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
    payload_string = None
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
    headers[SIGNATURE_HEADER] = settings.validator_keypair.sign(signature_string.encode()).hex()
    return headers, payload_string


@asynccontextmanager
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


@asynccontextmanager
async def patch(miner_ss58: str, url: str, payload: Dict[str, Any], **kwargs):
    """
    Perform a patch request to a miner.
    """
    async with aiohttp.ClientSession() as session:
        headers = kwargs.pop("headers", {})
        new_headers, payload_data = sign_request(miner_ss58, payload=payload)
        headers.update(new_headers)
        async with session.patch(url, data=payload_data, headers=headers, **kwargs) as response:
            yield response


@asynccontextmanager
async def get(miner_ss58: str, url: str, purpose: str, **kwargs):
    """
    Perform a get request to a miner.
    """
    async with aiohttp.ClientSession() as session:
        headers = kwargs.pop("headers", {})
        new_headers, payload_data = sign_request(miner_ss58, purpose=purpose)
        headers.update(new_headers)
        async with session.get(url, headers=headers, **kwargs) as response:
            yield response


async def get_real_axon(miner_ss58: str):
    """
    Attempt refreshing the axon's real host/port via porter.
    """
    if (cached := await settings.redis_client.get(f"real_axon:{miner_ss58}")) is not None:
        return cached.split(":__:")
    async with get_session() as session:
        if (miner := await get_miner_by_hotkey(miner_ss58, session)) is None:
            return None
        try:
            async with get(
                miner_ss58,
                f"http://{miner.ip}:{miner.port}/axon",
                purpose="porter",
                timeout=5.0,
            ) as resp:
                result = await resp.json()
                logger.debug(f"Received response from {miner_ss58=} porter: {result}")
                if result["host"] and miner.real_host != result["host"]:
                    miner.real_host = result["host"]
                    miner.real_port = result["port"]
                    await session.commit()
                    await session.refresh(miner)
                await settings.redis_client.set(
                    f"real_axon:{miner_ss58}",
                    f"{miner.real_host}:__:{miner.real_port}",
                    ex=300,
                )
        except Exception as exc:
            logger.warning(f"Error refreshing real axon: {exc}")
        return f"http://{miner.real_host or miner.ip}:{miner.real_port or miner.port}"


@asynccontextmanager
async def axon_post(miner_ss58: str, path: str, payload: Dict[str, Any], **kwargs):
    """
    Perform a post request to a miner's axon (redirecting from their porter instance, if any).
    """
    real_axon = await get_real_axon(miner_ss58)
    async with post(miner_ss58, f"{real_axon}{path}", payload=payload, **kwargs) as response:
        yield response


@asynccontextmanager
async def axon_patch(miner_ss58: str, path: str, payload: Dict[str, Any], **kwargs):
    """
    Perform a patch request to a miner's axon (redirecting from their porter instance, if any).
    """
    real_axon = await get_real_axon(miner_ss58)
    async with patch(miner_ss58, f"{real_axon}{path}", payload=payload, **kwargs) as response:
        yield response


@asynccontextmanager
async def axon_get(miner_ss58: str, path: str, purpose: str, **kwargs):
    """
    Perform a get request to a miner's axon (redirecting from their porter instance, if any).
    """
    real_axon = await get_real_axon(miner_ss58)
    async with get(miner_ss58, f"{real_axon}{path}", purpose=purpose, **kwargs) as response:
        yield response
