"""
Miner authentication.
"""

import re
import time
import hashlib
import pybase64 as base64
from fastapi import APIRouter, Request, Response, HTTPException, status
from substrateinterface import Keypair, KeypairType
from run_api.config import settings


router = APIRouter()


async def authenticate_request(request: Request, purpose: str = None) -> None:
    """
    Helper to authenticate miner requests.
    """
    ss58_address = request.headers.get("X-Parachutes-Hotkey")
    signature = request.headers.get("X-Parachutes-Signature")
    if not ss58_address or not signature:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing credentials",
        )
    if not purpose:
        body = await request.body()
        nonce = request.headers.get("X-Parachutes-Nonce")
        if (
            not body
            or not nonce
            or not nonce.isdigit()
            or abs(time.time() - int(nonce)) >= 600
        ):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="No request body to verify",
            )
        payload = ":".join(
            [
                ss58_address,
                nonce,
                hashlib.sha256(body).hexdigest(),
            ]
        )
    else:
        payload = request.headers.get("X-Parachutes-Auth") or ""
        payload_match = re.match(r"^([a-z]+):([0-9]+):([a-z0-9]+)$", payload, re.I)
        if not payload_match:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid X-Parachutes-Auth header format",
            )
        a_purpose, a_nonce, a_ss58 = payload_match.groups()
        if (
            a_purpose != purpose
            or a_ss58 != ss58_address
            or abs(time.time() - int(a_nonce)) >= 600
        ):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid X-Parachutes-Auth header value",
            )
    if not payload:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="No authorization payload to verify",
        )
    if not Keypair(ss58_address=ss58_address, crypto_type=KeypairType.SR25519).verify(
        payload.encode(), bytes.fromhex(signature)
    ):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid request signature"
        )


@router.get("/registry")
async def registry_auth(request: Request, response: Response):
    """
    Authentication registry/docker pull requests.
    """
    # TODO: ensure hotkey is registered to the subnet.
    await authenticate_request(request, purpose="registry")
    auth_string = base64.b64encode(f":{settings.registry_password}".encode())
    response.headers["Authorization"] = f"Basic {auth_string}"
    return {"authenticated": True}
