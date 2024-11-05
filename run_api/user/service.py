"""
User logic/code.
"""

import uuid
import re
import time
from sqlalchemy import exists
from sqlalchemy.future import select
from fastapi import APIRouter, Request, HTTPException, status
from substrateinterface import Keypair, KeypairType
from run_api.config import settings
from run_api.metasync import MetagraphNode
from run_api.database import SessionLocal
from run_api.user.schemas import User

router = APIRouter()


def get_current_user(
    purpose: str = None,
    registered_to: int = None,
    raise_not_found: bool = False,
    allow_api_key=False,
):
    """
    Authentication dependency builder.
    """

    async def _authenticate(request: Request):
        """
        Helper to authenticate requests.
        """
        hotkey = request.headers.get("X-Parachutes-Hotkey")
        signature = request.headers.get("X-Parachutes-Signature")
        if not hotkey or not signature:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid or missing credentials",
            )
        if not purpose:
            body_sha256 = getattr(request.state, "body_sha256", None)
            nonce = request.headers.get("X-Parachutes-Nonce")
            if (
                not body_sha256
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
                    hotkey,
                    nonce,
                    body_sha256,
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
                or a_ss58 != hotkey
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
        if not Keypair(ss58_address=hotkey, crypto_type=KeypairType.SR25519).verify(
            payload.encode(), bytes.fromhex(signature)
        ):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid request signature",
            )

        # Require a registered hotkey?
        if registered_to is not None:
            async with SessionLocal() as session:
                if not (
                    await session.execute(
                        select(
                            exists()
                            .where(MetagraphNode.hotkey == hotkey)
                            .where(MetagraphNode.netuid == registered_to)
                        )
                    )
                ).scalar():
                    raise HTTPException(
                        status_code=status.HTTP_401_UNAUTHORIZED,
                        detail=f"Hotkey is not registered on netuid {settings.netuid}",
                    )

        # Fetch the actual user.
        async with SessionLocal() as session:
            result = await session.execute(
                select(User).where(
                    User.user_id == str(uuid.uuid5(uuid.NAMESPACE_OID, hotkey))
                )
            )
            user = result.scalar_one_or_none()
            if not user and raise_not_found:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Invalid token or user not found",
                )
            return user

    return _authenticate
