"""
JWTs, for non-bittensor aware users.
"""

import jwt
import hashlib
from sqlalchemy import select
from aiocache import cached, Cache
from datetime import datetime, timedelta, timezone
from fastapi import HTTPException, status
from api.user.schemas import User
from api.database import get_session
from api.config import settings


@cached(ttl=60 * 60, cache=Cache.MEMORY)
async def get_user_fingerprint_hash(user_id: str) -> str:
    """
    Load a user's fingerprint hash.
    """
    async with get_session() as session:
        user = (
            await session.execute(select(User).where(User.user_id == user_id))
        ).scalar_one_or_none()
        if user:
            return user.fingerprint_hash
    return None


def create_token(user: User) -> str:
    """
    Create JWT token using user's fingerprint as signing key.
    """
    now = datetime.now(timezone.utc)
    expires_at = now + timedelta(days=7)
    payload = {
        "exp": expires_at.replace(tzinfo=None),
        "sub": user.user_id,
        "iat": now.replace(tzinfo=None),
    }
    signing_key = hashlib.sha256(user.fingerprint_hash.encode()).hexdigest()
    encoded_jwt = jwt.encode(payload, signing_key, algorithm="HS256")
    return encoded_jwt


async def get_user_from_token(token: str) -> User:
    """
    Verify a token.
    """
    payload = jwt.decode(token, options={"verify_signature": False})
    user_id = payload.get("sub")

    # Squad access?
    if payload.get("iss") == "squad":
        try:
            payload = jwt.decode(
                token,
                settings.squad_cert,
                algorithms=["RS256"],
                options={
                    "verify_signature": True,
                    "verify_exp": True,
                    "verify_iat": True,
                    "verify_iss": True,
                    "require": ["exp", "iat", "iss"],
                },
                issuer="squad",
            )
            async with get_session() as session:
                return (
                    await session.execute(select(User).where(User.user_id == user_id))
                ).scalar_one_or_none()
        except jwt.InvalidTokenError:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token.",
            )

    # Normal user JWT access.
    fingerprint_hash = await get_user_fingerprint_hash(user_id)
    if fingerprint_hash:
        try:
            payload = jwt.decode(
                token, hashlib.sha256(fingerprint_hash.encode()).hexdigest(), algorithms=["HS256"]
            )
            async with get_session() as session:
                return (
                    await session.execute(select(User).where(User.user_id == user_id))
                ).scalar_one_or_none()
        except jwt.ExpiredSignatureError:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Token expired.")
        except jwt.PyJWTError:
            ...
    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Invalid token.",
    )
