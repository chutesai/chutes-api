"""
Per-endpoint rate limiting for FastAPI.

Add to any route with Depends(rate_limit("endpoint_key", requests_per_minute=60)), or
Depends(rate_limit_miner(...)) to authenticate a miner and meter that miner individually.
Redis-backed so limits are shared across API replicas. Pass 0 to disable.
"""

import time
from typing import Callable, Optional

from fastapi import Depends, Header, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from api.config import settings
from api.constants import HOTKEY_HEADER
from api.database import get_db_session
from api.miner.util import is_miner_blacklisted
from api.user.schemas import User
from api.user.service import get_current_user


async def check_rate_limit(
    endpoint_key: str,
    limit: int,
    window_seconds: int = 60,
    identity: str = "",
) -> None:
    """
    Count one call against a fixed window and raise 429 once it exceeds ``limit``.

    Shared by both dependencies below. An ``identity`` gives each caller their own counter;
    without one the counter is global to the endpoint. Pass 0 for ``limit`` to disable.
    """
    if limit <= 0:
        return
    window = int(time.time() // window_seconds)
    key = (
        f"rate_limit:{endpoint_key}:{identity}:{window}"
        if identity
        else f"rate_limit:{endpoint_key}:{window}"
    )
    redis = settings.redis_client
    count = await redis.incr(key)
    if count == 1:
        await redis.expire(key, window_seconds * 2)
    if count > limit:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Rate limit exceeded. Try again later.",
        )


def rate_limit(endpoint_key: str, requests_per_minute: int) -> Callable:
    """
    FastAPI dependency that enforces a rate limit for this endpoint.

    Usage:
        @router.get("/evidence")
        async def get_evidence(
            _: None = Depends(rate_limit("tee_evidence", 60)),
            ...
        ):
            ...
    """

    async def _rate_limit() -> None:
        await check_rate_limit(endpoint_key, requests_per_minute, window_seconds=60)

    return _rate_limit


def rate_limit_miner(
    endpoint_key: str,
    limit: int,
    window_seconds: int = 60,
    global_limit: int = 0,
) -> Callable:
    """
    FastAPI dependency that authenticates a miner and meters that miner's own quota.

    Authentication is a SUB-dependency here rather than a sibling ``Depends`` on the route, and
    that ordering is the point: FastAPI resolves sub-dependencies first, so the request signature
    is verified and the hotkey confirmed registered and un-blacklisted before anything is counted.
    Siblings have no guaranteed order, and metering an unverified hotkey header would let anyone
    exhaust a real miner's quota by spoofing it.

    Returns the authenticated user, so a route declares auth and metering in one dependency:

        @router.post("/tdx/host_profiles")
        async def submit_host_profile(
            _: User = Depends(rate_limit_miner("host_profile_submit", 10, window_seconds=3600)),
            ...
        ):
            ...

    ``global_limit`` adds a ceiling across all miners over the same window, metered here rather
    than via a second ``Depends(rate_limit(...))`` for the same ordering reason: a pre-auth global
    counter can be run down by unauthenticated junk, locking out every legitimate miner.

    Note that the user may be None: a hotkey registered on the subnet does not necessarily have a
    users row, and ``get_current_user`` returns None rather than raising in that case whenever
    ``registered_to`` is set.
    """

    async def _rate_limit_miner(
        hotkey: str | None = Header(None, alias=HOTKEY_HEADER),
        db: AsyncSession = Depends(get_db_session),
        user: Optional[User] = Depends(get_current_user(registered_to=settings.netuid)),
    ) -> Optional[User]:
        reason = await is_miner_blacklisted(db, hotkey)
        if reason:
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=reason)
        await check_rate_limit(endpoint_key, limit, window_seconds=window_seconds, identity=hotkey)
        await check_rate_limit(
            f"{endpoint_key}:global", global_limit, window_seconds=window_seconds
        )
        return user

    return _rate_limit_miner
