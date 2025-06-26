"""
Helpers and application logic related to API keys.
"""

import re
import pickle
from async_lru import alru_cache
from sqlalchemy import select
from fastapi import Request, HTTPException, status
from api.config import settings
from api.api_key.schemas import APIKey
from api.database import get_session


def reinject_dash(uuid_str: str) -> str:
    """
    Re-inject the dashes into a uuid string.
    """
    return f"{uuid_str[0:8]}-{uuid_str[8:12]}-{uuid_str[12:16]}-{uuid_str[16:20]}-{uuid_str[20:32]}"


@alru_cache(maxsize=1000, ttl=600)
async def _load_key(token_id: str):
    """
    Load API key from database with memcache caching.
    """
    cache_key = f"api_key:{token_id}".encode()
    cached = await settings.memcache.get(cache_key)
    if cached:
        try:
            return pickle.loads(cached)
        except Exception:
            await settings.memcache.delete(cache_key)
    async with get_session() as session:
        api_key = (
            (await session.execute(select(APIKey).where(APIKey.api_key_id == token_id)))
            .unique()
            .scalar_one_or_none()
        )
        if api_key:
            serialized = pickle.dumps(api_key)
            await settings.memcache.set(cache_key, serialized, exptime=600)
        return api_key


async def get_and_check_api_key(key: str, request: Request):
    """
    Take the `key` from the authorization header which comprosises of the user_id and token_id,
    then check them against the available scopes.
    """
    if not APIKey.could_be_valid(key):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing or invalid authorization header(s)",
        )
    part_match = re.match(r"^cpk_([a-f0-9]{32})\.([a-f0-9]{32})\.([a-zA-Z0-9]{32})$", key)
    if not part_match:
        return False
    token_id, user_id, _ = part_match.groups()
    user_id = reinject_dash(user_id)
    token_id = reinject_dash(token_id)

    api_token = await _load_key(token_id)
    if not api_token or not api_token.verify(key):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token or user not found",
        )
    if not api_token.has_access(
        request.state.auth_object_type,
        request.state.auth_object_id,
        request.state.auth_method,
    ):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token or user not found",
        )

    # TODO: Add checking of the user_id?
    return api_token
