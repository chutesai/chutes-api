"""
Helpers and application logic related to API keys and OAuth access tokens.
"""

import re
import pickle
from async_lru import alru_cache
from sqlalchemy import select
from sqlalchemy.orm import joinedload
from fastapi import Request, HTTPException, status
from api.config import settings
from api.api_key.schemas import APIKey
from api.database import get_session
from api.user.schemas import User


def reinject_dash(uuid_str: str) -> str:
    """
    Re-inject the dashes into a uuid string.
    """
    return f"{uuid_str[0:8]}-{uuid_str[8:12]}-{uuid_str[12:16]}-{uuid_str[16:20]}-{uuid_str[20:32]}"


@alru_cache(maxsize=1000, ttl=60)
async def _load_key(token_id: str):
    """
    Load API key (normal, not oauth2) from database with caching.
    """
    cache_key = f"akey:{token_id}"
    cached = await settings.redis_client.get(cache_key)
    if cached:
        try:
            return pickle.loads(cached)
        except Exception:
            await settings.redis_client.delete(cache_key)
    async with get_session() as session:
        api_key = (
            (
                await session.execute(
                    select(APIKey)
                    .options(joinedload(APIKey.user).joinedload(User.current_balance))
                    .where(APIKey.api_key_id == token_id)
                )
            )
            .unique()
            .scalar_one_or_none()
        )
        if api_key:
            if api_key.user:
                _ = api_key.user.current_balance
                if api_key.user.current_balance:
                    _ = api_key.user.current_balance.effective_balance
            serialized = pickle.dumps(api_key)
            await settings.redis_client.set(cache_key, serialized, ex=60)
        return api_key


async def get_user_from_oauth_token(token: str, request: Request):
    """
    Validate an OAuth access token (cak_ prefix) and return the user and scopes.
    Returns (None, None) if invalid.
    """
    from api.idp.schemas import OAuthAccessToken

    if not OAuthAccessToken.could_be_valid(token):
        return None, None

    from api.idp.service import validate_access_token

    result = await validate_access_token(token)
    if not result:
        return None, None

    # Result is now a TokenValidationResult with user and scopes
    user = result.user
    scopes = result.scopes

    # Store OAuth context on request state
    request.state.oauth_token = True
    request.state.oauth_scopes = scopes

    return user, scopes


class OAuthTokenWrapper:
    """
    Wrapper to make OAuth tokens behave like API keys for compatibility.
    Implements the same interface as APIKey for scope checking.
    """

    def __init__(self, user, scopes: list):
        self.user = user
        self.scopes = scopes or []
        self.admin = "admin" in self.scopes

    def has_access(self, object_type: str, object_id: str, action: str) -> bool:
        """
        Check if the OAuth token has access to the specified resource.
        """
        from api.idp.schemas import check_scope_access

        if self.admin:
            return True

        if object_type == "chutes" and object_id in (
            "__megallm__",
            "__megadiffuser__",
            "__megaembed__",
        ):
            if check_scope_access(self.scopes, "chutes", None, "invoke"):
                return True

        return check_scope_access(self.scopes, object_type, object_id, action)


async def get_and_check_api_key(key: str, request: Request):
    """
    Check the API key, check scopes, then fetch the user (oauth tokens AND API keys).
    """
    # OAuth2 style tokens.
    if key.startswith("cak_"):
        user, scopes = await get_user_from_oauth_token(key, request)
        if user:
            wrapper = OAuthTokenWrapper(user, scopes)
            if not wrapper.has_access(
                request.state.auth_object_type,
                request.state.auth_object_id,
                request.state.auth_method,
            ):
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="Token does not have permission for this resource",
                )
            return wrapper
        return None

    # Normal API keys.
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

    return api_token
