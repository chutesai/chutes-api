"""
Service layer for OAuth2/IDP functionality.
"""

import hashlib
import json
import uuid
from datetime import datetime, timedelta, timezone
from typing import List, Optional, Tuple
from passlib.hash import argon2
from sqlalchemy import select
from sqlalchemy.orm import joinedload

from api.config import settings
from api.database import get_session
from api.idp.schemas import (
    OAuthAccessToken,
    OAuthApp,
    OAuthAuthorization,
    OAuthAuthorizationCode,
    OAuthRefreshToken,
)
from api.user.schemas import User
from api.constants import (
    ACCESS_TOKEN_EXPIRY_SECONDS,
    AUTH_CODE_EXPIRY_SECONDS,
    LOGIN_NONCE_EXPIRY_SECONDS,
)


async def create_login_nonce() -> str:
    """
    Create a nonce for hotkey signature login.
    Returns a UUID4 string that's stored in Redis with a TTL.
    """
    nonce = str(uuid.uuid4())
    await settings.redis_client.set(
        f"idp:login_nonce:{nonce}",
        "pending",
        ex=LOGIN_NONCE_EXPIRY_SECONDS,
    )
    return nonce


async def verify_and_consume_login_nonce(nonce: str) -> bool:
    """
    Verify a login nonce exists and hasn't been used.
    Consumes the nonce (deletes it) if valid.
    """
    key = f"idp:login_nonce:{nonce}"
    value = await settings.redis_client.get(key)
    if value:
        await settings.redis_client.delete(key)
        return True
    return False


async def get_app_by_client_id(client_id: str) -> Optional[OAuthApp]:
    """Load an OAuth app by client_id with caching."""
    # Cache the app_id to avoid repeated lookups, but always load fresh from DB
    # to ensure we have proper ORM relationships
    cache_key = f"idp:app_id:{client_id}"
    cached = await settings.redis_client.get(cache_key)

    async with get_session() as session:
        if cached:
            app_id = cached.decode() if isinstance(cached, bytes) else cached
            if app_id == "__none__":
                return None
            app = (
                (
                    await session.execute(
                        select(OAuthApp)
                        .options(joinedload(OAuthApp.user))
                        .where(OAuthApp.app_id == app_id, OAuthApp.active.is_(True))
                    )
                )
                .unique()
                .scalar_one_or_none()
            )
            if app:
                return app
            # App was deleted or deactivated, clear cache
            await settings.redis_client.delete(cache_key)

        # Full lookup by client_id
        app = (
            (
                await session.execute(
                    select(OAuthApp)
                    .options(joinedload(OAuthApp.user))
                    .where(OAuthApp.client_id == client_id, OAuthApp.active.is_(True))
                )
            )
            .unique()
            .scalar_one_or_none()
        )
        if app:
            await settings.redis_client.set(cache_key, app.app_id, ex=300)
        else:
            # Cache negative result briefly
            await settings.redis_client.set(cache_key, "__none__", ex=60)
        return app


async def invalidate_app_cache(client_id: str):
    """Invalidate the cache for an OAuth app."""
    await settings.redis_client.delete(f"idp:app_id:{client_id}")


async def create_authorization_code(
    app_id: str,
    user_id: str,
    redirect_uri: str,
    scopes: list[str],
    state: Optional[str] = None,
    code_challenge: Optional[str] = None,
    code_challenge_method: Optional[str] = None,
) -> str:
    """
    Create an authorization code for the OAuth2 authorization code flow.
    Stored in Redis with TTL (ephemeral, single-use).
    Returns the plain code (to be sent to client).
    """
    code = OAuthAuthorizationCode.generate_code()
    code_hash = OAuthAuthorizationCode.hash_code(code)

    auth_code = OAuthAuthorizationCode(
        app_id=app_id,
        user_id=user_id,
        redirect_uri=redirect_uri,
        scopes=scopes,
        state=state,
        code_challenge=code_challenge,
        code_challenge_method=code_challenge_method,
    )

    await settings.redis_client.set(
        OAuthAuthorizationCode.redis_key(code_hash),
        auth_code.to_json(),
        ex=AUTH_CODE_EXPIRY_SECONDS,
    )

    return code


async def exchange_authorization_code(
    code: str,
    client_id: str,
    client_secret: Optional[str],
    redirect_uri: str,
    code_verifier: Optional[str] = None,
) -> Tuple[Optional[str], Optional[str], Optional[int], Optional[str]]:
    """
    Exchange an authorization code for access and refresh tokens.
    Returns (access_token, refresh_token, expires_in, error).
    """
    # Look up the code in Redis (atomic get-and-delete for single use)
    code_hash = OAuthAuthorizationCode.hash_code(code)
    redis_key = OAuthAuthorizationCode.redis_key(code_hash)
    code_data = await settings.redis_client.getdel(redis_key)

    if not code_data:
        return None, None, None, "invalid_grant"

    try:
        auth_code = OAuthAuthorizationCode.from_json(code_data)
    except Exception:
        return None, None, None, "invalid_grant"

    # Verify redirect_uri matches
    if auth_code.redirect_uri != redirect_uri:
        return None, None, None, "invalid_grant"

    # Load the app to verify client credentials
    app = await get_app_by_client_id(client_id)
    if not app or app.app_id != auth_code.app_id:
        return None, None, None, "invalid_client"

    # Verify client secret or PKCE
    if auth_code.code_challenge:
        # PKCE flow
        if not code_verifier:
            return None, None, None, "invalid_grant"

        if auth_code.code_challenge_method == "S256":
            expected = hashlib.sha256(code_verifier.encode()).hexdigest()
            if expected != auth_code.code_challenge:
                return None, None, None, "invalid_grant"
        elif auth_code.code_challenge_method == "plain":
            if code_verifier != auth_code.code_challenge:
                return None, None, None, "invalid_grant"
    else:
        # Standard flow - verify client secret
        if not client_secret or not app.verify_secret(client_secret):
            return None, None, None, "invalid_client"

    async with get_session() as session:
        # Get or create authorization
        auth = (
            (
                await session.execute(
                    select(OAuthAuthorization)
                    .options(joinedload(OAuthAuthorization.app))
                    .where(
                        OAuthAuthorization.user_id == auth_code.user_id,
                        OAuthAuthorization.app_id == auth_code.app_id,
                    )
                )
            )
            .unique()
            .scalar_one_or_none()
        )

        if not auth:
            auth = OAuthAuthorization(
                user_id=auth_code.user_id,
                app_id=auth_code.app_id,
                scopes=auth_code.scopes,
            )
            session.add(auth)
            await session.flush()
            # Load app relationship for refresh token lifetime
            await session.refresh(auth, ["app"])
        elif auth.revoked:
            # Re-enable if previously revoked
            auth.revoked = False
            auth.revoked_at = None
            auth.scopes = auth_code.scopes

        # Create access token
        access_token = OAuthAccessToken.generate_token()
        access_token_hash = OAuthAccessToken.hash_token(access_token)
        access_expires = datetime.now(timezone.utc) + timedelta(seconds=ACCESS_TOKEN_EXPIRY_SECONDS)

        access_token_obj = OAuthAccessToken(
            token_hash=access_token_hash,
            authorization_id=auth.authorization_id,
            scopes=auth_code.scopes,
            expires_at=access_expires,
        )
        session.add(access_token_obj)

        # Create refresh token with per-app configurable lifetime
        refresh_token = OAuthRefreshToken.generate_token()
        refresh_token_hash = OAuthRefreshToken.hash_token(refresh_token)
        refresh_lifetime_days = auth.app.refresh_token_lifetime_days
        refresh_expires = datetime.now(timezone.utc) + timedelta(days=refresh_lifetime_days)

        refresh_token_obj = OAuthRefreshToken(
            token_hash=refresh_token_hash,
            authorization_id=auth.authorization_id,
            expires_at=refresh_expires,
        )
        session.add(refresh_token_obj)

        await session.commit()

        return access_token, refresh_token, ACCESS_TOKEN_EXPIRY_SECONDS, None


async def refresh_access_token(
    refresh_token: str,
    client_id: str,
    client_secret: Optional[str],
) -> Tuple[Optional[str], Optional[str], Optional[int], Optional[str]]:
    """
    Use a refresh token to get a new access token.
    Returns (access_token, refresh_token, expires_in, error).
    """
    async with get_session() as session:
        # Find the refresh token
        result = await session.execute(
            select(OAuthRefreshToken)
            .options(joinedload(OAuthRefreshToken.authorization).joinedload(OAuthAuthorization.app))
            .where(
                OAuthRefreshToken.used.is_(False),
                OAuthRefreshToken.revoked.is_(False),
            )
        )
        refresh_tokens = result.unique().scalars().all()

        # Find matching token by verifying hash
        refresh_obj = None
        for rt in refresh_tokens:
            try:
                if argon2.verify(refresh_token, rt.token_hash):
                    refresh_obj = rt
                    break
            except Exception:
                continue

        if not refresh_obj:
            return None, None, None, "invalid_grant"

        # Check expiration
        if refresh_obj.expires_at.replace(tzinfo=timezone.utc) < datetime.now(timezone.utc):
            refresh_obj.revoked = True
            await session.commit()
            return None, None, None, "invalid_grant"

        auth = refresh_obj.authorization

        # Check authorization isn't revoked
        if auth.revoked:
            return None, None, None, "invalid_grant"

        # Verify client
        if auth.app.client_id != client_id:
            return None, None, None, "invalid_client"

        # Verify client secret if provided
        if client_secret and not auth.app.verify_secret(client_secret):
            return None, None, None, "invalid_client"

        # Mark old refresh token as used
        refresh_obj.used = True

        # Create new access token
        new_access_token = OAuthAccessToken.generate_token()
        access_token_hash = OAuthAccessToken.hash_token(new_access_token)
        access_expires = datetime.now(timezone.utc) + timedelta(seconds=ACCESS_TOKEN_EXPIRY_SECONDS)

        access_token_obj = OAuthAccessToken(
            token_hash=access_token_hash,
            authorization_id=auth.authorization_id,
            scopes=auth.scopes,
            expires_at=access_expires,
        )
        session.add(access_token_obj)

        # Create new refresh token (token rotation with sliding expiration)
        # Each refresh grants a fresh lifetime based on app config
        new_refresh_token = OAuthRefreshToken.generate_token()
        new_refresh_hash = OAuthRefreshToken.hash_token(new_refresh_token)
        refresh_lifetime_days = auth.app.refresh_token_lifetime_days
        refresh_expires = datetime.now(timezone.utc) + timedelta(days=refresh_lifetime_days)

        new_refresh_obj = OAuthRefreshToken(
            token_hash=new_refresh_hash,
            authorization_id=auth.authorization_id,
            expires_at=refresh_expires,
        )
        session.add(new_refresh_obj)

        await session.commit()

        return new_access_token, new_refresh_token, ACCESS_TOKEN_EXPIRY_SECONDS, None


class TokenValidationResult:
    """Result of token validation including user and scopes."""

    def __init__(self, user: User, scopes: List[str], authorization: OAuthAuthorization = None):
        self.user = user
        self.scopes = scopes or []
        self.authorization = authorization


async def validate_access_token(token: str) -> Optional[TokenValidationResult]:
    """
    Validate an access token and return the associated user and scopes.
    Returns None if invalid.
    """
    if not OAuthAccessToken.could_be_valid(token):
        return None

    # Check cache first
    cache_key = f"idp:token:{hashlib.sha256(token.encode()).hexdigest()[:32]}"
    cached = await settings.redis_client.get(cache_key)
    if cached:
        try:
            data = json.loads(cached.decode() if isinstance(cached, bytes) else cached)
            if data.get("valid"):
                async with get_session() as session:
                    user = (
                        await session.execute(select(User).where(User.user_id == data["user_id"]))
                    ).scalar_one_or_none()
                    if user:
                        # Return cached scopes
                        return TokenValidationResult(
                            user=user,
                            scopes=data.get("scopes", []),
                        )
            return None
        except Exception:
            await settings.redis_client.delete(cache_key)

    async with get_session() as session:
        # Find the token
        result = await session.execute(
            select(OAuthAccessToken)
            .options(
                joinedload(OAuthAccessToken.authorization).joinedload(OAuthAuthorization.user),
                joinedload(OAuthAccessToken.authorization).joinedload(OAuthAuthorization.app),
            )
            .where(
                OAuthAccessToken.revoked.is_(False),
            )
        )
        tokens = result.unique().scalars().all()

        # Find matching token
        token_obj = None
        for t in tokens:
            try:
                if argon2.verify(token, t.token_hash):
                    token_obj = t
                    break
            except Exception:
                continue

        if not token_obj:
            # Cache negative result briefly
            await settings.redis_client.set(cache_key, json.dumps({"valid": False}), ex=60)
            return None

        # Check expiration
        if token_obj.expires_at.replace(tzinfo=timezone.utc) < datetime.now(timezone.utc):
            return None

        auth = token_obj.authorization

        # Check authorization isn't revoked
        if auth.revoked:
            return None

        # Check app is active
        if not auth.app.active:
            return None

        # Use the scopes from the access token (not the authorization)
        # This allows for per-token scope restrictions
        token_scopes = token_obj.scopes or []

        # Cache positive result with scopes
        ttl_seconds = int(
            (
                token_obj.expires_at.replace(tzinfo=timezone.utc) - datetime.now(timezone.utc)
            ).total_seconds()
        )
        await settings.redis_client.set(
            cache_key,
            json.dumps(
                {
                    "valid": True,
                    "user_id": auth.user_id,
                    "scopes": token_scopes,
                }
            ),
            ex=min(300, max(1, ttl_seconds)),
        )

        return TokenValidationResult(
            user=auth.user,
            scopes=token_scopes,
            authorization=auth,
        )


async def revoke_authorization(user_id: str, app_id: str) -> bool:
    """Revoke a user's authorization for an app."""
    async with get_session() as session:
        auth = (
            (
                await session.execute(
                    select(OAuthAuthorization).where(
                        OAuthAuthorization.user_id == user_id,
                        OAuthAuthorization.app_id == app_id,
                    )
                )
            )
            .unique()
            .scalar_one_or_none()
        )

        if not auth:
            return False

        auth.revoked = True
        auth.revoked_at = datetime.now(timezone.utc)

        # Revoke all tokens
        await session.execute(
            select(OAuthAccessToken).where(
                OAuthAccessToken.authorization_id == auth.authorization_id
            )
        )
        for token in (
            (
                await session.execute(
                    select(OAuthAccessToken).where(
                        OAuthAccessToken.authorization_id == auth.authorization_id
                    )
                )
            )
            .scalars()
            .all()
        ):
            token.revoked = True

        for token in (
            (
                await session.execute(
                    select(OAuthRefreshToken).where(
                        OAuthRefreshToken.authorization_id == auth.authorization_id
                    )
                )
            )
            .scalars()
            .all()
        ):
            token.revoked = True

        await session.commit()
        return True


async def revoke_token(token: str) -> bool:
    """Revoke a specific token (access or refresh)."""
    async with get_session() as session:
        # Try access token first
        if token.startswith("cak_"):
            result = await session.execute(select(OAuthAccessToken))
            for t in result.scalars().all():
                try:
                    if argon2.verify(token, t.token_hash):
                        t.revoked = True
                        await session.commit()
                        return True
                except Exception:
                    continue

        # Try refresh token
        if token.startswith("crt_"):
            result = await session.execute(select(OAuthRefreshToken))
            for t in result.scalars().all():
                try:
                    if argon2.verify(token, t.token_hash):
                        t.revoked = True
                        await session.commit()
                        return True
                except Exception:
                    continue

        return False
