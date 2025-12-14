"""
OAuth2/IDP Router for authentication and authorization endpoints.
"""

import base64
import hashlib
import json
import uuid
from datetime import datetime, timezone
from typing import Optional
from urllib.parse import quote, urlencode

from bittensor_wallet.keypair import Keypair
from fastapi import APIRouter, Depends, Form, HTTPException, Query, Request, status
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from loguru import logger
from sqlalchemy import or_, select, func
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import joinedload

from api.config import settings
from api.database import get_db_session
from api.pagination import PaginatedResponse
from api.permissions import Permissioning
from api.user.schemas import User
from api.user.service import get_current_user
from api.user.tokens import get_user_from_token, create_token
from api.idp.schemas import (
    OAuthAccessToken,
    OAuthApp,
    OAuthAppCreateRequest,
    OAuthAppShare,
    OAuthAppShareArgs,
    OAuthAppUpdateRequest,
    OAuthAuthorization,
    OAuthRefreshToken,
    get_available_scopes,
    get_scope_descriptions,
    validate_requested_scopes,
)
from api.idp.response import (
    OAuthAppCreationResponse,
    OAuthAppResponse,
    OAuthAppSecretRegenerateResponse,
    OAuthAuthorizationResponse,
    TokenResponse,
)
from api.idp.service import (
    create_authorization_code,
    create_login_nonce,
    exchange_authorization_code,
    get_app_by_client_id,
    invalidate_app_cache,
    refresh_access_token,
    revoke_authorization,
    revoke_token,
    validate_access_token,
    verify_and_consume_login_nonce,
)
from api.idp.templater import authorize_page, error_page, login_page

router = APIRouter()


@router.get("/scopes")
async def list_scopes():
    """
    List all available OAuth2 scopes with descriptions.
    This endpoint is public and can be used for documentation or scope selection UIs.
    """
    return {"scopes": get_available_scopes()}


@router.get("/cli_login/nonce")
async def get_cli_login_nonce():
    """
    Get a nonce for CLI-based hotkey signature login.
    """
    nonce = await create_login_nonce()
    return {"nonce": nonce}


@router.get("/cli_login", response_class=HTMLResponse)
async def cli_login(
    hotkey: str = Query(...),
    signature: str = Query(...),
    nonce: str = Query(...),
    db: AsyncSession = Depends(get_db_session),
):
    """
    CLI login endpoint for hotkey signature authentication.
    """
    # Verify nonce exists and hasn't been used
    if not await verify_and_consume_login_nonce(nonce):
        return HTMLResponse(
            content=error_page("invalid_nonce", "Invalid or expired nonce. Please try again."),
            status_code=400,
        )

    # Verify signature
    try:
        signature_bytes = bytes.fromhex(signature)
        keypair = Keypair(hotkey)
        if not keypair.verify(nonce, signature_bytes):
            return HTMLResponse(
                content=error_page("invalid_signature", "Invalid signature."),
                status_code=400,
            )
    except Exception as e:
        logger.warning(f"CLI login signature verification failed: {e}")
        return HTMLResponse(
            content=error_page("invalid_signature", "Invalid signature format."),
            status_code=400,
        )

    # Find user by hotkey
    user = (await db.execute(select(User).where(User.hotkey == hotkey))).scalar_one_or_none()

    if not user:
        return HTMLResponse(
            content=error_page("no_account", "No account found for this hotkey."),
            status_code=404,
        )

    # Create session token and set cookie
    session_token = create_token(user)
    response = RedirectResponse(
        url=f"https://{settings.base_domain}/app",
        status_code=302,
    )
    response.set_cookie(
        key="chutes-session-token",
        value=session_token,
        max_age=7 * 24 * 60 * 60,  # 7 days
        httponly=True,
        secure=True,
        samesite="lax",
        domain=f".{settings.base_domain}",
    )
    return response


@router.get("/apps", response_model=PaginatedResponse)
async def list_apps(
    include_public: Optional[bool] = True,
    include_shared: Optional[bool] = True,
    search: Optional[str] = None,
    page: Optional[int] = 0,
    limit: Optional[int] = 25,
    user_id: Optional[str] = None,
    db: AsyncSession = Depends(get_db_session),
    current_user: User = Depends(get_current_user()),
):
    """
    List OAuth applications.

    By default, returns apps owned by the current user, public apps, and apps shared with the user.
    Set include_public=false to exclude public apps.
    Set include_shared=false to exclude apps shared with the user.
    Use search to filter by name or description.
    """
    if user_id == "me":
        user_id = current_user.user_id
    if (
        user_id
        and user_id != current_user.user_id
        and not current_user.has_role(Permissioning.billing_admin)
    ):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="This action can only be performed by billing admin accounts.",
        )
    if not user_id:
        user_id = current_user.user_id

    query = select(OAuthApp).where(OAuthApp.active.is_(True))

    # Build visibility conditions
    if include_public or include_shared:
        visibility_conditions = [OAuthApp.user_id == user_id]

        if include_public:
            visibility_conditions.append(OAuthApp.public.is_(True))

        if include_shared:
            shared_subquery = select(OAuthAppShare.app_id).where(
                OAuthAppShare.shared_to == current_user.user_id
            )
            visibility_conditions.append(OAuthApp.app_id.in_(shared_subquery))

        query = query.where(or_(*visibility_conditions))
    else:
        query = query.where(OAuthApp.user_id == user_id)

    # Filter by app names.
    if search and search.strip():
        search_filter = or_(
            OAuthApp.name.ilike(f"%{search}%"),
            OAuthApp.description.ilike(f"%{search}%"),
            OAuthApp.homepage_url.ilike(f"%{search}%"),
        )
        query = query.where(search_filter)

    # Pagination.
    total_query = select(func.count()).select_from(query.subquery())
    total_result = await db.execute(total_query)
    total = total_result.scalar() or 0
    query = query.order_by(OAuthApp.created_at.desc())
    query = query.offset((page or 0) * (limit or 25)).limit(limit or 25)

    result = await db.execute(query)
    items = [OAuthAppResponse.model_validate(app) for app in result.scalars().unique().all()]
    return {
        "total": total,
        "page": page,
        "limit": limit,
        "items": items,
    }


@router.post("/apps", response_model=OAuthAppCreationResponse)
async def create_app(
    args: OAuthAppCreateRequest,
    db: AsyncSession = Depends(get_db_session),
    current_user: User = Depends(get_current_user()),
):
    """Create a new OAuth application."""
    existing = (
        await db.execute(
            select(OAuthApp).where(
                OAuthApp.name.ilike(args.name.strip()),
            )
        )
    ).scalar_one_or_none()
    if existing:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="An application with this name already exists",
        )

    app, client_secret = OAuthApp.create(current_user.user_id, args)
    db.add(app)
    await db.commit()
    await db.refresh(app)

    response = OAuthAppCreationResponse.model_validate(app)
    response.client_secret = client_secret
    return response


@router.get("/apps/{app_id}", response_model=OAuthAppResponse)
async def get_app(
    app_id: str,
    db: AsyncSession = Depends(get_db_session),
    current_user: User = Depends(get_current_user()),
):
    """Get details of an OAuth application."""
    app = (
        await db.execute(
            select(OAuthApp).where(
                OAuthApp.app_id == app_id,
                OAuthApp.user_id == current_user.user_id,
            )
        )
    ).scalar_one_or_none()

    if not app:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Application not found",
        )

    return OAuthAppResponse.model_validate(app)


@router.patch("/apps/{app_id}", response_model=OAuthAppResponse)
async def update_app(
    app_id: str,
    args: OAuthAppUpdateRequest,
    db: AsyncSession = Depends(get_db_session),
    current_user: User = Depends(get_current_user()),
):
    """Update an OAuth application."""
    app = (
        await db.execute(
            select(OAuthApp).where(
                OAuthApp.app_id == app_id,
                OAuthApp.user_id == current_user.user_id,
            )
        )
    ).scalar_one_or_none()

    if not app:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Application not found",
        )

    # Check for duplicate name if changing
    if args.name and args.name != app.name:
        existing = (
            await db.execute(
                select(OAuthApp).where(
                    OAuthApp.user_id == current_user.user_id,
                    OAuthApp.name == args.name,
                    OAuthApp.app_id != app_id,
                )
            )
        ).scalar_one_or_none()

        if existing:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="An application with this name already exists",
            )

    # Update fields
    if args.name is not None:
        app.name = args.name
    if args.description is not None:
        app.description = args.description
    if args.redirect_uris is not None:
        app.redirect_uris = args.redirect_uris
    if args.homepage_url is not None:
        app.homepage_url = args.homepage_url
    if args.logo_url is not None:
        app.logo_url = args.logo_url
    if args.active is not None:
        app.active = args.active
    if args.public is not None:
        app.public = args.public
    if args.refresh_token_lifetime_days is not None:
        app.refresh_token_lifetime_days = args.refresh_token_lifetime_days

    await db.commit()
    await db.refresh(app)
    await invalidate_app_cache(app.client_id)

    return OAuthAppResponse.model_validate(app)


@router.delete("/apps/{app_id}")
async def delete_app(
    app_id: str,
    db: AsyncSession = Depends(get_db_session),
    current_user: User = Depends(get_current_user()),
):
    """Delete an OAuth application."""
    app = (
        await db.execute(
            select(OAuthApp).where(
                OAuthApp.app_id == app_id,
                OAuthApp.user_id == current_user.user_id,
            )
        )
    ).scalar_one_or_none()

    if not app:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Application not found",
        )

    await invalidate_app_cache(app.client_id)
    await db.delete(app)
    await db.commit()

    return {"app_id": app_id, "deleted": True}


@router.post("/apps/{app_id}/regenerate-secret", response_model=OAuthAppSecretRegenerateResponse)
async def regenerate_app_secret(
    app_id: str,
    db: AsyncSession = Depends(get_db_session),
    current_user: User = Depends(get_current_user()),
):
    """Regenerate the client secret for an OAuth application."""
    app = (
        await db.execute(
            select(OAuthApp).where(
                OAuthApp.app_id == app_id,
                OAuthApp.user_id == current_user.user_id,
            )
        )
    ).scalar_one_or_none()

    if not app:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Application not found",
        )

    new_secret = app.regenerate_secret()
    await db.commit()
    await invalidate_app_cache(app.client_id)

    return OAuthAppSecretRegenerateResponse(
        app_id=app.app_id,
        client_id=app.client_id,
        client_secret=new_secret,
    )


@router.post("/apps/{app_id}/share")
async def share_app(
    app_id: str,
    args: OAuthAppShareArgs,
    db: AsyncSession = Depends(get_db_session),
    current_user: User = Depends(get_current_user()),
):
    """Share an OAuth application with another user."""
    # Find the app (must be owned by current user)
    app = (
        await db.execute(
            select(OAuthApp).where(
                OAuthApp.app_id == app_id,
                OAuthApp.user_id == current_user.user_id,
            )
        )
    ).scalar_one_or_none()

    if not app:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Application not found",
        )

    # Find the target user
    target_user = (
        await db.execute(
            select(User).where(
                (User.user_id == args.user_id_or_name) | (User.username == args.user_id_or_name)
            )
        )
    ).scalar_one_or_none()

    if not target_user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found",
        )

    if target_user.user_id == current_user.user_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Cannot share app with yourself",
        )

    # Check if already shared
    existing = (
        await db.execute(
            select(OAuthAppShare).where(
                OAuthAppShare.app_id == app_id,
                OAuthAppShare.shared_to == target_user.user_id,
            )
        )
    ).scalar_one_or_none()

    if existing:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="App is already shared with this user",
        )

    # Create share
    share = OAuthAppShare(
        app_id=app_id,
        shared_by=current_user.user_id,
        shared_to=target_user.user_id,
    )
    db.add(share)
    await db.commit()

    return {
        "app_id": app_id,
        "shared_to": target_user.user_id,
        "shared_to_username": target_user.username,
    }


@router.delete("/apps/{app_id}/share/{user_id}")
async def unshare_app(
    app_id: str,
    user_id: str,
    db: AsyncSession = Depends(get_db_session),
    current_user: User = Depends(get_current_user()),
):
    """Remove sharing of an OAuth application with a user."""
    # Find the app (must be owned by current user)
    app = (
        await db.execute(
            select(OAuthApp).where(
                OAuthApp.app_id == app_id,
                OAuthApp.user_id == current_user.user_id,
            )
        )
    ).scalar_one_or_none()

    if not app:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Application not found",
        )

    # Find and delete the share
    share = (
        await db.execute(
            select(OAuthAppShare).where(
                OAuthAppShare.app_id == app_id,
                OAuthAppShare.shared_to == user_id,
            )
        )
    ).scalar_one_or_none()

    if not share:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Share not found",
        )

    await db.delete(share)
    await db.commit()

    return {"app_id": app_id, "unshared_from": user_id}


@router.get("/apps/{app_id}/shares")
async def list_app_shares(
    app_id: str,
    db: AsyncSession = Depends(get_db_session),
    current_user: User = Depends(get_current_user()),
):
    """List users an OAuth application is shared with."""
    # Find the app (must be owned by current user)
    app = (
        await db.execute(
            select(OAuthApp).where(
                OAuthApp.app_id == app_id,
                OAuthApp.user_id == current_user.user_id,
            )
        )
    ).scalar_one_or_none()

    if not app:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Application not found",
        )

    # Get all shares
    result = await db.execute(
        select(OAuthAppShare)
        .options(joinedload(OAuthAppShare.shared_to_user))
        .where(OAuthAppShare.app_id == app_id)
    )
    shares = result.scalars().unique().all()

    return {
        "app_id": app_id,
        "shares": [
            {
                "user_id": s.shared_to,
                "username": s.shared_to_user.username if s.shared_to_user else None,
                "shared_at": s.shared_at,
            }
            for s in shares
        ],
    }


@router.get("/authorizations", response_model=PaginatedResponse)
async def list_authorizations(
    page: Optional[int] = 0,
    limit: Optional[int] = 25,
    db: AsyncSession = Depends(get_db_session),
    current_user: User = Depends(get_current_user()),
):
    """List apps the current user has authorized."""
    query = (
        select(OAuthAuthorization)
        .options(joinedload(OAuthAuthorization.app))
        .where(
            OAuthAuthorization.user_id == current_user.user_id,
            OAuthAuthorization.revoked.is_(False),
        )
    )

    total_query = select(func.count()).select_from(
        select(OAuthAuthorization)
        .where(
            OAuthAuthorization.user_id == current_user.user_id,
            OAuthAuthorization.revoked.is_(False),
        )
        .subquery()
    )
    total_result = await db.execute(total_query)
    total = total_result.scalar() or 0

    query = query.order_by(OAuthAuthorization.created_at.desc())
    query = query.offset((page or 0) * (limit or 25)).limit(limit or 25)

    result = await db.execute(query)
    items = []
    for auth in result.scalars().unique().all():
        items.append(
            OAuthAuthorizationResponse(
                authorization_id=auth.authorization_id,
                app_id=auth.app_id,
                app_name=auth.app.name if auth.app else "Unknown",
                app_description=auth.app.description if auth.app else None,
                app_logo_url=auth.app.logo_url if auth.app else None,
                scopes=auth.scopes,
                created_at=auth.created_at,
                revoked=auth.revoked,
            )
        )

    return {
        "total": total,
        "page": page,
        "limit": limit,
        "items": items,
    }


@router.delete("/authorizations/{app_id}")
async def revoke_app_authorization(
    app_id: str,
    db: AsyncSession = Depends(get_db_session),
    current_user: User = Depends(get_current_user()),
):
    """Revoke authorization for an app."""
    success = await revoke_authorization(current_user.user_id, app_id)
    if not success:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Authorization not found",
        )
    return {"app_id": app_id, "revoked": True}


@router.get("/authorize", response_class=HTMLResponse)
async def authorize_get(
    request: Request,
    response_type: str = Query(...),
    client_id: str = Query(...),
    redirect_uri: str = Query(...),
    scope: Optional[str] = Query(None),
    state: Optional[str] = Query(None),
    code_challenge: Optional[str] = Query(None),
    code_challenge_method: Optional[str] = Query(None),
    db: AsyncSession = Depends(get_db_session),
):
    """
    OAuth2 Authorization Endpoint.
    Displays login page if not authenticated, consent page if authenticated.
    Checks for existing chutes-session-token cookie for SSO.
    """
    # Validate response_type
    if response_type != "code":
        return HTMLResponse(
            content=error_page(
                "unsupported_response_type", "Only 'code' response type is supported"
            ),
            status_code=400,
        )

    # Validate client
    app = await get_app_by_client_id(client_id)
    if not app:
        return HTMLResponse(
            content=error_page("invalid_client", "Unknown client_id"),
            status_code=400,
        )

    # Validate redirect_uri
    if not app.is_valid_redirect_uri(redirect_uri):
        return HTMLResponse(
            content=error_page(
                "invalid_redirect_uri", "Redirect URI not registered for this application"
            ),
            status_code=400,
        )

    # Validate PKCE if provided
    if code_challenge and code_challenge_method not in ("plain", "S256"):
        return HTMLResponse(
            content=error_page("invalid_request", "Invalid code_challenge_method"),
            status_code=400,
        )

    # Check for existing session cookie (SSO)
    session_token = request.cookies.get("chutes-session-token")
    if session_token:
        try:
            # Mock request state for get_user_from_token
            request.state.auth_method = "read"
            user = await get_user_from_token(session_token, request)
            if user:
                # User is already authenticated - skip to consent page
                session_id = str(uuid.uuid4())
                session_data = json.dumps(
                    {
                        "user_id": user.user_id,
                        "client_id": client_id,
                        "redirect_uri": redirect_uri,
                        "scope": scope or "",
                        "state": state or "",
                        "code_challenge": code_challenge or "",
                        "code_challenge_method": code_challenge_method or "",
                    }
                )
                await settings.redis_client.set(
                    f"idp:session:{session_id}",
                    session_data,
                    ex=600,
                )
                return RedirectResponse(
                    url=f"/idp/authorize/consent?session_id={session_id}",
                    status_code=302,
                )
        except Exception:
            # Invalid session token - fall through to login page
            pass

    # Build the current authorization URL for the create account redirect
    current_url = str(request.url)

    # Generate login nonce for hotkey auth
    nonce = await create_login_nonce()

    # Show login page
    return HTMLResponse(
        content=login_page(
            client_id=client_id,
            redirect_uri=redirect_uri,
            state=state or "",
            scope=scope or "",
            app_name=app.name,
            app_description=app.description or "",
            nonce=nonce,
            code_challenge=code_challenge or "",
            code_challenge_method=code_challenge_method or "",
            create_account_url=f"https://{settings.base_domain}/auth/start?redirect_to={quote(current_url, safe='')}",
            login_url=f"https://{settings.base_domain}/auth?redirect_to={quote(current_url, safe='')}",
        )
    )


@router.post("/login", response_class=HTMLResponse)
async def login_post(
    request: Request,
    client_id: str = Form(...),
    redirect_uri: str = Form(...),
    state: str = Form(""),
    scope: str = Form(""),
    auth_method: str = Form(...),
    code_challenge: str = Form(""),
    code_challenge_method: str = Form(""),
    fingerprint: Optional[str] = Form(None),
    hotkey: Optional[str] = Form(None),
    signature: Optional[str] = Form(None),
    nonce: Optional[str] = Form(None),
    db: AsyncSession = Depends(get_db_session),
):
    """Handle login form submission."""
    # Validate client
    app = await get_app_by_client_id(client_id)
    if not app:
        return HTMLResponse(
            content=error_page("invalid_client", "Unknown client_id"),
            status_code=400,
        )

    # Validate redirect_uri
    if not app.is_valid_redirect_uri(redirect_uri):
        return HTMLResponse(
            content=error_page("invalid_redirect_uri", "Redirect URI not registered"),
            status_code=400,
        )

    user = None

    if auth_method == "fingerprint":
        # Fingerprint authentication
        if not fingerprint:
            new_nonce = await create_login_nonce()
            return HTMLResponse(
                content=login_page(
                    client_id=client_id,
                    redirect_uri=redirect_uri,
                    state=state,
                    scope=scope,
                    app_name=app.name,
                    app_description=app.description or "",
                    nonce=new_nonce,
                    error="Please enter your fingerprint",
                    code_challenge=code_challenge,
                    code_challenge_method=code_challenge_method,
                )
            )

        fingerprint_hash = hashlib.blake2b(fingerprint.encode()).hexdigest()
        user = (
            await db.execute(select(User).where(User.fingerprint_hash == fingerprint_hash))
        ).scalar_one_or_none()

        if not user:
            new_nonce = await create_login_nonce()
            return HTMLResponse(
                content=login_page(
                    client_id=client_id,
                    redirect_uri=redirect_uri,
                    state=state,
                    scope=scope,
                    app_name=app.name,
                    app_description=app.description or "",
                    nonce=new_nonce,
                    error="Invalid fingerprint",
                    code_challenge=code_challenge,
                    code_challenge_method=code_challenge_method,
                )
            )

    elif auth_method == "hotkey":
        # Hotkey signature authentication
        if not hotkey or not signature or not nonce:
            new_nonce = await create_login_nonce()
            return HTMLResponse(
                content=login_page(
                    client_id=client_id,
                    redirect_uri=redirect_uri,
                    state=state,
                    scope=scope,
                    app_name=app.name,
                    app_description=app.description or "",
                    nonce=new_nonce,
                    error="Please provide hotkey, signature, and sign the nonce",
                    code_challenge=code_challenge,
                    code_challenge_method=code_challenge_method,
                )
            )

        # Verify nonce exists and hasn't been used
        if not await verify_and_consume_login_nonce(nonce):
            new_nonce = await create_login_nonce()
            return HTMLResponse(
                content=login_page(
                    client_id=client_id,
                    redirect_uri=redirect_uri,
                    state=state,
                    scope=scope,
                    app_name=app.name,
                    app_description=app.description or "",
                    nonce=new_nonce,
                    error="Invalid or expired nonce. Please try again.",
                    code_challenge=code_challenge,
                    code_challenge_method=code_challenge_method,
                )
            )

        # Verify signature
        try:
            signature_bytes = bytes.fromhex(signature)
            keypair = Keypair(hotkey)
            if not keypair.verify(nonce, signature_bytes):
                new_nonce = await create_login_nonce()
                return HTMLResponse(
                    content=login_page(
                        client_id=client_id,
                        redirect_uri=redirect_uri,
                        state=state,
                        scope=scope,
                        app_name=app.name,
                        app_description=app.description or "",
                        nonce=new_nonce,
                        error="Invalid signature",
                        code_challenge=code_challenge,
                        code_challenge_method=code_challenge_method,
                    )
                )
        except Exception as e:
            logger.warning(f"Hotkey signature verification failed: {e}")
            new_nonce = await create_login_nonce()
            return HTMLResponse(
                content=login_page(
                    client_id=client_id,
                    redirect_uri=redirect_uri,
                    state=state,
                    scope=scope,
                    app_name=app.name,
                    app_description=app.description or "",
                    nonce=new_nonce,
                    error="Invalid signature format",
                    code_challenge=code_challenge,
                    code_challenge_method=code_challenge_method,
                )
            )

        # Find user by hotkey
        user = (await db.execute(select(User).where(User.hotkey == hotkey))).scalar_one_or_none()

        if not user:
            new_nonce = await create_login_nonce()
            return HTMLResponse(
                content=login_page(
                    client_id=client_id,
                    redirect_uri=redirect_uri,
                    state=state,
                    scope=scope,
                    app_name=app.name,
                    app_description=app.description or "",
                    nonce=new_nonce,
                    error="No account found for this hotkey",
                    code_challenge=code_challenge,
                    code_challenge_method=code_challenge_method,
                )
            )
    else:
        return HTMLResponse(
            content=error_page("invalid_request", "Invalid authentication method"),
            status_code=400,
        )

    # User authenticated successfully - store session with full authorization context
    # This prevents session replay attacks where attacker reuses session_id with different params
    session_id = str(uuid.uuid4())
    session_data = json.dumps(
        {
            "user_id": user.user_id,
            "client_id": client_id,
            "redirect_uri": redirect_uri,
            "scope": scope,
            "state": state,
            "code_challenge": code_challenge,
            "code_challenge_method": code_challenge_method,
        }
    )
    await settings.redis_client.set(
        f"idp:session:{session_id}",
        session_data,
        ex=600,  # 10 minutes
    )

    # Create session token for SSO
    session_token = create_token(user)

    # Auto-submit to consent page, setting the session cookie for future SSO
    response = RedirectResponse(
        url=f"/idp/authorize/consent?session_id={session_id}",
        status_code=302,
    )
    response.set_cookie(
        key="chutes-session-token",
        value=session_token,
        max_age=7 * 24 * 60 * 60,  # 7 days (matches token expiry)
        httponly=True,
        secure=True,
        samesite="lax",
        domain=f".{settings.base_domain}",
    )
    return response


async def _get_session_data(session_id: str) -> Optional[dict]:
    """Get and parse session data from Redis."""
    if not session_id:
        return None
    session_raw = await settings.redis_client.get(f"idp:session:{session_id}")
    if not session_raw:
        return None
    try:
        data = session_raw.decode() if isinstance(session_raw, bytes) else session_raw
        return json.loads(data)
    except Exception:
        return None


@router.get("/authorize/consent", response_class=HTMLResponse)
async def authorize_consent_page(
    session_id: str = Query(...),
    db: AsyncSession = Depends(get_db_session),
):
    """Show authorization consent page."""
    session_data = await _get_session_data(session_id)
    if not session_data:
        return HTMLResponse(
            content=error_page("invalid_request", "Session expired. Please try again."),
            status_code=400,
        )

    user_id = session_data["user_id"]
    client_id = session_data["client_id"]
    redirect_uri = session_data["redirect_uri"]
    scope = session_data.get("scope", "")
    code_challenge = session_data.get("code_challenge", "")
    code_challenge_method = session_data.get("code_challenge_method", "")

    app = await get_app_by_client_id(client_id)
    if not app:
        return HTMLResponse(
            content=error_page("invalid_client", "Unknown client_id"),
            status_code=400,
        )

    user = (await db.execute(select(User).where(User.user_id == user_id))).scalar_one_or_none()
    if not user:
        return HTMLResponse(
            content=error_page("invalid_request", "User not found"),
            status_code=400,
        )

    scopes_list = scope.split() if scope else ["profile"]

    return HTMLResponse(
        content=authorize_page(
            client_id=client_id,
            redirect_uri=redirect_uri,
            state=session_data.get("state", ""),
            scope=scope,
            app_name=app.name,
            app_description=app.description or "",
            app_logo_url=app.logo_url or "",
            user_name=user.username,
            scopes=get_scope_descriptions(scopes_list),
            code_challenge=code_challenge,
            code_challenge_method=code_challenge_method,
            refresh_token_lifetime_days=app.refresh_token_lifetime_days,
        ).replace(
            'action="/idp/authorize/consent"',
            f'action="/idp/authorize/consent?session_id={session_id}"',
        )
    )


@router.post("/authorize/consent", response_class=HTMLResponse)
async def authorize_consent(
    session_id: str = Query(...),
    action: str = Form(...),
    db: AsyncSession = Depends(get_db_session),
):
    """Handle authorization consent form submission."""
    # Get session data (contains all authorization context)
    session_data = await _get_session_data(session_id)
    if not session_data:
        return HTMLResponse(
            content=error_page("invalid_request", "Session expired. Please try again."),
            status_code=400,
        )

    # Extract data from session (NOT from form - prevents tampering)
    user_id = session_data["user_id"]
    client_id = session_data["client_id"]
    redirect_uri = session_data["redirect_uri"]
    scope = session_data.get("scope", "")
    state = session_data.get("state", "")
    code_challenge = session_data.get("code_challenge", "")
    code_challenge_method = session_data.get("code_challenge_method", "")

    # Get app
    app = await get_app_by_client_id(client_id)
    if not app:
        return HTMLResponse(
            content=error_page("invalid_client", "Unknown client_id"),
            status_code=400,
        )

    # Get user
    user = (await db.execute(select(User).where(User.user_id == user_id))).scalar_one_or_none()
    if not user:
        return HTMLResponse(
            content=error_page("invalid_request", "User not found"),
            status_code=400,
        )

    # Clean up session (single use)
    await settings.redis_client.delete(f"idp:session:{session_id}")

    if action == "deny":
        # User denied - redirect with error
        params = {"error": "access_denied", "error_description": "User denied the request"}
        if state:
            params["state"] = state
        return RedirectResponse(
            url=f"{redirect_uri}?{urlencode(params)}",
            status_code=302,
        )

    # User approved - validate and create authorization code
    scopes_list = scope.split() if scope else ["profile"]

    # Validate scopes against app's allowed scopes
    is_valid, error_msg, validated_scopes = validate_requested_scopes(
        scopes_list, app.allowed_scopes
    )
    if not is_valid:
        params = {"error": "invalid_scope", "error_description": error_msg}
        if state:
            params["state"] = state
        return RedirectResponse(
            url=f"{redirect_uri}?{urlencode(params)}",
            status_code=302,
        )

    code = await create_authorization_code(
        app_id=app.app_id,
        user_id=user.user_id,
        redirect_uri=redirect_uri,
        scopes=validated_scopes,
        state=state,
        code_challenge=code_challenge if code_challenge else None,
        code_challenge_method=code_challenge_method if code_challenge_method else None,
    )

    # Redirect with code
    params = {"code": code}
    if state:
        params["state"] = state

    return RedirectResponse(
        url=f"{redirect_uri}?{urlencode(params)}",
        status_code=302,
    )


@router.post("/token")
async def token_endpoint(
    request: Request,
    grant_type: str = Form(...),
    code: Optional[str] = Form(None),
    redirect_uri: Optional[str] = Form(None),
    client_id: Optional[str] = Form(None),
    client_secret: Optional[str] = Form(None),
    refresh_token: Optional[str] = Form(None),
    code_verifier: Optional[str] = Form(None),
):
    """OAuth2 Token Endpoint."""
    # Support client credentials in Authorization header
    auth_header = request.headers.get("Authorization")
    if auth_header and auth_header.startswith("Basic "):
        try:
            decoded = base64.b64decode(auth_header[6:]).decode()
            header_client_id, header_client_secret = decoded.split(":", 1)
            client_id = client_id or header_client_id
            client_secret = client_secret or header_client_secret
        except Exception:
            pass

    if grant_type == "authorization_code":
        if not code or not redirect_uri or not client_id:
            return JSONResponse(
                content={
                    "error": "invalid_request",
                    "error_description": "Missing required parameters",
                },
                status_code=400,
            )

        access_token, refresh_tok, expires_in, scopes, error = await exchange_authorization_code(
            code=code,
            client_id=client_id,
            client_secret=client_secret,
            redirect_uri=redirect_uri,
            code_verifier=code_verifier,
        )

        if error:
            return JSONResponse(
                content={"error": error},
                status_code=400,
            )

        return TokenResponse(
            access_token=access_token,
            token_type="Bearer",
            expires_in=expires_in,
            refresh_token=refresh_tok,
            scope=" ".join(scopes) if scopes else None,
        )

    elif grant_type == "refresh_token":
        if not refresh_token or not client_id:
            return JSONResponse(
                content={
                    "error": "invalid_request",
                    "error_description": "Missing required parameters",
                },
                status_code=400,
            )

        access_token, new_refresh, expires_in, scopes, error = await refresh_access_token(
            refresh_token=refresh_token,
            client_id=client_id,
            client_secret=client_secret,
        )

        if error:
            return JSONResponse(
                content={"error": error},
                status_code=400,
            )

        return TokenResponse(
            access_token=access_token,
            token_type="Bearer",
            expires_in=expires_in,
            refresh_token=new_refresh,
            scope=" ".join(scopes) if scopes else None,
        )

    else:
        return JSONResponse(
            content={"error": "unsupported_grant_type"},
            status_code=400,
        )


@router.post("/token/revoke")
async def revoke_token_endpoint(
    token: str = Form(...),
    token_type_hint: Optional[str] = Form(None),
):
    """OAuth2 Token Revocation Endpoint (RFC 7009)."""
    await revoke_token(token)
    # Always return 200 per RFC 7009, even if token not found
    return {"revoked": True}


@router.get("/userinfo")
async def userinfo_endpoint(
    request: Request,
    db: AsyncSession = Depends(get_db_session),
):
    """OpenID Connect UserInfo Endpoint."""
    # Get token from Authorization header
    auth_header = request.headers.get("Authorization")
    if not auth_header or not auth_header.startswith("Bearer "):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing or invalid authorization header",
        )

    token = auth_header[7:]
    result = await validate_access_token(token)

    if not result:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token",
        )

    # Check for required scope (openid or profile)
    has_required_scope = any(
        s in result.scopes for s in ("openid", "profile", "account:read", "admin")
    )
    if not has_required_scope:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Token does not have required scope (openid, profile, or account:read)",
        )

    user = result.user

    return {
        "sub": user.user_id,
        "username": user.username,
        "created_at": user.created_at.isoformat() if user.created_at else None,
    }


@router.post("/token/introspect")
async def introspect_token(
    token: str = Form(...),
    token_type_hint: Optional[str] = Form(None),
    client_id: Optional[str] = Form(None),
    client_secret: Optional[str] = Form(None),
    db: AsyncSession = Depends(get_db_session),
):
    """
    OAuth2 Token Introspection Endpoint (RFC 7662).

    Token format includes embedded token_id for O(1) lookup, so client auth is optional.

    Allows clients to check if a token is still valid and get metadata about it.
    Useful for determining if a user needs to re-authenticate.

    Returns:
        - active: Whether the token is currently valid
        - exp: Expiration timestamp (Unix epoch)
        - iat: Issued at timestamp
        - scope: Space-separated list of scopes
        - client_id: The client that the token was issued to
        - username: The user's username
        - sub: The user's ID
    """
    # Optionally verify client credentials if provided
    if client_id and client_secret:
        app = await get_app_by_client_id(client_id)
        if not app or not app.verify_secret(client_secret):
            return JSONResponse(
                content={"error": "invalid_client"},
                status_code=401,
            )

    # Check access token - O(1) lookup by embedded token_id
    if token.startswith("cak_") or token_type_hint == "access_token":
        token_id, _ = OAuthAccessToken.parse_token(token)
        if not token_id:
            return {"active": False}

        token_result = await db.execute(
            select(OAuthAccessToken)
            .options(
                joinedload(OAuthAccessToken.authorization).joinedload(OAuthAuthorization.user),
                joinedload(OAuthAccessToken.authorization).joinedload(OAuthAuthorization.app),
            )
            .where(OAuthAccessToken.token_id == token_id)
        )
        token_obj = token_result.unique().scalar_one_or_none()

        if token_obj:
            # Verify token secret (single argon2 verify)
            if not token_obj.verify_secret(token):
                return {"active": False}

            # Check if revoked or expired
            if token_obj.revoked:
                return {"active": False}

            exp_dt = token_obj.expires_at.replace(tzinfo=timezone.utc)
            if exp_dt < datetime.now(timezone.utc):
                return {"active": False}

            if token_obj.authorization.revoked:
                return {"active": False}

            iat_dt = (
                token_obj.created_at.replace(tzinfo=timezone.utc) if token_obj.created_at else None
            )

            return {
                "active": True,
                "token_type": "access_token",
                "exp": int(exp_dt.timestamp()),
                "iat": int(iat_dt.timestamp()) if iat_dt else None,
                "scope": " ".join(token_obj.scopes or []),
                "client_id": token_obj.authorization.app.client_id,
                "username": token_obj.authorization.user.username,
                "sub": token_obj.authorization.user.user_id,
            }

    # Check refresh token - O(1) lookup by embedded token_id
    if token.startswith("crt_") or token_type_hint == "refresh_token":
        token_id, _ = OAuthRefreshToken.parse_token(token)
        if not token_id:
            return {"active": False}

        token_result = await db.execute(
            select(OAuthRefreshToken)
            .options(
                joinedload(OAuthRefreshToken.authorization).joinedload(OAuthAuthorization.user),
                joinedload(OAuthRefreshToken.authorization).joinedload(OAuthAuthorization.app),
            )
            .where(OAuthRefreshToken.token_id == token_id)
        )
        token_obj = token_result.unique().scalar_one_or_none()

        if token_obj:
            # Verify token secret (single argon2 verify)
            if not token_obj.verify_secret(token):
                return {"active": False}

            # Check if used, revoked, or expired
            if token_obj.used or token_obj.revoked:
                return {"active": False}

            exp_dt = token_obj.expires_at.replace(tzinfo=timezone.utc)
            if exp_dt < datetime.now(timezone.utc):
                return {"active": False}

            if token_obj.authorization.revoked:
                return {"active": False}

            iat_dt = (
                token_obj.created_at.replace(tzinfo=timezone.utc) if token_obj.created_at else None
            )

            return {
                "active": True,
                "token_type": "refresh_token",
                "exp": int(exp_dt.timestamp()),
                "iat": int(iat_dt.timestamp()) if iat_dt else None,
                "scope": " ".join(token_obj.authorization.scopes or []),
                "client_id": token_obj.authorization.app.client_id,
                "username": token_obj.authorization.user.username,
                "sub": token_obj.authorization.user.user_id,
            }

    # Token is not active (invalid, expired, or revoked)
    return {"active": False}
