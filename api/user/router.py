"""
User routes.
"""

import uuid
import time
import secrets
import hashlib
import orjson as json
from loguru import logger
from datetime import datetime
from typing import Optional
from pydantic import BaseModel
from fastapi import APIRouter, Depends, HTTPException, Header, status, Request
from api.database import get_db_session
from api.user.schemas import (
    UserRequest,
    User,
    AdminUserRequest,
    InvocationQuota,
    InvocationDiscount,
)
from api.user.response import RegistrationResponse, SelfResponse
from api.user.service import get_current_user
from api.user.events import generate_uid as generate_user_uid
from api.user.tokens import create_token
from api.payment.schemas import AdminBalanceChange
from api.logo.schemas import Logo
from sqlalchemy import func
from sqlalchemy.ext.asyncio import AsyncSession
from api.constants import (
    HOTKEY_HEADER,
    COLDKEY_HEADER,
    NONCE_HEADER,
    SIGNATURE_HEADER,
    AUTHORIZATION_HEADER,
)
from api.permissions import Permissioning
from api.config import settings
from api.api_key.schemas import APIKey, APIKeyArgs
from api.api_key.response import APIKeyCreationResponse
from api.user.util import validate_the_username, generate_payment_address
from api.payment.schemas import UsageData
from bittensor_wallet.keypair import Keypair
from scalecodec.utils.ss58 import is_valid_ss58_address
from sqlalchemy import select, text, delete

router = APIRouter()


class FingerprintChange(BaseModel):
    fingerprint: str


class BalanceRequest(BaseModel):
    user_id: str
    amount: float
    reason: str


async def _link_hotkey(
    hotkey: str,
    signature: str,
    attribute: str,
    allow_list: list[str],
    db: AsyncSession,
    current_user: User,
):
    """
    Link a validator or subnet owner hotkey to this account, allowing free usage and developer access.
    """
    signature_string = f"{hotkey}:{current_user.username}"
    if hotkey in allow_list:
        if Keypair(ss58_address=hotkey).verify(signature_string, bytes.fromhex(signature)):
            # Any other accounts already associated?
            existing = (
                await db.execute(
                    select(User)
                    .where(getattr(User, attribute) == hotkey)
                    .where(User.user_id != current_user.user_id)
                )
            ).scalar_one_or_none()
            if existing:
                raise HTTPException(
                    status_code=status.HTTP_409_CONFLICT,
                    detail=f"Hotkey already associated with user {existing.username}",
                )

            # Reload the user since current_user isn't bound to a session, then update.
            user = (
                await db.execute(select(User).where(User.user_id == current_user.user_id))
            ).scalar_one_or_none()
            setattr(user, attribute, hotkey)
            Permissioning.enable(user, Permissioning.free_account)
            Permissioning.enable(user, Permissioning.developer)
            await db.commit()
            await db.refresh(user)
            return user
    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Invalid hotkey or signature.",
    )


@router.get("/growth")
async def get_user_growth(
    db: AsyncSession = Depends(get_db_session),
):
    cache_key = "user_growth".encode()
    cached = await settings.memcache.get(cache_key)
    if cached:
        return json.loads(cached)
    query = text("""
        SELECT
            date(created_at) as date,
            count(*) as daily_count,
            sum(count(*)) OVER (ORDER BY date(created_at)) as cumulative_count
        FROM users
        GROUP BY date(created_at)
        ORDER BY date DESC;
    """)
    result = await db.execute(query)
    rows = result.fetchall()
    response = [
        {
            "date": row.date,
            "daily_count": int(row.daily_count),
            "cumulative_count": int(row.cumulative_count),
        }
        for row in rows
    ]
    await settings.memcache.set(cache_key, json.dumps(response), exptime=600)
    return response


@router.post("/admin_balance_change")
async def admin_balance_change(
    balance_req: BalanceRequest,
    db: AsyncSession = Depends(get_db_session),
    current_user: User = Depends(get_current_user()),
):
    if not current_user.has_role(Permissioning.billing_admin):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="This action can only be performed by billing admin accounts.",
        )
    user = (
        (await db.execute(select(User).where(User.user_id == balance_req.user_id)))
        .unique()
        .scalar_one_or_none()
    )
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail=f"User not found: {balance_req.user_id}"
        )
    user.balance += balance_req.amount
    event_id = str(uuid.uuid4())
    event_data = AdminBalanceChange(
        event_id=event_id,
        user_id=user.user_id,
        amount=balance_req.amount,
        reason=balance_req.reason,
        timestamp=func.now(),
    )
    db.add(event_data)
    await db.commit()
    await db.refresh(user)
    return {"new_balance": user.balance, "event_id": event_id}


@router.post("/{user_id}/quotas")
async def admin_quotas_change(
    user_id: str,
    request: Request,
    db: AsyncSession = Depends(get_db_session),
    current_user: User = Depends(get_current_user()),
):
    if not current_user.has_role(Permissioning.billing_admin):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="This action can only be performed by billing admin accounts.",
        )

    # Validate payload.
    quotas = await request.json()
    for key, value in quotas.items():
        if not isinstance(value, int) or value < 0:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid quota value {key=} {value=}",
            )
        if key == "*":
            continue
        try:
            uuid.UUID(key)
        except ValueError:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid chute_id specified: {key}",
            )

    user = (
        (await db.execute(select(User).where(User.user_id == user_id)))
        .unique()
        .scalar_one_or_none()
    )
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail=f"User not found: {user_id}"
        )

    # Delete old quota values.
    result = await db.execute(
        delete(InvocationQuota)
        .where(InvocationQuota.user_id == user_id)
        .returning(InvocationQuota.chute_id)
    )
    deleted_chute_ids = [row[0] for row in result]

    # Purge the cache.
    for chute_id in deleted_chute_ids:
        key = f"quota:{user_id}:{chute_id}".encode()
        await settings.memcache.delete(key)

    # Add the new values.
    for key, quota in quotas.items():
        db.add(InvocationQuota(user_id=user_id, chute_id=key, quota=quota))
    await db.commit()
    logger.success(f"Updated quotas for {user.user_id=} [{user.username}] to {quotas=}")
    return quotas


@router.post("/{user_id}/discounts")
async def admin_discounts_change(
    user_id: str,
    request: Request,
    db: AsyncSession = Depends(get_db_session),
    current_user: User = Depends(get_current_user()),
):
    if not current_user.has_role(Permissioning.billing_admin):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="This action can only be performed by billing admin accounts.",
        )

    # Validate payload.
    discounts = await request.json()
    for key, value in discounts.items():
        if not isinstance(value, float) or not 0.0 < value < 1.0:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid discount value {key=} {value=}",
            )
        if key == "*":
            continue
        try:
            uuid.UUID(key)
        except ValueError:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid chute_id specified: {key}",
            )

    user = (
        (await db.execute(select(User).where(User.user_id == user_id)))
        .unique()
        .scalar_one_or_none()
    )
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail=f"User not found: {user_id}"
        )

    # Delete old discount values.
    result = await db.execute(
        delete(InvocationDiscount)
        .where(InvocationDiscount.user_id == user_id)
        .returning(InvocationDiscount.chute_id)
    )
    deleted_chute_ids = [row[0] for row in result]
    for chute_id in deleted_chute_ids:
        key = f"idiscount:{user_id}:{chute_id}".encode()
        await settings.memcache.delete(key)

    # Add the new values.
    for key, discount in discounts.items():
        db.add(InvocationDiscount(user_id=user_id, chute_id=key, discount=discount))
    await db.commit()
    logger.success(f"Updated discounts for {user.user_id=} [{user.username}] to {discounts=}")
    return discounts


@router.post("/{user_id}/enable_invoicing", response_model=SelfResponse)
async def admin_enable_invoicing(
    user_id: str,
    request: Request,
    db: AsyncSession = Depends(get_db_session),
    current_user: User = Depends(get_current_user()),
):
    if not current_user.has_role(Permissioning.billing_admin):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="This action can only be performed by billing admin accounts.",
        )
    unlimited = False
    try:
        if (await request.json()).get("unlimited"):
            unlimited = True
    except Exception:
        ...
    user = (
        (await db.execute(select(User).where(User.user_id == user_id)))
        .unique()
        .scalar_one_or_none()
    )
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail=f"User not found: {user_id}"
        )
    Permissioning.enable(user, Permissioning.invoice_billing)
    if unlimited:
        Permissioning.enable(user, Permissioning.unlimited)
    await db.commit()
    await db.refresh(user)
    return user


@router.get("/me", response_model=SelfResponse)
async def me(
    db: AsyncSession = Depends(get_db_session),
    current_user: User = Depends(get_current_user()),
):
    """
    Get a detailed response for the current user.
    """
    return current_user


@router.get("/me/quotas")
async def my_quotas(
    db: AsyncSession = Depends(get_db_session),
    current_user: User = Depends(get_current_user()),
):
    """
    Load quotas for the current user.
    """
    if current_user.has_role(Permissioning.free_account) or current_user.has_role(
        Permissioning.invoice_billing
    ):
        return {}
    quotas = (
        (
            await db.execute(
                select(InvocationQuota).where(InvocationQuota.user_id == current_user.user_id)
            )
        )
        .unique()
        .scalars()
        .all()
    )
    if not quotas:
        return settings.default_quotas
    return {quota.chute_id: quota.quota for quota in quotas}


@router.get("/me/quota_usage/{chute_id}")
async def chute_quota_usage(
    chute_id: str,
    db: AsyncSession = Depends(get_db_session),
    current_user: User = Depends(get_current_user()),
):
    """
    Check the current quota usage for a chute.
    """
    if current_user.has_role(Permissioning.free_account) or current_user.has_role(
        Permissioning.invoice_billing
    ):
        return {"quota": "unlimited", "used": 0}
    quota = await InvocationQuota.get(current_user.user_id, chute_id)
    key = await InvocationQuota.quota_key(current_user.user_id, chute_id)
    used_raw = await settings.quota_client.get(key)
    used = 0.0
    try:
        used = float(used_raw or "0.0")
    except ValueError:
        await settings.quota_client.delete(key)
    return {"quota": quota, "used": used}


@router.delete("/me")
async def delete_my_user(
    db: AsyncSession = Depends(get_db_session),
    authorization: str = Header(
        ..., description="Authorization header", alias=AUTHORIZATION_HEADER
    ),
):
    """
    Delete account.
    """
    fingerprint = authorization.strip().split(" ")[-1]
    fingerprint_hash = hashlib.blake2b(fingerprint.encode()).hexdigest()
    current_user = (
        await db.execute(select(User).where(User.fingerprint_hash == fingerprint_hash))
    ).scalar_one_or_none()
    if not current_user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not authorized",
        )

    await db.execute(
        text("DELETE FROM usage_data WHERE user_id = :user_id"), {"user_id": current_user.user_id}
    )
    await db.execute(
        text("DELETE FROM users WHERE user_id = :user_id"), {"user_id": current_user.user_id}
    )
    await db.commit()
    return {"deleted": True}


@router.get("/set_logo", response_model=SelfResponse)
async def set_logo(
    logo_id: str,
    db: AsyncSession = Depends(get_db_session),
    current_user: User = Depends(get_current_user()),
):
    """
    Get a detailed response for the current user.
    """
    logo = (
        (await db.execute(select(Logo).where(Logo.logo_id == logo_id)))
        .unique()
        .scalar_one_or_none()
    )
    if not logo:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail=f"Logo not found: {logo_id}"
        )
    # Reload user.
    user = (
        (await db.execute(select(User).where(User.user_id == current_user.user_id)))
        .unique()
        .scalar_one_or_none()
    )
    user.logo_id = logo_id
    await db.commit()
    await db.refresh(user)
    return user


@router.get("/link_validator", response_model=SelfResponse)
async def link_validator(
    hotkey: str,
    signature: str,
    db: AsyncSession = Depends(get_db_session),
    current_user: User = Depends(get_current_user(purpose="link_account")),
):
    """
    Link a validator hotkey to this account, allowing free usage.
    """
    return await _link_hotkey(
        hotkey, signature, "validator_hotkey", settings.validators, db, current_user
    )


@router.get("/link_subnet_owner", response_model=SelfResponse)
async def link_subnet_owner(
    hotkey: str,
    signature: str,
    db: AsyncSession = Depends(get_db_session),
    current_user: User = Depends(get_current_user(purpose="link_account")),
):
    """
    Link a subnet owner hotkey to this account, allowing free usage.
    """
    return await _link_hotkey(
        hotkey, signature, "subnet_owner_hotkey", settings.subnet_owners, db, current_user
    )


async def _validate_username(db, username):
    """
    Check validity and availability of a username.
    """
    try:
        validate_the_username(username)
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        ) from e
    existing_user = await db.execute(select(User).where(User.username.ilike(username)))
    if existing_user.first() is not None:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Username {username} already exists, sorry! Please choose another.",
        )


def _registration_response(user, fingerprint):
    """
    Generate a response for a newly registered user.
    """
    return RegistrationResponse(
        username=user.username,
        user_id=user.user_id,
        created_at=user.created_at,
        hotkey=user.hotkey,
        coldkey=user.coldkey,
        payment_address=user.payment_address,
        developer_payment_address=user.developer_payment_address,
        fingerprint=fingerprint,
    )


@router.get("/name_check")
async def check_username(username: str, db: AsyncSession = Depends(get_db_session)):
    """
    Check if a username is valid and available.
    """
    try:
        validate_the_username(username)
    except ValueError:
        return {"valid": False, "available": False}
    existing_user = await db.execute(select(User).where(User.username.ilike(username)))
    if existing_user.first() is not None:
        return {"valid": True, "available": False}
    return {"valid": True, "available": True}


# NOTE: Allow registertation without a hotkey and coldkey, for normal plebs?
@router.post(
    "/register",
    response_model=RegistrationResponse,
)
async def register(
    user_args: UserRequest,
    request: Request,
    db: AsyncSession = Depends(get_db_session),
    current_user: User = Depends(get_current_user(raise_not_found=False)),
    hotkey: str = Header(..., description="The hotkey of the user", alias=HOTKEY_HEADER),
):
    """
    Register a user.
    """
    x_forwarded_for = request.headers.get("X-Forwarded-For")
    actual_ip = x_forwarded_for.split(",")[0] if x_forwarded_for else request.client.host
    attempts = await settings.redis_client.get(f"user_signup:{actual_ip}")
    if attempts and int(attempts) > 3:
        logger.warning(
            f"Attempted multiple registrations from the same IP: {actual_ip} {attempts=}"
        )
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Too may registration requests.",
        )

    if current_user:
        # NOTE: Change when we allow register without a hotkey
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="This hotkey is already registered to a user!",
        )

    # Validate the username
    await _validate_username(db, user_args.username)

    # Create.
    user, fingerprint = User.create(
        username=user_args.username,
        coldkey=user_args.coldkey,
        hotkey=hotkey,
    )
    user.payment_address, user.wallet_secret = await generate_payment_address()
    user.developer_payment_address, user.developer_wallet_secret = await generate_payment_address()
    if settings.all_accounts_free:
        user.permissions_bitmask = 0
        Permissioning.enable(user, Permissioning.free_account)
    db.add(user)
    await db.commit()
    await db.refresh(user)

    await settings.redis_client.incr(f"user_signup:{actual_ip}")
    await settings.redis_client.expire(f"user_signup:{actual_ip}", 24 * 60 * 60)

    return _registration_response(user, fingerprint)


@router.post(
    "/create_user",
    response_model=RegistrationResponse,
)
async def admin_create_user(
    user_args: AdminUserRequest,
    request: Request,
    db: AsyncSession = Depends(get_db_session),
    current_user: User = Depends(get_current_user()),
):
    """
    Create a new user manually from an admin account, no bittensor stuff necessary.
    """
    actual_ip = (
        request.headers.get("CF-Connecting-IP", request.headers.get("X-Forwarded-For"))
        or request.client.host
    )
    actual_ip = actual_ip.split(",")[0]
    logger.info(f"USERCREATION: {actual_ip} username={user_args.username}")

    # Prevent multiple signups from the same IP.
    ip_signups = await settings.redis_client.get(f"ip_signups:{actual_ip}")
    if ip_signups and int(ip_signups) >= 2:
        logger.warning(
            f"Attempted multiple registrations from the same IP: {actual_ip} {ip_signups=}"
        )
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Too may registration requests from this IP.",
        )

    if not current_user.has_role(Permissioning.create_user):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="This action can only be performed by user admin accounts.",
        )

    # Validate the username
    await _validate_username(db, user_args.username)

    # Create the user, faking the hotkey and using the payment address as the coldkey, since this
    # user is API/APP only and not really cognisant of bittensor.
    user, fingerprint = User.create(
        username=user_args.username,
        coldkey=secrets.token_hex(24),
        hotkey=secrets.token_hex(24),
    )
    generate_user_uid(None, None, user)
    user.payment_address, user.wallet_secret = await generate_payment_address()
    user.coldkey = user.payment_address
    user.developer_payment_address, user.developer_wallet_secret = await generate_payment_address()
    if settings.all_accounts_free:
        user.permissions_bitmask = 0
        Permissioning.enable(user, Permissioning.free_account)
    db.add(user)

    # Automatically create an API key for the user as well.
    api_key, one_time_secret = APIKey.create(user.user_id, APIKeyArgs(name="default", admin=True))
    db.add(api_key)
    await db.commit()
    await db.refresh(user)
    await db.refresh(api_key)
    key_response = APIKeyCreationResponse.model_validate(api_key)
    key_response.secret_key = one_time_secret
    response = _registration_response(user, fingerprint)
    response.api_key = key_response

    # Track signups per IP.
    await settings.redis_client.incr(f"ip_signups:{actual_ip}")
    await settings.redis_client.expire(f"ip_signups:{actual_ip}", 24 * 60 * 60)

    return response


@router.post("/change_fingerprint")
async def change_fingerprint(
    args: FingerprintChange,
    db: AsyncSession = Depends(get_db_session),
    hotkey: str | None = Header(None, alias=HOTKEY_HEADER),
    coldkey: str | None = Header(None, alias=COLDKEY_HEADER),
    nonce: str = Header(..., description="Nonce", alias=NONCE_HEADER),
    signature: str = Header(..., description="Hotkey signature", alias=SIGNATURE_HEADER),
):
    """
    Reset a user's fingerprint using either the hotkey or coldkey.
    """
    fingerprint = args.fingerprint

    # Get the signature bytes.
    try:
        signature_hex = bytes.fromhex(signature)
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid signature",
        )

    # Check the nonce.
    valid_nonce = False
    if nonce.isdigit():
        nonce_val = int(nonce)
        now = int(time.time())
        if now - 300 <= nonce_val <= now + 300:
            valid_nonce = True
    if not valid_nonce:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Invalid nonce: {nonce}",
        )
    if not coldkey and not hotkey or not signature:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="You must provide either coldkey or hotkey along with a signature and nonce.",
        )

    # Check hotkey or coldkey, depending on what was passed.
    def _check(header):
        if not header:
            return False
        signing_message = f"{header}:{fingerprint}:{nonce}"
        keypair = Keypair(hotkey)
        try:
            if keypair.verify(signing_message, signature_hex):
                return True
        except Exception:
            ...
        return False

    user = None
    if _check(coldkey):
        user = (
            (await db.execute(select(User).where(User.coldkey == coldkey)))
            .unique()
            .scalar_one_or_none()
        )
    elif _check(hotkey):
        user = (
            (await db.execute(select(User).where(User.hotkey == hotkey)))
            .unique()
            .scalar_one_or_none()
        )
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="No user found with the provided hotkey/coldkey",
        )

    # If we have a user, and the signature passed, we can change the fingerprint.
    user.fingerprint_hash = hashlib.blake2b(fingerprint.encode()).hexdigest()
    await db.commit()
    await db.refresh(user)
    return {"status": "Fingerprint updated"}


@router.post("/login")
async def fingerprint_login(
    request: Request,
    db: AsyncSession = Depends(get_db_session),
):
    """
    Exchange the fingerprint for a JWT.
    """
    body = await request.json()
    fingerprint = body.get("fingerprint")
    if fingerprint and isinstance(fingerprint, str) and fingerprint.strip():
        fingerprint_hash = hashlib.blake2b(fingerprint.encode()).hexdigest()
        user = (
            await db.execute(select(User).where(User.fingerprint_hash == fingerprint_hash))
        ).scalar_one_or_none()
        if user:
            return {
                "token": create_token(user),
            }
    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Missing or invalid fingerprint provided.",
    )


@router.post("/change_bt_auth", response_model=SelfResponse)
async def change_bt_auth(
    request: Request,
    fingerprint: str = Header(alias=AUTHORIZATION_HEADER),
    db: AsyncSession = Depends(get_db_session),
):
    """
    Change the bittensor hotkey/coldkey associated with an account via fingerprint auth.
    """
    body = await request.json()
    fingerprint_hash = hashlib.blake2b(fingerprint.encode()).hexdigest()
    user = (
        await db.execute(select(User).where(User.fingerprint_hash == fingerprint_hash))
    ).scalar_one_or_none()
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing or invalid fingerprint provided.",
        )
    coldkey = body.get("coldkey")
    hotkey = body.get("hotkey")
    changed = False
    error_message = None
    if coldkey:
        if is_valid_ss58_address(coldkey):
            user.coldkey = coldkey
            changed = True
        else:
            error_message = f"Invalid coldkey: {coldkey}"
    if hotkey:
        if is_valid_ss58_address(hotkey):
            existing = (
                await db.execute(select(User).where(User.hotkey == hotkey))
            ).scalar_one_or_none()
            if existing:
                raise HTTPException(
                    status_code=status.HTTP_409_CONFLICT,
                    detail=f"Hotkey already associated with another user: {hotkey}",
                )
            user.hotkey = hotkey
            changed = True
        else:
            error_message = f"Invalid hotkey: {hotkey}"
    if changed:
        await db.commit()
        await db.refresh(user)
        return user
    raise HTTPException(
        status_code=status.HTTP_400_BAD_REQUEST,
        detail=error_message or "Invalid request, please provide a coldkey and/or hotkey",
    )


@router.put("/squad_access")
async def update_squad_access(
    request: Request,
    db: AsyncSession = Depends(get_db_session),
    user: User = Depends(get_current_user()),
):
    """
    Enable squad access.
    """
    user = await db.merge(user)
    body = await request.json()
    if body.get("enable") in (True, "true", "True"):
        user.squad_enabled = True
    elif "enable" in body:
        user.squad_enabled = False
    else:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail='Invalid request, payload should be {"enable": true|false}',
        )
    await db.commit()
    await db.refresh(user)
    return {"squad_enabled": user.squad_enabled}


@router.get("/me/usage")
async def list_usage(
    page: Optional[int] = 0,
    limit: Optional[int] = 24,
    per_chute: Optional[bool] = False,
    chute_id: Optional[str] = None,
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None,
    current_user: User = Depends(get_current_user()),
    db: AsyncSession = Depends(get_db_session),
):
    """
    List usage summary data.
    """
    base_query = select(UsageData).where(UsageData.user_id == current_user.user_id)
    if chute_id:
        base_query = base_query.where(UsageData.chute_id == chute_id)
    if start_date:
        base_query = base_query.where(UsageData.bucket >= start_date)
    if end_date:
        base_query = base_query.where(UsageData.bucket <= end_date)

    if per_chute:
        query = base_query
        total_query = select(func.count()).select_from(query.subquery())
        total_result = await db.execute(total_query)
        total = total_result.scalar() or 0

        query = (
            query.order_by(UsageData.bucket.desc(), UsageData.amount.desc())
            .offset(page * limit)
            .limit(limit)
        )

        results = []
        for data in (await db.execute(query)).unique().scalars().all():
            results.append(
                dict(
                    bucket=data.bucket.isoformat(),
                    chute_id=data.chute_id,
                    amount=data.amount,
                    count=data.count,
                )
            )
    else:
        query = select(
            UsageData.bucket,
            func.sum(UsageData.amount).label("amount"),
            func.sum(UsageData.count).label("count"),
        ).where(UsageData.user_id == current_user.user_id)

        if chute_id:
            query = query.where(UsageData.chute_id == chute_id)
        if start_date:
            query = query.where(UsageData.bucket >= start_date)
        if end_date:
            query = query.where(UsageData.bucket <= end_date)

        query = query.group_by(UsageData.bucket)

        count_subquery = select(UsageData.bucket).where(UsageData.user_id == current_user.user_id)
        if chute_id:
            count_subquery = count_subquery.where(UsageData.chute_id == chute_id)
        if start_date:
            count_subquery = count_subquery.where(UsageData.bucket >= start_date)
        if end_date:
            count_subquery = count_subquery.where(UsageData.bucket <= end_date)

        count_query = select(func.count()).select_from(
            count_subquery.group_by(UsageData.bucket).subquery()
        )

        total_result = await db.execute(count_query)
        total = total_result.scalar() or 0
        query = query.order_by(UsageData.bucket.desc()).offset(page * limit).limit(limit)
        results = []
        for row in (await db.execute(query)).all():
            results.append(
                dict(
                    bucket=row.bucket.isoformat(),
                    amount=row.amount,
                    count=row.count,
                )
            )

    response = {
        "total": total,
        "page": page,
        "limit": limit,
        "items": results,
    }
    return response
