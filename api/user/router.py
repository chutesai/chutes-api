"""
User routes.
"""

import re
import uuid
import time
import aiohttp
import secrets
import hashlib
import orjson as json
from loguru import logger
from datetime import datetime
from typing import Optional
from pydantic import BaseModel
from collections import defaultdict
from fastapi import APIRouter, Depends, HTTPException, Header, status, Request
from fastapi.responses import HTMLResponse
from api.database import get_db_session
from api.chute.schemas import ChuteShare
from api.user.schemas import (
    UserRequest,
    User,
    PriceOverride,
    AdminUserRequest,
    InvocationQuota,
    InvocationDiscount,
)
from api.util import (
    is_cloudflare_ip,
    has_minimum_balance_for_registration,
)
from api.user.response import RegistrationResponse, SelfResponse
from api.user.service import get_current_user, bt_user_exists
from api.user.events import generate_uid as generate_user_uid
from api.user.tokens import create_token
from api.payment.schemas import AdminBalanceChange
from api.logo.schemas import Logo
from sqlalchemy import func, or_, and_
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm.attributes import flag_modified
from api.constants import (
    HOTKEY_HEADER,
    COLDKEY_HEADER,
    NONCE_HEADER,
    SIGNATURE_HEADER,
    AUTHORIZATION_HEADER,
    MIN_REG_BALANCE,
)
from api.permissions import Permissioning
from api.config import settings
from api.api_key.schemas import APIKey, APIKeyArgs
from api.api_key.response import APIKeyCreationResponse
from api.user.util import validate_the_username, generate_payment_address
from api.user.templater import registration_token_form, registration_token_success, error_page
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


class SubnetRoleRequest(BaseModel):
    user: str
    netuid: int
    admin: bool


class SubnetRoleRevokeRequest(BaseModel):
    user: str
    netuid: int


@router.get("/growth")
async def get_user_growth(
    db: AsyncSession = Depends(get_db_session),
):
    cache_key = "user_growth"
    cached = await settings.redis_client.get(cache_key)
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
    await settings.redis_client.set(cache_key, json.dumps(response), ex=600)
    return response


@router.get("/{user_id}/shares")
async def list_chute_shares(
    user_id: str,
    db: AsyncSession = Depends(get_db_session),
    current_user: User = Depends(get_current_user()),
):
    if user_id == "me":
        user_id = current_user.user_id
    if user_id != current_user.user_id and not current_user.has_role(Permissioning.billing_admin):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="This action can only be performed by billing admin accounts.",
        )
    shares = (
        (await db.execute(select(ChuteShare).where(ChuteShare.shared_by == user_id)))
        .unique()
        .scalars()
        .all()
    )
    return shares


@router.get("/user_id_lookup")
async def admin_user_id_lookup(
    username: str,
    db: AsyncSession = Depends(get_db_session),
    current_user: User = Depends(get_current_user()),
):
    if not current_user.has_role(Permissioning.billing_admin):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="This action can only be performed by billing admin accounts.",
        )
    user = (
        (await db.execute(select(User).where(User.username == username)))
        .unique()
        .scalar_one_or_none()
    )
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail=f"User not found: {username}"
        )
    return {"user_id": user.user_id}


@router.get("/{user_id_or_username}/balance")
async def admin_balance_lookup(
    user_id_or_username: str,
    db: AsyncSession = Depends(get_db_session),
    current_user: User = Depends(get_current_user()),
):
    if not current_user.has_role(Permissioning.billing_admin):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="This action can only be performed by billing admin accounts.",
        )
    user = (
        (
            await db.execute(
                select(User).where(
                    or_(User.username == user_id_or_username, User.user_id == user_id_or_username)
                )
            )
        )
        .unique()
        .scalar_one_or_none()
    )
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail=f"User not found: {user_id_or_username}"
        )
    return {
        "user_id": user.user_id,
        "balance": user.current_balance.effective_balance if user.current_balance else 0.0,
    }


@router.get("/invoiced_user_list", response_model=list[SelfResponse])
async def admin_invoiced_user_list(
    db: AsyncSession = Depends(get_db_session),
    current_user: User = Depends(get_current_user()),
):
    if not current_user.has_role(Permissioning.billing_admin):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="This action can only be performed by billing admin accounts.",
        )
    query = select(User).where(
        and_(
            User.permissions_bitmask.op("&")(Permissioning.invoice_billing.bitmask) != 0,
            User.permissions_bitmask.op("&")(Permissioning.free_account.bitmask) == 0,
            User.user_id != "5682c3e0-3635-58f7-b7f5-694962450dfc",
        )
    )
    result = await db.execute(query)
    users = []
    for user in result.unique().scalars().all():
        ur = SelfResponse.from_orm(user)
        ur.balance = user.current_balance.effective_balance if user.current_balance else 0.0
        users.append(ur)
    return users


@router.post("/batch_user_lookup")
async def admin_batch_user_lookup(
    request: Request,
    db: AsyncSession = Depends(get_db_session),
    current_user: User = Depends(get_current_user()),
):
    if not current_user.has_role(Permissioning.billing_admin):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="This action can only be performed by billing admin accounts.",
        )
    body = await request.json()
    user_ids = body.get("user_ids") or []
    if not isinstance(user_ids, list):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="`user_ids` must be a list.",
        )
    if not user_ids:
        return []

    # Fetch all requested users
    user_query = select(User).where(User.user_id.in_(user_ids))
    result = await db.execute(user_query)
    db_users = result.unique().scalars().all()
    if not db_users:
        return []
    users_by_id = {u.user_id: u for u in db_users}
    ordered_users = [users_by_id[uid] for uid in user_ids if uid in users_by_id]
    quota_query = select(InvocationQuota).where(
        InvocationQuota.user_id.in_([u.user_id for u in db_users])
    )
    quota_result = await db.execute(quota_query)
    all_quotas = quota_result.scalars().all()
    quotas_by_user = defaultdict(list)
    for q in all_quotas:
        quotas_by_user[q.user_id].append(q)

    users = []
    for user in ordered_users:
        ur = SelfResponse.from_orm(user)
        ur.balance = (
            user.current_balance.effective_balance
            if getattr(user, "current_balance", None)
            else 0.0
        )
        user_quota_entries = []
        if user.has_role(Permissioning.free_account) or user.has_role(
            Permissioning.invoice_billing
        ):
            user_quota_entries.append(
                {
                    "chute_id": None,
                    "quota": "unlimited",
                    "used": 0.0,
                }
            )
        else:
            for quota in quotas_by_user.get(user.user_id, []):
                key = await InvocationQuota.quota_key(user.user_id, quota.chute_id)
                used_raw = await settings.redis_client.get(key)
                used = 0.0
                try:
                    used = float(used_raw or "0.0")
                except (TypeError, ValueError):
                    if used_raw is not None:
                        await settings.redis_client.delete(key)
                user_quota_entries.append(
                    {
                        "chute_id": quota.chute_id,
                        "quota": quota.quota,
                        "used": used,
                    }
                )
        ur.quotas = user_quota_entries
        users.append(ur)
    return users


@router.post("/admin_balance_change")
async def admin_balance_change(
    request: Request,
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

    origin_ip = request.headers.get("x-forwarded-for", "").split(",")[0]
    raw_request_data = {}
    try:
        payload = await request.json()
        raw_request_data = {
            "source_ip": origin_ip,
            "payload": payload,
            "headers": {
                k: v
                for k, v in request.headers.items()
                if k.lower() not in {"authorization", "x-api-key", "x-auth-token"}
            },
        }
    except Exception as exc:
        logger.error(f"Error gathering raw request data: {str(exc)}")
    logger.warning(
        f"admin_balance_change requested from {current_user.user_id=} "
        f"{current_user.username=} {origin_ip=} {balance_req=} {event_id=}"
    )

    event_data = AdminBalanceChange(
        event_id=event_id,
        user_id=user.user_id,
        amount=balance_req.amount,
        reason=balance_req.reason,
        timestamp=func.now(),
        created_by=current_user.user_id,
        raw_request=raw_request_data,
    )
    db.add(event_data)
    await db.commit()
    await db.refresh(user)
    return {"new_balance": user.balance, "event_id": event_id}


@router.post("/grant_subnet_role")
async def grant_subnet_role(
    args: SubnetRoleRequest,
    db: AsyncSession = Depends(get_db_session),
    current_user: User = Depends(get_current_user()),
):
    # Validate permissions to make the request.
    if not current_user.has_role(Permissioning.subnet_admin_assign) or args.netuid not in (
        current_user.netuids or []
    ):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Missing required subnet admin role assign permissions.",
        )

    # Load the target user.
    user = (
        (
            await db.execute(
                select(User)
                .where(or_(User.user_id == args.user, User.username == args.user))
                .limit(1)
            )
        )
        .unique()
        .scalar_one_or_none()
    )
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Target user not found.",
        )

    # Enable the role (and netuid on the user if necessary).
    role = Permissioning.subnet_invoke if not args.admin else Permissioning.subnet_admin
    Permissioning.enable(user, role)
    netuids = user.netuids or []
    if args.netuid not in netuids:
        netuids.append(args.netuid)
    user.netuids = netuids
    flag_modified(user, "netuids")
    await db.commit()
    return {
        "status": f"Successfully enabled {role.description=} {role.bitmask=} for {user.user_id=} {user.username=} on {args.netuid=}"
    }


@router.post("/revoke_subnet_role")
async def revoke_subnet_role(
    args: SubnetRoleRevokeRequest,
    db: AsyncSession = Depends(get_db_session),
    current_user: User = Depends(get_current_user()),
):
    # Validate permissions to make the request.
    if not current_user.has_role(Permissioning.subnet_admin_assign) or args.netuid not in (
        current_user.netuids or []
    ):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Missing required subnet admin role assign permissions.",
        )

    # Load the target user.
    user = (
        (
            await db.execute(
                select(User)
                .where(or_(User.user_id == args.user, User.username == args.user))
                .limit(1)
            )
        )
        .unique()
        .scalar_one_or_none()
    )
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Target user not found.",
        )

    # Remove the netuid from the user.
    if user.netuids and args.netuid in user.netuids:
        user.netuids.remove(args.netuid)

    # If the user no longer has a netuid tracked, remove any subnet roles.
    if not user.netuids:
        Permissioning.disable(user, Permissioning.subnet_admin)
        Permissioning.disable(user, Permissioning.subnet_invoke)
    await db.commit()
    return {
        "status": f"Successfully revoked {args.netuid=} subnet roles from {user.user_id=} {user.username=}"
    }


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
        key = f"qta:{user_id}:{chute_id}"
        await settings.redis_client.delete(key)

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
        key = f"idiscount:{user_id}:{chute_id}"
        await settings.redis_client.delete(key)

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
    ur = SelfResponse.from_orm(user)
    ur.balance = user.current_balance.effective_balance if user.current_balance else 0.0
    return ur


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
    return quotas


@router.get("/{user_id}/quotas")
async def admin_get_user_quotas(
    user_id: str,
    db: AsyncSession = Depends(get_db_session),
    current_user: User = Depends(get_current_user()),
):
    """
    Load quotas for a user.
    """
    if not current_user.has_role(Permissioning.billing_admin):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="This action can only be performed by billing admin accounts.",
        )
    quotas = (
        (await db.execute(select(InvocationQuota).where(InvocationQuota.user_id == user_id)))
        .unique()
        .scalars()
        .all()
    )
    if not quotas:
        return settings.default_quotas
    return quotas


@router.get("/me/discounts")
async def my_discounts(
    db: AsyncSession = Depends(get_db_session),
    current_user: User = Depends(get_current_user()),
):
    """
    Load discounts for the current user.
    """
    discounts = (
        (
            await db.execute(
                select(InvocationDiscount).where(InvocationDiscount.user_id == current_user.user_id)
            )
        )
        .unique()
        .scalars()
        .all()
    )
    return discounts


@router.get("/{user_id}/discounts")
async def admin_list_discounts(
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
    user = (
        (
            await db.execute(
                select(User).where(or_(User.user_id == user_id, User.username == user_id))
            )
        )
        .unique()
        .scalar_one_or_none()
    )
    if not user:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found.")
    discounts = (
        (
            await db.execute(
                select(InvocationDiscount).where(InvocationDiscount.user_id == user.user_id)
            )
        )
        .unique()
        .scalars()
        .all()
    )
    return discounts


@router.get("/me/price_overrides")
async def my_price_overrides(
    db: AsyncSession = Depends(get_db_session),
    current_user: User = Depends(get_current_user()),
):
    """
    Load price overrides for the current user.
    """
    overrides = (
        (
            await db.execute(
                select(PriceOverride).where(PriceOverride.user_id == current_user.user_id)
            )
        )
        .unique()
        .scalars()
        .all()
    )
    return overrides


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
    used_raw = await settings.redis_client.get(key)
    used = 0.0
    try:
        used = float(used_raw or "0.0")
    except ValueError:
        await settings.redis_client.delete(key)
    return {"quota": quota, "used": used}


@router.get("/me/subscription_usage")
async def my_subscription_usage(
    current_user: User = Depends(get_current_user()),
    db: AsyncSession = Depends(get_db_session),
):
    """
    Get current subscription usage and caps for the authenticated user.
    Returns monthly and 4-hour window usage vs limits.
    """
    from api.config import (
        get_subscription_tier,
        SUBSCRIPTION_MONTHLY_CAP_MULTIPLIER,
        SUBSCRIPTION_4H_CAP_MULTIPLIER,
        FOUR_HOUR_CHUNKS_PER_MONTH,
    )

    quota = await InvocationQuota.get(current_user.user_id, "*")
    monthly_price = get_subscription_tier(quota)
    if monthly_price is None:
        return {"subscription": False}

    monthly_cap = monthly_price * SUBSCRIPTION_MONTHLY_CAP_MULTIPLIER
    four_hour_cap = (monthly_price / FOUR_HOUR_CHUNKS_PER_MONTH) * SUBSCRIPTION_4H_CAP_MULTIPLIER

    now = datetime.now()
    month_suffix = now.strftime("%Y%m")
    four_hour_bucket = int(time.time()) // (4 * 3600)

    # Try Redis first for both, fall back to DB.
    monthly_usage = None
    four_hour_usage = None

    month_key = f"sub_cap_m:{month_suffix}:{current_user.user_id}"
    four_hour_key = f"sub_cap_4h:{four_hour_bucket}:{current_user.user_id}"
    cached_month = await settings.redis_client.get(month_key)
    cached_4h = await settings.redis_client.get(four_hour_key)

    if cached_month is not None:
        monthly_usage = float(
            cached_month.decode() if isinstance(cached_month, bytes) else cached_month
        )
    if cached_4h is not None:
        four_hour_usage = float(cached_4h.decode() if isinstance(cached_4h, bytes) else cached_4h)

    if monthly_usage is None or four_hour_usage is None:
        result = await db.execute(
            text("""
                SELECT
                    GREATEST(COALESCE(SUM(paygo_amount), 0) - COALESCE(SUM(amount), 0), 0) AS monthly,
                    GREATEST(
                        COALESCE(SUM(CASE WHEN bucket >= now() - interval '4 hours' THEN paygo_amount ELSE 0 END), 0)
                        - COALESCE(SUM(CASE WHEN bucket >= now() - interval '4 hours' THEN amount ELSE 0 END), 0),
                        0
                    ) AS four_hour
                FROM usage_data
                WHERE user_id = :user_id
                AND bucket >= date_trunc('month', now())
            """),
            {"user_id": current_user.user_id},
        )
        row = result.one()
        if monthly_usage is None:
            monthly_usage = float(row.monthly)
        if four_hour_usage is None:
            four_hour_usage = float(row.four_hour)

    return {
        "subscription": True,
        "monthly_price": monthly_price,
        "monthly": {
            "usage": monthly_usage,
            "cap": monthly_cap,
            "remaining": max(monthly_cap - monthly_usage, 0.0),
        },
        "four_hour": {
            "usage": four_hour_usage,
            "cap": four_hour_cap,
            "remaining": max(four_hour_cap - four_hour_usage, 0.0),
        },
    }


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
    ur = SelfResponse.from_orm(user)
    ur.balance = user.current_balance.effective_balance if user.current_balance else 0.0
    return ur


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
        fingerprint=fingerprint,
    )


@router.get("/name_check")
async def check_username(
    username: str, readonly: Optional[bool] = None, db: AsyncSession = Depends(get_db_session)
):
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


@router.post(
    "/register",
    response_model=RegistrationResponse,
)
async def register(
    user_args: UserRequest,
    request: Request,
    token: Optional[str] = None,
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
    if attempts and int(attempts) > 2:
        logger.warning(
            f"Attempted multiple registrations from the same IP: {actual_ip} {attempts=}"
        )
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Too may registration requests.",
        )

    # Check the registration token.
    if not token:
        logger.warning(
            f"RTOK: Attempted registration without token: {x_forwarded_for=} {actual_ip=}"
        )
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Missing registration token in URL query params, please ensure you have upgraded to chutes>=0.3.33 and try again.",
        )
    allowed_ip = None
    if re.match(r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$", token, re.I):
        allowed_ip = await settings.redis_client.get(f"regtoken:{token}")
        if allowed_ip:
            allowed_ip = allowed_ip.decode()
    if not allowed_ip:
        logger.warning(f"RTOK: token not found: {token=}")
        await settings.redis_client.delete(f"regtoken:{token}")
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Invalid registration token, or registration token does not match expected IP address",
        )
    elif allowed_ip != actual_ip:
        logger.warning(
            f"RTOK: Expected IP {allowed_ip=} but got {actual_ip=}, allowing but probably should not..."
        )

    # Prevent duplicate hotkeys.
    if current_user:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="This hotkey is already registered to a user!",
        )

    # Validate the username
    await _validate_username(db, user_args.username)

    # Check min balance.
    if not await has_minimum_balance_for_registration(user_args.coldkey, hotkey):
        raise HTTPException(
            status_code=status.HTTP_402_PAYMENT_REQUIRED,
            detail=f"You must have at least {MIN_REG_BALANCE} tao on your coldkey to register an account.",
        )

    # Create.
    user, fingerprint = User.create(
        username=user_args.username,
        coldkey=user_args.coldkey,
        hotkey=hotkey,
    )
    generate_user_uid(None, None, user)
    user.payment_address, user.wallet_secret = await generate_payment_address()
    if settings.all_accounts_free:
        user.permissions_bitmask = 0
        Permissioning.enable(user, Permissioning.free_account)
    db.add(user)

    # Create the quota object.
    quota = InvocationQuota(
        user_id=user.user_id,
        chute_id="*",
        quota=0.0,
        is_default=True,
        payment_refresh_date=None,
        updated_at=None,
    )
    db.add(quota)

    await db.commit()
    await db.refresh(user)

    await settings.redis_client.incr(f"user_signup:{actual_ip}")
    await settings.redis_client.expire(f"user_signup:{actual_ip}", 24 * 60 * 60)

    return _registration_response(user, fingerprint)


@router.get("/registration_token")
async def get_registration_token(request: Request):
    """
    Initial form with cloudflare + hcaptcha to generate a registration token.
    """
    x_forwarded_for = request.headers.get("X-Forwarded-For")
    ip_chain = (x_forwarded_for or "").split(",")
    cf_ip = ip_chain[1].strip() if len(ip_chain) >= 2 else None
    actual_ip = ip_chain[0].strip() if ip_chain else None
    logger.info(f"RTOK [get token]: {x_forwarded_for=} {actual_ip=} {cf_ip=}")
    hostname = (request.headers.get("host", "") or "").lower()
    if not cf_ip or not await is_cloudflare_ip(cf_ip) or hostname != "rtok.chutes.ai":
        logger.warning(
            f"RTOK [get token]: request attempted to bypass cloudflare: {x_forwarded_for=} {actual_ip=} {cf_ip=}"
        )
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Request blocked, are you trying to bypass security measures?",
        )

    # Rate limits.
    attempts = await settings.redis_client.get(f"rtoken_fetch:{actual_ip}")
    if attempts and int(attempts) > 3:
        logger.warning(f"RTOK [get token]: too many requests from {actual_ip=}")
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=f"Too many registration token attempts from {actual_ip}",
        )
    await settings.redis_client.incr(f"rtoken_fetch:{actual_ip}")
    await settings.redis_client.expire(f"rtoken_fetch:{actual_ip}", 24 * 60 * 60)

    return HTMLResponse(content=registration_token_form(settings.hcaptcha_sitekey))


@router.post("/registration_token")
async def post_rtok(request: Request):
    """
    Verify hCaptcha and get a short-lived registration token.
    """
    # Check Cloudflare IP
    x_forwarded_for = request.headers.get("X-Forwarded-For")
    ip_chain = (x_forwarded_for or "").split(",")
    cf_ip = ip_chain[1].strip() if len(ip_chain) >= 2 else None
    actual_ip = ip_chain[0].strip() if ip_chain else None
    logger.info(f"RTOK: {x_forwarded_for=} {actual_ip=} {cf_ip=}")
    if not cf_ip or not await is_cloudflare_ip(cf_ip):
        logger.warning(
            f"RTOK: request attempted to bypass cloudflare: {x_forwarded_for=} {actual_ip=} {cf_ip=}"
        )
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Request blocked; are you trying to bypass security measures?",
        )

    # Validate captcha.
    form_data = await request.form()
    h_captcha_response = form_data.get("h-captcha-response")
    if not h_captcha_response:
        logger.warning(f"RTOK: missing hCaptcha response from {actual_ip}")
        return HTMLResponse(content=error_page("hCaptcha verification required"), status_code=400)
    async with aiohttp.ClientSession() as session:
        try:
            async with session.post(
                "https://hcaptcha.com/siteverify",
                data={
                    "secret": settings.hcaptcha_secret,
                    "response": h_captcha_response,
                    "remoteip": actual_ip,
                },
            ) as response:
                verify_data = await response.json()
                if not verify_data.get("success"):
                    logger.warning(
                        f"RTOK: hCaptcha verification failed for {actual_ip}: {verify_data}"
                    )
                    return HTMLResponse(
                        content=error_page("hCaptcha verification failed. Please try again."),
                        status_code=400,
                    )
        except Exception as e:
            logger.error(f"RTOK: hCaptcha verification error: {e}")
            return HTMLResponse(
                content=error_page("Verification error. Please try again."), status_code=500
            )

    # Create the token and render it.
    token = str(uuid.uuid4())
    await settings.redis_client.set(f"regtoken:{token}", actual_ip, ex=300)
    return HTMLResponse(content=registration_token_success(token))


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

    # Only admins can create users.
    if not current_user.has_role(Permissioning.create_user):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="This action can only be performed by user admin accounts.",
        )

    # Validate the username
    await _validate_username(db, user_args.username)

    # Validate hotkey/coldkey if either is specified.
    if user_args.coldkey or user_args.hotkey:
        if (
            not user_args.coldkey
            or not user_args.hotkey
            or not is_valid_ss58_address(user_args.coldkey)
            or not is_valid_ss58_address(user_args.hotkey)
            or await bt_user_exists(db, hotkey=user_args.hotkey)
        ):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Missing or invalid coldkey/hotkey values (or hotkey is already registered).",
            )

    # Create the user, faking the hotkey and using the payment address as the coldkey, since this
    # user is API/APP only and not really cognisant of bittensor.
    user, fingerprint = User.create(
        username=user_args.username,
        coldkey=user_args.coldkey or secrets.token_hex(24),
        hotkey=user_args.hotkey or secrets.token_hex(24),
    )
    generate_user_uid(None, None, user)
    user.payment_address, user.wallet_secret = await generate_payment_address()
    if not user_args.coldkey:
        user.coldkey = user.payment_address
    if settings.all_accounts_free:
        user.permissions_bitmask = 0
        Permissioning.enable(user, Permissioning.free_account)
    db.add(user)

    # Automatically create an API key for the user as well.
    api_key, one_time_secret = APIKey.create(user.user_id, APIKeyArgs(name="default", admin=True))
    db.add(api_key)

    # Create the quota object.
    quota = InvocationQuota(
        user_id=user.user_id,
        chute_id="*",
        quota=0.0,
        is_default=True,
        payment_refresh_date=None,
        updated_at=None,
    )
    db.add(quota)

    await db.commit()
    await db.refresh(user)
    await db.refresh(api_key)

    key_response = APIKeyCreationResponse.model_validate(api_key)
    key_response.secret_key = one_time_secret
    response = _registration_response(user, fingerprint)
    response.api_key = key_response

    return response


@router.post("/change_fingerprint")
async def change_fingerprint(
    args: FingerprintChange,
    db: AsyncSession = Depends(get_db_session),
    authorization: str | None = Header(None, alias=AUTHORIZATION_HEADER),
    hotkey: str | None = Header(None, alias=HOTKEY_HEADER),
    coldkey: str | None = Header(None, alias=COLDKEY_HEADER),
    nonce: str = Header(None, description="Nonce", alias=NONCE_HEADER),
    signature: str = Header(None, description="Hotkey signature", alias=SIGNATURE_HEADER),
):
    """
    Reset a user's fingerprint using either the hotkey or coldkey.
    """
    fingerprint = args.fingerprint

    # Using the existing fingerprint?
    if authorization:
        fingerprint_hash = hashlib.blake2b(authorization.encode()).hexdigest()
        user = (
            await db.execute(select(User).where(User.fingerprint_hash == fingerprint_hash))
        ).scalar_one_or_none()
        if not user:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Missing or invalid fingerprint provided.",
            )
        user.fingerprint_hash = hashlib.blake2b(fingerprint.encode()).hexdigest()
        await db.commit()
        await db.refresh(user)
        return {"status": "Fingerprint updated"}

    if not nonce or not signature:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing or invalid BT signature.",
        )

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


@router.get("/login/nonce")
async def get_login_nonce():
    """
    Get a nonce for hotkey signature login.
    The nonce is a UUID4 string that must be signed by the user's hotkey.
    Valid for 5 minutes.
    """
    from api.idp.service import create_login_nonce

    nonce = await create_login_nonce()
    return {"nonce": nonce, "expires_in": 300}


@router.post("/login")
async def login(
    request: Request,
    db: AsyncSession = Depends(get_db_session),
):
    """
    Exchange credentials for a JWT.

    Supports two authentication methods:
    1. Fingerprint: {"fingerprint": "your-fingerprint"}
    2. Hotkey signature: {"hotkey": "5...", "signature": "hex...", "nonce": "uuid"}

    For hotkey auth, first call GET /users/login/nonce to get a nonce,
    sign it with your hotkey (e.g., `btcli w sign --message <nonce>`),
    then submit the hotkey, signature, and nonce.
    """
    body = await request.json()

    # Method 1: Fingerprint authentication
    fingerprint = body.get("fingerprint")
    if fingerprint and isinstance(fingerprint, str) and fingerprint.strip():
        fingerprint_hash = hashlib.blake2b(fingerprint.encode()).hexdigest()
        user = (
            await db.execute(select(User).where(User.fingerprint_hash == fingerprint_hash))
        ).scalar_one_or_none()
        if user:
            return {"token": create_token(user)}
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid fingerprint.",
        )

    # Method 2: Hotkey signature authentication
    hotkey = body.get("hotkey")
    signature = body.get("signature")
    nonce = body.get("nonce")

    if hotkey and signature and nonce:
        from api.idp.service import verify_and_consume_login_nonce

        # Verify nonce exists and hasn't been used
        if not await verify_and_consume_login_nonce(nonce):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid or expired nonce. Please request a new one.",
            )

        # Verify signature
        try:
            signature_bytes = bytes.fromhex(signature)
            keypair = Keypair(hotkey)
            if not keypair.verify(nonce, signature_bytes):
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Invalid signature.",
                )
        except ValueError:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid signature format.",
            )
        except Exception as e:
            logger.warning(f"Hotkey signature verification failed: {e}")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Signature verification failed.",
            )

        # Find user by hotkey
        user = (await db.execute(select(User).where(User.hotkey == hotkey))).scalar_one_or_none()

        if not user:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="No account found for this hotkey.",
            )

        return {"token": create_token(user)}

    raise HTTPException(
        status_code=status.HTTP_400_BAD_REQUEST,
        detail="Please provide either 'fingerprint' or 'hotkey'+'signature'+'nonce' for authentication.",
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
        ur = SelfResponse.from_orm(user)
        ur.balance = user.current_balance.effective_balance if user.current_balance else 0.0
        return ur
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


@router.get("/{user_id}/usage")
async def list_usage(
    user_id: str,
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
    if user_id == "me":
        user_id = current_user.user_id
    else:
        if user_id != current_user.user_id and not current_user.has_role(
            Permissioning.billing_admin
        ):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="This action can only be performed by billing admin accounts.",
            )

    base_query = select(UsageData).where(UsageData.user_id == user_id)
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
                    input_tokens=int(data.input_tokens),
                    output_tokens=int(data.output_tokens),
                )
            )
    else:
        query = select(
            UsageData.bucket,
            func.sum(UsageData.amount).label("amount"),
            func.sum(UsageData.count).label("count"),
            func.sum(UsageData.input_tokens).label("input_tokens"),
            func.sum(UsageData.output_tokens).label("output_tokens"),
        ).where(UsageData.user_id == user_id)

        if chute_id:
            query = query.where(UsageData.chute_id == chute_id)
        if start_date:
            query = query.where(UsageData.bucket >= start_date)
        if end_date:
            query = query.where(UsageData.bucket <= end_date)

        query = query.group_by(UsageData.bucket)

        count_subquery = select(UsageData.bucket).where(UsageData.user_id == user_id)
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
                    input_tokens=int(row.input_tokens or 0),
                    output_tokens=int(row.output_tokens or 0),
                )
            )

    response = {
        "total": total,
        "page": page,
        "limit": limit,
        "items": results,
    }
    return response


@router.get("/{user_id}", response_model=SelfResponse)
async def get_user_info(
    user_id: str,
    db: AsyncSession = Depends(get_db_session),
    current_user: User = Depends(get_current_user(purpose="me")),
):
    """
    Get user info.
    """
    if user_id == "me":
        user_id = current_user.user_id
    user = (
        (
            await db.execute(
                select(User).where(or_(User.user_id == user_id, User.username == user_id))
            )
        )
        .unique()
        .scalar_one_or_none()
    )
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found",
        )
    if user.user_id != current_user.user_id and not current_user.has_role(
        Permissioning.billing_admin
    ):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="This action can only be performed by billing admin accounts.",
        )

    ur = SelfResponse.from_orm(user)
    ur.balance = user.current_balance.effective_balance if user.current_balance else 0.0
    return ur
