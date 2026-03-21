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
from datetime import datetime, timezone, timedelta
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
from api.invocation.util import build_subscription_periods, SUBSCRIPTION_CACHE_PREFIX
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
from api.agent_registration.schemas import (
    AgentRegistration,
    AgentRegistrationRequest,
    AgentRegistrationResponse,
    AgentRegistrationStatusResponse,
    AgentSetupRequest,
    AgentSetupResponse,
)
from api.payment.schemas import UsageData
from bittensor_wallet.keypair import Keypair
from scalecodec.utils.ss58 import is_valid_ss58_address
from sqlalchemy import bindparam, select, text, delete
from sqlalchemy.dialects.postgresql import JSONB as SA_JSONB

router = APIRouter()


class FingerprintChange(BaseModel):
    fingerprint: str


class BalanceRequest(BaseModel):
    user_id: str
    amount: float
    reason: str


class BalanceTransferRequest(BaseModel):
    user_id: str  # target user_id (uuid) or username
    amount: Optional[float] = None  # defaults to entire balance


class QuotaConfigRequest(BaseModel):
    quota: int
    effective_date: Optional[datetime] = None


class EffectiveDateRequest(BaseModel):
    effective_date: Optional[datetime] = None


class SubnetRoleRequest(BaseModel):
    user: str
    netuid: int
    admin: bool


class SubnetRoleRevokeRequest(BaseModel):
    user: str
    netuid: int


def _normalize_effective_date_input(
    effective_date: Optional[datetime], detail: str
) -> Optional[datetime]:
    if effective_date is None:
        return None
    if effective_date.tzinfo is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=detail,
        )

    normalized = effective_date.astimezone(timezone.utc)
    now = datetime.now(timezone.utc)
    if normalized > now:
        normalized = now
    elif normalized < now - timedelta(days=31):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="effective_date cannot be older than 31 days.",
        )
    return normalized


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


@router.post("/balance_transfer")
async def balance_transfer(
    transfer_req: BalanceTransferRequest,
    request: Request,
    db: AsyncSession = Depends(get_db_session),
    hotkey: str | None = Header(None, alias=HOTKEY_HEADER),
    signature: str | None = Header(None, alias=SIGNATURE_HEADER),
    nonce: str | None = Header(None, alias=NONCE_HEADER),
    authorization: str | None = Header(None, alias=AUTHORIZATION_HEADER),
):
    """
    Transfer balance from the authenticated user to a target user.
    Supports three authentication methods:
      1. Hotkey authentication (X-Chutes-Hotkey + X-Chutes-Signature + X-Chutes-Nonce)
      2. Admin API key (Authorization: cpk_...)
      3. Fingerprint (Authorization: <fingerprint>)
    """
    from api.api_key.util import get_and_check_api_key
    from api.util import nonce_is_valid, get_signing_message

    current_user = None

    # Method 1: Hotkey authentication.
    if hotkey and signature and nonce:
        if not nonce_is_valid(nonce):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid nonce.",
            )
        body_sha256 = getattr(request.state, "body_sha256", None)
        signing_message = get_signing_message(
            hotkey=hotkey,
            nonce=nonce,
            payload_hash=body_sha256,
            purpose=None,
            payload_str=None,
        )
        if not signing_message:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Bad signing message.",
            )
        try:
            signature_hex = bytes.fromhex(signature)
        except ValueError:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid signature format.",
            )
        try:
            keypair = Keypair(hotkey)
            if not keypair.verify(signing_message, signature_hex):
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Invalid request signature.",
                )
        except HTTPException:
            raise
        except Exception:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid request signature.",
            )
        current_user = (
            (await db.execute(select(User).where(User.hotkey == hotkey)))
            .unique()
            .scalar_one_or_none()
        )
        if not current_user:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="No user found with that hotkey.",
            )

    # Method 2 & 3: Authorization header (API key or fingerprint).
    elif authorization:
        token = authorization.strip().split(" ")[-1]

        # Try API key (full format validation).
        if APIKey.could_be_valid(token):
            api_key = await get_and_check_api_key(token, request)
            if not api_key:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Invalid API key.",
                )
            if not api_key.admin:
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="Balance transfers require an admin API key.",
                )
            current_user = api_key.user

        # Try fingerprint (raw string).
        else:
            fingerprint_hash = hashlib.blake2b(token.encode()).hexdigest()
            current_user = (
                await db.execute(select(User).where(User.fingerprint_hash == fingerprint_hash))
            ).scalar_one_or_none()
            if not current_user:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Invalid fingerprint.",
                )
    else:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required. Provide hotkey auth, admin API key, or fingerprint.",
        )

    # Resolve target user.
    target_user = (
        (
            await db.execute(
                select(User).where(
                    or_(User.user_id == transfer_req.user_id, User.username == transfer_req.user_id)
                )
            )
        )
        .unique()
        .scalar_one_or_none()
    )
    if not target_user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Target user not found: {transfer_req.user_id}",
        )

    # Cannot transfer to yourself.
    if current_user.user_id == target_user.user_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Cannot transfer balance to yourself.",
        )

    # Validate amount upfront.
    if transfer_req.amount is not None and transfer_req.amount <= 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Transfer amount must be positive.",
        )

    # Atomic transfer: debit source and credit target in a single statement.
    # The CTE locks the source row (FOR UPDATE), computes the transfer amount,
    # and the balance check is enforced atomically — no race conditions.
    debit_event_id = str(uuid.uuid4())
    credit_event_id = str(uuid.uuid4())
    origin_ip = request.headers.get("x-forwarded-for", "").split(",")[0]

    debit_raw = {"source_ip": origin_ip, "type": "balance_transfer", "direction": "debit"}
    credit_raw = {"source_ip": origin_ip, "type": "balance_transfer", "direction": "credit"}

    transfer_sql = text("""
        WITH source AS (
            SELECT user_id, username, COALESCE(balance, 0) AS balance
            FROM users
            WHERE user_id = :source_user_id
            FOR UPDATE
        ),
        transfer AS (
            SELECT
                CASE
                    WHEN :amount IS NOT NULL THEN CAST(:amount AS double precision)
                    ELSE source.balance
                END AS transfer_amount,
                source.balance AS source_balance
            FROM source
        ),
        debit AS (
            UPDATE users
            SET balance = COALESCE(balance, 0) - transfer.transfer_amount
            FROM transfer
            WHERE users.user_id = :source_user_id
              AND transfer.transfer_amount > 0
              AND transfer.source_balance >= transfer.transfer_amount
            RETURNING users.balance AS new_source_balance
        ),
        credit AS (
            UPDATE users
            SET balance = COALESCE(balance, 0) + transfer.transfer_amount
            FROM transfer, debit
            WHERE users.user_id = :target_user_id
            RETURNING users.balance AS new_target_balance
        ),
        debit_log AS (
            INSERT INTO admin_balance_changes (event_id, user_id, amount, reason, created_by, raw_request, timestamp)
            SELECT
                :debit_event_id,
                :source_user_id,
                -transfer.transfer_amount,
                :debit_reason,
                :source_user_id,
                :debit_raw_request,
                now()
            FROM transfer, debit
        ),
        credit_log AS (
            INSERT INTO admin_balance_changes (event_id, user_id, amount, reason, created_by, raw_request, timestamp)
            SELECT
                :credit_event_id,
                :target_user_id,
                transfer.transfer_amount,
                :credit_reason,
                :source_user_id,
                :credit_raw_request,
                now()
            FROM transfer, debit
        )
        SELECT
            transfer.transfer_amount,
            transfer.source_balance,
            debit.new_source_balance,
            credit.new_target_balance
        FROM transfer
        LEFT JOIN debit ON true
        LEFT JOIN credit ON true
    """)

    result = await db.execute(
        transfer_sql.bindparams(
            bindparam("source_user_id", value=current_user.user_id),
            bindparam("target_user_id", value=target_user.user_id),
            bindparam("amount", value=transfer_req.amount),
            bindparam("debit_event_id", value=debit_event_id),
            bindparam("credit_event_id", value=credit_event_id),
            bindparam(
                "debit_reason",
                value=f"Balance transfer to {target_user.username} ({target_user.user_id})",
            ),
            bindparam(
                "credit_reason",
                value=f"Balance transfer from {current_user.username} ({current_user.user_id})",
            ),
            bindparam("debit_raw_request", value=debit_raw, type_=SA_JSONB),
            bindparam("credit_raw_request", value=credit_raw, type_=SA_JSONB),
        ),
    )
    row = result.fetchone()

    if not row or row.new_source_balance is None:
        source_balance = float(row.source_balance) if row and row.source_balance else 0.0
        if source_balance <= 0:
            detail = "No balance to transfer."
        elif transfer_req.amount and transfer_req.amount > source_balance:
            detail = f"Insufficient balance. Current balance: {source_balance:.6f}"
        else:
            detail = "Transfer failed."
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=detail,
        )

    await db.commit()

    logger.warning(
        f"balance_transfer: {current_user.user_id} ({current_user.username}) -> "
        f"{target_user.user_id} ({target_user.username}) amount={row.transfer_amount:.6f} "
        f"{origin_ip=} {debit_event_id=} {credit_event_id=}"
    )

    return {
        "transferred": float(row.transfer_amount),
        "from_user": current_user.user_id,
        "to_user": target_user.user_id,
        "from_balance": float(row.new_source_balance),
        "to_balance": float(row.new_target_balance),
        "debit_event_id": debit_event_id,
        "credit_event_id": credit_event_id,
    }


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
    raw_quotas = await request.json()
    quotas = {}
    for key, value in raw_quotas.items():
        if isinstance(value, int):
            parsed = QuotaConfigRequest(quota=value, effective_date=None)
        elif isinstance(value, dict):
            try:
                parsed = QuotaConfigRequest(**value)
            except Exception:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Invalid quota payload {key=} {value=}",
                )
        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid quota value {key=} {value=}",
            )
        if parsed.quota < 0:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid quota value {key=} {value=}",
            )
        parsed.effective_date = _normalize_effective_date_input(
            parsed.effective_date,
            f"effective_date must be a UTC timestamp for {key=}",
        )
        quotas[key] = parsed
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

    # Remove quotas not in the new payload.
    incoming_chute_ids = set(quotas.keys())
    result = await db.execute(
        delete(InvocationQuota)
        .where(InvocationQuota.user_id == user_id)
        .where(InvocationQuota.chute_id.notin_(incoming_chute_ids))
        .returning(InvocationQuota.chute_id)
    )
    deleted_chute_ids = [row[0] for row in result]

    # Upsert incoming quotas.
    for key, quota_config in quotas.items():
        effective_date = quota_config.effective_date
        if effective_date is not None:
            effective_date = effective_date.replace(tzinfo=None)
        existing = await db.execute(
            select(InvocationQuota)
            .where(InvocationQuota.user_id == user_id)
            .where(InvocationQuota.chute_id == key)
        )
        row = existing.scalar_one_or_none()
        if row:
            row.quota = quota_config.quota
            row.updated_at = func.now()
            if effective_date is not None:
                row.effective_date = effective_date
        else:
            db.add(
                InvocationQuota(
                    user_id=user_id,
                    chute_id=key,
                    quota=quota_config.quota,
                    effective_date=effective_date if effective_date is not None else func.now(),
                )
            )
    await db.commit()

    # Purge cache for deleted and updated keys.
    for chute_id in deleted_chute_ids:
        await settings.redis_client.delete(f"qta:{user_id}:{chute_id}")
    for chute_id in incoming_chute_ids:
        await settings.redis_client.delete(f"qta:{user_id}:{chute_id}")
    await settings.redis_client.delete(f"subq:{user_id}")

    # Read back actual stored values for the response.
    result = await db.execute(
        select(InvocationQuota)
        .where(InvocationQuota.user_id == user_id)
        .where(InvocationQuota.chute_id.in_(incoming_chute_ids))
    )
    stored_rows = {r.chute_id: r for r in result.scalars()}
    response = {
        key: {
            "quota": stored_rows[key].quota,
            "effective_date": stored_rows[key].effective_date.isoformat()
            if stored_rows[key].effective_date
            else None,
            "updated_at": stored_rows[key].updated_at.isoformat()
            if stored_rows[key].updated_at
            else None,
        }
        for key in quotas
        if key in stored_rows
    }
    logger.success(f"Updated quotas for {user.user_id=} [{user.username}] to {response=}")
    return response


@router.put("/{user_id}/quotas/{chute_id}/effective_date")
async def admin_quota_effective_date_change(
    user_id: str,
    chute_id: str,
    body: EffectiveDateRequest,
    db: AsyncSession = Depends(get_db_session),
    current_user: User = Depends(get_current_user()),
):
    if not current_user.has_role(Permissioning.billing_admin):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="This action can only be performed by billing admin accounts.",
        )

    if chute_id != "*":
        try:
            uuid.UUID(chute_id)
        except ValueError:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid chute_id specified: {chute_id}",
            )

    normalized_effective_date = _normalize_effective_date_input(
        body.effective_date,
        "effective_date must be a UTC timestamp.",
    )

    result = await db.execute(
        select(InvocationQuota)
        .where(InvocationQuota.user_id == user_id)
        .where(InvocationQuota.chute_id == chute_id)
    )
    quota = result.scalar_one_or_none()
    if quota is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Quota row not found for {user_id=} {chute_id=}",
        )

    quota.effective_date = (
        normalized_effective_date.replace(tzinfo=None)
        if normalized_effective_date is not None
        else None
    )
    await db.commit()
    await settings.redis_client.delete(f"qta:{user_id}:{chute_id}")
    if chute_id == "*":
        await settings.redis_client.delete(f"subq:{user_id}")

    return {
        "user_id": user_id,
        "chute_id": chute_id,
        "quota": quota.quota,
        "effective_date": normalized_effective_date.isoformat()
        if normalized_effective_date
        else None,
        "updated_at": quota.updated_at.isoformat() if quota.updated_at else None,
    }


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
        is_custom_subscription,
        SUBSCRIPTION_MONTHLY_CAP_MULTIPLIER,
        SUBSCRIPTION_4H_CAP_MULTIPLIER,
        FOUR_HOUR_CHUNKS_PER_MONTH,
    )

    (
        quota,
        subscription_anchor,
        effective_date,
        updated_at,
    ) = await InvocationQuota.get_subscription_record(current_user.user_id)
    monthly_price = get_subscription_tier(quota)
    if monthly_price is None:
        return {"subscription": False}

    custom_sub = is_custom_subscription(quota)
    four_hour_cap = (monthly_price / FOUR_HOUR_CHUNKS_PER_MONTH) * SUBSCRIPTION_4H_CAP_MULTIPLIER

    periods = build_subscription_periods(subscription_anchor)

    # Try Redis first, fall back to DB.
    four_hour_usage = None
    four_hour_key = (
        f"{SUBSCRIPTION_CACHE_PREFIX}_{periods['four_hour_period']}:{current_user.user_id}"
    )
    cached_4h = await settings.redis_client.get(four_hour_key)
    if cached_4h is not None:
        four_hour_usage = float(cached_4h.decode() if isinstance(cached_4h, bytes) else cached_4h)

    # Custom subs only enforce 4h burst caps, not monthly.
    monthly_usage = None
    monthly_cap = None
    if not custom_sub:
        monthly_cap = monthly_price * SUBSCRIPTION_MONTHLY_CAP_MULTIPLIER
        month_key = (
            f"{SUBSCRIPTION_CACHE_PREFIX}_{periods['monthly_period']}:{current_user.user_id}"
        )
        cached_month = await settings.redis_client.get(month_key)
        if cached_month is not None:
            monthly_usage = float(
                cached_month.decode() if isinstance(cached_month, bytes) else cached_month
            )

    if (not custom_sub and monthly_usage is None) or four_hour_usage is None:
        result = await db.execute(
            text("""
                SELECT
                    COALESCE(
                        SUM(
                            CASE
                                WHEN ud.bucket >= :cycle_start
                                THEN GREATEST(COALESCE(ud.paygo_amount, 0) - COALESCE(ud.amount, 0), 0)
                                ELSE 0
                            END
                        ),
                        0
                    ) AS monthly,
                    COALESCE(
                        SUM(
                            CASE
                                WHEN ud.bucket >= :four_hour_start
                                THEN GREATEST(COALESCE(ud.paygo_amount, 0) - COALESCE(ud.amount, 0), 0)
                                ELSE 0
                            END
                        ),
                        0
                    ) AS four_hour
                FROM usage_data ud
                WHERE ud.user_id = :user_id
                AND ud.bucket >= LEAST(:cycle_start, :four_hour_start)
                AND EXISTS (
                    SELECT 1
                    FROM chutes c
                    WHERE c.chute_id = ud.chute_id
                    AND c.public IS TRUE
                )
            """),
            {
                "user_id": current_user.user_id,
                "cycle_start": periods["cycle_start"].replace(tzinfo=None),
                "four_hour_start": periods["four_hour_start"].replace(tzinfo=None),
            },
        )
        row = result.one()
        if not custom_sub and monthly_usage is None:
            monthly_usage = float(row.monthly)
        if four_hour_usage is None:
            four_hour_usage = float(row.four_hour)

    response = {
        "subscription": True,
        "custom": custom_sub,
        "monthly_price": monthly_price,
        "anchor_date": subscription_anchor.isoformat() if subscription_anchor else None,
        "effective_date": effective_date.isoformat() if effective_date else None,
        "updated_at": updated_at.isoformat() if updated_at else None,
        "four_hour": {
            "usage": four_hour_usage,
            "cap": four_hour_cap,
            "remaining": max(four_hour_cap - four_hour_usage, 0.0),
            "reset_at": periods["four_hour_end"].isoformat(),
        },
    }
    if custom_sub:
        response["monthly"] = {"uncapped": True}
    else:
        response["monthly"] = {
            "usage": monthly_usage,
            "cap": monthly_cap,
            "remaining": max(monthly_cap - monthly_usage, 0.0),
            "reset_at": periods["cycle_end"].isoformat(),
        }
    return response


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
    existing_agent_reg = await db.execute(
        select(AgentRegistration).where(
            AgentRegistration.username.ilike(username),
            AgentRegistration.deleted_at.is_(None),
        )
    )
    if existing_agent_reg.first() is not None:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Username {username} is reserved by a pending registration. Please choose another.",
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
    existing_agent = await db.execute(
        select(AgentRegistration).where(
            AgentRegistration.username.ilike(username),
            AgentRegistration.deleted_at.is_(None),
        )
    )
    if existing_agent.first() is not None:
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


@router.post(
    "/agent_registration",
    response_model=AgentRegistrationResponse,
)
async def agent_registration(
    args: AgentRegistrationRequest,
    db: AsyncSession = Depends(get_db_session),
):
    """
    Register an AI agent programmatically using hotkey/coldkey/signature.
    Returns a payment address where the agent must send TAO to complete registration.
    """
    # Validate signature: message = "chutes_signup:{hotkey}:{coldkey}", signed by hotkey.
    signing_message = f"chutes_signup:{args.hotkey}:{args.coldkey}"
    try:
        signature_bytes = bytes.fromhex(args.signature.removeprefix("0x"))
        keypair = Keypair(args.hotkey)
        if not keypair.verify(signing_message, signature_bytes):
            raise ValueError("Invalid signature")
    except Exception:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid signature. Message must be 'chutes_signup:{hotkey}:{coldkey}' signed by the hotkey.",
        )

    # Validate hotkey format.
    if not is_valid_ss58_address(args.hotkey) or not is_valid_ss58_address(args.coldkey):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid hotkey or coldkey SS58 address.",
        )

    # Check hotkey not already registered as a user.
    existing_user = (
        await db.execute(select(User).where(User.hotkey == args.hotkey))
    ).scalar_one_or_none()
    if existing_user:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="This hotkey is already registered to a user.",
        )

    # Check hotkey not in any agent registration (pending, expired, or completed).
    existing_reg = (
        await db.execute(
            select(AgentRegistration).where(
                AgentRegistration.hotkey == args.hotkey,
            )
        )
    ).scalar_one_or_none()
    if existing_reg:
        if existing_reg.deleted_at is None:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="This hotkey already has a pending agent registration.",
            )
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="This hotkey has already been used for agent registration. A new hotkey is required.",
        )

    # Handle username: validate if provided, auto-generate if not.
    if args.username:
        await _validate_username(db, args.username)
        username = args.username
    else:
        # Auto-generate unique username (check both users and active agent registrations).
        while True:
            username = f"chuter_{secrets.token_hex(3)}"
            existing_user = await db.execute(select(User).where(User.username.ilike(username)))
            if existing_user.first() is not None:
                continue
            existing_agent = await db.execute(
                select(AgentRegistration).where(
                    AgentRegistration.username.ilike(username),
                    AgentRegistration.deleted_at.is_(None),
                )
            )
            if existing_agent.first() is None:
                break

    # Pre-generate user_id.
    user_id = str(uuid.uuid4())

    # Generate payment address.
    payment_address, wallet_secret = await generate_payment_address()

    # Create registration row.
    registration = AgentRegistration(
        registration_id=str(uuid.uuid4()),
        user_id=user_id,
        hotkey=args.hotkey,
        coldkey=args.coldkey,
        username=username,
        payment_address=payment_address,
        wallet_secret=wallet_secret,
    )
    db.add(registration)
    await db.commit()
    await db.refresh(registration)

    # Fetch current TAO price to calculate required amount.
    from api.fmv.fetcher import get_fetcher

    fetcher = get_fetcher()
    tao_price = await fetcher.get_price("tao")
    required_tao = settings.agent_registration_threshold / tao_price if tao_price > 0 else 0

    return AgentRegistrationResponse(
        registration_id=registration.registration_id,
        user_id=registration.user_id,
        hotkey=registration.hotkey,
        coldkey=registration.coldkey,
        payment_address=registration.payment_address,
        required_amount=round(required_tao, 4),
        message=f"Send at least {required_tao:.4f} TAO (${settings.agent_registration_threshold} USD) to the payment address. A 10% tolerance is applied for price fluctuations.",
    )


@router.get(
    "/agent_registration/{hotkey}",
    response_model=AgentRegistrationStatusResponse,
)
async def get_agent_registration_status(
    hotkey: str,
    db: AsyncSession = Depends(get_db_session),
):
    """
    Check the status of an agent registration by hotkey.
    Handles all states: pending payment, completed (converted to user), or expired.
    """
    from api.fmv.fetcher import get_fetcher

    fetcher = get_fetcher()
    tao_price = await fetcher.get_price("tao")
    threshold_usd = settings.agent_registration_threshold
    tolerance = settings.agent_registration_tolerance
    required_tao = threshold_usd / tao_price if tao_price > 0 else 0

    # Check for active (pending) registration first.
    registration = (
        await db.execute(
            select(AgentRegistration).where(
                AgentRegistration.hotkey == hotkey,
                AgentRegistration.deleted_at.is_(None),
            )
        )
    ).scalar_one_or_none()
    if registration:
        remaining_usd = threshold_usd * (1 - tolerance) - registration.received_amount
        remaining_tao = remaining_usd / tao_price if tao_price > 0 else 0
        message = (
            f"Awaiting payment. Received ${registration.received_amount:.2f} of "
            f"${threshold_usd:.2f} required. Send ~{remaining_tao:.4f} more TAO to {registration.payment_address}."
        )
        return AgentRegistrationStatusResponse(
            registration_id=registration.registration_id,
            user_id=registration.user_id,
            hotkey=registration.hotkey,
            coldkey=registration.coldkey,
            payment_address=registration.payment_address,
            received_amount=registration.received_amount,
            required_amount=round(required_tao, 4),
            status="pending_payment",
            message=message,
        )

    # Check if already converted to a user.
    user = (await db.execute(select(User).where(User.hotkey == hotkey))).scalar_one_or_none()
    if user:
        # Look up the completed registration for received_amount.
        completed_reg = (
            await db.execute(
                select(AgentRegistration).where(
                    AgentRegistration.hotkey == hotkey,
                    AgentRegistration.deleted_at.isnot(None),
                )
            )
        ).scalar_one_or_none()

        return AgentRegistrationStatusResponse(
            registration_id=completed_reg.registration_id if completed_reg else None,
            user_id=user.user_id,
            hotkey=user.hotkey,
            coldkey=user.coldkey,
            payment_address=user.payment_address,
            received_amount=completed_reg.received_amount if completed_reg else 0.0,
            required_amount=round(required_tao, 4),
            status="completed",
            message=(
                "Registration complete. Your account has been created. "
                f"Call POST /users/{user.user_id}/agent_setup to get your API key and config."
            ),
        )

    # Check if there's an expired (deleted) registration with no corresponding user.
    expired_reg = (
        await db.execute(
            select(AgentRegistration).where(
                AgentRegistration.hotkey == hotkey,
                AgentRegistration.deleted_at.isnot(None),
            )
        )
    ).scalar_one_or_none()
    if expired_reg:
        return AgentRegistrationStatusResponse(
            registration_id=expired_reg.registration_id,
            user_id=expired_reg.user_id,
            hotkey=expired_reg.hotkey,
            coldkey=expired_reg.coldkey,
            payment_address=expired_reg.payment_address,
            received_amount=expired_reg.received_amount,
            required_amount=round(required_tao, 4),
            status="expired",
            message="Registration expired. Please create a new registration.",
        )

    raise HTTPException(
        status_code=status.HTTP_404_NOT_FOUND,
        detail="No agent registration found for this hotkey.",
    )


@router.post(
    "/{user_id}/agent_setup",
    response_model=AgentSetupResponse,
)
async def agent_setup(
    user_id: str,
    args: AgentSetupRequest,
    db: AsyncSession = Depends(get_db_session),
):
    """
    One-time setup endpoint for agent-registered users.
    Requires hotkey signature to prove ownership.
    Returns API key and config.ini template.
    """
    # Validate: user exists and was created from agent registration.
    user = (await db.execute(select(User).where(User.user_id == user_id))).scalar_one_or_none()
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found. Ensure payment has been received and account was created.",
        )

    # Verify this user came from an agent registration (check completed registrations).
    agent_reg = (
        await db.execute(
            select(AgentRegistration).where(
                AgentRegistration.user_id == user_id,
                AgentRegistration.deleted_at.isnot(None),
            )
        )
    ).scalar_one_or_none()
    if not agent_reg:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="This user was not created via agent registration.",
        )

    # Authenticate: verify hotkey signature.
    # Message format: "chutes_setup:{user_id}", signed by the registration hotkey.
    if args.hotkey != user.hotkey:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Hotkey does not match the registered hotkey for this user.",
        )
    signing_message = f"chutes_setup:{user_id}"
    try:
        signature_bytes = bytes.fromhex(args.signature.removeprefix("0x"))
        keypair = Keypair(args.hotkey)
        if not keypair.verify(signing_message, signature_bytes):
            raise ValueError("Invalid signature")
    except Exception:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Invalid signature. Message must be 'chutes_setup:{user_id}' signed by the hotkey.",
        )

    # One-time gate via Redis.
    redis_key = f"agent_setup_done:{user_id}"
    already_done = await settings.redis_client.get(redis_key)
    if already_done:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Agent setup has already been completed for this user. This endpoint is one-time only.",
        )

    # Create admin API key.
    api_key, one_time_secret = APIKey.create(user.user_id, APIKeyArgs(name="default", admin=True))
    db.add(api_key)

    await db.commit()

    # Set Redis gate (no expiry — permanent).
    await settings.redis_client.set(redis_key, "1")

    # Build config.ini matching the real format.
    config_ini = f"""[api]
base_url = https://api.chutes.ai

[auth]
username = {user.username}
user_id = {user_id}
hotkey_seed = REPLACE_WITH_YOUR_HOTKEY_SEED
hotkey_name = default
hotkey_ss58address = {user.hotkey}

[payment]
address = {user.payment_address}
"""

    setup_instructions = (
        "Save the config above to ~/.chutes/config.ini and replace "
        "'REPLACE_WITH_YOUR_HOTKEY_SEED' with your actual hotkey seed (hex). "
        "The payment address accepts both TAO and subnet alpha tokens to top up your balance. "
        "Store your API key securely — this endpoint cannot be called again. "
        "You can use either the API key or hotkey signature for authentication."
    )

    return AgentSetupResponse(
        user_id=user_id,
        api_key=one_time_secret,
        hotkey_ss58address=user.hotkey,
        payment_address=user.payment_address,
        username=user.username,
        config_ini=config_ini,
        setup_instructions=setup_instructions,
    )
