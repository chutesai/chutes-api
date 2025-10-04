"""
Payments router.
"""

import orjson as json
from typing import Optional
from pydantic import BaseModel
from fastapi import APIRouter, status, HTTPException, Depends, Request
from fastapi_cache.decorator import cache
from sqlalchemy.ext.asyncio import AsyncSession
from api.gpu import SUPPORTED_GPUS, COMPUTE_UNIT_PRICE_BASIS, COMPUTE_MIN
from api.fmv.fetcher import get_fetcher
from api.config import settings
from api.database import get_db_session
from api.user.util import refund_deposit
from api.user.schemas import User
from api.user.service import get_current_user
from api.util import memcache_get, memcache_set
from api.payment.schemas import Payment
from sqlalchemy import select, desc, func, text

router = APIRouter()


class ReturnDepositArgs(BaseModel):
    address: str


@cache(expire=300)
@router.get("/daily_revenue_summary")
async def get_daily_revenue_summary(db: AsyncSession = Depends(get_db_session)):
    """
    Get the summary of daily revenue including paygo, invoiced users, subscriptions and pending private instances.
    """
    result = await db.execute(
        text("SELECT * FROM daily_revenue_summary ORDER BY date DESC LIMIT 30")
    )
    rows = result.fetchall()
    return [
        {
            "date": row.date.isoformat() if row.date else None,
            "new_subscriber_count": row.new_subscriber_count,
            "new_subscriber_revenue": float(row.new_subscriber_revenue),
            "paygo_revenue": float(row.paygo_revenue),
            "pending_instance_revenue": float(row.instance_revenue),
            "total_revenue": float(
                row.new_subscriber_revenue + row.paygo_revenue + row.instance_revenue
            ),
        }
        for row in rows
    ]


@cache(expire=300)
@router.get("/payments/summary/tao")
async def get_tao_payment_totals(db: AsyncSession = Depends(get_db_session)):
    """
    Get the amount (as USD equivalent) of payments made by tao for
    today, the current month, and total.
    """
    query = """
    SELECT
        COALESCE(SUM(CASE
            WHEN DATE(created_at) = CURRENT_DATE
            THEN usd_amount
        END), 0) AS today,
        COALESCE(SUM(CASE
            WHEN DATE_TRUNC('month', created_at) = DATE_TRUNC('month', CURRENT_DATE)
            THEN usd_amount
        END), 0) AS this_month,
        COALESCE(SUM(usd_amount), 0) AS total
    FROM payments
    WHERE purpose = 'credits'
    """
    result = await db.execute(text(query))
    row = result.fetchone()
    return {
        "today": float(row.today),
        "this_month": float(row.this_month),
        "total": float(row.total),
    }


@cache(expire=90)
@router.get("/quota_unlock_amount")
async def get_quota_unlock_amount():
    """
    Amount, in USD, of payment history that provides the "free" tier quota.
    """
    tao_usd = await get_fetcher().get_price("tao")
    return {
        "usd": settings.quota_unlock_amount,
        # For tao, we'll add a bit onto the estimate to hopefully accomodate any instantaneous price fluctuations.
        "tao": (settings.quota_unlock_amount / tao_usd) * 1.05,
    }


@cache(expire=60)
@router.get("/fmv")
async def get_fmv():
    """
    Get the current FMV for tao.
    """
    prices = await get_fetcher().get_prices()
    if not prices or not prices.get("tao"):
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to fetch prices",
        )
    return prices


@cache(expire=60)
@router.get("/pricing")
async def get_pricing():
    """
    Get the current compute unit pricing.
    """
    current_tao_price = await get_fetcher().get_price("tao")
    if current_tao_price is None:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to fetch prices",
        )

    # Calculate prices for each GPU.
    gpu_price = {}
    for model, info in SUPPORTED_GPUS.items():
        usd_price = info["hourly_rate"]
        tao_price = usd_price / current_tao_price
        gpu_price[model] = {
            "usd": {
                "hour": usd_price,
                "second": usd_price / 3600,
            },
            "tao": {
                "hour": tao_price,
                "second": tao_price / 3600,
            },
        }

    usd_compute_price = COMPUTE_UNIT_PRICE_BASIS * COMPUTE_MIN / 3600
    tao_compute_price = usd_compute_price / current_tao_price
    return {
        "tao_usd": current_tao_price,
        "compute_unit_estimate": {
            "usd": usd_compute_price,
            "tao": tao_compute_price,
            "description": "a single compute unit is considered one second of compute time on the 'worst' currently supported GPU",
        },
        "gpu_price_estimates": gpu_price,
    }


@router.post("/return_developer_deposit")
async def return_developer_deposit(
    args: ReturnDepositArgs,
    db: AsyncSession = Depends(get_db_session),
    current_user: User = Depends(get_current_user()),
):
    query = (
        select(Payment)
        .where(Payment.user_id == current_user.user_id)
        .where(Payment.purpose == "developer")
        .order_by(desc(Payment.created_at))
        .limit(1)
    )
    recent_payment = (await db.execute(query)).scalar_one_or_none()
    if not recent_payment:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"You have not made any payments to the developer deposit address: {current_user.developer_payment_address}",
        )
    result, message = await refund_deposit(current_user.user_id, args.address)
    if not result:
        return HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=message,
        )
    return {
        "status": "transferred",
        "message": message,
    }


@router.get("/payments")
async def list_payments(
    page: Optional[int] = 0,
    limit: Optional[int] = 25,
    db: AsyncSession = Depends(get_db_session),
    request: Request = None,
):
    """
    List all payments.
    """
    if request:
        cache_key = f"payment_list:{page}:{limit}".encode()
        if cached := await memcache_get(cache_key):
            return json.loads(cached)
    query = (
        select(Payment, User)
        .join(User, Payment.user_id == User.user_id)
        .where(Payment.purpose == "credits")
    )
    total_query = select(func.count()).select_from(query.subquery())
    total_result = await db.execute(total_query)
    total = total_result.scalar() or 0

    query = (
        query.order_by(Payment.created_at.desc())
        .offset((page or 0) * (limit or 25))
        .limit((limit or 25))
    )
    results = []
    for payment, user in (await db.execute(query)).all():
        transfer_link = (
            f"https://taostats.io/extrinsic/{payment.block}-{payment.extrinsic_idx:04d}"
            if payment.extrinsic_idx is not None
            else f"https://taostas.io/block/{payment.block}/extrinsics"
        )
        results.append(
            dict(
                payment_id=payment.payment_id,
                ss58_address=user.payment_address,
                block=payment.block,
                rao_amount=payment.rao_amount,
                fmv=payment.fmv,
                usd_amount=payment.usd_amount,
                transaction_hash=payment.transaction_hash,
                timestamp=payment.created_at.isoformat(),
                tx_link=transfer_link,
                transactions_link=f"https://taostats.io/account/{user.payment_address}/transactions",
            )
        )
        if payment.refunded:
            results[-1]["refunded"] = True
    response = {
        "total": total,
        "page": page,
        "limit": limit,
        "items": results,
    }
    await memcache_set(cache_key, json.dumps(response), exptime=300)
    return response
