"""
Payments router.
"""

import orjson as json
from typing import Optional
from pydantic import BaseModel
from datetime import datetime, timedelta, timezone
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
from api.payment.schemas import Payment
from sqlalchemy import select, desc, func

router = APIRouter()


class ReturnDepositArgs(BaseModel):
    address: str


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
        "first_payment_bonus": {
            "minimum_payment_usd": settings.first_payment_bonus_threshold,
            "bonus_amount_usd": settings.first_payment_bonus,
        },
        "compute_unit_estimate": {
            "usd": usd_compute_price,
            "tao": tao_compute_price,
            "description": "a single compute unit is considered one second of compute time on the 'worst' currently supported GPU",
        },
        "gpu_price_estimates": gpu_price,
    }


@router.get("/developer_deposit")
async def get_developer_deposit():
    """
    Get the USD/tao amount required to enable developer mode.
    """
    current_tao_price = await get_fetcher().get_price("tao")
    return {
        "usd": settings.developer_deposit,
        "tao_estimate": settings.developer_deposit / current_tao_price,
        "message": "Price fluctuations dictate you should probably send a bit more than the estimate.",
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
        .order_by(desc(Payment.created_at))
        .limit(1)
    )
    recent_payment = (await db.execute(query)).scalar_one_or_none()
    if not recent_payment:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"You have not made any payments to the developer deposit address: {current_user.developer_deposit_address}",
        )
    if datetime.now(timezone.utc) - recent_payment.created_at <= timedelta(days=7):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"You must wait at least 7 days between payment and cancellation, most recent payment: {recent_payment.created_at}",
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
        if cached := await settings.memcache.get(cache_key):
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
                tx_link=f"https://taostats.io/transfer/{payment.transaction_hash}",
                transactions_link=f"https://taostats.io/account/{user.payment_address}/transactions",
            )
        )
    response = {
        "total": total,
        "page": page,
        "limit": limit,
        "items": results,
    }
    await settings.memcache.set(cache_key, json.dumps(response), exptime=300)
    return response
