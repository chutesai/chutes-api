"""
Payments router.
"""

from fastapi import APIRouter, status, HTTPException
from fastapi_cache.decorator import cache
from api.gpu import COMPUTE_MULTIPLIER, COMPUTE_MIN
from api.payment.constants import COMPUTE_UNIT_PRICE_BASIS, PAYOUT_STRUCTURE
from api.fmv.fetcher import get_fetcher

router = APIRouter()


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
    for model, scaler in COMPUTE_MULTIPLIER.items():
        usd_price = COMPUTE_UNIT_PRICE_BASIS * scaler
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


@router.get("/payout_structure")
async def get_payout_structure():
    """
    Get the current theoretical payout structure.
    """
    return {
        "notice": "This is not in effect at the moment, simply a theoretically mechansim.",
        "theoretical_structure": {
            key: {k: v for k, v in value.items() if not k.startswith("address")}
            for key, value in PAYOUT_STRUCTURE.items()
        },
    }
