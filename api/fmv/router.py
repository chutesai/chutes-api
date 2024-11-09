"""
Fair market value router.
"""

from fastapi import APIRouter, status, HTTPException
from api.fmv.fetcher import get_fetcher

router = APIRouter()


@router.get("/")
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
