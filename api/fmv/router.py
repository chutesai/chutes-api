"""
Fair market value router.
"""

from fastapi import APIRouter, status, HTTPException
from functools import lru_cache
from api.fmv.fetcher import FMVFetcher

router = APIRouter()


@lru_cache()
def _fetcher():
    return FMVFetcher()


@router.get("/")
async def get_fmv():
    """
    Get the current FMV for tao.
    """
    prices = await _fetcher().get_prices()
    if not prices or not prices.get("tao"):
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to fetch prices",
        )
    return prices
