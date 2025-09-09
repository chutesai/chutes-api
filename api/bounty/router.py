"""
Routes for bounties.
"""

from datetime import datetime
from fastapi import APIRouter
from pydantic import BaseModel
from api.chute.response import ChuteResponse
from api.bounty.util import list_bounties

router = APIRouter()


class Bounty(BaseModel):
    bounty: int
    last_increased_at: datetime
    chute: ChuteResponse


@router.get("/")
async def get_bounty_list():
    """
    List available bounties, if any.
    """
    return await list_bounties()
