"""
Routes for bounties.
"""

from datetime import datetime
from fastapi import APIRouter, Depends
from pydantic import BaseModel
from sqlalchemy import text, select
from api.chute.schemas import Chute
from api.chute.response import ChuteResponse
from api.database import SessionLocal
from api.user.schemas import User
from api.user.service import get_current_user
from api.config import settings

router = APIRouter()


class Bounty(BaseModel):
    bounty: int
    last_increased_at: datetime
    chute: ChuteResponse


@router.get("/")
async def list_bounties(
    _: User = Depends(get_current_user(raise_not_found=False, registered_to=settings.netuid)),
):
    """
    List available bounties, if any.
    """
    async with SessionLocal() as session:
        query = (
            select(Chute, text("bounties.bounty"), text("bounties.last_increased_at"))
            .select_from(
                Chute.__table__.join(text("bounties"), text("bounties.chute_id = chutes.chute_id"))
            )
            .order_by(text("bounties.bounty DESC"))
        )
        results = await session.execute(query)
        bounties = []
        for chute, bounty, last_increased_at in results.all():
            bounties.append(Bounty(bounty=bounty, last_increased_at=last_increased_at, chute=chute))
        return bounties
