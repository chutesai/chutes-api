"""
Routes for bounties.
"""

import time
from typing import Optional
from loguru import logger
from datetime import datetime
from fastapi import APIRouter, status, HTTPException, Depends
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from api.config import settings
from api.database import get_db_session
from api.chute.schemas import Chute
from api.chute.response import ChuteResponse
from api.bounty.util import (
    create_bounty_if_not_exists,
    get_bounty_amount,
    send_bounty_notification,
    list_bounties,
)
from api.user.schemas import User
from api.user.service import get_current_user
from api.permissions import Permissioning

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


@router.get("/{chute_id}/increase")
async def increase_chute_bounty(
    chute_id: str,
    boost: Optional[float] = None,
    db: AsyncSession = Depends(get_db_session),
    current_user: User = Depends(get_current_user()),
):
    """
    Increase bounty value (creating if not exists).
    """
    if not current_user.has_role(Permissioning.billing_admin):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Only billing admins may perform this action",
        )

    chute = (
        (await db.execute(select(Chute).where(Chute.chute_id == chute_id)))
        .unique()
        .scalar_one_or_none()
    )
    if not chute:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Chute not found",
        )

    bounty_lifetime = 3600
    notice_sent = False
    bounty_created = False
    if await create_bounty_if_not_exists(chute_id, lifetime=bounty_lifetime):
        logger.success(f"Successfully created a bounty for {chute_id=}")
        bounty_created = True
    amount = await get_bounty_amount(chute_id)
    if amount:
        current_time = int(time.time())
        window = current_time - (current_time % 30)
        notification_key = f"bounty_notification:{chute_id}:{window}"
        if await settings.redis_client.setnx(notification_key, b"1"):
            await settings.redis_client.expire(notification_key, 33)
            logger.info(f"Bounty for {chute_id=} is now {amount}")
            await send_bounty_notification(chute_id, amount)
            notice_sent = True
    else:
        logger.warning(f"No bounty for {chute_id=}")

    # Boosting?
    if boost and 1.0 <= boost <= 4.0:
        chute.boost = boost
        await db.commit()

    return {
        "bounty_created": bounty_created,
        "bounty_existed": amount and not bounty_created,
        "notification_sent": notice_sent,
        "amount": amount,
    }
