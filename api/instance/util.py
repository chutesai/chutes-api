"""
Helper functions for instances.
"""

import time
import asyncio
import orjson as json
from loguru import logger
from api.instance.schemas import Instance
from api.config import settings
from api.database import get_session
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from sqlalchemy import func, case, text
from sqlalchemy.orm import aliased
from datetime import datetime

# Define an alias for the Instance model to use in a subquery
InstanceAlias = aliased(Instance)


async def discover_chute_targets(session: AsyncSession, chute_id: str, max_wait: int = 0):
    """
    Fancy query to attempt evenly distributing queries based on coldkey and
    last invocation timestamp.
    """
    subquery = (
        select(InstanceAlias.miner_coldkey, func.count().label("instance_count"))
        .where(InstanceAlias.active.is_(True))
        .where(InstanceAlias.verified.is_(True))
        .where(InstanceAlias.chute_id == chute_id)
        .group_by(InstanceAlias.miner_coldkey)
        .subquery()
    )

    # Main query to fetch instances filtered by chute_id and ordered as needed
    query = (
        select(Instance)
        .join(subquery, Instance.miner_coldkey == subquery.c.miner_coldkey)
        .where(Instance.active.is_(True))
        .where(Instance.verified.is_(True))
        .where(Instance.chute_id == chute_id)
        .order_by(
            subquery.c.instance_count,
            case(
                (Instance.last_queried_at.is_(None), datetime.min),
                else_=Instance.last_queried_at,
            ).asc(),
        )
        .limit(3)
    )

    # Execute the query asynchronously
    result = await session.execute(query)
    instances = result.scalars().unique().all()
    started_at = time.time()
    if max_wait > 0:
        try:
            current_bounty = 0
            while not instances and time.time() - started_at < max_wait:
                async with get_session() as bounty_session:
                    result = await bounty_session.execute(
                        text("SELECT * FROM increase_bounty(:chute_id)"),
                        {"chute_id": chute_id},
                    )
                    bounty, last_increased_at = result.one()
                    await bounty_session.commit()
                if bounty != current_bounty:
                    logger.info(f"Bounty for {chute_id=} is now {bounty}")
                    current_bounty = bounty
                    await settings.redis_client.publish(
                        "miner_broadcast",
                        json.dumps(
                            {
                                "reason": "bounty_change",
                                "data": {"chute_id": chute_id, "bounty": bounty},
                            }
                        ).decode(),
                    )
                    await settings.redis_client.publish(
                        "events",
                        json.dumps(
                            {
                                "reason": "bounty_change",
                                "message": f"Chute {chute_id} bounty has been set to {bounty} compute units.",
                                "data": {
                                    "chute_id": chute_id,
                                    "bounty": bounty,
                                },
                            }
                        ).decode(),
                    )
                await asyncio.sleep(1)
                result = await session.execute(query)
                instances = result.scalars().unique().all()
        except asyncio.CancelledError:
            logger.warning("Target discovery cancelled")
            return []
    return instances


async def get_instance_by_chute_and_id(db, instance_id, chute_id, hotkey):
    """
    Helper to load an instance by ID.
    """
    if not instance_id:
        return None
    query = (
        select(Instance)
        .where(Instance.instance_id == instance_id)
        .where(Instance.chute_id == chute_id)
        .where(Instance.miner_hotkey == hotkey)
    )
    result = await db.execute(query)
    return result.unique().scalar_one_or_none()
