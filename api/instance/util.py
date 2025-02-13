"""
Helper functions for instances.
"""

import time
import asyncio
import random
import orjson as json
from async_lru import alru_cache
from loguru import logger
from api.instance.schemas import Instance
from api.config import settings
from api.database import get_session
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from sqlalchemy import text
from sqlalchemy.orm import aliased

# Define an alias for the Instance model to use in a subquery
InstanceAlias = aliased(Instance)


@alru_cache(maxsize=100, ttl=30)
async def load_chute_targets(chute_id: str, nonce: float = 0):
    query = (
        select(Instance)
        .where(Instance.active.is_(True))
        .where(Instance.verified.is_(True))
        .where(Instance.chute_id == chute_id)
    )
    async with get_session() as session:
        result = await session.execute(query)
        return result.scalars().unique().all()


async def discover_chute_targets(session: AsyncSession, chute_id: str, max_wait: int = 0):
    """
    Evenly distribute queries to miners.
    """
    instances = await load_chute_targets(chute_id, nonce=0)
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
                instances = await load_chute_targets(chute_id, nonce=time.time())
        except asyncio.CancelledError:
            logger.warning("Target discovery cancelled")
            return []
    if not instances:
        return []
    return random.sample(instances, min(9, len(instances)))


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
