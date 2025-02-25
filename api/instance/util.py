"""
Helper functions for instances.
"""

import time
import uuid
import asyncio
import random
import aiohttp
import orjson as json
from async_lru import alru_cache
from loguru import logger
from contextlib import asynccontextmanager
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


MANAGERS = {}


class LeastConnManager:
    def __init__(self, instances: list[Instance], connection_expiry: int = 600):
        self.instances = {instance.instance_id: instance for instance in instances}
        self.connection_expiry = connection_expiry
        self.lock = asyncio.Lock()
        self._session = None

    async def initialize(self):
        if self._session is None:
            self.session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(connect=5.0, total=600.0),
                read_bufsize=8 * 1024 * 1024,
            )

    async def close(self):
        if self._session:
            await self._session.close()
            self._session = None

    async def get_targets(self):
        counts = {}
        for instance in self.instances:
            counts[instance.instance_id] = await settings.least_conn_redis.zcard(
                f"conn:{instance.instance_id}"
            )
        grouped_by_count = {}
        for instance_id, count in counts.items():
            if count not in grouped_by_count:
                grouped_by_count[count] = []
            grouped_by_count[count].append(instance_id)
        for count in grouped_by_count:
            random.shuffle(grouped_by_count[count])
        result = []
        for count in sorted(grouped_by_count.keys()):
            result.extend(grouped_by_count[count])
        return result

    @asynccontextmanager
    async def get_target(self, avoid=[]):
        conn_id = str(uuid.uuid4())
        now = time.time()
        try:
            targets = [inst for _id, inst in await self.get_targets() if _id not in avoid]
        except Exception as exc:
            logger.error(f"Failed to sort chute targets: {exc}, using random order")
            return random.choice([inst for _id, inst in self.instances.items() if _id not in avoid])

        if not targets:
            yield None
            return
        instance = targets[0]
        await settings.least_conn_redis.zadd(f"conn:{instance.instance_id}", {conn_id: now})
        await settings.least_conn_redis.expire(
            f"conn:{instance.instance_id}", self.connection_expiry
        )
        try:
            yield instance
        except Exception:
            await settings.least_conn_redis.zrem(f"conn:{instance.instance_id}", conn_id)
            raise
        finally:
            await settings.least_conn_redis.zrem(f"conn:{instance.instance_id}", conn_id)


async def get_chute_target_manager(session: AsyncSession, chute_id: str, max_wait: int = 0):
    """
    Select target instances by least connections (with random on equal counts).
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
                    if not await settings.memcache.get(
                        f"bounty_broadcast:{chute_id}:{bounty}".encode()
                    ):
                        await settings.memcache.set(
                            f"bounty_broadcast:{chute_id}:{bounty}".encode(), b"1", exptime=60
                        )
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
    if chute_id not in MANAGERS:
        MANAGERS[chute_id] = LeastConnManager(instances=instances)
        async with MANAGERS[chute_id].lock:
            await MANAGERS[chute_id].initialize()
    async with MANAGERS[chute_id].lock:
        MANAGERS[chute_id].instances = {instance.instance_id: instance for instance in instances}
    return MANAGERS[chute_id]


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
