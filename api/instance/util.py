"""
Helper functions for instances.
"""

import jwt
import time
import uuid
import asyncio
import random
import aiohttp
import orjson as json
from fastapi import HTTPException, status
from datetime import datetime, timedelta, timezone
from async_lru import alru_cache
from loguru import logger
from contextlib import asynccontextmanager
from api.instance.schemas import Instance, LaunchConfig
from api.config import settings
from api.database import get_session
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from sqlalchemy import text, func
from sqlalchemy.orm import aliased, joinedload

# Define an alias for the Instance model to use in a subquery
InstanceAlias = aliased(Instance)


@alru_cache(maxsize=100, ttl=30)
async def load_chute_targets(chute_id: str, nonce: float = 0):
    query = (
        select(Instance)
        .where(Instance.active.is_(True))
        .where(Instance.verified.is_(True))
        .where(Instance.chute_id == chute_id)
        .options(joinedload(Instance.nodes))
    )
    async with get_session() as session:
        result = await session.execute(query)
        return result.scalars().unique().all()


MANAGERS = {}


class LeastConnManager:
    def __init__(self, chute_id: str, instances: list[Instance], connection_expiry: int = 600):
        self.chute_id = chute_id
        self.redis_client = settings.cm_redis_client[
            uuid.UUID(chute_id).int % len(settings.cm_redis_client)
        ]
        self.instances = {instance.instance_id: instance for instance in instances}
        self.connection_expiry = connection_expiry
        self.lock = asyncio.Lock()
        self._session = None
        self._last_cleanup = time.time()
        self.mean_count = None

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

    async def get_targets(self, avoid=[], prefixes=None):
        instance_ids = list(self.instances)
        now = time.time()

        # Get connection counts for each instance via redis pipe.
        pipe = self.redis_client.pipeline()
        to_query = [instance_id for instance_id in instance_ids if instance_id not in avoid]
        for instance_id in to_query:
            pipe.zcount(f"conn:{instance_id}", now - self.connection_expiry, now)
        raw_counts = await pipe.execute()
        counts = dict(zip(to_query, raw_counts))
        min_count = min(raw_counts) if raw_counts else 1
        if not avoid:
            self.mean_count = int(sum(raw_counts) / (len(self.instances) or 1))

        # Periodically log counts for debugging.
        if random.random() < 0.05 and self.instances:
            logger.info(
                f"Instance counts for {self.chute_id=}: {min_count=} mean_count={self.mean_count} instance_count={len(self.instances)}"
            )
        if not counts:
            return []

        # Too many connections?
        if min_count >= 25:
            logger.warning(f"Instances overwhelmed: {min_count=}, pausing requests...")
            return []

        # Randomize the ordering for instances that have the same counts.
        grouped_by_count = {}
        for instance_id, count in counts.items():
            if count >= 25:
                logger.warning(f"Too many connections to {instance_id=} at the moment, skipping...")
                continue
            if count not in grouped_by_count:
                grouped_by_count[count] = []
            if instance := self.instances.get(instance_id):
                grouped_by_count[count].append(instance)
        for count in grouped_by_count:
            random.shuffle(grouped_by_count[count])

        # Prefix aware routing for LLM requests.
        if prefixes and random.random() <= 0.75:
            likely_cached = set([])
            for size, prefix_hash in prefixes:
                try:
                    instance_ids = list(counts)
                    has_prefix = await settings.memcache.multi_get(
                        *[
                            f"pfx:{prefix_hash}:{instance_id}".encode()
                            for instance_id in instance_ids
                        ]
                    )
                    for idx in range(len(instance_ids)):
                        if has_prefix[idx]:
                            likely_cached.add(instance_ids[idx])
                    if likely_cached:
                        break
                except Exception as exc:
                    logger.error(f"Error performing prefix-aware routing lookups: {exc}")
                    break
            if likely_cached:
                # Allow a small amount of discrepancy on active connection counts when
                # there is likely a prefix cache hit since it's much better for the user.
                routable = [
                    instance_id
                    for instance_id in likely_cached
                    if abs(counts[instance_id] - min_count) <= 3
                ]
                if routable:
                    result = sorted(
                        [
                            self.instances[instance_id]
                            for instance_id in routable
                            if instance_id in self.instances
                        ],
                        key=lambda inst: counts[inst.instance_id],
                    )[:3]
                    for count in sorted(grouped_by_count.keys()):
                        result.extend(
                            [
                                instance
                                for instance in grouped_by_count[count]
                                if instance.instance_id not in routable
                            ]
                        )
                    return result

        result = []
        for count in sorted(grouped_by_count.keys()):
            result.extend(grouped_by_count[count])
        return result

    @asynccontextmanager
    async def get_target(self, avoid=[], prefixes=None):
        await self.clean_up_expired_connections()
        conn_id = str(uuid.uuid4())
        now = time.time()
        try:
            targets = await self.get_targets(avoid=avoid, prefixes=prefixes)
        except Exception as exc:
            logger.error(f"Failed to sort chute targets: {exc}, using random order")
            yield random.choice([inst for _id, inst in self.instances.items() if _id not in avoid])
            return
        if not targets:
            yield None
            return
        instance = targets[0]
        try:
            await self.redis_client.zadd(f"conn:{instance.instance_id}", {conn_id: now})
            await self.redis_client.expire(f"conn:{instance.instance_id}", self.connection_expiry)
        except Exception as exc:
            logger.warning(f"Error tracking connection counts: {exc}")
        try:
            yield instance
        finally:
            try:
                await self.redis_client.zrem(f"conn:{instance.instance_id}", conn_id)
            except Exception as exc:
                logger.warning(f"Error cleaning up connection counts: {exc}")

    async def clean_up_expired_connections(self):
        now = time.time()
        if now - self._last_cleanup < 60:
            return
        for instance_id in self.instances:
            try:
                removed_count = await self.redis_client.zremrangebyscore(
                    f"conn:{instance_id}",
                    0,
                    now - self.connection_expiry,
                )
                if removed_count:
                    logger.info(
                        f"Successfully cleared {removed_count} expired connections from {instance_id=}"
                    )
            except Exception as exc:
                logger.warning(f"Error purging expired connection counts: {exc}")
        self._last_cleanup = now


async def get_chute_target_manager(session: AsyncSession, chute_id: str, max_wait: int = 0):
    """
    Select target instances by least connections (with random on equal counts).
    """
    instances = await load_chute_targets(chute_id, nonce=0)
    started_at = time.time()
    while not instances:
        # Increase the bounty.
        bounty, last_increased_at, was_increased = None, None, False
        async with get_session() as bounty_session:
            update_result = await bounty_session.execute(
                text("SELECT 1 FROM rolling_updates WHERE chute_id = :chute_id"),
                {"chute_id": chute_id},
            )
            if update_result.first() is not None:
                logger.warning(
                    f"Skipping bounty event for {chute_id=} due to in-progress rolling update."
                )
            else:
                result = await bounty_session.execute(
                    text("SELECT * FROM increase_bounty(:chute_id)"),
                    {"chute_id": chute_id},
                )
                bounty, last_increased_at, was_increased = result.one()
                await bounty_session.commit()

        # Broadcast unique bounty events.
        if was_increased:
            logger.info(f"Bounty for {chute_id=} is now {bounty}")
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
        if not max_wait or time.time() - started_at >= max_wait:
            break
        await asyncio.sleep(1.0)
        instances = await load_chute_targets(chute_id, nonce=time.time())
    if not instances:
        return None
    if chute_id not in MANAGERS:
        MANAGERS[chute_id] = LeastConnManager(chute_id=chute_id, instances=instances)
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
        .options(joinedload(Instance.nodes))
    )
    result = await db.execute(query)
    return result.unique().scalar_one_or_none()


def create_launch_jwt(launch_config) -> str:
    """
    Create JWT for a given launch config (updated chutes lib with new graval etc).
    """
    now = datetime.now(timezone.utc)
    expires_at = now + timedelta(hours=1)
    payload = {
        "exp": expires_at.replace(tzinfo=None),
        "sub": launch_config.config_id,
        "iat": now.replace(tzinfo=None),
        "url": f"https://api.{settings.base_domain}/instances/launch_config/{launch_config.config_id}",
        "env_key": launch_config.env_key,
        "iss": "chutes",
    }
    if launch_config.job_id:
        payload["job_id"] = launch_config.job_id
    encoded_jwt = jwt.encode(payload, settings.launch_config_key, algorithm="HS256")
    return encoded_jwt


async def load_launch_config_from_jwt(
    db, config_id: str, token: str, allow_retrieved: bool = False
) -> str:
    detail = "Missing or invalid launch config JWT"
    try:
        payload = jwt.decode(
            token,
            settings.launch_config_key,
            options={
                "verify_signature": True,
                "verify_exp": True,
                "verify_iat": True,
                "verify_iss": True,
                "require": ["exp", "iat", "iss"],
            },
            issuer="chutes",
        )
        if config_id == payload["sub"]:
            config = (
                (await db.execute(select(LaunchConfig).where(LaunchConfig.config_id == config_id)))
                .unique()
                .scalar_one_or_none()
            )
            if config:
                if not config.retrieved_at:
                    config.retrieved_at = func.now()
                    return config
                elif allow_retrieved:
                    return config
                detail = f"Launch config {config_id=} has already been retrieved: {token=} {config.retrieved_at=}"
                logger.warning(detail)
            else:
                detail = f"Launch config {config_id} not found in database."
        else:
            detail = f"Launch config {config_id=} does not match token!"
    except jwt.InvalidTokenError:
        logger.warning(f"Attempted to use invalid token for launch config: {config_id=} {token=}")
    except Exception as exc:
        logger.warning(f"Unhandled exception checking launch config JWT: {exc}")

    # If we got here, it failed somewhere.
    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail=detail,
    )
