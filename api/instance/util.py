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
    def __init__(
        self,
        chute_id: str,
        instances: list[Instance],
        connection_expiry: int = 600,
        cleanup_interval: int = 5,
    ):
        self.chute_id = chute_id
        self.redis_client = settings.cm_redis_client[
            uuid.UUID(chute_id).int % len(settings.cm_redis_client)
        ]
        self.instances = {instance.instance_id: instance for instance in instances}
        self.connection_expiry = connection_expiry
        self.cleanup_interval = cleanup_interval
        self._session = None
        self.mean_count = None

        # Start continuous cleanup task immediately
        self._cleanup_task = asyncio.create_task(self._continuous_cleanup())

        # Pre-register Lua scripts for better performance
        self._register_lua_scripts()

        self.lock = asyncio.Lock()

    def _register_lua_scripts(self):
        # Track new connection.
        self.lua_add_connection = """
        local key = KEYS[1]
        local conn_id = ARGV[1]
        local now = tonumber(ARGV[2])
        local expiry = tonumber(ARGV[3])
        redis.call('ZADD', key, now, conn_id)
        redis.call('EXPIRE', key, expiry)
        return redis.call('ZCOUNT', key, now - expiry, now)
        """

        # Remove "completed" connection.
        self.lua_remove_connection = """
        local key = KEYS[1]
        local conn_id = ARGV[1]
        local now = tonumber(ARGV[2])
        local expiry = tonumber(ARGV[3])
        local removed = redis.call('ZREM', key, conn_id)
        local expired = redis.call('ZRANGEBYSCORE', key, 0, now - expiry, 'LIMIT', 0, 10)
        if #expired > 0 then
            redis.call('ZREM', key, unpack(expired))
        end
        return removed
        """

        # Batch cleanup all keys.
        self.lua_batch_cleanup = """
        local pattern = ARGV[1]
        local now = tonumber(ARGV[2])
        local expiry = tonumber(ARGV[3])
        local cutoff = now - expiry
        local cursor = "0"
        local total_removed = 0
        repeat
            local result = redis.call('SCAN', cursor, 'MATCH', pattern, 'COUNT', 100)
            cursor = result[1]
            local keys = result[2]
            for i, key in ipairs(keys) do
                local removed = redis.call('ZREMRANGEBYSCORE', key, 0, cutoff)
                total_removed = total_removed + removed
            end
        until cursor == "0"
        return total_removed
        """

    async def initialize(self):
        if self._session is None:
            self._session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(connect=5.0, total=600.0),
                read_bufsize=8 * 1024 * 1024,
            )

    async def close(self):
        if self._session:
            await self._session.close()
            self._session = None

        if hasattr(self, "_cleanup_task") and self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass

    async def _continuous_cleanup(self):
        """
        Run cleanup continuously while CM is alive.
        """
        while True:
            try:
                await self._cleanup_expired_connections()
                await asyncio.sleep(self.cleanup_interval)
            except asyncio.CancelledError:
                logger.info(f"Cleanup task cancelled for chute {self.chute_id}")
                break
            except Exception as e:
                logger.error(f"Error in cleanup loop: {e}", exc_info=True)
                await asyncio.sleep(self.cleanup_interval)

    async def _cleanup_expired_connections(self):
        now = int(time.time())
        try:
            pattern = f"conn:{self.chute_id}:*"
            started_at = time.time()
            total_removed = await self.redis_client.eval(
                self.lua_batch_cleanup, 0, pattern, now, self.connection_expiry
            )
            if total_removed:
                logger.info(
                    f"Cleaned {total_removed} expired connections for chute {self.chute_id} in {time.time() - started_at} seconds"
                )
        except Exception as e:
            logger.error(f"Error in batch cleanup: {e}", exc_info=True)

    async def get_connection_counts(self, instance_ids: list[str]) -> dict[str, int]:
        """
        Get current valid connection counts for instances.
        """
        now = time.time()
        cutoff = now - self.connection_expiry
        pipe = self.redis_client.pipeline()
        for instance_id in instance_ids:
            key = f"conn:{self.chute_id}:{instance_id}"
            pipe.zcount(key, cutoff, now)
        try:
            counts = await pipe.execute()
            return dict(zip(instance_ids, counts))
        except Exception as e:
            logger.error(f"Error getting connection counts: {e}")
            return {iid: 0 for iid in instance_ids}

    async def get_targets(self, avoid=[], prefixes=None):
        # Get instances not in avoid list
        available_instances = [iid for iid in self.instances.keys() if iid not in avoid]
        if not available_instances:
            return []
        started_at = time.time()
        counts = await self.get_connection_counts(available_instances)
        time_taken = time.time() - started_at
        if not counts:
            return []
        min_count = min(counts.values())

        # Update mean count for monitoring
        if not avoid:
            self.mean_count = int(sum(counts.values()) / (len(counts) or 1))

        # Periodic logging
        if random.random() < 0.05:
            logger.info(
                f"Connection counts for {self.chute_id}: "
                f"min={min_count}, mean={self.mean_count}, "
                f"instances={len(self.instances)}, "
                f"{time_taken=}"
            )

        # Check if all instances are overwhelmed
        if min_count >= 25:
            logger.warning(f"All instances overwhelmed for {self.chute_id}: min_count={min_count}")
            return []

        # Group instances by connection count
        grouped_by_count = {}
        for instance_id, count in counts.items():
            if count >= 25:
                logger.warning(f"Instance {instance_id} has too many connections: {count}")
                continue
            if count not in grouped_by_count:
                grouped_by_count[count] = []
            if instance := self.instances.get(instance_id):
                grouped_by_count[count].append(instance)

        # Randomize within each count group
        for instances in grouped_by_count.values():
            random.shuffle(instances)

        # Handle prefix-aware routing if enabled
        if prefixes and random.random() <= 0.75:
            result = await self._handle_prefix_routing(
                counts, grouped_by_count, min_count, prefixes
            )
            if result:
                return result

        # Return instances sorted by connection count
        result = []
        for count in sorted(grouped_by_count.keys()):
            result.extend(grouped_by_count[count])

        return result

    async def _handle_prefix_routing(self, counts, grouped_by_count, min_count, prefixes):
        likely_cached = set()
        for size, prefix_hash in prefixes:
            try:
                instance_ids = list(counts.keys())
                cache_keys = [f"pfx:{prefix_hash}:{iid}".encode() for iid in instance_ids]
                has_prefix = await settings.memcache.multi_get(*cache_keys)
                for idx, iid in enumerate(instance_ids):
                    if has_prefix[idx]:
                        likely_cached.add(iid)

                if likely_cached:
                    break
            except Exception as e:
                logger.error(f"Error in prefix-aware routing: {e}")
                return None
        if not likely_cached:
            return None

        # Select instances with cache that have reasonable connection counts
        routable = [iid for iid in likely_cached if abs(counts[iid] - min_count) <= 2]
        if not routable:
            return None

        # Sort routable instances by connection count
        result = sorted(
            [self.instances[iid] for iid in routable if iid in self.instances],
            key=lambda inst: counts[inst.instance_id],
        )[:3]

        # Add remaining instances
        for count in sorted(grouped_by_count.keys()):
            result.extend(
                [inst for inst in grouped_by_count[count] if inst.instance_id not in routable]
            )

        return result

    @asynccontextmanager
    async def get_target(self, avoid=[], prefixes=None):
        conn_id = str(uuid.uuid4())
        instance = None
        try:
            targets = await asyncio.wait_for(
                self.get_targets(avoid=avoid, prefixes=prefixes), timeout=7.0
            )
            if not targets:
                yield None
                return
            instance = targets[0]
            try:
                key = f"conn:{self.chute_id}:{instance.instance_id}"
                started_at = time.time()
                count = await asyncio.wait_for(
                    self.redis_client.eval(
                        self.lua_add_connection,
                        1,
                        key,
                        conn_id,
                        int(time.time()),
                        self.connection_expiry,
                    ),
                    timeout=3.0,
                )
                time_taken = time.time() - started_at
                logger.info(
                    f"Assigned {conn_id=} of {self.chute_id} to {instance.instance_id} {count=} {time_taken=}"
                )
            except asyncio.TimeoutError:
                logger.warning(
                    f"Timeout adding connection to {instance.instance_id}, proceeding anyway"
                )
            except Exception as e:
                logger.error(f"Error tracking connection: {e}")
            yield instance
        except asyncio.TimeoutError:
            logger.error("Timeout getting targets")
            # Fallback to random instance
            available = [inst for iid, inst in self.instances.items() if iid not in avoid]
            if available:
                yield random.choice(available)
            else:
                yield None
        except Exception as e:
            logger.error(f"Error getting target: {e}", exc_info=True)
            yield None
        finally:
            if instance:
                try:
                    key = f"conn:{self.chute_id}:{instance.instance_id}"
                    await asyncio.wait_for(
                        self.redis_client.eval(
                            self.lua_remove_connection,
                            1,
                            key,
                            conn_id,
                            int(time.time()),
                            self.connection_expiry,
                        ),
                        timeout=3.0,
                    )
                except asyncio.TimeoutError:
                    logger.warning(f"Timeout cleaning up connection {conn_id}")
                except Exception as e:
                    logger.error(f"Error cleaning up connection {conn_id}: {e}")

    def __del__(self):
        if hasattr(self, "_cleanup_task") and self._cleanup_task:
            self._cleanup_task.cancel()


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
