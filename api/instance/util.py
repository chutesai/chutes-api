"""
Helper functions for instances.
"""

import jwt
import time
import uuid
import base64
import asyncio
import random
import pickle
import traceback
from fastapi import HTTPException, status
from datetime import datetime, timedelta, timezone
from async_lru import alru_cache
from loguru import logger
from contextlib import asynccontextmanager
from api.exceptions import InfraOverload
from api.chute.schemas import Chute
from api.instance.schemas import Instance, LaunchConfig
from api.config import settings
from api.job.schemas import Job
from api.database import get_session
from api.util import has_legacy_private_billing, memcache_get, memcache_set, memcache_delete
from api.user.service import chutes_user_id
from api.bounty.util import create_bounty_if_not_exists, get_bounty_amount, send_bounty_notification
from sqlalchemy.future import select
from sqlalchemy import text, func
from sqlalchemy.orm import aliased, joinedload
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import ec

# Define an alias for the Instance model to use in a subquery
InstanceAlias = aliased(Instance)


@alru_cache(maxsize=1000, ttl=60)
async def load_chute_target_ids(chute_id: str, nonce: int) -> list[str]:
    query = (
        select(Instance.instance_id)
        .where(Instance.active.is_(True))
        .where(Instance.verified.is_(True))
        .where(Instance.chute_id == chute_id)
    )
    async with get_session() as session:
        result = await session.execute(query)
        return [row[0] for row in result.all()]


@alru_cache(maxsize=5000, ttl=300)
async def load_chute_target(instance_id: str) -> Instance:
    # Try memcache first.
    cache_key = f"instance:{instance_id}"
    cached = await memcache_get(cache_key)
    if cached:
        try:
            return pickle.loads(cached)
        except Exception:
            await memcache_delete(cache_key)

    # Load from DB.
    query = (
        select(Instance)
        .where(Instance.instance_id == instance_id)
        .options(
            joinedload(Instance.nodes),
            joinedload(Instance.chute),
            joinedload(Instance.job),
            joinedload(Instance.config),
        )
    )
    async with get_session() as session:
        instance = (await session.execute(query)).unique().scalar_one_or_none()
        if instance:
            # Warm up relationships for serialization
            _ = instance.nodes
            _ = instance.chute
            _ = instance.job
            _ = instance.config

            # Warm up nested relationships on nodes
            if instance.chute:
                _ = instance.chute.image
                _ = instance.chute.logo
                _ = instance.chute.rolling_update
                _ = instance.chute.user
            try:
                serialized = pickle.dumps(instance)
                await memcache_set(cache_key, serialized, exptime=300)
            except Exception:
                ...
        return instance


@alru_cache(maxsize=1000, ttl=60)
async def load_chute_targets(chute_id: str, nonce: int = 0) -> list[Instance]:
    instance_ids = await load_chute_target_ids(chute_id, nonce=nonce)
    instances = []
    for instance_id in instance_ids:
        if (instance := await load_chute_target(instance_id)) is not None:
            instances.append(instance)
    return instances


async def invalidate_instance_cache(chute_id, instance_id: str = None):
    load_chute_target_ids.cache_invalidate(chute_id, nonce=0)
    load_chute_targets.cache_invalidate(chute_id, nonce=0)
    load_chute_target.cache_invalidate(instance_id)
    await memcache_delete(f"instance:{instance_id}".encode())


MANAGERS = {}


class LeastConnManager:
    def __init__(
        self,
        chute_id: str,
        concurrency: int,
        instances: list[Instance],
        connection_expiry: int = 600,
        cleanup_interval: int = 5,
    ):
        self.concurrency = concurrency or 1
        self.chute_id = chute_id
        _u_idx = uuid.UUID(self.chute_id).int % len(settings.cm_redis_client)
        self.redis_client = settings.cm_redis_client[_u_idx]
        self.redis_client_index = _u_idx
        self.instances = {instance.instance_id: instance for instance in instances}
        self.connection_expiry = connection_expiry
        self.cleanup_interval = cleanup_interval
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

    async def close(self):
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
                break
            except Exception as e:
                logger.error(f"Error in cleanup loop: {e}", exc_info=True)
                await asyncio.sleep(self.cleanup_interval)

    async def _cleanup_expired_connections(self):
        now = int(time.time())
        try:
            pattern = f"conn:{self.chute_id}:*"
            now = time.time()
            cutoff = now - self.connection_expiry
            total_removed = 0
            cursor = 0
            while True:
                cursor, keys = await self.redis_client.scan(cursor, match=pattern, count=100)
                if keys:
                    pipe = self.redis_client.pipeline()
                    for key in keys:
                        pipe.zremrangebyscore(key, 0, cutoff)
                    results = await pipe.execute()
                    total_removed += sum(results)
                if cursor == 0:
                    break
            return total_removed
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
        if min_count >= self.concurrency:
            raise InfraOverload(
                f"All instances overwhelmed for {self.chute_id}: min_count={min_count}"
            )

        # Group instances by connection count
        grouped_by_count = {}
        for instance_id, count in counts.items():
            if count >= self.concurrency:
                continue
            if count not in grouped_by_count:
                grouped_by_count[count] = []
            if instance := self.instances.get(instance_id):
                grouped_by_count[count].append(instance)

        # Randomize within each count group
        for instances in grouped_by_count.values():
            random.shuffle(instances)

        # Handle prefix-aware routing if enabled
        if prefixes and random.random() <= 0.95:
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
                yield None, "No infrastructure available to serve request"
                return
            instance = targets[0]
            try:
                key = f"conn:{self.chute_id}:{instance.instance_id}"
                _ = await asyncio.wait_for(
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
            except asyncio.TimeoutError:
                logger.warning(
                    f"Timeout adding connection to {self.redis_client_index=} {instance.instance_id}, proceeding anyway"
                )
            except Exception as e:
                logger.error(f"Error tracking connection: {e}")
            yield instance, None
        except asyncio.TimeoutError:
            logger.error(f"Timeout getting targets on {self.redis_client_index=}")
            # Fallback to random instance
            available = [inst for iid, inst in self.instances.items() if iid not in avoid]
            if available:
                yield random.choice(available), None
            else:
                yield None, "No infrastructure available to serve request"
        except Exception as e:
            if isinstance(e, InfraOverload):
                yield None, "infra_overload"
                return
            logger.error("Error getting target")
            logger.error(str(e))
            logger.error(traceback.format_exc())
            yield None, f"No infrastructure available to serve request, error code: {str(e)}"
        finally:
            if instance:
                try:
                    key = f"conn:{self.chute_id}:{instance.instance_id}"
                    await asyncio.shield(
                        self.redis_client.eval(
                            self.lua_remove_connection,
                            1,
                            key,
                            conn_id,
                            int(time.time()),
                            self.connection_expiry,
                        )
                    )
                except asyncio.TimeoutError:
                    logger.warning(
                        f"Timeout cleaning up connection {conn_id} on {self.redis_client_index=}"
                    )
                except Exception as e:
                    logger.error(f"Error cleaning up connection {conn_id}: {e}")

    def __del__(self):
        if hasattr(self, "_cleanup_task") and self._cleanup_task:
            self._cleanup_task.cancel()


async def get_chute_target_manager(
    chute: Chute, max_wait: int = 0, no_bounty: bool = False, dynonce: bool = False
):
    """
    Select target instances by least connections (with random on equal counts).
    """
    chute_id = chute.chute_id
    nonce = int(time.time()) if dynonce else 0
    instances = await load_chute_targets(chute_id, nonce=nonce)
    started_at = time.time()
    while not instances:
        # Private chutes have a very short-lived bounty, so users aren't billed if they stop making requests.
        bounty_lifetime = 86400
        if (
            not chute.public
            and not has_legacy_private_billing(chute)
            and chute.user_id != await chutes_user_id()
        ):
            bounty_lifetime = 3600 if "/affine" not in chute.name.lower() else 7200

        # Increase the bounty.
        async with get_session() as bounty_session:
            update_result = await bounty_session.execute(
                text("SELECT 1 FROM rolling_updates WHERE chute_id = :chute_id"),
                {"chute_id": chute_id},
            )
            if update_result.first() is not None:
                logger.warning(
                    f"Skipping bounty event for {chute_id=} due to in-progress rolling update."
                )
            elif not no_bounty:
                if await create_bounty_if_not_exists(chute_id, lifetime=bounty_lifetime):
                    logger.success(f"Successfully created a bounty for {chute_id=}")
                current_time = int(time.time())
                window = current_time - (current_time % 30)
                notification_key = f"bounty_notification:{chute_id}:{window}"
                if await settings.redis_client.setnx(notification_key, b"1"):
                    await settings.redis_client.expire(notification_key, 33)
                    amount = await get_bounty_amount(chute_id)
                    if amount:
                        logger.info(f"Bounty for {chute_id=} is now {amount}")
                        await send_bounty_notification(chute_id, amount)
        if not max_wait or time.time() - started_at >= max_wait:
            break
        await asyncio.sleep(1.0)
        instances = await load_chute_targets(chute_id, nonce=time.time())
    if not instances:
        return None
    if chute_id not in MANAGERS:
        MANAGERS[chute_id] = LeastConnManager(
            chute_id=chute_id, concurrency=chute.concurrency or 1, instances=instances
        )
    async with MANAGERS[chute_id].lock:
        MANAGERS[chute_id].instances = {instance.instance_id: instance for instance in instances}
        MANAGERS[chute_id].concurrency = chute.concurrency or 1
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


def _get_es256_verify_key():
    if getattr(settings, "launch_config_public_key_bytes", None):
        return settings.launch_config_public_key_bytes
    if getattr(settings, "launch_config_private_key", None):
        try:
            return settings.launch_config_private_key.public_key()
        except Exception:
            pass
    if getattr(settings, "launch_config_private_key_bytes", None):
        return settings.launch_config_private_key_bytes
    raise RuntimeError("No ES256 verification key configured")


def _decode_chutes_jwt(token: str, *, require_exp: bool) -> dict:
    """
    Decode either the legacy or new asymmetric JWT.
    """
    try:
        header = jwt.get_unverified_header(token)
    except Exception:
        raise jwt.InvalidTokenError("Malformed JWT header")
    alg = header.get("alg")
    if alg not in ("HS256", "ES256"):
        raise jwt.InvalidTokenError("Unsupported JWT alg")
    if alg == "HS256":
        key = settings.launch_config_key
    else:  # ES256
        key = _get_es256_verify_key()
    options = {
        "verify_signature": True,
        "verify_exp": require_exp,
        "verify_iat": True,
        "verify_iss": True,
        "require": ["iat", "iss"] + (["exp"] if require_exp else []),
    }
    return jwt.decode(
        token,
        key,
        algorithms=[alg],
        issuer="chutes",
        options=options,
    )


def create_launch_jwt_v2(launch_config, disk_gb: int = None, egress: bool = False) -> str:
    now = datetime.now(timezone.utc)
    expires_at = now + timedelta(hours=2)
    payload = {
        "exp": int(expires_at.timestamp()),
        "sub": launch_config.config_id,
        "chute_id": launch_config.chute_id,
        "iat": int(now.timestamp()),
        "url": f"https://api.{settings.base_domain}/instances/launch_config/{launch_config.config_id}",
        "env_key": launch_config.env_key,
        "iss": "chutes",
        "egress": egress,
    }
    if launch_config.job_id:
        payload["job_id"] = launch_config.job_id
    if disk_gb:
        payload["disk_gb"] = disk_gb
    encoded_jwt = jwt.encode(payload, settings.launch_config_private_key_bytes, algorithm="ES256")
    return encoded_jwt


def generate_fs_key(launch_config) -> str:
    """
    Generate a chutes secure FS code to unlock encrypted files.
    """
    timestamp = int(time.time())
    message = f"{timestamp}:{launch_config.chute_id}:{launch_config.config_id}".encode()
    signature = settings.launch_config_private_key.sign(message, ec.ECDSA(hashes.SHA256()))
    encoded_signature = base64.urlsafe_b64encode(signature).decode().rstrip("=")
    return f"{timestamp}:{encoded_signature}"


def create_launch_jwt(launch_config: LaunchConfig, disk_gb: int = None) -> str:
    """
    Create JWT for a given launch config (updated chutes lib with new graval etc).
    """
    now = datetime.now(timezone.utc)
    expires_at = now + timedelta(hours=2)
    payload = {
        "exp": int(expires_at.timestamp()),
        "sub": launch_config.config_id,
        "chute_id": launch_config.chute_id,
        "iat": int(now.timestamp()),
        "url": f"https://api.{settings.base_domain}/instances/launch_config/{launch_config.config_id}",
        "env_key": launch_config.env_key,
        "iss": "chutes",
        "env_type": launch_config.env_type
    }
    if launch_config.job_id:
        payload["job_id"] = launch_config.job_id
    if disk_gb:
        payload["disk_gb"] = disk_gb
    encoded_jwt = jwt.encode(payload, settings.launch_config_key, algorithm="HS256")
    return encoded_jwt


def create_job_jwt(job_id, filename: str = None) -> str:
    """
    Create JWT for a single job.
    """
    now = datetime.now(timezone.utc)
    payload = {
        "sub": job_id,
        "iat": int(now.timestamp()),
        "iss": "chutes",
    }
    if filename:
        payload["filename"] = filename
    encoded_jwt = jwt.encode(payload, settings.launch_config_key, algorithm="HS256")
    return encoded_jwt


async def load_launch_config_from_jwt(
    db, config_id: str, token: str, allow_retrieved: bool = False
) -> LaunchConfig:
    detail = "Missing or invalid launch config JWT"
    try:
        payload = _decode_chutes_jwt(token, require_exp=True)
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


async def load_job_from_jwt(db, job_id: str, token: str, filename: str = None) -> Job:
    """
    Load a job from a given JWT, ensuring the sub/chute/etc. match.
    """
    detail = "Missing or invalid JWT"
    try:
        payload = _decode_chutes_jwt(token, require_exp=False)
        assert job_id == payload["sub"], "Job ID in JWT does not match!"
        if filename:
            assert filename == payload["filename"], "Filename mismatch!"
        job = (
            (await db.execute(select(Job).where(Job.job_id == job_id)))
            .unique()
            .scalar_one_or_none()
        )
        job_namespace = uuid.UUID(job_id)
        file_id = str(uuid.uuid5(job_namespace, filename)) if filename else None
        if not job:
            detail = f"{job_id=} not found!"
            logger.warning(detail)
        elif filename and job.output_files and file_id not in job.output_files:
            detail = f"{job_id=} did not have any output file with {filename=}"
            logger.warning(detail)
        else:
            return job
    except jwt.InvalidTokenError:
        logger.warning(f"Attempted to use invalid token for job: {job_id=} {token=}")
    except Exception as exc:
        logger.warning(f"Unhandled exception checking job config JWT: {exc}")

    # If we got here, it failed somewhere.
    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail=detail,
    )


async def update_shutdown_timestamp(instance_id: str):
    query = """
WITH target AS (
    SELECT i.instance_id, COALESCE(c.shutdown_after_seconds, 300) AS shutdown_after_seconds
    FROM instances i
    JOIN chutes c ON i.chute_id = c.chute_id
    WHERE i.instance_id = :instance_id
    FOR UPDATE OF i SKIP LOCKED
)
UPDATE instances
SET stop_billing_at = NOW() + (target.shutdown_after_seconds * INTERVAL '1 second')
FROM target
WHERE instances.instance_id = target.instance_id
RETURNING instances.instance_id;
"""
    try:
        async with get_session() as session:
            await session.execute(text("SET LOCAL lock_timeout = '1s'"))
            await session.execute(text(query), {"instance_id": instance_id})
    except Exception as exc:
        logger.warning(f"Failed to push back instance shutdown time for {instance_id=}: {str(exc)}")
