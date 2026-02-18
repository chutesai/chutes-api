"""
Helper functions for instances.
"""

import jwt
import time
import uuid
import pybase64 as base64
import asyncio
import random
import pickle
import traceback
from fastapi import HTTPException, status
from datetime import datetime, timedelta, timezone
from async_lru import alru_cache
from loguru import logger
from contextlib import asynccontextmanager
from api.constants import (
    INSTANCE_DISABLE_BASE_TIMEOUT,
    MAX_INSTANCE_DISABLES,
    CASCADE_FAILURE_THRESHOLD,
    CASCADE_DETECTION_DELAY,
    CASCADE_PENDING_TTL,
    THRASH_WINDOW_HOURS,
    THRASH_PENALTY_HOURS,
)
from api.exceptions import InfraOverload
from api.chute.schemas import Chute
from api.instance.schemas import Instance, LaunchConfig
from api.config import settings
from api.job.schemas import Job
from api.database import get_session
from api.util import has_legacy_private_billing, notify_deleted
from api.user.service import chutes_user_id
from api.bounty.util import create_bounty_if_not_exists, get_bounty_amount, send_bounty_notification
from sqlalchemy.future import select
from sqlalchemy import text, func
from sqlalchemy.orm import aliased, joinedload
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import ec
from api.server.client import TeeServerClient
from api.server.schemas import Server
from api.server.exceptions import GetEvidenceError
from api.server.service import verify_quote, verify_gpu_evidence
from api.server.util import get_public_key_hash

# Define an alias for the Instance model to use in a subquery
InstanceAlias = aliased(Instance)


@alru_cache(maxsize=3000, ttl=30)
async def load_chute_target_ids(chute_id: str, nonce: int) -> list[str]:
    cache_key = f"inst_ids:{chute_id}:{nonce}"
    cached = await settings.redis_client.get(cache_key)
    if cached is not None:
        if isinstance(cached, bytes):
            cached = cached.decode()
        if not cached:
            return []
        return cached.split("|")

    query = (
        select(Instance.instance_id)
        .where(Instance.active.is_(True))
        .where(Instance.verified.is_(True))
        .where(Instance.chute_id == chute_id)
    )
    async with get_session() as session:
        result = await session.execute(query)
        instance_ids = [row[0] for row in result.all()]
        await settings.redis_client.set(cache_key, "|".join(instance_ids), ex=300)
        return instance_ids


@alru_cache(maxsize=5000, ttl=30)
async def load_chute_target(instance_id: str) -> Instance:
    cache_key = f"instance:{instance_id}"
    cached = await settings.redis_client.get(cache_key)
    if cached:
        try:
            return await asyncio.to_thread(pickle.loads, cached)
        except Exception as exc:
            logger.error(f"Error loading cached instance: {str(exc)}")
            await settings.redis_client.delete(cache_key)

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
                serialized = await asyncio.to_thread(pickle.dumps, instance)
                await settings.redis_client.set(cache_key, serialized, ex=300)
            except Exception as exc:
                logger.error(f"Error setting cache for {instance.instance_id=}: {str(exc)}")
        return instance


@alru_cache(maxsize=3000, ttl=30)
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
    await settings.redis_client.delete(f"inst_ids:{chute_id}:0")
    await settings.redis_client.delete(f"instance:{instance_id}")


async def is_instance_disabled(instance_id: str) -> bool:
    disabled = await settings.redis_client.get(f"instance_disabled:{instance_id}")
    return disabled is not None


async def get_instance_disable_count(instance_id: str) -> int:
    count = await settings.redis_client.get(f"instance_disable_count:{instance_id}")
    if count is None:
        return 0
    return int(count)


class _InstanceInfo:
    """Simple class to hold instance info for notify_deleted."""

    def __init__(self, instance_id: str, miner_hotkey: str, chute_id: str, config_id: str = None):
        self.instance_id = instance_id
        self.miner_hotkey = miner_hotkey
        self.chute_id = chute_id
        self.config_id = config_id


async def _execute_instance_deletion(
    instance_id: str,
    chute_id: str,
    miner_hotkey: str,
    reason: str,
    config_id: str = None,
) -> bool:
    """Actually delete an instance from the database."""
    async with get_session() as session:
        delete_result = await session.execute(
            text("DELETE FROM instances WHERE instance_id = :instance_id"),
            {"instance_id": instance_id},
        )
        if delete_result.rowcount > 0:
            await invalidate_instance_cache(chute_id, instance_id=instance_id)
            await session.execute(
                text(
                    "UPDATE instance_audit SET deletion_reason = :reason WHERE instance_id = :instance_id"
                ),
                {"instance_id": instance_id, "reason": reason},
            )
            await session.commit()
            logger.warning(f"INSTANCE DELETED: {instance_id}: {reason}")
            asyncio.create_task(
                notify_deleted(
                    _InstanceInfo(instance_id, miner_hotkey, chute_id, config_id),
                    message=f"Instance {instance_id} of miner {miner_hotkey} has been deleted: {reason}",
                )
            )
            return True
    return False


async def _check_cascade_and_delete(
    instance_id: str,
    chute_id: str,
    miner_hotkey: str,
    reason: str,
):
    await asyncio.sleep(CASCADE_DETECTION_DELAY)

    # Count how many instances are pending deletion
    pending_key = f"pending_deletion:{instance_id}"
    try:
        cursor = 0
        pending_count = 0
        while True:
            cursor, keys = await settings.lite_redis_client.client.scan(
                cursor, match="pending_deletion:*", count=1000
            )
            pending_count += len(keys)
            if cursor == 0:
                break

        if pending_count >= CASCADE_FAILURE_THRESHOLD:
            # Cascade failure detected - don't delete, just disable temporarily
            logger.error(
                f"CASCADE FAILURE DETECTED: {pending_count} instances pending deletion, "
                f"skipping deletion of {instance_id}"
            )
            # Extend the disable timeout instead of deleting
            disabled_key = f"instance_disabled:{instance_id}"
            await settings.redis_client.expire(disabled_key, INSTANCE_DISABLE_BASE_TIMEOUT * 3)
        else:
            # Safe to delete
            await _execute_instance_deletion(instance_id, chute_id, miner_hotkey, reason)
    except Exception as e:
        logger.error(f"Error in cascade check for {instance_id}: {e}")
    finally:
        # Clean up pending marker
        await settings.redis_client.delete(pending_key)


async def disable_instance(
    instance_id: str,
    chute_id: str,
    miner_hotkey: str,
    skip_disable_loop: bool = False,
    instant_delete: bool = False,
) -> bool:
    disabled_key = f"instance_disabled:{instance_id}"
    count_key = f"instance_disable_count:{instance_id}"

    # Use the raw redis client to ensure we don't silently discard.
    acquired = await settings.redis_client.client.set(
        disabled_key, b"1", nx=True, ex=INSTANCE_DISABLE_BASE_TIMEOUT
    )
    if not acquired:
        return False

    disable_count = await settings.redis_client.incr(count_key)
    await settings.redis_client.expire(count_key, 3600)

    should_delete = disable_count > MAX_INSTANCE_DISABLES or skip_disable_loop or instant_delete

    if should_delete:
        if instant_delete:
            reason = "catastrophic error (invalid/empty response or verification failure)"
        elif skip_disable_loop:
            reason = f"server error after {disable_count} consecutive failure events"
        else:
            reason = (
                f"max consecutive failures reached after {disable_count - 1} temporary disables"
            )

        # For catastrophic errors (instant_delete) or server errors (skip_disable_loop),
        # delete immediately - no cascade check needed because connection worked
        if instant_delete or skip_disable_loop:
            await _execute_instance_deletion(instance_id, chute_id, miner_hotkey, reason)
        else:
            # For network-related deletions (timeouts, disconnects), use cascade detection
            # Mark as pending deletion and schedule background check
            pending_key = f"pending_deletion:{instance_id}"
            await settings.lite_redis_client.set(pending_key, b"1", ex=CASCADE_PENDING_TTL)
            asyncio.create_task(
                _check_cascade_and_delete(instance_id, chute_id, miner_hotkey, reason)
            )
    else:
        # Disable temporarily with increasing timeout
        timeout_seconds = INSTANCE_DISABLE_BASE_TIMEOUT * disable_count
        await settings.redis_client.expire(disabled_key, timeout_seconds)
        logger.warning(
            f"INSTANCE DISABLED: {instance_id} for {timeout_seconds}s (disable_count={disable_count})"
        )

    return True


async def clear_instance_disable_state(instance_id: str) -> None:
    await settings.redis_client.delete(f"instance_disabled:{instance_id}")
    await settings.redis_client.delete(f"instance_disable_count:{instance_id}")


MANAGERS = {}


class LeastConnManager:
    def __init__(
        self,
        chute_id: str,
        concurrency: int,
        instances: list[Instance],
        connection_expiry: int = 3600,
    ):
        self.concurrency = concurrency or 1
        self.chute_id = chute_id
        self.redis_client = settings.redis_client
        self.instances = {instance.instance_id: instance for instance in instances}
        self.connection_expiry = connection_expiry
        self.mean_count = None
        self._last_instance_utilization = None
        self._last_conn_used = None
        self.lock = asyncio.Lock()

    async def get_connection_counts(self, instance_ids: list[str]) -> dict[str, int]:
        """
        Get current connection counts for instances via MGET.
        """
        keys = [f"cc:{self.chute_id}:{iid}" for iid in instance_ids]
        try:
            values = await self.redis_client.client.mget(keys)
            if not values:
                return {iid: 0 for iid in instance_ids}
            return {iid: int(v or 0) for iid, v in zip(instance_ids, values)}
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
        if prefixes:
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
                cache_keys = [f"pfx:{prefix_hash}:{iid}" for iid in instance_ids]
                has_prefix = await settings.redis_client.mget(cache_keys)
                if has_prefix is not None:
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
        )[:2]
        r_inst_ids = {r.instance_id for r in result}

        # Add remaining instances
        for count in sorted(grouped_by_count.keys()):
            result.extend(
                [inst for inst in grouped_by_count[count] if inst.instance_id not in r_inst_ids]
            )

        return result

    async def _track_active(self, instance_id: str):
        """Fire-and-forget tracking of active chutes/instances for gauge enumeration."""
        try:
            pipe = self.redis_client.client.pipeline()
            pipe.sadd("active_chutes", self.chute_id)
            pipe.expire("active_chutes", self.connection_expiry)
            pipe.sadd(f"cc_inst:{self.chute_id}", instance_id)
            pipe.expire(f"cc_inst:{self.chute_id}", self.connection_expiry)
            pipe.set(f"cc_conc:{self.chute_id}", self.concurrency, ex=self.connection_expiry)
            await pipe.execute()
        except Exception as e:
            logger.error(f"Error tracking active chute/instance: {e}")

    @asynccontextmanager
    async def get_target(self, avoid=[], prefixes=None):
        instance = None
        try:
            targets = await asyncio.wait_for(
                self.get_targets(avoid=avoid, prefixes=prefixes), timeout=7.0
            )
            if not targets:
                yield None, "No infrastructure available to serve request"
                return

            # Find first non-disabled instance (lazy check with caching)
            instance = None
            for candidate in targets:
                if not await is_instance_disabled(candidate.instance_id):
                    instance = candidate
                    break

            if not instance:
                yield None, "infra_overload"
                return

            key = f"cc:{self.chute_id}:{instance.instance_id}"
            try:
                await asyncio.wait_for(
                    self.redis_client.client.incr(key),
                    timeout=3.0,
                )
                await self.redis_client.expire(key, self.connection_expiry)
            except asyncio.TimeoutError:
                logger.warning(
                    f"Timeout incrementing connection count for {instance.instance_id}, proceeding anyway"
                )
            except Exception as e:
                logger.error(f"Error tracking connection: {e}")

            # Track active chute/instance for gauge enumeration (fire-and-forget)
            asyncio.create_task(self._track_active(instance.instance_id))

            yield instance, None
        except asyncio.TimeoutError:
            logger.error("Timeout getting targets")
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
                    key = f"cc:{self.chute_id}:{instance.instance_id}"

                    async def _decr():
                        val = await self.redis_client.client.decr(key)
                        if val < 0:
                            await self.redis_client.client.set(key, 0, ex=self.connection_expiry)

                    await asyncio.shield(_decr())
                except asyncio.TimeoutError:
                    logger.warning(f"Timeout cleaning up connection for {instance.instance_id}")
                except Exception as e:
                    logger.error(f"Error cleaning up connection for {instance.instance_id}: {e}")


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
        if not no_bounty:
            if await create_bounty_if_not_exists(chute_id, lifetime=bounty_lifetime):
                logger.success(f"Successfully created a bounty for {chute_id=}")
            amount = await get_bounty_amount(chute_id)
            if amount:
                current_time = int(time.time())
                window = current_time - (current_time % 30)
                notification_key = f"bounty_notification:{chute_id}:{window}"
                if await settings.redis_client.setnx(notification_key, b"1"):
                    await settings.redis_client.expire(notification_key, 33)
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


def create_launch_jwt_v2(
    launch_config: LaunchConfig, disk_gb: int = None, egress: bool = False
) -> str:
    now = datetime.now(timezone.utc)
    expires_at = now + timedelta(hours=2)
    env_type = launch_config.env_type if launch_config.env_type else "graval"
    payload = {
        "exp": int(expires_at.timestamp()),
        "sub": launch_config.config_id,
        "chute_id": launch_config.chute_id,
        "iat": int(now.timestamp()),
        "url": f"https://api.{settings.base_domain}/instances/launch_config/{launch_config.config_id}",
        "env_key": launch_config.env_key,
        "iss": "chutes",
        "egress": egress,
        "env_type": env_type,
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
    env_type = launch_config.env_type if launch_config.env_type else "graval"
    payload = {
        "exp": int(expires_at.timestamp()),
        "sub": launch_config.config_id,
        "chute_id": launch_config.chute_id,
        "iat": int(now.timestamp()),
        "url": f"https://api.{settings.base_domain}/instances/launch_config/{launch_config.config_id}",
        "env_key": launch_config.env_key,
        "iss": "chutes",
        "env_type": env_type,
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


async def _update_shutdown_timestamp(instance_id: str):
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
            logger.success(f"Updated instance shutdown timestamp: {instance_id=}")
    except Exception as exc:
        logger.warning(f"Failed to push back instance shutdown time for {instance_id=}: {str(exc)}")


async def update_shutdown_timestamp(instance_id: str):
    key = f"shutdownlock:{instance_id}"
    try:
        acquired = await settings.redis_client.set(key, b"1", nx=True, ex=60)
        if not acquired:
            return
        await _update_shutdown_timestamp(instance_id)
    except Exception as exc:
        logger.warning(f"Failed to push back instance shutdown time for {instance_id=}: {exc}")
        try:
            await settings.redis_client.delete(key)
        except Exception:
            pass


async def verify_tee_chute(
    db,
    instance,
    launch_config,
    deployment_id: str,
    expected_nonce: str,
):
    """
    Verify TEE chute by fetching evidence from the attestation proxy and validating it.

    Args:
        db: Database session
        instance: Instance object
        launch_config: LaunchConfig object
        deployment_id: Deployment ID for the chute
        expected_nonce: Expected nonce for verification

    Raises:
        HTTPException: If verification fails
    """
    try:
        # Get the server from the database
        server_query = select(Server).where(
            Server.ip == instance.host, Server.miner_hotkey == launch_config.miner_hotkey
        )
        server = (await db.execute(server_query)).scalar_one_or_none()
        if not server:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Server not found for IP {instance.host} and miner {launch_config.miner_hotkey}",
            )

        # Use the TeeServerClient to get evidence from the chute proxy
        client = TeeServerClient(server)

        # Get quote, GPU evidence, cert from chute verify endpoint (no nonce; chute uses stored nonce)
        quote, gpu_evidence, cert = await client.get_chute_evidence(deployment_id)
        expected_cert_hash = get_public_key_hash(cert)

        # Verify the quote against the expected nonce and cert hash
        await verify_quote(quote, expected_nonce, expected_cert_hash)

        # Verify GPU attestation evidence with expected nonce
        await verify_gpu_evidence(gpu_evidence, expected_nonce)

        logger.success(f"Successfully verified attestation for chute deployment {deployment_id}")
    except GetEvidenceError as exc:
        logger.error(f"Failed to get evidence from chute proxy for {instance.host}: {exc}")
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="Attestation service unavailable. The chute attestation proxy could not be reached or returned an error. Please ensure the server is accessible and the attestation service is running.",
        )
    except HTTPException:
        raise
    except Exception as exc:
        logger.error(f"Unexpected error verifying chute evidence: {exc}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to verify chute attestation: {str(exc)}",
        )


async def is_thrashing_miner(
    db, miner_hotkey: str, chute_id: str, instance_created_at: datetime = None
) -> bool:
    """
    Check if a miner is thrashing (deleted an active instance of the same chute
    within THRASH_WINDOW_HOURS before creating this new instance).

    If instance_created_at is None, uses database NOW() for the check.

    Returns True if the miner is thrashing and should not receive bounty/urgency boosts.
    """
    if instance_created_at is None:
        # Use database NOW() for timestamp
        result = await db.execute(
            text(f"""
                SELECT EXISTS (
                    SELECT 1
                    FROM instance_audit
                    WHERE miner_hotkey = :miner_hotkey
                      AND chute_id = :chute_id
                      AND activated_at IS NOT NULL
                      AND deleted_at IS NOT NULL
                      AND deleted_at > NOW() - INTERVAL '{THRASH_WINDOW_HOURS} hours'
                      AND deleted_at <= NOW()
                      AND valid_termination IS NOT TRUE
                )
            """),
            {
                "miner_hotkey": miner_hotkey,
                "chute_id": chute_id,
            },
        )
    else:
        # Ensure instance_created_at is timezone-naive for comparison
        if instance_created_at.tzinfo is not None:
            created_at_naive = instance_created_at.replace(tzinfo=None)
        else:
            created_at_naive = instance_created_at

        result = await db.execute(
            text("""
                SELECT EXISTS (
                    SELECT 1
                    FROM instance_audit
                    WHERE miner_hotkey = :miner_hotkey
                      AND chute_id = :chute_id
                      AND activated_at IS NOT NULL
                      AND deleted_at IS NOT NULL
                      AND deleted_at > :window_start
                      AND deleted_at <= :created_at
                )
            """),
            {
                "miner_hotkey": miner_hotkey,
                "chute_id": chute_id,
                "window_start": created_at_naive - timedelta(hours=THRASH_WINDOW_HOURS),
                "created_at": created_at_naive,
            },
        )
    return result.scalar()


async def is_instance_in_thrash_penalty(
    db, instance_id: str, miner_hotkey: str, chute_id: str, activated_at: datetime
) -> bool:
    """
    Check if an instance is still within its thrash penalty period.

    Used by the autoscaler to skip compute_multiplier updates for thrashing instances.
    """
    if activated_at is None:
        return False

    # Ensure activated_at is timezone-naive for comparison
    if activated_at.tzinfo is not None:
        activated_at_naive = activated_at.replace(tzinfo=None)
    else:
        activated_at_naive = activated_at

    # Check if still in penalty period
    now = datetime.utcnow()
    if now >= activated_at_naive + timedelta(hours=THRASH_PENALTY_HOURS):
        return False

    # Check if this miner thrashed when creating this instance
    # Look for deleted active instances before this instance's creation
    result = await db.execute(
        text("""
            SELECT i.created_at
            FROM instances i
            WHERE i.instance_id = :instance_id
        """),
        {"instance_id": instance_id},
    )
    row = result.fetchone()
    if not row or not row.created_at:
        return False

    instance_created_at = row.created_at
    if instance_created_at.tzinfo is not None:
        instance_created_at = instance_created_at.replace(tzinfo=None)

    return await is_thrashing_miner(db, miner_hotkey, chute_id, instance_created_at)
