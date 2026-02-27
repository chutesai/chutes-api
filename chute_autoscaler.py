"""
Auto-scale chutes based on utilization.
"""

import gc
import os
import math
import asyncio
import argparse
import random
import uuid
from contextlib import asynccontextmanager
from functools import wraps
from collections import defaultdict
from loguru import logger
from datetime import timedelta, datetime, timezone
from typing import Dict, Optional, Set, List, Tuple
import aiohttp
from sqlalchemy import (
    text,
    select,
    func,
    and_,
    or_,
)
from sqlalchemy.exc import OperationalError
import api.database.orms  # noqa
from sqlalchemy.orm import selectinload, joinedload
from api.database import get_session
from api.config import settings
from api.bounty.util import (
    check_bounty_exists,
    delete_bounty,
    get_bounty_info,
    get_bounty_infos,
    send_bounty_notification,
)
from api.user.service import chutes_user_id
from api.util import has_legacy_private_billing, notify_deleted
from api.chute.schemas import Chute, NodeSelector, RollingUpdate
from api.instance.schemas import Instance, LaunchConfig
from api.instance.util import invalidate_instance_cache
from api.metrics.util import reconcile_connection_counts
from api.capacity_log.schemas import CapacityLog
from watchtower import purge, purge_and_notify  # noqa
from api.constants import (
    UNDERUTILIZED_CAP,
    UTILIZATION_SCALE_UP,
    UTILIZATION_SCALE_DOWN,
    RATE_LIMIT_SCALE_UP,
    SCALE_DOWN_LOOKBACK_MINUTES,
    SCALE_DOWN_MAX_DROP_RATIO,
)


# Distributed lock to prevent concurrent autoscaler runs
AUTOSCALER_LOCK_KEY = "autoscaler:lock"
AUTOSCALER_LOCK_TTL = 180  # 3 minutes max (should complete in <1 min normally)


class LockNotAcquired(Exception):
    """Raised when the autoscaler lock cannot be acquired."""

    pass


@asynccontextmanager
async def autoscaler_lock(soft_mode: bool = False, skip_lock: bool = False):
    """
    Distributed lock using Redis to prevent concurrent autoscaler runs.

    Uses SET NX with expiry to atomically acquire lock.
    Releases lock on exit (or lets it expire if process crashes).

    If soft_mode=True and lock is held, exits quietly instead of raising.
    Full mode retries for up to 60 seconds before failing.
    If skip_lock=True (for dry-run), yields immediately without acquiring lock.
    """
    if skip_lock:
        logger.info("Skipping lock acquisition (dry-run mode)")
        yield
        return

    lock_id = str(uuid.uuid4())
    acquired = False
    max_retries = 12  # 12 retries * 5 seconds = 60 seconds
    retry_delay = 5

    try:
        for attempt in range(max_retries):
            acquired = await settings.redis_client.set(
                AUTOSCALER_LOCK_KEY,
                lock_id,
                nx=True,
                ex=AUTOSCALER_LOCK_TTL,
            )
            if acquired:
                break

            ttl = await settings.redis_client.ttl(AUTOSCALER_LOCK_KEY)
            if soft_mode:
                logger.info(f"Lock held (TTL: {ttl}s), skipping soft mode run")
                raise LockNotAcquired()

            # Full mode: retry with backoff
            if attempt < max_retries - 1:
                logger.info(
                    f"Lock held (TTL: {ttl}s), retrying in {retry_delay}s "
                    f"(attempt {attempt + 1}/{max_retries})"
                )
                await asyncio.sleep(retry_delay)
            else:
                raise RuntimeError(
                    f"Another autoscaler is running (lock TTL: {ttl}s). "
                    f"Gave up after {max_retries} retries (~{max_retries * retry_delay}s)."
                )

        logger.info(f"Acquired autoscaler lock: {lock_id}")
        yield
    finally:
        if acquired:
            current = await settings.redis_client.get(AUTOSCALER_LOCK_KEY)
            if current and current.decode() == lock_id:
                await settings.redis_client.delete(AUTOSCALER_LOCK_KEY)
                logger.info(f"Released autoscaler lock: {lock_id}")


def retry_on_db_failure(max_retries=3, delay=1.0):
    """
    Decorator to retry async DB operations on OperationalError (timeouts/deadlocks).
    """

    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            last_error = None
            for attempt in range(max_retries):
                try:
                    return await func(*args, **kwargs)
                except OperationalError as e:
                    last_error = e
                    logger.warning(
                        f"Database operation {func.__name__} failed (attempt {attempt + 1}/{max_retries}): {e}. "
                        f"Retrying in {delay}s..."
                    )
                    await asyncio.sleep(delay)
            logger.error(f"Database operation {func.__name__} failed after {max_retries} attempts.")
            raise last_error

        return wrapper

    return decorator


@retry_on_db_failure()
async def get_scale_down_permission(
    chute_id: str, current_count: int, proposed_target: int, current_rate_limit: float = 0
) -> Tuple[bool, str]:
    """
    Check if scale-down is permitted based on historical capacity_log trends.

    Returns (permitted, reason) tuple.

    Scale-down is permitted if:
    1. We have enough historical data (at least 3 samples)
    2. Proposed target isn't drastically below recent average (within SCALE_DOWN_MAX_DROP_RATIO)
    3. No significant rate limiting is currently occurring (uses current metrics, not historical)

    This prevents thrashing and respects bursty traffic patterns.
    """
    # Check for current rate limiting first (using live metrics passed from context)
    # Previously this checked MAX(rate_limit) over 90 minutes which was too conservative -
    # old rate limiting would block scale-downs even after traffic had subsided
    if current_rate_limit >= 0.01:
        return False, f"rate_limiting_in_window ({current_rate_limit:.1%})"

    async with get_session() as session:
        await session.execute(text("SET LOCAL statement_timeout = '5s'"))
        result = await session.execute(
            text("""
                SELECT
                    AVG(target_count) as avg_target,
                    MAX(target_count) as max_target,
                    AVG(utilization_15m) as avg_util,
                    COUNT(*) as sample_count
                FROM capacity_log
                WHERE chute_id = :chute_id
                  AND timestamp >= NOW() - make_interval(mins => :lookback_minutes)
            """),
            {"chute_id": chute_id, "lookback_minutes": SCALE_DOWN_LOOKBACK_MINUTES},
        )
        row = result.fetchone()

        if not row or row.sample_count < 3:
            return False, "insufficient_history"

        # Check if proposed target is within acceptable range of rolling average
        min_allowed_target = max(1, int(float(row.avg_target) * SCALE_DOWN_MAX_DROP_RATIO))
        if proposed_target < min_allowed_target:
            return (
                False,
                f"below_moving_avg (proposed={proposed_target}, avg={row.avg_target:.1f}, min_allowed={min_allowed_target})",
            )

        return True, "permitted"


# Constants
PROMETHEUS_URL = os.getenv("PROMETHEUS_URL", "http://prometheus-server")
MIN_CHUTES_FOR_SCALING = 10
PRICE_COMPATIBILITY_THRESHOLD = 0.67
AUTOSCALER_FULL_INTERVAL_SECONDS = int(os.getenv("AUTOSCALER_FULL_INTERVAL_SECONDS", "1200"))
AUTOSCALER_SOFT_INTERVAL_SECONDS = int(os.getenv("AUTOSCALER_SOFT_INTERVAL_SECONDS", "120"))

# Forced donation anti-thrashing settings
# Chutes that were starving recently cannot be donors (prevents A->B->A donation cycles)
STARVING_COOLDOWN_MINUTES = 90
# Redis key prefix for tracking recently starving chutes
STARVING_HISTORY_KEY_PREFIX = "starving:"

# Higher min instance counts for some chutes...
LIMIT_OVERRIDES = {}
FAILSAFE = {
    "0d7184a2-32a3-53e0-9607-058c37edaab5": 30,
    "722df757-203b-58df-b54b-22130fd1fc53": 16,
    "398651e1-5f85-5e50-a513-7c5324e8e839": 15,
    "e51e818e-fa63-570d-9f68-49d7d1b4d12f": 10,
    "08a7a60f-6956-5a9e-9983-5603c3ac5a38": 10,
    "2ff25e81-4586-5ec8-b892-3a6f342693d7": 6,
    "8f3bb827-b9e6-5487-88bc-ee8f0c6f5810": 6,
    "6320ab82-9e94-5d63-8e38-d136f61dc157": 3,
    "bbb7f19b-45e4-5172-ae38-6088612c9889": 3,
}


class AutoScaleContext:
    def __init__(
        self,
        chute_id,
        metrics,
        info,
        supported_gpus,
        instances: List[Instance],
        db_now: datetime,
        gpu_count=None,
    ):
        self.chute_id = chute_id
        self.metrics = metrics
        self.info = info
        self.supported_gpus = supported_gpus
        if gpu_count is None and info:
            node_selector = getattr(info, "node_selector", None) or {}
            gpu_count = node_selector.get("gpu_count")
        self.gpu_count = gpu_count
        self.tee = info.tee if info else False
        self.current_version = info.version if info else None
        self.instances = instances
        self.db_now = db_now

        # Map actual hardware to specific instance objects
        # Only include established instances (active for 1+ hour) for donor consideration
        self.hardware_map = defaultdict(list)
        self.established_instance_count = 0
        self.old_instance_count = 0
        for inst in instances:
            if inst.nodes:
                is_established = db_now.replace(tzinfo=None) - inst.activated_at.replace(
                    tzinfo=None
                ) >= timedelta(minutes=63)
                if is_established:
                    gpu_id = inst.nodes[0].gpu_identifier
                    self.hardware_map[gpu_id].append(inst)
                    self.established_instance_count += 1
            if self.current_version and inst.version != self.current_version:
                self.old_instance_count += 1

        # Computed metrics
        self.utilization_basis = max(
            metrics["utilization"].get("5m", 0), metrics["utilization"].get("15m", 0)
        )
        # Track all rate limit windows
        self.rate_limit_5m = metrics["rate_limit_ratio"].get("5m", 0)
        self.rate_limit_15m = metrics["rate_limit_ratio"].get("15m", 0)
        self.rate_limit_1h = metrics["rate_limit_ratio"].get("1h", 0)
        # For scale-up decisions, use the most recent rate limit values
        self.rate_limit_basis = max(self.rate_limit_5m, self.rate_limit_15m)
        # For scale-down prevention, ANY rate limiting in any window blocks it
        self.any_rate_limiting = (
            self.rate_limit_5m > 0 or self.rate_limit_15m > 0 or self.rate_limit_1h > 0
        )

        # Request volume for demand-based scaling
        self.completed_5m = metrics["completed_requests"].get("5m", 0)
        self.completed_15m = metrics["completed_requests"].get("15m", 0)
        self.rate_limited_count_5m = metrics["rate_limited_requests"].get("5m", 0)
        self.rate_limited_count_15m = metrics["rate_limited_requests"].get("15m", 0)
        self.current_count = info.instance_count if info else 0
        self.threshold = info.scaling_threshold if info else UTILIZATION_SCALE_UP
        if not self.threshold:
            self.threshold = UTILIZATION_SCALE_UP
        # Scale-down threshold is proportionally lower than scale-up threshold
        # Default: 0.35/0.6 = 0.583 ratio
        self.scale_down_threshold = self.threshold * (UTILIZATION_SCALE_DOWN / UTILIZATION_SCALE_UP)
        # Starving threshold: when utilization is high enough to justify forced donations
        # Fixed at 80%, unless the chute's scaling threshold is >= 80%, then 10% above that (capped at 100%)
        self.starving_threshold = min(1.0, max(0.80, self.threshold * 1.10))
        self.has_rolling_update = info.has_rolling_update if info else False
        # max_instances: None means unbounded, use a large number for comparisons
        self.max_instances = info.max_instances if (info and info.max_instances) else 10000
        self.public = info.public if info else True
        # Pending instances: unverified but recently created (in process of starting up)
        self.pending_instance_count = info.pending_instance_count if info else 0
        # Concurrency: how many concurrent requests per instance before rate limiting
        self.concurrency = info.concurrency if info else 1
        # Optional manual boost multiplier (fine-tuning)
        self.manual_boost = info.manual_boost if info else 1.0

        # Decision outputs
        self.target_count = self.current_count
        self.action = "no_action"
        self.urgency_score = 0.0
        self.smoothed_urgency = 0.0
        self.smoothed_util = self.utilization_basis
        self.is_starving = False
        self.is_donor = False
        self.is_critical_donor = False
        self.blocked_by_starving = False
        self.downscale_amount = 0
        self.upscale_amount = 0
        self.preferred_downscale_gpus = set()
        self.boost = 1.0
        self.base_multiplier = 0.0  # Base compute multiplier from node selector
        self.effective_multiplier = 0.0  # Total effective compute multiplier for miners
        self.cm_delta_ratio = 0.0  # Ratio of effective/base (how much boost overall)
        self.has_bounty = False


async def instance_cleanup():
    """
    Clean up instances that should have been verified by now.
    """
    async with get_session() as session:
        query = (
            select(Instance)
            .join(LaunchConfig, Instance.config_id == LaunchConfig.config_id, isouter=True)
            .where(
                or_(
                    and_(
                        Instance.verified.is_(False),
                        or_(
                            and_(
                                Instance.config_id.isnot(None),
                                Instance.created_at <= func.now() - timedelta(hours=3, minutes=0),
                            ),
                            and_(
                                Instance.config_id.is_(None),
                                Instance.created_at <= func.now() - timedelta(hours=3, minutes=0),
                            ),
                        ),
                    ),
                    and_(
                        Instance.verified.is_(True),
                        Instance.active.is_(False),
                        Instance.config_id.isnot(None),
                        LaunchConfig.verified_at <= func.now() - timedelta(hours=3, minutes=0),
                    ),
                )
            )
            .options(joinedload(Instance.chute))
        )
        total = 0
        for instance in (await session.execute(query)).unique().scalars().all():
            delta = int((datetime.now() - instance.created_at.replace(tzinfo=None)).total_seconds())
            logger.warning(
                f"Purging instance {instance.instance_id} of {instance.chute.name} "
                f"which was created {instance.created_at} ({delta} seconds ago)..."
            )
            logger.warning(f"  {instance.verified=} {instance.active=}")
            await purge_and_notify(
                instance, reason="Instance failed to verify within a reasonable amount of time"
            )
            total += 1
        if total:
            logger.success(f"Purged {total} total unverified+old instances.")


# EMA smoothing for stability
EMA_ALPHA_URGENCY = 0.3  # For urgency/boost calculations
EMA_ALPHA_UTIL = 0.4  # For utilization (slightly more reactive)
EMA_REDIS_TTL = 7200  # 2 hours - survive missed runs but not stale forever


async def get_smoothed_metrics(chute_ids: List[str]) -> Dict[str, Dict[str, float]]:
    """
    Fetch previously smoothed metrics from Redis for all chutes.
    Returns dict of chute_id -> {"urgency": float, "util": float}
    """
    if not chute_ids:
        return {}

    pipe = settings.redis_client.pipeline()
    for chute_id in chute_ids:
        pipe.hgetall(f"smooth:{chute_id}")
    results = await pipe.execute()

    smoothed = {}
    for chute_id, data in zip(chute_ids, results):
        if data:
            smoothed[chute_id] = {
                "urgency": float(data.get(b"urgency", 0)),
                "util": float(data.get(b"util", 0)),
            }
    return smoothed


async def save_smoothed_metrics(metrics: Dict[str, Dict[str, float]]):
    """
    Save smoothed metrics to Redis.
    metrics: dict of chute_id -> {"urgency": float, "util": float}
    """
    if not metrics:
        return

    pipe = settings.redis_client.pipeline()
    for chute_id, values in metrics.items():
        pipe.hset(
            f"smooth:{chute_id}",
            mapping={
                "urgency": str(values["urgency"]),
                "util": str(values["util"]),
            },
        )
        pipe.expire(f"smooth:{chute_id}", EMA_REDIS_TTL)
    await pipe.execute()


def calculate_ema(current: float, previous: float | None, alpha: float) -> float:
    """
    Calculate exponential moving average.
    If no previous value, return current (first data point).
    """
    if previous is None:
        return current
    return alpha * current + (1 - alpha) * previous


async def mark_chute_as_starving(chute_id: str):
    """
    Mark a chute as recently starving in Redis.
    This prevents the chute from being used as a forced donation donor
    for STARVING_COOLDOWN_MINUTES, avoiding A->B->A donation thrashing.
    """
    key = f"{STARVING_HISTORY_KEY_PREFIX}{chute_id}"
    await settings.redis_client.set(key, "1", ex=STARVING_COOLDOWN_MINUTES * 60)


async def get_recently_starving_chutes(chute_ids: List[str]) -> Set[str]:
    """
    Check which chutes have been starving recently (within STARVING_COOLDOWN_MINUTES).
    Returns set of chute_ids that were recently starving.
    """
    if not chute_ids:
        return set()

    pipe = settings.redis_client.pipeline()
    for chute_id in chute_ids:
        pipe.exists(f"{STARVING_HISTORY_KEY_PREFIX}{chute_id}")
    results = await pipe.execute()

    return {chute_id for chute_id, exists in zip(chute_ids, results) if exists}


# Compute multiplier adjustment timing constants
# No adjustment for the first N hours after activation (miners keep their original boost)
COMPUTE_MULTIPLIER_HOLD_HOURS = 2.0
# Total hours until fully adjusted to target (includes hold period)
COMPUTE_MULTIPLIER_FULL_ADJUST_HOURS = 8.0
# Ramp duration (after hold period)
COMPUTE_MULTIPLIER_RAMP_HOURS = COMPUTE_MULTIPLIER_FULL_ADJUST_HOURS - COMPUTE_MULTIPLIER_HOLD_HOURS


def _calculate_blended_multiplier(
    current: float | None,
    target: float,
    hours_since_activation: float,
) -> float | None:
    """
    Calculate the blended compute_multiplier based on time since activation.

    Returns None if no update needed (still in hold period with existing value).
    """
    if hours_since_activation <= COMPUTE_MULTIPLIER_HOLD_HOURS:
        # In hold period - only set if NULL
        return target if current is None else None

    if hours_since_activation >= COMPUTE_MULTIPLIER_FULL_ADJUST_HOURS:
        # Past full adjustment - clamp to target
        return target

    # In ramp period - ease-in blend (t² curve)
    current_val = current if current is not None else target
    t = (hours_since_activation - COMPUTE_MULTIPLIER_HOLD_HOURS) / COMPUTE_MULTIPLIER_RAMP_HOURS
    blend = t * t
    return current_val * (1 - blend) + target * blend


async def simulate_miner_scores(
    chute_effective_multipliers: Dict[str, float],
) -> Dict[str, Dict]:
    """
    Simulate what miner scores would be if the updated effective compute multipliers
    were applied to instances. This uses the exact same scoring logic as
    metasync/shared.py:get_scoring_data but with projected multiplier changes.

    Args:
        chute_effective_multipliers: Dict mapping chute_id -> new effective_compute_multiplier
                                    (calculated from updated boosts in dry-run)

    Returns:
        Dict with:
        - current_scores: current normalized scores per miner
        - simulated_scores: projected scores if multipliers were updated
        - current_raw: current raw compute_units per miner
        - simulated_raw: projected raw compute_units per miner
        - instance_changes: list of instance-level multiplier changes
        - miner_changes: summary of score changes per miner
    """
    from metasync.constants import SCORING_INTERVAL

    logger.info("Simulating miner scores with updated compute multipliers...")

    # Use the same interval as metasync scoring
    interval = SCORING_INTERVAL

    async with get_session() as session:
        await session.execute(text("SET LOCAL statement_timeout = '30s'"))

        metagraph_result = await session.execute(
            text(f"""
                SELECT coldkey, hotkey, blacklist_reason
                FROM metagraph_nodes
                WHERE netuid = {settings.netuid} AND node_id >= 0
            """)
        )
        hot_cold_map = {}
        blacklisted_hotkeys = set()
        for coldkey, hotkey, blacklist_reason in metagraph_result:
            hot_cold_map[hotkey] = coldkey
            if blacklist_reason:
                blacklisted_hotkeys.add(hotkey)

        coldkey_counts = {}
        for hotkey, coldkey in hot_cold_map.items():
            coldkey_counts[coldkey] = coldkey_counts.get(coldkey, 0) + 1

        current_query = text(f"""
            WITH billed_instances AS (
                SELECT
                    ia.miner_hotkey,
                    ia.instance_id,
                    ia.chute_id,
                    ia.created_at,
                    ia.activated_at,
                    ia.deleted_at,
                    ia.stop_billing_at,
                    ia.compute_multiplier,
                    ia.bounty,
                    GREATEST(ia.created_at, now() - interval '{interval}') as billing_start,
                    LEAST(
                        COALESCE(ia.stop_billing_at, now()),
                        COALESCE(ia.deleted_at, now()),
                        now()
                    ) as billing_end
                FROM instance_audit ia
                WHERE ia.activated_at IS NOT NULL
                  AND (
                      (
                        ia.billed_to IS NULL
                        AND ia.deleted_at IS NOT NULL
                        AND ia.deleted_at - ia.activated_at >= INTERVAL '1 hour'
                      )
                      OR ia.valid_termination IS TRUE
                      OR ia.deletion_reason in (
                          'job has been terminated due to insufficient user balance',
                          'user-defined/private chute instance has not been used since shutdown_after_seconds',
                          'user has zero/negative balance (private chute)'
                      )
                      OR ia.deletion_reason LIKE '%has an old version%'
                      OR ia.deleted_at IS NULL
                  )
                  AND (ia.deleted_at IS NULL OR ia.deleted_at >= now() - interval '{interval}')
            ),
            instance_weighted_compute AS (
                SELECT
                    bi.instance_id,
                    bi.miner_hotkey,
                    bi.chute_id,
                    bi.billing_start,
                    bi.billing_end,
                    bi.bounty,
                    bi.compute_multiplier as fallback_multiplier,
                    COALESCE(
                        SUM(
                            EXTRACT(EPOCH FROM (
                                LEAST(COALESCE(ich.ended_at, now()), bi.billing_end)
                                - GREATEST(ich.started_at, bi.billing_start)
                            )) * ich.compute_multiplier
                        ),
                        EXTRACT(EPOCH FROM (bi.billing_end - bi.billing_start)) * COALESCE(bi.compute_multiplier, 1.0)
                    ) AS weighted_compute_units,
                    EXTRACT(EPOCH FROM (bi.billing_end - bi.billing_start)) AS billing_seconds
                FROM billed_instances bi
                LEFT JOIN instance_compute_history ich
                       ON ich.instance_id = bi.instance_id
                      AND ich.started_at < bi.billing_end
                      AND (ich.ended_at IS NULL OR ich.ended_at > bi.billing_start)
                WHERE bi.billing_end > bi.billing_start
                GROUP BY bi.instance_id, bi.miner_hotkey, bi.chute_id, bi.billing_start, bi.billing_end, bi.bounty, bi.compute_multiplier
            )
            SELECT
                instance_id,
                miner_hotkey,
                chute_id,
                billing_seconds,
                weighted_compute_units,
                fallback_multiplier
            FROM instance_weighted_compute
        """)

        instance_result = await session.execute(current_query)
        instances_data = []
        for row in instance_result:
            instances_data.append(
                {
                    "instance_id": row.instance_id,
                    "miner_hotkey": row.miner_hotkey,
                    "chute_id": row.chute_id,
                    "billing_seconds": float(row.billing_seconds or 0),
                    "current_compute_units": float(row.weighted_compute_units or 0),
                    "current_multiplier": float(row.fallback_multiplier or 1.0),
                }
            )

        active_instances_result = await session.execute(
            text("""
                SELECT
                    i.instance_id,
                    i.chute_id,
                    i.miner_hotkey,
                    i.compute_multiplier,
                    i.activated_at,
                    i.created_at
                FROM instances i
                WHERE i.active = true
                  AND i.verified = true
                  AND i.activated_at IS NOT NULL
            """)
        )
        active_instances = {}
        for row in active_instances_result:
            active_instances[row.instance_id] = {
                "chute_id": row.chute_id,
                "miner_hotkey": row.miner_hotkey,
                "current_multiplier": float(row.compute_multiplier or 1.0),
                "activated_at": row.activated_at,
            }

    now = datetime.now()
    instance_changes = []

    for instance_id, inst_data in active_instances.items():
        chute_id = inst_data["chute_id"]
        target = chute_effective_multipliers.get(chute_id)

        if target is None:
            continue

        hours_since = (
            now - inst_data["activated_at"].replace(tzinfo=None)
        ).total_seconds() / 3600.0
        current = inst_data["current_multiplier"]
        new_value = _calculate_blended_multiplier(current, target, hours_since)

        if new_value is not None and abs(current - new_value) >= 0.001:
            instance_changes.append(
                {
                    "instance_id": instance_id,
                    "chute_id": chute_id,
                    "miner_hotkey": inst_data["miner_hotkey"],
                    "current_multiplier": current,
                    "new_multiplier": new_value,
                    "target_multiplier": target,
                    "hours_since_activation": hours_since,
                }
            )

    new_multipliers = {ic["instance_id"]: ic["new_multiplier"] for ic in instance_changes}

    current_raw = defaultdict(float)
    simulated_raw = defaultdict(float)

    for inst in instances_data:
        hotkey = inst["miner_hotkey"]
        if not hotkey or hotkey not in hot_cold_map or hotkey in blacklisted_hotkeys:
            continue

        instance_id = inst["instance_id"]
        billing_seconds = inst["billing_seconds"]
        current_units = inst["current_compute_units"]

        current_raw[hotkey] += current_units

        # Simulated score: if this instance would get a new multiplier, recalculate
        if instance_id in new_multipliers:
            old_mult = inst["current_multiplier"]
            new_mult = new_multipliers[instance_id]
            if old_mult > 0:
                simulated_units = current_units * (new_mult / old_mult)
            else:
                simulated_units = billing_seconds * new_mult
            simulated_raw[hotkey] += simulated_units
        else:
            simulated_raw[hotkey] += current_units

    for coldkey in set(hot_cold_map.values()):
        if coldkey_counts.get(coldkey, 0) > 1:
            coldkey_hotkeys = [
                hk for hk, ck in hot_cold_map.items() if ck == coldkey and hk in current_raw
            ]
            if len(coldkey_hotkeys) > 1:
                coldkey_hotkeys.sort(key=lambda hk: current_raw.get(hk, 0.0), reverse=True)
                for hk in coldkey_hotkeys[1:]:
                    current_raw.pop(hk, None)
                    simulated_raw.pop(hk, None)

    current_sum = sum(max(0.0, v) for v in current_raw.values())
    simulated_sum = sum(max(0.0, v) for v in simulated_raw.values())

    if current_sum > 0:
        current_scores = {hk: max(0.0, v) / current_sum for hk, v in current_raw.items()}
    else:
        n = max(len(current_raw), 1)
        current_scores = {hk: 1.0 / n for hk in current_raw.keys()}

    if simulated_sum > 0:
        simulated_scores = {hk: max(0.0, v) / simulated_sum for hk, v in simulated_raw.items()}
    else:
        n = max(len(simulated_raw), 1)
        simulated_scores = {hk: 1.0 / n for hk in simulated_raw.keys()}

    miner_changes = []
    all_hotkeys = set(current_scores.keys()) | set(simulated_scores.keys())
    for hk in all_hotkeys:
        curr = current_scores.get(hk, 0.0)
        sim = simulated_scores.get(hk, 0.0)
        curr_raw = current_raw.get(hk, 0.0)
        sim_raw = simulated_raw.get(hk, 0.0)
        if abs(curr - sim) > 0.000001:
            miner_changes.append(
                {
                    "hotkey": hk,
                    "current_score": curr,
                    "simulated_score": sim,
                    "score_change": sim - curr,
                    "score_change_pct": ((sim - curr) / curr * 100) if curr > 0 else 0,
                    "current_raw": curr_raw,
                    "simulated_raw": sim_raw,
                }
            )

    miner_changes.sort(key=lambda x: abs(x["score_change"]), reverse=True)

    return {
        "current_scores": current_scores,
        "simulated_scores": simulated_scores,
        "current_raw": dict(current_raw),
        "simulated_raw": dict(simulated_raw),
        "instance_changes": instance_changes,
        "miner_changes": miner_changes,
    }


@retry_on_db_failure()
async def refresh_instance_compute_multipliers(chute_ids: List[str] = None):
    """
    Refresh compute_multiplier for active instances based on current chute state.

    Uses a gradual adjustment curve to prevent "rug pull" scenarios where miners
    deploy based on a high boost that immediately drops:

    - 0-2 hours after activation: No change (instance keeps original multiplier)
    - 2-8 hours: Ease-in blend toward target (slow at first, accelerates)
      Uses t² curve where t is normalized time in the ramp window
    - 8+ hours: Clamp to target value

    Additionally, instances in the thrash penalty period are skipped entirely.

    Pre-loads all instance values first, calculates new values in Python,
    then issues static UPDATE statements to avoid read-modify-write locks.
    """
    from api.chute.util import calculate_effective_compute_multiplier
    from api.constants import THRASH_WINDOW_HOURS, THRASH_PENALTY_HOURS

    logger.info("Refreshing compute multipliers for active instances...")

    async with get_session() as session:
        await session.execute(text("SET LOCAL statement_timeout = '10s'"))

        # Load chutes (optionally filtered)
        query = select(Chute)
        if chute_ids:
            query = query.where(Chute.chute_id.in_(chute_ids))
        result = await session.execute(query)
        chutes = {c.chute_id: c for c in result.scalars().all()}

        if not chutes:
            logger.info("No chutes to process")
            return

        # Pre-load all active instances for these chutes
        instance_query = select(Instance).where(
            Instance.chute_id.in_(chutes.keys()),
            Instance.active.is_(True),
            Instance.verified.is_(True),
            Instance.activated_at.isnot(None),
        )
        instance_result = await session.execute(instance_query)
        instances = instance_result.scalars().all()

        if not instances:
            logger.info("No active instances to update")
            return

        # Identify instances in thrash penalty period (single query for efficiency)
        thrash_penalty_result = await session.execute(
            text(
                """
                SELECT i.instance_id
                FROM instances i
                WHERE i.chute_id = ANY(:chute_ids)
                  AND i.active = true
                  AND i.verified = true
                  AND i.activated_at IS NOT NULL
                  AND i.activated_at + INTERVAL ':penalty_hours hours' > NOW()
                  AND EXISTS (
                      SELECT 1
                      FROM instance_audit ia
                      WHERE ia.miner_hotkey = i.miner_hotkey
                        AND ia.chute_id = i.chute_id
                        AND ia.activated_at IS NOT NULL
                        AND ia.deleted_at IS NOT NULL
                        AND ia.deleted_at > i.created_at - INTERVAL ':window_hours hours'
                        AND ia.deleted_at <= i.created_at
                        AND ia.valid_termination IS NOT TRUE
                  )
            """.replace(":penalty_hours", str(THRASH_PENALTY_HOURS)).replace(
                    ":window_hours", str(THRASH_WINDOW_HOURS)
                )
            ),
            {"chute_ids": list(chutes.keys())},
        )
        thrash_penalty_instances = {row.instance_id for row in thrash_penalty_result}
        if thrash_penalty_instances:
            logger.info(
                f"Skipping {len(thrash_penalty_instances)} instances in thrash penalty period"
            )

        # Calculate target multipliers for each chute (without bounty)
        chute_targets = {}
        for chute_id, chute in chutes.items():
            effective_data = await calculate_effective_compute_multiplier(
                chute, include_bounty=False
            )
            chute_targets[chute_id] = effective_data["effective_compute_multiplier"]

        # Calculate new values in Python
        now = datetime.now()
        updates = []  # List of (instance_id, new_multiplier)

        for inst in instances:
            # Skip instances in thrash penalty period
            if inst.instance_id in thrash_penalty_instances:
                continue

            target = chute_targets.get(inst.chute_id)
            if target is None:
                continue

            hours_since = (now - inst.activated_at.replace(tzinfo=None)).total_seconds() / 3600.0
            current = inst.compute_multiplier

            new_value = _calculate_blended_multiplier(current, target, hours_since)

            # Skip if no update needed or value unchanged
            if new_value is None:
                continue
            if current is not None and abs(current - new_value) < 0.001:
                continue

            updates.append((inst.instance_id, new_value))

        # Batch update with static values (no read-modify-write)
        if updates:
            for instance_id, new_multiplier in updates:
                await session.execute(
                    text("""
                        UPDATE instances
                        SET compute_multiplier = :multiplier
                        WHERE instance_id = :instance_id
                    """),
                    {"instance_id": instance_id, "multiplier": new_multiplier},
                )
            await session.commit()
            logger.success(f"Updated compute_multiplier for {len(updates)} instances")
        else:
            logger.info("No compute_multiplier updates needed")


async def _log_thrashing_instances():
    """
    Log all instances currently in thrash penalty period for debugging/monitoring.
    Shows current multiplier vs what it would be without thrash penalty.
    """
    from api.constants import THRASH_WINDOW_HOURS, THRASH_PENALTY_HOURS
    from api.bounty.util import get_bounty_info

    async with get_session() as session:
        await session.execute(text("SET LOCAL statement_timeout = '10s'"))

        # Find all instances in thrash penalty period with details about the prior deletion
        result = await session.execute(
            text(
                """
                SELECT
                    i.instance_id,
                    i.chute_id,
                    i.miner_hotkey,
                    i.created_at AS instance_created_at,
                    i.activated_at,
                    i.compute_multiplier,
                    i.bounty,
                    c.name AS chute_name,
                    c.boost AS chute_boost,
                    prior.deleted_at AS prior_deleted_at,
                    prior.activated_at AS prior_activated_at,
                    prior.compute_multiplier AS prior_compute_multiplier,
                    prior.bounty AS prior_bounty,
                    EXTRACT(EPOCH FROM (i.created_at - prior.deleted_at)) / 60 AS minutes_after_deletion,
                    EXTRACT(EPOCH FROM (NOW() - i.activated_at)) / 3600 AS hours_since_activation,
                    EXTRACT(EPOCH FROM (i.activated_at + INTERVAL ':penalty_hours hours' - NOW())) / 60 AS penalty_minutes_remaining
                FROM instances i
                JOIN chutes c ON c.chute_id = i.chute_id
                JOIN LATERAL (
                    SELECT ia.deleted_at, ia.activated_at, ia.compute_multiplier, ia.bounty
                    FROM instance_audit ia
                    WHERE ia.miner_hotkey = i.miner_hotkey
                      AND ia.chute_id = i.chute_id
                      AND ia.activated_at IS NOT NULL
                      AND ia.deleted_at IS NOT NULL
                      AND ia.deleted_at > i.created_at - INTERVAL ':window_hours hours'
                      AND ia.deleted_at <= i.created_at
                      AND ia.valid_termination IS NOT TRUE
                    ORDER BY ia.deleted_at DESC
                    LIMIT 1
                ) prior ON true
                WHERE i.active = true
                  AND i.verified = true
                  AND i.activated_at IS NOT NULL
                  AND i.activated_at + INTERVAL ':penalty_hours hours' > NOW()
                ORDER BY penalty_minutes_remaining ASC
            """.replace(":penalty_hours", str(THRASH_PENALTY_HOURS)).replace(
                    ":window_hours", str(THRASH_WINDOW_HOURS)
                )
            )
        )
        rows = result.fetchall()

        if not rows:
            logger.info("=== THRASHING INSTANCES === None detected")
            return

        logger.info(f"=== THRASHING INSTANCES === {len(rows)} in penalty period")
        logger.info(
            f"{'Instance':<12} {'Chute':<20} {'Miner':<15} "
            f"{'Penalty Left':<12} {'Curr Mult':<10} {'Would Be':<10} {'Blocked':<20}"
        )
        logger.info("-" * 110)

        for row in rows:
            current_mult = row.compute_multiplier or 1.0
            chute_boost = row.chute_boost or 1.0

            # Calculate what multiplier would be if thrash penalty wasn't applied
            # Start with current multiplier
            would_be_mult = current_mult

            # Add chute boost that was blocked
            if chute_boost > 1.0:
                would_be_mult *= chute_boost

            # Check if there's a current bounty that would have applied
            bounty_info = await get_bounty_info(row.chute_id)
            bounty_boost = 1.0
            if bounty_info and not row.bounty:
                # There's a bounty and this instance didn't get it
                bounty_boost = bounty_info.get("boost", 1.0)
                would_be_mult *= bounty_boost

            # Build blocked components string
            blocked = []
            if chute_boost > 1.0:
                blocked.append(f"boost:{chute_boost:.1f}x")
            if bounty_boost > 1.0:
                blocked.append(f"bounty:{bounty_boost:.1f}x")
            blocked_str = ", ".join(blocked) if blocked else "none"

            logger.info(
                f"{row.instance_id[:10]}.. {row.chute_name[:18]:<20} {row.miner_hotkey[:13]}.. "
                f"{row.penalty_minutes_remaining:>8.1f} min "
                f"{current_mult:>9.2f} {would_be_mult:>9.2f} {blocked_str:<20}"
            )

        logger.info("=== END THRASHING INSTANCES ===")


@retry_on_db_failure()
async def manage_rolling_updates(
    db_now: datetime,
    chute_target_counts: Dict[str, int] | None = None,
    chute_rate_limiting: Dict[str, bool] | None = None,
    interval_seconds: int = AUTOSCALER_FULL_INTERVAL_SECONDS,
):
    """
    Manage rolling updates by replacing old-version instances with new-version capacity.
    Enforces a hard 3-hour cap; after that, all remaining old instances are deleted.
    """
    max_duration = timedelta(hours=12)
    async with get_session() as session:
        await session.execute(text("SET LOCAL statement_timeout = '10s'"))
        result = await session.execute(select(RollingUpdate))
        rolling_updates = result.scalars().all()

        for rolling_update in rolling_updates:
            started_at = rolling_update.started_at or db_now
            elapsed = db_now.replace(tzinfo=None) - started_at.replace(tzinfo=None)

            chute = (
                (
                    await session.execute(
                        select(Chute).where(Chute.chute_id == rolling_update.chute_id)
                    )
                )
                .unique()
                .scalar_one_or_none()
            )
            if not chute:
                await session.delete(rolling_update)
                await session.commit()
                continue

            current_version = chute.version
            old_instances = (
                (
                    await session.execute(
                        select(Instance)
                        .where(
                            Instance.chute_id == rolling_update.chute_id,
                            Instance.version != current_version,
                            Instance.active.is_(True),
                            Instance.verified.is_(True),
                        )
                        .order_by(Instance.activated_at.asc().nullsfirst())
                    )
                )
                .unique()
                .scalars()
                .all()
            )

            if not old_instances:
                await session.delete(rolling_update)
                await session.commit()
                continue

            to_delete = []
            if elapsed >= max_duration:
                to_delete = old_instances
                logger.warning(
                    f"Rolling update exceeded 3h cap for {rolling_update.chute_id=}, forcing cleanup"
                )
            else:
                permitted_capacity = None
                if rolling_update.permitted:
                    try:
                        permitted_capacity = sum(rolling_update.permitted.values())
                    except Exception:
                        permitted_capacity = None

                total_active = (
                    await session.execute(
                        select(func.count())
                        .select_from(Instance)
                        .where(
                            Instance.chute_id == rolling_update.chute_id,
                            Instance.active.is_(True),
                            Instance.verified.is_(True),
                        )
                    )
                ).scalar_one()
                target = None
                if chute_target_counts is not None:
                    target = chute_target_counts.get(rolling_update.chute_id)
                if target is not None:
                    remaining_seconds = max(0, int((max_duration - elapsed).total_seconds()))
                    remaining_cycles = max(1, math.ceil(remaining_seconds / interval_seconds))
                    max_per_cycle = max(1, math.ceil(len(old_instances) / remaining_cycles))

                    if permitted_capacity is None:
                        permitted_capacity = total_active
                    baseline_capacity = permitted_capacity
                    if target is not None and target < baseline_capacity:
                        baseline_capacity = target

                    initial_old = max(permitted_capacity or 0, len(old_instances))
                    elapsed_seconds = max(0, int(elapsed.total_seconds()))
                    max_seconds = max(1, int(max_duration.total_seconds()))
                    expected_deleted = int((initial_old * elapsed_seconds) / max_seconds)
                    deleted_so_far = max(0, initial_old - len(old_instances))
                    schedule_needed = max(0, expected_deleted - deleted_so_far)

                    available_capacity = max(0, total_active - baseline_capacity)
                    surplus_based = min(available_capacity, max_per_cycle)

                    deletable = max(schedule_needed, surplus_based)
                    deletable = min(deletable, max_per_cycle, len(old_instances))
                    if deletable > 0:
                        to_delete = old_instances[:deletable]

            if not to_delete:
                continue

            reason = (
                "Rolling update timeout (3h cap)"
                if elapsed >= max_duration
                else "Rolling update replacement"
            )
            instance_ids = [instance.instance_id for instance in to_delete]
            await session.execute(
                text(
                    "UPDATE instance_audit SET deletion_reason = :reason, valid_termination = true WHERE instance_id = ANY(:instance_ids)"
                ),
                {"reason": reason, "instance_ids": instance_ids},
            )

            for instance in to_delete:
                await session.delete(instance)

            if len(old_instances) <= len(to_delete):
                await session.delete(rolling_update)

            await session.commit()
            for instance in to_delete:
                await notify_deleted(instance, message=reason)
                await invalidate_instance_cache(instance.chute_id, instance_id=instance.instance_id)


async def query_prometheus_batch(
    queries: Dict[str, str], prometheus_url: str = PROMETHEUS_URL
) -> Dict[str, Optional[float]]:
    """
    Execute multiple Prometheus queries concurrently.
    Raises exception if any query fails to ensure script safety.
    """
    results = {}

    async def query_single(session: aiohttp.ClientSession, name: str, query: str) -> tuple:
        try:
            async with session.get(
                f"{prometheus_url}/api/v1/query",
                params={"query": query},
                timeout=aiohttp.ClientTimeout(total=30),
            ) as response:
                response.raise_for_status()
                data = await response.json()
                if data["status"] == "success" and data["data"]["result"]:
                    chute_results = {}
                    for result in data["data"]["result"]:
                        chute_id = result["metric"].get("chute_id")
                        value = float(result["value"][1])
                        if chute_id:
                            chute_results[chute_id] = value
                    return (name, chute_results)
                return (name, {})
        except Exception as e:
            logger.error(f"Critical error querying Prometheus for {name}: {e}")
            raise Exception(f"Prometheus query failed for {name}: {e}")

    async with aiohttp.ClientSession() as session:
        tasks = [query_single(session, name, query) for name, query in queries.items()]
        query_results = await asyncio.gather(*tasks)
        for name, result in query_results:
            results[name] = result

    return results


async def get_all_chutes_from_db() -> Set[str]:
    """
    Get all chute IDs from the database.
    """
    async with get_session() as session:
        result = await session.execute(text("SELECT chute_id FROM chutes"))
        return {row.chute_id for row in result}


async def get_all_chute_metrics() -> Dict[str, Dict]:
    """
    Get metrics for all chutes from Prometheus, including zero defaults for chutes without metrics.
    """
    # First, get all chute IDs from the database
    all_db_chute_ids = await get_all_chutes_from_db()
    logger.info(f"Found {len(all_db_chute_ids)} chutes in database")

    queries = {
        # Current utilization
        "utilization_current": "avg by (chute_id) (utilization)",
        # Average utilization over time windows
        "utilization_5m": "avg by (chute_id) (avg_over_time(utilization[5m]))",
        "utilization_15m": "avg by (chute_id) (avg_over_time(utilization[15m]))",
        "utilization_1h": "avg by (chute_id) (avg_over_time(utilization[1h]))",
        # Completed requests
        "completed_5m": "sum by (chute_id) (increase(requests_completed_total[5m]))",
        "completed_15m": "sum by (chute_id) (increase(requests_completed_total[15m]))",
        "completed_1h": "sum by (chute_id) (increase(requests_completed_total[1h]))",
        # Rate limited requests
        "rate_limited_5m": "sum by (chute_id) (increase(requests_rate_limited_total[5m]))",
        "rate_limited_15m": "sum by (chute_id) (increase(requests_rate_limited_total[15m]))",
        "rate_limited_1h": "sum by (chute_id) (increase(requests_rate_limited_total[1h]))",
    }

    try:
        results = await query_prometheus_batch(queries)
    except Exception as e:
        logger.error(f"Failed to query Prometheus, aborting autoscale: {e}")
        raise

    # Initialize metrics for all chutes with zero defaults
    chute_metrics = {}
    for chute_id in all_db_chute_ids:
        chute_metrics[chute_id] = {
            "utilization": {"current": 0.0, "5m": 0.0, "15m": 0.0, "1h": 0.0},
            "completed_requests": {"5m": 0.0, "15m": 0.0, "1h": 0.0},
            "rate_limited_requests": {"5m": 0.0, "15m": 0.0, "1h": 0.0},
            "total_requests": {"5m": 0.0, "15m": 0.0, "1h": 0.0},
            "rate_limit_ratio": {"5m": 0.0, "15m": 0.0, "1h": 0.0},
        }

    # Process Prometheus results and update metrics where data exists
    prometheus_chute_ids = set()
    for metric_name, chute_values in results.items():
        for chute_id, value in chute_values.items():
            prometheus_chute_ids.add(chute_id)
            if chute_id in chute_metrics:  # Only update if chute exists in DB
                if metric_name.startswith("utilization_"):
                    window = metric_name.replace("utilization_", "")
                    chute_metrics[chute_id]["utilization"][window] = value
                elif metric_name.startswith("completed_"):
                    window = metric_name.replace("completed_", "")
                    chute_metrics[chute_id]["completed_requests"][window] = value
                elif metric_name.startswith("rate_limited_"):
                    window = metric_name.replace("rate_limited_", "")
                    chute_metrics[chute_id]["rate_limited_requests"][window] = value

    # Calculate derived metrics
    for chute_id in chute_metrics:
        metrics = chute_metrics[chute_id]
        for window in ["5m", "15m", "1h"]:
            completed = metrics["completed_requests"].get(window, 0) or 0
            rate_limited = metrics["rate_limited_requests"].get(window, 0) or 0
            total = completed + rate_limited
            metrics["total_requests"][window] = total
            if total > 0:
                metrics["rate_limit_ratio"][window] = rate_limited / total
            else:
                metrics["rate_limit_ratio"][window] = 0.0

    # Log information about chutes without metrics
    chutes_without_metrics = all_db_chute_ids - prometheus_chute_ids
    if chutes_without_metrics:
        logger.info(
            f"Found {len(chutes_without_metrics)} chutes in DB without Prometheus metrics (set to zero defaults)"
        )

    return chute_metrics


@retry_on_db_failure()
async def update_chute_boosts(chute_boosts: Dict[str, float]):
    """
    Update the boost column for all chutes based on urgency-calculated values.
    """
    if not chute_boosts:
        return

    async with get_session() as session:
        await session.execute(text("SET LOCAL statement_timeout = '5s'"))
        # Batch update all boosts
        for chute_id, boost in chute_boosts.items():
            await session.execute(
                text("UPDATE chutes SET boost = :boost WHERE chute_id = :chute_id"),
                {"chute_id": chute_id, "boost": boost},
            )
        await session.commit()
        logger.info(f"Updated boost values for {len(chute_boosts)} chutes")


@retry_on_db_failure()
async def log_capacity_metrics(
    chute_metrics: Dict[str, Dict],
    chute_actions: Dict[str, str],
    chute_target_counts: Dict[str, int],
    chute_effective_multipliers: Dict[str, float] = None,
):
    """
    Log all chute metrics to the capacity_log table.
    """
    if chute_effective_multipliers is None:
        chute_effective_multipliers = {}
    async with get_session() as session:
        await session.execute(text("SET LOCAL statement_timeout = '5s'"))
        instance_counts = {}
        result = await session.execute(
            text("""
                SELECT chute_id, COUNT(*) as count
                FROM instances
                WHERE verified = true AND active = true
                GROUP BY chute_id
            """)
        )
        for row in result:
            instance_counts[row.chute_id] = row.count

        # Track in the database.
        logged_count = 0
        for chute_id, metrics in chute_metrics.items():
            capacity_log = CapacityLog(
                timestamp=func.now(),
                chute_id=chute_id,
                utilization_current=metrics["utilization"].get("current"),
                utilization_5m=metrics["utilization"].get("5m"),
                utilization_15m=metrics["utilization"].get("15m"),
                utilization_1h=metrics["utilization"].get("1h"),
                rate_limit_ratio_5m=metrics["rate_limit_ratio"].get("5m"),
                rate_limit_ratio_15m=metrics["rate_limit_ratio"].get("15m"),
                rate_limit_ratio_1h=metrics["rate_limit_ratio"].get("1h"),
                total_requests_5m=metrics["total_requests"].get("5m"),
                total_requests_15m=metrics["total_requests"].get("15m"),
                total_requests_1h=metrics["total_requests"].get("1h"),
                completed_requests_5m=metrics["completed_requests"].get("5m"),
                completed_requests_15m=metrics["completed_requests"].get("15m"),
                completed_requests_1h=metrics["completed_requests"].get("1h"),
                rate_limited_requests_5m=metrics["rate_limited_requests"].get("5m"),
                rate_limited_requests_15m=metrics["rate_limited_requests"].get("15m"),
                rate_limited_requests_1h=metrics["rate_limited_requests"].get("1h"),
                instance_count=instance_counts.get(chute_id, 0),
                action_taken=chute_actions.get(chute_id, "no_action"),
                target_count=chute_target_counts.get(chute_id, UNDERUTILIZED_CAP),
                effective_multiplier=chute_effective_multipliers.get(chute_id),
            )
            session.add(capacity_log)
            logged_count += 1

        if logged_count:
            await session.commit()
            logger.info(f"Logged capacity metrics for {logged_count} chutes")


async def perform_autoscale(
    dry_run: bool = False,
    soft_mode: bool = False,
    dry_run_csv: str = None,
    refresh_multipliers: bool = False,
    simulate_scores: bool = False,
    show_thrashing: bool = False,
):
    """
    Gather utilization data and make decisions on scaling up/down (or nothing).

    Modes:
    - dry_run: Logging only. No Redis writes, no DB writes, no instance changes.
    - soft_mode: Updates Redis targets, compute multipliers, boosts, rolling updates,
                 and logs to capacity_log, but skips all scale-downs.
                 Exits quietly if another autoscaler is running.
    - (default): Full mode - does everything including scale-downs.

    Args:
        dry_run_csv: Path to export CSV with all chute data (only in dry-run mode)
        refresh_multipliers: If True, refresh instance compute_multipliers. Should be
                             run hourly (before :05 when validators snapshot) rather
                             than every autoscaler run.
        simulate_scores: If True (requires dry_run), simulate what miner scores would
                        be if the updated compute multipliers were applied.
        show_thrashing: If True, show instances currently in thrash penalty period.
    """
    try:
        async with autoscaler_lock(soft_mode=soft_mode, skip_lock=dry_run):
            await _perform_autoscale_impl(
                dry_run,
                soft_mode,
                dry_run_csv,
                refresh_multipliers,
                simulate_scores,
                show_thrashing,
            )
    except LockNotAcquired:
        # Soft mode couldn't acquire lock, exit quietly
        return


async def _perform_autoscale_impl(
    dry_run: bool = False,
    soft_mode: bool = False,
    dry_run_csv: str = None,
    refresh_multipliers: bool = False,
    simulate_scores: bool = False,
    show_thrashing: bool = False,
):
    """Internal implementation of autoscale logic (called within lock)."""
    if dry_run and soft_mode:
        logger.warning("Both --dry-run and --soft specified; --dry-run takes precedence")
        soft_mode = False

    mode_str = "DRY-RUN" if dry_run else ("SOFT" if soft_mode else "FULL")
    logger.info(f"Starting autoscaler in {mode_str} mode...")

    if not dry_run:
        logger.info("Performing instance cleanup...")
        await instance_cleanup()

    # Reconcile connection counts from instance ground truth before scaling decisions.
    logger.info("Reconciling connection counts...")
    try:
        await asyncio.wait_for(reconcile_connection_counts(), timeout=30.0)
    except Exception as exc:
        logger.warning(f"Connection count reconciliation failed: {exc}")

    logger.info("Fetching metrics from Prometheus and database...")
    chute_metrics = await get_all_chute_metrics()

    # Safety check - ensure we have enough data
    if len(chute_metrics) < MIN_CHUTES_FOR_SCALING:
        logger.warning(
            f"Only found {len(chute_metrics)} chutes total, need at least {MIN_CHUTES_FOR_SCALING}. Aborting."
        )
        return
    logger.info(f"Processing metrics for {len(chute_metrics)} chutes")

    # Fetch detailed chute info and ALL active instances (with nodes)
    chute_info_map = {}
    all_active_instances = []
    db_now = datetime.now(timezone.utc)

    for attempt in range(3):
        try:
            async with get_session() as session:
                await session.execute(text("SET LOCAL statement_timeout = '10s'"))
                chute_result = await session.execute(
                    text("""
                        SELECT
                            c.chute_id,
                            c.public,
                            c.name,
                            c.user_id,
                            c.created_at,
                            c.concurrency,
                            c.node_selector,
                            c.tee,
                            c.version,
                            c.boost,
                            c.disabled,
                            MAX(COALESCE(ucb.effective_balance, 0)) AS user_balance,
                            c.max_instances,
                            c.scaling_threshold,
                            COALESCE(MAX(cmb.boost), 1.0) AS manual_boost,
                            NOW() - c.created_at <= INTERVAL '3 hours' AS new_chute,
                            COUNT(DISTINCT CASE WHEN i.active = true AND i.verified = true THEN i.instance_id END) AS instance_count,
                            COUNT(DISTINCT CASE WHEN i.verified = false AND i.created_at > NOW() - INTERVAL '30 minutes' THEN i.instance_id END) AS pending_instance_count,
                            EXISTS(SELECT 1 FROM rolling_updates ru WHERE ru.chute_id = c.chute_id) AS has_rolling_update,
                            NOW() AS db_now
                        FROM chutes c
                        LEFT JOIN instances i ON c.chute_id = i.chute_id
                        LEFT JOIN user_current_balance ucb on ucb.user_id = c.user_id
                        LEFT JOIN chute_manual_boosts cmb on cmb.chute_id = c.chute_id
                        WHERE c.jobs IS NULL
                              OR c.jobs = '[]'::jsonb
                              OR c.jobs = '{}'::jsonb
                        GROUP BY c.chute_id
                    """)
                )
                chute_info_map = {row.chute_id: row for row in chute_result}
                if chute_info_map:
                    db_now = next(iter(chute_info_map.values())).db_now

                instance_result = await session.execute(
                    select(Instance)
                    .where(Instance.active.is_(True), Instance.verified.is_(True))
                    .options(selectinload(Instance.nodes))
                )
                all_active_instances = instance_result.scalars().all()
                break
        except OperationalError as e:
            if attempt == 2:
                logger.error(f"Failed to fetch system state after 3 attempts: {e}")
                raise
            logger.warning(
                f"Failed to fetch system state (attempt {attempt + 1}/3): {e}. Retrying..."
            )
            await asyncio.sleep(1)

    instances_by_chute = defaultdict(list)
    for inst in all_active_instances:
        instances_by_chute[inst.chute_id].append(inst)

    # Fetch previous smoothed metrics from Redis for EMA calculation
    all_chute_ids = list(chute_metrics.keys())
    previous_smoothed = {}
    if not dry_run:
        previous_smoothed = await get_smoothed_metrics(all_chute_ids)

    # Load sponsored chute IDs (no demand boost for sponsored chutes)
    from api.invocation.util import get_all_sponsored_chute_ids

    sponsored_chute_ids = await get_all_sponsored_chute_ids()

    # 1. Initialize Contexts and Calculate Urgency
    contexts: Dict[str, AutoScaleContext] = {}
    starving_chutes: List[AutoScaleContext] = []
    # Track filtered chutes for accurate capacity logging
    filtered_chutes: Dict[str, int] = {}
    # Collect new smoothed metrics to save
    new_smoothed_metrics: Dict[str, Dict[str, float]] = {}
    # Cache chutes_user_id for preemptible check
    chutes_uid = await chutes_user_id()

    for chute_id, metrics in chute_metrics.items():
        info = chute_info_map.get(chute_id)
        if not info:
            # Chute filtered out by query (e.g., has jobs) - write safe target to avoid stale Redis
            # Use current instance count from instances_by_chute, or 0 if none
            current_instances = len(instances_by_chute.get(chute_id, []))
            if not dry_run:
                await settings.redis_client.set(f"scale:{chute_id}", current_instances, ex=3700)
            filtered_chutes[chute_id] = current_instances
            continue

        # Parse node selector to understand hardware needs
        try:
            ns = NodeSelector(**info.node_selector)
            supported_gpus = set(ns.supported_gpus)
            gpu_count = ns.gpu_count
        except Exception:
            logger.warning(f"Failed to parse node selector for {chute_id}")
            supported_gpus = set()
            gpu_count = None

        ctx = AutoScaleContext(
            chute_id,
            metrics,
            info,
            supported_gpus,
            instances_by_chute[chute_id],
            db_now,
            gpu_count,
        )
        contexts[chute_id] = ctx

        # Calculate Urgency Score (raw/instantaneous)
        # The goal is to reflect "how urgently do we need capacity RIGHT NOW"
        #
        # Components:
        # 1. Recency-weighted rate limiting: Recent RL matters much more than historical
        #    - 5m window: full weight (this is happening now)
        #    - 15m window: 30% weight (recent but may have resolved)
        #    - 1h window: 5% weight (mostly historical, slight memory)
        # 2. Volume significance: Rate limiting must represent meaningful demand
        #    - Uses dynamic thresholds based on actual throughput, not magic numbers
        #    - Prevents gaming by spamming a few requests to trigger RL ratios
        #    - Only applies to preemptible (per-request billed) chutes
        # 3. Threshold-relative utilization: How far over YOUR threshold
        #    - A chute at 9% util with threshold=1% is 9x over, very urgent
        #    - A chute at 60% util with threshold=60% is at threshold, moderate
        # 4. Capacity pressure multiplier: Low utilization dampens urgency
        #    - If util=6% but historical RL, we have 94% spare capacity, less urgent
        #
        # This prevents high boosts for chutes that HAD problems but now have spare capacity,
        # and prevents gaming by triggering RL with tiny request volumes.

        # Determine if this chute is preemptible (per-request billing) vs hourly billing
        # Preemptible: public OR legacy_private_billing OR chutes_user_id (semi-private)
        # Non-preemptible (hourly): private AND NOT legacy AND NOT chutes_user_id
        is_preemptible = (
            ctx.public
            or has_legacy_private_billing(ctx.info)
            or (ctx.info and ctx.info.user_id == chutes_uid)
        )

        # Volume significance check for rate limiting (dynamic, no magic numbers)
        # Rate limiting is "significant" if it represents meaningful unmet demand:
        # 1. RL count should be at least 10% of completed count (real additional demand)
        # 2. OR RL count should fill at least one "slot" per instance (concurrency worth)
        # Non-preemptible (hourly billed) chutes skip this - user pays for capacity anyway.
        def is_rl_significant(rl_count: float, completed_count: float) -> bool:
            if not is_preemptible:
                # Hourly billed: user pays regardless, any RL is significant to them
                return rl_count > 0
            if rl_count <= 0:
                return False
            # Minimum bar: at least 10% of completed requests were denied
            # This filters out noise like 2 RL out of 1000 completed
            if completed_count > 0 and rl_count >= completed_count * 0.1:
                return True
            # Alternative: RL count fills at least one slot per instance
            # This catches cases where completed is low but RL is meaningful
            slots_per_instance = max(1, ctx.concurrency)
            instances = max(1, ctx.current_count)
            if rl_count >= slots_per_instance * instances:
                return True
            return False

        # Apply significance filter to each time window
        rl_5m_significant = (
            ctx.rate_limit_5m
            if is_rl_significant(ctx.rate_limited_count_5m, ctx.completed_5m)
            else 0
        )
        rl_15m_significant = (
            ctx.rate_limit_15m
            if is_rl_significant(ctx.rate_limited_count_15m, ctx.completed_15m)
            else 0
        )
        # 1h: use 15m counts as proxy (don't have 1h counts)
        rl_1h_significant = (
            ctx.rate_limit_1h
            if is_rl_significant(ctx.rate_limited_count_15m, ctx.completed_15m)
            else 0
        )

        # Recency-weighted rate limiting (0-1 scale)
        rl_weighted = (
            rl_5m_significant * 1.0  # Full weight for current
            + rl_15m_significant * 0.3  # Partial weight for recent
            + rl_1h_significant * 0.05  # Minimal weight for historical
        )
        # Normalize (theoretical max is 1.35 if all windows at 100%)
        rl_weighted = min(1.0, rl_weighted / 1.35)

        # Threshold-relative utilization score
        # How far over threshold as a ratio (1.0 = at threshold, 2.0 = 2x threshold)
        if ctx.threshold > 0:
            threshold_ratio = ctx.utilization_basis / ctx.threshold
        else:
            threshold_ratio = ctx.utilization_basis * 100  # Effectively infinite if threshold=0

        # Capacity pressure: how much of capacity is actually being used
        # This dampens urgency when we have lots of spare capacity
        # Range: 0.2 (at 0% util) to 1.0 (at 100% util)
        capacity_pressure = 0.2 + (ctx.utilization_basis * 0.8)

        # Combine components:
        # - Rate limiting is the primary signal (scaled 0-500)
        # - Threshold-relative util adds urgency for overloaded chutes (scaled 0-100)
        # - Capacity pressure dampens everything if we have spare capacity
        rl_score = rl_weighted * 500
        util_score = min(100, (threshold_ratio - 1.0) * 50) if threshold_ratio > 1.0 else 0

        ctx.urgency_score = (rl_score + util_score) * capacity_pressure

        # Calculate smoothed values using EMA
        # These provide stability for boost calculations and scale-down decisions
        prev = previous_smoothed.get(chute_id)
        prev_urgency = prev["urgency"] if prev else None
        prev_util = prev["util"] if prev else None

        ctx.smoothed_urgency = calculate_ema(ctx.urgency_score, prev_urgency, EMA_ALPHA_URGENCY)
        ctx.smoothed_util = calculate_ema(ctx.utilization_basis, prev_util, EMA_ALPHA_UTIL)

        # Store for saving to Redis later (not in dry_run)
        new_smoothed_metrics[chute_id] = {
            "urgency": ctx.smoothed_urgency,
            "util": ctx.smoothed_util,
        }

        # Identify Starving Chutes (High Demand) - eligible for forced donations from other chutes
        # Use RAW metrics here for fast reaction to demand spikes
        # "Starving" requires severe capacity pressure, not just being above scale-up threshold:
        # - Very high utilization (at or above starving_threshold) where rate limiting is imminent, OR
        # - Sustained rate limiting: both 5m and 15m windows show >5% rate limiting,
        #   AND 5m rate > 15m rate (indicates ongoing/worsening pressure, not just historical)
        # Chutes just above the scale-up threshold can still scale up via normal means,
        # but won't force donations from other chutes in the stable zone.
        is_severely_loaded = ctx.utilization_basis >= ctx.starving_threshold
        # Require sustained rate limiting: both windows > 5%, and 5m >= 15m (not improving)
        is_sustained_rate_limiting = (
            ctx.rate_limit_5m >= 0.05
            and ctx.rate_limit_15m >= 0.05
            and ctx.rate_limit_5m >= ctx.rate_limit_15m
        )
        if is_severely_loaded or is_sustained_rate_limiting:
            ctx.is_starving = True
            starving_chutes.append(ctx)

        # Identify Potential Donors (for forced donations during arbitration)
        # Private chutes are not donors unless they belong to chutes_user_id (semi-private).
        # Chutes in LIMIT_OVERRIDES should never be preempted.
        # Use SMOOTHED utilization for donor determination to prevent flip-flopping
        allow_donor = ctx.public or (ctx.info and ctx.info.user_id == await chutes_user_id())
        if (
            not ctx.any_rate_limiting
            and ctx.current_count > 0
            and allow_donor
            and ctx.chute_id not in LIMIT_OVERRIDES
        ):
            # Voluntary scale-down candidate: below scale_down_threshold
            # These will scale down on their own (gated by moving average)
            # Use smoothed_util to prevent borderline chutes from flip-flopping
            if ctx.smoothed_util < ctx.scale_down_threshold:
                ctx.is_critical_donor = True
                ctx.is_donor = True
            # Forced donation candidate: in stable zone (below threshold but above scale_down_threshold)
            # These won't scale down voluntarily but can be forced to donate when others are starving
            elif ctx.smoothed_util < ctx.threshold:
                ctx.is_donor = True

    # If any starving chutes exist, block non-starving scale-ups that are GPU-compatible.
    if starving_chutes:
        for ctx in contexts.values():
            if ctx.is_starving:
                continue
            if not ctx.public:
                continue
            for starving_ctx in starving_chutes:
                if starving_ctx.chute_id == ctx.chute_id:
                    continue
                if ctx.tee != starving_ctx.tee:
                    continue
                if (
                    ctx.gpu_count is not None
                    and starving_ctx.gpu_count is not None
                    and ctx.gpu_count != starving_ctx.gpu_count
                ):
                    continue
                if not ctx.supported_gpus or not starving_ctx.supported_gpus:
                    ctx.blocked_by_starving = True
                elif ctx.supported_gpus & starving_ctx.supported_gpus:
                    ctx.blocked_by_starving = True
                if ctx.blocked_by_starving:
                    break

    bounty_infos = await get_bounty_infos(list(contexts.keys()))
    for ctx in contexts.values():
        if ctx.chute_id in bounty_infos:
            ctx.has_bounty = True
            # Delete bounty if chute has active/hot instances (active=true AND verified=true)
            if ctx.current_count > 0 and not dry_run:
                if await delete_bounty(ctx.chute_id):
                    logger.info(
                        f"Deleted bounty for {ctx.chute_id} - chute has {ctx.current_count} active instances"
                    )
                    ctx.has_bounty = False

    # 2. Local Decision Making (Ideal World)
    for ctx in contexts.values():
        await calculate_local_decision(ctx)

    # 3. Global Arbitration (The Real World Matchmaking)
    # Force donors to give up capacity for starving chutes, but conservatively:
    # - Only force a portion of needed instances (traffic may subside)
    # - Max 1 instance per donor per interval (prevents destabilizing any single chute)
    # - Don't take from chutes that were recently starving (prevents A->B->A cycles)
    MAX_FORCED_DONATIONS_PER_CHUTE = 3  # Reduced: be conservative, traffic may subside
    MAX_FORCED_DONATIONS_TOTAL = 10  # Reduced total cap
    # Each donor can only give 1 instance per full interval (20 min)
    MAX_DONATIONS_PER_DONOR = 1

    # Get set of chutes that were starving in the past 90 minutes
    # These cannot be donors to prevent A->B->A donation thrashing
    recently_starving = await get_recently_starving_chutes(list(contexts.keys()))

    # Mark current starving chutes in Redis (for future intervals)
    if not dry_run:
        for ctx in starving_chutes:
            await mark_chute_as_starving(ctx.chute_id)

    total_forced = 0
    if starving_chutes:
        starving_chutes.sort(key=lambda x: x.urgency_score, reverse=True)

        for hungry_ctx in starving_chutes:
            if total_forced >= MAX_FORCED_DONATIONS_TOTAL:
                break

            # How many instances does this chute need?
            # Subtract pending instances (already spinning up) from the demand
            instances_needed = hungry_ctx.upscale_amount - hungry_ctx.pending_instance_count
            if instances_needed <= 0:
                continue

            # Be conservative: only force a portion of needed instances
            # Traffic may subside soon, and we can always force more next interval
            # Force at most ceil(needed / 3) instances, minimum 1
            conservative_needed = max(1, math.ceil(instances_needed / 3))
            instances_needed = min(instances_needed, conservative_needed)

            # Match strictly by TEE status and actual available hardware
            needed_gpus = hungry_ctx.supported_gpus

            # Build list of all eligible donors with matching hardware
            eligible_donors = []
            for donor in contexts.values():
                # Skip ineligible donors
                if (
                    donor.chute_id == hungry_ctx.chute_id
                    or not donor.is_donor
                    or donor.tee != hungry_ctx.tee
                ):
                    continue
                # Skip chutes that were recently starving (prevents A->B->A cycles)
                if donor.chute_id in recently_starving:
                    continue
                # Skip donors that have already donated this interval
                if donor.downscale_amount >= MAX_DONATIONS_PER_DONOR:
                    continue
                # Donor must have established instances (1+ hour old) to donate
                if donor.established_instance_count == 0:
                    continue
                # Donor must have capacity above minimum after any pending downscales
                # Use chute-specific failsafe minimum, not just global UNDERUTILIZED_CAP
                donor_failsafe = FAILSAFE.get(donor.chute_id, UNDERUTILIZED_CAP)
                remaining_capacity = donor.current_count - donor.downscale_amount
                if remaining_capacity <= donor_failsafe:
                    continue

                # Check if donor actually has hardware the starving chute can use
                available_matching_gpus = set(donor.hardware_map.keys()) & needed_gpus
                if available_matching_gpus:
                    # Each donor can only give 1 instance per interval (MAX_DONATIONS_PER_DONOR)
                    # This prevents destabilizing any single donor
                    can_give = MAX_DONATIONS_PER_DONOR - donor.downscale_amount

                    # For non-critical donors (in stable zone), check if donation would cause thrashing
                    # Account for pending instances in projected utilization
                    if can_give > 0 and not donor.is_critical_donor:
                        # Calculate what utilization would be after donation
                        # Include pending instances as they'll soon be active
                        effective_donor_count = donor.current_count + donor.pending_instance_count
                        new_count = effective_donor_count - donor.downscale_amount - can_give
                        if new_count > 0:
                            # Project utilization based on current load spread across new capacity
                            projected_util = (
                                donor.utilization_basis * donor.current_count
                            ) / new_count
                            # Don't donate if it would push donor above scale-up threshold
                            if projected_util >= donor.threshold:
                                can_give = 0

                    if can_give > 0:
                        eligible_donors.append((donor, available_matching_gpus, can_give))

            if not eligible_donors:
                continue

            # Prioritize critical donors (already scaling down voluntarily) over stable-zone donors
            # This prevents forcing capacity away from stable chutes when there are chutes
            # that would scale down anyway
            critical_donors = [(d, g, c) for d, g, c in eligible_donors if d.is_critical_donor]
            stable_donors = [(d, g, c) for d, g, c in eligible_donors if not d.is_critical_donor]
            random.shuffle(critical_donors)
            random.shuffle(stable_donors)
            eligible_donors = critical_donors + stable_donors
            donations_for_this_chute = 0
            max_for_this_chute = min(
                instances_needed,
                MAX_FORCED_DONATIONS_PER_CHUTE,
                MAX_FORCED_DONATIONS_TOTAL - total_forced,
            )

            for donor, available_matching_gpus, can_give in eligible_donors:
                if donations_for_this_chute >= max_for_this_chute:
                    break

                # Take up to what this donor can give, but not more than we need
                # With MAX_DONATIONS_PER_DONOR=1, this will always be 1
                take_from_donor = min(
                    can_give,
                    max_for_this_chute - donations_for_this_chute,
                )
                if take_from_donor <= 0:
                    continue

                chosen_gpu = random.choice(list(available_matching_gpus))
                donor.downscale_amount += take_from_donor
                donor.target_count = donor.current_count - donor.downscale_amount
                donor.action = "forced_downscale"
                donor.preferred_downscale_gpus.add(chosen_gpu)

                donations_for_this_chute += take_from_donor
                total_forced += take_from_donor

                logger.info(
                    f"Arbitration: {donor.chute_id} giving up {take_from_donor}x {chosen_gpu} "
                    f"for {hungry_ctx.chute_id} (Urgency={hungry_ctx.urgency_score:.1f})"
                )

            if donations_for_this_chute > 0:
                logger.info(
                    f"Arbitration summary: {hungry_ctx.chute_id} received {donations_for_this_chute} "
                    f"forced donations (needed {instances_needed})"
                )

    # 3b. Priority Locking & Boost Calculation
    # Boost multipliers for chutes wanting to scale up.
    # Base boost is calculated from SMOOTHED urgency for stability.
    #
    # With the new urgency formula:
    # - Max theoretical urgency: (500 rl + 100 util) * 1.0 pressure = 600
    # - High urgency (active RL + high util): ~300-400
    # - Moderate urgency (some RL or high util): ~100-200
    # - Low urgency (historical RL, low util): ~20-50
    #
    # We set URGENCY_MAX_FOR_BOOST at 300 so that genuinely urgent chutes
    # hit max boost, while chutes with only historical issues get modest boost.
    URGENCY_MAX_FOR_BOOST = 300
    URGENCY_BOOST_MIN = 1.0
    URGENCY_BOOST_MAX = 1.5
    RELATIVE_ADJUSTMENT_MAX = 0.2  # ±20% adjustment based on relative urgency

    # Collect SMOOTHED urgency scores for chutes wanting to scale up
    # Using smoothed values prevents boost from oscillating between runs
    scaling_chutes = [ctx for ctx in contexts.values() if ctx.upscale_amount > 0]
    if scaling_chutes:
        smoothed_scores = [ctx.smoothed_urgency for ctx in scaling_chutes]
        avg_urgency = sum(smoothed_scores) / len(smoothed_scores)
        max_urgency = max(smoothed_scores)
    else:
        avg_urgency = 0
        max_urgency = 0

    for ctx in contexts.values():
        if ctx.chute_id in sponsored_chute_ids:
            ctx.boost = 1.0
        elif not ctx.public and ctx.current_count >= ctx.max_instances:
            ctx.boost = 1.0
        elif ctx.upscale_amount > 0:
            # Base boost from SMOOTHED individual urgency (stable across runs)
            normalized_urgency = min(ctx.smoothed_urgency / URGENCY_MAX_FOR_BOOST, 1.0)
            base_boost = URGENCY_BOOST_MIN + (
                normalized_urgency * (URGENCY_BOOST_MAX - URGENCY_BOOST_MIN)
            )

            # Relative adjustment based on position vs other scaling chutes.
            # Above average: bonus up to +20%
            # Below average: reduction up to -20%, but final boost never below 1.0
            if max_urgency > 0:
                # Normalize to [-1, 1] range
                spread = max(max_urgency, 1)
                relative_position = (ctx.smoothed_urgency - avg_urgency) / spread
                relative_position = max(-1.0, min(1.0, relative_position))
                relative_factor = 1.0 + (relative_position * RELATIVE_ADJUSTMENT_MAX)
            else:
                relative_factor = 1.0

            # Sustained urgency factor: dampen boost for sudden spikes
            # If raw urgency is much higher than smoothed, this is a new spike - be conservative.
            # If raw ≈ smoothed, urgency has been sustained - reward more.
            # This prevents gaming by burst-spam → deploy → collect boost → repeat.
            #
            # Formula: sustainability = smoothed / max(raw, smoothed)
            # - If smoothed == raw: sustainability = 1.0 (fully sustained)
            # - If smoothed << raw: sustainability approaches 0 (sudden spike)
            # We then blend: effective_boost = 1.0 + (base_boost - 1.0) * sustainability
            # This keeps minimum boost at 1.0 but scales the bonus by sustainability.
            if ctx.urgency_score > 0:
                sustainability = ctx.smoothed_urgency / max(ctx.urgency_score, ctx.smoothed_urgency)
                # Apply a floor so sustained urgency still gets decent boost even if slightly declining
                sustainability = max(0.3, sustainability)
            else:
                sustainability = 1.0 if ctx.smoothed_urgency == 0 else 0.3

            adjusted_boost = base_boost * relative_factor
            # Scale the boost bonus (amount above 1.0) by sustainability
            ctx.boost = max(1.0, 1.0 + (adjusted_boost - 1.0) * sustainability)
        else:
            ctx.boost = 1.0

    # Calculate effective compute multiplier for each chute (for CSV export and logging)
    # This mirrors the logic in api/chute/util.py:calculate_effective_compute_multiplier
    # but uses ctx.boost (which may not be saved to DB yet in dry-run mode)
    from api.constants import PRIVATE_INSTANCE_BONUS, INTEGRATED_SUBNET_BONUS, TEE_BONUS
    from api.chute.util import INTEGRATED_SUBNETS

    for ctx in contexts.values():
        try:
            ns = NodeSelector(**ctx.info.node_selector)
            base_mult = ns.compute_multiplier
        except Exception:
            base_mult = 1.0

        ctx.base_multiplier = base_mult
        total = base_mult

        # Private/integrated bonus
        if not ctx.public and ctx.info:
            is_integrated = any(
                config["model_substring"] in ctx.info.name.lower()
                for config in INTEGRATED_SUBNETS.values()
            )
            if is_integrated:
                total *= INTEGRATED_SUBNET_BONUS
            else:
                total *= PRIVATE_INSTANCE_BONUS

        # Urgency boost
        if ctx.boost > 1.0:
            total *= ctx.boost

        # Manual boost
        if ctx.manual_boost and ctx.manual_boost > 0:
            total *= min(ctx.manual_boost, 20.0)

        # Bounty boost
        bounty_info = bounty_infos.get(ctx.chute_id)
        if bounty_info and bounty_info.get("boost", 1.0) > 1.0:
            total *= bounty_info["boost"]

        # TEE bonus
        if ctx.tee:
            total *= TEE_BONUS

        ctx.effective_multiplier = total
        ctx.cm_delta_ratio = total / base_mult if base_mult > 0 else 1.0

    # Kinda hacky, because it's not actually creating bounties, but we'll
    # send bounty notifications for miners to have instant feedback when
    # there are urgent scaling needs.
    # Only send notifications for chutes with NO active instances (cold start).
    URGENCY_THRESHOLD_FOR_NOTIFICATION = 100
    if not dry_run:
        from api.chute.util import calculate_effective_compute_multiplier

        for ctx in starving_chutes:
            # Never send bounty notifications if chute has active/hot instances
            if ctx.current_count > 0:
                continue

            # Only public chutes or semi-private (chutes user) get notifications
            if not ctx.public and (not ctx.info or ctx.info.user_id != await chutes_user_id()):
                continue

            # Only notify if urgency is high enough and we actually need instances
            if ctx.urgency_score < URGENCY_THRESHOLD_FOR_NOTIFICATION or ctx.upscale_amount <= 0:
                continue

            # Calculate effective multiplier (without bounty since there may not be one)
            effective_data = await calculate_effective_compute_multiplier(
                ctx.info, include_bounty=False
            )
            effective_mult = effective_data["effective_compute_multiplier"]

            # Check if there's an existing bounty to include its boost
            bounty_info = await get_bounty_info(ctx.chute_id)
            bounty_amount = bounty_info["amount"] if bounty_info else 0
            bounty_boost = bounty_info["boost"] if bounty_info else None
            if bounty_boost:
                effective_mult *= bounty_boost

            await send_bounty_notification(
                chute_id=ctx.chute_id,
                bounty=bounty_amount,
                effective_multiplier=effective_mult,
                bounty_boost=bounty_boost,
                urgency="cold",
            )
            logger.info(
                f"Sent scaling notification for {ctx.chute_id}: urgency=cold, "
                f"effective_mult={effective_mult:.1f}x, upscale_amount={ctx.upscale_amount}"
            )

    # 4. Finalize Actions
    chute_actions = {}
    chute_target_counts = {}
    chute_rate_limiting = {}
    chute_boosts = {}
    to_downsize: List[Tuple[str, int, Set[str]]] = []

    for ctx in contexts.values():
        apply_overrides(ctx)

        # For voluntary scale-downs (not forced donations), check moving average permission
        # Skip this check in soft_mode since we won't execute scale-downs anyway
        if ctx.action == "scale_down_candidate" and ctx.downscale_amount > 0 and not soft_mode:
            permitted, reason = await get_scale_down_permission(
                ctx.chute_id, ctx.current_count, ctx.target_count, ctx.rate_limit_basis
            )
            if not permitted:
                # Moving average check blocked voluntary scale-down
                logger.info(
                    f"Scale down blocked: {ctx.chute_id} - {reason}, "
                    f"keeping at {ctx.current_count} instances"
                )
                ctx.target_count = ctx.current_count
                ctx.downscale_amount = 0
                ctx.action = "scale_down_blocked"

        # In soft_mode, clear all scale-down decisions (we still track them for logging)
        if soft_mode and ctx.downscale_amount > 0:
            original_action = ctx.action
            ctx.action = f"{original_action}_skipped"
            ctx.downscale_amount = 0
            # Keep target_count at current to avoid Redis showing lower targets
            ctx.target_count = ctx.current_count

        chute_actions[ctx.chute_id] = ctx.action
        chute_target_counts[ctx.chute_id] = ctx.target_count
        chute_rate_limiting[ctx.chute_id] = ctx.any_rate_limiting
        chute_boosts[ctx.chute_id] = ctx.boost

        # In dry_run, skip Redis writes entirely
        if not dry_run:
            await settings.redis_client.set(f"scale:{ctx.chute_id}", ctx.target_count, ex=3700)

        if ctx.downscale_amount > 0:
            to_downsize.append((ctx.chute_id, ctx.downscale_amount, ctx.preferred_downscale_gpus))

    if dry_run:
        logger.warning("DRY RUN MODE: Skipping all writes (Redis, DB, instance changes)")

        # Categorize actions
        scale_ups = [c for c in contexts.values() if "scale_up" in c.action]
        scale_downs = [
            c
            for c in contexts.values()
            if "scale_down" in c.action or "forced_downscale" in c.action
        ]
        no_actions = [c for c in contexts.values() if c.action == "no_action"]

        logger.info("=== DRY RUN SUMMARY ===")
        logger.info(
            f"Scale ups: {len(scale_ups)}, Scale downs: {len(scale_downs)}, No action: {len(no_actions)}"
        )

        # Log scale-up details (soft mode would do these)
        if scale_ups:
            logger.info("--- SCALE UPS (soft mode would execute) ---")
            for ctx in sorted(scale_ups, key=lambda x: x.smoothed_urgency, reverse=True):
                # Show actual delta (target - current), not upscale_amount which may differ
                # due to failsafe minimums being applied after initial calculation
                actual_delta = ctx.target_count - ctx.current_count
                logger.info(
                    f"  {ctx.chute_id} | {ctx.current_count} -> {ctx.target_count} (+{actual_delta}) | "
                    f"util={ctx.utilization_basis:.2f} rl={ctx.rate_limit_basis:.3f} | "
                    f"urgency={ctx.urgency_score:.0f} smoothed={ctx.smoothed_urgency:.0f} boost={ctx.boost:.2f}"
                )

        # Log scale-down details (only full mode would execute)
        if scale_downs:
            logger.info("--- SCALE DOWNS (only full mode would execute) ---")
            for ctx in sorted(scale_downs, key=lambda x: x.downscale_amount, reverse=True):
                logger.info(
                    f"  {ctx.chute_id} | {ctx.current_count} -> {ctx.target_count} (-{ctx.downscale_amount}) | "
                    f"util={ctx.utilization_basis:.2f} smoothed={ctx.smoothed_util:.2f} | action={ctx.action}"
                )

        # Log no-action chutes
        if no_actions:
            logger.info("--- NO ACTION ---")
            for ctx in no_actions:
                logger.info(
                    f"  {ctx.chute_id} | count={ctx.current_count} | "
                    f"util={ctx.utilization_basis:.2f} rl={ctx.rate_limit_basis:.3f}"
                )

        # Log boost distribution
        if chute_boosts:
            boosts = list(chute_boosts.values())
            logger.info(
                f"--- BOOST STATS --- min={min(boosts):.2f} max={max(boosts):.2f} "
                f"avg={sum(boosts) / len(boosts):.2f}"
            )

        # Run score simulation if requested
        simulation_results = None
        if simulate_scores:
            # Build effective multipliers map from contexts
            # These use the updated boost values calculated during this dry-run
            chute_effective_multipliers = {
                ctx.chute_id: ctx.effective_multiplier
                for ctx in contexts.values()
                if ctx.effective_multiplier > 0
            }
            simulation_results = await simulate_miner_scores(chute_effective_multipliers)

            logger.info("=== SCORE SIMULATION ===")
            logger.info(
                f"Instance multiplier changes: {len(simulation_results['instance_changes'])}"
            )
            logger.info(f"Miners with score changes: {len(simulation_results['miner_changes'])}")

            # Log top miner changes (biggest absolute changes first)
            if simulation_results["miner_changes"]:
                logger.info("--- TOP MINER SCORE CHANGES ---")
                logger.info(
                    f"{'Hotkey':<48} {'Current':<10} {'Simulated':<10} {'Change':<12} {'Change %':<10}"
                )
                logger.info("-" * 90)
                for mc in simulation_results["miner_changes"][:20]:
                    logger.info(
                        f"{mc['hotkey']:<48} "
                        f"{mc['current_score']:<10.6f} "
                        f"{mc['simulated_score']:<10.6f} "
                        f"{mc['score_change']:+<12.6f} "
                        f"{mc['score_change_pct']:+.2f}%"
                    )

            # Log instance-level changes for transparency
            if simulation_results["instance_changes"]:
                logger.info(
                    f"--- INSTANCE MULTIPLIER CHANGES (first 20 of {len(simulation_results['instance_changes'])}) ---"
                )
                for ic in simulation_results["instance_changes"][:20]:
                    logger.info(
                        f"  {ic['instance_id'][:8]}... | "
                        f"chute={ic['chute_id'][:8]}... | "
                        f"miner={ic['miner_hotkey'][:12]}... | "
                        f"mult: {ic['current_multiplier']:.2f} -> {ic['new_multiplier']:.2f} "
                        f"(target={ic['target_multiplier']:.2f}, {ic['hours_since_activation']:.1f}h)"
                    )

            logger.info("=== END SCORE SIMULATION ===")

        # Export CSV if requested
        if dry_run_csv:
            import csv

            with open(dry_run_csv, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(
                    [
                        "chute_id",
                        "action",
                        "current_count",
                        "target_count",
                        "upscale_amount",
                        "downscale_amount",
                        "utilization_basis",
                        "smoothed_util",
                        "rate_limit_basis",
                        "urgency_score",
                        "smoothed_urgency",
                        "boost",
                        "base_multiplier",
                        "effective_multiplier",
                        "cm_delta_ratio",
                        "public",
                        "threshold",
                        "scale_down_threshold",
                        "is_starving",
                    ]
                )
                for ctx in contexts.values():
                    writer.writerow(
                        [
                            ctx.chute_id,
                            ctx.action,
                            ctx.current_count,
                            ctx.target_count,
                            ctx.upscale_amount,
                            ctx.downscale_amount,
                            ctx.utilization_basis,
                            ctx.smoothed_util,
                            ctx.rate_limit_basis,
                            ctx.urgency_score,
                            ctx.smoothed_urgency,
                            ctx.boost,
                            ctx.base_multiplier,
                            ctx.effective_multiplier,
                            ctx.cm_delta_ratio,
                            ctx.public,
                            ctx.threshold,
                            ctx.scale_down_threshold,
                            ctx.is_starving,
                        ]
                    )
            logger.info(f"Exported dry run data to {dry_run_csv}")

            # Export simulation CSVs if simulation was run
            if simulation_results:
                # Export miner score changes
                base_path = dry_run_csv.rsplit(".", 1)[0]
                miner_csv = f"{base_path}_miner_scores.csv"
                with open(miner_csv, "w", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow(
                        [
                            "hotkey",
                            "current_score",
                            "simulated_score",
                            "score_change",
                            "score_change_pct",
                            "current_raw_compute_units",
                            "simulated_raw_compute_units",
                        ]
                    )
                    for mc in simulation_results["miner_changes"]:
                        writer.writerow(
                            [
                                mc["hotkey"],
                                mc["current_score"],
                                mc["simulated_score"],
                                mc["score_change"],
                                mc["score_change_pct"],
                                mc["current_raw"],
                                mc["simulated_raw"],
                            ]
                        )
                logger.info(f"Exported miner score simulation to {miner_csv}")

                # Export instance-level multiplier changes
                instance_csv = f"{base_path}_instance_changes.csv"
                with open(instance_csv, "w", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow(
                        [
                            "instance_id",
                            "chute_id",
                            "miner_hotkey",
                            "current_multiplier",
                            "new_multiplier",
                            "target_multiplier",
                            "hours_since_activation",
                        ]
                    )
                    for ic in simulation_results["instance_changes"]:
                        writer.writerow(
                            [
                                ic["instance_id"],
                                ic["chute_id"],
                                ic["miner_hotkey"],
                                ic["current_multiplier"],
                                ic["new_multiplier"],
                                ic["target_multiplier"],
                                ic["hours_since_activation"],
                            ]
                        )
                logger.info(f"Exported instance multiplier changes to {instance_csv}")

        # Show thrashing instances if requested
        if show_thrashing:
            await _log_thrashing_instances()

        logger.info("=== END DRY RUN ===")
        return
    else:
        # Update boost values in database
        await update_chute_boosts(chute_boosts)

        # Save smoothed metrics to Redis for next run's EMA calculation
        await save_smoothed_metrics(new_smoothed_metrics)

        # Refresh instance compute_multipliers (only if requested - should run hourly, not every run)
        if refresh_multipliers:
            await refresh_instance_compute_multipliers()

        # Manage rolling updates (replacement + hard cap enforcement)
        # In soft_mode, still manage rolling updates (they're not scale-downs, they're version transitions)
        if not soft_mode:
            await manage_rolling_updates(
                db_now,
                chute_target_counts,
                chute_rate_limiting,
                interval_seconds=AUTOSCALER_FULL_INTERVAL_SECONDS,
            )

    # Include filtered chutes in capacity logging with their actual targets
    for chute_id, target in filtered_chutes.items():
        chute_actions[chute_id] = "filtered"
        chute_target_counts[chute_id] = target

    # Build effective_multipliers dict from contexts
    chute_effective_multipliers = {
        ctx.chute_id: ctx.effective_multiplier
        for ctx in contexts.values()
        if ctx.effective_multiplier > 0
    }

    await log_capacity_metrics(
        chute_metrics, chute_actions, chute_target_counts, chute_effective_multipliers
    )

    # 5. Execute Downsizing (skip in soft_mode)
    if soft_mode:
        if to_downsize:
            logger.info(f"SOFT MODE: Skipping {len(to_downsize)} scale-down operations")
        return

    await execute_downsizing(to_downsize, db_now)


def calculate_demand_based_instances(ctx: AutoScaleContext) -> int:
    """
    Calculate how many additional instances are needed based on request volume.

    The idea: if we're rate limiting, we have unmet demand. We estimate how many
    additional instances would be needed to handle that demand.

    Assumptions:
    - Each instance handles roughly (completed_requests / current_instances) requests
    - Not all rate-limited requests are unique (many are retries)
    - We estimate ~40% of rate-limited requests are unique demand (conservative)
    """
    if ctx.current_count == 0:
        return 1

    # Use 5m window for most responsive scaling, fall back to 15m if 5m has no data
    completed = ctx.completed_5m if ctx.completed_5m > 0 else ctx.completed_15m
    rate_limited = ctx.rate_limited_count_5m if ctx.completed_5m > 0 else ctx.rate_limited_count_15m

    if rate_limited == 0:
        return 0

    # Edge case: everything is being rate-limited (completed=0, rate_limited>0)
    # This means we have demand but zero capacity is getting through
    if completed == 0:
        # Conservative: add 1 instance to start getting some throughput data
        # Can't estimate demand without knowing per-instance throughput
        return 1

    # Throughput per instance
    throughput_per_instance = completed / ctx.current_count
    if throughput_per_instance <= 0:
        return 1

    # Estimate unique rate-limited requests (exclude retries)
    # Conservative estimate: 40% are unique, 60% are retries.
    # In reality, it's probably orders of magnitude more retries.
    RETRY_FACTOR = 0.4
    estimated_unique_unmet = rate_limited * RETRY_FACTOR

    # How many additional instances needed to handle the unmet demand?
    additional_needed = math.ceil(estimated_unique_unmet / throughput_per_instance)

    # Cap the addition to prevent runaway scaling, don't more
    # than double the current count in one cycle
    max_addition = max(ctx.current_count, 5)
    additional_needed = min(additional_needed, max_addition)

    return additional_needed


def clamp_to_max_instances(ctx: AutoScaleContext):
    """
    Ensure target_count never exceeds the chute's configured max_instances.
    """
    effective_max = ctx.max_instances
    if ctx.has_rolling_update and ctx.old_instance_count:
        effective_max = ctx.max_instances + ctx.old_instance_count
    if ctx.target_count > effective_max:
        ctx.target_count = effective_max
        # Recalculate upscale_amount based on clamped target
        ctx.upscale_amount = max(0, ctx.target_count - ctx.current_count)
        if ctx.upscale_amount == 0 and ctx.action == "scale_up_candidate":
            ctx.action = "no_action"


async def calculate_local_decision(ctx: AutoScaleContext):
    """
    Determine what a chute WANTS to do based purely on its own metrics.
    """
    # Disabled chutes should not have any instances
    if ctx.info and ctx.info.disabled:
        ctx.target_count = 0
        ctx.action = "no_action"
        logger.info(f"Chute {ctx.chute_id} is disabled, target_count set to 0.")
        return

    # Private Chutes logic
    if (
        ctx.info
        and not ctx.info.public
        and not has_legacy_private_billing(ctx.info)
        and ctx.info.user_id != await chutes_user_id()
    ):
        if ctx.info.user_balance <= 0:
            ctx.target_count = 0
            ctx.action = "no_action"
            logger.info(f"User for private chute {ctx.chute_id=} has no balance, unable to scale.")
            return

        # Private chutes use a higher default threshold (0.75) than public (0.6)
        private_threshold = ctx.info.scaling_threshold or 0.75
        # For private chutes, max_instances defaults to 1 if not set
        private_max = ctx.info.max_instances if ctx.info.max_instances else 1
        if ctx.current_count:
            if ctx.utilization_basis >= private_threshold and ctx.current_count < private_max:
                ctx.upscale_amount = 1
                ctx.target_count = ctx.current_count + 1
                ctx.action = "scale_up_candidate"
                logger.info(f"Private chute {ctx.chute_id=} high util, adding capacity")
            elif ctx.utilization_basis < private_threshold and ctx.current_count > 1:
                ctx.downscale_amount = 1
                ctx.target_count = ctx.current_count - 1
                ctx.action = "scaled_down"
                logger.info(f"Private chute {ctx.chute_id=} low util, removing instance")
        elif await check_bounty_exists(ctx.chute_id):
            # Bounty was created via user request (invocation or warmup) - scale up
            ctx.upscale_amount = 1
            ctx.target_count = 1
            ctx.action = "scale_up_candidate"
            logger.info(f"Private chute {ctx.chute_id=} has active bounty, adding initial capacity")
        else:
            ctx.target_count = 0
            ctx.action = "no_action"
            logger.info(f"Private chute {ctx.chute_id=} has no bounty, waiting for user request.")
        return

    failsafe_min = FAILSAFE.get(ctx.chute_id, UNDERUTILIZED_CAP)
    if ctx.chute_id in LIMIT_OVERRIDES:
        limit = LIMIT_OVERRIDES[ctx.chute_id]
        ctx.target_count = limit
        if ctx.current_count > limit:
            ctx.downscale_amount = ctx.current_count - limit
            ctx.action = "scaled_down"
            logger.info(f"Chute {ctx.chute_id}: limit override, scaling down to {limit}")
        elif ctx.current_count < limit:
            ctx.upscale_amount = limit - ctx.current_count
            ctx.action = "scale_up_candidate"
            logger.info(f"Chute {ctx.chute_id}: limit override, scaling up to {limit}")
        return

    if ctx.public and ctx.current_count == 0:
        ctx.target_count = max(failsafe_min, 2)
        ctx.upscale_amount = max(1, ctx.target_count - ctx.current_count)
        ctx.action = "scale_up_candidate"
        clamp_to_max_instances(ctx)
        logger.info(
            f"Scale up: {ctx.chute_id} - bounty with no instances, target={ctx.target_count}"
        )
        return

    # Rolling updates: allow scaling up to ensure smooth transition
    if ctx.has_rolling_update:
        if ctx.is_starving:
            # High demand during rolling update - scale up aggressively
            num_to_add = max(2, int(ctx.current_count * 0.2))
            ctx.upscale_amount = num_to_add
            ctx.target_count = max(failsafe_min, ctx.current_count + num_to_add)
            ctx.action = "scale_up_candidate"
            clamp_to_max_instances(ctx)
            logger.info(
                f"Scale up: {ctx.chute_id} - rolling update with high demand, "
                f"util={ctx.utilization_basis:.1%}, adding {ctx.upscale_amount} instances"
            )
        else:
            # Rolling update without high demand - still allow +1 for buffer
            ctx.upscale_amount = 1
            ctx.target_count = max(failsafe_min, ctx.current_count + 1)
            ctx.action = "scale_up_candidate"
            clamp_to_max_instances(ctx)
            logger.info(
                f"Scale up: {ctx.chute_id} - rolling update buffer, adding {ctx.upscale_amount} instance(s)"
            )
        return

    if ctx.is_starving:
        num_to_add = 1

        # Account for pending instances in effective capacity calculations
        # Pending instances are already spinning up and will soon be available
        effective_count = ctx.current_count + ctx.pending_instance_count

        # Calculate demand-based scaling if we have rate limiting
        demand_based_add = 0
        if ctx.rate_limit_basis > 0:
            demand_based_add = calculate_demand_based_instances(ctx)

        # Very high utilization - aggressive scale up (but account for pending)
        if ctx.utilization_basis >= 0.85:
            num_to_add = max(5, int(ctx.current_count * 0.8))
        elif ctx.utilization_basis >= ctx.threshold * 1.5:
            num_to_add = max(3, int(ctx.current_count * 0.5))
        # Rate limiting - use demand-based calculation
        elif demand_based_add > 0:
            # Use the demand-based calculation, but ensure minimum based on ratio severity
            # Only apply ratio-based minimums if we have significant volume (>50 rate-limited requests)
            # to avoid over-scaling for low-volume spikes
            significant_volume = ctx.rate_limited_count_5m >= 50 or ctx.rate_limited_count_15m >= 50
            if ctx.rate_limit_basis >= 0.2 and significant_volume:
                num_to_add = max(demand_based_add, 3)
            elif ctx.rate_limit_basis >= 0.1 and significant_volume:
                num_to_add = max(demand_based_add, 2)
            else:
                num_to_add = max(demand_based_add, 1)
        # Only historical rate limiting (1h only) - minimal scale up
        elif ctx.rate_limit_1h > 0 and ctx.rate_limit_basis < RATE_LIMIT_SCALE_UP:
            num_to_add = 1

        # Subtract pending instances - they're already in flight
        # This prevents repeatedly requesting new instances while previous batch is still spinning up
        num_to_add = max(0, num_to_add - ctx.pending_instance_count)

        if num_to_add == 0:
            # We have enough instances spinning up, no additional scaling needed
            logger.info(
                f"Scale up skipped: {ctx.chute_id} - {ctx.pending_instance_count} pending instances "
                f"already spinning up, effective_count={effective_count}"
            )
            return

        ctx.upscale_amount = num_to_add
        ctx.target_count = max(failsafe_min, ctx.current_count + num_to_add)
        ctx.action = "scale_up_candidate"
        clamp_to_max_instances(ctx)
        logger.info(
            f"Scale up: {ctx.chute_id} - high demand, util={ctx.utilization_basis:.1%}, "
            f"rate_limit(5m={ctx.rate_limit_5m:.1%}, 15m={ctx.rate_limit_15m:.1%}, 1h={ctx.rate_limit_1h:.1%}), "
            f"completed_5m={ctx.completed_5m:.0f}, rate_limited_5m={ctx.rate_limited_count_5m:.0f}, "
            f"demand_based_add={demand_based_add}, pending={ctx.pending_instance_count}, "
            f"adding {ctx.upscale_amount} instances, target={ctx.target_count}"
        )
        return

    # Voluntary Scale-Down: if SMOOTHED utilization is below scale_down_threshold
    # Using smoothed_util prevents borderline chutes from flip-flopping between runs
    # This is conservative - gated by moving average check during execution
    # Separate from forced donations which happen in arbitration phase
    # Don't scale down if there are pending instances - they'll affect utilization soon
    if (
        ctx.smoothed_util < ctx.scale_down_threshold
        and ctx.current_count > failsafe_min
        and not ctx.any_rate_limiting
        and ctx.pending_instance_count == 0
    ):
        # Calculate ideal target count that would bring utilization up to threshold
        # ideal_count = current_count * (util / threshold)
        # Example: 29 instances at 6.2% util with 40% threshold -> 29 * 0.062/0.4 = 4.5 -> 5 instances
        if ctx.utilization_basis > 0:
            ideal_count = max(
                failsafe_min, math.ceil(ctx.current_count * ctx.utilization_basis / ctx.threshold)
            )
        else:
            ideal_count = failsafe_min

        excess = ctx.current_count - ideal_count

        # Remove 1/4 of excess per cycle to avoid thrashing
        # This allows gradual convergence while still being responsive
        # e.g., 29 instances, ideal=5, excess=24 -> remove 6 this cycle
        # Next cycle: ~23 instances, utilization adjusts, recalculate
        num_to_remove = max(1, excess // 5)

        # Ensure we don't go below failsafe
        num_to_remove = min(num_to_remove, ctx.current_count - failsafe_min)

        if num_to_remove > 0:
            proposed_target = ctx.current_count - num_to_remove
            # Verify projected utilization stays below scale-up threshold
            projected_util = (ctx.utilization_basis * ctx.current_count) / proposed_target

            if projected_util < min(ctx.threshold, UTILIZATION_SCALE_DOWN):
                ctx.downscale_amount = num_to_remove
                ctx.target_count = proposed_target
                ctx.action = "scale_down_candidate"
                logger.info(
                    f"Scale down candidate: {ctx.chute_id} - smoothed_util={ctx.smoothed_util:.1%} < {ctx.scale_down_threshold:.1%}, "
                    f"raw_util={ctx.utilization_basis:.1%}, ideal_count={ideal_count}, excess={excess}, "
                    f"removing {num_to_remove} (1/3 of excess), projected_util={projected_util:.1%}, target={ctx.target_count}"
                )
                return

    # Moderate Scale-Up: at/above threshold but not starving or rate limiting.
    # Scale to bring utilization back to the target threshold.
    if ctx.utilization_basis >= ctx.threshold and not ctx.blocked_by_starving:
        # Account for pending instances in capacity calculations
        effective_count = ctx.current_count + ctx.pending_instance_count
        # target_count ~= current_count * (util / threshold)
        desired_count = math.ceil(ctx.current_count * (ctx.utilization_basis / ctx.threshold))
        desired_count = max(ctx.current_count + 1, desired_count)
        ctx.target_count = max(failsafe_min, desired_count)
        ctx.upscale_amount = max(0, ctx.target_count - ctx.current_count)
        # Subtract pending instances - they're already in flight
        ctx.upscale_amount = max(0, ctx.upscale_amount - ctx.pending_instance_count)
        if ctx.upscale_amount > 0:
            ctx.action = "scale_up_candidate"
            clamp_to_max_instances(ctx)
            logger.info(
                f"Scale up: {ctx.chute_id} - util={ctx.utilization_basis:.1%} >= "
                f"{ctx.threshold:.1%}, pending={ctx.pending_instance_count}, "
                f"adding {ctx.upscale_amount} instance(s), target={ctx.target_count}"
            )
        elif ctx.pending_instance_count > 0:
            logger.info(
                f"Scale up skipped: {ctx.chute_id} - {ctx.pending_instance_count} pending instances "
                f"already spinning up"
            )
    elif ctx.utilization_basis >= ctx.threshold and ctx.blocked_by_starving:
        logger.info(
            f"Scale up blocked: {ctx.chute_id} - util={ctx.utilization_basis:.1%} >= "
            f"{ctx.threshold:.1%} but GPU-compatible starving chutes exist"
        )
    else:
        # Default/Stable - maintain current count (respecting failsafe minimum)
        # Chutes in stable zone (between scale_down_threshold and threshold) can still
        # be forced to donate capacity during arbitration if others are starving
        ctx.target_count = max(failsafe_min, ctx.current_count)

    if ctx.info.new_chute:
        # New chutes get a boost, but still respect max_instances
        target_for_new = min(10, ctx.max_instances)
        ctx.target_count = max(target_for_new, ctx.target_count)
        if ctx.target_count > ctx.current_count:
            ctx.upscale_amount = ctx.target_count - ctx.current_count
            ctx.action = "scale_up_candidate"
            logger.info(
                f"Scale up: {ctx.chute_id} - new chute, "
                f"adding {ctx.upscale_amount} instances, target={ctx.target_count}"
            )

    # Always clamp to max_instances at the end
    clamp_to_max_instances(ctx)


def apply_overrides(ctx: AutoScaleContext):
    """
    Apply failsafe minimums and other overrides to the scaling decision.
    This should cap decisions, not override them with potentially more aggressive values.

    Only applies to public chutes - private chutes handle their own minimums.
    """
    # Private chutes are not subject to UNDERUTILIZED_CAP failsafe
    if not ctx.public:
        return

    # For public chutes, ensure we don't go below failsafe minimum
    failsafe_min = FAILSAFE.get(ctx.chute_id, UNDERUTILIZED_CAP)
    if ctx.target_count < failsafe_min:
        ctx.target_count = failsafe_min
        # Cap downscale_amount to not go below failsafe
        max_allowed_downscale = max(0, ctx.current_count - failsafe_min)
        if ctx.downscale_amount > max_allowed_downscale:
            ctx.downscale_amount = max_allowed_downscale
            if max_allowed_downscale == 0:
                ctx.action = "no_action"


@retry_on_db_failure()
async def execute_downsizing(to_downsize: List[Tuple[str, int, Set[str]]], db_now: datetime):
    """
    Perform the actual removal of instances.
    """
    instances_removed = 0
    gpus_removed = 0

    for chute_id, num_to_remove, preferred_gpus in to_downsize:
        if num_to_remove <= 0:
            continue

        async with get_session() as session:
            await session.execute(text("SET LOCAL statement_timeout = '5s'"))
            chute_q = await session.execute(
                select(Chute)
                .where(Chute.chute_id == chute_id)
                .options(selectinload(Chute.instances).selectinload(Instance.nodes))
            )
            chute = chute_q.unique().scalar_one_or_none()
            if not chute:
                continue

            active_instances = [
                inst
                for inst in chute.instances
                if inst.verified and inst.active and (not inst.config or not inst.config.job_id)
            ]

            # Prefer removing broken or unestablished instances first
            valid_candidates = []
            for inst in active_instances:
                if len(inst.nodes) != (chute.node_selector.get("gpu_count") or 1):
                    await purge_and_notify(inst, "Instance node count mismatch")
                    num_to_remove -= 1
                    instances_removed += 1
                elif db_now.replace(tzinfo=None) - inst.activated_at.replace(
                    tzinfo=None
                ) >= timedelta(minutes=63):
                    valid_candidates.append(inst)

            if num_to_remove <= 0 or not valid_candidates:
                continue

            # Target instances matching the preferred hardware identified in arbitration
            for _ in range(num_to_remove):
                if not valid_candidates:
                    break

                match_found = False
                if preferred_gpus:
                    for i, inst in enumerate(valid_candidates):
                        if inst.nodes and inst.nodes[0].gpu_identifier in preferred_gpus:
                            targeted_instance = valid_candidates.pop(i)
                            match_found = True
                            break

                if not match_found:
                    targeted_instance = valid_candidates.pop(
                        random.randrange(len(valid_candidates))
                    )

                logger.info(
                    f"Downscaling {chute_id}: removing {targeted_instance.instance_id} ({targeted_instance.nodes[0].gpu_identifier if targeted_instance.nodes else 'unknown'})"
                )
                await purge_and_notify(
                    targeted_instance, "Autoscaler adjustment", valid_termination=True
                )
                await invalidate_instance_cache(chute_id, targeted_instance.instance_id)
                instances_removed += 1
                gpus_removed += len(targeted_instance.nodes)

    if instances_removed:
        logger.success(
            f"Scaled down total: {instances_removed} instances, {gpus_removed} GPUs freed."
        )


if __name__ == "__main__":
    gc.set_threshold(5000, 50, 50)
    parser = argparse.ArgumentParser(description="Auto-scale chutes based on utilization")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Logging only - no Redis writes, no DB writes, no instance changes",
    )
    parser.add_argument(
        "--soft",
        action="store_true",
        help="Soft mode - updates Redis targets, boosts, rolling updates, "
        "and logs to capacity_log, but skips all scale-downs (both voluntary and forced)",
    )
    parser.add_argument(
        "--csv",
        type=str,
        metavar="FILE",
        help="Export dry run results to CSV file (only works with --dry-run)",
    )
    parser.add_argument(
        "--simulate",
        action="store_true",
        help="Simulate miner scores with updated compute multipliers (only works with --dry-run)",
    )
    parser.add_argument(
        "--refresh-multipliers",
        action="store_true",
        help="Refresh instance compute_multipliers (should run hourly before :05, not every run)",
    )
    parser.add_argument(
        "--show-thrashing",
        action="store_true",
        help="Show instances currently in thrash penalty period (only works with --dry-run)",
    )
    args = parser.parse_args()
    if args.csv and not args.dry_run:
        parser.error("--csv requires --dry-run")
    if args.simulate and not args.dry_run:
        parser.error("--simulate requires --dry-run")
    if args.show_thrashing and not args.dry_run:
        parser.error("--show-thrashing requires --dry-run")
    asyncio.run(
        perform_autoscale(
            dry_run=args.dry_run,
            soft_mode=args.soft,
            dry_run_csv=args.csv,
            refresh_multipliers=args.refresh_multipliers,
            simulate_scores=args.simulate,
            show_thrashing=args.show_thrashing,
        )
    )
