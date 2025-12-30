"""
Auto-scale chutes based on utilization.
"""

import gc
import os
import math
import asyncio
import argparse
import random
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
from api.bounty.util import check_bounty_exists
from api.user.service import chutes_user_id
from api.util import has_legacy_private_billing, notify_deleted
from api.chute.schemas import Chute, NodeSelector, RollingUpdate
from api.instance.schemas import Instance, LaunchConfig
from api.instance.util import invalidate_instance_cache, cleanup_expired_connections
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
    chute_id: str, current_count: int, proposed_target: int
) -> Tuple[bool, str]:
    """
    Check if scale-down is permitted based on historical capacity_log trends.

    Returns (permitted, reason) tuple.

    Scale-down is permitted if:
    1. We have enough historical data (at least 3 samples)
    2. Proposed target isn't drastically below recent average (within SCALE_DOWN_MAX_DROP_RATIO)
    3. No significant rate limiting occurred in the lookback window

    This prevents thrashing and respects bursty traffic patterns.
    """
    async with get_session() as session:
        await session.execute(text("SET LOCAL statement_timeout = '5s'"))
        result = await session.execute(
            text("""
                SELECT
                    AVG(target_count) as avg_target,
                    MAX(target_count) as max_target,
                    AVG(utilization_15m) as avg_util,
                    MAX(GREATEST(
                        COALESCE(rate_limit_ratio_5m, 0),
                        COALESCE(rate_limit_ratio_15m, 0),
                        COALESCE(rate_limit_ratio_1h, 0)
                    )) as max_rate_limit,
                    COUNT(*) as sample_count
                FROM capacity_log
                WHERE chute_id = :chute_id
                  AND timestamp >= NOW() - INTERVAL :lookback
            """),
            {"chute_id": chute_id, "lookback": f"{SCALE_DOWN_LOOKBACK_MINUTES} minutes"},
        )
        row = result.fetchone()

        if not row or row.sample_count < 3:
            return False, "insufficient_history"

        # Check for rate limiting in the lookback window
        if row.max_rate_limit and row.max_rate_limit >= 0.01:
            return False, f"rate_limiting_in_window ({row.max_rate_limit:.1%})"

        # Check if proposed target is within acceptable range of rolling average
        min_allowed_target = max(1, int(row.avg_target * SCALE_DOWN_MAX_DROP_RATIO))
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

# Higher min instance counts for some chutes...
LIMIT_OVERRIDES = {
    "0596f791-79e9-51e1-bf93-93f22a4e8110": 1,
    "8f3bb827-b9e6-5487-88bc-ee8f0c6f5810": 4,
    "0d7184a2-32a3-53e0-9607-058c37edaab5": 36,
}
FAILSAFE = {
    "722df757-203b-58df-b54b-22130fd1fc53": 20,
    "4fa0c7f5-82f7-59d1-8996-661bb778893d": 15,
    "d711f181-5b21-5169-a011-ccb472a1604f": 10,
    "08a7a60f-6956-5a9e-9983-5603c3ac5a38": 10,
    "579ca543-dda4-51d0-83ef-5667d1a5ed5f": 9,
    "4f82321e-3e58-55da-ba44-051686ddbfe5": 8,
    "8d008c10-60d3-51e8-9272-c428ed6ff576": 6,
    "02636d63-c996-5779-a0a2-25712469a7ca": 6,
    "b2b7a64c-b203-5a5f-8982-a9c5cc12058c": 6,
    "4bbc44e9-6bfc-5e21-a91d-129bff2fb6d4": 5,
    "ae3b9d04-28fa-543a-9276-290da772dc23": 5,
    "aef797d4-f375-5beb-9986-3ad245947469": 5,
    "689d2caa-01c1-5de1-ba69-39c5398be0c6": 5,
}


class AutoScaleContext:
    def __init__(
        self, chute_id, metrics, info, supported_gpus, instances: List[Instance], db_now: datetime
    ):
        self.chute_id = chute_id
        self.metrics = metrics
        self.info = info
        self.supported_gpus = supported_gpus
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
        self.has_rolling_update = info.has_rolling_update if info else False
        # max_instances: None means unbounded, use a large number for comparisons
        self.max_instances = info.max_instances if (info and info.max_instances) else 10000
        self.public = info.public if info else True

        # Decision outputs
        self.target_count = self.current_count
        self.action = "no_action"
        self.urgency_score = 0.0
        self.is_starving = False
        self.is_donor = False
        self.is_critical_donor = False
        self.downscale_amount = 0
        self.upscale_amount = 0
        self.preferred_downscale_gpus = set()
        self.boost = 1.0  # Compute multiplier boost (1.0 - 4.0)
        self.locked_for_priority = False  # True if locked to let higher-urgency chutes scale


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
                                Instance.created_at <= func.now() - timedelta(hours=1, minutes=30),
                            ),
                            and_(
                                Instance.config_id.is_(None),
                                Instance.created_at <= func.now() - timedelta(hours=1, minutes=30),
                            ),
                        ),
                    ),
                    and_(
                        Instance.verified.is_(True),
                        Instance.active.is_(False),
                        Instance.config_id.isnot(None),
                        LaunchConfig.verified_at <= func.now() - timedelta(hours=1, minutes=30),
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


# Compute multiplier adjustment timing constants
# No adjustment for the first N hours after activation (miners keep their original boost)
COMPUTE_MULTIPLIER_HOLD_HOURS = 2.0
# Total hours until fully adjusted to target (includes hold period)
COMPUTE_MULTIPLIER_FULL_ADJUST_HOURS = 8.0
# Ramp duration (after hold period)
COMPUTE_MULTIPLIER_RAMP_HOURS = COMPUTE_MULTIPLIER_FULL_ADJUST_HOURS - COMPUTE_MULTIPLIER_HOLD_HOURS


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

    For bounty instances, the target includes the decaying bounty boost.
    """
    from api.chute.util import calculate_effective_compute_multiplier
    from metasync.constants import BOUNTY_BOOST_INITIAL, BOUNTY_BOOST_DECAY_HOURS

    logger.info("Refreshing compute multipliers for active instances...")

    async with get_session() as session:
        await session.execute(text("SET LOCAL statement_timeout = '10s'"))
        # Load chutes (optionally filtered)
        query = select(Chute)
        if chute_ids:
            query = query.where(Chute.chute_id.in_(chute_ids))
        result = await session.execute(query)
        chutes = result.scalars().all()

        instances_updated = 0
        for chute in chutes:
            # Get base multiplier without bounty
            effective_data = await calculate_effective_compute_multiplier(
                chute, include_bounty=False
            )
            base_multiplier = effective_data["effective_compute_multiplier"]

            # Update instances without bounty
            # Blend formula with ease-in curve:
            #   hours = time since activation in hours
            #   if hours <= hold_hours: no change
            #   elif hours >= full_adjust_hours: clamp to target
            #   else: t = (hours - hold) / ramp; blend = t²; result = original*(1-blend) + target*blend
            result = await session.execute(
                text("""
                    UPDATE instances
                    SET compute_multiplier = CASE
                        -- Before hold period ends: don't touch (but initialize if NULL)
                        WHEN EXTRACT(EPOCH FROM (NOW() - activated_at)) / 3600.0 <= :hold_hours
                            THEN COALESCE(compute_multiplier, :target)
                        -- After full adjustment period: clamp to target
                        WHEN EXTRACT(EPOCH FROM (NOW() - activated_at)) / 3600.0 >= :full_hours
                            THEN :target
                        -- During ramp: ease-in blend (t² curve)
                        ELSE (
                            SELECT
                                compute_multiplier * (1 - blend) + :target * blend
                            FROM (
                                SELECT POWER(
                                    (EXTRACT(EPOCH FROM (NOW() - activated_at)) / 3600.0 - :hold_hours)
                                    / :ramp_hours,
                                    2
                                ) AS blend
                            ) AS b
                        )
                    END
                    WHERE chute_id = :chute_id
                      AND (bounty IS NULL OR bounty = false)
                      AND active = true
                      AND verified = true
                      AND activated_at IS NOT NULL
                      AND (
                          compute_multiplier IS NULL
                          OR (
                              EXTRACT(EPOCH FROM (NOW() - activated_at)) / 3600.0 > :hold_hours
                              AND ABS(compute_multiplier - :target) > 0.001
                          )
                      )
                    RETURNING instance_id
                """),
                {
                    "chute_id": chute.chute_id,
                    "target": base_multiplier,
                    "hold_hours": COMPUTE_MULTIPLIER_HOLD_HOURS,
                    "full_hours": COMPUTE_MULTIPLIER_FULL_ADJUST_HOURS,
                    "ramp_hours": COMPUTE_MULTIPLIER_RAMP_HOURS,
                },
            )
            instances_updated += len(result.fetchall())

            # Update instances with bounty: target includes decaying bounty boost
            # Same ease-in blend logic, but target is base_multiplier * bounty_decay
            result = await session.execute(
                text("""
                    UPDATE instances
                    SET compute_multiplier = CASE
                        -- Before hold period ends: don't touch (but initialize if NULL)
                        WHEN EXTRACT(EPOCH FROM (NOW() - activated_at)) / 3600.0 <= :hold_hours
                            THEN COALESCE(compute_multiplier, target_mult)
                        -- After full adjustment period: clamp to target
                        WHEN EXTRACT(EPOCH FROM (NOW() - activated_at)) / 3600.0 >= :full_hours
                            THEN target_mult
                        -- During ramp: ease-in blend (t² curve)
                        ELSE (
                            compute_multiplier * (1 - POWER(
                                (EXTRACT(EPOCH FROM (NOW() - instances.activated_at)) / 3600.0 - :hold_hours)
                                / :ramp_hours,
                                2
                            )) + target_mult * POWER(
                                (EXTRACT(EPOCH FROM (NOW() - instances.activated_at)) / 3600.0 - :hold_hours)
                                / :ramp_hours,
                                2
                            )
                        )
                    END
                    FROM (
                        SELECT
                            instance_id,
                            :base_multiplier * GREATEST(
                                1.0,
                                :initial_boost - (
                                    LEAST(
                                        EXTRACT(EPOCH FROM (NOW() - activated_at)) / 3600.0,
                                        :decay_hours
                                    ) / :decay_hours * (:initial_boost - 1.0)
                                )
                            ) AS target_mult
                        FROM instances
                        WHERE chute_id = :chute_id
                          AND bounty = true
                          AND active = true
                          AND verified = true
                          AND activated_at IS NOT NULL
                    ) AS targets
                    WHERE instances.instance_id = targets.instance_id
                      AND (
                          instances.compute_multiplier IS NULL
                          OR (
                              EXTRACT(EPOCH FROM (NOW() - instances.activated_at)) / 3600.0 > :hold_hours
                              AND ABS(instances.compute_multiplier - targets.target_mult) > 0.001
                          )
                      )
                    RETURNING instances.instance_id
                """),
                {
                    "chute_id": chute.chute_id,
                    "base_multiplier": base_multiplier,
                    "initial_boost": BOUNTY_BOOST_INITIAL,
                    "decay_hours": BOUNTY_BOOST_DECAY_HOURS,
                    "hold_hours": COMPUTE_MULTIPLIER_HOLD_HOURS,
                    "full_hours": COMPUTE_MULTIPLIER_FULL_ADJUST_HOURS,
                    "ramp_hours": COMPUTE_MULTIPLIER_RAMP_HOURS,
                },
            )
            instances_updated += len(result.fetchall())

        if instances_updated:
            await session.commit()
            logger.success(f"Updated compute_multiplier for {instances_updated} instances")
        else:
            logger.info("No compute_multiplier updates needed")


@retry_on_db_failure()
async def manage_rolling_updates(
    db_now: datetime,
    chute_target_counts: Dict[str, int] | None = None,
    chute_rate_limiting: Dict[str, bool] | None = None,
):
    """
    Manage rolling updates by replacing old-version instances with new-version capacity.
    Enforces a hard 3-hour cap; after that, all remaining old instances are deleted.
    """
    max_duration = timedelta(hours=3)
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
                    autoscaler_interval = 30 * 60
                    remaining_cycles = max(1, math.ceil(remaining_seconds / autoscaler_interval))
                    deletions_needed = math.ceil(len(old_instances) / remaining_cycles)

                    is_rate_limited = False
                    if chute_rate_limiting is not None:
                        is_rate_limited = chute_rate_limiting.get(rolling_update.chute_id, False)

                    deletable = 0
                    if total_active > target:
                        deletable = max(total_active - target, deletions_needed)
                    elif not is_rate_limited:
                        deletable = deletions_needed

                    deletable = min(deletable, len(old_instances))
                    if deletable > 0:
                        to_delete = old_instances[:deletable]

            if not to_delete:
                continue

            for instance in to_delete:
                await session.delete(instance)

            if len(old_instances) <= len(to_delete):
                await session.delete(rolling_update)

            await session.commit()

            reason = (
                "Rolling update timeout (3h cap)"
                if elapsed >= max_duration
                else "Rolling update replacement"
            )
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
):
    """
    Log all chute metrics to the capacity_log table.
    """
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
            )
            session.add(capacity_log)
            logged_count += 1

        if logged_count:
            await session.commit()
            logger.info(f"Logged capacity metrics for {logged_count} chutes")


async def perform_autoscale(dry_run: bool = False, soft_mode: bool = False):
    """
    Gather utilization data and make decisions on scaling up/down (or nothing).

    Modes:
    - dry_run: Logging only. No Redis writes, no DB writes, no instance changes.
    - soft_mode: Updates Redis targets, compute multipliers, boosts, rolling updates,
                 and logs to capacity_log, but skips all scale-downs.
    - (default): Full mode - does everything including scale-downs.
    """
    if dry_run and soft_mode:
        logger.warning("Both --dry-run and --soft specified; --dry-run takes precedence")
        soft_mode = False

    mode_str = "DRY-RUN" if dry_run else ("SOFT" if soft_mode else "FULL")
    logger.info(f"Starting autoscaler in {mode_str} mode...")

    if not dry_run:
        logger.info("Performing instance cleanup...")
        await instance_cleanup()
        await cleanup_expired_connections()

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
                            MAX(COALESCE(ucb.effective_balance, 0)) AS user_balance,
                            c.max_instances,
                            c.scaling_threshold,
                            NOW() - c.created_at <= INTERVAL '3 hours' AS new_chute,
                            COUNT(DISTINCT CASE WHEN i.active = true AND i.verified = true THEN i.instance_id END) AS instance_count,
                            EXISTS(SELECT 1 FROM rolling_updates ru WHERE ru.chute_id = c.chute_id) AS has_rolling_update,
                            NOW() AS db_now
                        FROM chutes c
                        LEFT JOIN instances i ON c.chute_id = i.chute_id AND i.verified = true AND i.active = true
                        LEFT JOIN user_current_balance ucb on ucb.user_id = c.user_id
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

    # 1. Initialize Contexts and Calculate Urgency
    contexts: Dict[str, AutoScaleContext] = {}
    starving_chutes: List[AutoScaleContext] = []
    # Track filtered chutes for accurate capacity logging
    filtered_chutes: Dict[str, int] = {}

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
        except Exception:
            logger.warning(f"Failed to parse node selector for {chute_id}")
            supported_gpus = set()

        ctx = AutoScaleContext(
            chute_id, metrics, info, supported_gpus, instances_by_chute[chute_id], db_now
        )
        contexts[chute_id] = ctx

        # Calculate Urgency Score
        # Formula: (RateLimitRatio * 5000) + (Utilization * 100)
        # Prioritizes error-reduction over pure utilization.
        util_score = min(100, ctx.utilization_basis * 100)
        rl_score = ctx.rate_limit_basis * 5000
        ctx.urgency_score = util_score + rl_score

        # Identify Starving Chutes (High Demand)
        if ctx.utilization_basis >= ctx.threshold or ctx.rate_limit_basis >= RATE_LIMIT_SCALE_UP:
            ctx.is_starving = True
            starving_chutes.append(ctx)

        # Identify Potential Donors (for forced donations during arbitration)
        # Private chutes are not donors unless they belong to chutes_user_id (semi-private).
        # Chutes in LIMIT_OVERRIDES should never be preempted.
        allow_donor = ctx.public or (ctx.info and ctx.info.user_id == await chutes_user_id())
        if (
            not ctx.any_rate_limiting
            and ctx.current_count > 0
            and allow_donor
            and ctx.chute_id not in LIMIT_OVERRIDES
        ):
            # Voluntary scale-down candidate: below scale_down_threshold
            # These will scale down on their own (gated by moving average)
            if ctx.utilization_basis < ctx.scale_down_threshold:
                ctx.is_critical_donor = True
                ctx.is_donor = True
            # Forced donation candidate: in stable zone (below threshold but above scale_down_threshold)
            # These won't scale down voluntarily but can be forced to donate when others are starving
            elif ctx.utilization_basis < ctx.threshold:
                ctx.is_donor = True

    # 2. Local Decision Making (Ideal World)
    for ctx in contexts.values():
        await calculate_local_decision(ctx)

    # 3. Global Arbitration (The Real World Matchmaking)
    # Force multiple donors per starving chute based on need, up to a cap
    MAX_FORCED_DONATIONS_PER_CHUTE = 5
    MAX_FORCED_DONATIONS_TOTAL = 20

    total_forced = 0
    if starving_chutes:
        starving_chutes.sort(key=lambda x: x.urgency_score, reverse=True)

        for hungry_ctx in starving_chutes:
            if total_forced >= MAX_FORCED_DONATIONS_TOTAL:
                break

            # How many instances does this chute need?
            instances_needed = hungry_ctx.upscale_amount
            if instances_needed <= 0:
                continue

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
                # Donor must have established instances (1+ hour old) to donate
                if donor.established_instance_count == 0:
                    continue
                # Donor must have capacity above minimum after any pending downscales
                remaining_capacity = donor.current_count - donor.downscale_amount
                if remaining_capacity <= UNDERUTILIZED_CAP:
                    continue

                # Check if donor actually has hardware the starving chute can use
                available_matching_gpus = set(donor.hardware_map.keys()) & needed_gpus
                if available_matching_gpus:
                    # Calculate how many this donor can give (stay above UNDERUTILIZED_CAP)
                    can_give = remaining_capacity - UNDERUTILIZED_CAP
                    eligible_donors.append((donor, available_matching_gpus, can_give))

            if not eligible_donors:
                continue

            # Shuffle for fairness, then take from donors until we have enough
            random.shuffle(eligible_donors)
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
    # High-urgency chutes that still need scaling get priority:
    # - They get a boost multiplier (1.0 - 2.5) based on urgency
    # - Compatible lower-priority chutes get locked from scaling up
    URGENCY_LOCK_THRESHOLD = 100  # Urgency score above which we lock competitors
    URGENCY_MAX_FOR_BOOST = 500  # Urgency score that maps to max boost (2.5)
    URGENCY_BOOST_MIN = 1.0
    URGENCY_BOOST_MAX = 2.5

    high_urgency_chutes = [
        ctx
        for ctx in starving_chutes
        if ctx.urgency_score >= URGENCY_LOCK_THRESHOLD and ctx.upscale_amount > 0
    ]

    for ctx in contexts.values():
        if ctx.is_starving and ctx.upscale_amount > 0:
            # Calculate boost based on urgency (linear scale from 1.0 to 2.5)
            # urgency 0 -> 1.0, urgency >= URGENCY_MAX_FOR_BOOST -> 2.5
            normalized_urgency = min(ctx.urgency_score / URGENCY_MAX_FOR_BOOST, 1.0)
            ctx.boost = URGENCY_BOOST_MIN + (
                normalized_urgency * (URGENCY_BOOST_MAX - URGENCY_BOOST_MIN)
            )
        else:
            ctx.boost = 1.0

    # Lock compatible chutes that are trying to scale up but aren't as urgent
    for hungry_ctx in high_urgency_chutes:
        for ctx in contexts.values():
            if ctx.chute_id == hungry_ctx.chute_id:
                continue
            if ctx.action != "scale_up_candidate":
                continue
            if ctx.urgency_score >= hungry_ctx.urgency_score:
                continue  # Don't lock equally or more urgent chutes
            if ctx.tee != hungry_ctx.tee:
                continue  # Different TEE mode, not competing
            if not (ctx.supported_gpus & hungry_ctx.supported_gpus):
                continue  # Different hardware, not competing

            # This chute is competing for same hardware with lower priority - lock it
            ctx.target_count = ctx.current_count
            ctx.upscale_amount = 0
            ctx.action = "locked_for_priority"
            ctx.locked_for_priority = True
            ctx.boost = 1.0  # No boost for locked chutes
            logger.info(
                f"Priority lock: {ctx.chute_id} locked (urgency={ctx.urgency_score:.0f}) "
                f"to prioritize {hungry_ctx.chute_id} (urgency={hungry_ctx.urgency_score:.0f})"
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
                ctx.chute_id, ctx.current_count, ctx.target_count
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
        # Log what would have happened
        scale_ups = [c for c in contexts.values() if "scale_up" in c.action]
        scale_downs = [
            c
            for c in contexts.values()
            if "scale_down" in c.action or "forced_downscale" in c.action
        ]
        logger.info(
            f"Would scale up: {len(scale_ups)} chutes, Would scale down: {len(scale_downs)} chutes"
        )
        return
    else:
        # Update boost values in database
        await update_chute_boosts(chute_boosts)

        # Refresh instance compute_multipliers based on current chute state and bounty decay
        await refresh_instance_compute_multipliers()

        # Manage rolling updates (replacement + hard cap enforcement)
        # In soft_mode, still manage rolling updates (they're not scale-downs, they're version transitions)
        await manage_rolling_updates(db_now, chute_target_counts, chute_rate_limiting)

    # Include filtered chutes in capacity logging with their actual targets
    for chute_id, target in filtered_chutes.items():
        chute_actions[chute_id] = "filtered"
        chute_target_counts[chute_id] = target

    await log_capacity_metrics(chute_metrics, chute_actions, chute_target_counts)

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

        # Calculate demand-based scaling if we have rate limiting
        demand_based_add = 0
        if ctx.rate_limit_basis > 0:
            demand_based_add = calculate_demand_based_instances(ctx)

        # Very high utilization - aggressive scale up
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

        ctx.upscale_amount = num_to_add
        ctx.target_count = max(failsafe_min, ctx.current_count + num_to_add)
        ctx.action = "scale_up_candidate"
        clamp_to_max_instances(ctx)
        logger.info(
            f"Scale up: {ctx.chute_id} - high demand, util={ctx.utilization_basis:.1%}, "
            f"rate_limit(5m={ctx.rate_limit_5m:.1%}, 15m={ctx.rate_limit_15m:.1%}, 1h={ctx.rate_limit_1h:.1%}), "
            f"completed_5m={ctx.completed_5m:.0f}, rate_limited_5m={ctx.rate_limited_count_5m:.0f}, "
            f"demand_based_add={demand_based_add}, adding {ctx.upscale_amount} instances, target={ctx.target_count}"
        )
        return

    # Voluntary Scale-Down: if utilization is below scale_down_threshold
    # This is conservative - gated by moving average check during execution
    # Separate from forced donations which happen in arbitration phase
    if (
        ctx.utilization_basis < ctx.scale_down_threshold
        and ctx.current_count > failsafe_min
        and not ctx.any_rate_limiting
    ):
        # Calculate what utilization would be after removing one instance
        if ctx.current_count > 1:
            projected_util = (ctx.utilization_basis * ctx.current_count) / (ctx.current_count - 1)
        else:
            projected_util = 1.0

        # Only scale down if projected utilization stays below scale-up threshold
        if projected_util < ctx.threshold:
            ctx.downscale_amount = 1
            ctx.target_count = ctx.current_count - 1
            ctx.action = "scale_down_candidate"
            logger.info(
                f"Scale down candidate: {ctx.chute_id} - util={ctx.utilization_basis:.1%} < {ctx.scale_down_threshold:.1%}, "
                f"projected_util={projected_util:.1%}, target={ctx.target_count}"
            )
            return

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
        help="Soft mode - updates Redis targets, compute multipliers, boosts, rolling updates, "
        "and logs to capacity_log, but skips all scale-downs (both voluntary and forced)",
    )
    args = parser.parse_args()
    asyncio.run(perform_autoscale(dry_run=args.dry_run, soft_mode=args.soft))
