"""
Auto-scale chutes based on utilization.
"""

import gc
import os
import math
import asyncio
import argparse
import random
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
import api.database.orms  # noqa
from sqlalchemy.orm import selectinload, joinedload
from api.database import get_session
from api.config import settings
from api.bounty.util import (
    check_bounty_exists,
    create_bounty_if_not_exists,
    get_bounty_amount,
    send_bounty_notification,
)
from api.user.service import chutes_user_id
from api.util import has_legacy_private_billing
from api.chute.schemas import Chute, NodeSelector
from api.instance.schemas import Instance, LaunchConfig
from api.instance.util import invalidate_instance_cache, cleanup_expired_connections
from api.capacity_log.schemas import CapacityLog
from watchtower import purge, purge_and_notify  # noqa
from api.constants import (
    UNDERUTILIZED_CAP,
    UTILIZATION_SCALE_UP,
    RATE_LIMIT_SCALE_UP,
)


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
        self.instances = instances
        self.db_now = db_now

        # Map actual hardware to specific instance objects
        # Only include established instances (active for 1+ hour) for donor consideration
        self.hardware_map = defaultdict(list)
        self.established_instance_count = 0
        for inst in instances:
            if inst.nodes:
                is_established = db_now.replace(tzinfo=None) - inst.activated_at.replace(
                    tzinfo=None
                ) >= timedelta(minutes=63)
                if is_established:
                    gpu_id = inst.nodes[0].gpu_identifier
                    self.hardware_map[gpu_id].append(inst)
                    self.established_instance_count += 1

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


async def log_capacity_metrics(
    chute_metrics: Dict[str, Dict],
    chute_actions: Dict[str, str],
    chute_target_counts: Dict[str, int],
):
    """
    Log all chute metrics to the capacity_log table.
    """
    async with get_session() as session:
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


async def perform_autoscale(dry_run: bool = False):
    """
    Gather utilization data and make decisions on scaling up/down (or nothing).
    """
    logger.info("Performing instance cleanup...")
    await instance_cleanup()

    # Cleanup the connections while we are at it...
    await cleanup_expired_connections()

    logger.info(f"Fetching metrics from Prometheus and database... (dry_run={dry_run})")
    chute_metrics = await get_all_chute_metrics()

    # Safety check - ensure we have enough data
    if len(chute_metrics) < MIN_CHUTES_FOR_SCALING:
        logger.warning(
            f"Only found {len(chute_metrics)} chutes total, need at least {MIN_CHUTES_FOR_SCALING}. Aborting."
        )
        return
    logger.info(f"Processing metrics for {len(chute_metrics)} chutes")

    # Fetch detailed chute info and ALL active instances (with nodes)
    async with get_session() as session:
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
        db_now = (
            next(iter(chute_info_map.values())).db_now
            if chute_info_map
            else datetime.now(timezone.utc)
        )

        instance_result = await session.execute(
            select(Instance)
            .where(Instance.active.is_(True), Instance.verified.is_(True))
            .options(selectinload(Instance.nodes))
        )
        all_active_instances = instance_result.scalars().all()
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

        # Identify Potential Donors (Low Utilization)
        # ANY rate limiting in any time window prevents being a donor
        if not ctx.any_rate_limiting and ctx.current_count > 0:
            # Critical Donor: Absolute waste, always downscale.
            if ctx.utilization_basis < ctx.threshold * 0.2:
                ctx.is_critical_donor = True
                ctx.is_donor = True
            # Optional Donor: Buffer, only downscale to help others.
            elif ctx.utilization_basis < ctx.threshold * 0.6:
                ctx.is_donor = True

    # 2. Local Decision Making (Ideal World)
    for ctx in contexts.values():
        await calculate_local_decision(ctx)

    # 3. Global Arbitration (The Real World Matchmaking)
    if starving_chutes:
        starving_chutes.sort(key=lambda x: x.urgency_score, reverse=True)

        for hungry_ctx in starving_chutes:
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
                # Donor must not already be tagged for downscale and have capacity above minimum
                if donor.downscale_amount > 0 or donor.current_count <= UNDERUTILIZED_CAP:
                    continue

                # Check if donor actually has hardware the starving chute can use
                available_matching_gpus = set(donor.hardware_map.keys()) & needed_gpus
                if available_matching_gpus:
                    eligible_donors.append((donor, available_matching_gpus))

            # Randomly select one donor from eligible candidates
            if eligible_donors:
                donor, available_matching_gpus = random.choice(eligible_donors)
                chosen_gpu = random.choice(list(available_matching_gpus))
                donor.downscale_amount = 1
                donor.target_count = donor.current_count - 1
                donor.action = "forced_downscale"
                donor.preferred_downscale_gpus.add(chosen_gpu)
                logger.info(
                    f"Arbitration: {donor.chute_id} giving up {chosen_gpu} for {hungry_ctx.chute_id} "
                    f"(Urgency={hungry_ctx.urgency_score:.1f}, eligible_donors={len(eligible_donors)})"
                )

    # 4. Finalize Actions
    chute_actions = {}
    chute_target_counts = {}
    to_downsize: List[Tuple[str, int, Set[str]]] = []

    for ctx in contexts.values():
        apply_overrides(ctx)

        chute_actions[ctx.chute_id] = ctx.action
        chute_target_counts[ctx.chute_id] = ctx.target_count

        await settings.redis_client.set(f"scale:{ctx.chute_id}", ctx.target_count, ex=3700)

        if ctx.downscale_amount > 0:
            to_downsize.append((ctx.chute_id, ctx.downscale_amount, ctx.preferred_downscale_gpus))

    # Include filtered chutes in capacity logging with their actual targets
    for chute_id, target in filtered_chutes.items():
        chute_actions[chute_id] = "filtered"
        chute_target_counts[chute_id] = target

    await log_capacity_metrics(chute_metrics, chute_actions, chute_target_counts)

    # 5. Execute Downsizing
    if dry_run and to_downsize:
        logger.warning("DRY RUN MODE: Skipping actual instance removal")
        for cid, amt, pref in to_downsize:
            logger.info(f"Would remove {amt} from {cid} (Preferred GPUs: {pref or 'any'})")
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
    if ctx.target_count > ctx.max_instances:
        ctx.target_count = ctx.max_instances
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
                if await create_bounty_if_not_exists(ctx.chute_id, lifetime=3600):
                    logger.success(f"Created additional bounty for private chute {ctx.chute_id=}")
                amount = await get_bounty_amount(ctx.chute_id)
                if amount:
                    await send_bounty_notification(ctx.chute_id, amount)
            elif ctx.utilization_basis < private_threshold and ctx.current_count > 1:
                ctx.downscale_amount = 1
                ctx.target_count = ctx.current_count - 1
                ctx.action = "scaled_down"
                logger.info(f"Private chute {ctx.chute_id=} low util, removing instance")
        elif await check_bounty_exists(ctx.chute_id):
            ctx.upscale_amount = 1
            ctx.target_count = 1
            ctx.action = "scale_up_candidate"
            logger.info(f"Private chute {ctx.chute_id=} has active bounty, adding initial capacity")
        else:
            ctx.target_count = 0
            ctx.action = "no_action"
            logger.info(f"Private chute {ctx.chute_id=} has no bounty, not scalable.")
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
            logger.info(f"Scale up: {ctx.chute_id} - rolling update buffer, adding {ctx.upscale_amount} instance(s)")
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

    # Critical Donors always scale down locally.
    # Optional Donors stay 'no_action' until forced by Arbitration.
    # ANY rate limiting prevents scale-down (checked via is_critical_donor which requires !any_rate_limiting)
    if ctx.is_critical_donor and ctx.current_count > failsafe_min:
        # Calculate safe ceiling using 85% of threshold to prevent flapping
        target_ceil_util = ctx.threshold * 0.85
        safe_count = (
            math.ceil((ctx.utilization_basis * ctx.current_count) / target_ceil_util)
            if target_ceil_util > 0
            else failsafe_min
        )
        safe_count = max(safe_count, failsafe_min)

        actual_remove = min(ctx.current_count - safe_count, max(1, int(ctx.current_count * 0.2)))
        if actual_remove > 0:
            ctx.downscale_amount = actual_remove
            ctx.target_count = ctx.current_count - actual_remove
            ctx.action = "scaled_down"
            logger.info(
                f"Scale down: {ctx.chute_id} - critical donor, util={ctx.utilization_basis:.1%}, "
                f"removing {actual_remove} instances, target={ctx.target_count}"
            )
            return

    # Default/Stable
    # Public chutes get failsafe_min as their floor (FAILSAFE[id] or UNDERUTILIZED_CAP)
    # Scaling down happens through critical donor logic when underutilized
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
        help="Run without actually removing instances (simulation mode)",
    )
    args = parser.parse_args()
    asyncio.run(perform_autoscale(dry_run=args.dry_run))
