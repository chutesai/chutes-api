"""
Auto-scale chutes based on utilization.
"""

import gc
import os
import math
import asyncio
import argparse
import random
from loguru import logger
from datetime import timedelta, datetime, timezone
from typing import Dict, Optional, Set
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
from api.gpu import SUPPORTED_GPUS
from api.bounty.util import (
    check_bounty_exists,
    create_bounty_if_not_exists,
    get_bounty_amount,
    send_bounty_notification,
)
from api.user.service import chutes_user_id
from api.util import notify_deleted, has_legacy_private_billing
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
    "398651e1-5f85-5e50-a513-7c5324e8e839": 20,
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
    "689d2caa-01c1-5de1-ba69-39c5398be0c6": 5,
}


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
                                Instance.created_at <= func.now() - timedelta(hours=2, minutes=0),
                            ),
                            and_(
                                Instance.config_id.is_(None),
                                Instance.created_at <= func.now() - timedelta(hours=2, minutes=0),
                            ),
                        ),
                    ),
                    and_(
                        Instance.verified.is_(True),
                        Instance.active.is_(False),
                        Instance.config_id.isnot(None),
                        LaunchConfig.verified_at <= func.now() - timedelta(hours=2, minutes=0),
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

    to_downsize = []
    scale_up_candidates = []
    chute_actions = {}
    chute_target_counts = {}

    # Also need to check which chutes are being updated.
    async with get_session() as session:
        result = await session.execute(
            text("""
                SELECT
                    c.chute_id,
                    c.public,
                    c.name,
                    c.user_id,
                    c.created_at,
                    c.concurrency,
                    MAX(COALESCE(ucb.effective_balance, 0)) AS user_balance,
                    COALESCE(c.max_instances, 1) AS max_instances,
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
        chute_info = {row.chute_id: row for row in result}
        db_now = (
            next(iter(chute_info.values())).db_now if chute_info else datetime.now(timezone.utc)
        )

    # Analyze each chute.
    for chute_id, metrics in chute_metrics.items():
        info = chute_info.get(chute_id)
        if not info:
            logger.warning(f"Chute {chute_id} found in metrics but not in chute_info query")
            # Set default target count for chutes not found in query
            chute_target_counts[chute_id] = UNDERUTILIZED_CAP
            continue

        # Private chute handling, quite different...
        if (
            info
            and not info.public
            and not has_legacy_private_billing(info)
            and info.user_id != await chutes_user_id()
        ):
            # User has no balance, can't scale up of course.
            if info.user_balance <= 0:
                await settings.redis_client.set(f"scale:{chute_id}", 0)
                chute_target_counts[chute_id] = 0
                chute_actions[chute_id] = "no_action"
                logger.info(f"Private chute {chute_id=} has no balance, unable to scale.")
                continue

            # Private chutes that do already have instances.
            if info.instance_count:
                utilization_5m = metrics["utilization"].get("5m", 0)
                utilization_15m = metrics["utilization"].get("15m", 0)
                utilization_basis = max(utilization_5m, utilization_15m)

                # Need to scale up?
                threshold = info.scaling_threshold or 0.75
                if utilization_basis >= threshold and info.instance_count < info.max_instances:
                    await settings.redis_client.set(f"scale:{chute_id}", info.instance_count + 1)
                    chute_target_counts[chute_id] = info.instance_count + 1
                    chute_actions[chute_id] = "scale_up_candidate"
                    logger.info(
                        f"Private chute {chute_id=} has reached {utilization_basis=} with {info.max_instances=}, adding capacity"
                    )
                    if await create_bounty_if_not_exists(chute_id, lifetime=3600):
                        logger.success(
                            f"Successfully created additional bounty for private chute {chute_id=}"
                        )
                    amount = await get_bounty_amount(chute_id)
                    if amount:
                        logger.info(f"Bounty for {chute_id=} is now {amount}")
                        await send_bounty_notification(chute_id, amount)
                    continue

                # Need to scale down?
                elif utilization_basis < threshold and info.instance_count > 1:
                    await settings.redis_client.set(f"scale:{chute_id}", info.instance_count - 1)
                    chute_target_counts[chute_id] = info.instance_count - 1
                    chute_actions[chute_id] = "scaled_down"
                    logger.info(
                        f"Private chute {chute_id=} has fallen to {utilization_basis=} with {info.max_instances=}, removing instance"
                    )
                    to_downsize.append((chute_id, 1))
                    continue

            # No instances, but a bounty exists so we allow one instance.
            elif await check_bounty_exists(chute_id):
                await settings.redis_client.set(f"scale:{chute_id}", 1)
                chute_target_counts[chute_id] = 1
                chute_actions[chute_id] = "scale_up_candidate"
                logger.info(f"Private chute {chute_id=} has an active bounty, adding capacity")
                continue

            # No bounty, no usage, disallow.
            else:
                await settings.redis_client.set(f"scale:{chute_id}", 0)
                chute_target_counts[chute_id] = 0
                chute_actions[chute_id] = "no_action"
                logger.info(f"Private chute {chute_id=} has no bounty, not scalable.")
                continue

            # Default, do nothing.
            await settings.redis_client.set(f"scale:{chute_id}", info.instance_count)
            chute_target_counts[chute_id] = info.instance_count
            chute_actions[chute_id] = "no_action"
            logger.info(f"Private chute {chute_id=} has expected capacity, no-op.")
            continue

        if not info or not info.instance_count:
            # Check if there's a failsafe minimum for this chute
            failsafe_min = FAILSAFE.get(chute_id, UNDERUTILIZED_CAP)
            target_count = max(UNDERUTILIZED_CAP, failsafe_min)
            await settings.redis_client.set(f"scale:{chute_id}", target_count, ex=3700)
            chute_actions[chute_id] = "scale_up_candidate"
            chute_target_counts[chute_id] = target_count
            scale_up_candidates.append((chute_id, target_count))
            logger.info(
                f"Scale up candidate: {chute_id} - no instances for past hour! Target: {target_count}"
            )
            continue

        # XXX Manual configurations, just in case, e.g. here kimi-k2-tools on vllm with b200s.
        if chute_id in LIMIT_OVERRIDES or (
            info.public and info.max_instances and info.max_instances > 1
        ):
            limit = LIMIT_OVERRIDES.get(chute_id, info.max_instances)
            logger.warning(f"Setting manual override value to {chute_id=}: {limit=}")
            await settings.redis_client.set(f"scale:{chute_id}", limit, ex=3700)
            chute_target_counts[chute_id] = limit
            if info.instance_count < limit:
                scale_up_candidates.append((chute_id, limit - info.instance_count))
                chute_actions[chute_id] = "scale_up_candidate"
                continue
            elif info.instance_count > limit:
                num_to_remove = info.instance_count - limit
                to_downsize.append((chute_id, num_to_remove))
                chute_actions[chute_id] = "scaled_down"
                chute_target_counts[chute_id] = limit
                logger.info(
                    f"Scale down candidate: {chute_id} - manual limit override, "
                    f"instances: {info.instance_count} - target: {limit}"
                )
                continue

        # Check scale up conditions
        rate_limit_5m = metrics["rate_limit_ratio"].get("5m", 0)
        rate_limit_15m = metrics["rate_limit_ratio"].get("15m", 0)
        rate_limit_1h = metrics["rate_limit_ratio"].get("1h", 0)
        utilization_15m = metrics["utilization"].get("15m", 0)
        utilization_5m = metrics["utilization"].get("5m", 0)
        rate_limit_basis = max(rate_limit_15m, rate_limit_5m)
        utilization_basis = max(utilization_15m, utilization_5m)

        # Scale up candidate: high utilization
        threshold = info.scaling_threshold or UTILIZATION_SCALE_UP
        if utilization_basis >= threshold:
            num_to_add = 1
            if utilization_basis >= 0.85:
                num_to_add = max(5, int(info.instance_count * 0.8))
            elif utilization_basis >= threshold * 1.5:
                num_to_add = max(3, int(info.instance_count * 0.5))
            elif utilization_basis >= threshold * 1.25:
                num_to_add = max(2, int(info.instance_count * 0.25))
            target_count = max(FAILSAFE.get(chute_id, 0), info.instance_count + num_to_add)
            scale_up_candidates.append((chute_id, num_to_add))
            await settings.redis_client.set(f"scale:{chute_id}", target_count, ex=3700)
            chute_actions[chute_id] = "scale_up_candidate"
            chute_target_counts[chute_id] = target_count
            logger.info(
                f"Scale up candidate: {chute_id} - high utilization: {utilization_basis:.1%} "
                f"- allowing {num_to_add} additional instances, {target_count=}"
            )
        # Scale up candidate: increasing rate limiting and significant rate limiting
        elif rate_limit_basis >= RATE_LIMIT_SCALE_UP:
            num_to_add = 1
            if rate_limit_basis >= 0.2:
                num_to_add = max(3, int(info.instance_count * 0.3))
            elif rate_limit_basis >= 0.1:
                num_to_add = max(2, int(info.instance_count * 0.15))
            else:
                num_to_add = max(1, int(info.instance_count * 0.05))
            target_count = max(FAILSAFE.get(chute_id, 0), info.instance_count + num_to_add)
            scale_up_candidates.append((chute_id, num_to_add))
            chute_actions[chute_id] = "scale_up_candidate"
            chute_target_counts[chute_id] = target_count
            await settings.redis_client.set(f"scale:{chute_id}", target_count, ex=3700)
            logger.info(
                f"Scale up candidate: {chute_id} - rate limiting increasing: "
                f"5m={rate_limit_5m:.1%}, 15m={rate_limit_15m:.1%}, 1h={rate_limit_1h:.1%} "
                f"- allowing {num_to_add} additional instances, {target_count=}"
            )

        # Scale down candidate: low utilization, no rate limiting, and has enough instances
        elif (
            info.instance_count >= UNDERUTILIZED_CAP
            and utilization_basis < threshold
            and rate_limit_5m == 0
            and rate_limit_15m == 0
            and rate_limit_1h == 0
            and not info.new_chute
            and chute_id not in LIMIT_OVERRIDES
        ):
            num_to_remove = 1
            # Calculate the number of instances to remove, based on how far away the
            # current utilization is from the scale-up threshold.
            excess_instances = info.instance_count - UNDERUTILIZED_CAP
            if utilization_basis < threshold * 0.25:
                removal_percentage = 0.3 + (0.1 * (1 - utilization_basis / (threshold * 0.25)))
            elif utilization_basis < threshold * 0.5:
                removal_percentage = 0.2 + (0.1 * (1 - utilization_basis / (threshold * 0.5)))
            else:
                removal_percentage = 0.1 + (0.1 * (1 - utilization_basis / threshold))
            removal_percentage = min(removal_percentage, 0.4)
            num_to_remove = max(1, int(excess_instances * removal_percentage))

            # Ensure post-removal utilization stays well below scale-up threshold to prevent flapping
            # Use threshold * 0.85 as target ceiling for better hysteresis
            target_utilization = threshold * 0.85
            post_removal_count = info.instance_count - num_to_remove
            post_removal_utilization = (
                utilization_basis * info.instance_count
            ) / post_removal_count
            if post_removal_utilization > target_utilization:
                safe_count = max(
                    UNDERUTILIZED_CAP,
                    math.ceil((utilization_basis * info.instance_count) / target_utilization),
                )
                num_to_remove = max(info.instance_count - safe_count, 0)

            # Final validation - never scale down if it would trigger scale-up
            if num_to_remove > 0:
                final_utilization = (utilization_basis * info.instance_count) / (
                    info.instance_count - num_to_remove
                )
                if final_utilization >= threshold:
                    num_to_remove = 0  # Abort scale-down to prevent flapping

            # Check failsafe minimum
            failsafe_min = FAILSAFE.get(chute_id, UNDERUTILIZED_CAP)
            target_count = info.instance_count - num_to_remove

            # Ensure we don't go below failsafe minimum
            if target_count < failsafe_min:
                if info.instance_count > failsafe_min:
                    # Scale down to failsafe minimum only
                    num_to_remove = info.instance_count - failsafe_min
                    target_count = failsafe_min
                    logger.info(f"Scaling down {chute_id} to failsafe minimum: {failsafe_min}")
                else:
                    # Already at or below failsafe, don't scale down
                    num_to_remove = 0
                    target_count = info.instance_count
                    logger.info(
                        f"Chute {chute_id} already at/below failsafe minimum: {failsafe_min}"
                    )

            if num_to_remove > 0:
                await settings.redis_client.set(f"scale:{chute_id}", target_count, ex=3700)
                to_downsize.append((chute_id, num_to_remove))
                chute_actions[chute_id] = "scaled_down"
                chute_target_counts[chute_id] = target_count
                logger.info(
                    f"Scale down candidate: {chute_id} - low utilization: {utilization_basis:.1%}, "
                    f"instances: {info.instance_count} - removing {num_to_remove} instances, target: {target_count}"
                )
            else:
                chute_actions[chute_id] = "no_action"
                target_count = max(failsafe_min, info.instance_count)
                chute_target_counts[chute_id] = target_count
                await settings.redis_client.set(f"scale:{chute_id}", target_count, ex=3700)
        elif info.new_chute:
            # Allow scaling new chutes, to a point.
            failsafe_min = FAILSAFE.get(chute_id, UNDERUTILIZED_CAP)
            # For new chutes, target is the max of 10, current count, or failsafe
            target_count = failsafe_min
            if "affine" not in info.name.lower():
                target_count = max(10, failsafe_min)
            num_to_add = max(0, target_count - info.instance_count)
            await settings.redis_client.set(f"scale:{chute_id}", target_count, ex=3700)
            chute_target_counts[chute_id] = target_count
            if num_to_add >= 1:
                scale_up_candidates.append((chute_id, num_to_add))
                chute_actions[chute_id] = "scale_up_candidate"
            elif info.instance_count > target_count:
                num_to_remove = info.instance_count - target_count
                to_downsize.append((chute_id, num_to_remove))
                chute_actions[chute_id] = "scaled_down"
                logger.info(
                    f"Scale down candidate: {chute_id} - new chute override "
                    f"instances: {info.instance_count} - removing {num_to_remove} instances to target: {target_count}"
                )
            else:
                chute_actions[chute_id] = "no_action"
        else:
            # Nothing to do.
            failsafe_min = FAILSAFE.get(chute_id, UNDERUTILIZED_CAP)
            target_count = max(failsafe_min, info.instance_count)
            await settings.redis_client.set(f"scale:{chute_id}", target_count, ex=3700)
            chute_actions[chute_id] = "no_action"
            chute_target_counts[chute_id] = target_count

    # Log all metrics and actions
    await log_capacity_metrics(chute_metrics, chute_actions, chute_target_counts)

    logger.success(
        f"Found {len(scale_up_candidates)} scale up candidates and {len(to_downsize)} scale down candidates"
    )

    # Don't do any actual downscaling in dry-run mode.
    if dry_run and to_downsize:
        logger.warning("DRY RUN MODE: Skipping actual instance removal")
        total_instances_to_remove = sum(num for _, num in to_downsize)
        logger.info(
            f"Would remove {total_instances_to_remove} instances across {len(to_downsize)} chutes:"
        )
        for chute_id, num_to_remove in to_downsize:
            logger.info(f"  - Chute {chute_id}: would remove {num_to_remove} instances")
        return 0

    # Perform the actual scale downs
    instances_removed = 0
    gpus_removed = 0
    for chute_id, num_to_remove in to_downsize:
        async with get_session() as session:
            chute = (
                (
                    await session.execute(
                        select(Chute)
                        .where(Chute.chute_id == chute_id)
                        .options(selectinload(Chute.instances).selectinload(Instance.nodes))
                    )
                )
                .unique()
                .scalar_one_or_none()
            )
            if not chute:
                logger.warning(f"Chute not found: {chute_id=}")
                continue

            active = [
                inst
                for inst in chute.instances
                if inst.verified and inst.active and not inst.config.job_id
            ]
            instances = []
            for instance in active:
                if len(instance.nodes) != chute.node_selector.get("gpu_count"):
                    logger.warning(f"Bad instance? {instance.instance_id=} {instance.verified=}")
                    reason = "instance node count does not match node selector"
                    await purge(instance, reason=reason)
                    await notify_deleted(instance, message=reason)
                    await invalidate_instance_cache(
                        instance.chute_id, instance_id=instance.instance_id
                    )
                    num_to_remove -= 1
                    instances_removed += 1
                    gpus_removed += len(instance.nodes)
                else:
                    instances.append(instance)

            # Sanity check.
            if len(instances) < UNDERUTILIZED_CAP or num_to_remove <= 0:
                logger.warning(
                    f"Instance count for {chute_id=} is now below underutilized cap, skipping..."
                )
                continue

            # Calculate compatible_chute_ids once per chute (before the removal loop)
            # Get minimum GPU price for current chute
            current_chute_min_rate = float("inf")
            current_node_selector = NodeSelector(**chute.node_selector)
            current_chute_gpus = set(current_node_selector.supported_gpus)
            for gpu in current_node_selector.supported_gpus:
                if gpu in SUPPORTED_GPUS:
                    current_chute_min_rate = min(
                        current_chute_min_rate, SUPPORTED_GPUS[gpu]["hourly_rate"]
                    )

            # Get all chutes and their hardware requirements
            chutes_query = text("""
                SELECT c.chute_id, c.node_selector
                FROM chutes c
            """)
            chutes_result = await session.execute(chutes_query)

            # Find chutes that the instance's nodes could run
            compatible_chute_ids = set()
            for row in chutes_result:
                node_selector = NodeSelector(**row.node_selector)
                supported_gpus = set(node_selector.supported_gpus)
                if current_chute_gpus & supported_gpus:
                    chute_min_rate = float("inf")
                    for gpu in supported_gpus:
                        if gpu in SUPPORTED_GPUS:
                            chute_min_rate = min(chute_min_rate, SUPPORTED_GPUS[gpu]["hourly_rate"])
                    # Only compatible if this chute's min price is at least threshold of current chute's min price
                    if chute_min_rate >= (current_chute_min_rate * PRICE_COMPATIBILITY_THRESHOLD):
                        compatible_chute_ids.add(row.chute_id)
            compatible_chute_ids.add(chute_id)  # Always include current chute

            logger.info(
                f"Downsizing chute {chute_id}, current count = {len(instances)}, removing {num_to_remove} unlucky instances"
            )
            kicked = set()

            for idx in range(num_to_remove):
                unlucky_instance = None
                unlucky_reason = None
                instances = [i for i in instances if i.instance_id not in kicked]

                # Filter to only established instances (online for at least 1 hour)
                established_instances = [
                    instance
                    for instance in instances
                    if instance.active
                    and db_now.replace(tzinfo=None) - instance.activated_at.replace(tzinfo=None)
                    >= timedelta(minutes=63)
                ]
                if not established_instances:
                    logger.warning(
                        f"No established instances (>1 hour) available to remove for {chute_id=}, "
                        f"skipping removal {idx + 1} of {num_to_remove}"
                    )
                    continue
                instances = established_instances

                # Completely random instance selection to purge.
                unlucky_instance = random.choice(instances)
                unlucky_reason = (
                    f"Selected an unlucky instance at random: {chute.chute_id=} "
                    f"{unlucky_instance.instance_id=} {unlucky_instance.miner_hotkey=} "
                    f"{idx + 1} of {num_to_remove}"
                )
                logger.info(unlucky_reason)

                # Purge the unlucky one
                kicked.add(unlucky_instance.instance_id)
                await purge(unlucky_instance, reason=unlucky_reason)
                await notify_deleted(unlucky_instance, message=unlucky_reason)
                await invalidate_instance_cache(
                    unlucky_instance.chute_id, instance_id=unlucky_instance.instance_id
                )

                instances_removed += 1
                gpus_removed += len(unlucky_instance.nodes)

    if instances_removed:
        logger.success(f"Scaled down, {instances_removed=} and {gpus_removed=}")

    return instances_removed


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
