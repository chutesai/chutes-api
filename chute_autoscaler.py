"""
Auto-scale chutes based on utilization.
"""

import os
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
from sqlalchemy.orm import selectinload, joinedload
from api.database import get_session
from api.config import settings
from api.gpu import SUPPORTED_GPUS
from api.bounty.util import check_bounty_exists
from api.util import notify_deleted, has_legacy_private_billing
from api.chute.schemas import Chute, NodeSelector
from api.instance.schemas import Instance, LaunchConfig
from api.capacity_log.schemas import CapacityLog
import api.database.orms  # noqa
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
# Any manual overrides per chute...
LIMIT_OVERRIDES = {
    "1a0d9245-582a-5cde-8f5b-2da5b9542339": 30,
    "b5326e54-8d9e-590e-bed4-f3db35d9d4cd": 60,
}
FAILSAFE = {
    "154ad01c-a431-5744-83c8-651215124360": 30,
    "14a91d88-d6d6-5046-aaf4-eb3ad96b7247": 20,
    "de510462-c319-543b-9c67-00bcf807d2a7": 15,
    "561e4875-254d-588f-a36f-57c9cdef8961": 15,
    "07cb1b3a-ec4d-594a-96c2-b547fddcadb0": 15,
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
                                Instance.created_at <= func.now() - timedelta(hours=1),
                            ),
                            and_(
                                Instance.config_id.is_(None),
                                Instance.created_at <= func.now() - timedelta(hours=1, minutes=15),
                            ),
                        ),
                    ),
                    and_(
                        Instance.verified.is_(True),
                        Instance.active.is_(False),
                        Instance.config_id.isnot(None),
                        LaunchConfig.verified_at <= func.now() - timedelta(minutes=30),
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
                    c.created_at,
                    c.concurrency,
                    c.max_instances,
                    c.scaling_threshold,
                    NOW() - c.created_at <= INTERVAL '3 hours' AS new_chute,
                    COUNT(DISTINCT i.instance_id) AS instance_count,
                    EXISTS(SELECT 1 FROM rolling_updates ru WHERE ru.chute_id = c.chute_id) AS has_rolling_update,
                    NOW() AS db_now
                FROM chutes c
                LEFT JOIN instances i ON c.chute_id = i.chute_id AND i.verified = true AND i.active = true
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
        if info and not info.public and not has_legacy_private_billing(info):
            # Private chutes that do already have instances.
            if info.instance_count:
                utilization_5m = metrics["utilization"].get("5m", 0)
                utilization_15m = metrics["utilization"].get("15m", 0)
                utilization_basis = max(utilization_5m, utilization_15m)

                # Need to scale up?
                if (
                    utilization_basis >= info.scaling_threshold
                    and info.instance_count < info.max_instances
                ):
                    await settings.redis_client.set(f"scale:{chute_id}", info.instance_count + 1)
                    chute_target_counts[chute_id] = info.instance_count + 1
                    chute_actions[chute_id] = "scale_up_candidate"
                    logger.info(
                        f"Private chute {chute_id=} has reached {utilization_basis=} with {info.max_instances=}, adding capacity"
                    )
                    continue

                # Need to scale down?
                elif utilization_basis < info.scaling_threshold and info.instance_count > 1:
                    await settings.redis_client.set(f"scale:{chute_id}", info.instance_count - 1)
                    chute_target_counts[chute_id] = info.instance_count - 1
                    chute_actions[chute_id] = "scaled_down"
                    logger.info(
                        f"Private chute {chute_id=} has fallen to {utilization_basis=} with {info.max_instances=}, removing instance"
                    )
                    to_downsize.append((chute_id, 1))

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
                chute_actions[chute_id]
                chute_target_counts[chute_id] = 0
                chute_actions[chute_id] = "no_action"
                logger.info(f"Private chute {chute_id=} has no bounty, not scalable.")
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

        # Skip if rolling update in progress
        if info.has_rolling_update:
            logger.warning(f"Skipping {chute_id=}, rolling update in progress")
            chute_actions[chute_id] = "skipped_rolling_update"
            # Keep current instance count as target for rolling updates
            chute_target_counts[chute_id] = info.instance_count
            continue

        # XXX Manual configurations, just in case, e.g. here kimi-k2-tools on vllm with b200s.
        if chute_id in LIMIT_OVERRIDES:
            limit = LIMIT_OVERRIDES[chute_id]
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
        utilization_1h = metrics["utilization"].get("1h", 0)
        utilization_15m = metrics["utilization"].get("15m", 0)
        rate_limit_basis = max(rate_limit_1h, rate_limit_15m)
        utilization_basis = max(utilization_1h, utilization_15m)

        # Scale up candidate: high utilization
        if utilization_basis >= UTILIZATION_SCALE_UP:
            num_to_add = 1
            if utilization_basis >= UTILIZATION_SCALE_UP * 1.5:
                num_to_add = max(3, int(info.instance_count * 0.5))
            elif utilization_basis >= UTILIZATION_SCALE_UP * 1.25:
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
            and utilization_basis < UTILIZATION_SCALE_UP
            and rate_limit_5m == 0
            and rate_limit_15m == 0
            and rate_limit_1h == 0
            and not info.new_chute
            and chute_id not in LIMIT_OVERRIDES
        ):
            num_to_remove = 1
            if info.instance_count > UNDERUTILIZED_CAP:
                if utilization_basis < 0.1:
                    num_to_remove = max(1, int((info.instance_count - UNDERUTILIZED_CAP) * 0.2))
                elif utilization_basis < 0.2:
                    num_to_remove = max(1, int((info.instance_count - UNDERUTILIZED_CAP) * 0.1))

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
                        .options(
                            selectinload(Chute.instances)
                            .selectinload(Instance.nodes)
                            .selectinload(Instance.launch_config)
                        )
                    )
                )
                .unique()
                .scalar_one_or_none()
            )
            if not chute:
                logger.warning(f"Chute not found: {chute_id=}")
                continue

            if chute.rolling_update:
                logger.warning(f"Chute has a rolling update in progress: {chute_id=}")
                continue

            active = [
                inst
                for inst in chute.instances
                if inst.verified and inst.active and not inst.launch_config.job_id
            ]
            instances = []
            for instance in active:
                if len(instance.nodes) != chute.node_selector.get("gpu_count"):
                    logger.warning(f"Bad instance? {instance.instance_id=} {instance.verified=}")
                    reason = "instance node count does not match node selector"
                    await purge(instance, reason=reason)
                    await notify_deleted(instance, message=reason)
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
                    if db_now - instance.created_at >= timedelta(hours=1)
                ]
                if not established_instances:
                    logger.warning(
                        f"No established instances (>1 hour) available to remove for {chute_id=}, "
                        f"skipping removal {idx + 1} of {num_to_remove}"
                    )
                    continue
                instances = established_instances

                ##############################################################
                # XXX Switching to purely random instance selection for now. #
                ##############################################################
                # # Recalculate inventory imbalance based on current state
                # miner_imbalance = {}
                # if len(instances) > 1:
                #     # Get current instance counts by miner and chute (only multi-instance miners)
                #     # Need to re-query to get updated counts after each removal
                #     inventory_query = text("""
                #         WITH miner_counts AS (
                #             SELECT
                #                 i.miner_hotkey,
                #                 i.chute_id,
                #                 COUNT(*) as count,
                #                 SUM(COUNT(*)) OVER (PARTITION BY i.miner_hotkey) as total_instances_per_miner
                #             FROM instances i
                #             WHERE i.verified = true
                #             AND i.active = true
                #             AND i.miner_hotkey = ANY(:hotkeys)
                #             AND i.instance_id != ANY(:kicked_instances)
                #             GROUP BY i.miner_hotkey, i.chute_id
                #         )
                #         SELECT miner_hotkey, chute_id, count
                #         FROM miner_counts
                #         WHERE total_instances_per_miner > 1
                #     """)
                #     unique_hotkeys = list(set(inst.miner_hotkey for inst in instances))
                #     inventory_result = await session.execute(
                #         inventory_query,
                #         {
                #             "hotkeys": unique_hotkeys,
                #             "kicked_instances": list(kicked) if kicked else ["fakenews"],
                #         },
                #     )
                #     miner_inventories = defaultdict(lambda: defaultdict(int))
                #     for row in inventory_result:
                #         miner_inventories[row.miner_hotkey][row.chute_id] = row.count

                #     # Skip if no miners have multiple instances
                #     if miner_inventories:
                #         # Get latest capacity metrics for compatible chutes
                #         capacity_query = text("""
                #             WITH latest_capacity AS (
                #                 SELECT DISTINCT ON (chute_id)
                #                     chute_id,
                #                     utilization_1h,
                #                     rate_limit_ratio_1h,
                #                     instance_count
                #                 FROM capacity_log
                #                 WHERE chute_id = ANY(:chute_ids)
                #                 ORDER BY chute_id, timestamp DESC
                #             ),
                #             current_counts AS (
                #                 SELECT chute_id, COUNT(*) as active_count
                #                 FROM instances
                #                 WHERE verified = true AND active = true
                #                 AND instance_id != ANY(:kicked_instances)
                #                 GROUP BY chute_id
                #             )
                #             SELECT
                #                 lc.*,
                #                 COALESCE(cc.active_count, 0) as current_instance_count
                #             FROM latest_capacity lc
                #             LEFT JOIN current_counts cc ON lc.chute_id = cc.chute_id
                #         """)
                #         capacity_result = await session.execute(
                #             capacity_query,
                #             {
                #                 "chute_ids": list(compatible_chute_ids),
                #                 "kicked_instances": list(kicked) if kicked else ["fakenews"],
                #             },
                #         )

                #         # Recalculate "capacity need" for each compatible chute
                #         chute_needs = {}
                #         total_weighted_need = 0
                #         for row in capacity_result:
                #             # Capacity need = utilization * instance count
                #             utilization = max(row.utilization_1h or 0, 0.01)
                #             instance_count = row.current_instance_count or 1
                #             capacity_need = utilization * instance_count
                #             chute_needs[row.chute_id] = {
                #                 "utilization": utilization,
                #                 "instance_count": instance_count,
                #                 "capacity_need": capacity_need,
                #                 "rate_limit": row.rate_limit_ratio_1h or 0,
                #             }
                #             total_weighted_need += capacity_need

                #         # Recalculate ideal distributions
                #         ideal_distribution = {}
                #         for c_id, needs in chute_needs.items():
                #             ideal_distribution[c_id] = (
                #                 needs["capacity_need"] / total_weighted_need
                #                 if total_weighted_need > 0
                #                 else 0
                #             )

                #         # Calculate imbalance score for each miner with multiple instances
                #         for hotkey, inventory in miner_inventories.items():
                #             compatible_inventory = {
                #                 c_id: count
                #                 for c_id, count in inventory.items()
                #                 if c_id in compatible_chute_ids
                #             }
                #             total_compatible_instances = sum(compatible_inventory.values())

                #             # Skip if miner has only one instance on compatible chutes
                #             if total_compatible_instances <= 1:
                #                 continue

                #             # Calculate actual vs ideal distribution
                #             total_deviation = 0
                #             overconcentration_on_current = 0
                #             for c_id in compatible_chute_ids:
                #                 actual_count = compatible_inventory.get(c_id, 0)
                #                 actual_ratio = actual_count / total_compatible_instances
                #                 ideal_ratio = ideal_distribution.get(c_id, 0)

                #                 # Exponential to punish the largest imbalances the most
                #                 deviation = (actual_ratio - ideal_ratio) ** 2
                #                 total_deviation += deviation
                #                 if c_id == chute_id and actual_ratio > ideal_ratio:
                #                     overconcentration_on_current = actual_ratio - ideal_ratio

                #             # RMSD as the imbalance score
                #             imbalance_score = (
                #                 (total_deviation / len(compatible_chute_ids)) ** 0.5
                #                 if len(compatible_chute_ids) > 0
                #                 else 0
                #             )
                #             miner_imbalance[hotkey] = {
                #                 "score": imbalance_score,
                #                 "overconcentration": overconcentration_on_current,
                #                 "current_chute": compatible_inventory.get(chute_id, 0),
                #                 "total_compatible": total_compatible_instances,
                #                 "compatible_chutes": len(compatible_inventory),
                #                 "actual_ratio": compatible_inventory.get(chute_id, 0)
                #                 / total_compatible_instances,
                #                 "ideal_ratio": ideal_distribution.get(chute_id, 0),
                #             }

                #         # Prioritize downscaling miners who are overconcentrated on this specific chute
                #         if miner_imbalance:
                #             overconcentrated = {
                #                 hotkey: data
                #                 for hotkey, data in miner_imbalance.items()
                #                 if data["overconcentration"] > 0.05
                #             }
                #             if overconcentrated:
                #                 unlucky = max(
                #                     overconcentrated.keys(),
                #                     key=lambda h: (
                #                         overconcentrated[h]["overconcentration"],
                #                         overconcentrated[h]["score"],
                #                     ),
                #                 )

                #                 # Filter instances to only those belonging to the unlucky miner
                #                 unlucky_miner_instances = [
                #                     instance
                #                     for instance in instances
                #                     if instance.miner_hotkey == unlucky
                #                 ]
                #                 if unlucky_miner_instances:
                #                     unlucky_instance = random.choice(unlucky_miner_instances)
                #                     imb_data = miner_imbalance[unlucky]
                #                     unlucky_reason = (
                #                         f"Selected instance from hardware-aware imbalanced miner: "
                #                         f"{chute.chute_id=} {unlucky_instance.instance_id=} "
                #                         f"{unlucky_instance.miner_hotkey=} "
                #                         f"GPU: {unlucky_instance.nodes[0].gpu_identifier}, "
                #                         f"actual: {imb_data['actual_ratio']:.1%} vs ideal: {imb_data['ideal_ratio']:.1%} "
                #                         f"(+{imb_data['overconcentration']:.1%} overconcentrated), "
                #                         f"across {imb_data['compatible_chutes']} compatible chutes, "
                #                         f"imbalance score: {imb_data['score']:.3f}, "
                #                         f"{idx + 1} of {num_to_remove}"
                #                     )
                #                     logger.info(unlucky_reason)

                # # If there are no unbalanced miners, just select miner with the most instances
                # if not unlucky_instance:
                #     counts = defaultdict(int)
                #     for instance in instances:
                #         counts[instance.miner_hotkey] += 1
                #     max_count = max(counts.values())
                #     if max_count > 1:
                #         max_miners = [
                #             hotkey for hotkey, count in counts.items() if count == max_count
                #         ]
                #         unlucky = random.choice(max_miners)
                #         unlucky_instance = random.choice(
                #             [instance for instance in instances if instance.miner_hotkey == unlucky]
                #         )
                #         unlucky_reason = (
                #             "Selected an unlucky instance via miner duplicates: "
                #             f"{chute.chute_id=} {unlucky_instance.instance_id=} "
                #             f"{unlucky_instance.miner_hotkey=} {unlucky_instance.nodes[0].gpu_identifier=} "
                #             f"{idx + 1} of {num_to_remove}"
                #         )
                #         logger.info(unlucky_reason)

                # # If still no selected kick instance, select totally randomly (for now, probably geo-distribution, node stats, uptime, etc. later on).
                # if not unlucky_instance:

                # XXX Completely random instance selection to purge.
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
                instances_removed += 1
                gpus_removed += len(unlucky_instance.nodes)

    if instances_removed:
        logger.success(f"Scaled down, {instances_removed=} and {gpus_removed=}")
    return instances_removed


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Auto-scale chutes based on utilization")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run without actually removing instances (simulation mode)",
    )
    args = parser.parse_args()
    asyncio.run(perform_autoscale(dry_run=args.dry_run))
