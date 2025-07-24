"""
Scale down underutilized chutes using Prometheus metrics.
"""

import os
import asyncio
import random
from loguru import logger
from collections import defaultdict
from datetime import timedelta, datetime, timezone
from typing import Dict, Optional
import aiohttp
from sqlalchemy import (
    text,
    select,
)
from sqlalchemy.orm import selectinload
from api.database import get_session
from api.config import settings
from api.chute.schemas import Chute
from api.instance.schemas import Instance
from api.capacity_log.schemas import CapacityLog
import api.database.orms  # noqa
from watchtower import purge_and_notify
from api.constants import (
    UNDERUTILIZED_CAP,
    UTILIZATION_SCALE_UP,
    RATE_LIMIT_SCALE_UP,
)


# Constants
PROMETHEUS_URL = os.getenv("PROMETHEUS_URL", "http://prometheus-server")
MIN_CHUTES_FOR_SCALING = 10


async def query_prometheus_batch(
    queries: Dict[str, str], prometheus_url: str = PROMETHEUS_URL
) -> Dict[str, Optional[float]]:
    """
    Execute multiple Prometheus queries concurrently.
    """
    results = {}

    async def query_single(session: aiohttp.ClientSession, name: str, query: str) -> tuple:
        try:
            async with session.get(
                f"{prometheus_url}/api/v1/query", params={"query": query}
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
            logger.error(f"Error querying Prometheus for {name}: {e}")
            return (name, {})

    async with aiohttp.ClientSession() as session:
        tasks = [query_single(session, name, query) for name, query in queries.items()]
        query_results = await asyncio.gather(*tasks)
        for name, result in query_results:
            results[name] = result

    return results


async def get_all_chute_metrics() -> Dict[str, Dict]:
    """
    Get metrics for all chutes from Prometheus.
    """

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
    results = await query_prometheus_batch(queries)
    chute_metrics = defaultdict(
        lambda: {
            "utilization": {},
            "completed_requests": {},
            "rate_limited_requests": {},
            "total_requests": {},
            "rate_limit_ratio": {},
        }
    )

    # Process results
    all_chute_ids = set()
    for metric_name, chute_values in results.items():
        for chute_id, value in chute_values.items():
            all_chute_ids.add(chute_id)
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
    for chute_id in all_chute_ids:
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

    return dict(chute_metrics)


async def log_capacity_metrics(chute_metrics: Dict[str, Dict], chute_actions: Dict[str, str]):
    """
    Log all chute metrics to the capacity_log table.
    """
    async with get_session() as session:
        result = await session.execute(text("SELECT NOW()"))
        timestamp = result.scalar()

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

        # Prepare insert data
        insert_data = []
        for chute_id, metrics in chute_metrics.items():
            insert_data.append(
                {
                    "timestamp": timestamp,
                    "chute_id": chute_id,
                    "utilization_current": metrics["utilization"].get("current"),
                    "utilization_5m": metrics["utilization"].get("5m"),
                    "utilization_15m": metrics["utilization"].get("15m"),
                    "utilization_1h": metrics["utilization"].get("1h"),
                    "rate_limit_ratio_5m": metrics["rate_limit_ratio"].get("5m"),
                    "rate_limit_ratio_15m": metrics["rate_limit_ratio"].get("15m"),
                    "rate_limit_ratio_1h": metrics["rate_limit_ratio"].get("1h"),
                    "total_requests_5m": metrics["total_requests"].get("5m"),
                    "total_requests_15m": metrics["total_requests"].get("15m"),
                    "total_requests_1h": metrics["total_requests"].get("1h"),
                    "completed_requests_5m": metrics["completed_requests"].get("5m"),
                    "completed_requests_15m": metrics["completed_requests"].get("15m"),
                    "completed_requests_1h": metrics["completed_requests"].get("1h"),
                    "rate_limited_requests_5m": metrics["rate_limited_requests"].get("5m"),
                    "rate_limited_requests_15m": metrics["rate_limited_requests"].get("15m"),
                    "rate_limited_requests_1h": metrics["rate_limited_requests"].get("1h"),
                    "instance_count": instance_counts.get(chute_id, 0),
                    "action_taken": chute_actions.get(chute_id, "no_action"),
                }
            )

        # Bulk insert
        if insert_data:
            await session.execute(CapacityLog.insert(), insert_data)
            await session.commit()
            logger.info(f"Logged capacity metrics for {len(insert_data)} chutes")


async def perform_autoscale():
    """
    Gather utilization data and make decisions on scaling up/down (or nothing).
    """
    logger.info("Fetching metrics from Prometheus...")
    chute_metrics = await get_all_chute_metrics()

    # Safety check - ensure we have enough data
    if len(chute_metrics) < MIN_CHUTES_FOR_SCALING:
        logger.warning(
            f"Only found metrics for {len(chute_metrics)} chutes, need at least {MIN_CHUTES_FOR_SCALING}. Aborting."
        )
        return

    logger.info(f"Found metrics for {len(chute_metrics)} chutes")

    # Identify chutes to scale down and scale up candidates
    to_downsize = []
    scale_up_candidates = []
    chute_actions = {}

    # Also need to check which chutes are being updated.
    async with get_session() as session:
        result = await session.execute(
            text("""
                SELECT 
                    c.chute_id,
                    NOW() - c.created_at <= INTERVAL '3 hours' AS new_chute,
                    COUNT(DISTINCT i.instance_id) as instance_count,
                    EXISTS(SELECT 1 FROM rolling_updates ru WHERE ru.chute_id = c.chute_id) as has_rolling_update
                FROM chutes c
                LEFT JOIN instances i ON c.chute_id = i.chute_id AND i.verified = true AND i.active = true
                GROUP BY c.chute_id
            """)
        )
        chute_info = {row.chute_id: row for row in result}

    # Analyze each chute
    for chute_id, metrics in chute_metrics.items():
        info = chute_info.get(chute_id)
        if not info:
            logger.warning(f"No data for {chute_id=}")
            num_to_remove = 1
            if info.instance_count > UNDERUTILIZED_CAP:
                num_to_remove = max(1, int((info.instance_count - UNDERUTILIZED_CAP) * 0.5))
            if num_to_remove > 0:
                to_downsize.append((chute_id, num_to_remove))
                await settings.redis_client.set(f"scale:{chute_id}", 0, ex=600)
                chute_actions[chute_id] = "scaled_down"
                logger.info(
                    f"Scale down candidate: {chute_id} - no metrics available for past hour, "
                    f"instances: {info.instance_count} - removing {num_to_remove} instances"
                )
            continue

        # Skip if rolling update in progress
        if info.has_rolling_update:
            logger.warning(f"Skipping {chute_id=}, rolling update in progress")
            continue

        # Skip if new chute.
        if info.new_chute:
            logger.info(f"Allowing scale-up of new chute {chute_id=}")
            chute_actions[chute_id] = "scale_up_candidate"
            num_to_add = max(0, 10 - info.instance_count)
            if num_to_add >= 1:
                scale_up_candidates.append((chute_id, num_to_add))
                await settings.redis_client.set(f"scale:{chute_id}", num_to_add, ex=600)
            continue

        # Check scale up conditions
        rate_limit_5m = metrics["rate_limit_ratio"].get("5m", 0)
        rate_limit_15m = metrics["rate_limit_ratio"].get("15m", 0)
        rate_limit_1h = metrics["rate_limit_ratio"].get("1h", 0)
        utilization_1h = metrics["utilization"].get("1h", 0)

        # Scale up candidate: increasing rate limiting and significant rate limiting
        if rate_limit_5m > rate_limit_15m > rate_limit_1h and rate_limit_15m >= RATE_LIMIT_SCALE_UP:
            num_to_add = 1
            if rate_limit_15m >= 0.5:
                num_to_add = max(3, int(info.instance_count * 0.5))
            elif rate_limit_15m >= 0.3:
                num_to_add = max(2, int(info.instance_count * 0.3))
            else:
                num_to_add = max(1, int(info.instance_count * 0.2))
            scale_up_candidates.append((chute_id, num_to_add))
            chute_actions[chute_id] = "scale_up_candidate"
            await settings.redis_client.set(f"scale:{chute_id}", num_to_add, ex=600)
            logger.info(
                f"Scale up candidate: {chute_id} - rate limiting increasing: "
                f"5m={rate_limit_5m:.1%}, 15m={rate_limit_15m:.1%}, 1h={rate_limit_1h:.1%} "
                f"- allowing {num_to_add} additional instances"
            )

        # Scale up candidate: high utilization
        elif utilization_1h >= UTILIZATION_SCALE_UP:
            num_to_add = 1
            if utilization_1h >= 0.9:
                num_to_add = max(2, int(info.instance_count * 0.4))
            elif utilization_1h >= 0.8:
                num_to_add = max(1, int(info.instance_count * 0.25))
            scale_up_candidates.append((chute_id, num_to_add))
            await settings.redis_client.set(f"scale:{chute_id}", num_to_add, ex=600)
            chute_actions[chute_id] = "scale_up_candidate"
            logger.info(
                f"Scale up candidate: {chute_id} - high utilization: {utilization_1h:.1%} "
                f"- allowing {num_to_add} additional instances"
            )

        # Scale down candidate: low utilization, no rate limiting, and has enough instances
        elif (
            info.instance_count >= UNDERUTILIZED_CAP
            and utilization_1h < UTILIZATION_SCALE_UP
            and rate_limit_5m == 0
            and rate_limit_15m == 0
            and rate_limit_1h == 0
            and metrics["total_requests"].get("1h", 0) > 0
        ):
            num_to_remove = 1
            if info.instance_count > UNDERUTILIZED_CAP:
                if utilization_1h < 0.1:
                    num_to_remove = max(1, int((info.instance_count - UNDERUTILIZED_CAP) * 0.33))
                elif utilization_1h < 0.3:
                    num_to_remove = max(1, int((info.instance_count - UNDERUTILIZED_CAP) * 0.1))
            if num_to_remove > 0:
                to_downsize.append((chute_id, num_to_remove))
                await settings.redis_client.set(f"scale:{chute_id}", 0, ex=600)
                chute_actions[chute_id] = "scaled_down"
                logger.info(
                    f"Scale down candidate: {chute_id} - low utilization: {utilization_1h:.1%}, "
                    f"instances: {info.instance_count} - removing {num_to_remove} instances"
                )
        else:
            await settings.redis_client.set(f"scale:{chute_id}", 0, ex=600)

    # Log all metrics and actions
    await log_capacity_metrics(chute_metrics, chute_actions)

    logger.success(
        f"Found {len(scale_up_candidates)} scale up candidates and {len(to_downsize)} scale down candidates"
    )

    # Perform the actual scale downs
    instances_removed = 0
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

            if chute.rolling_update:
                logger.warning(f"Chute has a rolling update in progress: {chute_id=}")
                continue

            active = [inst for inst in chute.instances if inst.verified and inst.active]
            instances = []
            for instance in active:
                if len(instance.nodes) != chute.node_selector.get("gpu_count"):
                    logger.warning(f"Bad instance? {instance.instance_id=} {instance.verified=}")
                    # XXX re-enable after testing.
                    # await purge_and_notify(
                    #     instance, reason="instance node count does not match node selector"
                    # )
                    num_to_remove -= 1
                    instances_removed += 1
                else:
                    instances.append(instance)

            # Sanity check.
            if len(instances) < UNDERUTILIZED_CAP or num_to_remove <= 0:
                logger.warning(
                    f"Instance count for {chute_id=} is now below underutilized cap, skipping..."
                )
                continue

            logger.info(
                f"Downsizing chute {chute_id}, current count = {len(instances)}, removing {num_to_remove} unlucky instances"
            )
            kicked = set()
            for idx in range(num_to_remove):
                instances = [i for i in instances if i.instance_id not in kicked]

                # Kick a miner with highest instance counts, when > 1.
                unlucky_instance = None
                unlucky_reason = None
                counts = defaultdict(int)
                for instance in instances:
                    counts[instance.miner_hotkey] += 1
                max_count = max(counts.values())
                if max_count > 1:
                    max_miners = [hotkey for hotkey, count in counts.items() if count == max_count]
                    unlucky = random.choice(max_miners)
                    unlucky_instance = random.choice(
                        [instance for instance in instances if instance.miner_hotkey == unlucky]
                    )
                    unlucky_reason = (
                        "Selected an unlucky instance via miner duplicates: "
                        f"{chute.chute_id=} {unlucky_instance.instance_id=} "
                        f"{unlucky_instance.miner_hotkey=} {unlucky_instance.nodes[0].gpu_identifier=} "
                        f"{idx + 1} of {num_to_remove}"
                    )
                    logger.info(unlucky_reason)

                # Random for now, but will be maxing geographical distribution and other metrics once available.
                if not unlucky_instance:
                    established = [
                        instance
                        for instance in instances
                        if datetime.now(timezone.utc) - instance.created_at >= timedelta(hours=1)
                    ]
                    if established:
                        unlucky_instance = random.choice(established)
                        unlucky_reason = (
                            f"Selected an unlucky instance at random: {chute.chute_id=} "
                            f"{unlucky_instance.instance_id=} {unlucky_instance.miner_hotkey=} "
                            f"{idx + 1} of {num_to_remove}"
                        )
                        logger.info(unlucky_reason)

                # Purge the unlucky one.
                if unlucky_instance:
                    kicked.add(unlucky_instance.instance_id)
                    # XXX re-enable after testing.
                    # await purge_and_notify(unlucky_instance, reason=unlucky_reason)
                    instances_removed += 1

    if instances_removed:
        logger.success(f"Scaled down {instances_removed} underutilized instances")
    return instances_removed


if __name__ == "__main__":
    asyncio.run(perform_autoscale())
