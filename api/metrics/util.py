"""
Connection count reconciliation via /_conn_stats endpoint.
Called from chute_autoscaler.py to correct drift from missed DECRs.

Also provides a lightweight redis-only gauge refresh loop for API pods
to keep prometheus utilization gauges current between requests.
"""

import asyncio
from loguru import logger
from api.config import settings
from api.instance.util import load_chute_target, cleanup_instance_conn_tracking
from api.miner_client import get as miner_get
from api.metrics.capacity import track_capacity

CONNECTION_EXPIRY = 3600
GAUGE_REFRESH_INTERVAL = 10  # seconds


async def _query_conn_stats(instance) -> dict | None:
    """Query an instance's /_conn_stats endpoint for ground-truth connection info."""
    try:
        async with miner_get(
            miner_ss58=instance.miner_hotkey,
            url="/_conn_stats",
            instance=instance,
            purpose="conn_stats",
            timeout=5,
        ) as resp:
            if resp.status == 200:
                return await resp.json()
    except Exception:
        pass
    return None


async def _reconcile_instance(chute_id: str, instance_id: str) -> bool:
    """Reconcile a single instance. Returns True if corrected."""
    cm_redis = settings.cm_redis_client
    instance = await load_chute_target(instance_id)
    if not instance:
        await cleanup_instance_conn_tracking(chute_id, instance_id)
        return True

    stats = await _query_conn_stats(instance)
    key = f"cc:{chute_id}:{instance_id}"

    if stats is None:
        # Instance unreachable — leave redis value as-is (fallback).
        return False

    in_flight = stats.get("in_flight")
    if in_flight is None:
        return False

    current = await cm_redis.get(key)
    current = int(current or 0)

    if current != in_flight:
        await cm_redis.set(key, in_flight, ex=CONNECTION_EXPIRY)
        return True
    return False


async def reconcile_connection_counts():
    """
    Query each active instance's /_conn_stats endpoint concurrently and SET
    the redis counter to the ground-truth in_flight value.
    Instances that time out or are unreachable keep their current redis value.
    """
    redis_client = settings.redis_client

    chute_ids = await redis_client.smembers("active_chutes")
    if not chute_ids:
        return

    # Collect all (chute_id, instance_id) pairs.
    tasks = []
    for raw_chute_id in chute_ids:
        chute_id = raw_chute_id if isinstance(raw_chute_id, str) else raw_chute_id.decode()
        try:
            instance_ids_raw = await redis_client.smembers(f"cc_inst:{chute_id}")
            if not instance_ids_raw:
                continue
            for raw_iid in instance_ids_raw:
                instance_id = raw_iid if isinstance(raw_iid, str) else raw_iid.decode()
                tasks.append(_reconcile_instance(chute_id, instance_id))
        except Exception as exc:
            logger.error(f"Failed enumerating instances for {chute_id}: {exc}")

    if not tasks:
        return

    # Run all reconciliations concurrently with a semaphore to bound concurrency.
    sem = asyncio.Semaphore(50)

    async def bounded(coro):
        async with sem:
            try:
                return await coro
            except Exception as exc:
                logger.debug(f"Reconciliation task failed: {exc}")
                return False

    results = await asyncio.gather(*[bounded(t) for t in tasks])
    reconciled = sum(1 for r in results if r)
    if reconciled:
        logger.info(f"Reconciled {reconciled}/{len(tasks)} instance connection counts")


async def _refresh_gauges_once():
    """
    Read connection counts + concurrency from Redis and update prometheus gauges.
    Enumeration keys (active_chutes, cc_inst, cc_conc) on primary redis,
    cc: counters on sharded cm_redis. No DB queries.
    """
    redis_client = settings.redis_client
    chute_ids_raw = await redis_client.smembers("active_chutes")
    if not chute_ids_raw:
        return

    for raw_cid in chute_ids_raw:
        chute_id = raw_cid if isinstance(raw_cid, str) else raw_cid.decode()
        try:
            conc_raw = await redis_client.get(f"cc_conc:{chute_id}")
            if not conc_raw:
                continue
            chute_concurrency = int(conc_raw)
            if chute_concurrency <= 0:
                continue

            instance_ids_raw = await redis_client.smembers(f"cc_inst:{chute_id}")
            if not instance_ids_raw:
                await track_capacity(chute_id, 0, chute_concurrency, instance_utilization=0.0)
                continue

            keys = []
            for raw_iid in instance_ids_raw:
                iid = raw_iid if isinstance(raw_iid, str) else raw_iid.decode()
                keys.append(f"cc:{chute_id}:{iid}")

            cm_redis = settings.cm_redis_client
            values = await cm_redis.mget(keys)
            if values is None:
                continue
            total_conns = sum(int(v or 0) for v in values)
            instance_count = len(keys)
            mean_conn = total_conns / instance_count if instance_count else 0
            util = mean_conn / chute_concurrency

            await track_capacity(chute_id, mean_conn, chute_concurrency, instance_utilization=util)
        except Exception as exc:
            logger.debug(f"Error refreshing gauge for {chute_id}: {exc}")


async def keep_gauges_fresh():
    """
    Background loop that periodically refreshes prometheus utilization gauges
    from Redis connection counts. Lightweight — no DB queries.
    """
    while True:
        try:
            await _refresh_gauges_once()
        except Exception as exc:
            logger.warning(f"Gauge refresh failed: {exc}")
        await asyncio.sleep(GAUGE_REFRESH_INTERVAL)
