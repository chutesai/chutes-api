"""
Connection count reconciliation via /_conn_stats endpoint.
Called from chute_autoscaler.py to correct drift from missed DECRs.
"""

import asyncio
from loguru import logger
from api.config import settings
from api.instance.util import load_chute_target
from api.miner_client import get as miner_get
import aiohttp

CONNECTION_EXPIRY = 3600


async def _query_conn_stats(instance) -> dict | None:
    """Query an instance's /_conn_stats endpoint for ground-truth connection info."""
    try:
        url = f"http://{instance.host}:{instance.port}/_conn_stats"
        async with miner_get(
            miner_ss58=instance.miner_hotkey,
            url=url,
            purpose="conn_stats",
            timeout=aiohttp.ClientTimeout(total=5),
        ) as resp:
            if resp.status == 200:
                return await resp.json()
    except Exception:
        pass
    return None


async def _reconcile_instance(chute_id: str, instance_id: str) -> bool:
    """Reconcile a single instance. Returns True if corrected."""
    redis_client = settings.redis_client
    instance = await load_chute_target(instance_id)
    if not instance:
        return False

    stats = await _query_conn_stats(instance)
    key = f"cc:{chute_id}:{instance_id}"

    if stats is None:
        # Instance unreachable — leave redis value as-is (fallback).
        return False

    in_flight = stats.get("in_flight")
    if in_flight is None:
        return False

    current = await redis_client.get(key)
    current = int(current or 0)

    if current != in_flight:
        await redis_client.set(key, in_flight, ex=CONNECTION_EXPIRY)
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
