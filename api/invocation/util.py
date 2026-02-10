"""
Helpers for invocations.
"""

import os
import hashlib
import aiohttp
import orjson as json
from datetime import datetime, timezone
from typing import Dict
from async_lru import alru_cache
from loguru import logger
from api.gpu import COMPUTE_UNIT_PRICE_BASIS
from api.config import settings
from api.database import get_session, get_inv_session
from api.chute.schemas import NodeSelector
from sqlalchemy import text

TOKEN_METRICS_QUERY = """
INSERT INTO vllm_metrics
SELECT * FROM get_llm_metrics('2025-01-30', DATE_TRUNC('day', NOW())::date)
ORDER BY date DESC, name;
"""

DIFFUSION_METRICS_QUERY = """
INSERT INTO diffusion_metrics
SELECT * FROM get_diffusion_metrics('2025-01-30', DATE_TRUNC('day', NOW())::date)
ORDER BY date DESC, name;
"""

PROMETHEUS_URL = os.getenv("PROMETHEUS_URL", "http://prometheus-server")


@alru_cache(maxsize=500, ttl=1200)
async def get_sponsored_chute_ids(user_id: str) -> frozenset[str]:
    """Get the set of chute IDs with active sponsorships for a given user."""
    redis_key = f"sponsored_chutes:{user_id}"
    cached = await settings.redis_client.get(redis_key)
    if cached is not None:
        return frozenset(json.loads(cached))

    query = text("""
        SELECT sc.chute_id
        FROM inference_sponsorships isp
        JOIN sponsorship_chutes sc ON isp.id = sc.sponsorship_id
        WHERE isp.user_id = :user_id
        AND isp.start_date <= CURRENT_DATE
        AND (isp.end_date IS NULL OR isp.end_date >= CURRENT_DATE)
    """)
    async with get_session(readonly=True) as session:
        result = await session.execute(query, {"user_id": user_id})
        chute_ids = frozenset(row[0] for row in result)

    await settings.redis_client.set(redis_key, json.dumps(list(chute_ids)), ex=1200)
    return chute_ids


@alru_cache(maxsize=1, ttl=1200)
async def get_all_sponsored_chute_ids() -> frozenset[str]:
    """Get the set of all chute IDs with any active sponsorship."""
    redis_key = "all_sponsored_chutes"
    cached = await settings.redis_client.get(redis_key)
    if cached is not None:
        return frozenset(json.loads(cached))

    query = text("""
        SELECT DISTINCT sc.chute_id
        FROM inference_sponsorships isp
        JOIN sponsorship_chutes sc ON isp.id = sc.sponsorship_id
        WHERE isp.start_date <= CURRENT_DATE
        AND (isp.end_date IS NULL OR isp.end_date >= CURRENT_DATE)
    """)
    async with get_session(readonly=True) as session:
        result = await session.execute(query)
        chute_ids = frozenset(row[0] for row in result)

    await settings.redis_client.set(redis_key, json.dumps(list(chute_ids)), ex=1200)
    return chute_ids


async def query_prometheus(
    queries: Dict[str, str], prometheus_url: str = PROMETHEUS_URL
) -> Dict[str, Dict[str, float]]:
    """
    Execute multiple Prometheus queries concurrently and return results keyed by chute_id.
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
            logger.warning(f"Error querying Prometheus for {name}: {e}")
            return (name, {})

    try:
        async with aiohttp.ClientSession() as session:
            import asyncio

            tasks = [query_single(session, name, query) for name, query in queries.items()]
            query_results = await asyncio.gather(*tasks)
            for name, result in query_results:
                results[name] = result
    except Exception as e:
        logger.error(f"Failed to query Prometheus: {e}")

    return results


async def gather_metrics(interval: str = "1 hour"):
    """
    Generate chute metrics from Prometheus (utilization, request counts, rate limits).
    Falls back to cached data if Prometheus is unavailable.
    """
    cache_key = f"gather_metrics_{interval}"
    cached = await settings.redis_client.get(cache_key)
    if cached:
        rows = json.loads(cached)
        for item in rows:
            yield item
        return

    # Map interval string to Prometheus duration
    interval_map = {"1 hour": "1h", "1 day": "1d", "1 week": "7d"}
    prom_interval = interval_map.get(interval, "1h")

    # Query Prometheus for metrics matching what autoscaler uses
    queries = {
        "utilization": f"avg by (chute_id) (avg_over_time(utilization[{prom_interval}]))",
        "completed": f"sum by (chute_id) (increase(requests_completed_total[{prom_interval}]))",
        "rate_limited": f"sum by (chute_id) (increase(requests_rate_limited_total[{prom_interval}]))",
        "usage_usd": f"sum by (chute_id) (increase(usage_usd_total[{prom_interval}]))",
        "compute_seconds": f"sum by (chute_id) (increase(compute_seconds_total[{prom_interval}]))",
    }

    prom_results = await query_prometheus(queries)

    # Get all chute IDs and their node_selectors from DB
    chute_data = {}
    async with get_session() as session:
        result = await session.execute(
            text(
                """
                SELECT c.chute_id, c.name, c.node_selector
                FROM chutes c
                WHERE EXISTS (
                   SELECT FROM instances i WHERE i.chute_id = c.chute_id
                )
                """
            )
        )
        for row in result:
            try:
                node_selector = NodeSelector(**row.node_selector)
                compute_multiplier = node_selector.compute_multiplier
            except Exception:
                compute_multiplier = 1.0
            chute_data[row.chute_id] = {
                "name": row.name,
                "compute_multiplier": compute_multiplier,
            }

    # Get active instance counts per chute
    instance_counts = {}
    async with get_session() as session:
        result = await session.execute(
            text(
                """
                SELECT chute_id, COUNT(*) as instance_count
                FROM instances
                WHERE active = true AND verified = true
                GROUP BY chute_id
                """
            )
        )
        for row in result:
            instance_counts[row.chute_id] = int(row.instance_count)

    # Build metrics for each chute
    items = []
    all_chute_ids = set(chute_data.keys())
    for chute_id in prom_results.get("completed", {}).keys():
        all_chute_ids.add(chute_id)

    now = datetime.now(timezone.utc)
    for chute_id in all_chute_ids:
        if chute_id not in chute_data:
            continue

        compute_multiplier = chute_data[chute_id]["compute_multiplier"]
        completed = prom_results.get("completed", {}).get(chute_id, 0)
        rate_limited = prom_results.get("rate_limited", {}).get(chute_id, 0)
        utilization = prom_results.get("utilization", {}).get(chute_id, 0)
        usage_usd = prom_results.get("usage_usd", {}).get(chute_id, 0)
        compute_seconds = prom_results.get("compute_seconds", {}).get(chute_id, 0)

        item = {
            "chute_id": chute_id,
            "end_date": now.isoformat(),
            "start_date": now.isoformat(),  # Approximate, Prometheus handles the range
            "compute_multiplier": compute_multiplier,
            "total_invocations": int(completed),
            "total_compute_time": compute_seconds,
            "error_count": 0,  # Could add error metric if available
            "rate_limit_count": int(rate_limited),
            "instance_count": instance_counts.get(chute_id, 0),
            "utilization": utilization,
            "per_second_price_usd": compute_multiplier * COMPUTE_UNIT_PRICE_BASIS / 3600,
            "total_usage_usd": usage_usd,
        }
        items.append(item)
        yield item

    if items:
        await settings.redis_client.set(cache_key, json.dumps(items), ex=120)


def get_prompt_prefix_hashes(payload: dict) -> list:
    """
    Given an LLM prompt, generate a list of prefix hashes that can be used
    in prefix-aware routing for higher KV cache hit rate. Exponential size,
    powers of 2, using only characters not tokens for performance, as well
    as md5 since collections don't really matter here, cache miss is fine.
    """
    if (prompt := payload.get("prompt")) is None:
        if (messages := payload.get("messages")) is None:
            return []
        if all([isinstance(v, dict) and isinstance(v.get("content"), str) for v in messages]):
            prompt = "".join([v["content"] for v in messages])
        else:
            return []
    if not prompt or len(prompt) <= 1024:
        return []
    size = 1024
    hashes = []
    while len(prompt) > size:
        hashes.append((size, hashlib.md5(prompt[:size].encode()).hexdigest()))
        size *= 2
    return hashes[::-1]


async def generate_invocation_history_metrics():
    """
    Generate all vllm/diffusion metrics through time.
    """
    async with get_inv_session() as session:
        await session.execute(text("TRUNCATE TABLE vllm_metrics RESTART IDENTITY"))
        await session.execute(text("TRUNCATE TABLE diffusion_metrics RESTART IDENTITY"))
        await session.execute(text(TOKEN_METRICS_QUERY))
        await session.execute(text(DIFFUSION_METRICS_QUERY))
