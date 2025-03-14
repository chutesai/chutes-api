"""
Helpers for invocations.
"""

import hashlib
import orjson as json
from api.gpu import COMPUTE_UNIT_PRICE_BASIS
from api.database import get_session
from api.config import settings
from sqlalchemy import text


async def gather_metrics(interval: str = "1 hour"):
    """
    Generate invocation metrics for the last (interval).
    """
    cached = await settings.memcache.get(b"miner_metrics_stream")
    if cached:
        rows = json.loads(cached)
        for item in rows:
            yield item
        return

    query = text(
        f"""
SELECT
    i.chute_id,
    current_timestamp AS end_date,
    current_timestamp - INTERVAL '{interval}' AS start_date,
    AVG(i.compute_multiplier) AS compute_multiplier,
    COUNT(DISTINCT i.parent_invocation_id) as total_invocations,
    SUM(EXTRACT(EPOCH FROM (i.completed_at - i.started_at))) AS total_compute_time,
    COUNT(CASE WHEN i.error_message IS NOT NULL THEN 1 END) AS error_count,
    COUNT(CASE WHEN i.error_message = 'RATE_LIMIT' THEN 1 END) AS rate_limit_count,
    COUNT(DISTINCT CASE WHEN inst.active AND inst.verified THEN i.instance_id END) AS instance_count
FROM invocations i
LEFT JOIN instances inst ON i.instance_id = inst.instance_id
INNER JOIN chutes c ON i.chute_id = c.chute_id
WHERE i.started_at > NOW() - INTERVAL '{interval}'
AND i.error_message IS NULL
AND i.completed_at IS NOT NULL
AND NOT EXISTS (
    SELECT 1
    FROM reports
    WHERE invocation_id = i.parent_invocation_id
    AND confirmed_at IS NOT NULL
)
GROUP BY i.chute_id"""
    )
    items = []
    async with get_session() as session:
        result = await session.stream(query)
        async for row in result:
            item = dict(row._mapping)
            item["per_second_price_usd"] = (
                float(item.pop("compute_multiplier")) * COMPUTE_UNIT_PRICE_BASIS / 3600
            )
            item["total_compute_time"] = (
                float(item["total_compute_time"]) if item.get("total_compute_time") else 0
            )
            item["total_usage_usd"] = item["per_second_price_usd"] * item["total_compute_time"]
            items.append(item)
            yield item
    await settings.memcache.set(b"miner_metrics_stream", json.dumps(items), exptime=600)


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
