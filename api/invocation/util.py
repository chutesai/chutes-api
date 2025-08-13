"""
Helpers for invocations.
"""

import hashlib
import orjson as json
from api.gpu import COMPUTE_UNIT_PRICE_BASIS
from api.database import get_session
from api.util import memcache_get, memcache_set
from sqlalchemy import text

TOKEN_METRICS_QUERY = """
CREATE TABLE vllm_metrics_temp AS
    SELECT * FROM get_vllm_metrics('2025-01-30', DATE_TRUNC('day', NOW())::date)
    ORDER BY date DESC, name;
"""

DIFFUSION_METRICS_QUERY = """
CREATE TABLE diffusion_metrics_temp AS WITH min_date AS (
  SELECT MIN(DATE(started_at)) AS min_date
  FROM invocations
  JOIN chutes ON invocations.chute_id = chutes.chute_id
  WHERE chutes.standard_template = 'diffusion'
  AND metrics->>'steps' IS NOT NULL
),
date_series AS (
  SELECT generate_series(
    (SELECT min_date FROM min_date),
    DATE_TRUNC('day', NOW()),
    '1 day'::interval
  )::date AS date
),
all_chutes AS (
  SELECT chute_id, name
  FROM chutes
  WHERE standard_template = 'diffusion'
),
chute_dates AS (
  SELECT c.chute_id, c.name, d.date
  FROM all_chutes c
  CROSS JOIN date_series d
),
metrics_data AS (
  SELECT
    chutes.chute_id,
    chutes.name,
    DATE(started_at) AS date,
    SUM((metrics->>'steps')::float)::int AS total_steps,
    COUNT(*) AS total_requests,
    AVG((metrics->>'sps')::float) AS average_sps
  FROM invocations
  JOIN chutes ON invocations.chute_id = chutes.chute_id
  WHERE chutes.standard_template = 'diffusion'
  AND metrics->>'steps' IS NOT NULL
  AND error_message IS NULL
  AND completed_at IS NOT NULL
  GROUP BY chutes.chute_id, chutes.name, DATE(started_at)
)
SELECT
  cd.chute_id,
  cd.name,
  cd.date,
  COALESCE(md.total_steps, 0) AS total_steps,
  COALESCE(md.total_requests, 0) AS total_requests,
  COALESCE(md.average_sps, 0) AS average_sps
FROM chute_dates cd
LEFT JOIN metrics_data md ON cd.chute_id = md.chute_id AND cd.date = md.date
ORDER BY cd.date DESC, cd.name;
"""


async def gather_metrics(interval: str = "1 hour"):
    """
    Generate invocation metrics for the last (interval).
    """
    cached = await memcache_get(b"miner_metrics_stream")
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
    COUNT(DISTINCT CASE WHEN i.error_message IS NULL AND i.completed_at IS NOT NULL THEN i.parent_invocation_id END) as total_invocations,
    SUM(CASE WHEN i.error_message IS NULL AND i.completed_at IS NOT NULL THEN EXTRACT(EPOCH FROM (i.completed_at - i.started_at)) END) AS total_compute_time,
    COUNT(CASE WHEN i.error_message IS NOT NULL THEN 1 END) AS error_count,
    COUNT(CASE WHEN i.error_message = 'RATE_LIMIT' THEN 1 END) AS rate_limit_count,
    COUNT(DISTINCT CASE WHEN inst.active AND inst.verified THEN i.instance_id END) AS instance_count
FROM invocations i
LEFT JOIN instances inst ON i.instance_id = inst.instance_id
INNER JOIN chutes c ON i.chute_id = c.chute_id
WHERE i.started_at > NOW() - INTERVAL '{interval}'
AND i.completed_at IS NOT NULL
GROUP BY i.chute_id"""
    )
    items = []
    async with get_session() as session:
        result = await session.stream(query)
        async for row in result:
            item = dict(row._mapping)
            item["per_second_price_usd"] = (
                float(item["compute_multiplier"]) * COMPUTE_UNIT_PRICE_BASIS / 3600
            )
            item["total_compute_time"] = (
                float(item["total_compute_time"]) if item.get("total_compute_time") else 0
            )
            item["total_usage_usd"] = item["per_second_price_usd"] * item["total_compute_time"]
            items.append(item)
            yield item
    await memcache_set(b"miner_metrics_stream", json.dumps(items), exptime=600)


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
    async with get_session() as session:
        await session.execute(text("DROP TABLE IF EXISTS vllm_metrics_temp"))
        await session.execute(text("DROP TABLE IF EXISTS diffusion_metrics_temp"))
        await session.execute(text(TOKEN_METRICS_QUERY))
        await session.execute(text(DIFFUSION_METRICS_QUERY))
    async with get_session() as session:
        await session.execute(text("DROP TABLE IF EXISTS vllm_metrics"))
        await session.execute(text("DROP TABLE IF EXISTS diffusion_metrics"))
        await session.execute(text("ALTER TABLE vllm_metrics_temp RENAME TO vllm_metrics"))
        await session.execute(
            text("ALTER TABLE diffusion_metrics_temp RENAME to diffusion_metrics")
        )
