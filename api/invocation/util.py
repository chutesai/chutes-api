"""
Helpers for invocations.
"""

from api.database import get_session
from api.payment.constants import COMPUTE_UNIT_PRICE_BASIS
from sqlalchemy import text


async def gather_metrics(interval: str = "1 hour"):
    """
    Generate invocation metrics for the last (interval).
    """
    query = text(
        f"""
SELECT
    i.chute_id,
    current_timestamp AS end_date,
    current_timestamp - INTERVAL '{interval}' AS start_date,
    AVG(i.compute_multiplier) AS compute_multiplier,
    COUNT(DISTINCT i.invocation_id) as total_invocations,
    SUM(EXTRACT(EPOCH FROM (i.completed_at - i.started_at))) AS total_compute_time,
    COUNT(CASE WHEN i.error_message IS NOT NULL THEN 1 END) AS error_count,
    COUNT(CASE WHEN i.error_message = 'RATE_LIMIT' THEN 1 END) AS rate_limit_count,
    COUNT(DISTINCT CASE WHEN inst.active AND inst.verified THEN i.instance_id END) AS instance_count
FROM invocations i
LEFT JOIN instances inst ON i.instance_id = inst.instance_id
INNER JOIN chutes c ON i.chute_id = c.chute_id
WHERE i.started_at > NOW() - INTERVAL '{interval}'
GROUP BY i.chute_id"""
    )
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
            yield item
