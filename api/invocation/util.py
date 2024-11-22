"""
Helpers for invocations.
"""

from api.database import SessionLocal
from api.payment.constants import COMPUTE_UNIT_PRICE_BASIS
from sqlalchemy import text


async def gather_metrics(interval: str = "5 minutes"):
    """
    Generate invocation metrics for the last (interval).
    """
    query = text(
        f"""
       SELECT
           chute_id,
           current_timestamp AS end_date,
           current_timestamp - INTERVAL '{interval}' AS start_date,
           AVG(compute_multiplier) AS compute_multiplier,
           COUNT(DISTINCT invocation_id) as total_invocations,
           SUM(EXTRACT(EPOCH FROM (completed_at - started_at))) AS total_compute_time,
           COUNT(CASE WHEN error_message IS NOT NULL THEN 1 END) AS error_count,
           COUNT(DISTINCT instance_id) AS instance_count
       FROM invocations
       WHERE started_at > NOW() - INTERVAL '{interval}'
       GROUP BY chute_id
   """
    )
    async with SessionLocal() as session:
        result = await session.stream(query)
        async for row in result:
            item = dict(row._mapping)
            item["per_second_price_usd"] = (
                float(item.pop("compute_multiplier")) * COMPUTE_UNIT_PRICE_BASIS / 3600
            )
            item["total_compute_time"] = float(item["total_compute_time"])
            item["total_usage_usd"] = item["per_second_price_usd"] * item["total_compute_time"]
            yield item
