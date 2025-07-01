"""
Cache warmer to avoid hammering the database on the live endpoints.
"""

import time
import asyncio
import json
from loguru import logger
from api.config import settings
from api.database import get_session
from sqlalchemy import text
import api.database.orms  # noqa
from api.chute.util import update_chute_utilization, refresh_all_llm_details
from api.invocation.util import generate_invocation_history_metrics


PAST_DAY_METRICS_QUERY = """
UPDATE chutes
SET invocation_count = COALESCE(usage_summary.total_count, 0)
FROM (
    SELECT chute_id, SUM(count) as total_count
    FROM usage_data
    WHERE bucket >= now() - interval '1 day'
    GROUP BY chute_id
) usage_summary
WHERE chutes.chute_id = usage_summary.chute_id;
"""


async def warm_up_cache():
    """
    Keep some of the DB-heavy endpoints warm in cache so API requests are always fast.
    """
    from api.miner.router import get_scores, get_stats, get_utilization, get_utilization_instances
    from api.invocation.router import get_usage

    logger.info("Warming up miner and chute usage endpoints...")
    async with get_session() as session:
        await get_stats(miner_hotkey=None, session=session, per_chute=False, request=None)
        logger.success("Warmed up stats endpoint, per_chute=False")
        await get_stats(miner_hotkey=None, session=session, per_chute=True, request=None)
        logger.success("Warmed up stats endpoint, per_chute=True")
    await get_scores(hotkey=None, request=None)
    logger.success("Warmed up scores endpoint")
    await get_utilization(hotkey=None, request=None)
    logger.success("Warmed up utilization score endpoint")
    await get_utilization_instances(hotkey=None, request=None)
    logger.success("Warmed up utilization per instance endpoint")
    await get_usage(request=None)
    logger.success("Warmed up usage metrics")
    await refresh_all_llm_details()
    logger.success("Warmed up LLM details")


async def warm_up_chute_history():
    """
    Update the miner unique chute count history endpoint.
    """
    from api.metasync import get_unique_chute_history

    logger.info("Attempting to warm up unique chute history...")
    history = None
    started_at = time.time()
    history = await get_unique_chute_history()
    for hotkey, values in history.items():
        cache_key = f"uqhist:{hotkey}".encode()
        await settings.memcache.set(cache_key, json.dumps(values).encode())

    delta = time.time() - started_at
    logger.success(
        f"Successfully warmed up unique chute history for {len(history)} hotkeys in {int(delta)} seconds."
    )


async def update_past_day_metrics():
    """
    Update the past day invocation counts for sorting.
    """
    logger.info("Updating past day metrics...")
    async with get_session() as session:
        # Set counts to 0 since not all chutes will have data from the usage_data table.
        await session.execute(text("UPDATE chutes SET invocation_count = 0"))

        # Then, update for the chutes that do have metrics.
        await session.execute(text(PAST_DAY_METRICS_QUERY))
    logger.success("Updated past day invocation metric on chutes.")


async def main():
    """
    Warm up all heavy cache endpoints/invocation count for past day.
    """
    await update_past_day_metrics()
    await warm_up_chute_history()
    await warm_up_cache()
    await generate_invocation_history_metrics()
    await update_chute_utilization()


if __name__ == "__main__":
    asyncio.run(main())
