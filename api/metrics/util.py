"""
Script to keep the utilization Gauges up-to-date even when traffic stops.
"""

import asyncio
import traceback
from loguru import logger
from sqlalchemy import select
import api.database.orms  # noqa
from api.chute.schemas import Chute
from api.database import get_session
from api.instance.util import get_chute_target_manager
from api.metrics.capacity import track_capacity


async def update_gauges():
    async with get_session() as session:
        chutes = (await session.execute(select(Chute))).unique().scalars().all()
        for chute in chutes:
            logger.info(f"Updating gauges for {chute.chute_id=} {chute.name=}")
            tm = await get_chute_target_manager(session, chute, no_bounty=True)
            if not tm:
                logger.warning(f"No instances for {chute.chute_id=} {chute.name=}")
                track_capacity(
                    chute.chute_id, mean_conn=0, chute_concurrency=chute.concurrency or 1
                )
                continue
            try:
                _ = await tm.get_targets()
                if tm.mean_count is not None:
                    track_capacity(
                        chute.chute_id,
                        mean_conn=tm.mean_count,
                        chute_concurrency=chute.concurrency or 1,
                    )
                    logger.success(
                        f"Updated utilization gauges with {tm.mean_count=} and {chute.concurrency=}"
                    )
                else:
                    logger.warning("Mean count is none?")
                    track_capacity(
                        chute.chute_id, mean_conn=0, chute_concurrency=chute.concurrency or 1
                    )
            except Exception as exc:
                logger.error(
                    f"Failed here updating gauges for {chute.chute_id=} and {chute.concurrency=}: : {str(exc)}"
                )


async def keep_gauges_fresh():
    while True:
        try:
            await asyncio.wait_for(update_gauges(), 120.0)
        except Exception as exc:
            logger.warning(f"Failed to update gauges: {str(exc)}\n{traceback.format_exc()}")
        await asyncio.sleep(300)


if __name__ == "__main__":
    asyncio.run(update_gauges())
