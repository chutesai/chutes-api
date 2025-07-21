import json
import asyncio
from loguru import logger
from sqlalchemy import text, select
from sqlalchemy.orm import selectinload
import api.database.orms  # noqa
from api.config import settings
from api.database import get_session
from api.chute.schemas import Chute


QUERY = """
WITH verified_deployments AS (
    SELECT DISTINCT chute_id, version
    FROM instance_audit
    WHERE verified_at IS NOT NULL
),
miner_counts AS (
    SELECT
        chute_id,
        version,
        COUNT(DISTINCT miner_hotkey) as miner_count
    FROM instance_audit
    WHERE created_at < NOW() - INTERVAL '90 minutes'
    GROUP BY chute_id, version
),
active_instances AS (
    SELECT DISTINCT instance_id
    FROM instances
)
SELECT
    c.name,
    c.chute_id,
    c.version,
    c.created_at,
    TRUE as never_deployed,
    mc.miner_count >= 5 as has_five_miners
FROM chutes c
LEFT JOIN verified_deployments vd
    ON c.chute_id = vd.chute_id
    AND c.version = vd.version
LEFT JOIN miner_counts mc
    ON c.chute_id = mc.chute_id
    AND c.version = mc.version
LEFT JOIN active_instances ai
    ON c.chute_id = ai.instance_id
WHERE (c.jobs IS NULL OR c.jobs = '{}' OR c.jobs = '[]')
    AND vd.chute_id IS NULL
    AND ai.instance_id IS NULL
    AND COALESCE(mc.miner_count, 0) >= 5;
"""


async def clean_failed_chutes():
    """
    Find chutes that were attempted to be deployed by at least 5 miners
    without success and are at least 90 minutes old. These will never
    work and should be culled.
    """
    to_broadcast = []
    async with get_session() as session:
        result = await session.execute(text(QUERY))
        for name, chute_id, version, created_at, _, __ in result:
            chute = (
                (
                    await session.execute(
                        select(Chute)
                        .where(Chute.chute_id == chute_id, Chute.version == version)
                        .options(selectinload(Chute.instances))
                    )
                )
                .unique()
                .scalar_one_or_none()
            )
            if not chute:
                continue  # how would it possibly get here?
            if chute.instances:
                logger.warning(
                    f"{chute.chute_id=} {chute.name=} has instances, refusing to clean up..."
                )
                continue
            logger.warning(
                f"Chute {name} {chute_id} created {created_at} failed to deploy, wiping..."
            )
            to_broadcast.append(
                {
                    "chute_id": chute.chute_id,
                    "version": chute.version,
                }
            )
            await session.delete(chute)
        await session.commit()

    for data in to_broadcast:
        await settings.redis_client.publish(
            "miner_broadcast",
            json.dumps(
                {
                    "reason": "chute_deleted",
                    "data": data,
                }
            ),
        )
    if to_broadcast:
        logger.success(f"Successfully purged {len(to_broadcast)} chutes that failed to deploy.")


if __name__ == "__main__":
    asyncio.run(clean_failed_chutes())
