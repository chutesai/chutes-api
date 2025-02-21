import json
import asyncio
from loguru import logger
from sqlalchemy import text, select
import api.database.orms  # noqa
from api.config import settings
from api.database import get_session
from api.chute.schemas import Chute


QUERY = """
SELECT 
    chutes.name,
    chutes.chute_id,
    chutes.version,
    chutes.created_at,
    NOT EXISTS (
        SELECT 1 
        FROM instance_audit 
        WHERE instance_audit.chute_id = chutes.chute_id 
        AND instance_audit.version = chutes.version
        AND verified_at IS NOT NULL
    ) as never_deployed,
    (
        SELECT COUNT(DISTINCT miner_hotkey)
        FROM instance_audit
        WHERE instance_audit.chute_id = chutes.chute_id
        AND instance_audit.version = chutes.version
        AND created_at < NOW() - INTERVAL '3 hours'
    ) >= 5 as has_five_miners
FROM chutes
WHERE NOT EXISTS (
    SELECT 1 
    FROM instance_audit 
    WHERE instance_audit.chute_id = chutes.chute_id 
    AND instance_audit.version = chutes.version
    AND verified_at IS NOT NULL
)
AND (
    SELECT COUNT(DISTINCT miner_hotkey)
    FROM instance_audit
    WHERE instance_audit.chute_id = chutes.chute_id
    AND instance_audit.version = chutes.version
    AND created_at < NOW() - INTERVAL '90 minutes'
) >= 5
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
        for name, chute_id, created_at, _, __ in result:
            chute = (
                (await session.execute(select(Chute).where(Chute.chute_id == chute_id)))
                .unique()
                .scalar_one_or_none()
            )
            if not chute:
                continue  # how would it possibly get here?
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
