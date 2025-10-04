import json
import asyncio
from loguru import logger
from sqlalchemy import text, select
import api.database.orms  # noqa
from api.config import settings
from api.chute.schemas import Chute
from api.database import get_session
from api.user.service import chutes_user_id

QUERY = """
WITH deployment_attempts AS (
  SELECT instance_audit.chute_id,
    COUNT(CASE WHEN activated_at IS NOT NULL THEN 1 END) AS activated_count,
    COUNT(*) AS total_count,
    COUNT(DISTINCT(miner_hotkey)) AS hotkey_count
  FROM instance_audit
  JOIN chutes
  ON chutes.chute_id = instance_audit.chute_id AND chutes.version = instance_audit.version
  WHERE instance_audit.created_at >= NOW() - INTERVAL '1 week'
  AND instance_audit.created_at <= NOW() - INTERVAL '8 hours'
  AND NOT EXISTS (
    SELECT
    FROM instances
    WHERE chute_id = instance_audit.chute_id
    AND active
  )
  GROUP BY instance_audit.chute_id
)
SELECT chutes.chute_id, name, activated_count, total_count, hotkey_count
FROM deployment_attempts
JOIN chutes ON deployment_attempts.chute_id = chutes.chute_id
WHERE activated_count = 0
AND total_count > 20
AND hotkey_count >= 3;
"""


async def clean_failed_chutes():
    to_broadcast = []
    async with get_session() as session:
        result = await session.execute(text(QUERY))
        for chute_id, name, activated_count, total_count, hotkey_count in result:
            chute = (
                (await session.execute(select(Chute).where(Chute.chute_id == chute_id)))
                .unique()
                .scalar_one_or_none()
            )
            if not chute or chute.user_id == await chutes_user_id():
                continue
            logger.warning(
                f"Deleting chute that failed all deployment attempts: {chute_id=} {name=} {activated_count=} {total_count=} {hotkey_count=}"
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
