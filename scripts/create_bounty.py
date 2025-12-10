import sys
import time
import asyncio
from loguru import logger
from api.config import settings
from api.bounty.util import create_bounty_if_not_exists, get_bounty_amount, send_bounty_notification


async def create_bounty(chute_id: str):
    bounty_lifetime = 3600
    if await create_bounty_if_not_exists(chute_id, lifetime=bounty_lifetime):
        logger.success(f"Successfully created a bounty for {chute_id=}")
    amount = await get_bounty_amount(chute_id)
    if amount:
        current_time = int(time.time())
        window = current_time - (current_time % 30)
        notification_key = f"bounty_notification:{chute_id}:{window}"
        if await settings.redis_client.setnx(notification_key, b"1"):
            await settings.redis_client.expire(notification_key, 33)
            logger.info(f"Bounty for {chute_id=} is now {amount}")
            await send_bounty_notification(chute_id, amount)
    else:
        logger.warning(f"No bounty for {chute_id=}")


if __name__ == "__main__":
    asyncio.run(create_bounty(sys.argv[1]))
