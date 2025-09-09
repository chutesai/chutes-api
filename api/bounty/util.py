import time
import json
from loguru import logger
from datetime import datetime, timezone
from typing import Optional
from api.config import settings


CLAIM_BOUNTY_LUA = """
local bounty_key = KEYS[1]
local bounty_data = redis.call('GET', bounty_key)
if bounty_data then
    redis.call('DEL', bounty_key)
    return bounty_data
else
    return nil
end
"""

CREATE_BOUNTY_LUA = """
local bounty_key = KEYS[1]
local bounty_data = ARGV[1]
local expire_time = ARGV[2]
if redis.call('EXISTS', bounty_key) == 0 then
    redis.call('SET', bounty_key, bounty_data, 'EX', expire_time)
    return 1
else
    return 0
end
"""


async def create_bounty_if_not_exists(chute_id: str, lifetime: int = 86400) -> bool:
    """
    Create a bounty timestamp if one doesn't already exist.
    """
    bounty_key = f"bounty:{chute_id}"
    bounty_data = {
        "created_at": datetime.now(timezone.utc).timestamp(),
        "chute_id": chute_id,
    }
    try:
        result = await settings.redis_client.eval(
            CREATE_BOUNTY_LUA,
            1,
            bounty_key,
            json.dumps(bounty_data),
            lifetime,
        )
        return bool(result)
    except Exception as exc:
        logger.warning(f"Failed to create bounty: {exc}")
    return False


async def claim_bounty(chute_id: str) -> Optional[int]:
    """
    Atomically claim a bounty.
    """
    bounty_key = f"bounty:{chute_id}"
    try:
        bounty_data = await settings.redis_client.eval(
            CLAIM_BOUNTY_LUA,
            1,
            bounty_key,
        )
        if not bounty_data:
            return None
        data = json.loads(bounty_data)
        created_at = data["created_at"]
        seconds_elapsed = int(time.time() - created_at)
        bounty_amount = min(3 * seconds_elapsed + 100, 86400)
        return bounty_amount
    except Exception as exc:
        logger.warning(f"Failed to claim bounty: {exc}")
    return None


async def check_bounty_exists(chute_id: str) -> bool:
    """
    Check if a bounty exists without claiming it.
    """
    bounty_key = f"bounty:{chute_id}"
    try:
        exists = await settings.redis_client.exists(bounty_key)
        return bool(exists)
    except Exception as exc:
        logger.warning(f"Failed to check bounty existence: {exc}")
    return False


async def get_bounty_amount(chute_id: str) -> int:
    """
    Get bounty amount and creation time without claiming it.
    """
    bounty_key = f"bounty:{chute_id}"
    try:
        bounty_data = await settings.redis_client.get(bounty_key)
        if not bounty_data:
            return None
        data = json.loads(bounty_data)
        created_at = data["created_at"]
        seconds_elapsed = int(time.time() - created_at)
        bounty_amount = min(3 * seconds_elapsed + 100, 86400)
        return bounty_amount
    except (json.JSONDecodeError, KeyError) as exc:
        logger.warning(f"Failed to get bounty info: {exc}")
        return None
    except Exception as exc:
        logger.warning(f"Unexpected error getting bounty info: {exc}")
        return None


async def delete_bounty(chute_id: str) -> bool:
    """
    Manually delete a bounty.
    """
    bounty_key = f"bounty:{chute_id}"
    try:
        result = await settings.redis_client.delete(bounty_key)
        return bool(result)
    except Exception as exc:
        logger.warning(f"Failed to delete bounty: {exc}")
    return False


async def send_bounty_notification(chute_id: str, bounty: int) -> None:
    try:
        await settings.redis_client.publish(
            "miner_broadcast",
            json.dumps(
                {
                    "reason": "bounty_change",
                    "data": {"chute_id": chute_id, "bounty": bounty},
                }
            ),
        )
        await settings.redis_client.publish(
            "events",
            json.dumps(
                {
                    "reason": "bounty_change",
                    "message": f"Chute {chute_id} bounty has been set to {bounty} compute units.",
                    "data": {
                        "chute_id": chute_id,
                        "bounty": bounty,
                    },
                }
            ),
        )
    except Exception as exc:
        logger.error(f"Failed to send bounty notification: {exc}")


async def list_bounties() -> list[dict]:
    """
    List all available bounties with their current amounts.
    """
    bounties = []
    try:
        cursor = 0
        pattern = "bounty:*"
        while True:
            cursor, keys = await settings.redis_client.scan(cursor, match=pattern, count=100)
            for key in keys:
                try:
                    bounty_data = await settings.redis_client.get(key)
                    if bounty_data:
                        data = json.loads(bounty_data)
                        chute_id = data.get("chute_id")
                        created_at = data.get("created_at")
                        seconds_elapsed = int(time.time() - created_at)
                        bounty_amount = min(3 * seconds_elapsed, 86400)
                        ttl = await settings.redis_client.ttl(key)
                        bounties.append(
                            {
                                "chute_id": chute_id,
                                "bounty_amount": bounty_amount,
                                "seconds_elapsed": seconds_elapsed,
                                "time_remaining": ttl if ttl > 0 else 0,
                                "created_at": created_at,
                            }
                        )
                except (json.JSONDecodeError, KeyError) as exc:
                    logger.warning(f"Failed to parse bounty data for key {key}: {exc}")
                    continue
            if cursor == 0:
                break
        bounties.sort(key=lambda x: x["created_at"])
    except Exception as exc:
        logger.error(f"Failed to list bounties: {exc}")
    return bounties
