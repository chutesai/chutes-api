import time
import json
from loguru import logger
from datetime import datetime, timezone
from typing import Optional
from api.config import settings
from metasync.constants import (
    BOUNTY_BOOST_MIN,
    BOUNTY_BOOST_MAX,
    BOUNTY_BOOST_RAMP_MINUTES,
)


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
        result = await settings.lite_redis_client.eval(
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


async def claim_bounty(chute_id: str) -> Optional[dict]:
    """
    Atomically claim a bounty. Returns dict with bounty info including age for boost calculation.
    """
    bounty_key = f"bounty:{chute_id}"
    try:
        bounty_data = await settings.lite_redis_client.eval(
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
        return {
            "amount": bounty_amount,
            "created_at": created_at,
            "age_seconds": seconds_elapsed,
        }
    except Exception as exc:
        logger.warning(f"Failed to claim bounty: {exc}")
    return None


def calculate_bounty_boost(age_seconds: int) -> float:
    """
    Calculate compute multiplier boost based on bounty age.

    The boost ramps up from BOUNTY_BOOST_MIN (1.5x) to BOUNTY_BOOST_MAX (4x)
    over BOUNTY_BOOST_RAMP_MINUTES (60 minutes).

    This incentivizes miners to respond to older/more urgent bounties.
    """
    age_minutes = age_seconds / 60.0

    if age_minutes <= 0:
        return BOUNTY_BOOST_MIN
    elif age_minutes >= BOUNTY_BOOST_RAMP_MINUTES:
        return BOUNTY_BOOST_MAX
    else:
        # Linear ramp from min to max
        t = age_minutes / BOUNTY_BOOST_RAMP_MINUTES
        return BOUNTY_BOOST_MIN + t * (BOUNTY_BOOST_MAX - BOUNTY_BOOST_MIN)


async def get_bounty_info(chute_id: str) -> Optional[dict]:
    """
    Get full bounty info for a chute without claiming it.

    Returns dict with:
    - amount: bounty amount (based on age)
    - boost: dynamic boost multiplier (1.5x at 0min → 4x at 60min+)
    - age_seconds: how old the bounty is
    - created_at: timestamp when bounty was created

    Returns None if no bounty exists.
    """
    bounty_key = f"bounty:{chute_id}"
    try:
        bounty_data = await settings.lite_redis_client.get(bounty_key)
        if not bounty_data:
            return None
        data = json.loads(bounty_data)
        created_at = data.get("created_at")
        if not created_at:
            return None
        age_seconds = int(time.time() - created_at)
        bounty_amount = min(3 * age_seconds + 100, 86400)
        return {
            "amount": bounty_amount,
            "boost": calculate_bounty_boost(age_seconds),
            "age_seconds": age_seconds,
            "created_at": created_at,
        }
    except Exception as exc:
        logger.warning(f"Failed to get bounty info: {exc}")
    return None


async def get_bounty_boost(chute_id: str) -> float:
    """
    Get the current bounty boost for a chute without claiming it.
    Returns the dynamic boost based on bounty age, or 1.0 if no bounty exists.
    """
    info = await get_bounty_info(chute_id)
    return info["boost"] if info else 1.0


async def check_bounty_exists(chute_id: str) -> bool:
    """
    Check if a bounty exists without claiming it.
    """
    bounty_key = f"bounty:{chute_id}"
    try:
        exists = await settings.lite_redis_client.exists(bounty_key)
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
        bounty_data = await settings.lite_redis_client.get(bounty_key)
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


async def get_bounty_amounts(chute_ids: list[str]) -> dict[str, int]:
    """
    Fetch bounty amounts for multiple chutes in a single Redis round-trip.
    """
    if not chute_ids:
        return {}

    keys = [f"bounty:{chute_id}" for chute_id in chute_ids]
    results: dict[str, int] = {}
    try:
        values = await settings.lite_redis_client.mget(*keys)
        for chute_id, bounty_data in zip(chute_ids, values):
            if not bounty_data:
                continue
            try:
                data = json.loads(bounty_data)
                created_at = data.get("created_at")
                if not created_at:
                    continue
                seconds_elapsed = int(time.time() - created_at)
                bounty_amount = min(3 * seconds_elapsed + 100, 86400)
                results[chute_id] = bounty_amount
            except (json.JSONDecodeError, KeyError) as exc:
                logger.warning(f"Failed to parse bounty data for key bounty:{chute_id}: {exc}")
                continue
    except Exception as exc:
        logger.warning(f"Failed to get bounty info for chutes: {exc}")
    return results


async def delete_bounty(chute_id: str) -> bool:
    """
    Manually delete a bounty.
    """
    bounty_key = f"bounty:{chute_id}"
    try:
        result = await settings.lite_redis_client.delete(bounty_key)
        return bool(result)
    except Exception as exc:
        logger.warning(f"Failed to delete bounty: {exc}")
    return False


async def send_bounty_notification(
    chute_id: str,
    bounty: int,
    effective_multiplier: Optional[float] = None,
    bounty_boost: Optional[float] = None,
    urgency: Optional[str] = None,
) -> None:
    """
    Send bounty notification to miners.

    Args:
        chute_id: The chute ID
        bounty: Bounty amount
        effective_multiplier: Total effective compute multiplier for this chute
        bounty_boost: Current dynamic bounty boost (1.5x-4x based on age)
        urgency: Optional urgency level ("cold", "scaling", "critical")
    """
    try:
        data = {
            "chute_id": chute_id,
            "bounty": bounty,
        }
        if effective_multiplier is not None:
            data["effective_multiplier"] = effective_multiplier
        if bounty_boost is not None:
            data["bounty_boost"] = bounty_boost
        if urgency is not None:
            data["urgency"] = urgency

        await settings.lite_redis_client.publish(
            "miner_broadcast",
            json.dumps({"reason": "bounty_change", "data": data}),
        )

        urgency_str = f" ({urgency})" if urgency else ""
        await settings.lite_redis_client.publish(
            "events",
            json.dumps(
                {
                    "reason": "bounty_change",
                    "message": f"Chute {chute_id} bounty{urgency_str}: {bounty} compute units, {effective_multiplier:.1f}x multiplier",
                    "data": data,
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
            cursor, keys = await settings.lite_redis_client.client.scan(
                cursor, match=pattern, count=1000
            )
            for key in keys:
                try:
                    bounty_data = await settings.lite_redis_client.get(key)
                    if bounty_data:
                        data = json.loads(bounty_data)
                        chute_id = data.get("chute_id")
                        created_at = data.get("created_at")
                        seconds_elapsed = int(time.time() - created_at)
                        bounty_amount = min(3 * seconds_elapsed + 100, 86400)
                        ttl = await settings.lite_redis_client.ttl(key)
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
