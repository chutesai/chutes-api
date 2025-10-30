from loguru import logger
from api.metasync import get_miner_by_hotkey
from api.config import settings


async def is_miner_blacklisted(db, hotkey):
    reason = None
    mgnode = await get_miner_by_hotkey(hotkey, db)
    if not mgnode:
        reason = f"Your hotkey is not registered on {settings.netuid}"
    elif mgnode.blacklist_reason:
        logger.warning(f"MINERBLACKLIST: {hotkey=} reason={mgnode.blacklist_reason}")
        reason = f"Your hotkey has been blacklisted: {mgnode.blacklist_reason}"
    return reason