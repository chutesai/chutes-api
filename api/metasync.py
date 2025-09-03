from api.database import Base, get_session
from api.config import settings
from metasync.shared import create_metagraph_node_class
from metasync.constants import (
    SCORING_INTERVAL,
    UNIQUE_CHUTE_HISTORY_QUERY,
)
from sqlalchemy import select, text

MetagraphNode = create_metagraph_node_class(Base)


async def get_miner_by_hotkey(hotkey, db):
    """
    Helper to load a node by ID.
    """
    if not hotkey:
        return None
    query = (
        select(MetagraphNode)
        .where(MetagraphNode.hotkey == hotkey)
        .where(MetagraphNode.netuid == settings.netuid)
    )
    result = await db.execute(query)
    return result.scalar_one_or_none()


async def get_unique_chute_history():
    query = text(UNIQUE_CHUTE_HISTORY_QUERY.format(interval=SCORING_INTERVAL))
    values = {}
    async with get_session() as session:
        result = await session.execute(query)
        for hotkey, timepoint, count in result:
            if hotkey not in values:
                values[hotkey] = []
            values[hotkey].append({"time": timepoint, "count": count})
    return values
