from api.database import Base
from api.config import settings
from metasync.shared import create_metagraph_node_class
from sqlalchemy import select

MetagraphNode = create_metagraph_node_class(Base)


async def get_miner_by_hotkey(hotkey, db):
    """
    Helper to load a node by ID.
    """
    if not hotkey:
        return None
    query = select(MetagraphNode).where(MetagraphNode.hotkey == hotkey).where(MetagraphNode.netuid == settings.netuid)
    result = await db.execute(query)
    return result.scalar_one_or_none()
