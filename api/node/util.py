"""
Utility functions for nodes.
"""

from sqlalchemy import select
from api.node.schemas import Node


async def get_node_by_id(node_id, db, hotkey):
    """
    Helper to load a node by ID.
    """
    if not node_id:
        return None
    query = (
        select(Node).where(Node.miner_hotkey == hotkey).where(Node.node_id == node_id)
    )
    result = await db.execute(query)
    return result.scalar_one_or_none()
