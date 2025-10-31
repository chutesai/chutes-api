"""
Utility functions for nodes.
"""

from typing import Optional
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from api.gpu import SUPPORTED_GPUS
from api.node.schemas import Node, NodeArgs


async def get_node_by_id(node_id, db, hotkey):
    """
    Helper to load a node by ID.
    """
    if not node_id:
        return None
    query = select(Node).where(Node.miner_hotkey == hotkey).where(Node.uuid == node_id)
    result = await db.execute(query)
    return result.unique().scalar_one_or_none()

async def _track_nodes(
    db: AsyncSession, hotkey: str, server_id: str, nodes_args: list[NodeArgs], 
    seed: str, verified_at = None
):
    nodes = []
    for node_args in nodes_args:
        node = Node(
            **{
                **node_args.model_dump(),
                **{
                    "miner_hotkey": hotkey,
                    "seed": seed,
                    "verified_at": verified_at,
                    "server_id": server_id
                },
            }
        )
        # Legacy flags for backwards graval compatibility.
        gpu_info = SUPPORTED_GPUS[node.gpu_identifier]
        if "major" in gpu_info:
            for key in ["major", "minor", "tensor_cores", "concurrent_kernels", "ecc", "sxm"]:
                setattr(node, key, gpu_info.get(key))
        db.add(node)
        nodes.append(node)
    await db.commit()
    for idx in range(len(nodes)):
        await db.refresh(nodes[idx])

    return nodes

async def check_node_inventory(db, node_uuids):
    # Check if any of the nodes are already in inventory.
    existing_uuids = []
    existing = (
        (await db.execute(select(Node).where(Node.uuid.in_(node_uuids)))).unique().scalars().all()
    )
    if existing:
        existing_uuids = [node.uuid for node in existing]

    return existing_uuids