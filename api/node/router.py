"""
Routes for nodes.
"""

from typing import Annotated
from fastapi import APIRouter, Depends, HTTPException, status, Header
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from api.database import get_db_session
from api.config import settings
from api.node.schemas import Node, NodeArgs
from api.user.schemas import User
from api.user.service import get_current_user

router = APIRouter()


@router.post("/")
async def create_node(
    node_args: NodeArgs,
    db: AsyncSession = Depends(get_db_session),
    hotkey: Annotated[str | None, Header()] = None,
    _: User = Depends(get_current_user(raise_not_found=False, registered_to=settings.netuid)),
):
    # If we got here, the authorization succeeded, meaning it's from a registered hotkey.
    node = Node(**{**node_args.dict(), **{"miner_hotkey": hotkey}})
    db.add(node)
    await db.commit()
    await db.refresh(node)
    return node


@router.delete("/{node_id}")
async def delete_node(
    node_id: str,
    db: AsyncSession = Depends(get_db_session),
    hotkey: Annotated[str | None, Header()] = None,
    _: User = Depends(get_current_user(purpose="nodes", raise_not_found=False, registered_to=settings.netuid)),
):
    query = select(Node).where(Node.miner_hotkey == hotkey).where(Node.uuid == node_id)
    result = await db.execute(query)
    node = result.scalar_one_or_none()
    if not node:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Node does not exist, or does not belong to you",
        )
    await db.delete(node)
    await db.commit()
    return {"node_id": node_id, "deleted": True}
