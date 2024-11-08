"""
Routes for nodes.
"""

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
from api.database import get_db_session
from api.config import settings
from api.node.schemas import Node, NodeArgs
from api.user.schemas import User
from api.user.service import get_current_user

router = APIRouter()


@router.get("/")
async def dummy():
    return {"ok": True}


@router.post("/")
async def create_node(
    node_args: NodeArgs,
    db: AsyncSession = Depends(get_db_session),
    _: User = Depends(
        get_current_user(raise_not_found=False, registered_to=settings.netuid)
    ),
):
    # If we got here, the authorization succeeded, meaning it's from a registered hotkey.
    node = Node(**node_args.dict())
    db.add(node)
    await db.commit()
    await db.refresh(node)
    return node
