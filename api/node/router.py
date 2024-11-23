"""
Routes for nodes.
"""

import asyncio
import random
from taskiq_redis.exceptions import ResultIsMissingError
from typing import Annotated
from fastapi import APIRouter, Depends, HTTPException, status, Header
from sqlalchemy import select, func, delete
from sqlalchemy.ext.asyncio import AsyncSession
from api.database import get_db_session
from api.config import settings
from api.util import is_valid_host
from api.node.schemas import Node, MultiNodeArgs
from api.node.graval import validate_gpus, broker
from api.user.schemas import User
from api.user.service import get_current_user
from api.constants import HOTKEY_HEADER

router = APIRouter()


@router.post("/", status_code=status.HTTP_202_ACCEPTED)
async def create_nodes(
    args: MultiNodeArgs,
    db: AsyncSession = Depends(get_db_session),
    hotkey: str | None = Header(None, alias=HOTKEY_HEADER),
    _: User = Depends(get_current_user(raise_not_found=False, registered_to=settings.netuid)),
):
    # If we got here, the authorization succeeded, meaning it's from a registered hotkey.
    nodes_args = args.nodes

    # Random seed.
    seed = random.randint(1, 2**63 - 1)
    nodes = []
    verified_at = func.now() if settings.skip_gpu_verification else None
    if not all(await asyncio.gather(*[is_valid_host(n.verification_host) for n in args.nodes])):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="One or more invalid verification_hosts provided.",
        )
    for node_args in nodes_args:
        node = Node(
            **{
                **node_args.dict(),
                **{"miner_hotkey": hotkey, "seed": seed, "verified_at": verified_at},
            }
        )
        db.add(node)
        nodes.append(node)
    await db.commit()
    for idx in range(len(nodes)):
        await db.refresh(nodes[idx])

    # Purge any old challenges.
    node_uuids = [node.uuid for node in args.nodes]
    await db.execute(delete(Challenge).where(Challenge.uuid.in_(node_uuids)))

    task_id = "skip"
    if not verified_at:
        task = await validate_gpus.kiq([node.uuid for node in nodes])
        task_id = f"{hotkey}::{task.task_id}"
    return {"nodes": nodes, "task_id": task_id}


@router.get("/verification_status")
async def check_verification_status(
    task_id: str,
    db: AsyncSession = Depends(get_db_session),
    hotkey: str | None = Header(None, alias=HOTKEY_HEADER),
    _: User = Depends(get_current_user(raise_not_found=False, registered_to=settings.netuid, purpose="graval")),
):
    task_parts = task_id.split("::")
    if len(task_parts) != 2 or task_parts[0] != hotkey:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="go away",
        )
    task_id = task_parts[1]
    if task_id == "skip":
        return {"status": "verified"}
    if not broker.result_backend.is_result_ready(task_id):
        return {"status": "pending"}
    try:
        result = await broker.result_backend.get_result(task_id)
    except ResultIsMissingError:
        return {"status": "pending"}
    if result.is_err:
        return {"status": "error", "error": result.error}
    success, error_message = result.return_value
    if not success:
        return {"status": "failed", "detail": error_message}
    return {"status": "verified"}


@router.delete("/{node_id}")
async def delete_node(
    node_id: str,
    db: AsyncSession = Depends(get_db_session),
    hotkey: Annotated[str | None, Header()] = None,
    _: User = Depends(
        get_current_user(purpose="nodes", raise_not_found=False, registered_to=settings.netuid)
    ),
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
