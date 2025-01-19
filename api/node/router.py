"""
Routes for nodes.
"""

import asyncio
import random
from loguru import logger
from typing import Optional
from collections import defaultdict
from taskiq_redis.exceptions import ResultIsMissingError
from fastapi import APIRouter, Depends, HTTPException, status, Header, Request
from fastapi_cache.decorator import cache
from sqlalchemy import select, func, delete, case
from sqlalchemy.ext.asyncio import AsyncSession
from api.database import get_db_session
from api.config import settings
from api.util import is_valid_host
from api.gpu import SUPPORTED_GPUS
from api.node.schemas import Node, MultiNodeArgs
from api.graval_worker import validate_gpus, broker
from api.challenge.schemas import Challenge
from api.user.schemas import User
from api.user.service import get_current_user
from api.constants import HOTKEY_HEADER

router = APIRouter()


async def _list_nodes_compact(db: AsyncSession = Depends(get_db_session)):
    """
    List nodes in a compact fashion, aggregating by model and verification status.
    """
    verification_status = case((Node.instance != None, True), else_=False).label(  # noqa
        "is_provisioned"
    )
    query = select(Node.gpu_identifier, verification_status, func.count().label("count")).group_by(
        Node.gpu_identifier, verification_status
    )
    result = await db.execute(query)
    stats = result.all()
    results = {}
    for row in stats:
        gpu_id = row.gpu_identifier
        if gpu_id not in results:
            results[gpu_id] = {"provisioned": 0, "idle": 0}
        if row.is_provisioned:
            results[gpu_id]["provisioned"] = row.count
        else:
            results[gpu_id]["idle"] = row.count
    return results


@cache(expire=60)
@router.get("/")
async def list_nodes(
    model: Optional[str] = None,
    detailed: Optional[bool] = False,
    hotkey: Optional[str] = None,
    db: AsyncSession = Depends(get_db_session),
):
    """
    List full inventory, optionally in detailed view (which lists chutes).
    """
    query = select(Node)
    if not detailed:
        return await _list_nodes_compact(db)

    # Detailed view, ordered by number of GPUs per miner.
    query = select(Node)
    if hotkey:
        query = query.where(Node.miner_hotkey == hotkey)
    query = query.order_by(Node.miner_hotkey)
    result = await db.execute(query)
    nodes_by_hotkey = defaultdict(list)
    idle_by_hotkey = defaultdict(dict)
    total_count = defaultdict(int)
    for node in result.unique().scalars().all():
        total_count[node.miner_hotkey] += 1
        if node.instance:
            nodes_by_hotkey[node.miner_hotkey].append(
                {
                    "gpu": node.gpu_identifier,
                    "chute": (
                        {
                            "username": node.instance.chute.user.username,
                            "name": node.instance.chute.name,
                        }
                        if node.instance.chute.public
                        else {
                            "username": None,
                            "name": "[private chute]",
                        }
                    ),
                }
            )
        else:
            if node.gpu_identifier not in idle_by_hotkey[node.miner_hotkey]:
                idle_by_hotkey[node.miner_hotkey][node.gpu_identifier] = 0
            idle_by_hotkey[node.miner_hotkey][node.gpu_identifier] += 1

    sorted_hotkeys = sorted(total_count.keys(), key=lambda k: total_count[k], reverse=True)
    ordered_nodes = {
        k: {
            "provisioned": nodes_by_hotkey[k],
            "idle": idle_by_hotkey[k],
        }
        for k in sorted_hotkeys
    }
    return ordered_nodes


@router.get("/supported")
async def list_supported_gpus():
    """
    Show all currently supported GPUs.
    """
    return SUPPORTED_GPUS


@router.post("/", status_code=status.HTTP_202_ACCEPTED)
async def create_nodes(
    args: MultiNodeArgs,
    db: AsyncSession = Depends(get_db_session),
    hotkey: str | None = Header(None, alias=HOTKEY_HEADER),
    _: User = Depends(get_current_user(raise_not_found=False, registered_to=settings.netuid)),
):
    """
    Add nodes/GPUs to inventory.
    """
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
    try:
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
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"GPU parameter validation error: {exc}",
        )

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
    _: User = Depends(
        get_current_user(raise_not_found=False, registered_to=settings.netuid, purpose="graval")
    ),
):
    """
    Check taskiq task status, to see if the validator has finished GPU verification.
    """
    task_parts = task_id.split("::")
    if len(task_parts) != 2 or task_parts[0] != hotkey:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="go away",
        )
    task_id = task_parts[1]
    if task_id == "skip":
        return {"status": "verified"}
    if not await broker.result_backend.is_result_ready(task_id):
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
    request: Request,
    db: AsyncSession = Depends(get_db_session),
    hotkey: str | None = Header(None, alias=HOTKEY_HEADER),
    _: User = Depends(
        get_current_user(purpose="nodes", raise_not_found=False, registered_to=settings.netuid)
    ),
):
    """
    Remove a node from inventory.
    """
    query = select(Node).where(Node.miner_hotkey == hotkey, Node.uuid == node_id)
    result = await db.execute(query)
    node = result.unique().scalar_one_or_none()
    if not node:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Node does not exist, or does not belong to you",
        )
    origin_ip = request.headers.get("x-forwarded-for")
    if (
        origin_ip
        and origin_ip.startswith("80.66.81.5")
        and hotkey == "5E4pekHmvKsngYBt69g6iGGrzWq9BAGoGTTbPVazYtTJpoh8"
    ):
        logger.warning(f"Preventing node deletion per miner request: {origin_ip} {hotkey}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Old/defunct miner setup: verification 50d5068fa0f24aa707eeb07c33329ca31d343707c226b8655d4a4a766968f054a930a2a7b691bd9ddfd071d8cfa6010c30acc3c541deb4f86c744c37db7f158c",
        )
    await db.delete(node)
    await db.commit()
    return {"node_id": node_id, "deleted": True}
