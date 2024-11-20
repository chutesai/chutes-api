"""
Routes for instances.
"""

from typing import Annotated
from fastapi import APIRouter, Depends, HTTPException, status, Header
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.exc import IntegrityError
from api.database import get_db_session, generate_uuid
from api.config import settings
from api.chute.util import get_chute_by_id_or_name
from api.node.util import get_node_by_id
from api.instance.schemas import InstanceArgs, Instance, instance_nodes
from api.instance.util import get_instance_by_chute_and_id
from api.user.schemas import User
from api.user.service import get_current_user
from api.metasync import get_miner_by_hotkey

router = APIRouter()


@router.post("/{chute_id}/", status_code=status.HTTP_202_ACCEPTED)
async def create_instance(
    chute_id: str,
    instance_args: InstanceArgs,
    db: AsyncSession = Depends(get_db_session),
    hotkey: Annotated[str | None, Header()] = None,
    _: User = Depends(get_current_user(raise_not_found=False, registered_to=settings.netuid)),
):
    chute = await get_chute_by_id_or_name(chute_id)
    if not chute:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Chute {chute_id} not found",
        )

    # Load the miner.
    miner = await get_miner_by_hotkey(hotkey)
    if not miner:
        raise HTTPException(
            status_code=status.HTTP_401_FORBIDDEN,
            detail=f"Did not find miner with {hotkey=}",
        )

    # Instantiate the instance.
    instance = Instance(
        instance_id=generate_uuid(),
        host=instance_args.host,
        port=instance_args.port,
        chute_id=chute_id,
        miner_uid=miner.node_id,
        miner_hotkey=hotkey,
        miner_coldkey=miner.coldkey,
        region="n/a",
        active=True,
        verified=False,
    )
    db.add(instance)

    # Verify the GPUs are suitable.
    gpu_count = chute.node_selector.get("gpu_count", 1)
    if len(instance_args.node_ids) != gpu_count:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Chute {chute_id} requires exactly {gpu_count} GPUs.",
        )
    for node_id in instance_args.node_ids:
        node = get_node_by_id(node_id)
        if not node:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Node {node_id} not found",
            )
        if not node.is_suitable(chute):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Node {node_id} is not compatible with chute node selector!",
            )
        db.execute(
            instance_nodes.insert().values(instance_id=instance.instance_id, node_id=node_id)
        )

    # Persist, which will raise a unique constraint error when the node is already allocated.
    try:
        await db.commit()
    except IntegrityError as exc:
        await db.rollback()
        if "uq_instance_node" in str(exc):
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=f"Node {node_id} is already provisioned to another instance",
            )
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Unknown database integrity error",
        )
    await db.refresh(instance)
    return instance


@router.delete("/{chute_id}/{instance_id}")
async def delete_instance(
    chute_id: str,
    instance_id: str,
    db: AsyncSession = Depends(get_db_session),
    hotkey: Annotated[str | None, Header()] = None,
    _: User = Depends(
        get_current_user(purpose="nodes", raise_not_found=False, registered_to=settings.netuid)
    ),
):
    instance = await get_instance_by_chute_and_id(db, instance_id, chute_id, hotkey)
    if not instance:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Instance with {chute_id=} {instance_id} associated with {hotkey=} not found",
        )
    await db.delete(instance)
    await db.commit()
    return {"instance_id": instance_id, "deleted": True}
