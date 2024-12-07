"""
Routes for instances.
"""

import orjson as json
from loguru import logger
from fastapi import APIRouter, Depends, HTTPException, status, Header
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.exc import IntegrityError
from api.database import get_db_session, generate_uuid
from api.config import settings
from api.constants import HOTKEY_HEADER
from api.node.util import get_node_by_id
from api.chute.schemas import Chute
from api.instance.schemas import InstanceArgs, Instance, instance_nodes, ActivateArgs
from api.instance.util import get_instance_by_chute_and_id
from api.user.schemas import User
from api.user.service import get_current_user
from api.metasync import get_miner_by_hotkey
from api.util import is_valid_host
from api.graval_worker import verify_instance

router = APIRouter()


@router.post("/{chute_id}/", status_code=status.HTTP_202_ACCEPTED)
async def create_instance(
    chute_id: str,
    instance_args: InstanceArgs,
    db: AsyncSession = Depends(get_db_session),
    hotkey: str | None = Header(None, alias=HOTKEY_HEADER),
    _: User = Depends(get_current_user(raise_not_found=False, registered_to=settings.netuid)),
):
    chute = (await db.execute(select(Chute).where(Chute.chute_id == chute_id))).scalar_one_or_none()
    if not chute:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Chute {chute_id} not found",
        )

    # Load the miner.
    miner = await get_miner_by_hotkey(hotkey, db)
    if not miner:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Did not find miner with {hotkey=}",
        )

    # Validate the hostname.
    if not await is_valid_host(instance_args.host):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid instance host: {instance_args.host}",
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
        active=False,
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
    gpu_type = None
    for node_id in instance_args.node_ids:
        node = await get_node_by_id(node_id, db, hotkey)
        gpu_type = node.name
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
        await db.execute(
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

    # Broadcast the event.
    try:
        await settings.redis_client.publish(
            "events",
            json.dumps(
                {
                    "reason": "instance_created",
                    "message": f"Miner {instance.miner_hotkey} has provisioned an instance of chute {chute.chute_id} on {gpu_count}x{gpu_type}",
                    "data": {
                        "chute_id": instance.chute_id,
                        "gpu_count": gpu_count,
                        "gpu_model_name": gpu_type,
                        "miner_hotkey": instance.miner_hotkey,
                    },
                }
            ).decode(),
        )
    except Exception as exc:
        logger.warning(f"Error broadcasting instance event: {exc}")

    return instance


@router.patch("/{chute_id}/{instance_id}")
async def activate_instance(
    chute_id: str,
    instance_id: str,
    args: ActivateArgs,
    db: AsyncSession = Depends(get_db_session),
    hotkey: str | None = Header(None, alias=HOTKEY_HEADER),
    _: User = Depends(get_current_user(raise_not_found=False, registered_to=settings.netuid)),
):
    if not args.active:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail='Patch endpoint only supports {"active": true} as request body.',
        )
    instance = await get_instance_by_chute_and_id(db, instance_id, chute_id, hotkey)
    if not instance.active:
        instance.active = True
    await db.commit()
    await db.refresh(instance)

    # Broadcast the event.
    try:
        await settings.redis_client.publish(
            "events",
            json.dumps(
                {
                    "reason": "instance_activated",
                    "message": f"Miner {instance.miner_hotkey} has activated instance {instance.instance_id} chute {instance.chute_id}, waiting for verification...",
                    "data": {
                        "chute_id": instance.chute_id,
                        "miner_hotkey": instance.miner_hotkey,
                    },
                }
            ).decode(),
        )
    except Exception as exc:
        logger.warning(f"Error broadcasting instance event: {exc}")

    # Kick off validation.
    await verify_instance.kiq(instance_id)

    return instance


@router.delete("/{chute_id}/{instance_id}")
async def delete_instance(
    chute_id: str,
    instance_id: str,
    db: AsyncSession = Depends(get_db_session),
    hotkey: str | None = Header(None, alias=HOTKEY_HEADER),
    _: User = Depends(get_current_user(purpose="instances", registered_to=settings.netuid)),
):
    instance = await get_instance_by_chute_and_id(db, instance_id, chute_id, hotkey)
    if not instance:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Instance with {chute_id=} {instance_id} associated with {hotkey=} not found",
        )
    await db.delete(instance)
    await db.commit()

    await settings.redis_client.publish(
        "events",
        json.dumps(
            {
                "reason": "instance_deleted",
                "message": f"Miner {instance.miner_hotkey} has deleted instance an instance of chute {chute_id}.",
                "data": {
                    "chute_id": chute_id,
                    "miner_hotkey": hotkey,
                },
            }
        ).decode(),
    )

    return {"instance_id": instance_id, "deleted": True}
