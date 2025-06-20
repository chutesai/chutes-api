"""
Routes for instances.
"""

import orjson as json
import traceback
from loguru import logger
from fastapi import APIRouter, Depends, HTTPException, status, Header, Request
from sqlalchemy import select, text
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.exc import IntegrityError
from api.database import get_db_session, generate_uuid
from api.config import settings
from api.constants import HOTKEY_HEADER, EXPANSION_UTILIZATION_THRESHOLD, UNDERUTILIZED_CAP
from api.node.util import get_node_by_id
from api.chute.schemas import Chute
from api.instance.schemas import InstanceArgs, Instance, instance_nodes, ActivateArgs
from api.instance.util import get_instance_by_chute_and_id
from api.user.schemas import User
from api.user.service import get_current_user
from api.metasync import get_miner_by_hotkey
from api.util import is_valid_host, generate_ip_token
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
    mgnode = await get_miner_by_hotkey(hotkey, db)
    if mgnode.blacklist_reason:
        logger.warning(f"MINERBLACKLIST: {hotkey=} reason={mgnode.blacklist_reason}")
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=f"Your hotkey has been blacklisted: {mgnode.blacklist_reason}",
        )

    chute = (
        (await db.execute(select(Chute).where(Chute.chute_id == chute_id)))
        .unique()
        .scalar_one_or_none()
    )
    if not chute:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Chute {chute_id} not found",
        )

    # Rolling update handling.
    if chute.rolling_update:
        limit = chute.rolling_update.permitted.get(hotkey, 0)
        if not limit and chute.rolling_update.permitted:
            logger.warning(
                f"SCALELOCK: chute {chute_id=} {chute.name} is currently undergoing a rolling update"
            )
            raise HTTPException(
                status_code=status.HTTP_423_LOCKED,
                detail=f"Chute {chute_id} is currently undergoing a rolling update and you have no quota, try again later.",
            )
        if limit:
            chute.rolling_update.permitted[hotkey] -= 1

    # Limit underutilized chutes.
    try:
        query = text(
            "SELECT * FROM chute_utilization "
            "WHERE chute_id = :chute_id "
            "AND NOT EXISTS ("
            "  SELECT FROM chutes "
            "  WHERE chute_id = :chute_id "
            "  AND updated_at >= now() - INTERVAL '1 hour' "
            ")"
        )
        results = await db.execute(query, {"chute_id": chute_id})
        utilization = results.mappings().first()
        if (
            utilization
            and utilization["avg_busy_ratio"] < EXPANSION_UTILIZATION_THRESHOLD
            and not utilization["total_rate_limit_errors"]
        ):
            query = text(
                "SELECT COUNT(*) AS total_count, "
                "COUNT(CASE WHEN miner_hotkey = :hotkey THEN 1 ELSE NULL END) AS hotkey_count "
                "FROM instances WHERE chute_id = :chute_id"
            )
            count_result = (
                (await db.execute(query, {"chute_id": chute_id, "hotkey": hotkey}))
                .mappings()
                .first()
            )
            if count_result["total_count"] >= UNDERUTILIZED_CAP or count_result.hotkey_count:
                logger.warning(
                    f"SCALELOCK: chute {chute_id=} {chute.name} is currently capped: {count_result}"
                )
                raise HTTPException(
                    status_code=status.HTTP_423_LOCKED,
                    detail=f"Chute {chute_id} is underutilized and either at capacity or you already have an instance.",
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
            logger.warning(
                f"INSTANCEFAIL: Attempt to post bad host: {instance_args.host} from {hotkey=}"
            )
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
            version=chute.version,
            miner_uid=miner.node_id,
            miner_hotkey=hotkey,
            miner_coldkey=miner.coldkey,
            region="n/a",
            active=False,
            verified=False,
            chutes_version=chute.chutes_version,
        )
        db.add(instance)

        # Verify the GPUs are suitable.
        gpu_count = chute.node_selector.get("gpu_count", 1)
        if len(instance_args.node_ids) != gpu_count:
            logger.warning(
                f"INSTANCEFAIL: Attempt to post incorrect GPU count: {len(instance_args.node_ids)} vs {gpu_count} from {hotkey=}"
            )
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Chute {chute_id} requires exactly {gpu_count} GPUs.",
            )
        gpu_type = None
        node_hosts = set()
        for node_id in instance_args.node_ids:
            # Make sure the node is in the miner's inventory.
            node = await get_node_by_id(node_id, db, hotkey)
            if not node:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"Node {node_id} not found",
                )
            node_hosts.add(node.verification_host)

            # Not verified?
            if not node.verified_at:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"GPU {node_id} is not yet verified, and cannot be associated with an instance",
                )

            # Already associated with an instance?
            if node.instance:
                raise HTTPException(
                    status_code=status.HTTP_409_CONFLICT,
                    detail=f"GPU {node_id} is already assigned to an instance: {instance.instance_id=} {instance.host=} {instance.port=} {instance.chute_id=}",
                )

            # Valid GPU for this chute?
            gpu_type = node.name
            if not node.is_suitable(chute):
                logger.warning(
                    f"INSTANCEFAIL: attempt to post incompatible GPUs: {node.name} for {chute.node_selector}"
                )
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Node {node_id} is not compatible with chute node selector!",
                )

            # Create the association record.
            await db.execute(
                instance_nodes.insert().values(instance_id=instance.instance_id, node_id=node_id)
            )

        # The hostname used in verifying the node must match the hostname of the instance.
        if len(node_hosts) > 1 or list(node_hosts)[0].lower() != instance.host.lower():
            logger.warning(
                f"Instance hostname does not match the node verification hostname: {instance.host=} vs {node_hosts=}"
            )
            # XXX disable for now to allow domain-based DDoS protection.
            # raise HTTPException(
            #     status_code=status.HTTP_400_BAD_REQUEST,
            #     detail=f"Instance hostname does not match the node verification hostname: {instance.host=} vs {node_hosts=}",
            # )

        # Persist, which will raise a unique constraint error when the node is already allocated.
        await db.commit()
    except IntegrityError as exc:
        detail = f"INTEGRITYERROR {hotkey=}: {exc}\n{traceback.format_exc()}"
        logger.error(detail)
        await db.rollback()
        if "uq_inode" in str(exc):
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=f"One or more nodes already provisioned to an instance: {detail=}",
            )
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unhandled DB integrity error: {detail}",
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


@router.get("/token_check")
async def get_token(salt: str = None, request: Request = None):
    origin_ip = request.headers.get("x-forwarded-for", "").split(",")[0]
    return {"token": generate_ip_token(origin_ip, extra_salt=salt)}


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
    if not instance:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Instance not found.",
        )
    if instance.active and instance.verified:
        return instance
    elif not instance.active:
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
    if await settings.redis_client.get(f"verify:lock:{instance_id}"):
        logger.warning("Ignoring verification request, already in progress...")
        return instance

    if await settings.redis_client.get(f"verify:lock:{instance_id}"):
        logger.info(f"Verification request is currently in progress: {instance_id=}")
        return instance
    attempts = await settings.redis_client.get(f"verify_instance:{instance_id}")
    if attempts and int(attempts) > 5:
        logger.warning(
            f"Refusing to attempt verification more than 7 times for {instance_id=} {attempts=}"
        )
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Too may verification requests.",
        )
    await settings.redis_client.incr(f"verify_instance:{instance_id}")
    await settings.redis_client.expire(f"verify_instance:{instance_id}", 600)
    await verify_instance.kiq(instance_id)
    return instance


@router.delete("/{chute_id}/{instance_id}")
async def delete_instance(
    chute_id: str,
    instance_id: str,
    request: Request,
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
    origin_ip = request.headers.get("x-forwarded-for")
    logger.info(f"INSTANCE DELETION INITIALIZED: {instance_id=} {hotkey=} {origin_ip=}")
    await db.delete(instance)
    await db.execute(
        text(
            "UPDATE instance_audit SET deletion_reason = 'miner initialized' WHERE instance_id = :instance_id"
        ),
        {"instance_id": instance_id},
    )
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
