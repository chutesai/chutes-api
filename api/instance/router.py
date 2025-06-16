"""
Routes for instances.
"""

import re
import uuid
import orjson as json
import traceback
import random
import secrets
import base64
from datetime import datetime, timedelta, timezone
from loguru import logger
from fastapi import APIRouter, Depends, HTTPException, status, Header, Request
from sqlalchemy import select, text, func
from sqlalchemy.orm import joinedload
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.exc import IntegrityError
from api.gpu import SUPPORTED_GPUS
from api.database import get_db_session, generate_uuid, get_session
from api.config import settings
from api.constants import (
    HOTKEY_HEADER,
    EXPANSION_UTILIZATION_THRESHOLD,
    UNDERUTILIZED_CAP,
    AUTHORIZATION_HEADER,
)
from api.node.util import get_node_by_id
from api.chute.schemas import Chute
from api.instance.schemas import (
    InstanceArgs,
    Instance,
    instance_nodes,
    ActivateArgs,
    LaunchConfig,
    LaunchConfigArgs,
)
from api.job.schemas import Job
from api.instance.util import (
    get_instance_by_chute_and_id,
    create_launch_jwt,
    load_launch_config_from_jwt,
)
from api.user.schemas import User
from api.user.service import get_current_user
from api.metasync import get_miner_by_hotkey
from api.util import is_valid_host, generate_ip_token, aes_decrypt
from api.graval_worker import verify_instance, graval_encrypt
from watchtower import get_expected_command, decrypt_envdump_cipher, is_kubernetes_env

router = APIRouter()


async def _load_chute(db, chute_id: str):
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


async def _check_blacklisted(db, hotkey):
    mgnode = await get_miner_by_hotkey(hotkey, db)
    if mgnode.blacklist_reason:
        logger.warning(f"MINERBLACKLIST: {hotkey=} reason={mgnode.blacklist_reason}")
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=f"Your hotkey has been blacklisted: {mgnode.blacklist_reason}",
        )
    return mgnode


async def _check_scalable(db, chute, hotkey):
    chute_id = chute.chute_id
    if chute.rolling_update:
        limit = chute.rolling_update.permitted.get(hotkey, 0)
        if not limit:
            logger.warning(
                f"SCALELOCK: chute {chute_id=} {chute.name} is currently undergoing a rolling update"
            )
            raise HTTPException(
                status_code=status.HTTP_423_LOCKED,
                detail=f"Chute {chute_id} is currently undergoing a rolling update and you have no quota, try again later.",
            )
    else:
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


async def _validate_node(db, chute, node_id: str, hotkey: str):
    node = await get_node_by_id(node_id, db, hotkey)
    if not node:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Node {node_id} not found",
        )

    # Not verified?
    if not node.verified_at:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"GPU {node_id} is not yet verified, and cannot be associated with an instance",
        )

    # Already associated with an instance?
    if node.instance:
        instance = node.instance
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"GPU {node_id} is already assigned to an instance: {instance.instance_id=} {instance.chute_id=}",
        )

    # Valid GPU for this chute?
    if not node.is_suitable(chute):
        logger.warning(
            f"INSTANCEFAIL: attempt to post incompatible GPUs: {node.name} for {chute.node_selector} {hotkey=}"
        )
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Node {node_id} is not compatible with chute node selector!",
        )
    return node


async def _validate_nodes(db, chute, node_ids: list[str], hotkey: str, instance: Instance):
    host = instance.host
    gpu_count = chute.node_selector.get("gpu_count", 1)
    if len(set(node_ids)) != gpu_count:
        logger.warning(
            f"INSTANCEFAIL: Attempt to post incorrect GPU count: {len(node_ids)} vs {gpu_count} from {hotkey=}"
        )
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"{chute.chute_id=} {chute.name=} requires exactly {gpu_count} GPUs.",
        )

    node_hosts = set()
    nodes = []
    for node_id in set(node_ids):
        node = await _validate_node(db, chute, node_id, hotkey)
        nodes.append(node)
        node_hosts.add(node.verification_host)

        # Create the association record.
        await db.execute(
            instance_nodes.insert().values(instance_id=instance.instance_id, node_id=node_id)
        )

    # The hostname used in verifying the node must match the hostname of the instance.
    if len(node_hosts) > 1 or list(node_hosts)[0].lower() != host.lower():
        logger.warning("INSTANCEFAIL: Instance hostname mismatch: {node_hosts=} {host=}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Instance hostname does not match the node verification hostname: {host=} vs {node_hosts=}",
        )
    return nodes


async def _validate_host_port(db, host, port):
    existing = (
        (
            await db.execute(
                select(Instance).where(Instance.host == host, Instance.port == port).limit(1)
            )
        )
        .unique()
        .scalar_one_or_none()
    )
    if existing:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Host/port {host}:{port} is already in use by another instance.",
        )

    if not await is_valid_host(host):
        logger.warning(f"INSTANCEFAIL: Attempt to post bad host: {host}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid instance host: {host}",
        )


@router.get("/launch_config")
async def get_launch_config(
    chute_id: str,
    host: str,
    port: int,
    job_id: str = None,
    db: AsyncSession = Depends(get_db_session),
    hotkey: str | None = Header(None, alias=HOTKEY_HEADER),
    _: User = Depends(get_current_user(raise_not_found=False, registered_to=settings.netuid)),
):
    miner = await _check_blacklisted(db, hotkey)

    # Load the chute and check if it's scalable.
    chute = await _load_chute(db, chute_id)
    await _check_scalable(db, chute, hotkey)

    # Associated with a job?
    if job_id:
        job = (
            (await db.execute(select(Job).where(Job.chute_id == chute_id, Job.job_id == job_id)))
            .unique()
            .scalar_one_or_none()
        )
        if not job:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Job {job_id} for chute {chute_id} not found",
            )
        if job.miner_hotkey:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=f"Job {job_id} has already been claimed!",
            )

        existing = (
            (await db.execute(select(LaunchConfig).where(LaunchConfig.job_id == job_id)))
            .unique()
            .scalar_one_or_none()
        )
        if existing:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=f"Job {job_id} has already been claimed!",
            )

        # Update job with miner info.
        job.miner_uid = miner.node_id
        job.miner_hotkey = miner.hotkey
        job.miner_coldkey = miner.coldkey

    # Create the launch config and JWT.
    try:
        launch_config = LaunchConfig(
            config_id=str(uuid.uuid4()),
            seed=random.randint(1, 2**63 - 1),
            env_key=secrets.token_bytes(16).hex(),
            chute_id=chute_id,
            job_id=job_id,
            miner_hotkey=hotkey,
            miner_uid=miner.node_id,
            miner_coldkey=miner.coldkey,
        )
        db.add(launch_config)
        await db.commit()
        await db.refresh(launch_config)
    except IntegrityError as exc:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Launch config conflict/unique constraint error: {exc}",
        )

    # Generate the JWT.
    return {
        "token": create_launch_jwt(launch_config),
        "config_id": launch_config.config_id,
    }


@router.post("/launch_config/{config_id}")
async def claim_launch_config(
    config_id: str,
    args: LaunchConfigArgs,
    db: AsyncSession = Depends(get_db_session),
    authorization: str = Header(None, alias=AUTHORIZATION_HEADER),
):
    # Load the launch config, verifying the token.
    token = authorization.strip().split(" ")[-1]
    launch_config = await load_launch_config_from_jwt(db, config_id, token)
    chute = await _load_chute(launch_config.chute_id)

    # Verify the code is what is expected.
    try:
        dump = json.loads(
            decrypt_envdump_cipher(args.dump, launch_config.env_key, chute.chutes_version)
        )
        process = dump[1] if isinstance(dump, list) else dump["process"]
        assert process["pid"] == 1
        command_line = re.sub(
            r"([^ ]+/)?python3?(\.[0-9]+)", "python", " ".join(process["cmdline"])
        )
        if command_line != get_expected_command(chute, token=token):
            logger.error(f"Attempt to claim {config_id=} failed, invalid command: {command_line=}")
            launch_config.failed_at = func.now()
            launch_config.verification_error = f"Invalid command: {command_line=}"
            await db.commit()
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"You are not running the correct command, sneaky devil: {command_line=}",
            )
        log_prefix = f"{config_id=} {chute.chute_id=}"
        if not is_kubernetes_env(chute, dump, log_prefix):
            logger.error(f"{log_prefix} is not running a valid kubernetes environment")
            launch_config.failed_at = func.now()
            launch_config.verification_error = "Failed kubernetes environment check."
            await db.commit()
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=launch_config.verification_error,
            )

    except Exception as exc:
        logger.error(
            f"Attempt to claim {config_id=} failed, unable to verify command: {exc=} {args=}"
        )
        launch_config.failed_at = func.now()
        launch_config.verification_error = f"Unable to verify: {exc=} {args=}"
        await db.commit()
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=launch_config.verification_error,
        )

    # Create the instance on the fly.
    instance = Instance(
        instance_id=generate_uuid(),
        host=launch_config.host,
        port=launch_config.port,
        chute_id=launch_config.chute_id,
        version=chute.version,
        miner_uid=launch_config.miner_uid,
        miner_hotkey=launch_config.miner_hotkey,
        miner_coldkey=launch_config.miner_coldkey,
        region="n/a",
        active=False,
        verified=False,
        chutes_version=chute.chutes_version,
        symmetric_key=secrets.token_bytes(16).hex(),
        job_id=launch_config.job_id,
        config_id=launch_config.config_id,
    )
    db.add(instance)

    # Mark the job as associated with this instance.
    if launch_config.job_id:
        job = (
            (await db.execute(select(Job).where(Job.job_id == launch_config.job_id)))
            .unique()
            .scalar_one_or_none()
        )
        if not job:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Job {launch_config.job_id} no longer exists!",
            )
        job.instance_id = instance.instance_id

    # Verify the GPUs are suitable.
    node_ids = [node["uuid"] for node in args.gpus]
    try:
        nodes = await _validate_nodes(
            db,
            chute,
            node_ids,
            launch_config.miner_hotkey,
            instance,
        )
    except Exception:
        await db.rollback()
        async with get_session() as error_session:
            await error_session.execute(
                text(
                    "UPDATE launch_configs SET failed_at = NOW(), "
                    "verification_error = 'invalid GPU/nodes configuration provided' "
                    "WHERE config_id = :config_id"
                ),
                {"config_id": launch_config.config_id},
            )
        raise

    # Generate a ciphertext for this instance to decrypt.
    node_idx = random.choice(list(range(len(nodes))))
    node = nodes[node_idx]
    iterations = SUPPORTED_GPUS[node.gpu_identifier]["graval"]["iterations"]
    ciphertext = await graval_encrypt(
        node,
        instance.symmetric_key,
        with_chutes=True,
        cuda=False,
        seed=launch_config.seed,
        iterations=iterations,
    )
    logger.success(
        f"Generated ciphertext for {node.uuid} "
        f"with seed={launch_config.seed} "
        f"instance_id={instance.instance_id} "
        f"for symmetric key validation/PovW check: {ciphertext=}"
    )

    # Store the timestamp so we can verify the graval challenges completed in the expected time.
    launch_config.retrieved_at = func.now()
    await db.commit()
    await db.refresh(launch_config)

    # Set timestamp in a fresh transaction so it's not affected by the long cipher gen time.
    async with get_session() as session:
        await session.execute(
            text("UPDATE launch_configs SET retrieved_at = NOW() WHERE config_id = :config_id"),
            {"config_id": config_id},
        )

    # The miner must decrypt the proposed symmetric key from this response payload,
    # then encrypt something using this symmetric key within the expected graval timeout.
    return {
        "seed": launch_config.seed,
        "iterations": iterations,
        "job_id": launch_config.job_id,
        "symmetric_key": {
            "ciphertext": ciphertext,
            "device_index": node_idx,
            "plaintext_response": f"secret is {launch_config.config_id} {launch_config.seed}",
        },
    }


@router.put("/launch_config/{config_id}")
async def verify_launch_config_instance(
    config_id: str,
    request: Request,
    db: AsyncSession = Depends(get_db_session),
    authorization: str = Header(None, alias=AUTHORIZATION_HEADER),
):
    token = authorization.strip().split(" ")[-1]
    launch_config = await load_launch_config_from_jwt(db, config_id, token, allow_retrieved=True)

    # Validate the launch config.
    if launch_config.verified_at:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Launch config has already been verified: {config_id}",
        )
    if launch_config.failed_at:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=f"Launch config failed verification: {launch_config.failed_at=} {launch_config.verification_error=}",
        )

    # Check decryption time.
    now = datetime.now(timezone.utc).replace(tzinfo=None)
    start = launch_config.retrieved_at.replace(tzinfo=None)
    query = (
        select(Instance)
        .where(Instance.config_id == launch_config.config_id)
        .options(joinedload(Instance.nodes).joinedload(Instance.job))
    )
    instance = (await db.execute(query)).unique().scalar_one_or_none()
    estimate = SUPPORTED_GPUS[instance.gpu_identifier]["graval"]["estimate"]
    max_duration = estimate * 1.3
    if (delta := now - start) >= timedelta(seconds=max_duration):
        launch_config.failed_at = func.now()
        launch_config.verification_error = f"GraVal challenge took {delta}, expected completion time {estimate} with buffer of up to {max_duration}"
        await db.commit()
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=launch_config.verification_error,
        )

    # Valid response cipher?
    try:
        ciphertext = (await request.json())["response"]
        bytes_ = base64.b64decode(ciphertext[32:])
        iv = bytes.fromhex(ciphertext[:32])
        response = aes_decrypt(bytes_, instance.symmetric_key, iv)
        assert response == f"secret is {launch_config.config_id} {launch_config.seed}"
    except Exception as exc:
        launch_config.failed_at = func.now()
        launch_config.verification_error = f"PoVW encrypted response was invalid: {exc}"
        await db.commit()
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=launch_config.verification_error,
        )

    # Everything checks out (so far).
    launch_config.verified_at = func.now()
    job = instance.job
    if job:
        job.started_at = func.now()
    await db.commit()
    await db.refresh(launch_config)
    if job:
        await db.refresh(job)
    return_value = {
        "chute_id": launch_config.chute_id,
        "instance_id": instance.instance_id,
        "verified_at": launch_config.verified_at.isoformat(),
    }
    if job:
        return_value.update(
            {
                "job_id": instance.job_id,
                "job_method": instance.job.method,
                "job_data": instance.job.job_args,
            }
        )
    return return_value


@router.post("/{chute_id}/", status_code=status.HTTP_202_ACCEPTED)
async def create_instance(
    chute_id: str,
    instance_args: InstanceArgs,
    db: AsyncSession = Depends(get_db_session),
    hotkey: str | None = Header(None, alias=HOTKEY_HEADER),
    _: User = Depends(get_current_user(raise_not_found=False, registered_to=settings.netuid)),
):
    await _check_blacklisted(db, hotkey)

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

    # Host port already used?
    await _validate_host_port(db, instance_args.host, instance_args.port)

    # Scalable?
    await _check_scalable(db, chute, hotkey)
    if chute.rolling_update:
        chute.rolling_update.permitted[hotkey] -= 1

    gpu_type = None
    gpu_count = chute.node_selector.get("gpu_count", 1)
    try:
        # Load the miner.
        miner = await get_miner_by_hotkey(hotkey, db)
        if not miner:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail=f"Did not find miner with {hotkey=}",
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
        nodes = await _validate_nodes(db, chute, instance_args.node_ids, hotkey, instance)
        gpu_type = nodes[0].gpu_identifier

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
