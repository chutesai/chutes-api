"""
Routes for instances.
"""

import os
import uuid
import base64
import traceback
import random
import socket
import secrets
import asyncio
import api.miner_client as miner_client
from loguru import logger
from typing import Optional
from datetime import timedelta
from fastapi import APIRouter, Depends, HTTPException, status, Header, Request
from sqlalchemy import select, text, func, update
from sqlalchemy.orm import joinedload
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.exc import IntegrityError
from api.gpu import SUPPORTED_GPUS
from api.database import get_db_session, generate_uuid, get_session
from api.config import settings
from api.permissions import Permissioning
from api.constants import (
    HOTKEY_HEADER,
    AUTHORIZATION_HEADER,
    PRIVATE_INSTANCE_MULTIPLIER,
)
from api.node.util import get_node_by_id
from api.chute.schemas import Chute, NodeSelector
from api.instance.schemas import (
    Instance,
    instance_nodes,
    LaunchConfig,
    LaunchConfigArgs,
)
from api.job.schemas import Job
from api.instance.util import (
    get_instance_by_chute_and_id,
    create_launch_jwt,
    create_job_jwt,
    load_launch_config_from_jwt,
)
from api.user.schemas import User
from api.user.service import get_current_user, chutes_user_id
from api.metasync import get_miner_by_hotkey
from api.util import (
    semcomp,
    is_valid_host,
    generate_ip_token,
    aes_decrypt,
    notify_created,
    notify_deleted,
    notify_verified,
    notify_activated,
    has_legacy_private_billing,
)
from api.bounty.util import check_bounty_exists, delete_bounty
from starlette.responses import StreamingResponse
from api.graval_worker import graval_encrypt, verify_proof, generate_fs_hash
from watchtower import is_kubernetes_env, verify_expected_command

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
    return chute


async def _check_blacklisted(db, hotkey):
    mgnode = await get_miner_by_hotkey(hotkey, db)
    if not mgnode:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Miner with hotkey {hotkey} not found in metagraph",
        )
    if mgnode.blacklist_reason:
        logger.warning(f"MINERBLACKLIST: {hotkey=} reason={mgnode.blacklist_reason}")
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=f"Your hotkey has been blacklisted: {mgnode.blacklist_reason}",
        )
    return mgnode


async def _check_scalable(db, chute, hotkey):
    chute_id = chute.chute_id
    query = text("""
        SELECT
            COUNT(*) AS total_count,
            COUNT(CASE WHEN active IS true AND verified IS true THEN 1 ELSE NULL END) AS active_count,
            COUNT(CASE WHEN miner_hotkey = :hotkey THEN 1 ELSE NULL END) AS hotkey_count
        FROM instances
        WHERE chute_id = :chute_id
    """)
    count_result = (
        (await db.execute(query, {"chute_id": chute_id, "hotkey": hotkey})).mappings().first()
    )
    current_count = count_result["total_count"]
    active_count = count_result["active_count"]
    hotkey_count = count_result["hotkey_count"]

    # Get target count from Redis
    scale_value = await settings.redis_client.get(f"scale:{chute_id}")
    if scale_value:
        target_count = int(scale_value)
    else:
        # Fallback to database.
        capacity_query = text("""
            SELECT target_count
            FROM capacity_log
            WHERE chute_id = :chute_id
            ORDER BY timestamp DESC
            LIMIT 1
        """)
        capacity_result = await db.execute(capacity_query, {"chute_id": chute_id})
        capacity_row = capacity_result.first()
        if capacity_row and capacity_row.target_count is not None:
            target_count = capacity_row.target_count
            logger.info(f"Retrieved target_count from CapacityLog for {chute_id}: {target_count}")
        else:
            target_count = current_count
            logger.warning(
                f"No target_count in Redis or CapacityLog for {chute_id}, "
                f"using conservative current count as default: {target_count}"
            )

    # Edge case where we have too many instances already but none are active.
    if current_count <= target_count + 2 and active_count == 0:
        logger.warning("No active instances, allowing!")
        return

    # When there is a rolling update in progress, use the permitted list.
    if chute.rolling_update and current_count >= target_count - 2:
        limit = chute.rolling_update.permitted.get(hotkey, 0) or 0
        if limit <= 0 and chute.rolling_update.permitted:
            logger.warning(
                f"SCALELOCK: chute {chute_id=} {chute.name} is currently undergoing a rolling update"
            )
            raise HTTPException(
                status_code=status.HTTP_423_LOCKED,
                detail=f"Chute {chute_id} is currently undergoing a rolling update and you have no quota, try again later.",
            )
        elif limit:
            chute.rolling_update.permitted[hotkey] -= 1
        return

    # Check if scaling is allowed based on target count.
    if current_count >= target_count:
        logger.warning(
            f"SCALELOCK: chute {chute_id=} {chute.name} has reached target capacity: "
            f"{current_count=}, {active_count=}, {target_count=}, {hotkey_count=}"
        )
        raise HTTPException(
            status_code=status.HTTP_423_LOCKED,
            detail=f"Chute {chute_id} has reached its target capacity of {target_count} instances.",
        )

    # Prevent monopolizing capped chutes.
    remaining_slots = target_count - current_count
    if remaining_slots <= 2 and hotkey_count == current_count:
        logger.warning(
            f"SCALELOCK: chute {chute_id=} {chute.name} - miner {hotkey} already has {hotkey_count} instances, "
            f"only {remaining_slots} slots remaining"
        )
        raise HTTPException(
            status_code=status.HTTP_423_LOCKED,
            detail=f"Chute {chute_id} is near capacity with only {remaining_slots} slots remaining. "
            f"You already have {hotkey_count} instance(s).",
        )


async def _check_scalable_private(db, chute):
    """
    Special scaling logic for private chutes (without legacy billing).
    """
    chute_id = chute.chute_id
    query = text("""
        SELECT
            COUNT(*) AS total_count,
            COUNT(CASE WHEN active IS true AND verified IS true THEN 1 ELSE NULL END) AS active_count
        FROM instances
        WHERE chute_id = :chute_id
    """)
    count_result = (await db.execute(query, {"chute_id": chute_id})).mappings().first()
    active_count = count_result["active_count"]
    scale_value = await settings.redis_client.get(f"scale:{chute_id}")
    target_count = int(scale_value) if scale_value else 0
    bounty_exists = await check_bounty_exists(chute_id)
    if active_count == 0 and not bounty_exists:
        raise HTTPException(
            status_code=status.HTTP_423_LOCKED,
            detail=f"Private chute {chute_id} has no active bounty and cannot be scaled.",
        )
    if active_count >= target_count and target_count > 0:
        raise HTTPException(
            status_code=status.HTTP_423_LOCKED,
            detail=f"Private chute {chute_id} has reached its target capacity of {target_count} instances.",
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
    job_id: Optional[str] = None,
    db: AsyncSession = Depends(get_db_session),
    hotkey: str | None = Header(None, alias=HOTKEY_HEADER),
    _: User = Depends(
        get_current_user(raise_not_found=False, registered_to=settings.netuid, purpose="launch")
    ),
):
    miner = await _check_blacklisted(db, hotkey)

    # Load the chute and check if it's scalable.
    chute = await _load_chute(db, chute_id)
    if not job_id:
        if (
            not chute.public
            and not has_legacy_private_billing(chute)
            and chute.user_id != await chutes_user_id()
        ):
            await _check_scalable_private(db, chute)
        else:
            await _check_scalable(db, chute, hotkey)

    # Associated with a job?
    disk_gb = None
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

        # Don't allow too many miners to try to claim the job...
        if len(job.miner_history) >= 15:
            raise HTTPException(
                status_code=status.HTTP_423_LOCKED,
                detail=f"Job {job_id} for chute {chute_id} is already in a race between {len(job.miner_history)} miners",
            )

        # Don't allow miners to try claiming a job more than once.
        if hotkey in job.miner_history:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Your hotkey has already attempted to claim {job_id=}",
            )

        # Track this miner in the job history.
        await db.execute(
            text(
                "UPDATE jobs SET miner_history = miner_history || jsonb_build_array(CAST(:hotkey AS TEXT))"
                "WHERE job_id = :job_id"
            ),
            {"job_id": job_id, "hotkey": hotkey},
        )
        disk_gb = job.job_args["_disk_gb"]

    # Create the launch config and JWT.
    try:
        launch_config = LaunchConfig(
            config_id=str(uuid.uuid4()),
            env_key=secrets.token_bytes(16).hex(),
            chute_id=chute_id,
            job_id=job_id,
            miner_hotkey=hotkey,
            miner_uid=miner.node_id,
            miner_coldkey=miner.coldkey,
            seed=0,
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
        "token": create_launch_jwt(launch_config, disk_gb=disk_gb),
        "config_id": launch_config.config_id,
    }


@router.post("/launch_config/{config_id}")
async def claim_launch_config(
    config_id: str,
    args: LaunchConfigArgs,
    request: Request,
    db: AsyncSession = Depends(get_db_session),
    authorization: str = Header(None, alias=AUTHORIZATION_HEADER),
):
    from chutes.envdump import DUMPER

    # Load the launch config, verifying the token.
    token = authorization.strip().split(" ")[-1]
    launch_config = await load_launch_config_from_jwt(db, config_id, token)
    chute = await _load_chute(db, launch_config.chute_id)

    # Generate a tentative instance ID.
    new_instance_id = generate_uuid()

    # Re-check scalable...
    if not launch_config.job_id:
        if (
            not chute.public
            and not has_legacy_private_billing(chute)
            and chute.user_id != await chutes_user_id()
        ):
            await _check_scalable_private(db, chute)
        else:
            await _check_scalable(db, chute, launch_config.miner_hotkey)

    # IP matches?
    x_forwarded_for = request.headers.get("X-Forwarded-For")
    actual_ip = x_forwarded_for.split(",")[0] if x_forwarded_for else request.client.host
    if launch_config.job_id and actual_ip != args.host:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Egress and ingress IPs much match for jobs: {actual_ip} vs {args.host}",
        )

    # Verify, decrypt, parse the envdump payload.
    if "ENVDUMP_UNLOCK" in os.environ:
        code = None
        try:
            dump = DUMPER.decrypt(launch_config.env_key, args.env)
            code_data = DUMPER.decrypt(launch_config.env_key, args.code)
            code = base64.b64decode(code_data["content"]).decode()
        except Exception as exc:
            logger.error(
                f"Attempt to claim {config_id=} failed, invalid envdump payload received: {exc}"
            )
            launch_config.failed_at = func.now()
            launch_config.verification_error = f"Unable to verify: {exc=} {args=}"
            await db.commit()
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=launch_config.verification_error,
            )

        # Check the environment.
        try:
            await verify_expected_command(
                dump,
                chute,
                miner_hotkey=launch_config.miner_hotkey,
            )
            assert code == chute.code, f"Incorrect code:\n{code=}\n{chute.code=}"
        except AssertionError as exc:
            logger.error(f"Attempt to claim {config_id=} failed, invalid command: {exc}")
            launch_config.failed_at = func.now()
            launch_config.verification_error = f"Invalid command: {exc}"
            await db.commit()
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"You are not running the correct command, sneaky devil: {exc}",
            )

        # K8S check.
        log_prefix = f"ENVDUMP: {config_id=} {chute.chute_id=}"
        if not is_kubernetes_env(chute, dump, log_prefix=log_prefix):
            logger.error(f"{log_prefix} is not running a valid kubernetes environment")
            launch_config.failed_at = func.now()
            launch_config.verification_error = "Failed kubernetes environment check."
            await db.commit()
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=launch_config.verification_error,
            )

    else:
        logger.warning("Unable to perform extended validation, skipping...")

    # Valid filesystem/integrity?
    if semcomp(chute.chutes_version, "0.3.1") >= 0:
        image_id = chute.image_id
        patch_version = chute.image.patch_version
        if "CFSV_OP" in os.environ:
            task = await generate_fs_hash.kiq(
                image_id,
                patch_version,
                launch_config.config_id,
                sparse=False,
                exclude_path=f"/app/{chute.filename}",
            )
            result = await task.wait_result()
            expected_hash = result.return_value
            if expected_hash != args.fsv:
                logger.error(
                    f"Filesystem challenge failed for {launch_config.config_id=} {launch_config.miner_hotkey=}, "
                    f"{expected_hash=} for {chute.image_id=} {patch_version=} but received {args.fsv}"
                )
                launch_config.failed_at = func.now()
                launch_config.verification_error = (
                    "File system verification failure, mismatched hash"
                )
                await db.commit()
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail=launch_config.verification_error,
                )
        else:
            logger.warning("Extended filesystem verification disabled, skipping...")

    # Assign the job to this launch config.
    if launch_config.job_id:
        stmt = (
            update(Job)
            .where(
                Job.job_id == launch_config.job_id,
                Job.miner_hotkey.is_(None),
            )
            .values(
                miner_uid=launch_config.miner_uid,
                miner_hotkey=launch_config.miner_hotkey,
                miner_coldkey=launch_config.miner_coldkey,
            )
        )
        result = await db.execute(stmt)
        if result.rowcount == 0:
            # Job was already claimed by another miner
            logger.warning(
                f"Job {launch_config.job_id=} via {launch_config.config_id=} was already "
                f"claimed when miner {launch_config.miner_hotkey=} tried to claim it."
            )
            launch_config.failed_at = func.now()
            launch_config.verification_error = "Job was already claimed by another miner"
            await db.commit()
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=f"Job {launch_config.job_id} has already been claimed by another miner!",
            )

    # Create the instance now that we've verified the envdump/k8s env.
    node_selector = NodeSelector(**chute.node_selector)
    instance = Instance(
        instance_id=new_instance_id,
        host=args.host,
        port=args.port_mappings[0].external_port,
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
        config_id=launch_config.config_id,
        port_mappings=[item.model_dump() for item in args.port_mappings],
        compute_multiplier=node_selector.compute_multiplier,
        billed_to=None,
        hourly_rate=(await node_selector.current_estimated_price())["usd"]["hour"],
    )
    if launch_config.job_id or (
        not chute.public
        and not has_legacy_private_billing(chute)
        and chute.user_id != await chutes_user_id()
    ):
        instance.compute_multiplier *= PRIVATE_INSTANCE_MULTIPLIER
        instance.billed_to = chute.user_id

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
        job.port_mappings = [item.model_dump() for item in args.port_mappings]

        # Verify port mappings are correct.
        job_obj = next(j for j in chute.jobs if j["name"] == job.method)
        expected = set([f"{p['proto']}:{p['port']}".lower() for p in job_obj["ports"]])
        received = set(
            [
                f"{p.proto}:{p.internal_port}".lower()
                for p in args.port_mappings
                if p.internal_port not in [8000, 8001]
            ]
        )
        if expected != received:
            logger.error(
                f"{instance.instance_id=} from {config_id=} posted invalid ports: {expected=} vs {received=}"
            )
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Invalid port mappings provided: {expected=} {received=}",
            )

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
                {"config_id": config_id},
            )
            await error_session.commit()
        raise

    # Generate a ciphertext for this instance to decrypt.
    node = random.choice(nodes)
    iterations = SUPPORTED_GPUS[node.gpu_identifier]["graval"]["iterations"]
    encrypted_payload = await graval_encrypt(
        node,
        instance.symmetric_key,
        iterations=iterations,
        seed=None,
    )
    parts = encrypted_payload.split("|")
    seed = int(parts[0])
    ciphertext = parts[1]
    launch_config.seed = seed
    logger.success(
        f"Generated ciphertext for {node.uuid} "
        f"with seed={seed} "
        f"instance_id={instance.instance_id} "
        f"for symmetric key validation/PovW check: {ciphertext=}"
    )

    # Store the launch config.
    await db.commit()
    await db.refresh(launch_config)

    # Set timestamp in a fresh transaction so it's not affected by the long cipher gen time.
    async with get_session() as session:
        await session.execute(
            text("UPDATE launch_configs SET retrieved_at = NOW() WHERE config_id = :config_id"),
            {"config_id": config_id},
        )

    # Send event.
    await db.refresh(instance)
    gpu_count = len(nodes)
    gpu_type = nodes[0].gpu_identifier
    asyncio.create_task(notify_created(instance, gpu_count=gpu_count, gpu_type=gpu_type))

    # The miner must decrypt the proposed symmetric key from this response payload,
    # then encrypt something using this symmetric key within the expected graval timeout.
    return {
        "seed": launch_config.seed,
        "iterations": iterations,
        "job_id": launch_config.job_id,
        "symmetric_key": {
            "ciphertext": ciphertext,
            "uuid": node.uuid,
            "response_plaintext": f"secret is {launch_config.config_id} {launch_config.seed}",
        },
    }


@router.get("/launch_config/{config_id}/activate")
async def activate_launch_config_instance(
    config_id: str,
    request: Request,
    db: AsyncSession = Depends(get_db_session),
    authorization: str = Header(None, alias=AUTHORIZATION_HEADER),
):
    token = authorization.strip().split(" ")[-1]
    launch_config = await load_launch_config_from_jwt(db, config_id, token, allow_retrieved=True)
    if not launch_config.verified_at:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Launch config has not been verified.",
        )
    instance = launch_config.instance
    if not instance:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Instance has disappeared for launch_{config_id=}",
        )
    chute = (
        (await db.execute(select(Chute).where(Chute.chute_id == instance.chute_id)))
        .unique()
        .scalar_one_or_none()
    )

    # Prevent activation of private instances if we're already capped. This is necessary here
    # because we allow the miners to "race" to deploy so there are potentially more instances
    # inactive/pending than actually allowed.
    if (
        not chute.public
        and not has_legacy_private_billing(chute)
        and chute.user_id != await chutes_user_id()
    ):
        query = text("""
            SELECT COUNT(CASE WHEN active IS true AND verified IS true THEN 1 ELSE NULL END) AS active_count
            FROM instances
            WHERE chute_id = :chute_id
        """)
        count_result = (await db.execute(query, {"chute_id": chute.chute_id})).mappings().first()
        active_count = count_result["active_count"]
        scale_value = await settings.redis_client.get(f"scale:{chute.chute_id}")
        target_count = int(scale_value) if scale_value else 0
        can_scale = False
        if not active_count and await check_bounty_exists(chute.chute_id):
            can_scale = True
        elif active_count < target_count:
            can_scale = True
        if not can_scale:
            reason = f"Private chute {chute.chute_id=} {chute.name=} already has >= {target_count=} active instances"
            logger.warning(reason)
            await db.delete(instance)
            await asyncio.create_task(notify_deleted(instance))
            await db.execute(
                text(
                    "UPDATE instance_audit SET deletion_reason = :reason WHERE instance_id = :instance_id"
                ),
                {"instance_id": instance.instance_id, "reason": reason},
            )
            raise HTTPException(
                status_code=status.HTTP_423_LOCKED,
                detail=reason,
            )

    # Activate the instance (and trigger tentative billing stop time).
    if not instance.active:
        instance.active = True
        instance.activated_at = func.now()
        if launch_config.job_id or (
            not chute.public
            and not has_legacy_private_billing(chute)
            and chute.user_id != await chutes_user_id()
        ):
            instance.stop_billing_at = func.now() + timedelta(
                seconds=chute.shutdown_after_seconds or 300
            )
            # For private instances, we need to delete the bounty to prevent scaling back up
            # if this instance is terminated before a request is made. The miner will still a
            # bounty, however, since each private instance automatically counts as a bounty (see metasync/shared.py)
            await delete_bounty(chute.chute_id)
        await db.commit()
        asyncio.create_task(notify_activated(instance))
    return {"ok": True}


async def verify_port_map(instance, port_map):
    """
    Verify a port is open on the remote chute pod.
    """
    logger.info(f"Attempting to verify {port_map=} on {instance.instance_id=}")
    try:
        if port_map["proto"].lower() in ["tcp", "http"]:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(10)
            sock.connect((instance.host, port_map["external_port"]))
            logger.info(f"Connected to {instance.instance_id=} on {port_map=}")
            sock.send(b"test")
            logger.info(f"Sent a packet to {instance.instance_id=} on {port_map=}")
            response = sock.recv(1024).decode()
            logger.success(f"Received a response from {instance.instance_id=} on {port_map=}")
            sock.close()
        else:
            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            sock.settimeout(10)
            sock.sendto(b"test", (instance.host, port_map["external_port"]))
            logger.info(f"Sent a packet to {instance.instance_id=} on {port_map=}")
            response, _ = sock.recvfrom(1024)
            response = response.decode()
            logger.success(f"Received a response from {instance.instance_id=} on {port_map=}")
            sock.close()
        if "|" not in response:
            logger.error(f"Invalid socket response for {port_map=} {response=}")
            return False

        iv_hex, encrypted_response = response.split("|", 1)
        decrypted = aes_decrypt(encrypted_response, instance.symmetric_key, iv_hex)
        expected = f"response from {port_map['proto'].lower()} {port_map['internal_port']}"
        return decrypted.decode() == expected
    except Exception as e:
        logger.error(f"Port verification failed for {port_map}: {e}")
        return False


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
        logger.warning(f"Launch config {config_id} has already been verified!")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Launch config has already been verified: {config_id}",
        )
    if launch_config.failed_at:
        logger.warning(
            f"Launch config {config_id} has non-null failed_at: {launch_config.failed_at}"
        )
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=f"Launch config failed verification: {launch_config.failed_at=} {launch_config.verification_error=}",
        )

    # Check decryption time.
    now = (await db.scalar(select(func.now()))).replace(tzinfo=None)
    start = launch_config.retrieved_at.replace(tzinfo=None)
    query = (
        select(Instance)
        .where(Instance.config_id == launch_config.config_id)
        .options(
            joinedload(Instance.nodes),
            joinedload(Instance.job),
            joinedload(Instance.chute),
        )
    )
    instance = (await db.execute(query)).unique().scalar_one_or_none()
    if not instance:
        logger.error(
            f"Instance associated with lauch config has been deleted! {launch_config.config_id=}"
        )
        launch_config.failed_at = func.now()
        launch_config.verification_error = "Instance was deleted"
        await db.commit()
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Instance disappeared (did you update gepetto reconcile?)",
        )
    estimate = SUPPORTED_GPUS[instance.nodes[0].gpu_identifier]["graval"]["estimate"]
    max_duration = estimate * 2.15
    if (delta := (now - start).total_seconds()) >= max_duration:
        reason = (
            f"PoVW encrypted response for {config_id=} and {instance.instance_id=} "
            f"{instance.miner_hotkey=} took {delta} seconds, exceeding maximum estimate of {max_duration}"
        )
        logger.error(reason)
        launch_config.failed_at = func.now()
        launch_config.verification_error = reason
        await db.delete(instance)
        await db.execute(
            text(
                "UPDATE instance_audit SET deletion_reason = :reason WHERE instance_id = :instance_id"
            ),
            {"instance_id": instance.instance_id, "reason": reason},
        )
        await db.commit()
        asyncio.create_task(notify_deleted(instance))
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=launch_config.verification_error,
        )

    # Valid response cipher?
    response_body = await request.json()
    try:
        ciphertext = response_body["response"]
        iv = response_body["iv"]
        response = aes_decrypt(ciphertext, instance.symmetric_key, iv)
        assert response == f"secret is {launch_config.config_id} {launch_config.seed}".encode()
    except Exception as exc:
        reason = (
            f"PoVW encrypted response for {config_id=} and {instance.instance_id=} "
            f"{instance.miner_hotkey=} was invalid: {exc}\n{traceback.format_exc()}"
        )
        logger.error(reason)
        launch_config.failed_at = func.now()
        launch_config.verification_error = reason
        await db.delete(instance)
        await db.execute(
            text(
                "UPDATE instance_audit SET deletion_reason = :reason WHERE instance_id = :instance_id"
            ),
            {"instance_id": instance.instance_id, "reason": reason},
        )
        await db.commit()
        asyncio.create_task(notify_deleted(instance))
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=launch_config.verification_error,
        )

    # Valid proof?
    try:
        node_idx = random.randint(0, len(instance.nodes) - 1)
        node = instance.nodes[node_idx]
        work_product = response_body["proof"][node.uuid]["work_product"]
        logger.info(f"CHECKING PROOF: {work_product=}\n{response_body=}")
        assert await verify_proof(node, launch_config.seed, work_product)
    except Exception as exc:
        reason = (
            f"PoVW proof failed for {config_id=} and {instance.instance_id=} "
            f"{instance.miner_hotkey=}: {exc}\n{traceback.format_exc()}"
        )
        launch_config.failed_at = func.now()
        launch_config.verification_error = reason
        await db.delete(instance)
        await db.execute(
            text(
                "UPDATE instance_audit SET deletion_reason = :reason WHERE instance_id = :instance_id"
            ),
            {"instance_id": instance.instance_id, "reason": reason},
        )
        await db.commit()
        asyncio.create_task(notify_deleted(instance))
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=launch_config.verification_error,
        )

    # Valid filesystem/integrity?
    if semcomp(instance.chute.chutes_version, "0.3.1") < 0:
        image_id = instance.chute.image_id
        patch_version = instance.chute.image.patch_version
        if "CFSV_OP" in os.environ:
            task = await generate_fs_hash.kiq(
                image_id,
                patch_version,
                launch_config.seed,
                sparse=False,
                exclude_path=f"/app/{instance.chute.filename}",
            )
            result = await task.wait_result()
            expected_hash = result.return_value
            if expected_hash != response_body["fsv"]:
                reason = (
                    f"Filesystem challenge failed for {config_id=} and {instance.instance_id=} {instance.miner_hotkey=}, "
                    f"{expected_hash=} for {image_id=} {patch_version=} but received {response_body['fsv']}"
                )
                logger.error(reason)
                launch_config.failed_at = func.now()
                launch_config.verification_error = reason
                await db.delete(instance)
                await db.execute(
                    text(
                        "UPDATE instance_audit SET deletion_reason = :reason WHERE instance_id = :instance_id"
                    ),
                    {"instance_id": instance.instance_id, "reason": reason},
                )
                await db.commit()
                asyncio.create_task(notify_deleted(instance))
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail=launch_config.verification_error,
                )
        else:
            logger.warning("Extended filesystem verification disabled, skipping...")

    # Everything checks out.
    launch_config.verified_at = func.now()
    job = instance.job
    if job:
        # Test the ports are open.
        for port_map in instance.port_mappings:
            if port_map["internal_port"] in (8000, 8001):
                continue
            if not await verify_port_map(instance, port_map):
                reason = f"Failed port verification on {port_map=} for {instance.instance_id=} {instance.miner_hotkey=}"
                logger.error(reason)
                await db.execute(
                    text(
                        "UPDATE instance_audit SET deletion_reason = :reason WHERE instance_id = :instance_id"
                    ),
                    {"instance_id": instance.instance_id, "reason": reason},
                )
                asyncio.create_task(notify_deleted(instance))
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail=f"Failed port verification on {port_map=}",
                )

        # All good!
        job.started_at = func.now()

    # Can't do this via the instance attrs directly, circular dependency :/
    await db.execute(
        text(
            "UPDATE instances SET verified = true, verification_error = null, last_verified_at = now() WHERE instance_id = :instance_id"
        ),
        {"instance_id": instance.instance_id},
    )

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
        job_token = create_job_jwt(job.job_id)
        return_value.update(
            {
                "job_id": instance.job.job_id,
                "job_method": instance.job.method,
                "job_data": instance.job.job_args,
                "job_status_url": f"https://api.{settings.base_domain}/jobs/{instance.job.job_id}?token={job_token}",
            }
        )

    # Stealth models...
    if instance.chute.name in ("Zenith", "Meridian", "Proxima"):
        return_value.update(
            {
                "secrets": {
                    "HUGGING_FACE_HUB_TOKEN": os.getenv("STEALTH_0_TOKEN"),
                },
            }
        )

    return_value["activation_url"] = (
        f"https://api.{settings.base_domain}/instances/launch_config/{launch_config.config_id}/activate"
    )

    await db.refresh(instance)
    asyncio.create_task(notify_verified(instance))
    return return_value


@router.get("/token_check")
async def get_token(salt: str = None, request: Request = None):
    origin_ip = request.headers.get("x-forwarded-for", "").split(",")[0]
    return {"token": generate_ip_token(origin_ip, extra_salt=salt)}


@router.get("/{instance_id}/logs")
async def stream_logs(
    instance_id: str,
    request: Request,
    backfill: Optional[int] = 100,
    db: AsyncSession = Depends(get_db_session),
    hotkey: str | None = Header(None, alias=HOTKEY_HEADER),
    current_user: User = Depends(get_current_user()),
):
    """
    Fetch the raw kubernetes pod logs, but only if logging_enabled is
    true on the chute itself, meaning the developer of the chute has
    explicitly set the option.

    These are application-level logs, which for example would not include
    any prompts/responses/etc. by default for any sglang/vllm container.

    The caveat is that affine admins can view any affine chute pod logs.
    """
    # These are raw application (k8s pod) logs
    instance = (
        (
            await db.execute(
                select(Instance)
                .where(Instance.instance_id == instance_id)
                .options(joinedload(Instance.chute))
            )
        )
        .unique()
        .scalar_one_or_none()
    )
    if not instance:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Instance not found.",
        )
    if instance.chute.user_id != current_user.user_id or not instance.chute.logging_enabled:
        if (
            not current_user.has_role(Permissioning.affine_admin)
            or "affine" not in instance.chute.name.lower()
        ):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=(
                    "You may only view logs for your own chutes, IFF logging_enabled "
                    "is configured on the chute (or be affine admin for affine models)"
                ),
            )
    if not 0 <= backfill <= 10000:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="`backfill` must be between 0 and 10000 (lines of logs)",
        )

    async def _stream():
        log_port = next(p for p in instance.port_mappings if p["internal_port"] == 8001)[
            "external_port"
        ]
        async with miner_client.get(
            instance.miner_hotkey,
            f"http://{instance.host}:{log_port}/logs/stream",
            timeout=0,
            purpose="chutes",
            params={"backfill": str(backfill)},
        ) as resp:
            async for chunk in resp.content:
                yield chunk

    return StreamingResponse(
        _stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


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

    # Fail the job.
    job = (
        (await db.execute(select(Job).where(Job.instance_id == instance_id)))
        .unique()
        .scalar_one_or_none()
    )
    if job and not job.finished_at:
        job.status = "error"
        job.error_detail = f"Instance was terminated by miner: {hotkey=}"
        job.miner_terminated = True
        job.finished_at = func.now()

    await db.delete(instance)
    await db.execute(
        text(
            "UPDATE instance_audit SET deletion_reason = 'miner initialized' WHERE instance_id = :instance_id"
        ),
        {"instance_id": instance_id},
    )
    await db.commit()
    await notify_deleted(instance)

    return {"instance_id": instance_id, "deleted": True}
