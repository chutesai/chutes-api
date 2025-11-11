"""
Routes for instances.
"""

import csv
from io import StringIO
import os
import uuid
import base64
import ctypes
import traceback
import random
import socket
import secrets
import asyncio
import orjson as json  # noqa
from api.image.util import get_inspecto_hash
import api.miner_client as miner_client
from loguru import logger
from typing import Optional, Tuple
from datetime import timedelta
from fastapi import APIRouter, Depends, HTTPException, Response, status, Header, Request
from sqlalchemy import select, text, func, update, and_
from sqlalchemy.orm import joinedload
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.exc import IntegrityError
from sqlalchemy.dialects.postgresql import insert
from api.gpu import SUPPORTED_GPUS
from api.database import get_db_session, generate_uuid, get_session
from api.config import settings
from api.constants import (
    HOTKEY_HEADER,
    AUTHORIZATION_HEADER,
    PRIVATE_INSTANCE_BONUS,
)
from api.node.schemas import Node
from api.payment.util import decrypt_secret
from api.node.util import get_node_by_id
from api.chute.schemas import Chute, NodeSelector
from api.secret.schemas import Secret
from api.image.schemas import Image  # noqa
from api.instance.schemas import (
    GravalLaunchConfigArgs,
    TeeLaunchConfigArgs,
    Instance,
    instance_nodes,
    LaunchConfig,
    LaunchConfigArgs,
)
from api.job.schemas import Job
from api.instance.util import (
    create_launch_jwt_v2,
    generate_fs_key,
    get_instance_by_chute_and_id,
    create_launch_jwt,
    create_job_jwt,
    load_launch_config_from_jwt,
    invalidate_instance_cache,
)
from api.server.service import check_server_ownership, validate_and_consume_nonce, validate_request_nonce, verify_gpu_evidence, verify_server
from api.user.schemas import User
from api.user.service import get_current_user, chutes_user_id, subnet_role_accessible
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
    load_shared_object,
    has_legacy_private_billing,
)
from api.bounty.util import check_bounty_exists, delete_bounty, claim_bounty
from starlette.responses import StreamingResponse
from api.graval_worker import graval_encrypt, verify_proof, generate_fs_hash
from watchtower import is_kubernetes_env, verify_expected_command

router = APIRouter()

INSPECTO = load_shared_object("chutes", "chutes-inspecto.so")
INSPECTO.verify_hash.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_char_p]
INSPECTO.verify_hash.restype = ctypes.c_char_p

NETNANNY = load_shared_object("chutes", "chutes-netnanny.so")
NETNANNY.verify.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_uint8]
NETNANNY.verify.restype = ctypes.c_int


async def _load_chute(db, chute_id: str) -> Chute:
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

    # Check if scaling is allowed based on target count.
    if active_count >= target_count:
        logger.warning(
            f"SCALELOCK: chute {chute_id=} {chute.name} has reached target capacity: "
            f"{current_count=}, {active_count=}, {target_count=}, {hotkey_count=}"
        )
        raise HTTPException(
            status_code=status.HTTP_423_LOCKED,
            detail=f"Chute {chute_id} has reached its target capacity of {target_count} instances.",
        )


async def _check_scalable_private(db, chute, miner):
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

async def _validate_node(db, chute, node_id: str, hotkey: str) -> Node:
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
    result = await db.execute(
        select(instance_nodes.c.instance_id).where(instance_nodes.c.node_id == node_id)
    )
    existing_instance_id = result.scalar()
    if existing_instance_id:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"GPU {node_id} is already assigned to instance: {existing_instance_id}",
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


async def _validate_nodes(db, chute, node_ids: list[str], hotkey: str, instance: Instance) -> list[Node]:
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

        # Create the association record, handling dupes.
        stmt = (
            insert(instance_nodes)
            .values(instance_id=instance.instance_id, node_id=node_id)
            .on_conflict_do_nothing(index_elements=["node_id"])
        )
        result = await db.execute(stmt)
        if result.rowcount == 0:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=f"Node {node_id} is already assigned to another instance",
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
    
@router.get("/reconciliation_csv")
async def get_instance_reconciliation_csv(
    db: AsyncSession = Depends(get_db_session),
):
    """
    Get all instance audit instance_id, deleted_at records to help reconcile audit data.
    """
    query = """
        SELECT
            instance_id,
            deleted_at
        FROM instance_audit
        WHERE deleted_at IS NOT NULL
        AND activated_at IS NOT NULL
    """
    output = StringIO()
    writer = csv.writer(output)
    result = await db.execute(text(query))
    writer.writerow([col for col in result.keys()])
    writer.writerows(result)
    return Response(
        content=output.getvalue(),
        media_type="text/csv",
        headers={"Content-Disposition": 'attachment; filename="audit-reconciliation.csv"'},
    )

async def _validate_launch_config_env(
    db: AsyncSession,
    launch_config: LaunchConfig,
    chute: Chute,
    args: GravalLaunchConfigArgs,
    log_prefix: str
):
    from chutes.envdump import DUMPER
    # Verify, decrypt, parse the envdump payload.
    if "ENVDUMP_UNLOCK" in os.environ:
        code = None
        try:
            dump = DUMPER.decrypt(launch_config.env_key, args.env)
            if semcomp(chute.chutes_version or "0.0.0", "0.3.61") < 0:
                code_data = DUMPER.decrypt(launch_config.env_key, args.code)
                code = base64.b64decode(code_data["content"]).decode()
        except Exception as exc:
            logger.error(
                f"Attempt to claim {launch_config.config_id=} failed, invalid envdump payload received: {exc}"
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
            if semcomp(chute.chutes_version or "0.0.0", "0.3.61") < 0:
                assert code == chute.code, f"Incorrect code:\n{code=}\n{chute.code=}"
        except AssertionError as exc:
            logger.error(f"Attempt to claim {launch_config.config_id=} failed, invalid command: {exc}")
            launch_config.failed_at = func.now()
            launch_config.verification_error = f"Invalid command: {exc}"
            await db.commit()
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"You are not running the correct command, sneaky devil: {exc}",
            )

        # K8S check.
        if not is_kubernetes_env(
            chute, dump, log_prefix=log_prefix, standard_template=chute.standard_template
        ):
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
        
async def _validate_launch_config_inspecto(
    db: AsyncSession,
    launch_config: LaunchConfig,
    chute: Chute,
    args: GravalLaunchConfigArgs,
    log_prefix: str
):
    if semcomp(chute.chutes_version, "0.3.50") >= 0:
        # Inspecto
        if not args.inspecto:
            logger.error(f"{log_prefix} no inspecto hash provided")
            launch_config.failed_at = func.now()
            launch_config.verification_error = "Failed inspecto environment/lib verification."
            await db.commit()
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=launch_config.verification_error,
            )

        enforce_inspecto = "PS_OP" in os.environ
        inspecto_valid = True
        fail_reason = None
        if enforce_inspecto:
            inspecto_hash = await get_inspecto_hash(chute.image_id)
            if not inspecto_hash:
                logger.info(f"INSPECTO: image_id={chute.image_id} has no inspecto hash; allowing.")
                inspecto_valid = True
            else:
                if not args.inspecto:
                    inspecto_valid = False
                    fail_reason = "missing args.inspecto hash!"
                else:
                    raw = INSPECTO.verify_hash(
                        inspecto_hash.encode("utf-8"),
                        launch_config.config_id.encode("utf-8"),
                        args.inspecto.encode("utf-8"),
                    )
                    logger.info(
                        f"INSPECTO: verify_hash({inspecto_hash=}, {launch_config.config_id=}, {args.inspecto=}) -> {raw=}",
                    )
                    if not raw:
                        inspecto_valid = False
                        fail_reason = "inspecto returned NULL"
                    else:
                        try:
                            payload = json.loads(raw.decode("utf-8"))
                        except Exception as e:
                            inspecto_valid = False
                            fail_reason = f"inspecto returned non-JSON: {e}"
                        else:
                            if not payload.get("verified"):
                                inspecto_valid = False
                                fail_reason = f"inspecto verification failed: {payload}"
        if not inspecto_valid:
            logger.error(f"{log_prefix} has invalid inspecto verification: {fail_reason}")
            launch_config.failed_at = func.now()
            launch_config.verification_error = "Failed inspecto environment/lib verification."
            await db.commit()
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=launch_config.verification_error,
            )
        
async def _validate_launch_config_filesystem(
    db: AsyncSession,
    launch_config: LaunchConfig,
    chute: Chute,
    args: GravalLaunchConfigArgs
):
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

async def _validate_launch_config_instance(
    db: AsyncSession,
    request: Request,
    args: LaunchConfigArgs,
    launch_config: LaunchConfig,
    chute: Chute,
    log_prefix: str
) -> Tuple[LaunchConfig, list[Node], Instance]:

    miner = await _check_blacklisted(db, launch_config.miner_hotkey)

    config_id = launch_config.config_id

    # Generate a tentative instance ID.
    new_instance_id = generate_uuid()

    # Re-check scalable...
    if not launch_config.job_id:
        if (
            not chute.public
            and not has_legacy_private_billing(chute)
            and chute.user_id != await chutes_user_id()
        ):
            await _check_scalable_private(db, chute, miner)
        else:
            await _check_scalable(db, chute, launch_config.miner_hotkey)

    # IP matches?
    x_forwarded_for = request.headers.get("X-Forwarded-For")
    actual_ip = x_forwarded_for.split(",")[0] if x_forwarded_for else request.client.host
    if actual_ip != args.host:
        logger.warning(
            f"Instance with {launch_config.config_id=} {launch_config.miner_hotkey=} EGRESS INGRESS mismatch!: {actual_ip=} {args.host=}"
        )
        if launch_config.job_id:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Egress and ingress IPs much match for jobs: {actual_ip} vs {args.host}",
            )

    # Uniqueness of host/miner_hotkey.
    result = await db.scalar(
        select(Instance).where(
            and_(
                Instance.host == launch_config.host,
                Instance.miner_hotkey != launch_config.miner_hotkey,
            )
        )
    )
    if result:
        logger.warning(
            f"{launch_config.config_id=} {launch_config.miner_hotkey=} attempted to use host already used by another miner!"
        )
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Host {launch_config.host} is already assigned to at least one other miner_hotkey.",
        )

    if semcomp(chute.chutes_version, "0.3.50") >= 0:
        if not args.run_path or (
            chute.standard_template == "vllm"
            and os.path.dirname(args.run_path)
            != "/usr/local/lib/python3.12/dist-packages/chutes/entrypoint"
        ):
            logger.error(f"{log_prefix} has tampered with paths!")
            launch_config.failed_at = func.now()
            launch_config.verification_error = "Env tampering detected!"
            await db.commit()
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=launch_config.verification_error,
            )

        # NetNanny (match egress config and hash).
        nn_valid = True
        if chute.allow_external_egress != args.egress or not args.netnanny_hash:
            nn_valid = False
        else:
            if not NETNANNY.verify(
                launch_config.config_id.encode(),
                args.netnanny_hash.encode(),
                1,
            ):
                logger.error(
                    f"{log_prefix} netnanny hash mismatch for {launch_config.config_id=} and {chute.allow_external_egress=}"
                )
                nn_valid = False
            else:
                logger.success(
                    f"{log_prefix} netnanny hash challenge success: for {launch_config.config_id=} and {chute.allow_external_egress=} {args.netnanny_hash=}"
                )
        if not nn_valid:
            logger.error(
                f"{log_prefix} has tampered with netnanny? {args.netnanny_hash=} {args.egress=} {chute.allow_external_egress=}"
            )
            launch_config.failed_at = func.now()
            launch_config.verification_error = "Failed netnanny validation."
            await db.commit()
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=launch_config.verification_error,
            )

    await _validate_launch_config_filesystem(
        db, launch_config, chute, args
    )

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
        inspecto=args.inspecto,
        env_creation=args.model_dump(),
    )
    if launch_config.job_id or (
        not chute.public
        and not has_legacy_private_billing(chute)
        and chute.user_id != await chutes_user_id()
    ):
        instance.compute_multiplier *= PRIVATE_INSTANCE_BONUS
        instance.billed_to = chute.user_id

    # Add chute boost.
    if chute.boost is not None and chute.boost > 0 and chute.boost <= 20:
        instance.compute_multiplier *= chute.boost

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
    if len(set([node["uuid"] for node in args.gpus])) != len(args.gpus):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Duplicate GPUs in request!",
        )
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

    return launch_config, nodes, instance

async def _validate_graval_launch_config_instance(
    config_id: str,
    args: GravalLaunchConfigArgs,
    request: Request,
    db: AsyncSession,
    authorization: str,
) -> Tuple[LaunchConfig, list[Node], Instance]:

    token = authorization.strip().split(" ")[-1]
    launch_config = await load_launch_config_from_jwt(db, config_id, token)
    chute = await _load_chute(db, launch_config.chute_id)
    log_prefix = f"ENVDUMP: {launch_config.config_id=} {chute.chute_id=}"

    # This does change order from previous graval only implementation
    # If want to preserve order need to split up final shared config check
    await _validate_launch_config_env(
        db, launch_config, chute, args, log_prefix    
    )

    await _validate_launch_config_inspecto(
        db, launch_config, chute, args, log_prefix
    )

    return await _validate_launch_config_instance(
        db, request, args, launch_config, chute, log_prefix
    )

async def _validate_tee_launch_config_instance(
    config_id: str,
    args: TeeLaunchConfigArgs,
    request: Request,
    db: AsyncSession,
    authorization: str,
) -> Tuple[LaunchConfig, list[Node], Instance]:
    
    token = authorization.strip().split(" ")[-1]
    launch_config = await load_launch_config_from_jwt(db, config_id, token)
    chute = await _load_chute(db, launch_config.chute_id)
    log_prefix = f"ENVDUMP: {launch_config.config_id=} {chute.chute_id=}"

    return await _validate_launch_config_instance(
        db, request, args, launch_config, chute, log_prefix
    )


@router.get("/launch_config")
async def get_launch_config(
    chute_id: str,
    server_id: Optional[str] = None,
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
            await _check_scalable_private(db, chute, miner)
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

    if server_id:
        server = await check_server_ownership(db, server_id, hotkey)

        if not server:
            logger.error(f"Server {server_id} does not exist or does not belong to miner {hotkey=}.")
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Your hotkey does not own server {server_id=} or server does not exist.",
            )
        env_type = "tee" if server.is_tee else "graval"
    else:
        # Fallback to graval for backwards compatability
        env_type = "graval"


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
            env_type=env_type,
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
    token = None
    if semcomp(chute.chutes_version or "0.0.0", "0.3.61") >= 0:
        token = create_launch_jwt_v2(
            launch_config, egress=chute.allow_external_egress, disk_gb=disk_gb
        )
    else:
        token = create_launch_jwt(launch_config, disk_gb=disk_gb)


    return {
        "token": token,
        "config_id": launch_config.config_id,
    }


@router.post("/launch_config/{config_id}/attest")
async def validate_tee_launch_config_instance(
    config_id: str,
    args: TeeLaunchConfigArgs,
    request: Request,
    db: AsyncSession = Depends(get_db_session),
    authorization: str = Header(None, alias=AUTHORIZATION_HEADER),
    expected_nonce: str = Depends(validate_request_nonce())
):
    launch_config, nodes, instance = await _validate_tee_launch_config_instance(
        config_id, args, request, db, authorization
    )

    _validate_launch_config_not_expired(launch_config)

    # Store the launch config
    await db.commit()
    await db.refresh(launch_config)

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

    await verify_gpu_evidence(args.gpu_evidence, expected_nonce)

    request_body = await request.json()

    # Reload instance with chute relationship for filesystem validation
    # Lazy load fails 
    stmt = (
        select(Instance)
        .where(Instance.instance_id == instance.instance_id)
        .options(
            joinedload(Instance.chute).
            joinedload(Chute.image),
            joinedload(Instance.job),
        )
    )
    instance = (await db.execute(stmt)).scalar_one()

    # Filesystem integrity checks for < 0.3.1
    await _validate_legacy_filesystem(db, instance, launch_config, request_body)

    # Everything checks out.
    launch_config.verified_at = func.now()
    await _verify_job_ports(db, instance)
    await _mark_instance_verified(db, instance, launch_config)
    return_value = await _build_launch_config_verified_response(db, instance, launch_config)
    return_value["symmetric_key"] = instance.symmetric_key

    await db.refresh(instance)
    asyncio.create_task(notify_verified(instance))
    return return_value

@router.post("/launch_config/{config_id}")
async def claim_launch_config(
    config_id: str,
    args: GravalLaunchConfigArgs,
    request: Request,
    db: AsyncSession = Depends(get_db_session),
    authorization: str = Header(None, alias=AUTHORIZATION_HEADER),
):
    launch_config, nodes, instance = await _validate_graval_launch_config_instance(
        config_id, args, request, db, authorization
    )

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
        # If a bounty exists for this chute, claim it.
        bounty = await claim_bounty(instance.chute_id)
        if bounty is None:
            bounty = 0
        if bounty:
            instance.bounty = True

        # Verify egress.
        # net_success = True
        # if semcomp(chute.chutes_version, "0.3.56") >= 0:
        #    from conn_prober import check_instance_connectivity

        #    _, net_success = await check_instance_connectivity(instance, delete_on_failure=False)
        # if not net_success:
        #    reason = "Instance has failed network connectivity probes, based on allow_external_egress flag"
        #    logger.warning(reason)
        # XXX TODO
        # await db.delete(instance)
        # await asyncio.create_task(notify_deleted(instance))
        # await db.execute(
        #    text(
        #        "UPDATE instance_audit SET deletion_reason = :reason WHERE instance_id = :instance_id"
        #    ),
        #    {"instance_id": instance.instance_id, "reason": reason},
        # )
        # raise HTTPException(
        #    status_code=status.HTTP_403_FORBIDDEN,
        #    detail=reason,
        # )

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
        await db.commit()
        await invalidate_instance_cache(instance.chute_id, instance_id=instance.instance_id)
        await delete_bounty(chute.chute_id)
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

def _validate_launch_config_not_expired(launch_config):
    # Validate the launch config.
    config_id = launch_config.config_id
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
    
async def _validate_legacy_filesystem(
    db: AsyncSession, instance: Instance, launch_config: LaunchConfig, response_body
):
    config_id = launch_config.config_id
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

async def _verify_job_ports(db: AsyncSession, instance: Instance):
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
        await db.refresh(job)

async def _mark_instance_verified(db: AsyncSession, instance: Instance, launch_config: LaunchConfig):
    # Can't do this via the instance attrs directly, circular dependency :/
    await db.execute(
        text(
            "UPDATE instances SET verified = true, verification_error = null, last_verified_at = now() WHERE instance_id = :instance_id"
        ),
        {"instance_id": instance.instance_id},
    )

    await db.commit()
    await db.refresh(launch_config)

async def _build_launch_config_verified_response(
    db: AsyncSession, instance: Instance, launch_config: LaunchConfig
):
    
    return_value = {
        "chute_id": launch_config.chute_id,
        "instance_id": instance.instance_id,
        "verified_at": launch_config.verified_at.isoformat(),
    }
    if semcomp(instance.chutes_version or "0.0.0", "0.3.61") >= 0:
        return_value["code"] = instance.chute.code
        return_value["fs_key"] = generate_fs_key(launch_config),
        if not instance.chute.encrypted_fs:
            return_value["efs"] = True
    if instance.job:
        job_token = create_job_jwt(instance.job.job_id)
        return_value.update(
            {
                "job_id": instance.job.job_id,
                "job_method": instance.job.method,
                "job_data": instance.job.job_args,
                "job_status_url": f"https://api.{settings.base_domain}/jobs/{instance.job.job_id}?token={job_token}",
            }
        )

    # Secrets, e.g. private HF tokens etc.
    secrets = (
        (await db.execute(select(Secret).where(Secret.purpose == launch_config.chute_id)))
        .unique()
        .scalars()
        .all()
    )
    if secrets:
        return_value["secrets"] = {}
        for secret in secrets:
            value = await decrypt_secret(secret.value)
            return_value["secrets"][secret.key] = value

    return_value["activation_url"] = (
        f"https://api.{settings.base_domain}/instances/launch_config/{launch_config.config_id}/activate"
    )

    return return_value

@router.put("/launch_config/{config_id}")
async def verify_launch_config_instance(
    config_id: str,
    request: Request,
    db: AsyncSession = Depends(get_db_session),
    authorization: str = Header(None, alias=AUTHORIZATION_HEADER),
):
    token = authorization.strip().split(" ")[-1]
    launch_config = await load_launch_config_from_jwt(db, config_id, token, allow_retrieved=True)

    _validate_launch_config_not_expired(launch_config)

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

    await _validate_legacy_filesystem(db, instance, launch_config, response_body)

    # Everything checks out.
    launch_config.verified_at = func.now()
    await _verify_job_ports(db, instance)
    await _mark_instance_verified(db, instance, launch_config)
    return_value = await _build_launch_config_verified_response(db, instance, launch_config)
    

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
    Fetch the raw kubernetes pod logs, but only if the chute is private.

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
    if instance.chute.user_id != current_user.user_id or instance.chute.public:
        if not subnet_role_accessible(instance.chute, current_user, admin=True):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="You may only view logs for your own (private) chutes.",
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