"""
GraVal node validation worker.
"""

import os
import subprocess
import tempfile
import asyncio
import aiohttp
import binascii
import uuid
import random
import traceback
import re
import time
import base64
import backoff
import secrets
import orjson as json
import pkg_resources
from typing import List, Tuple
from pydantic import BaseModel
from loguru import logger
from ipaddress import ip_address
from api.config import settings
from api.util import (
    semcomp,
    aes_encrypt,
    aes_decrypt,
    use_encrypted_path,
    get_resolved_ips,
    generate_ip_token,
    notify_deleted,
)
from api.gpu import SUPPORTED_GPUS
from api.database import get_session
from api.node.schemas import Node
from api.chute.schemas import Chute, RollingUpdate
from api.instance.schemas import Instance
from sqlalchemy import update, func
from sqlalchemy.future import select
from sqlalchemy.orm import selectinload
from taskiq_redis import ListQueueBroker, RedisAsyncResultBackend
from watchtower import get_expected_command, verify_expected_command, is_kubernetes_env, get_dump
import api.database.orms  # noqa
import api.miner_client as miner_client

broker = ListQueueBroker(url=settings.redis_url, queue_name="graval").with_result_backend(
    RedisAsyncResultBackend(redis_url=settings.redis_url, result_ex_time=3600)
)


class CipherChallenge(BaseModel):
    ciphertext: str
    iv: str
    length: int
    device_id: int
    seed: int


def generate_device_info_challenge(device_count: int):
    """
    Generate a device info challenge.
    """
    bytes_array = secrets.token_bytes(32)
    bytes_list = list(bytes_array)
    device_id = bytes_list[0]
    if device_id >= device_count and bytes_list[0] > 25:
        bytes_list[0] = 0
    return binascii.hexlify(bytes(bytes_list)).decode("ascii")


def get_actual_path(instance, path):
    """
    Get the real path, which may be encrypted.
    """
    if not use_encrypted_path(instance.chutes_version):
        return path
    path = "/" + path.lstrip("/")
    return aes_encrypt(path.ljust(24, "?"), instance.symmetric_key, hex_encode=True)


@backoff.on_exception(
    backoff.constant,
    Exception,
    jitter=None,
    interval=10,
    max_tries=7,
)
async def verify_device_info_challenge(devices, challenge, response):
    """
    Verify a device info challenge.
    """
    url = f"{settings.opencl_graval_url}/verify_device_challenge"
    logger.info(f"Verifying device info challenge hash with {url=}")
    async with aiohttp.ClientSession(raise_for_status=True) as session:
        async with session.post(
            url,
            json={
                "devices": devices,
                "challenge": challenge,
                "response": response,
            },
        ) as resp:
            return (await resp.json())["result"]


async def get_encryption_settings(
    node,
    data,
    seed: int = None,
    iterations: int = None,
):
    """
    Determine the chute to use for encryption with validator infra fallback/option.
    """
    url = f"{settings.opencl_graval_url}/encrypt"
    headers = {}
    payload = {
        "device_info": node.graval_dict(),
    }
    if seed:
        payload["seed"] = int(seed)
    payload["iterations"] = iterations or 1
    payload.update({"payload": data if isinstance(data, str) else json.dumps(data).decode()})
    return url, payload, headers


@backoff.on_exception(
    backoff.constant,
    Exception,
    jitter=None,
    interval=3,
    max_tries=7,
)
async def graval_encrypt(node, payload, seed: int = None, iterations: int = None):
    """
    Encrypt data via the GraVal PoW mechanism.
    """
    url, data, headers = await get_encryption_settings(
        node,
        payload,
        seed=seed,
        iterations=iterations,
    )
    logger.info(f"Using {url=} for graval encryption with {payload=}")
    async with aiohttp.ClientSession(raise_for_status=True) as session:
        async with session.post(url, json=data, headers=headers, timeout=300) as resp:
            logger.success(f"Generated ciphertext for {node.uuid} via {url=}")
            return await resp.text()


@backoff.on_exception(
    backoff.constant,
    Exception,
    jitter=None,
    interval=3,
    max_tries=7,
)
async def verify_proof(
    node,
    seed,
    work_product,
    index: int = 0,
):
    """
    Verify a miner's proof.
    """
    url = f"{settings.opencl_graval_url}/check_proof"
    payload = {
        "device_info": node.graval_dict(),
        "seed": int(seed),
        "work_product": work_product,
        "check_index": index,
    }
    logger.info(f"Checking proof validity from {node.uuid=} {node.miner_hotkey=} using {seed=}")
    async with aiohttp.ClientSession(raise_for_status=True) as session:
        async with session.post(url, json=payload, timeout=120) as resp:
            verified = (await resp.json())["result"]
            if verified:
                logger.success(
                    f"Successfully verified proof from {node.uuid=} [{node.name}] {node.miner_hotkey=} using {seed=}"
                )
            return (await resp.json())["result"]


async def generate_cipher(node):
    """
    Encrypt some data on the validator side and see if the miner can decrypt it.
    """
    plaintext = f"decrypt me please: {uuid.uuid4()}"
    graval_config = SUPPORTED_GPUS[node.gpu_identifier]["graval"]
    cipher = await graval_encrypt(
        node,
        plaintext,
        seed=node.seed,
        iterations=graval_config["iterations"],
    )
    logger.info(f"Generated ciphertext for {node.uuid} from {plaintext=} {cipher=}")
    return plaintext, cipher


async def verify_povw_challenge(nodes: list[Node]) -> bool:
    """
    Generate a povw challenge for the miner and verify the proof.
    """

    # Generate the challenge.
    graval_config = SUPPORTED_GPUS[nodes[0].gpu_identifier]["graval"]
    cipher_data = await asyncio.gather(
        *[
            generate_cipher(
                node,
            )
            for node in nodes
        ]
    )
    ciphertexts = [c[1].split("|")[-1] for c in cipher_data]
    cipher_map = {node.uuid: ciphertext for node, ciphertext in zip(nodes, ciphertexts)}
    plaintext = {node.uuid: c[0] for node, c in zip(nodes, cipher_data)}
    challenge = {
        "seed": int(nodes[0].seed),
        "iterations": graval_config["iterations"],
        "ciphertext": {gpu_uuid: ciphertext for gpu_uuid, ciphertext in cipher_map.items()},
    }

    # Send the challenge over to the miner.
    node = nodes[0]
    url = f"http://{node.verification_host}:{node.verification_port}/prove"
    logger.info(f"Sending PoVW challenge to {url=} for {node.miner_hotkey=}: {challenge=}")
    error_message = None
    started_at = time.time()
    try:
        async with miner_client.post(
            node.miner_hotkey,
            url,
            payload=challenge,
            timeout=int(graval_config["estimate"] * 2.5),
        ) as response:
            if response.status != 200:
                error_message = (
                    f"Miner failed to generate a proof and/or decrypt from {url=} {node.miner_hotkey=}: "
                    f"{response.status=} {await response.text()}"
                )
            else:
                # Verify the decrypted responses are what we'd expect.
                data = await response.json()
                if not all(
                    [
                        data["plaintext"].get(gpu_uuid) == plaintext[gpu_uuid]
                        for gpu_uuid in plaintext
                    ]
                ):
                    error_message = (
                        f"Miner responded with incorrect plaintext {url=} {node.miner_hotkey=}: "
                        f"expected={plaintext}, received={data['plaintext']}"
                    )

                # Check if the time taken to generate the proof matches what we'd expect.
                delta = time.time() - started_at
                if not graval_config["estimate"] * 0.55 < delta < graval_config["estimate"] * 1.7:
                    error_message = (
                        f"GraVal decryption challenge completed in {int(delta)} seconds, "
                        f"but estimate is {graval_config['estimate']} seconds: {url=} {node.miner_hotkey=}"
                    )
                else:
                    logger.success(
                        f"Miner successfully decrypted via PoVW in {delta} seconds, "
                        f"expected {graval_config['estimate']} seconds: {url=} {node.miner_hotkey=}"
                    )

                # Verify the proofs.
                verified = await asyncio.gather(
                    *[
                        verify_proof(
                            nodes[idx],
                            nodes[0].seed,
                            data["proof"][nodes[idx].uuid]["work_product"],
                            index=0,
                        )
                        for idx in range(len(nodes))
                    ]
                )
                if not all(verified):
                    error_message = "Miner proof verification failed!"
                else:
                    logger.success(
                        f"All miner proofs verified successfully: {url=} {node.miner_hotkey=}"
                    )

    except Exception as exc:
        error_message = (
            "Unhandled exception gathering miner PoVW challenge after "
            f"{time.time() - started_at} seconds: {exc=}\n{traceback.format_exc()}"
        )

    # On failure, store the reason and reset the seed.
    if error_message:
        logger.error(error_message)
        new_seed = random.randint(1, 2**63 - 1)
        async with get_session() as session:
            await session.execute(
                update(Node)
                .where(Node.uuid.in_([node.uuid for node in nodes]))
                .values({"seed": new_seed, "verification_error": error_message})
            )
        await session.commit()
        return False

    return True


async def check_device_info_challenge(
    nodes: List[Node],
    url: str = None,
    purpose: str = "graval",
) -> bool:
    """
    Send a single device info challenge.
    """
    if not url:
        url = f"http://{nodes[0].verification_host}:{nodes[0].verification_port}/info"
    error_message = None
    try:
        challenge = generate_device_info_challenge(len(nodes))
        async with miner_client.get(
            nodes[0].miner_hotkey,
            url,
            purpose,
            params={"challenge": challenge},
            timeout=12.0,
        ) as response:
            if response.status != 200:
                error_message = f"Failed to perform device info challenge: {response.status=} {await response.text()}"
            else:
                response = await response.text()
                assert await verify_device_info_challenge(
                    [node.graval_dict() for node in nodes],
                    challenge,
                    response,
                )
    except Exception as exc:
        error_message = f"Failed to perform decryption challenge: [unhandled exception] {exc} {traceback.format_exc()}"
    if error_message:
        logger.error(error_message)
        async with get_session() as session:
            await session.execute(
                update(Node)
                .where(Node.uuid.in_([node.uuid for node in nodes]))
                .values({"verification_error": error_message})
            )
        await session.commit()
        return False
    return True


async def verify_outbound_ip(nodes):
    """
    Check if the advertised IP matches the outbound IP (via remote token fetch).
    """
    addrs = []
    if len(set([f"{n.verification_host}:{n.verification_port}" for n in nodes])) > 1:
        error_message = "Multiple host/port pairs for a single server."
    else:
        addrs = []
        try:
            ip_address(nodes[0].verification_host)
            addrs = [nodes[0].verification_host]
        except ValueError:
            try:
                addrs = await asyncio.wait_for(get_resolved_ips(nodes[0].verification_host), 5.0)
            except ValueError:
                error_message = f"Could not resolve IP addresses for {nodes[0].verification_host}"
        if not addrs:
            error_message = "Unable to determine IP address for {nodes[0].verification_host}"
        url = f"http://{nodes[0].verification_host}:{nodes[0].verification_port}/remote_token"
        salt = secrets.token_hex(16)
        token_url = f"https://api.{settings.base_domain}/instances/token_check?salt={salt}"
        try:
            async with miner_client.post(
                nodes[0].miner_hotkey,
                url,
                payload={"token_url": token_url},
                timeout=15.0,
            ) as response:
                if response.status != 200:
                    error_message = (
                        f"Unable to fetch remote IP check token: {await response.text()}"
                    )
                token = await response.text()
                expected = [generate_ip_token(i, extra_salt=salt) for i in addrs]
                if any([e == token for e in expected]):
                    logger.success(f"Verified {nodes[0].verification_host} remote IP token check.")
                    return True
                error_message = (
                    f"Failed IP token check, expected one of {expected} but received {token}"
                )
        except Exception as exc:
            error_message = f"Unhandled exception fetching remote IP check token: {exc=}"
    if error_message:
        logger.warning(f"Failed outbound IP check: {error_message=}")
        async with get_session() as session:
            await session.execute(
                update(Node)
                .where(Node.uuid.in_([node.uuid for node in nodes]))
                .values({"verification_error": error_message})
            )
            await session.commit()
            return False
    return True


@broker.task
async def validate_gpus(uuids: List[str]) -> Tuple[bool, str]:
    """
    Validate GPUs.
    """
    nodes = None
    async with get_session() as session:
        if not (
            nodes := (await session.execute(select(Node).where(Node.uuid.in_(uuids))))
            .scalars()
            .unique()
            .all()
        ):
            logger.warning("Found no matching nodes, did they disappear?")
            return False, "nodes not found"

    # Re-order nodes...
    node_map = {node.uuid: node for node in nodes}
    nodes = [node_map[uuid] for uuid in uuids if uuid in node_map]
    if len(nodes) != len(uuids):
        error_message = f"Expecting {len(uuids)} nodes but only found {len(nodes)}"
        logger.warning(error_message)
        return False, error_message

    # XXX Check if the advertised IP matches outbound IP, disabled temporarily.
    # if not await verify_outbound_ip(nodes):
    #     return False, "Outbound IP address does not match advertised IP address"

    # Check the basic device info challenges first.
    for _ in range(settings.device_info_challenge_count):
        if not await check_device_info_challenge(nodes):
            error_message = "one or more device info challenges failed"
            logger.warning(error_message)
            return False, error_message

    # PoVW challenge.
    if not await verify_povw_challenge(nodes):
        return False, "failed povw challenge(s)"

    # Validation success.
    logger.success(
        f"Successfully performed GraVal PoVW decryption challenges on {len(nodes)} devices."
    )
    async with get_session() as session:
        await session.execute(
            update(Node)
            .where(Node.uuid.in_(uuids))
            .values({"verified_at": func.now(), "verification_error": None})
        )

    # Notify the miner.
    async def _notify_one(gpu_id):
        try:
            event_data = {
                "reason": "gpu_verified",
                "data": {
                    "gpu_id": gpu_id,
                    "miner_hotkey": nodes[0].miner_hotkey,
                },
                "filter_recipients": [nodes[0].miner_hotkey],
            }
            await settings.redis_client.publish("miner_broadcast", json.dumps(event_data).decode())
        except Exception as exc:
            # Allow exceptions here since the miner can also check.
            logger.warning(
                f"Error notifying miner that GPU is verified: {exc}\n{traceback.format_exc()}"
            )

    await asyncio.gather(*[_notify_one(gpu_id) for gpu_id in uuids])
    return True, None


async def check_envdump_command(instance):
    """
    Check the running command via chutes envdump, for supported chutes versions.
    """
    chute = instance.chute
    if semcomp(chute.chutes_version or "0.0.0", "0.2.53") < 0:
        logger.warning(
            f"Unable to check envdump command line for {chute.chutes_version=} {chute.chute_id=}"
        )
        return True

    # Load the dump.
    dump = await get_dump(instance)
    log_prefix = f"ENVDUMP: {instance.instance_id=} {instance.miner_hotkey=} {instance.chute_id=}"
    if not is_kubernetes_env(instance, dump, log_prefix=log_prefix):
        logger.error(f"{log_prefix} is not running a valid kubernetes environment")
        return False

    # Check the running command.
    try:
        await verify_expected_command(
            dump, chute, miner_hotkey=instance.miner_hotkey, seed=instance.nodes[0].seed, tls=False
        )
    except AssertionError as exc:
        logger.error(f"{log_prefix} running invalid process: {exc=}")
        return False

    return True


@backoff.on_exception(
    backoff.constant,
    Exception,
    jitter=None,
    interval=5,
    max_tries=3,
)
async def check_live_code(instance: Instance) -> bool:
    """
    Check the running command.
    """
    if not await check_envdump_command(instance):
        return False

    # Filesystem version.
    payload = {"path": "/proc/1/cmdline"}
    payload = aes_encrypt(json.dumps(payload), instance.symmetric_key)
    iv = payload[:32]
    path = aes_encrypt("/_slurp", instance.symmetric_key, hex_encode=True)
    async with miner_client.post(
        instance.miner_hotkey,
        f"http://{instance.host}:{instance.port}/{path}",
        payload,
        timeout=15.0,
    ) as resp:
        data = await resp.json()
        command_line = (
            base64.b64decode(
                json.loads(aes_decrypt(data["json"], instance.symmetric_key, iv=iv))["contents"]
            )
            .decode()
            .replace("\x00", " ")
            .strip()
        )
        command_line = re.sub(r"([^ ]+/)?python3?(\.[0-9]+)", "python", command_line)
        command_line = re.sub(r"([^ ]+/)?chutes\b", "chutes", command_line)
        expected = get_expected_command(
            instance.chute, miner_hotkey=instance.miner_hotkey, seed=instance.nodes[0].seed
        )
        if command_line != expected:
            logger.error(
                f"Failed PID 1 lookup evaluation: {instance.instance_id=} {instance.miner_hotkey=}:\n\t{command_line}\n\t{expected}"
            )
            return False

    # Double check the code.
    payload = {"path": f"/app/{instance.chute.filename}"}
    payload = aes_encrypt(json.dumps(payload), instance.symmetric_key)
    iv = payload[:32]
    path = aes_encrypt("/_slurp", instance.symmetric_key, hex_encode=True)
    async with miner_client.post(
        instance.miner_hotkey,
        f"http://{instance.host}:{instance.port}/{path}",
        payload,
        timeout=12.0,
    ) as resp:
        data = await resp.json()
        code = base64.b64decode(
            json.loads(aes_decrypt(data["json"], instance.symmetric_key, iv=iv))["contents"]
        )
        if code != instance.chute.code.encode():
            logger.error(
                f"Failed code slurp evaluation: {instance.instance_id=} {instance.miner_hotkey=}:\n{code}"
            )
            return False
    logger.success(
        f"Code and proc validation success: {instance.instance_id=} {instance.miner_hotkey=}"
    )
    return True


@broker.task
async def handle_rolling_update(chute_id: str, version: str, reason: str = "code change"):
    """
    Handle a rolling update event.
    """
    logger.info(f"Received rolling update task for {chute_id=}")
    async with get_session() as session:
        chute = (
            (
                await session.execute(
                    select(Chute)
                    .where(Chute.chute_id == chute_id)
                    .options(selectinload(Chute.instances))
                )
            )
            .unique()
            .scalar_one_or_none()
        )
        if not chute:
            logger.warning(f"Chute no longer found? {chute_id=}")
            return
        if not chute.instances:
            logger.info(f"No instances to update? {chute_id=}")
            return

    # Calculate sleep per instance so we finish within 45 minutes.
    max_duration = 60 * 45
    sleep_per_instance = int(max_duration / len(chute.instances))
    if not sleep_per_instance:
        sleep_per_instance = 1

    # Cap sleep time per instance to 5 minutes per instance.
    sleep_per_instance = min(300, sleep_per_instance)

    # Iterate through instances slowly to avoid crashing the entire chute.
    logger.info(
        f"Triggering update for {len(chute.instances)} instances of {chute.chute_id=} {chute.name=}"
    )
    for inst in chute.instances:
        # Make sure this rolling update is still valid, and the instance still exists.
        async with get_session() as session:
            instance = (
                (
                    await session.execute(
                        select(Instance).where(Instance.instance_id == inst.instance_id)
                    )
                )
                .unique()
                .scalar_one_or_none()
            )
            rolling_update = (
                (
                    await session.execute(
                        select(RollingUpdate).where(RollingUpdate.chute_id == chute_id)
                    )
                )
                .unique()
                .scalar_one_or_none()
            )
            if not instance:
                logger.warning(
                    f"Instance {inst.instance_id} no longer exists, skipping rolling update of {chute_id=} {version=}"
                )
                continue
            if not rolling_update or rolling_update.new_version != version:
                logger.warning(f"Rolling update is now defunct {chute_id=} {version=}")
                return

        # Send the event.
        try:
            event_data = {
                "reason": "rolling_update",
                "data": {
                    "chute_id": chute_id,
                    "instance_id": instance.instance_id,
                    "miner_hotkey": instance.miner_hotkey,
                    "old_version": rolling_update.old_version,
                    "new_version": rolling_update.new_version,
                    "reason": reason,
                },
                "filter_recipients": [instance.miner_hotkey],
            }
            await settings.redis_client.publish("miner_broadcast", json.dumps(event_data).decode())
            logger.success(
                f"Sent a notification to instance {instance.instance_id} of miner {instance.miner_hotkey}"
            )
        except Exception:
            # Allow exceptions here since the miner can also check.
            logger.warning(
                f"Error notifying miner {instance.miner_hotkey} about rolling update of {instance.instance_id=} for {chute.name=}"
            )

        # Wait before notifying the next miner.
        await asyncio.sleep(sleep_per_instance)

    # Once finished, clean up all instances still bound to the old version.
    async with get_session() as session:
        rolling_update = (
            (await session.execute(select(RollingUpdate).where(RollingUpdate.chute_id == chute_id)))
            .unique()
            .scalar_one_or_none()
        )
        if not rolling_update or rolling_update.new_version != version:
            return
        chute = (
            (
                await session.execute(
                    select(Chute)
                    .where(Chute.chute_id == chute_id)
                    .options(selectinload(Chute.instances))
                )
            )
            .unique()
            .scalar_one_or_none()
        )
        if not chute:
            return
        for instance in chute.instances:
            if instance.version != version:
                await session.delete(instance)
                await notify_deleted(
                    instance,
                    message=f"Instance {instance.instance_id} of miner {instance.miner_hotkey} still bound to old version, deleting...",
                )
                logger.warning(
                    f"Instance did not respond to rolling update event: {instance.instance_id} of miner {instance.miner_hotkey}"
                )
        await session.delete(rolling_update)


@broker.task
async def generate_fs_hash(
    image_id: str, patch_version: str, seed: int, sparse: bool, exclude_path: str
):
    """
    Use the new cfsv mechanism to generate the expected filesystem hash for a given image/seed pair.
    """
    if not os.getenv("CFSV_OP"):
        return "__disabled__"

    chutes_location = pkg_resources.get_distribution("chutes").location
    cfsv_path = os.path.join(chutes_location, "chutes", "cfsv")
    mode = "sparse" if sparse else "full"
    seed_str = str(seed)

    # Make sure our FS datamap is cached.
    cache_path = f"/tmp/{image_id}.{patch_version}.data"
    if not os.path.exists(cache_path):
        logger.info(f"Downloading data file for image_id={image_id}, patch_version={patch_version}")
        s3_key = f"image_hash_blobs/{image_id}/{patch_version}.data"
        try:
            temp_fd, temp_path = tempfile.mkstemp(dir="/tmp", prefix=f"{image_id}.{patch_version}.")
            os.close(temp_fd)
            try:
                async with settings.s3_client() as s3:
                    await s3.download_file(settings.storage_bucket, s3_key, temp_path)
                os.rename(temp_path, cache_path)
                logger.info(
                    f"Successfully cached data file to {cache_path} for {image_id=} {patch_version=}"
                )
            except Exception:
                if os.path.exists(temp_path):
                    os.unlink(temp_path)
                raise
        except Exception as e:
            logger.error(f"Failed to download data file from S3: {e}")
            raise Exception(f"Failed to download image data from S3: {e}")
    else:
        logger.info(f"Using cached data file at {cache_path}")

    # Now generate the hash.
    cmd = [cfsv_path, "validate", seed_str, mode, cache_path, exclude_path]
    logger.info(
        f"Generating filesystem hash for {image_id=} {patch_version=} using seed={seed_str} and mode={mode}"
    )
    proc = await asyncio.create_subprocess_exec(
        *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
    )
    stdout, stderr = await proc.communicate()
    if proc.returncode != 0:
        logger.error(f"cfsv validate failed: {stderr.decode()}")
        raise subprocess.CalledProcessError(proc.returncode, cmd, stderr.decode())

    # Extract the hash from the output.
    fsv_hash = None
    for line in stdout.decode().strip().split("\n"):
        if line.startswith("RESULT:"):
            fsv_hash = line.split("RESULT:")[1].strip()
            logger.info(f"Filesystem verification hash: {fsv_hash}")
            break
    if not fsv_hash:
        logger.warning(
            "Failed to extract filesystem verification hash from cfsv output: {stdout.decode()}"
        )
        raise Exception("No RESULT line found in cfsv output: {stdout.decode()}")
    return fsv_hash
