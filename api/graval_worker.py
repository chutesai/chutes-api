"""
GraVal node validation worker.
"""

import asyncio
import aiohttp
import binascii
import uuid
import random
import hashlib
import traceback
import re
import time
import base64
import backoff
import secrets
import orjson as json
from async_lru import alru_cache
from typing import List, Tuple
from pydantic import BaseModel
from loguru import logger
from ipaddress import ip_address
from api.config import settings
from api.util import (
    aes_encrypt,
    aes_decrypt,
    use_encryption_v2,
    use_encrypted_path,
    should_slurp_code,
    decrypt_envdump_cipher,
    get_resolved_ips,
    generate_ip_token,
)
from api.gpu import SUPPORTED_GPUS
from api.database import get_session
from api.node.schemas import Node
from api.chute.schemas import Chute, RollingUpdate
from api.instance.schemas import Instance
from api.fs_challenge.schemas import FSChallenge
from sqlalchemy import update, func, and_, not_
from sqlalchemy.future import select
from sqlalchemy.orm import joinedload, selectinload
from sqlalchemy.ext.asyncio import AsyncSession
from taskiq_redis import ListQueueBroker, RedisAsyncResultBackend
from watchtower import get_env_dump, get_expected_command
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
async def verify_device_info_challenge(
    devices, challenge, response, with_chutes: bool = False, opencl: bool = False
):
    """
    Verify a device info challenge.
    """
    url = f"https://chutes-graval-device-challenge.{settings.base_domain}/verify_device_hash"
    headers = {}
    if not with_chutes:
        if opencl:
            url = f"{settings.opencl_graval_url}/verify_device_challenge"
        else:
            url = f"{settings.graval_url}/verify_device_challenge"
    else:
        headers["Authorization"] = settings.codecheck_key
    logger.info(f"Verifying device info challenge hash with {url=}")
    async with aiohttp.ClientSession(raise_for_status=True) as session:
        async with session.post(
            url,
            json={
                "devices": devices,
                "challenge": challenge,
                "response": response,
            },
            headers=headers,
        ) as resp:
            return (await resp.json())["result"]


async def get_encryption_settings(
    node,
    path,
    data,
    with_chutes: bool = True,
    cuda: bool = True,
    seed: int = None,
    iterations: int = None,
):
    """
    Determine the chute to use for encryption with validator infra fallback/option.
    """
    # Determine which chute to call, if we are using chutes rather than a validator GPU.
    memory = SUPPORTED_GPUS.get(node.gpu_identifier, {}).get("memory", 140)
    suffix = "large" if memory > 48 else "small"
    suffix = f"cuda-{suffix}" if cuda else f"opencl-{suffix}"
    slug = f"chutes-graval-{suffix}"

    # Double check this chute is available.
    if slug not in await available_verification_chutes():
        with_chutes = False

    # Encrypt the payload.
    url = f"https://{slug}.{settings.base_domain}/{path}"
    if not with_chutes:
        if cuda:
            url = f"{settings.graval_url}/{path}"
        else:
            url = f"{settings.opencl_graval_url}/{path}"
    headers = {}
    if with_chutes:
        headers["Authorization"] = settings.codecheck_key
    payload = {
        "device_info": node.graval_dict(),
        "seed": seed if seed is not None else random.randint(0, 999999999999),
    }
    if not cuda:
        payload["iterations"] = iterations or 1
    if path == "encrypt":
        payload.update({"payload": data if isinstance(data, str) else json.dumps(data).decode()})
        if with_chutes:
            payload.update(
                {
                    "key": secrets.token_bytes(16).hex(),
                }
            )
    else:
        if with_chutes:
            payload.update({"data": data})
        else:
            payload.update({"payload": data})

    return url, payload, headers, with_chutes


@backoff.on_exception(
    backoff.constant,
    Exception,
    jitter=None,
    interval=3,
    max_tries=7,
)
async def graval_encrypt(
    node, payload, with_chutes=True, cuda=True, seed: int = None, iterations: int = None
):
    """
    Encrypt data via the GraVal PoW mechanism.
    """
    url, data, headers, chute = await get_encryption_settings(
        node,
        "encrypt",
        payload,
        with_chutes=with_chutes,
        cuda=cuda,
        seed=seed,
        iterations=iterations,
    )
    logger.info(f"Using {url=} for graval encryption")
    async with aiohttp.ClientSession(raise_for_status=True) as session:
        async with session.post(url, json=data, headers=headers, timeout=300) as resp:
            logger.success(f"Generated ciphertext for {node.uuid} via {url=}")
            if chute:
                body = await resp.json()
                result = json.loads(
                    decrypt_envdump_cipher(body["cipher"], bytes.fromhex(data["key"]))
                )
                return base64.b64encode(
                    bytes.fromhex(result["iv"]) + bytes.fromhex(result["ciphertext"])
                ).decode()
            return await resp.text()


@backoff.on_exception(
    backoff.constant,
    Exception,
    jitter=None,
    interval=10,
    max_tries=7,
)
async def graval_decrypt(
    node, payload, with_chutes=True, cuda=True, iterations: int = None, seed: int = None
):
    """
    Decrypt data via the GraVal PoW mechanism.
    """
    url, data, headers, chute = await get_encryption_settings(
        node,
        "decrypt",
        payload,
        with_chutes=with_chutes,
        cuda=cuda,
        seed=seed,
        iterations=iterations,
    )
    logger.info(f"Using {url=} for graval decryption")
    async with aiohttp.ClientSession(raise_for_status=True) as session:
        async with session.post(url, json=data, headers=headers) as resp:
            logger.success(f"Decrypted ciphertext from node {node.uuid} via {url=}")
            if chute:
                text = (await resp.json())["plaintext"]
                return decrypt_envdump_cipher(text, data["key"])
            return await resp.text()


async def generate_cipher(node):
    """
    Encrypt some data on the validator side and see if the miner can decrypt it.
    """
    plaintext = f"decrypt me please: {uuid.uuid4()}"
    graval_config = SUPPORTED_GPUS[node.gpu_identifier]["graval"]
    cipher = await graval_encrypt(
        node,
        plaintext,
        with_chutes=False,
        cuda=False,
        seed=node.seed,
        iterations=graval_config["iterations"],
    )
    logger.info(f"Generated ciphertext for {node.uuid} from {plaintext=} {cipher=}")
    return plaintext, cipher


async def check_encryption_challenge(
    node: Node, plaintext: str, ciphertext: str, uuids: list[str], timeout: int = None
) -> bool:
    """
    Send a single device decryption challenge.
    """
    url = f"http://{node.verification_host}:{node.verification_port}/decrypt"
    graval_config = SUPPORTED_GPUS[node.gpu_identifier]["graval"]
    payload = {
        "ciphertext": ciphertext,
        "seed": node.seed,
        "iterations": graval_config["iterations"],
        "device_index": node.device_index,
    }

    # Send the request and verify the response.
    error_message = None
    started_at = time.time()
    try:
        async with miner_client.post(
            node.miner_hotkey,
            url,
            payload=payload,
            timeout=timeout or int(graval_config["estimate"] * 1.35),
        ) as response:
            if response.status != 200:
                error_message = (
                    "Failed to perform decryption challenge: "
                    f"{response.status=} {await response.text()}"
                )
            else:
                response_text = (await response.json())["plaintext"]
                if response_text != plaintext:
                    error_message = (
                        f"Miner response '{response_text}' does not match ciphertext: '{plaintext}'"
                    )
                # XXX disabled for the time being.
                # elif timeout is None:
                #     delta = time.time() - started_at
                #     if (
                #         not graval_config["estimate"] * 0.70
                #         < delta
                #         < graval_config["estimate"] * 1.3
                #     ):
                #         error_message = (
                #             f"GraVal decryption challenge completed in {int(delta)} seconds, "
                #             f"but estimate is {graval_config['estimate']} seconds"
                #         )
    except Exception as exc:
        error_message = (
            "Unhandled exception performing miner decryption challenge after "
            f"{time.time() - started_at} seconds: {exc=}\n{traceback.format_exc()}"
        )

    # Store the reason for the verification error, upon failure.
    if error_message:
        logger.error(error_message)
        async with get_session() as session:
            # On failure, mark all of the GPUs as failed.
            await session.execute(
                update(Node)
                .where(Node.uuid.in_(uuids))
                .values({"verification_error": error_message})
            )
        await session.commit()
        return False

    return True


async def check_device_info_challenge(
    nodes: List[Node],
    url: str = None,
    purpose: str = "graval",
    with_chutes: bool = False,
    opencl: bool = False,
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
                    with_chutes=with_chutes,
                    opencl=opencl,
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

    # Check if the advertised IP matches outbound IP, disabled temporarily.
    # if not await verify_outbound_ip(nodes):
    #     return False, "Outbound IP address does not match advertised IP address"

    # Fast pass, do simple device info challenges.
    for _ in range(settings.device_info_challenge_count):
        if not await check_device_info_challenge(nodes, opencl=True):
            error_message = "one or more device info challenges failed"
            logger.warning(error_message)
            return False, error_message

    # Generate ciphertexts for each GPU for PoVW
    challenges = await asyncio.gather(*[generate_cipher(node) for node in nodes])

    # See if they decrypt properly - send one challenge first, which triggers the graval
    # init for all GPUs, then if/when that passes we can call the decryption endpoint
    # for all other GPUs concurrently and it will be virtually instant.
    if not await check_encryption_challenge(
        nodes[0], challenges[0][0], challenges[0][1], uuids=uuids
    ):
        error_message = "one or more decryption challenges failed"
        logger.warning(error_message)
        return False, error_message
    if len(nodes) > 1:
        successes = await asyncio.gather(
            *[
                check_encryption_challenge(
                    nodes[idx],
                    challenges[idx][0],
                    challenges[idx][1],
                    uuids=uuids,
                )
                for idx in range(1, len(nodes))
            ]
        )
        if not all(successes):
            error_message = "one or more decryption challenges failed"
            logger.warning(error_message)
            return False, error_message

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


@backoff.on_exception(
    backoff.constant,
    Exception,
    jitter=None,
    interval=5,
    max_tries=3,
)
async def _verify_instance_graval(instance: Instance) -> bool:
    """
    Check decryption ability of an instance via the _ping endpoint.
    """
    if not settings.graval_url:
        logger.info("GraVal disabled, skipping _verify_instance_graval...")
        return True

    # Generate a ciphertext for a random GPU on the instance.
    target_index = random.choice(list(range(len(instance.nodes))))
    target_node = instance.nodes[target_index]
    expected = str(uuid.uuid4())
    ciphertext = await graval_encrypt(
        target_node,
        expected,
        with_chutes=True,
        cuda=True,
        seed=target_node.seed,
    )

    # Format the payload to accomodate the old mechanism.
    bytes_ = base64.b64decode(ciphertext)
    iv = bytes_[:16]
    cipher = bytes_[16:]
    payload = {
        "hello": {
            "ciphertext": base64.b64encode(cipher).decode(),
            "iv": iv.hex(),
            "length": len(cipher),
            "device_id": target_index,
            "seed": target_node.seed,
        },
    }

    logger.info(f"Sending encrypted payload to _ping endpoint for graval verification: {payload=}")
    path = get_actual_path(instance, "_ping")
    async with miner_client.post(
        instance.miner_hotkey,
        f"http://{instance.host}:{instance.port}/{path}",
        payload,
        headers={"X-Chutes-Encrypted": "true"},
        timeout=12.0,
    ) as resp:
        resp.raise_for_status()
        if (await resp.json())["hello"] == expected:
            return True
        logger.warning(f"Expected {expected}, result: {await resp.json()}")
        return False


@backoff.on_exception(
    backoff.constant,
    Exception,
    jitter=None,
    interval=5,
    max_tries=3,
)
async def exchange_symmetric_key(instance: Instance) -> bool:
    """
    Create a new symmetric key and send it over to the miner via GraVal encryption.
    """
    # Generate a ciphertext for a random GPU on the instance.
    target_index = random.choice(list(range(len(instance.nodes))))
    target_node = instance.nodes[target_index]
    ciphertext = await graval_encrypt(
        target_node,
        instance.symmetric_key,
        with_chutes=True,
        cuda=True,
        seed=target_node.seed,
    )

    # Format the payload to accomodate the old mechanism.
    bytes_ = base64.b64decode(ciphertext)
    iv = bytes_[:16]
    cipher = bytes_[16:]
    payload = {
        "symmetric_key": {
            "ciphertext": base64.b64encode(cipher).decode(),
            "iv": iv.hex(),
            "length": len(cipher),
            "device_id": target_index,
            "seed": target_node.seed,
        },
    }

    # Send the encrypted symmetric key over to the instance.
    async with miner_client.post(
        instance.miner_hotkey,
        f"http://{instance.host}:{instance.port}/_exchange",
        payload,
        timeout=12.0,
    ) as resp:
        if resp.status != 404:
            resp.raise_for_status()

        # Make sure the encryption/decryption flow works properly.
        expected = str(uuid.uuid4())
        payload = aes_encrypt(json.dumps({"hello": expected}), instance.symmetric_key)
        iv = bytes.fromhex(payload[:32])
        logger.info(f"Sending {payload=} to _ping of {instance.instance_id=}")
        path = get_actual_path(instance, "_ping")
        async with miner_client.post(
            instance.miner_hotkey,
            f"http://{instance.host}:{instance.port}/{path}",
            payload,
            timeout=12.0,
        ) as decrypted_response:
            decrypted_response.raise_for_status()
            ciphertext = json.loads(await decrypted_response.read())["json"]
            plaintext = aes_decrypt(ciphertext, instance.symmetric_key, iv)
            logger.info(f"Plain text response from _ping on {instance.instance_id=}: {plaintext}")
            if json.loads(plaintext).get("hello") != expected:
                logger.warning(f"Expected {expected}, result: {plaintext}")
                return False

    logger.success(f"Successfully exchanged new symmetric key for {instance.instance_id=}")
    return True


@backoff.on_exception(
    backoff.constant,
    Exception,
    jitter=None,
    interval=5,
    max_tries=3,
)
async def _verify_filesystem_challenge(instance: Instance, challenge: FSChallenge) -> bool:
    """
    Check a single filesystem challenge on the remote instance.
    """
    logger.info(
        f"Sending filesystem challenge {challenge.filename=} {challenge.length=} {challenge.offset=} to {instance.instance_id=}"
    )
    payload = dict(
        filename=challenge.filename,
        offset=challenge.offset,
        length=challenge.length,
    )
    if use_encryption_v2(instance.chutes_version):
        payload = aes_encrypt(json.dumps(payload), instance.symmetric_key)
    path = get_actual_path(instance, "_fs_challenge")
    async with miner_client.post(
        instance.miner_hotkey,
        f"http://{instance.host}:{instance.port}/{path}",
        payload=payload,
        timeout=12.0,
    ) as resp:
        resp.raise_for_status()
        result = await resp.text()
        if result != challenge.expected:
            logger.warning(
                f"Expected {challenge.expected}, got {result}: {challenge.filename} [{challenge.offset}:{challenge.length}]"
            )
            return False
        logger.success(f"Successfully processed filesystem challenge: {challenge}")
        return True


async def check_envdump_command(instance):
    """
    Check the running command via chutes envdump, for supported chutes versions.
    """
    chute = instance.chute
    if not re.match(
        r"^[0-9]+\.(2\.(4[1-9]|[5-9][0-9])|[3-9]\.[0-9]+)$", chute.chutes_version or ""
    ):
        logger.warning(
            f"Unable to check envdump command line for {chute.chutes_version=} {chute.chute_id=}"
        )
        return True

    # Load the dump.
    dump = await get_env_dump(instance)
    process = dump[1] if isinstance(dump, list) else dump["process"]
    assert process["pid"] == 1
    command_line = re.sub(r"([^ ]+/)?python3?(\.[0-9]+)", "python", " ".join(process["cmdline"]))
    if command_line != get_expected_command(instance, instance.chute):
        logger.error(
            f"ENVDUMP: {instance.instance_id=} {instance.miner_hotkey=} {instance.chute_id=} running invalid process: {command_line}"
        )
        return False
    logger.success(
        f"ENVDUMP: {instance.instance_id=} {instance.miner_hotkey=} {instance.chute_id=} code validation success: {command_line=}"
    )
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
        timeout=12.0,
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
        expected = get_expected_command(instance, instance.chute)
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


async def _verify_filesystem(session: AsyncSession, instance: Instance) -> bool:
    """
    Perform a variety of filesystem challenges.
    """

    async def _safe_verify_one(challenge):
        try:
            return await _verify_filesystem_challenge(instance, challenge)
        except Exception as exc:
            logger.error(
                f"Failed _verify_filesystem_challenge 3 times: {exc}\n{traceback.format_exc()}"
            )
            return False

    subquery = (
        select(
            FSChallenge.challenge_id,
            FSChallenge.challenge_type,
            FSChallenge.image_id,
            func.row_number()
            .over(partition_by=FSChallenge.challenge_type, order_by=func.random())
            .label("rn"),
        )
        .filter(
            and_(
                FSChallenge.image_id == instance.chute.image_id,
                not_(FSChallenge.filename.endswith(f"/{instance.chute.filename}")),
                not_(FSChallenge.filename.endswith(".pyc")),
                not_(FSChallenge.filename.endswith(".pyo")),
            )
        )
        .subquery()
    )
    result = await session.execute(
        select(FSChallenge)
        .join(subquery, FSChallenge.challenge_id == subquery.c.challenge_id)
        .where(subquery.c.rn <= 10)
    )
    challenges = list(result.scalars().all())

    # Add in challenges for the chute code file.
    if should_slurp_code(instance.chute.chutes_version):
        try:
            if not await check_live_code(instance):
                logger.warning(f"Failed live app code check: {instance.instance_id=}")
                return False
            return True
        except Exception as exc:
            logger.error(
                f"Error checking live code: {instance.instance_id=} {instance.miner_hotkey=}: {exc}"
            )
            return False

    # Use sha256 checks.
    for _ in range(10):
        length = random.randint(10, len(instance.chute.code))
        offset = (
            0
            if length >= len(instance.chute.code)
            else random.randint(0, len(instance.chute.code) - length - 1)
        )
        challenges.append(
            FSChallenge(
                **{
                    "challenge_id": "000",
                    "image_id": instance.chute.image_id,
                    "filename": f"/app/{instance.chute.filename}",
                    "length": length,
                    "offset": offset,
                    "challenge_type": "chute_code",
                    "expected": hashlib.sha256(
                        instance.chute.code[offset : offset + length].encode()
                    ).hexdigest(),
                }
            )
        )

    results = await asyncio.gather(*[_safe_verify_one(challenge) for challenge in challenges])
    passed = sum(1 for r in results if r)
    logger.info(
        f"{instance.instance_id=} passed {passed} of {len(challenges)} filesystem challenges"
    )
    return passed == len(challenges)


@alru_cache(maxsize=1, ttl=600)
async def available_verification_chutes():
    """
    Find chutes enabling device challenges and/or graval PoW challenges.
    """
    options = [
        "chutes-graval-device-challenge",
        "chutes-graval-cuda-large",
        "chutes-graval-cuda-small",
        "chutes-graval-opencl-large",
        "chutes-graval-opencl-small",
    ]
    available = {key: False for key in options}
    async with get_session() as session:
        query = select(Chute).where(Chute.slug.in_(options))
        result = (await session.execute(query)).unique().scalars()
        for chute in result:
            available[chute.slug] = True
            logger.success(f"Enabling chutes-based challenges for {chute.slug=}")
    return available


@broker.task
async def verify_instance(instance_id: str):
    """
    Verify a single instance.
    """
    if not await settings.redis_client.setnx(f"verify:lock:{instance_id}", b"1"):
        logger.warning(f"Instance {instance_id} is already being verified...")
        return
    await settings.redis_client.expire(f"verify:lock:{instance_id}", 180)
    query = (
        select(Instance)
        .where(Instance.instance_id == instance_id)
        .options(joinedload(Instance.nodes), joinedload(Instance.chute))
    )
    async with get_session() as session:
        instance = (await session.execute(query)).unique().scalar_one_or_none()
        if not instance:
            logger.warning("Found no matching nodes, did they disappear?")
            await settings.redis_client.delete(f"verify:lock:{instance_id}")
            return

        if not use_encryption_v2(instance.chutes_version):
            # Legacy/encryption V1 tests.
            try:
                if not await _verify_instance_graval(instance):
                    logger.warning(f"{instance_id=} failed GraVal verification!")
                    instance.verification_error = "Failed one or more GraVal encryption challenges."
                    await session.commit()
                    await settings.redis_client.delete(f"verify:lock:{instance_id}")
                    return
            except Exception as exc:
                error_message = f"Failed to perform GraVal validation for {instance_id=}: {exc}\n{traceback.format_exc()}"
                logger.error(error_message)
                instance.verification_error = error_message
                await session.commit()
                await settings.redis_client.delete(f"verify:lock:{instance_id}")
                return
        else:
            # Encryption V2, create and exchange an AES key.
            try:
                await exchange_symmetric_key(instance)
            except Exception as exc:
                error_message = f"Failed to exchange symmetric key via GraVal encryption for {instance_id=}: {exc}\n{traceback.format_exc()}"
                logger.error(error_message)
                instance.verification_error = error_message
                await session.commit()
                await settings.redis_client.delete(f"verify:lock:{instance_id}")
                return

        # Filesystem test.
        try:
            if not await _verify_filesystem(session, instance):
                logger.warning(f"{instance_id=} failed filesystem verification!")
                instance.verification_error = "Failed one or more filesystem challenges."
                await session.commit()
                await settings.redis_client.delete(f"verify:lock:{instance_id}")
                return
        except Exception as exc:
            error_message = f"Failed to perform filesystem validation for {instance_id=}: {exc}\n{traceback.format_exc()}"
            logger.error(error_message)
            instance.verification_error = error_message
            await session.commit()
            await settings.redis_client.delete(f"verify:lock:{instance_id}")
            return

        # Device info challenges.
        path = get_actual_path(instance, "_device_challenge")
        url = f"http://{instance.host}:{instance.port}/{path}"

        # Use chutes itself for verification, if not the verification chutes, thereby federating validation.
        chutes_based = await available_verification_chutes()
        verify_with_chutes = False
        if (
            not instance.chute.slug.startswith(("chutes-graval-", "chutes-device-challenge"))
            and "chutes-graval-device-challenge" in chutes_based
        ):
            verify_with_chutes = True
            logger.success(
                f"Using chutes-based device challenge verification API for {instance.chute.slug=}"
            )

        for idx in range(settings.device_info_challenge_count):
            if not await check_device_info_challenge(
                instance.nodes, url=url, purpose="chutes", with_chutes=verify_with_chutes
            ):
                error_message = f"{instance_id=} failed one or more device info challenges"
                logger.warning(error_message)
                instance.verification_error = error_message
                await session.commit()
                await settings.redis_client.delete(f"verify:lock:{instance_id}")
                return
            else:
                logger.success(
                    f"{instance_id=} passed device info challenge {idx + 1} of {settings.device_info_challenge_count}"
                )

        # Looks good!
        logger.success(f"Instance {instance_id=} has passed verification!")
        instance.verified = True
        instance.last_verified_at = func.now()
        instance.verification_error = None
        await session.commit()
        await session.refresh(instance)

        # Notify the miner.
        try:
            event_data = {
                "reason": "instance_verified",
                "data": {
                    "instance_id": instance_id,
                    "miner_hotkey": instance.miner_hotkey,
                },
                "filter_recipients": [instance.miner_hotkey],
            }
            await settings.redis_client.publish("miner_broadcast", json.dumps(event_data).decode())
        except Exception as exc:
            # Allow exceptions here since the miner can also check.
            logger.warning(
                f"Error notifying miner that instance/deployment is verified: {exc}\n{traceback.format_exc()}"
            )

        # Broadcast the event.
        try:
            await settings.redis_client.publish(
                "events",
                json.dumps(
                    {
                        "reason": "instance_hot",
                        "message": f"Miner {instance.miner_hotkey} instance {instance.instance_id} chute {instance.chute_id} has been verified, now 'hot'!",
                        "data": {
                            "chute_id": instance.chute_id,
                            "miner_hotkey": instance.miner_hotkey,
                        },
                    }
                ).decode(),
            )
        except Exception as exc:
            logger.warning(f"Error broadcasting instance event: {exc}")
        await settings.redis_client.delete(f"verify:lock:{instance_id}")


@broker.task
async def handle_rolling_update(chute_id: str, version: str):
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
                event_data = {
                    "reason": "instance_deleted",
                    "message": f"Instance {instance.instance_id} of miner {instance.miner_hotkey} still bound to old version, deleting...",
                    "data": {
                        "chute_id": instance.chute_id,
                        "instance_id": instance.instance_id,
                        "miner_hotkey": instance.miner_hotkey,
                    },
                    "filter_recipients": [instance.miner_hotkey],
                }
                asyncio.create_task(
                    settings.redis_client.publish(
                        "miner_broadcast", json.dumps(event_data).decode()
                    )
                )
                logger.warning(
                    f"Instance did not respond to rolling update event: {instance.instance_id} of miner {instance.miner_hotkey}"
                )
        await session.delete(rolling_update)
