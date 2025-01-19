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
import backoff
import secrets
import orjson as json
from typing import List, Tuple
from pydantic import BaseModel
from loguru import logger
from api.config import settings
from api.database import get_session
from api.node.schemas import Node
from api.instance.schemas import Instance
from api.fs_challenge.schemas import FSChallenge
from sqlalchemy import update, func, and_, not_
from sqlalchemy.future import select
from sqlalchemy.ext.asyncio import AsyncSession
from taskiq_redis import ListQueueBroker, RedisAsyncResultBackend
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
    async with aiohttp.ClientSession(raise_for_status=True) as session:
        async with session.post(
            f"{settings.graval_url}/verify_device_challenge",
            json={
                "devices": devices,
                "challenge": challenge,
                "response": response,
            },
        ) as resp:
            return (await resp.json())["result"]


@backoff.on_exception(
    backoff.constant,
    Exception,
    jitter=None,
    interval=10,
    max_tries=7,
)
async def generate_cipher(node):
    """
    Encrypt some data on the validator side and see if the miner can decrypt it.
    """
    plaintext = f"decrypt me please: {uuid.uuid4()}"
    async with aiohttp.ClientSession(raise_for_status=True) as session:
        async with session.post(
            f"{settings.graval_url}/encrypt",
            json={
                "payload": {
                    "plaintext": plaintext,
                },
                "device_info": node.graval_dict(),
                "device_id": node.device_index,
                "seed": node.seed,
            },
        ) as resp:
            data = (await resp.json())["plaintext"]
            logger.info(f"Generated ciphertext for {node.uuid} from {plaintext=} {data=}")
            return plaintext, CipherChallenge(
                ciphertext=data["ciphertext"],
                iv=data["iv"],
                length=data["length"],
                device_id=node.device_index,
                seed=node.seed,
            )


async def check_encryption_challenge(
    node: Node, challenge: CipherChallenge, plaintext: str
) -> bool:
    """
    Send a single device decryption challenge.
    """
    url = f"http://{node.verification_host}:{node.verification_port}/challenge/decrypt"
    error_message = None
    try:
        async with miner_client.post(
            node.miner_hotkey, url, payload=challenge.dict(), timeout=12.0
        ) as response:
            if response.status != 200:
                error_message = f"Failed to perform decryption challenge: {response.status=} {await response.text()}"
            else:
                response_text = (await response.json())["plaintext"]
                assert response_text == plaintext, (
                    f"Miner response '{response_text}' does not match ciphertext: '{plaintext}'"
                )
    except Exception as exc:
        logger.error(traceback.format_exc())
        error_message = f"Failed to perform decryption challenge: [unhandled exception] {exc}"
    if error_message:
        logger.error(error_message)
        async with get_session() as session:
            await session.execute(
                update(Node)
                .where(Node.uuid == node.uuid)
                .values({"verification_error": error_message})
            )
        await session.commit()
        return False
    return True


async def check_device_info_challenge(
    nodes: List[Node], url: str = None, purpose: str = "graval"
) -> bool:
    """
    Send a single device info challenge.
    """
    if not url:
        url = f"http://{nodes[0].verification_host}:{nodes[0].verification_port}/challenge/info"
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


@broker.task
async def validate_gpus(uuids: List[str]) -> Tuple[bool, str]:
    """
    Validate a single node.
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

    # Generate ciphertexts for each GPU.
    challenges = await asyncio.gather(*[generate_cipher(node) for node in nodes])

    # See if they decrypt properly.
    successes = await asyncio.gather(
        *[
            check_encryption_challenge(nodes[idx], challenges[idx][1], challenges[idx][0])
            for idx in range(len(uuids))
        ]
    )
    if not all(successes):
        error_message = "one or more decryption challenges failed"
        logger.warning(error_message)
        return False, error_message
    logger.success(
        f"All encryption checks [count={len(successes)}] passed successfully, trying device challenges..."
    )

    futures = []
    for _ in range(settings.device_info_challenge_count):
        futures.append(check_device_info_challenge(nodes))
        if len(futures) == 10:
            if not all(await asyncio.gather(*futures)):
                error_message = "one or more device info challenges failed"
                logger.warning(error_message)
                return False, error_message
            futures = []
    logger.success(f"Nodes {uuids} passed all preliminary node validation challenges!")
    async with get_session() as session:
        await session.execute(
            update(Node)
            .where(Node.uuid.in_(uuids))
            .values({"verified_at": func.now(), "verification_error": None})
        )

    # Notify the miner.
    async def _verify_one(gpu_id):
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

    await asyncio.gather(*[_verify_one(gpu_id) for gpu_id in uuids])

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
    device_dicts = [node.graval_dict() for node in instance.nodes]
    target_index = random.randint(0, len(device_dicts) - 1)
    target_device = device_dicts[target_index]
    seed = instance.nodes[0].seed
    expected = str(uuid.uuid4())
    payload = None
    logger.info(f"Trying to encrypt: {expected} via {settings.graval_url}/encrypt")
    async with aiohttp.ClientSession(raise_for_status=True) as graval_session:
        async with graval_session.post(
            f"{settings.graval_url}/encrypt",
            json={
                "payload": {
                    "hello": expected,
                },
                "device_info": target_device,
                "device_id": target_index,
                "seed": seed,
            },
            timeout=30.0,
        ) as resp:
            payload = await resp.json()

    logger.info(f"Sending encrypted payload to _ping endpoint for graval verification: {payload}")
    async with miner_client.post(
        instance.miner_hotkey,
        f"http://{instance.host}:{instance.port}/_ping",
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
async def _verify_filesystem_challenge(instance: Instance, challenge: FSChallenge) -> bool:
    """
    Check a single filesystem challenge on the remote instance.
    """
    logger.info(
        f"Sending filesystem challenge {challenge.filename=} {challenge.length=} {challenge.offset=} to {instance.instance_id=}"
    )
    async with miner_client.post(
        instance.miner_hotkey,
        f"http://{instance.host}:{instance.port}/_fs_challenge",
        payload=dict(
            filename=challenge.filename,
            offset=challenge.offset,
            length=challenge.length,
        ),
        timeout=12.0,
    ) as resp:
        resp.raise_for_status()
        result = await resp.text()
        if result != challenge.expected:
            logger.warning(f"Expected {challenge.expected}, got {result}: {challenge}")
            return False
        logger.success(f"Successfully processed filesystem challenge: {challenge}")
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
    logger.info(f"{instance.instance_id=} passed {passed} of {len(challenges)}")
    return passed == len(challenges)


@broker.task
async def verify_instance(instance_id: str):
    """
    Verify a single instance.
    """
    attempts = await settings.redis_client.incr(f"verify_instance:backend:{instance_id}")
    if attempts >= 8:
        return
    async with get_session() as session:
        instance = (
            (await session.execute(select(Instance).where(Instance.instance_id == instance_id)))
            .unique()
            .scalar_one_or_none()
        )
        if not instance:
            logger.warning("Found no matching nodes, did they disappear?")
            return

        # Ping test (decryption, ensures graval initialization happened).
        try:
            if not await _verify_instance_graval(instance):
                logger.warning(f"{instance_id=} failed GraVal verification!")
                instance.verification_error = "Failed one or more GraVal encryption challenges."
                await session.commit()
                return
        except Exception as exc:
            error_message = f"Failed to perform GraVal validation for {instance_id=}: {exc}\n{traceback.format_exc()}"
            logger.error(error_message)
            instance.verification_error = error_message
            await session.commit()
            return

        # Filesystem test.
        try:
            if not await _verify_filesystem(session, instance):
                logger.warning(f"{instance_id=} failed filesystem verification!")
                instance.verification_error = "Failed one or more filesystem challenges."
                await session.commit()
                return
        except Exception as exc:
            error_message = f"Failed to perform filesystem validation for {instance_id=}: {exc}\n{traceback.format_exc()}"
            logger.error(error_message)
            instance.verification_error = error_message
            await session.commit()
            return

        # Device info challenges.
        url = f"http://{instance.host}:{instance.port}/_device_challenge"
        futures = []
        for _ in range(settings.device_info_challenge_count):
            futures.append(check_device_info_challenge(instance.nodes, url=url, purpose="chutes"))
            if len(futures) == 10:
                if not all(await asyncio.gather(*futures)):
                    error_message = f"{instance_id=} failed one or more device info challenges"
                    logger.warning(error_message)
                    instance.verification_error = error_message
                    await session.commit()
                    return
                futures = []

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
