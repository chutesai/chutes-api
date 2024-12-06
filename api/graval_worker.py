"""
GraVal node validation worker.
"""

import re
import hashlib
import asyncio
import aiohttp
import uuid
import random
import traceback
import backoff
import pybase64 as base64
from typing import List, Tuple
from pydantic import BaseModel
from loguru import logger
from api.config import settings
from api.database import get_session
from api.node.schemas import Node
from api.image.schemas import Image
from api.instance.schemas import Instance
from api.fs_challenge.schemas import FSChallenge
from sqlalchemy import update, func
from sqlalchemy.orm import joinedload
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


@backoff.on_exception(
    backoff.constant,
    Exception,
    jitter=None,
    interval=10,
    max_tries=7,
)
async def generate_device_info_challenge(device_count: int):
    """
    Generate a device info challenge.
    """
    async with aiohttp.ClientSession(raise_for_status=True) as session:
        async with session.get(
            f"{settings.graval_url}/device_challenge",
            params={"device_count": str(device_count)},
        ) as resp:
            return (await resp.json())["challenge"]


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
        logger.info(f"POSTING: {challenge.dict()}")
        async with miner_client.post(
            node.miner_hotkey, url, payload=challenge.dict(), timeout=5.0
        ) as response:
            if response.status != 200:
                error_message = f"Failed to perform decryption challenge: {response.status=} {await response.text()}"
            else:
                response_text = (await response.json())["plaintext"]
                assert (
                    response_text == plaintext
                ), f"Miner response '{response_text}' does not match ciphertext: '{plaintext}'"
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


async def check_device_info_challenge(nodes: List[Node]) -> bool:
    """
    Send a single device info challenge.
    """
    url = f"http://{nodes[0].verification_host}:{nodes[0].verification_port}/challenge/info"
    error_message = None
    try:
        challenge = await generate_device_info_challenge(len(nodes))
        async with miner_client.get(
            nodes[0].miner_hotkey,
            url,
            "graval",
            params={"challenge": challenge},
            timeout=5.0,
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
            async with miner_client.axon_patch(
                nodes[0].miner_hotkey, f"/gpus/{gpu_id}", payload={"verified": True}
            ) as resp:
                resp.raise_for_status()
        except Exception as exc:
            # Allow exceptions here since the miner can also check.
            logger.warning(
                f"Error notifying miner that GPU is verified: {exc}\n{traceback.format_exc()}"
            )

    await asyncio.gather(*[_verify_one(gpu_id) for gpu_id in uuids])

    return True, None


@broker.task
async def generate_fs_challenges(image_id: str):
    """
    Generate filesystem challenges after an image is created.
    """
    async with get_session() as session:
        image = (
            (
                await session.execute(
                    select(Image)
                    .options(joinedload(Image.fs_challenges))
                    .where(Image.image_id == image_id)
                )
            )
            .unique()
            .scalar_one_or_none()
        )
        if not image:
            logger.warning(f"Unable to locate {image_id=}")
            return
        if image.fs_challenges:
            logger.warning(f"Already have filesystem challenges for {image_id=}")

    challenges = []
    async with settings.s3_client() as s3:
        for suffix in ["newsample", "workdir", "corelibs"]:
            key = f"fschallenge/{image.user_id}/{image.image_id}/fschallenge_{suffix}.data"
            entries = []
            try:
                response = await s3.get_object(Bucket=settings.storage_bucket, Key=key)
                data = (await response["Body"].read()).decode("utf-8")
                entries = [line for line in data.splitlines() if line.strip()]
            except Exception as exc:
                logger.error(f"Error loading challenge data @ {key}: {exc}")
                continue
            for line in entries:
                file_data_match = re.match(
                    r"^(.*)?:__size__:(.*):__checksum__:.*?:__head__:(.*)?:__tail__:(.*)", line
                )
                if not file_data_match:
                    logger.warning(f"Bad challenge data found in {key}: {line}")
                    continue
                filename, size, head, tail = file_data_match.groups()
                head = None if head == "NONE" else base64.b64decode(head.encode())
                tail = None if tail == "NONE" else base64.b64decode(tail.encode())
                size = int(size)
                if not size or not head:
                    continue
                current_count = len(challenges)
                target_count = {"newsample": 20, "workdir": 100, "corelibs": 50}[suffix]
                logger.info(f"Generating {target_count} challenges for {image_id=} {filename=}")
                while len(challenges) <= current_count + target_count:
                    length = random.randint(10, len(head))
                    offset = random.randint(0, len(head) - length - 1)
                    challenges.append(
                        {
                            "filename": filename,
                            "length": length,
                            "offset": offset,
                            "expected": hashlib.sha256(head[offset : offset + length]).hexdigest(),
                        }
                    )
                    if tail:
                        length = random.randint(10, len(tail))
                        offset = random.randint(size - len(tail), len(tail) - length - 1)
                        challenges.append(
                            {
                                "filename": filename,
                                "length": length,
                                "offset": offset,
                                "expected": hashlib.sha256(
                                    tail[offset - size : offset - size + length]
                                ).hexdigest(),
                            }
                        )

    # Persist all of the challenges to DB.
    logger.info(f"About to save {len(challenges)} challenges for {image_id=}")
    async with get_session() as session:
        for challenge in challenges:
            session.add(
                FSChallenge(
                    challenge_id=str(uuid.uuid4()),
                    image_id=image_id,
                    filename=challenge["filename"],
                    offset=challenge["offset"],
                    length=challenge["length"],
                    expected=challenge["expected"],
                )
            )
        await session.commit()
    logger.success(
        f"Successfully persisted {len(challenges)} filesystem challenges for {image_id=}"
    )


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
        timeout=5.0,
    ) as resp:
        resp.raise_for_status()
        assert (await resp.json())["hello"] == expected


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
        timeout=3.0,
    ) as resp:
        resp.raise_for_status()
        assert (await resp.text()) == challenge.expected


async def _verify_filesystem(session: AsyncSession, instance: Instance) -> bool:
    """
    Perform a variety of filesystem challenges.
    """

    async def _safe_verify_one(challenge):
        try:
            return await _verify_filesystem_challenge(instance, challenge)
        except Exception as exc:
            logger.error(f"Failed _verify_filesystem_challenge 3 times: {exc}")
            return False

    subquery = (
        select(FSChallenge)
        .add_columns(
            func.row_number()
            .over(partition_by=FSChallenge.challenge_type, order_by=func.random())
            .label("rn")
        )
        .subquery()
    )

    result = await session.execute(select(subquery.c).where(subquery.c.rn <= 10))
    challenges = result.scalars().all()

    results = await asyncio.gather(*[_safe_verify_one(challenge) for challenge in challenges])
    passed = sum(1 for r in results if r)

    logger.info(f"{instance.instance_id=} passed {passed} of {len(challenges)}")
    return passed == len(challenges)


@broker.task
async def verify_instance(instance_id: str):
    """
    Verify a single instance.
    """
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
            if not await _verify_instance_graval(instance_id):
                logger.warning(f"{instance_id=} failed GraVal verification!")
        except Exception:
            logger.error(f"Failed to perform GraVal validation for {instance_id=}")

        # Filesystem test.
        try:
            if not await _verify_filesystem(session, instance):
                logger.warning(f"{instance_id=} failed filesystem verification!")
        except Exception:
            logger.error(f"Failed to perform filesystem validation for {instance_id=}")

        # Looks good!
        instance.verified = True
        await session.commit()
        await session.refresh(instance)

        # Broadcast the event.
        try:
            await settings.redis_client.publish(
                "user_broadcast",
                json.dumps(
                    {
                        "reason": "instance_hot",
                        "message": f"Miner {instance.miner_hotkey} instance {instance.instance_id} '{instance.chute.name}' has been verified, now 'hot'!",
                        "data": {
                            "chute_id": instance.chute_id,
                            "miner_hotkey": instance.miner_hotkey,
                        },
                    }
                ).decode(),
            )
        except Exception as exc:
            logger.warning(f"Error broadcasting instance event: {exc}")
