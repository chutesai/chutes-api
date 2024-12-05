"""
GraVal node validation worker.
"""

import asyncio
import aiohttp
import uuid
import traceback
import backoff
from typing import List, Tuple
from pydantic import BaseModel
from loguru import logger
from api.config import settings
from api.database import get_session
from api.node.schemas import Node
from sqlalchemy import update, func
from sqlalchemy.future import select
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
