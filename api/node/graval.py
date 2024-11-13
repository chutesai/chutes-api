"""
GraVal node validation worker.
"""

import asyncio
import uuid
import pybase64 as base64
from typing import List
from pydantic import BaseModel
from graval.validator import Validator
from loguru import logger
from api.config import settings
from api.database import SessionLocal
from api.node.schemas import Node
from sqlalchemy import update
from sqlalchemy.future import select
from taskiq import TaskiqEvents
from taskiq_redis import ListQueueBroker, RedisAsyncResultBackend
import api.database.orms  # noqa
import api.miner_client as miner_client

broker = ListQueueBroker(url=settings.redis_url, queue_name="graval").with_result_backend(
    RedisAsyncResultBackend(redis_url=settings.redis_url, result_ex_time=3600)
)
validator = Validator()


class CipherChallenge(BaseModel):
    ciphertext: str
    iv: str
    length: int
    device_id: int
    seed: int


@broker.on_event(TaskiqEvents.WORKER_STARTUP)
async def initialize_graval(*_, **__):
    """
    Initialize the GraVal validator instance.
    """
    validator.initialize()


def generate_cipher(node):
    """
    Encrypt some data on the validator side and see if the miner can decrypt it.
    """
    plaintext = f"decrypt me please: {uuid.uuid4()}"
    ciphertext, iv, length = validator.encrypt(node.graval_dict(), plaintext, node.seed)
    logger.info(f"Generated {length} byte ciphertext from: {plaintext}")
    return plaintext, CipherChallenge(
        ciphertext=base64.b64encode(ciphertext).decode(),
        iv=iv.hex(),
        length=length,
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
            node.miner_hotkey, url, payload=challenge.dict(), timeout=5.0
        ) as response:
            if response.status_code != 200:
                error_message = f"Failed to perform decryption challenge: {response.status_code=} {await response.text()}"
            assert (await response.json())["plaintext"] == plaintext
    except Exception as exc:
        error_message = f"Failed to perform decryption challenge: [unhandled exception] {exc}"
    if error_message:
        async with SessionLocal() as session:
            await session.execute(
                update(Node)
                .where(Node.node_id == node.node_id)
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
        challenge = validator.generate_device_info_challenge(len(nodes))
        async with miner_client.get(
            nodes[0].miner_hotkey, url, params={"challenge": challenge}, timeout=5.0
        ) as response:
            if response.status_code != 200:
                error_message = f"Failed to perform device info challenge: {response.status_code=} {await response.text()}"
            response = await response.text()
            assert validator.verify_device_info_challenge(
                challenge, response, [node.graval_dict() for node in nodes]
            )
    except Exception as exc:
        error_message = f"Failed to perform decryption challenge: [unhandled exception] {exc}"
    if error_message:
        async with SessionLocal() as session:
            await session.execute(
                update(Node)
                .where(Node.node_id.in_([node.node_id for node in nodes]))
                .values({"verification_error": error_message})
            )
        await session.commit()
        return False
    return True


@broker.task
async def validate_gpus(node_ids: str):
    """
    Validate a single node.
    """
    async with SessionLocal() as session:
        if not (
            nodes := (
                await session.execute(select(Node).where(Node.node_id.in_(node_ids)))
            ).scalar()
        ):
            logger.warning("Found no matching nodes, did they disappear?")
            return

    # Generate ciphertexts for each GPU.
    challenges = [generate_cipher(node) for node in nodes]

    # See if they decrypt properly.
    successes = await asyncio.gather(
        *[
            check_encryption_challenge(nodes[idx], challenges[idx][1], challenges[idx][0])
            for idx in range(len(node_ids))
        ]
    )
    if not all(successes):
        logger.warning("Skipping remaining checks, one or more decryption challenges failed.")
        return
    logger.success(
        f"All encryption checks [count={len(successes)}] passed successfully, trying device challenges..."
    )

    for _ in range(settings.device_info_challenge_count):
        if not await check_device_info_challenge(nodes):
            logger.warning("Failed device info challenge!")
            return
