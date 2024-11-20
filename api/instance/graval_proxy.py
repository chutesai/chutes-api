"""
GraVal proxy - encrypt/decrypt traffic to/from instances.
"""

import aiohttp
import argparse
import uvicorn
import asyncio
import random
import pybase64 as base64
from loguru import logger
from sqlalchemy import select
from pydantic import BaseModel
from graval.validator import Validator
from fastapi import FastAPI, status, HTTPException
from fastapi.responses import ORJSONResponse
from starlette.responses import StreamingResponse
from api.instance.schemas import Instance
from api.database import SessionLocal
import api.database.orms  # noqa


class Cipher(BaseModel):
    ciphertext: str
    iv: str
    length: int
    device_id: int
    seed: int


class Invocation(BaseModel):
    args: str
    kwargs: str
    path: str
    stream: bool
    instance_id: str


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
    )
    args = parser.parse_args()
    gpu_lock = asyncio.Lock()
    validator = Validator()
    validator.initialize()

    app = FastAPI(
        title="GraVal proxy",
        description="Encryption plz.",
        version="0.0.1",
    )

    @app.post("/proxy")
    async def proxy_request(invocation: Invocation):
        """
        Proxy a single request upstream to an instance, adding encryption.
        """
        logger.debug(f"Received invocation request: {invocation}")
        # Load the instance (for host, port, and device info).
        async with SessionLocal() as session:
            instance = (
                (
                    await session.execute(
                        select(Instance).filter(Instance.instance_id == invocation.instance_id)
                    )
                )
                .unique()
                .scalar_one_or_none()
            )
            if not instance:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"{invocation.instance_id=} not found",
                )
            device_dicts = [node.graval_dict() for node in instance.nodes]

        # Encrypt.
        async with gpu_lock:
            target_index = random.randint(0, len(device_dicts) - 1)
            target_device = device_dicts[target_index]
            seed = instance.nodes[0].seed
            encrypted = {}
            for key, val in [("args", invocation.args), ("kwargs", invocation.kwargs)]:
                ciphertext, iv, length = validator.encrypt(target_device, val, seed)
                encrypted[key] = Cipher(
                    ciphertext=base64.b64encode(ciphertext).decode(),
                    iv=iv.hex(),
                    length=length,
                    device_id=target_index,
                    seed=seed,
                ).dict()

        # Decrypt response.
        async def _response_iterator():
            session = aiohttp.ClientSession(raise_for_status=True)
            response = await session.post(
                f"http://{instance.host}:{instance.port}/{invocation.path}",
                json=encrypted,
                headers={"X-Chutes-Encrypted": "true"},
            )
            if invocation.stream:
                try:
                    async for chunk in response.content:
                        yield chunk.decode()
                finally:
                    await response.release()
                    await session.close()
            else:
                data = await response.json()
                await response.release()
                await session.close()
                yield data

        if invocation.stream:
            return StreamingResponse(_response_iterator())
        else:
            result = None
            async for data in _response_iterator():
                result = data
            return ORJSONResponse(result)

    uvicorn.run(app=app, host="0.0.0.0", port=args.port)


if __name__ == "__main__":
    main()
