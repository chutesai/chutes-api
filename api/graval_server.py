"""
GraVal encryptor/challenge generator.
"""

import time
import uuid
import random
import hashlib
import argparse
import uvicorn
import asyncio
import pybase64 as base64
import orjson as json
from ipaddress import ip_address
from loguru import logger
from pydantic import BaseModel
from graval import Validator
from substrateinterface import Keypair, KeypairType
from fastapi import FastAPI, Request, status, HTTPException


class Cipher(BaseModel):
    ciphertext: str
    iv: str
    length: int


class Challenge(Cipher):
    seed: int
    plaintext: str


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
    )
    parser.add_argument(
        "--validator-whitelist",
        type=str,
        required=True,
    )
    args = parser.parse_args()

    validator = Validator()
    validator.initialize()
    app = FastAPI(
        title="GraVal challenge generator",
        description="GPU info plz",
        version="0.0.1",
    )
    gpu_lock = asyncio.Lock()

    def verify_request(request: Request, whitelist: list[str], extra_key: str = "graval") -> None:
        """
        Verify the authenticity of a request.
        """
        validator_hotkey = request.headers.get("X-Validator")
        nonce = request.headers.get("X-Nonce")
        signature = request.headers.get("X-Signature")
        if (
            any(not v for v in [validator_hotkey, nonce, signature])
            or validator_hotkey not in whitelist
            or int(time.time()) - int(nonce) >= 30
        ):
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="go away")
        signature_string = ":".join(
            [
                validator_hotkey,
                nonce,
                extra_key,
            ]
        )
        if not Keypair(ss58_address=validator_hotkey, crypto_type=KeypairType.SR25519).verify(
            signature_string, bytes.fromhex(signature)
        ):
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="go away")

    @app.post("/generate")
    async def generate_challenge(request: Request):
        """
        Generate a new encryption challenge.
        """
        request_body = await request.body()
        sha2 = hashlib.sha256(request_body).hexdigest()
        verify_request(request, args.validator_whitelist.split(","), extra_key=sha2)
        device_info = json.loads(request_body.decode())
        seed = device_info.pop("seed", random.randint(1, 2**63 - 1))
        async with gpu_lock:
            plaintext = str(uuid.uuid4())
            ciphertext, iv, length = validator.encrypt(device_info, plaintext, seed)
            logger.info(f"Generated {length} byte ciphertext from: {plaintext}")
            return Challenge(
                ciphertext=ciphertext.hex(),
                iv=iv.hex(),
                length=length,
                seed=seed,
                plaintext=plaintext,
            )

    @app.post("/encrypt")
    async def encrypt_payload(request: Request):
        """
        Encrypt an input payload for the specified device.
        """
        data = await request.json()
        x_forwarded_for = request.headers.get("X-Forwarded-For")
        actual_ip = x_forwarded_for.split(",")[0] if x_forwarded_for else request.client.host
        ip = ip_address(actual_ip)
        is_private = ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved
        if not is_private:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="go away")
        device_info = data["device_info"]
        payload = data["payload"]
        seed = data["seed"]
        device_id = data["device_id"]
        encrypted_payload = {}
        async with gpu_lock:
            for key, value in payload.items():
                plaintext = value if isinstance(value, str) else json.dumps(value).decode()
                ciphertext, iv, length = validator.encrypt(device_info, plaintext, seed)
                logger.info(f"Generated {length} byte ciphertext for {device_info['uuid']}")
                encrypted_payload[key] = dict(
                    ciphertext=base64.b64encode(ciphertext).decode(),
                    iv=iv.hex(),
                    length=length,
                    device_id=device_id,
                    seed=seed,
                )
        return encrypted_payload

    @app.get("/ping")
    async def ping():
        return "pong"

    uvicorn.run(app=app, host="0.0.0.0", port=args.port)


if __name__ == "__main__":
    main()
