"""
Application-wide settings.
"""

import os
import aioboto3
import redis.asyncio as redis
from boto3.session import Config
from typing import Optional
from substrateinterface import Keypair
from pydantic_settings import BaseSettings
from contextlib import asynccontextmanager


class Settings(BaseSettings):
    sqlalchemy: str = os.getenv(
        "POSTGRESQL", "postgresql+asyncpg://user:password@127.0.0.1:5432/chutes"
    )
    aws_access_key_id: str = os.getenv("AWS_ACCESS_KEY_ID", "REPLACEME")
    aws_secret_access_key: str = os.getenv("AWS_SECRET_ACCESS_KEY", "REPLACEME")
    aws_endpoint_url: Optional[str] = os.getenv("AWS_ENDPOINT_URL", "http://minio:9000")
    aws_region: str = os.getenv("AWS_REGION", "local")
    storage_bucket: str = os.getenv("STORAGE_BUCKET", "chutes")

    @property
    def s3_session(self) -> aioboto3.Session:
        session = aioboto3.Session(
            aws_access_key_id=self.aws_access_key_id,
            aws_secret_access_key=self.aws_secret_access_key,
            region_name=self.aws_region,
        )
        return session

    @asynccontextmanager
    async def s3_client(self):
        session = self.s3_session
        async with session.client(
            "s3",
            endpoint_url=self.aws_endpoint_url,
            config=Config(signature_version="s3v4"),
        ) as client:
            yield client

    wallet_key: Optional[str] = os.getenv(
        "WALLET_KEY", "967fcf63799171672b6b66dfe30d8cd678c8bc6fb44806f0cdba3d873b3dd60b"
    )
    pg_encryption_key: Optional[str] = os.getenv("PG_ENCRYPTION_KEY", "secret")

    validator_ss58: Optional[str] = os.getenv("VALIDATOR_SS58")
    validator_keypair: Optional[Keypair] = (
        Keypair.create_from_seed(os.environ["VALIDATOR_SEED"])
        if os.getenv("VALIDATOR_SEED")
        else None
    )
    storage_bucket: str = os.getenv("STORAGE_BUCKET", "REPLACEME")
    redis_url: str = os.getenv("REDIS_URL", "redis://127.0.0.1:6379/0")
    redis_client: redis.Redis = redis.Redis.from_url(
        os.getenv("REDIS_URL", "redis://127.0.0.1:6379/0")
    )
    registry_host: str = os.getenv("REGISTRY_HOST", "registry:5000")
    registry_external_host: str = os.getenv("REGISTRY_EXTERNAL_HOST", "registry.chutes.ai")
    registry_password: str = os.getenv("REGISTRY_PASSWORD", "registrypassword")
    registry_insecure: bool = os.getenv("REGISTRY_INSECURE", "false").lower() == "true"
    build_timeout: int = int(os.getenv("BUILD_TIMEOUT", "3600"))
    push_timeout: int = int(os.getenv("PUSH_TIMEOUT", "1800"))
    netuid: int = int(os.getenv("NETUID", "19"))
    subtensor: str = os.getenv("SUBTENSOR_ADDRESS", "wss://entrypoint-finney.opentensor.ai:443")
    first_payment_bonus: float = float(os.getenv("FIRST_PAYMENT_BONUS", "100.0"))
    first_payment_bonus_threshold: float = float(os.getenv("FIRST_PAYMENT_BONUS_THRESHOLD", 25.0))
    payment_recovery_blocks: int = int(os.getenv("PAYMENT_RECOVERY_BLOCKS", "32"))
    device_info_challenge_count: int = int(os.getenv("DEVICE_INFO_CHALLENGE_COUNT", "200"))
    skip_gpu_verification: bool = os.getenv("SKIP_GPU_VERIFICATION", "false").lower() == "true"
    graval_proxy_url: str = os.getenv("GRAVAL_PROXY_URL", "")

    db_pool_size: int = int(os.getenv("DB_POOL_SIZE", "256"))
    db_overflow: int = int(os.getenv("DB_OVERFLOW", "32"))
    debug: bool = os.getenv("DEBUG", "false").lower() == "true"

    # XXX unused for now - future in which payouts to various parties.
    miner_take: float = float(os.getenv("MINER_TAKE", "0.73"))
    maintainer_take: float = float(os.getenv("MAINTAINER_TAKE", "0.2"))
    moderator_take: float = float(os.getenv("MODERATOR_TAKE", "0.02"))
    contributor_take: float = float(os.getenv("CONTRIBUTOR_TAKE", "0.03"))
    image_creator_take: float = float(os.getenv("IMAGE_CREATOR_TAKE", "0.01"))
    chute_creator_take: float = float(os.getenv("CHUTE_CREATOR_TAKE", "0.01"))


settings = Settings()
