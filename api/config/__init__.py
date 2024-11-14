"""
Application-wide settings.
"""

import os
import hvac
import redis.asyncio as redis
from miniopy_async import Minio
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    sqlalchemy: str = os.getenv(
        "POSTGRESQL", "postgresql+asyncpg://user:password@127.0.0.1:5432/chutes"
    )
    storage_client: Minio = Minio(
        os.getenv("MINIO_ENDPOINT", "127.0.0.1"),
        access_key=os.getenv("MINIO_ACCESS_KEY", "REPLACEME"),
        secret_key=os.getenv("MINIO_SECRET_KEY", "REPLACEME"),
        secure=os.getenv("MINIO_SECURE", "false").lower() == "true",
    )
    vault_client: hvac.Client = hvac.Client(
        url=os.getenv("VAULT_URL", "http://vault:777"),
        token=os.getenv("VAULT_TOKEN", "supersecrettoken"),
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
    debug: bool = os.getenv("DEBUG", "false").lower() == "true"

    # XXX unused for now - future in which payouts to various parties.
    miner_take: float = float(os.getenv("MINER_TAKE", "0.73"))
    maintainer_take: float = float(os.getenv("MAINTAINER_TAKE", "0.2"))
    moderator_take: float = float(os.getenv("MODERATOR_TAKE", "0.02"))
    contributor_take: float = float(os.getenv("CONTRIBUTOR_TAKE", "0.03"))
    image_creator_take: float = float(os.getenv("IMAGE_CREATOR_TAKE", "0.01"))
    chute_creator_take: float = float(os.getenv("CHUTE_CREATOR_TAKE", "0.01"))
    maintainer_payout_addresses: list[str] = [
        address
        for address in os.getenv("MAINTAINER_PAYOUT_ADDRESSES", "").split(",")
        if address.strip()
    ]


settings = Settings()
