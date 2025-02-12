"""
Application-wide settings.
"""

import os
import aioboto3
import aiomcache
import redis.asyncio as redis
from boto3.session import Config
from typing import Optional
from substrateinterface import Keypair
from pydantic_settings import BaseSettings
from contextlib import asynccontextmanager


def load_squad_cert():
    if (path := os.getenv("SQUAD_CERT_PATH")) is not None:
        with open(path, "rb") as infile:
            return infile.read()
    return b""


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
    memcache: Optional[aiomcache.Client] = (
        aiomcache.Client(os.getenv("MEMCACHED", "memcached"), 11211)
        if os.getenv("MEMCACHED")
        else None
    )
    registry_host: str = os.getenv("REGISTRY_HOST", "registry:5000")
    registry_external_host: str = os.getenv("REGISTRY_EXTERNAL_HOST", "registry.chutes.ai")
    registry_password: str = os.getenv("REGISTRY_PASSWORD", "registrypassword")
    registry_insecure: bool = os.getenv("REGISTRY_INSECURE", "false").lower() == "true"
    build_timeout: int = int(os.getenv("BUILD_TIMEOUT", "3600"))
    push_timeout: int = int(os.getenv("PUSH_TIMEOUT", "1800"))
    scan_timeout: int = int(os.getenv("SCAN_TIMEOUT", "1200"))
    netuid: int = int(os.getenv("NETUID", "64"))
    subtensor: str = os.getenv("SUBTENSOR_ADDRESS", "wss://entrypoint-finney.opentensor.ai:443")
    first_payment_bonus: float = float(os.getenv("FIRST_PAYMENT_BONUS", "25.0"))
    first_payment_bonus_threshold: float = float(os.getenv("FIRST_PAYMENT_BONUS_THRESHOLD", 100.0))
    developer_deposit: float = float(os.getenv("DEVELOPER_DEPOSIT", "250.0"))
    payment_recovery_blocks: int = int(os.getenv("PAYMENT_RECOVERY_BLOCKS", "128"))
    device_info_challenge_count: int = int(os.getenv("DEVICE_INFO_CHALLENGE_COUNT", "200"))
    skip_gpu_verification: bool = os.getenv("SKIP_GPU_VERIFICATION", "false").lower() == "true"
    graval_url: str = os.getenv("GRAVAL_URL", "")

    # Database settings.
    db_pool_size: int = int(os.getenv("DB_POOL_SIZE", "64"))
    db_overflow: int = int(os.getenv("DB_OVERFLOW", "32"))

    # Debug logging.
    debug: bool = os.getenv("DEBUG", "false").lower() == "true"

    # IP hash check salt.
    ip_check_salt: str = os.getenv("IP_CHECK_SALT", "salt")

    # Validator and subnet owner keys allowed to link/get free+dev access.
    validators: list[str] = os.getenv("VALIDATOR_HOTKEYS", "").split(",")
    subnet_owners: list[str] = os.getenv("SUBNET_OWNER_HOTKEYS", "").split(",")

    # Flag indicating that all accounts created are free.
    all_accounts_free: bool = os.getenv("ALL_ACCOUNTS_FREE", "false").lower() == "true"

    # Squad cert (for JWT auth from agents).
    squad_cert: bytes = load_squad_cert()

    # Consecutive failure count that triggers instance deletion.
    consecutive_failure_limit: int = int(os.getenv("CONSECUTIVE_FAILURE_LIMIT", "7"))

    # Rate limits.
    rate_limit_count: int = int(os.getenv("RATE_LIMIT_COUNT", 15))
    rate_limit_window: int = int(os.getenv("RATE_LIMIT_WINDOW", 60))
    ip_rate_limit_count: int = int(os.getenv("IP_RATE_LIMIT_COUNT", 25))
    ip_rate_limit_window: int = int(os.getenv("IP_RATE_LIMIT_WINDOW", 60))

    # Chutes pinned version.
    chutes_version: str = os.getenv("CHUTES_VERSION", "0.2.18")

    # XXX unused for now - future in which payouts to various parties.
    miner_take: float = float(os.getenv("MINER_TAKE", "0.73"))
    maintainer_take: float = float(os.getenv("MAINTAINER_TAKE", "0.2"))
    moderator_take: float = float(os.getenv("MODERATOR_TAKE", "0.02"))
    contributor_take: float = float(os.getenv("CONTRIBUTOR_TAKE", "0.03"))
    image_creator_take: float = float(os.getenv("IMAGE_CREATOR_TAKE", "0.01"))
    chute_creator_take: float = float(os.getenv("CHUTE_CREATOR_TAKE", "0.01"))


settings = Settings()
