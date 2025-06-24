"""
Application-wide settings.
"""

import os
import aioboto3
import aiomcache
from functools import cached_property
import redis.asyncio as redis
from boto3.session import Config
from typing import Optional
from bittensor_wallet.keypair import Keypair
from pydantic_settings import BaseSettings, SettingsConfigDict
from contextlib import asynccontextmanager


def load_squad_cert():
    if (path := os.getenv("SQUAD_CERT_PATH")) is not None:
        with open(path, "rb") as infile:
            return infile.read()
    return b""


class Settings(BaseSettings):
    model_config = SettingsConfigDict(arbitrary_types_allowed=True)
    _validator_keypair: Optional[Keypair] = None

    @cached_property
    def validator_keypair(self) -> Optional[Keypair]:
        if not self._validator_keypair and os.getenv("VALIDATOR_SEED"):
            self._validator_keypair = Keypair.create_from_seed(
                bytes.fromhex(os.environ["VALIDATOR_SEED"])
            )
        return self._validator_keypair

    sqlalchemy: str = os.getenv(
        "POSTGRESQL", "postgresql+asyncpg://user:password@127.0.0.1:5432/chutes"
    )
    postgres_ro: Optional[str] = os.getenv("POSTGRESQL_RO")

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
    storage_bucket: str = os.getenv("STORAGE_BUCKET", "REPLACEME")
    redis_url: str = os.getenv("REDIS_URL", "redis://127.0.0.1:6379/0")
    redis_client: redis.Redis = redis.Redis.from_url(
        os.getenv("REDIS_URL", "redis://127.0.0.1:6379/0")
    )
    cm_redis_client: list[redis.Redis] = [
        redis.Redis.from_url(
            os.getenv("REDIS_URL", "redis://127.0.0.1:6379/0").replace(
                "@redis.chutes.svc.cluster.local", f"@cm-redis-{idx}.chutes.svc.cluster.local"
            )
        )
        for idx in range(int(os.getenv("CM_REDIS_SHARD_COUNT", "0")))
    ]
    llm_cache_client: redis.Redis = redis.Redis.from_url(
        os.getenv("REDIS_URL", "redis://127.0.0.1:6379/0").replace(
            "@redis.chutes.svc.cluster.local", "@llm-cache-redis.chutes.svc.cluster.local"
        )
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
    build_timeout: int = int(os.getenv("BUILD_TIMEOUT", "7200"))
    push_timeout: int = int(os.getenv("PUSH_TIMEOUT", "1800"))
    scan_timeout: int = int(os.getenv("SCAN_TIMEOUT", "1800"))
    netuid: int = int(os.getenv("NETUID", "64"))
    subtensor: str = os.getenv("SUBTENSOR_ADDRESS", "wss://entrypoint-finney.opentensor.ai:443")
    first_payment_bonus: float = float(os.getenv("FIRST_PAYMENT_BONUS", "25.0"))
    first_payment_bonus_threshold: float = float(os.getenv("FIRST_PAYMENT_BONUS_THRESHOLD", 100.0))
    developer_deposit: float = float(os.getenv("DEVELOPER_DEPOSIT", "250.0"))
    payment_recovery_blocks: int = int(os.getenv("PAYMENT_RECOVERY_BLOCKS", "128"))
    device_info_challenge_count: int = int(os.getenv("DEVICE_INFO_CHALLENGE_COUNT", "20"))
    skip_gpu_verification: bool = os.getenv("SKIP_GPU_VERIFICATION", "false").lower() == "true"
    graval_url: str = os.getenv("GRAVAL_URL", "")
    opencl_graval_url: str = os.getenv("OPENCL_GRAVAL_URL", "https://opencl-graval.chutes.ai")

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

    # API key for checking code.
    codecheck_key: Optional[str] = os.getenv("CODECHECK_KEY")

    # Chutes decryption key bit.
    envcheck_key: Optional[str] = os.getenv("ENVCHECK_KEY")
    envcheck_salt: Optional[str] = os.getenv("ENVCHECK_SALT")
    envcheck_52_key: Optional[str] = os.getenv("ENVCHECK_KEY_52")
    envcheck_52_salt: Optional[str] = os.getenv("ENVCHECK_SALT_52")
    kubecheck_salt: Optional[str] = os.getenv("KUBECHECK_SALT")
    kubecheck_prefix: Optional[str] = os.getenv("KUBECHECK_PREFIX")
    kubecheck_suffix: Optional[str] = os.getenv("KUBECHECK_SUFFIX")

    # Logos CDN hostname.
    logo_cdn: Optional[str] = os.getenv("LOGO_CDN", "https://logos.chutes.ai")

    # Base domain.
    base_domain: Optional[str] = os.getenv("BASE_DOMAIN", "chutes.ai")

    # Rate limits.
    rate_limit_count: int = int(os.getenv("RATE_LIMIT_COUNT", 10))
    rate_limit_window: int = int(os.getenv("RATE_LIMIT_WINDOW", 60))
    ip_rate_limit_count: int = int(os.getenv("IP_RATE_LIMIT_COUNT", 60))
    ip_rate_limit_window: int = int(os.getenv("IP_RATE_LIMIT_WINDOW", 60))

    # Chutes pinned version.
    chutes_version: str = os.getenv("CHUTES_VERSION", "0.2.53")

    # Auto stake amount when DCAing into alpha after receiving payments.
    autostake_amount: float = float(os.getenv("AUTOSTAKE_AMOUNT", "0.03"))


settings = Settings()
