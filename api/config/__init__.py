"""
Application-wide settings.
"""

import os
import hashlib
from pathlib import Path
import aioboto3
import aiomcache
import json
from functools import cached_property, lru_cache
import redis.asyncio as redis
from boto3.session import Config
from typing import Dict, Optional
from bittensor_wallet.keypair import Keypair
from pydantic_settings import BaseSettings, SettingsConfigDict
from contextlib import asynccontextmanager
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import ec


@lru_cache(maxsize=1)
def load_squad_cert():
    if (path := os.getenv("SQUAD_CERT_PATH")) is not None:
        with open(path, "rb") as infile:
            return infile.read()
    return b""


@lru_cache(maxsize=1)
def load_launch_config_private_key():
    if (path := os.getenv("LAUNCH_CONFIG_PRIVATE_KEY_PATH")) is not None:
        with open(path, "rb") as infile:
            return infile.read()
    return None


class Settings(BaseSettings):
    model_config = SettingsConfigDict(arbitrary_types_allowed=True)
    _validator_keypair: Optional[Keypair] = None

    @cached_property
    def validator_keypair(self) -> Optional[Keypair]:
        if not self._validator_keypair and os.getenv("VALIDATOR_SEED"):
            self._validator_keypair = Keypair.create_from_seed(os.environ["VALIDATOR_SEED"])
        return self._validator_keypair

    sqlalchemy: str = os.getenv(
        "POSTGRESQL", "postgresql+asyncpg://user:password@127.0.0.1:5432/chutes"
    )
    postgres_ro: Optional[str] = os.getenv("POSTGRESQL_RO")

    # Invocations database.
    invocations_db_url: Optional[str] = os.getenv(
        "INVOCATIONS_DB_URL",
        os.getenv("POSTGRESQL", "postgresql+asyncpg://user:password@127.0.0.1:5432/chutes"),
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
    storage_bucket: str = os.getenv("STORAGE_BUCKET", "REPLACEME")
    redis_url: str = os.getenv("REDIS_URL", "redis://127.0.0.1:6379/0")
    memcached_host: str = os.getenv("MEMCACHED", "memcached")

    _redis_client: Optional[redis.Redis] = None
    _cm_redis_clients: Optional[list[redis.Redis]] = None
    _quota_client: Optional[redis.Redis] = None
    _memcache: Optional[aiomcache.Client] = None
    cm_redis_shard_count: int = int(os.getenv("CM_REDIS_SHARD_COUNT", "1"))

    @property
    def redis_client(self) -> redis.Redis:
        if self._redis_client is None:
            self._redis_client = redis.Redis.from_url(self.redis_url)
        return self._redis_client

    @property
    def cm_redis_client(self) -> list[redis.Redis]:
        if self._cm_redis_clients is None:
            self._cm_redis_clients = [
                redis.Redis.from_url(
                    self.redis_url.replace(
                        "@redis.chutes.svc.cluster.local",
                        f"@cm-redis-{idx}.chutes.svc.cluster.local",
                    ),
                    socket_timeout=10.0,
                    socket_connect_timeout=3.0,
                    socket_keepalive=True,
                    health_check_interval=30,
                )
                for idx in range(self.cm_redis_shard_count)
            ]
        return self._cm_redis_clients

    @property
    def quota_client(self) -> redis.Redis:
        if self._quota_client is None:
            self._quota_client = redis.Redis.from_url(
                self.redis_url.replace(
                    "@redis.chutes.svc.cluster.local", "@quota-redis.chutes.svc.cluster.local"
                )
            )
        return self._quota_client

    @property
    def memcache(self) -> Optional[aiomcache.Client]:
        if self._memcache is None and self.memcached_host:
            self._memcache = aiomcache.Client(self.memcached_host, 11211, pool_size=4)
        return self._memcache

    registry_host: str = os.getenv("REGISTRY_HOST", "registry:5000")
    registry_external_host: str = os.getenv("REGISTRY_EXTERNAL_HOST", "registry.chutes.ai")
    registry_password: str = os.getenv("REGISTRY_PASSWORD", "registrypassword")
    registry_insecure: bool = os.getenv("REGISTRY_INSECURE", "false").lower() == "true"
    build_timeout: int = int(os.getenv("BUILD_TIMEOUT", "7200"))
    push_timeout: int = int(os.getenv("PUSH_TIMEOUT", "7200"))
    scan_timeout: int = int(os.getenv("SCAN_TIMEOUT", "7200"))
    netuid: int = int(os.getenv("NETUID", "64"))
    subtensor: str = os.getenv("SUBTENSOR_ADDRESS", "wss://entrypoint-finney.opentensor.ai:443")
    payment_recovery_blocks: int = int(os.getenv("PAYMENT_RECOVERY_BLOCKS", "128"))
    device_info_challenge_count: int = int(os.getenv("DEVICE_INFO_CHALLENGE_COUNT", "20"))
    skip_gpu_verification: bool = os.getenv("SKIP_GPU_VERIFICATION", "false").lower() == "true"
    graval_url: str = os.getenv("GRAVAL_URL", "https://graval.chutes.ai:11443")

    # Database settings.
    db_pool_size: int = int(os.getenv("DB_POOL_SIZE", "16"))
    db_overflow: int = int(os.getenv("DB_OVERFLOW", "3"))

    # Debug logging.
    debug: bool = os.getenv("DEBUG", "false").lower() == "true"

    # IP hash check salt.
    ip_check_salt: str = os.getenv("IP_CHECK_SALT", "salt")

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

    # Launch config JWT signing key.
    launch_config_key: str = hashlib.sha256(
        os.getenv("LAUNCH_CONFIG_KEY", "launch-secret").encode()
    ).hexdigest()

    # New, asymmetric launch config keys.
    launch_config_private_key_bytes: Optional[bytes] = load_launch_config_private_key()

    @cached_property
    def launch_config_private_key(self) -> Optional[ec.EllipticCurvePrivateKey]:
        if hasattr(self, "_launch_config_private_key"):
            return self._launch_config_private_key
        if (key_bytes := load_launch_config_private_key()) is not None:
            self._launch_config_private_key = serialization.load_pem_private_key(key_bytes, None)
        return self._launch_config_private_key

    # Default quotas/discounts.
    default_quotas: dict = json.loads(os.getenv("DEFAULT_QUOTAS", '{"*": 200}'))
    default_discounts: dict = json.loads(os.getenv("DEFAULT_DISCOUNTS", '{"*": 0.0}'))
    default_job_quotas: dict = json.loads(os.getenv("DEFAULT_JOB_QUOTAS", '{"*": 0}'))

    # Quota unlock amount (requires replacing the trigger function to actually work though!)
    quota_unlock_amount: float = float(os.getenv("QUOTA_UNLOCK_AMOUNT", "5.0"))

    # Reroll discount (i.e. duplicate prompts for re-roll in RP, or pass@k, etc.)
    reroll_multiplier: float = float(os.getenv("REROLL_MULTIPLIER", "0.1"))

    # Chutes pinned version.
    chutes_version: str = os.getenv("CHUTES_VERSION", "0.3.44")

    # Auto stake amount when DCAing into alpha after receiving payments.
    autostake_amount: float = float(os.getenv("AUTOSTAKE_AMOUNT", "1.0"))

    # Cosign Settings
    cosign_password: Optional[str] = os.getenv("COSIGN_PASSWORD")
    cosign_key: Optional[Path] = Path(os.getenv("COSIGN_KEY")) if os.getenv("COSIGN_KEY") else None

    # hCaptcha
    hcaptcha_sitekey: Optional[str] = os.getenv("HCAPTCHA_SITEKEY")
    hcaptcha_secret: Optional[str] = os.getenv("HCAPTCHA_SECRET")

    # TDX Attestation settings
    expected_mrtd: Optional[str] = os.getenv("TDX_EXPECTED_MRTD")
    expected_boot_rmtrs: Optional[Dict[str, str]] = (
        {
            pair.split("=")[0]: pair.split("=")[1]
            for pair in os.getenv("TDX_BOOT_RMTRS", "").split(",")
            if pair and "=" in pair and len(pair.split("=")) == 2
        }
        if os.getenv("TDX_BOOT_RMTRS")
        else None
    )
    expected_runtime_rmtrs: Optional[Dict[str, str]] = (
        {
            pair.split("=")[0]: pair.split("=")[1]
            for pair in os.getenv("TDX_RUNTIME_RMTRS", "").split(",")
            if pair and "=" in pair and len(pair.split("=")) == 2
        }
        if os.getenv("TDX_RUNTIME_RMTRS")
        else None
    )
    luks_passphrase: Optional[str] = os.getenv("LUKS_PASSPHRASE")

    # TDX verification service URLs (if using Intel's remote verification)
    tdx_verification_url: Optional[str] = os.getenv("TDX_VERIFICATION_URL")
    tdx_cert_chain_url: Optional[str] = os.getenv("TDX_CERT_CHAIN_URL")

    # Nonce expiration (minutes)
    attestation_nonce_expiry: int = int(os.getenv("ATTESTATION_NONCE_EXPIRY", "10"))


settings = Settings()
