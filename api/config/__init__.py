"""
Application-wide settings.
"""

import os
import hashlib
from pathlib import Path
import aioboto3
import json
import yaml
from dataclasses import dataclass
from api.safe_redis import SafeRedis
from functools import cached_property, lru_cache
import redis.asyncio as redis
from redis.retry import Retry
from redis.backoff import ConstantBackoff
from boto3.session import Config
from typing import Dict, List, Optional
from bittensor_wallet.keypair import Keypair
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict
from contextlib import asynccontextmanager
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.fernet import Fernet
from loguru import logger


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


@dataclass
class TeeMeasurementConfig:
    """Configuration for allowed measurements for a TEE VM."""

    version: str
    mrtd: str
    name: str
    boot_rtmrs: Dict[str, str]
    runtime_rtmrs: Dict[str, str]
    expected_gpus: List[str]
    gpu_count: Optional[int] = None


class Settings(BaseSettings):
    model_config = SettingsConfigDict(arbitrary_types_allowed=True)
    _validator_keypair: Optional[Keypair] = None

    def model_post_init(self, __context) -> None:
        """Validate configuration after initialization."""
        # Eagerly validate TEE measurement configuration only when the config file is mounted
        if self.tee_measurement_config_path.exists():
            _ = self.tee_measurements

    @cached_property
    def validator_keypair(self) -> Optional[Keypair]:
        if not self._validator_keypair and os.getenv("VALIDATOR_SEED"):
            self._validator_keypair = Keypair.create_from_seed(os.environ["VALIDATOR_SEED"])
        return self._validator_keypair

    @cached_property
    def fernet_key(self) -> Optional[Fernet]:
        """Get validated Fernet cipher for cache passphrase encryption.

        Returns:
            Fernet cipher instance, or None if CACHE_PASSPHRASE_KEY not configured

        Raises:
            ValueError: If CACHE_PASSPHRASE_KEY is invalid format
        """
        key = os.getenv("CACHE_PASSPHRASE_KEY")
        if not key:
            return None

        # Fernet keys must be 32 url-safe base64-encoded bytes (44 characters)
        if len(key) != 44:
            raise ValueError(
                f"CACHE_PASSPHRASE_KEY must be 44 characters (32 bytes base64-encoded), got {len(key)} characters. "
                "Generate a valid key with: python -c 'from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())'"
            )

        try:
            return Fernet(key.encode())
        except Exception as e:
            raise ValueError(f"Invalid CACHE_PASSPHRASE_KEY format: {e}")

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

    # Base redis settings.
    redis_host: str = Field(
        default="172.16.0.100",
        validation_alias="PRIMARY_REDIS_HOST",
    )
    redis_port: int = Field(
        default=1600,
        validation_alias="PRIMARY_REDIS_PORT",
    )
    redis_password: str = str(os.getenv("REDIS_PASSWORD", "password"))
    redis_db: int = int(os.getenv("REDIS_DB", "0"))
    redis_max_connections: int = int(os.getenv("REDIS_MAX_CONNECTIONS", 512))
    redis_connect_timeout: float = float(os.getenv("REDIS_CONNECT_TIMEOUT", "1.5"))
    redis_socket_timeout: float = float(os.getenv("REDIS_SOCKET_TIMEOUT", "2.5"))
    redis_op_timeout: float = float(
        os.getenv("REDIS_OP_TIMEOUT", os.getenv("REDIS_SOCKET_TIMEOUT", "2.5"))
    )

    _redis_client: Optional[redis.Redis] = None
    _lite_redis_client: Optional[redis.Redis] = None
    _billing_redis_client: Optional[redis.Redis] = None
    _cm_redis_clients: Optional[list[redis.Redis]] = None
    cm_redis_shard_count: int = int(os.getenv("CM_REDIS_SHARD_COUNT", "5"))
    cm_redis_start_port: int = int(os.getenv("CM_REDIS_START_PORT", "1700"))
    cm_redis_socket_timeout: float = float(os.getenv("CM_REDIS_SOCKET_TIMEOUT", "30.0"))
    cm_redis_op_timeout: float = float(os.getenv("CM_REDIS_OP_TIMEOUT", "2.5"))

    @property
    def redis_url(self) -> str:
        return f"redis://:{self.redis_password}@{self.redis_host}:{self.redis_port}/{self.redis_db}"

    @property
    def redis_client(self) -> redis.Redis:
        if self._redis_client is None:
            self._redis_client = SafeRedis(
                host=self.redis_host,
                port=self.redis_port,
                db=self.redis_db,
                password=self.redis_password,
                socket_connect_timeout=self.redis_connect_timeout,
                socket_timeout=self.redis_socket_timeout,
                op_timeout=self.redis_op_timeout,
                max_connections=self.redis_max_connections,
                socket_keepalive=True,
                health_check_interval=30,
                retry_on_timeout=True,
                retry=Retry(ConstantBackoff(0.5), 2),
            )
        return self._redis_client

    @property
    def lite_redis_client(self) -> redis.Redis:
        if self._lite_redis_client is None:
            self._lite_redis_client = SafeRedis(
                host=self.redis_host,
                port=self.redis_port,
                db=self.redis_db + 1,
                password=self.redis_password,
                socket_connect_timeout=self.redis_connect_timeout,
                socket_timeout=self.redis_socket_timeout,
                op_timeout=self.redis_op_timeout,
                max_connections=self.redis_max_connections,
                socket_keepalive=True,
                health_check_interval=30,
                retry_on_timeout=True,
                retry=Retry(ConstantBackoff(0.5), 2),
            )
        return self._lite_redis_client

    @property
    def billing_redis_client(self) -> redis.Redis:
        if self._billing_redis_client is None:
            self._billing_redis_client = SafeRedis(
                host=self.redis_host,
                port=self.redis_port,
                db=self.redis_db + 2,
                password=self.redis_password,
                socket_connect_timeout=self.redis_connect_timeout,
                socket_timeout=self.redis_socket_timeout,
                op_timeout=self.redis_op_timeout,
                max_connections=self.redis_max_connections,
                socket_keepalive=True,
                health_check_interval=30,
                retry_on_timeout=True,
                retry=Retry(ConstantBackoff(0.5), 2),
            )
        return self._billing_redis_client

    @property
    def cm_redis_client(self) -> list[redis.Redis]:
        if self._cm_redis_clients is None:
            self._cm_redis_clients = [
                SafeRedis(
                    host=self.redis_host,
                    port=self.cm_redis_start_port + idx,
                    db=self.redis_db,
                    password=self.redis_password,
                    socket_connect_timeout=self.redis_connect_timeout,
                    socket_timeout=self.cm_redis_socket_timeout,
                    op_timeout=self.cm_redis_op_timeout,
                    max_connections=self.redis_max_connections,
                    socket_keepalive=True,
                    health_check_interval=30,
                    retry_on_timeout=True,
                    retry=Retry(ConstantBackoff(0.5), 2),
                )
                for idx in range(self.cm_redis_shard_count)
            ]
        return self._cm_redis_clients

    registry_host: str = os.getenv("REGISTRY_HOST", "registry:5000")
    registry_external_host: str = os.getenv("REGISTRY_EXTERNAL_HOST", "registry.chutes.ai")
    registry_password: str = os.getenv("REGISTRY_PASSWORD", "registrypassword")
    registry_insecure: bool = os.getenv("REGISTRY_INSECURE", "false").lower() == "true"
    build_timeout: int = int(os.getenv("BUILD_TIMEOUT", "7200"))
    push_timeout: int = int(os.getenv("PUSH_TIMEOUT", "7200"))
    scan_timeout: int = int(os.getenv("SCAN_TIMEOUT", "7200"))
    netuid: int = int(os.getenv("NETUID", "64"))
    subtensor: str = os.getenv("SUBTENSOR_ADDRESS", "wss://entrypoint-finney.opentensor.ai:443")
    payment_recovery_blocks: int = int(os.getenv("PAYMENT_RECOVERY_BLOCKS", "256"))
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

    # Reroll discount (i.e. duplicate prompts for re-roll in RP, or pass@k, etc.)
    reroll_multiplier: float = float(os.getenv("REROLL_MULTIPLIER", "0.1"))

    # Chutes pinned version.
    chutes_version: str = os.getenv("CHUTES_VERSION", "0.4.46")

    # Auto stake amount when DCAing into alpha after receiving payments.
    autostake_amount: float = float(os.getenv("AUTOSTAKE_AMOUNT", "10.0"))

    # Cosign Settings
    cosign_password: Optional[str] = os.getenv("COSIGN_PASSWORD")
    cosign_key: Optional[Path] = Path(os.getenv("COSIGN_KEY")) if os.getenv("COSIGN_KEY") else None

    # hCaptcha
    hcaptcha_sitekey: Optional[str] = os.getenv("HCAPTCHA_SITEKEY")
    hcaptcha_secret: Optional[str] = os.getenv("HCAPTCHA_SECRET")

    # TDX Attestation settings - Measurement configuration loaded from ConfigMap
    tee_measurement_config_path: Path = Path("/etc/config/tee_measurements.yaml")

    @property
    def tee_measurements(self) -> List[TeeMeasurementConfig]:
        """Load TEE measurement configurations from YAML file (mounted from ConfigMap).

        Re-reads the file on every access so that ConfigMap updates propagated
        by Kubernetes are picked up without restarting the pod.
        """
        return self._load_tee_measurements()

    def _load_tee_measurements(self) -> List[TeeMeasurementConfig]:
        """Parse and validate TEE measurement configurations from the YAML file."""
        try:
            with open(self.tee_measurement_config_path) as f:
                config = yaml.safe_load(f)
        except Exception as e:
            logger.error(f"Failed to load TEE measurement config: {e}")
            return []

        measurements: List[TeeMeasurementConfig] = []
        for measurement_config in config.get("measurements", []):
            config_name = measurement_config.get("name", "unnamed")
            version = measurement_config.get("version")
            if not version or not str(version).strip():
                error_msg = (
                    f"Missing or empty 'version' for measurement config '{config_name}'. "
                    "Each measurement configuration must have a version."
                )
                logger.error(error_msg)
                raise ValueError(error_msg)

            rtmr0_upper = measurement_config["boot_rtmrs"]["rtmr0"].upper().strip()
            if len(rtmr0_upper) != 96:
                logger.warning(
                    f"Invalid RTMR0 length for measurement config {config_name}: {len(rtmr0_upper)} (expected 96)"
                )
                continue

            mrtd_upper = measurement_config["mrtd"].upper().strip()
            if len(mrtd_upper) != 96:
                logger.warning(
                    f"Invalid MRTD length for measurement config {config_name}: {len(mrtd_upper)} (expected 96)"
                )
                continue

            boot_rtmrs = {
                k.upper(): v.upper().strip() for k, v in measurement_config["boot_rtmrs"].items()
            }
            runtime_rtmrs = {
                k.upper(): v.upper().strip() for k, v in measurement_config["runtime_rtmrs"].items()
            }

            if boot_rtmrs.get("RTMR0") != runtime_rtmrs.get("RTMR0"):
                logger.warning(
                    f"RTMR0 mismatch between boot and runtime for measurement config {config_name}. "
                    "This is unexpected - RTMR0 should be the same (ACPI tables don't change)."
                )

            measurements.append(
                TeeMeasurementConfig(
                    version=str(version).strip(),
                    mrtd=mrtd_upper,
                    name=measurement_config["name"],
                    boot_rtmrs=boot_rtmrs,
                    runtime_rtmrs=runtime_rtmrs,
                    expected_gpus=[gpu.lower() for gpu in measurement_config["expected_gpus"]],
                    gpu_count=measurement_config.get("gpu_count"),
                )
            )

        logger.info(f"Loaded {len(measurements)} TEE measurement configurations")
        return measurements

    luks_passphrase: Optional[str] = os.getenv("LUKS_PASSPHRASE")
    cache_passphrase_key: Optional[str] = os.getenv("CACHE_PASSPHRASE_KEY")

    # TDX verification service URLs (if using Intel's remote verification)
    tdx_verification_url: Optional[str] = os.getenv("TDX_VERIFICATION_URL")
    tdx_cert_chain_url: Optional[str] = os.getenv("TDX_CERT_CHAIN_URL")

    # Nonce expiration (minutes)
    attestation_nonce_expiry: int = int(os.getenv("ATTESTATION_NONCE_EXPIRY", "10"))

    # OpenRouter free usage settings.
    or_free_user_id: str = os.getenv("OR_FREE_USER_ID", "replaceme")

    # Premium chute IDs (restricted from $3/mo sub users without balance).
    premium_chute_ids: list = json.loads(os.getenv("PREMIUM_CHUTE_IDS", "[]"))


# Subscription tier: quota -> monthly price in USD (canonical values only).
SUBSCRIPTION_TIERS = {
    300: 3.0,
    2000: 10.0,
    5000: 20.0,
}
SUBSCRIPTION_MONTHLY_CAP_MULTIPLIER = 5.0
SUBSCRIPTION_4H_CAP_MULTIPLIER = 75.0
FOUR_HOUR_CHUNKS_PER_MONTH = 180  # 30 days * 24 hours / 4 hours


def get_subscription_tier(quota: int) -> float | None:
    """
    Get the monthly price for a subscription quota value.
    Handles off-by-one quotas (e.g., 301, 2001, 5001) used for free/comped subs.
    """
    if quota in SUBSCRIPTION_TIERS:
        return SUBSCRIPTION_TIERS[quota]
    if quota - 1 in SUBSCRIPTION_TIERS:
        return SUBSCRIPTION_TIERS[quota - 1]
    return None


settings = Settings()
