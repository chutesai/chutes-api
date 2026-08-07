"""
Application-wide settings.
"""

import os
import hashlib
from pathlib import Path
import aioboto3
import json
import yaml
from dataclasses import dataclass, field as dataclass_field
from api.safe_redis import SafeRedis
from functools import cached_property, lru_cache
import redis.asyncio as redis
from redis.retry import Retry
from redis.backoff import ConstantBackoff
from boto3.session import Config
from typing import ClassVar, Dict, List, Optional
from bittensor_wallet.keypair import Keypair
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict
from contextlib import asynccontextmanager
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.fernet import Fernet
from loguru import logger
from api.semver_util import semcomp


@lru_cache(maxsize=1)
def load_launch_config_private_key():
    if (path := os.getenv("LAUNCH_CONFIG_PRIVATE_KEY_PATH")) is not None:
        with open(path, "rb") as infile:
            return infile.read()
    return None


# Boot RTMR3 is always all-zeros: it cannot be extended until runtime (after the
# boot process unlocks the root volume), so it is a constant rather than configured.
ZERO_RTMR = "0" * 96


@dataclass
class TeeMeasurementConfig:
    """Configuration for allowed measurements for a single TEE VM version + hardware variant.

    RTMR0/RTMR1/RTMR2 are identical between boot and runtime; boot RTMR3 is always zero.
    MRTD/RTMR1/RTMR2 and runtime RTMR3 are shared across hardware for a given version
    (firmware, kernel/initrd, and guest measurements); only RTMR0 varies per hardware
    (GPU topology). The verbose boot_rtmrs/runtime_rtmrs dicts consumed elsewhere are
    derived from these scalar fields.

    rc marks a release-candidate / actively-tested version: it is still accepted for
    attestation, but is excluded from the public GET /tee/measurements endpoint and from
    the tee_minimum_boot_version fallback so it does not affect released VMs.
    """

    version: str
    name: str
    mrtd: str  # shared across hardware for this version
    rtmr0: str  # per-hardware (GPU topology)
    rtmr1: str  # shared across hardware for this version
    rtmr2: str  # shared across hardware for this version
    runtime_rtmr3: str  # shared across hardware; runtime-only (boot RTMR3 is zero)
    expected_gpus: List[str]
    gpu_count: Optional[int] = None
    rc: bool = False  # release candidate / in-test: attestable but unpublished
    # rc authorization allowlists (both required non-empty when rc is True; ignored for published
    # measurements). The two gate modes use different primitives, matched to their environment:
    #   * authorized_hotkeys -- miner hotkeys allowed on the register/runtime (userspace,
    #     get_current_user-authenticated) paths.
    #   * authorized_signing_keys -- operator RSA *public* keys (PEM) allowed on the boot/provision
    #     (initramfs) paths, where the VM signs the nonce with `openssl dgst -sha256 -sign` and no
    #     sr25519/substrate signer is available.
    # See api.server.util.authorize_rc_measurement.
    authorized_hotkeys: List[str] = dataclass_field(default_factory=list)
    authorized_signing_keys: List[str] = dataclass_field(default_factory=list)

    @property
    def boot_rtmrs(self) -> Dict[str, str]:
        """RTMRs expected in a boot quote (RTMR3 is never extended before runtime)."""
        return {
            "RTMR0": self.rtmr0,
            "RTMR1": self.rtmr1,
            "RTMR2": self.rtmr2,
            "RTMR3": ZERO_RTMR,
        }

    @property
    def runtime_rtmrs(self) -> Dict[str, str]:
        """RTMRs expected in a runtime quote."""
        return {
            "RTMR0": self.rtmr0,
            "RTMR1": self.rtmr1,
            "RTMR2": self.rtmr2,
            "RTMR3": self.runtime_rtmr3,
        }


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
        """Get validated Fernet cipher for LUKS passphrase encryption at rest.

        Encrypts all LUKS passphrases stored in the database (root, storage, cache
        volumes) so that database read access alone is insufficient to obtain them.

        Returns:
            Fernet cipher instance, or None if PASSPHRASE_ENCRYPTION_KEY not configured

        Raises:
            ValueError: If PASSPHRASE_ENCRYPTION_KEY is invalid format
        """
        key = os.getenv("PASSPHRASE_ENCRYPTION_KEY")
        if not key:
            return None

        # Fernet keys must be 32 url-safe base64-encoded bytes (44 characters)
        if len(key) != 44:
            raise ValueError(
                f"PASSPHRASE_ENCRYPTION_KEY must be 44 characters (32 bytes base64-encoded), got {len(key)} characters. "
                "Generate a valid key with: python -c 'from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())'"
            )

        try:
            return Fernet(key.encode())
        except Exception as e:
            raise ValueError(f"Invalid PASSPHRASE_ENCRYPTION_KEY format: {e}")

    sqlalchemy: str = os.getenv(
        "POSTGRESQL", "postgresql+asyncpg://user:password@127.0.0.1:5432/chutes"
    )
    postgres_ro: Optional[str] = os.getenv("POSTGRESQL_RO")

    aws_access_key_id: str = os.getenv("AWS_ACCESS_KEY_ID", "REPLACEME")
    aws_secret_access_key: str = os.getenv("AWS_SECRET_ACCESS_KEY", "REPLACEME")
    aws_endpoint_url: Optional[str] = os.getenv("AWS_ENDPOINT_URL", "http://minio:9000")
    aws_region: str = os.getenv("AWS_REGION", "local")
    storage_bucket: str = os.getenv("STORAGE_BUCKET", "chutes")
    s3_proxy_url: Optional[str] = os.getenv("S3_PROXY_URL")

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
            config=Config(
                signature_version="s3v4",
                proxies={"https": self.s3_proxy_url} if self.s3_proxy_url else None,
            ),
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
        default_factory=lambda: os.getenv("REDIS_HOST", "172.16.0.100"),
        validation_alias="PRIMARY_REDIS_HOST",
    )
    redis_port: int = Field(
        default_factory=lambda: int(os.getenv("REDIS_PORT", "6378")),
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
    redis_cacert: Optional[str] = os.getenv("REDIS_CACERT")

    _redis_client: Optional[redis.Redis] = None
    _lite_redis_client: Optional[redis.Redis] = None
    _billing_redis_client: Optional[redis.Redis] = None
    _cm_redis_client: Optional[redis.Redis] = None

    @property
    def redis_url(self) -> str:
        scheme = "rediss" if self.redis_cacert else "redis"
        base = (
            f"{scheme}://:{self.redis_password}@{self.redis_host}:{self.redis_port}/{self.redis_db}"
        )
        if self.redis_cacert:
            return f"{base}?ssl_cert_reqs=required&ssl_ca_certs={self.redis_cacert}"
        return base

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
                ssl_ca_certs=self.redis_cacert,
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
                ssl_ca_certs=self.redis_cacert,
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
                ssl_ca_certs=self.redis_cacert,
            )
        return self._billing_redis_client

    @property
    def cm_redis_client(self) -> redis.Redis:
        if self._cm_redis_client is None:
            self._cm_redis_client = SafeRedis(
                host=self.redis_host,
                port=self.redis_port,
                db=self.redis_db + 3,
                password=self.redis_password,
                socket_connect_timeout=self.redis_connect_timeout,
                socket_timeout=self.redis_socket_timeout,
                op_timeout=self.redis_op_timeout,
                max_connections=self.redis_max_connections,
                socket_keepalive=True,
                health_check_interval=30,
                retry_on_timeout=True,
                retry=Retry(ConstantBackoff(0.5), 2),
                ssl_ca_certs=self.redis_cacert,
            )
        return self._cm_redis_client

    registry_host: str = os.getenv("REGISTRY_HOST", "registry:5000")
    registry_external_host: str = os.getenv("REGISTRY_EXTERNAL_HOST", "registry.chutes.ai")
    registry_insecure: bool = os.getenv("REGISTRY_INSECURE", "false").lower() == "true"
    build_timeout: int = int(os.getenv("BUILD_TIMEOUT", "7200"))
    push_timeout: int = int(os.getenv("PUSH_TIMEOUT", "7200"))
    scan_timeout: int = int(os.getenv("SCAN_TIMEOUT", "7200"))
    netuid: int = int(os.getenv("NETUID", "64"))
    subtensor: str = os.getenv("SUBTENSOR_ADDRESS", "wss://entrypoint-finney.opentensor.ai:443")
    mev_protection_enabled: bool = os.getenv("MEV_PROTECTION_ENABLED", "false").lower() == "true"
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

    # User JWT salt.
    user_jwt_salt: Optional[str] = os.getenv("USER_JWT_SALT", "replaceme")

    # Flag indicating that all accounts created are free.
    all_accounts_free: bool = os.getenv("ALL_ACCOUNTS_FREE", "false").lower() == "true"

    # Consecutive failure count that triggers instance deletion.
    consecutive_failure_limit: int = int(os.getenv("CONSECUTIVE_FAILURE_LIMIT", "7"))

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
    default_quotas: dict = json.loads(os.getenv("DEFAULT_QUOTAS", '{"*": 0}'))
    default_discounts: dict = json.loads(os.getenv("DEFAULT_DISCOUNTS", '{"*": 0.0}'))
    default_job_quotas: dict = json.loads(os.getenv("DEFAULT_JOB_QUOTAS", '{"*": 0}'))

    # Reroll discount (i.e. duplicate prompts for re-roll in RP, or pass@k, etc.)
    reroll_multiplier: float = float(os.getenv("REROLL_MULTIPLIER", "0.1"))

    # Magic discount header: when a request includes this header with the correct value,
    # a discount is applied to both quota increment and paygo charges.
    magic_discount_header_key: Optional[str] = os.getenv("MAGIC_DISCOUNT_HEADER_KEY")
    magic_discount_header_val: Optional[str] = os.getenv("MAGIC_DISCOUNT_HEADER_VAL")
    magic_discount_amount: float = float(os.getenv("MAGIC_DISCOUNT_AMOUNT", "0.5"))

    # Chutes pinned version.
    chutes_version: str = os.getenv("CHUTES_VERSION", "0.4.46")

    # Auto stake amount when DCAing into alpha after receiving payments.
    autostake_amount: float = float(os.getenv("AUTOSTAKE_AMOUNT", "10.0"))

    # Depot.dev settings (remote image building).
    depot_token: str = os.getenv("DEPOT_TOKEN", "")
    depot_project_id: str = os.getenv("DEPOT_PROJECT_ID", "")
    depot_registry: str = os.getenv("DEPOT_REGISTRY", "")
    depot_registry_token: str = os.getenv("DEPOT_REGISTRY_TOKEN", "")
    depot_registry_rw_token: str = os.getenv("DEPOT_REGISTRY_RW_TOKEN", "")

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

        def _hex96(value: str, field_name: str, owner: str) -> str:
            """Upper-case/strip a hex measurement and validate it is 96 chars."""
            cleaned = str(value).upper().strip()
            if len(cleaned) != 96:
                raise ValueError(
                    f"Invalid {field_name} length for measurement config '{owner}': "
                    f"{len(cleaned)} chars (expected 96)."
                )
            return cleaned

        measurements: List[TeeMeasurementConfig] = []
        for version_config in config.get("measurements", []):
            version = version_config.get("version")
            if not version or not str(version).strip():
                error_msg = (
                    "Missing or empty 'version' for a measurement config. "
                    "Each measurement configuration must have a version."
                )
                logger.error(error_msg)
                raise ValueError(error_msg)
            version = str(version).strip()

            # MRTD/RTMR1/RTMR2 and runtime RTMR3 are shared across all hardware variants for
            # a version (firmware, kernel/initrd, and guest measurements). Only RTMR0 varies
            # per hardware (GPU topology). Boot RTMR3 is always zero (set on the config props).
            mrtd = _hex96(version_config["mrtd"], "MRTD", version)
            rtmr1 = _hex96(version_config["rtmr1"], "RTMR1", version)
            rtmr2 = _hex96(version_config["rtmr2"], "RTMR2", version)
            runtime_rtmr3 = _hex96(version_config["runtime_rtmr3"], "runtime RTMR3", version)
            rc = bool(version_config.get("rc", False))

            # rc authorization allowlists. Meaningful only for rc measurements (published ones lock
            # down identical guest software for everyone, so operator identity is irrelevant there):
            #   authorized_hotkeys      -> register/runtime paths, authorized_signing_keys -> boot/provision.
            authorized_hotkeys = [
                str(hk).strip()
                for hk in (version_config.get("authorized_hotkeys") or [])
                if str(hk).strip()
            ]
            authorized_signing_keys = [
                str(k).strip()
                for k in (version_config.get("authorized_signing_keys") or [])
                if str(k).strip()
            ]

            # Load-time invariant: an rc measurement with no allowlist would be usable by anyone who
            # can build the same image -- exactly the exposure rc gating exists to prevent. A full VM
            # lifecycle hits BOTH gate modes, so require both allowlists non-empty, and require every
            # signing key to parse as a PEM public key (a bad key can't verify anything, so treating
            # it as usable would be a silent hole). Drop the whole version (all hardware variants) and
            # log loudly rather than raise, so one misconfigured rc entry can't take down attestation
            # for published VMs; a dropped entry never matches any quote, so affected VMs fail closed.
            drop_reason = None
            if rc and not authorized_hotkeys:
                drop_reason = "'authorized_hotkeys' allowlist is empty"
            elif rc and not authorized_signing_keys:
                drop_reason = "'authorized_signing_keys' allowlist is empty"
            elif rc:
                for pem in authorized_signing_keys:
                    try:
                        serialization.load_pem_public_key(pem.encode())
                    except Exception as e:
                        drop_reason = f"an 'authorized_signing_keys' entry is not a valid PEM public key ({e})"
                        break
            if drop_reason:
                logger.error(
                    f"Refusing to load rc measurement version '{version}': {drop_reason}. rc "
                    "measurements MUST declare non-empty, valid 'authorized_hotkeys' and "
                    "'authorized_signing_keys'. This version is UNUSABLE until fixed; VMs on it "
                    "will fail attestation."
                )
                continue

            hardware = version_config.get("hardware") or []
            if not hardware:
                raise ValueError(
                    f"Measurement config version '{version}' must define at least one "
                    "'hardware' variant."
                )

            for hw in hardware:
                hw_name = hw.get("name", "unnamed")
                owner = f"{version}/{hw_name}"
                rtmr0 = _hex96(hw["rtmr0"], "RTMR0", owner)

                gpu_count = hw.get("gpu_count")
                if gpu_count is None:
                    raise ValueError(
                        f"Missing 'gpu_count' for measurement config '{owner}'. "
                        "All TEE measurement hardware variants must specify gpu_count."
                    )

                measurements.append(
                    TeeMeasurementConfig(
                        version=version,
                        name=hw_name,
                        mrtd=mrtd,
                        rtmr0=rtmr0,
                        rtmr1=rtmr1,
                        rtmr2=rtmr2,
                        runtime_rtmr3=runtime_rtmr3,
                        expected_gpus=[gpu.lower() for gpu in hw["expected_gpus"]],
                        gpu_count=gpu_count,
                        rc=rc,
                        authorized_hotkeys=authorized_hotkeys,
                        authorized_signing_keys=authorized_signing_keys,
                    )
                )

        logger.info(f"Loaded {len(measurements)} TEE measurement configurations")
        return measurements

    @property
    def tee_minimum_boot_version(self) -> str:
        """Minimum VM version accepted for boot attestation.

        Returns TEE_MINIMUM_BOOT_VERSION when set, allowing new platform measurement
        configs to be added to the YAML incrementally without immediately enforcing a
        version bump for platforms not yet upgraded.  Falls back to the highest non-rc
        version found across all loaded measurement configs, or "0.0.0" if the config file
        is not present (e.g. pods that don't mount the TEE measurements ConfigMap).

        Release-candidate (rc) versions are excluded so an in-test version does not raise
        the minimum and lock out released VMs.
        """
        if pinned := os.getenv("TEE_MINIMUM_BOOT_VERSION"):
            return pinned
        if not self.tee_measurement_config_path.exists():
            return "0.0.0"
        versions = [m.version for m in self.tee_measurements if m.version and not m.rc]
        if not versions:
            return "0.0.0"
        latest = versions[0]
        for v in versions[1:]:
            if semcomp(v, latest) > 0:
                latest = v
        return latest

    signing_keys_bundle_path: Path = Path(
        os.getenv("SIGNING_KEYS_BUNDLE_PATH", "/etc/config/signing_keys_bundle.json")
    )

    _REQUIRED_SIGNING_KEY_NAMES: ClassVar[frozenset] = frozenset(
        ["cosign/chutes.pub", "cosign/dockerhub.pub", "helm-pubkey.gpg"]
    )

    @cached_property
    def signing_keys_bundle(self) -> Optional[dict]:
        """Load and validate the signing keys bundle from the configured path.

        Loaded once and cached for the lifetime of the process. Returns None if the
        file does not exist (pods without the ConfigMap mounted will return 503).

        Raises ValueError if the file exists but fails validation.
        """
        if not self.signing_keys_bundle_path.exists():
            logger.warning(
                f"Signing keys bundle not found at {self.signing_keys_bundle_path}; "
                "/servers/signing-keys will return 503"
            )
            return None

        with open(self.signing_keys_bundle_path) as fh:
            bundle = json.loads(fh.read())

        if not isinstance(bundle.get("version"), int):
            raise ValueError("signing_keys_bundle: 'version' must be an integer")
        for field in ("keys", "signatures"):
            if not isinstance(bundle.get(field), dict):
                raise ValueError(f"signing_keys_bundle: '{field}' must be an object")
        missing = self._REQUIRED_SIGNING_KEY_NAMES - bundle["keys"].keys()
        if missing:
            raise ValueError(f"signing_keys_bundle: missing required keys: {missing}")
        missing_sigs = self._REQUIRED_SIGNING_KEY_NAMES - bundle["signatures"].keys()
        if missing_sigs:
            raise ValueError(f"signing_keys_bundle: missing required signatures: {missing_sigs}")
        # Validate keys and signatures separately: they share the same names, so
        # merging them into one dict would let a valid signature mask an empty key.
        for section in ("keys", "signatures"):
            for name, value in bundle[section].items():
                if not isinstance(value, str) or not value.strip():
                    raise ValueError(
                        f"signing_keys_bundle: value for '{section}/{name}' must be a non-empty string"
                    )

        logger.info(
            f"Signing keys bundle loaded from {self.signing_keys_bundle_path} "
            f"(version={bundle['version']}, keys={list(bundle['keys'].keys())})"
        )
        return bundle

    # Attestation-proxy provenance secrets. Two proxies front the attestation endpoints during
    # the 1.3.x -> 1.4.0 migration; the API tells them apart by which secret matched
    # (see require_attestation_proxy). Every other proxy (esp. api.chutes.ai) must strip both.
    #   attestation_proxy_secret: injected by the attestation proxy (tdx-attestation.chutes.ai) as
    #     X-Attestation-Proxy-Auth; used by legacy 1.3.x VMs (throwaway-cert mTLS).
    #   cvm_proxy_secret: injected by the cvm proxy (cvm.chutes.ai) as X-Cvm-Proxy-Auth; used by
    #     1.4.0+ VMs (registered-CA mTLS). A match marks the request mTLS-verified and is
    #     required by the 1.4.0-only endpoints (provision, provision/confirm).
    attestation_proxy_secret: Optional[str] = os.getenv("ATTESTATION_PROXY_SECRET")
    cvm_proxy_secret: Optional[str] = os.getenv("CVM_PROXY_SECRET")

    # Chute log shipper (see api/chute_logs): the in-guest agent streams pre-registration chute pod
    # logs to POST /instances/launch_config/{config_id}/logs; the validator pushes them to a
    # dedicated in-namespace Loki (NOT the ops monitoring cluster) for a bounded window, read back
    # by owners / the miner CLI and surfaced to support via the ops Grafana. When loki_url is unset
    # the ingest endpoint accepts + drops (still returning the cutoff signal) so the guest is a no-op
    # until Loki is provisioned.
    loki_url: Optional[str] = os.getenv("LOKI_URL")
    loki_tenant_id: Optional[str] = os.getenv("LOKI_TENANT_ID")
    loki_timeout_seconds: float = float(os.getenv("LOKI_TIMEOUT_SECONDS", "10.0"))
    # Bounds on a single shipment so a misbehaving/hostile guest cannot flood the store.
    chute_logs_max_lines_per_shipment: int = int(
        os.getenv("CHUTE_LOGS_MAX_LINES_PER_SHIPMENT", "5000")
    )
    chute_logs_max_line_bytes: int = int(os.getenv("CHUTE_LOGS_MAX_LINE_BYTES", "32768"))
    # Logs stream in as batches, often several back-to-back per poll. Cache the (expensive) mTLS
    # authentication per (config_id, cert) so we verify the leaf + do the lookups once, not per batch.
    # The auth result is immutable for the life of a boot, so the TTL is generous.
    # TTL for the Redis-shared auth cache (config+cert → resolved identity). The identity is stable
    # for the config's lifetime, so this only bounds staleness on CA/ownership changes.
    chute_logs_auth_cache_seconds: int = int(os.getenv("CHUTE_LOGS_AUTH_CACHE_SECONDS", "300"))

    # Shared secret injected by the registry.chutes.ai nginx frontend as
    # X-Registry-Proxy-Auth.  When set, the /registry/auth handler refuses
    # requests that do not carry this header, preventing X-Client-Cert spoofing
    # from clients that bypass the registry nginx proxy.
    registry_proxy_secret: Optional[str] = os.getenv("REGISTRY_PROXY_SECRET")

    # Registry auth path selector (see api/registry/router.py): a VM whose attested
    # measurement version is >= this must authenticate to the registry via mTLS;
    # older VMs use legacy Bittensor hotkey/signature/nonce auth.  Defaults to the
    # 1.4.0 SEK8S release that ships mTLS registry support.  Set to "0.0.0" to force
    # every attested VM onto mTLS — the kill switch that retires legacy auth.
    registry_mtls_min_version: str = os.getenv("REGISTRY_MTLS_MIN_VERSION", "1.4.0")

    # Attestation mTLS gate (see gate_legacy_attestation): a VM whose attested measurement
    # version is >= this must reach the transitional attestation endpoints (nonce, luks/confirm)
    # via the cvm proxy rather than the legacy api.chutes.ai path. Older/unknown VMs still use
    # the legacy path. Set to "0.0.0" to force every attested VM onto the cvm proxy -- the kill
    # switch that closes the legacy attestation path once the fleet is migrated.
    tee_mtls_min_version: str = os.getenv("TEE_MTLS_MIN_VERSION", "1.4.0")

    luks_passphrase: Optional[str] = os.getenv("LUKS_PASSPHRASE")
    passphrase_encryption_key: Optional[str] = os.getenv("PASSPHRASE_ENCRYPTION_KEY")

    @cached_property
    def luks_passphrases(self) -> Dict[str, str]:
        """Root-volume LUKS passphrases keyed by measurement version.

        Parsed from the LUKS_PASSPHRASES env var (a JSON object mapping version -> passphrase).
        Each VM image bakes in a version-specific root-volume passphrase, so /boot/attestation
        returns the passphrase matching the attested measurement version. There is no fallback:
        every accepted version must have an entry.

        Raises:
            ValueError: If LUKS_PASSPHRASES is set but is not a non-empty {str: str} JSON object.
        """
        raw = os.getenv("LUKS_PASSPHRASES")
        if not raw:
            return {}
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError as e:
            raise ValueError(f"LUKS_PASSPHRASES must be valid JSON: {e}")
        if not isinstance(parsed, dict) or not parsed:
            raise ValueError(
                "LUKS_PASSPHRASES must be a non-empty JSON object mapping version -> passphrase."
            )
        if not all(isinstance(k, str) and isinstance(v, str) and v for k, v in parsed.items()):
            raise ValueError(
                "LUKS_PASSPHRASES must map string versions to non-empty string passphrases."
            )
        return parsed

    # TDX verification service URLs (if using Intel's remote verification)
    tdx_verification_url: Optional[str] = os.getenv("TDX_VERIFICATION_URL")
    tdx_cert_chain_url: Optional[str] = os.getenv("TDX_CERT_CHAIN_URL")

    # Nonce expiration (minutes)
    attestation_nonce_expiry: int = int(os.getenv("ATTESTATION_NONCE_EXPIRY", "10"))

    # OpenRouter free usage settings.
    or_free_user_id: str = os.getenv("OR_FREE_USER_ID", "replaceme")

    # Agent registration settings.
    agent_registration_threshold: float = float(os.getenv("AGENT_REGISTRATION_THRESHOLD", "50.0"))
    agent_registration_tolerance: float = float(os.getenv("AGENT_REGISTRATION_TOLERANCE", "0.10"))
    agent_registration_ttl_hours: int = int(os.getenv("AGENT_REGISTRATION_TTL_HOURS", "24"))

    # TEE server health prober settings.
    # Age of last successful probe past which a server is flagged degraded (default 12h) / offline (default 72h).
    server_health_degraded_threshold_seconds: int = int(
        os.getenv("SERVER_HEALTH_DEGRADED_THRESHOLD_SECONDS", str(12 * 3600))
    )
    server_health_offline_threshold_seconds: int = int(
        os.getenv("SERVER_HEALTH_OFFLINE_THRESHOLD_SECONDS", str(72 * 3600))
    )
    server_health_max_concurrent: int = int(os.getenv("SERVER_HEALTH_MAX_CONCURRENT", "32"))


# Subscription tier: quota -> monthly price in USD (canonical values only).
SUBSCRIPTION_TIERS = {
    300: 3.0,
    2000: 10.0,
    5000: 20.0,
}
SUBSCRIPTION_PAYGO_DISCOUNTS = {
    3.0: 0.03,
    10.0: 0.06,
    20.0: 0.1,
}
SUBSCRIPTION_MONTHLY_CAP_MULTIPLIER = 5.0
SUBSCRIPTION_4H_CAP_MULTIPLIER = 75.0
FOUR_HOUR_CHUNKS_PER_MONTH = 180  # 30 days * 24 hours / 4 hours


def get_subscription_tier(quota: int) -> float | None:
    """
    Get the monthly price for a subscription quota value.
    Handles off-by-one quotas (e.g., 301, 2001, 5001) used for custom subs.
    """
    if quota in SUBSCRIPTION_TIERS:
        return SUBSCRIPTION_TIERS[quota]
    if quota - 1 in SUBSCRIPTION_TIERS:
        return SUBSCRIPTION_TIERS[quota - 1]
    return None


def is_custom_subscription(quota: int) -> bool:
    """Off-by-one quotas represent custom subscriptions."""
    return quota not in SUBSCRIPTION_TIERS and quota - 1 in SUBSCRIPTION_TIERS


settings = Settings()
