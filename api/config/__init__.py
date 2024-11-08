"""
Application-wide settings.
"""

import os
import hvac
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
    registry_host: str = os.getenv("REGISTRY_HOST", "registry:5000")
    registry_external_host: str = os.getenv(
        "REGISTRY_EXTERNAL_HOST", "registry.chutes.ai"
    )
    registry_password: str = os.getenv("REGISTRY_PASSWORD", "registrypassword")
    registry_insecure: bool = os.getenv("REGISTRY_INSECURE", "false").lower() == "true"
    build_timeout: int = int(os.getenv("BUILD_TIMEOUT", "3600"))
    push_timeout: int = int(os.getenv("PUSH_TIMEOUT", "1800"))
    netuid: int = int(os.getenv("NETUID", "19"))
    subtensor: str = os.getenv(
        "SUBTENSOR_ADDRESS", "wss://entrypoint-finney.opentensor.ai:443"
    )
    registration_minimum_balance: float = float(
        os.getenv("REGISTRATION_MINIMUM_BALANCE", "0.5")
    )
    signup_bonus_balance: int = int(
        os.getenv("REGISTRATION_BONUS_BALANCE", str(1 * 10**9))
    )
    debug: bool = os.getenv("DEBUG", "false").lower() == "true"


settings = Settings()
