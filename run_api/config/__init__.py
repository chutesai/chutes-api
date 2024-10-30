"""
Application-wide settings.
"""

import os
from miniopy_async import Minio
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    sqlalchemy: str = os.getenv(
        "POSTGRESQL", "postgresql+asyncpg://user:password@127.0.0.1:5432/parachutes"
    )
    storage_client: Minio = Minio(
        os.getenv("MINIO_ENDPOINT", "127.0.0.1"),
        access_key=os.getenv("MINIO_ACCESS_KEY", "REPLACEME"),
        secret_key=os.getenv("MINIO_SECRET_KEY", "REPLACEME"),
        secure=os.getenv("MINIO_SECURE", "false").lower() == "true",
    )
    storage_bucket: str = os.getenv("STORAGE_BUCKET", "REPLACEME")
    redis_url: str = os.getenv("REDIS_URL", "redis://127.0.0.1:6379/0")
    registry_host: str = os.getenv("REGISTRY_HOST", "registry:5000")
    registry_external_host: str = os.getenv(
        "REGISTRY_EXTERNAL_HOST", "registry.parachutes.ai"
    )
    registry_insecure: bool = os.getenv("REGISTRY_INSECURE", "false").lower() == "true"
    build_timeout: int = int(os.getenv("BUILD_TIMEOUT", "3600"))
    push_timeout: int = int(os.getenv("PUSH_TIMEOUT", "1800"))


settings = Settings()
