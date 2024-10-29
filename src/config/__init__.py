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
    )
    storage_bucket: str = os.getenv("STORAGE_BUCKET", "REPLACEME")


settings = Settings()
