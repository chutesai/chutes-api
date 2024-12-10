"""
Application-wide settings.
"""

import os
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    sqlalchemy: str = os.getenv(
        "POSTGRESQL", "postgresql+asyncpg://user:password@127.0.0.1:5432/chutes"
    )
    subtensor: str = os.getenv("SUBTENSOR_ADDRESS", "wss://entrypoint-finney.opentensor.ai:443")
    netuid: int = os.getenv("NETUID", "19")
    redis_url: str = os.getenv("REDIS_URL", "redis://127.0.0.1:6379/0")
    debug: bool = os.getenv("DEBUG", "false").lower() == "true"


settings = Settings()
