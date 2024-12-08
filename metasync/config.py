"""
Application-wide settings.
"""

import os
from typing import Optional
from fiber import Keypair
from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    sqlalchemy: str = os.getenv(
        "POSTGRESQL", "postgresql+asyncpg://user:password@127.0.0.1:5432/chutes"
    )
    netuid: int = os.getenv("NETUID", "19")
    redis_url: str = os.getenv("REDIS_URL", "redis://127.0.0.1:6379/0")
    debug: bool = os.getenv("DEBUG", "false").lower() == "true"

    validator_ss58: Optional[str] = os.getenv("VALIDATOR_SS58")
    validator_keypair: Optional[Keypair] = (
        Keypair.create_from_seed(os.environ["VALIDATOR_SEED"])
        if os.getenv("VALIDATOR_SEED")
        else None
    )

    subtensor_network: Optional[str] = Field(None, alias="SUBTENSOR_NETWORK")
    subtensor_address: Optional[str] = Field(None, alias="SUBTENSOR_ADDRESS")


settings = Settings()
