import uuid
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from api.config import settings
from typing import AsyncGenerator
from contextlib import asynccontextmanager

engine = create_async_engine(
    settings.sqlalchemy,
    echo=settings.debug,
)

SessionLocal = sessionmaker(
    bind=engine,
    class_=AsyncSession,
    expire_on_commit=False,
)

Base = declarative_base()


async def get_db_session() -> AsyncGenerator[AsyncSession, None]:
    """
    Obtain a DB session.
    """
    async with SessionLocal() as session:
        yield session


@asynccontextmanager
async def get_db() -> AsyncGenerator[AsyncSession, None]:
    async with SessionLocal() as session:
        yield session


def generate_uuid():
    """
    Helper for uuid generation.
    """
    return str(uuid.uuid4())
