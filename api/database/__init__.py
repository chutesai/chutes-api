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
    pool_size=settings.db_pool_size,
    max_overflow=settings.db_overflow,
    pool_pre_ping=True,
    pool_reset_on_return="rollback",
    pool_timeout=30,
    pool_recycle=900,
    pool_use_lifo=True,
    connect_args={"ssl": "require"},
)

SessionLocal = sessionmaker(
    bind=engine,
    class_=AsyncSession,
    expire_on_commit=False,
)

ro_engine = (
    engine
    if not settings.postgres_ro
    else create_async_engine(
        settings.postgres_ro,
        echo=settings.debug,
        pool_size=settings.db_pool_size,
        max_overflow=settings.db_overflow,
        pool_pre_ping=True,
        pool_reset_on_return="rollback",
        pool_timeout=30,
        pool_recycle=900,
        pool_use_lifo=True,
        connect_args={"ssl": "require"},
    )
)
SessionLocalRead = sessionmaker(
    bind=ro_engine,
    class_=AsyncSession,
    expire_on_commit=False,
)

Base = declarative_base()


@asynccontextmanager
async def get_session(readonly=False) -> AsyncGenerator[AsyncSession, None]:
    session_maker = SessionLocalRead if readonly else SessionLocal
    async with session_maker() as session:
        try:
            yield session
            if not readonly:
                await session.commit()
        except Exception:
            if not readonly:
                try:
                    await session.rollback()
                except Exception:
                    pass
            raise


async def get_db_session():
    async with SessionLocal() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            try:
                await session.rollback()
            except Exception:
                pass
            raise


async def get_db_ro_session():
    async with SessionLocalRead() as session:
        yield session


def generate_uuid():
    """
    Helper for uuid generation.
    """
    return str(uuid.uuid4())
