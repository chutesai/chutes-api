import uuid
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from run_api.config import settings


engine = create_async_engine(
    settings.sqlalchemy,
    echo=True,
)

SessionLocal = sessionmaker(
    bind=engine,
    class_=AsyncSession,
    expire_on_commit=False,
)

Base = declarative_base()


async def get_db_session():
    """
    Obtain a DB session.
    """
    async with SessionLocal() as session:
        yield session


def generate_uuid():
    """
    Helper for uuid generation.
    """
    return str(uuid.uuid4())
