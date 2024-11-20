"""
Endpoints specific to miners.
"""

import orjson as json
from fastapi import APIRouter, Depends
from starlette.responses import StreamingResponse
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from sqlalchemy.orm import class_mapper
from typing import Any
from api.chute.schemas import Chute
from api.user.schemas import User
from api.user.service import get_current_user
from api.image.schemas import Image
from api.database import get_db_session
from api.config import settings

router = APIRouter()


def model_to_dict(obj):
    """
    Helper to convert object to dict.
    """
    return {column.key: getattr(obj, column.key) for column in class_mapper(obj.__class__).columns}


async def _stream_items(db: AsyncSession, clazz: Any):
    """
    Streaming results helper.
    """
    query = select(clazz)
    result = await db.stream(query)
    async for row in result:
        yield f"data: {json.dumps(model_to_dict(row[0])).decode()}\n\n"


@router.get("/chutes/")
async def list_chutes(
    db: AsyncSession = Depends(get_db_session),
    _: User = Depends(get_current_user(purpose="miner", registered_to=settings.netuid)),
):
    return StreamingResponse(_stream_items(db, Chute))


@router.get("/images/")
async def list_images(
    db: AsyncSession = Depends(get_db_session),
    _: User = Depends(get_current_user(purpose="miner", registered_to=settings.netuid)),
):
    return StreamingResponse(_stream_items(db, Image))
