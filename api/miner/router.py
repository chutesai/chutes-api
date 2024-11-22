"""
Endpoints specific to miners.
"""

import orjson as json
from fastapi import APIRouter, Depends, Header
from fastapi_cache.decorator import cache
from starlette.responses import StreamingResponse
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from sqlalchemy.orm import class_mapper
from typing import Any
from pydantic.fields import ComputedFieldInfo
import api.database.orms  # noqa
from api.user.schemas import User
from api.chute.schemas import Chute
from api.node.schemas import Node
from api.image.schemas import Image
from api.instance.schemas import Instance
from api.invocation.util import gather_metrics
from api.user.service import get_current_user
from api.database import get_db_session
from api.config import settings
from api.constants import HOTKEY_HEADER

router = APIRouter()


def model_to_dict(obj):
    """
    Helper to convert object to dict.
    """
    from loguru import logger

    mapper = class_mapper(obj.__class__)
    data = {column.key: getattr(obj, column.key) for column in mapper.columns}
    for name, value in vars(obj.__class__).items():
        if isinstance(getattr(value, "decorator_info", None), ComputedFieldInfo):
            logger.info(f"GOT THIS THING: {name=} {value=} {getattr(obj, name)}")
            data[name] = getattr(obj, name)
    return data


async def _stream_items(db: AsyncSession, clazz: Any, selector: Any = None):
    """
    Streaming results helper.
    """
    query = selector if selector is not None else select(clazz)
    result = await db.stream(query)
    async for row in result.unique():
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


@router.get("/nodes/")
async def list_nodes(
    db: AsyncSession = Depends(get_db_session),
    hotkey: str | None = Header(None, alias=HOTKEY_HEADER),
    _: User = Depends(get_current_user(purpose="miner", registered_to=settings.netuid)),
):
    return StreamingResponse(
        _stream_items(db, Node, selector=select(Node).where(Node.miner_hotkey == hotkey))
    )


@router.get("/instances/")
async def list_instances(
    db: AsyncSession = Depends(get_db_session),
    hotkey: str | None = Header(None, alias=HOTKEY_HEADER),
    _: User = Depends(get_current_user(purpose="miner", registered_to=settings.netuid)),
):
    return StreamingResponse(
        _stream_items(
            db,
            Instance,
            selector=select(Instance).where(Instance.miner_hotkey == hotkey),
        )
    )


@cache(expire=300)
@router.get("/metrics/")
async def metrics(
    hotkey: str | None = Header(None, alias=HOTKEY_HEADER),
    _: User = Depends(get_current_user(purpose="miner", registered_to=settings.netuid)),
):
    async def _stream():
        async for metric in gather_metrics():
            yield f"data: {json.dumps(metric).decode()}\n\n"

    return StreamingResponse(_stream())
