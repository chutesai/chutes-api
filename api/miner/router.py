"""
Endpoints specific to miners.
"""

import orjson as json
from sqlalchemy.ext.asyncio import AsyncSession
from fastapi_cache.decorator import cache
from fastapi import APIRouter, Depends, Header, status, HTTPException, Response
from starlette.responses import StreamingResponse
from sqlalchemy import text
from sqlalchemy.future import select
from sqlalchemy.orm import class_mapper
from typing import Any, Optional
from pydantic.fields import ComputedFieldInfo
import api.database.orms  # noqa
from api.user.schemas import User
from api.chute.schemas import Chute
from api.node.schemas import Node
from api.image.schemas import Image
from api.instance.schemas import Instance
from api.invocation.util import gather_metrics
from api.user.service import get_current_user
from api.database import get_session, get_db_session
from api.config import settings
from api.constants import HOTKEY_HEADER

router = APIRouter()


def model_to_dict(obj):
    """
    Helper to convert object to dict.
    """
    mapper = class_mapper(obj.__class__)
    data = {column.key: getattr(obj, column.key) for column in mapper.columns}
    for name, value in vars(obj.__class__).items():
        if isinstance(getattr(value, "decorator_info", None), ComputedFieldInfo):
            data[name] = getattr(obj, name)
    if isinstance(obj, Chute):
        data["image"] = f"{obj.image.user.username}/{obj.image.name}:{obj.image.tag}"
    return data


async def _stream_items(clazz: Any, selector: Any = None):
    """
    Streaming results helper.
    """
    async with get_session() as db:
        query = selector if selector is not None else select(clazz)
        result = await db.stream(query)
        async for row in result.unique():
            yield f"data: {json.dumps(model_to_dict(row[0])).decode()}\n\n"


@router.get("/chutes/")
async def list_chutes(
    _: User = Depends(get_current_user(purpose="miner", registered_to=settings.netuid)),
):
    return StreamingResponse(_stream_items(Chute))


@router.get("/images/")
async def list_images(
    _: User = Depends(get_current_user(purpose="miner", registered_to=settings.netuid)),
):
    return StreamingResponse(_stream_items(Image))


@router.get("/nodes/")
async def list_nodes(
    hotkey: str | None = Header(None, alias=HOTKEY_HEADER),
    _: User = Depends(get_current_user(purpose="miner", registered_to=settings.netuid)),
):
    return StreamingResponse(
        _stream_items(Node, selector=select(Node).where(Node.miner_hotkey == hotkey))
    )


@router.get("/instances/")
async def list_instances(
    hotkey: str | None = Header(None, alias=HOTKEY_HEADER),
    _: User = Depends(get_current_user(purpose="miner", registered_to=settings.netuid)),
):
    return StreamingResponse(
        _stream_items(
            Instance,
            selector=select(Instance).where(Instance.miner_hotkey == hotkey),
        )
    )


@router.get("/inventory")
async def get_full_inventory(
    hotkey: str | None = Header(None, alias=HOTKEY_HEADER),
    session: AsyncSession = Depends(get_db_session),
):
    query = text(
        f"""
    SELECT
      nodes.uuid AS gpu_id,
      instances.last_verified_at,
      instances.verification_error,
      chutes.chute_id,
      chutes.name AS chute_name
    FROM nodes
    JOIN instance_nodes ON nodes.uuid = instance_nodes.node_id
    JOIN instances ON instance_nodes.instance_id = instances.instance_id
    JOIN chutes ON instances.chute_id = chutes.chute_id
    WHERE nodes.miner_hotkey = '{hotkey}'
    """
    )
    result = await session.execute(query, {"hotkey": hotkey})
    return [dict(row._mapping) for row in result]


@router.get("/metrics/")
async def metrics(
    hotkey: str | None = Header(None, alias=HOTKEY_HEADER),
    _: User = Depends(get_current_user(purpose="miner", registered_to=settings.netuid)),
):
    async def _stream():
        async for metric in gather_metrics():
            yield f"data: {json.dumps(metric).decode()}\n\n"

    return StreamingResponse(_stream())


@router.get("/chutes/{chute_id}/{version}")
async def get_chute(
    chute_id: str,
    version: str,
    _: User = Depends(get_current_user(purpose="miner", registered_to=settings.netuid)),
):
    async with get_session() as db:
        chute = (
            await db.execute(
                select(Chute).where(Chute.chute_id == chute_id).where(Chute.version == version)
            )
        ).scalar_one_or_none()
        if not chute:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"{chute_id=} not found",
            )
        return model_to_dict(chute)


@cache(expire=60)
@router.get("/stats")
async def get_stats(
    session: AsyncSession = Depends(get_db_session), per_chute: Optional[bool] = False
) -> Response:
    """
    Get invocation status over different intervals.
    """
    bounty_query = """
        SELECT miner_hotkey, SUM(bounty) as total_bounty
          FROM invocations
         WHERE started_at >= NOW() - INTERVAL '{interval}'
           AND error_message IS NULL
           AND miner_uid > 0
         GROUP BY miner_hotkey
    """
    compute_query = """
        SELECT
            i.miner_hotkey,
            SUM(i.compute_multiplier * EXTRACT(EPOCH FROM (i.completed_at - i.started_at))) AS compute_units
        FROM invocations i
        WHERE i.started_at > NOW() - INTERVAL '{interval}'
        AND i.error_message IS NULL
        AND miner_uid > 0
        GROUP BY i.miner_hotkey
        HAVING SUM(i.compute_multiplier * EXTRACT(EPOCH FROM (i.completed_at - i.started_at))) > 0
        ORDER BY compute_units DESC
    """
    if per_chute:
        compute_query = """
        SELECT
            i.miner_hotkey,
            i.chute_id,
            SUM(i.compute_multiplier * EXTRACT(EPOCH FROM (i.completed_at - i.started_at))) AS compute_units
        FROM invocations i
        WHERE i.started_at > NOW() - INTERVAL '{interval}'
        AND i.error_message IS NULL
        AND miner_uid > 0
        GROUP BY i.miner_hotkey, i.chute_id
        HAVING SUM(i.compute_multiplier * EXTRACT(EPOCH FROM (i.completed_at - i.started_at))) > 0
        ORDER BY compute_units DESC
        """
    results = {}
    for interval, label in (("1 hour", "past_hour"), ("1 day", "past_day"), ("1 week", "all")):
        bounty_result = await session.execute(text(bounty_query.format(interval=interval)))
        compute_result = await session.execute(text(compute_query.format(interval=interval)))
        bounty_data = [
            {"miner_hotkey": row[0], "total_bounty": float(row[1])}
            for row in bounty_result.fetchall()
        ]
        compute_data = []
        if per_chute:
            compute_data = [
                {"miner_hotkey": row[0], "chute_id": row[1], "compute_units": float(row[2])}
                for row in compute_result.fetchall()
            ]
        else:
            compute_data = [
                {"miner_hotkey": row[0], "compute_units": float(row[1])}
                for row in compute_result.fetchall()
            ]
        results[label] = {
            "bounties": bounty_data,
            "compute_units": compute_data,
        }
    return results
