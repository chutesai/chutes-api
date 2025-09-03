"""
Endpoints specific to miners.
"""

import re
import orjson as json
from decimal import Decimal
from sqlalchemy.ext.asyncio import AsyncSession
from fastapi import APIRouter, Depends, Header, status, HTTPException, Response, Request
from starlette.responses import StreamingResponse
from sqlalchemy import text
from sqlalchemy.future import select
from sqlalchemy.orm import class_mapper
from typing import Any, Optional
from pydantic.fields import ComputedFieldInfo
import api.database.orms  # noqa
from api.user.schemas import User
from api.chute.schemas import Chute, NodeSelector
from api.node.schemas import Node
from api.image.schemas import Image
from api.instance.schemas import Instance
from api.job.schemas import Job
from api.invocation.util import gather_metrics
from api.user.service import get_current_user
from api.database import get_session, get_db_session
from api.config import settings
from api.constants import HOTKEY_HEADER
from api.metasync import get_miner_by_hotkey, MetagraphNode
from api.util import memcache_get, memcache_set
from metasync.shared import get_scoring_data
from metasync.constants import UTILIZATION_RATIO_QUERY

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
        if obj.image.patch_version not in (None, "initial"):
            data["image"] += f"-{obj.image.patch_version}"
        ns = NodeSelector(**obj.node_selector)
        data["node_selector"].update(
            {
                "compute_multiplier": ns.compute_multiplier,
                "supported_gpus": ns.supported_gpus,
            }
        )
    if isinstance(obj, Image):
        data["username"] = obj.user.username
    if isinstance(data.get("seed"), Decimal):
        data["seed"] = int(data["seed"])
    return data


async def _stream_items(clazz: Any, selector: Any = None, explicit_null: bool = False):
    """
    Streaming results helper.
    """
    async with get_session() as db:
        query = selector if selector is not None else select(clazz)
        result = await db.stream(query)
        any_found = False
        async for row in result.unique():
            yield f"data: {json.dumps(model_to_dict(row[0])).decode()}\n\n"
            any_found = True
        if explicit_null and not any_found:
            yield "data: NO_ITEMS\n"


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
    explicit_null: Optional[bool] = False,
    hotkey: str | None = Header(None, alias=HOTKEY_HEADER),
    _: User = Depends(get_current_user(purpose="miner", registered_to=settings.netuid)),
):
    return StreamingResponse(
        _stream_items(
            Instance,
            selector=select(Instance).where(Instance.miner_hotkey == hotkey),
            explicit_null=explicit_null,
        )
    )


@router.get("/jobs/")
async def list_available_jobs(
    _: User = Depends(get_current_user(purpose="miner", registered_to=settings.netuid)),
):
    return StreamingResponse(
        _stream_items(Job, selector=select(Job).where(Job.instance_id.is_(None)))
    )


@router.delete("/jobs/{job_id}")
async def release_job(
    job_id: str,
    hotkey: str | None = Header(None, alias=HOTKEY_HEADER),
    _: User = Depends(get_current_user(purpose="miner", registered_to=settings.netuid)),
    db: AsyncSession = Depends(get_db_session),
):
    job = (
        (
            await db.execute(
                select(Job).where(
                    Job.miner_hotkey == hotkey, Job.finished_at.is_(None), Job.job_id == job_id
                )
            )
        )
        .unique()
        .scalar_one_or_none()
    )
    if not job:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"{job_id=} not found or is not associated with your miner",
        )
    job.miner_uid = None
    job.miner_hotkey = None
    job.miner_coldkey = None
    job.instance_id = None
    await db.commit()
    await db.refresh(job)

    # Send a new job_created notification.
    node_selector = NodeSelector(**job.chute.node_selector)
    await settings.redis_client.publish(
        "miner_broadcast",
        json.dumps(
            {
                "reason": "job_created",
                "data": {
                    "job_id": job.job_id,
                    "method": job.method,
                    "chute_id": job.chute_id,
                    "image_id": job.chute.image.image_id,
                    "gpu_count": node_selector.gpu_count,
                    "compute_multiplier": job.compute_multiplier,
                    "exclude": job.miner_history,
                },
            }
        ).decode(),
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
      instances.active,
      chutes.chute_id,
      chutes.name AS chute_name
    FROM nodes
    JOIN instance_nodes ON nodes.uuid = instance_nodes.node_id
    JOIN instances ON instance_nodes.instance_id = instances.instance_id
    JOIN chutes ON instances.chute_id = chutes.chute_id
    JOIN metagraph_nodes on instances.miner_hotkey = metagraph_nodes.hotkey AND metagraph_nodes.netuid = 64
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
            (
                await db.execute(
                    select(Chute).where(Chute.chute_id == chute_id).where(Chute.version == version)
                )
            )
            .unique()
            .scalar_one_or_none()
        )
        if not chute:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"{chute_id=} not found",
            )
        return model_to_dict(chute)


@router.get("/stats")
async def get_stats(
    miner_hotkey: Optional[str] = None,
    session: AsyncSession = Depends(get_db_session),
    per_chute: Optional[bool] = False,
    request: Request = None,
) -> Response:
    """
    Get invocation status over different intervals.
    """

    cache_key = f"mstats:{per_chute}".encode()

    def _filter_by_key(mstats):
        if miner_hotkey:
            for _, data in mstats.items():
                for key in data:
                    data[key] = [v for v in data[key] if v["miner_hotkey"] == miner_hotkey]
        return mstats

    if request:
        cached = await memcache_get(cache_key)
        if cached:
            return _filter_by_key(json.loads(cached))

    if miner_hotkey and not re.match(r"^[a-zA-Z0-9]{48}$", miner_hotkey):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid hotkey parameter."
        )
    bounty_query = """
    SELECT miner_hotkey, SUM(bounty) as total_bounty
      FROM invocations i
      JOIN metagraph_nodes mn on i.miner_hotkey = mn.hotkey AND mn.netuid = 64
     WHERE started_at >= NOW() - INTERVAL '{interval}'
       AND error_message IS NULL
       AND miner_uid >= 0
       AND NOT EXISTS (
          SELECT 1
          FROM reports
          WHERE invocation_id = i.parent_invocation_id
          AND confirmed_at IS NOT NULL
       )
    GROUP BY miner_hotkey
    """
    compute_query = """
    SELECT
        i.miner_hotkey,
        COUNT(*) AS total_invocations,
        SUM(
            i.bounty +
            i.compute_multiplier *
            CASE
                WHEN i.metrics->>'steps' IS NOT NULL
                    AND (i.metrics->>'steps')::float > 0
                    AND i.metrics->>'masps' IS NOT NULL
                THEN (i.metrics->>'steps')::float * (i.metrics->>'masps')::float
                WHEN i.metrics->>'it' IS NOT NULL
                    AND i.metrics->>'ot' IS NOT NULL
                    AND (i.metrics->>'it')::float > 0
                    AND (i.metrics->>'ot')::float > 0
                    AND i.metrics->>'maspt' IS NOT NULL
                THEN ((i.metrics->>'it')::float + (i.metrics->>'ot')::float) * (i.metrics->>'maspt')::float
                ELSE EXTRACT(EPOCH FROM (i.completed_at - i.started_at))
            END
        ) AS compute_units
    FROM invocations i
    JOIN metagraph_nodes mn on i.miner_hotkey = mn.hotkey AND mn.netuid = 64
    WHERE i.started_at > NOW() - INTERVAL '{interval}'
    AND i.error_message IS NULL
    AND miner_uid >= 0
    AND i.completed_at IS NOT NULL
    AND NOT EXISTS (
        SELECT 1
        FROM reports
        WHERE invocation_id = i.parent_invocation_id
        AND confirmed_at IS NOT NULL
    )
    GROUP BY i.miner_hotkey
    HAVING SUM(
        i.bounty +
        i.compute_multiplier *
        CASE
            WHEN i.metrics->>'steps' IS NOT NULL
                AND (i.metrics->>'steps')::float > 0
                AND i.metrics->>'masps' IS NOT NULL
            THEN (i.metrics->>'steps')::float * (i.metrics->>'masps')::float
            WHEN i.metrics->>'it' IS NOT NULL
                AND i.metrics->>'ot' IS NOT NULL
                AND (i.metrics->>'it')::float > 0
                AND (i.metrics->>'ot')::float > 0
                AND i.metrics->>'maspt' IS NOT NULL
            THEN ((i.metrics->>'it')::float + (i.metrics->>'ot')::float) * (i.metrics->>'maspt')::float
            ELSE EXTRACT(EPOCH FROM (i.completed_at - i.started_at))
        END
    ) > 0
    ORDER BY compute_units DESC
    """
    if per_chute:
        compute_query = """
        SELECT
            i.miner_hotkey,
            i.chute_id,
            COUNT(*) AS total_invocations,
            SUM(
                i.bounty +
                i.compute_multiplier *
                CASE
                    WHEN i.metrics->>'steps' IS NOT NULL
                        AND (i.metrics->>'steps')::float > 0
                        AND i.metrics->>'masps' IS NOT NULL
                    THEN (i.metrics->>'steps')::float * (i.metrics->>'masps')::float
                    WHEN i.metrics->>'it' IS NOT NULL
                        AND i.metrics->>'ot' IS NOT NULL
                        AND (i.metrics->>'it')::float > 0
                        AND (i.metrics->>'ot')::float > 0
                        AND i.metrics->>'maspt' IS NOT NULL
                    THEN ((i.metrics->>'it')::float + (i.metrics->>'ot')::float) * (i.metrics->>'maspt')::float
                    ELSE EXTRACT(EPOCH FROM (i.completed_at - i.started_at))
                END
            ) AS compute_units
        FROM invocations i
        JOIN metagraph_nodes mn on i.miner_hotkey = mn.hotkey AND mn.netuid = 64
        WHERE i.started_at > NOW() - INTERVAL '{interval}'
        AND i.error_message IS NULL
        AND miner_uid >= 0
        AND i.completed_at IS NOT NULL
        AND NOT EXISTS (
            SELECT 1
            FROM reports
            WHERE invocation_id = i.parent_invocation_id
            AND confirmed_at IS NOT NULL
        )
        GROUP BY i.miner_hotkey, i.chute_id
        HAVING SUM(
            i.bounty +
            i.compute_multiplier *
            CASE
                WHEN i.metrics->>'steps' IS NOT NULL
                    AND (i.metrics->>'steps')::float > 0
                    AND i.metrics->>'masps' IS NOT NULL
                THEN (i.metrics->>'steps')::float * (i.metrics->>'masps')::float
                WHEN i.metrics->>'it' IS NOT NULL
                    AND i.metrics->>'ot' IS NOT NULL
                    AND (i.metrics->>'it')::float > 0
                    AND (i.metrics->>'ot')::float > 0
                    AND i.metrics->>'maspt' IS NOT NULL
                THEN ((i.metrics->>'it')::float + (i.metrics->>'ot')::float) * (i.metrics->>'maspt')::float
                ELSE EXTRACT(EPOCH FROM (i.completed_at - i.started_at))
            END
        ) > 0
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
                {
                    "miner_hotkey": row[0],
                    "chute_id": row[1],
                    "invocation_count": int(row[2]),
                    "compute_units": float(row[3]),
                }
                for row in compute_result.fetchall()
            ]
        else:
            compute_data = [
                {
                    "miner_hotkey": row[0],
                    "invocation_count": int(row[1]),
                    "compute_units": float(row[2]),
                }
                for row in compute_result.fetchall()
            ]
        results[label] = {
            "bounties": bounty_data,
            "compute_units": compute_data,
        }

    await memcache_set(cache_key, json.dumps(results))
    return _filter_by_key(results)


@router.get("/scores")
async def get_scores(hotkey: Optional[str] = None, request: Request = None):
    rv = None
    if request:
        cached = await memcache_get(b"miner_scores")
        if cached:
            rv = json.loads(cached)
        else:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Waiting for cache to populate.",
            )
    if not rv:
        rv = await get_scoring_data()
        await memcache_set(b"miner_scores", json.dumps(rv))
    if hotkey:
        for key in rv:
            if key != "totals":
                rv[key] = {hotkey: rv[key].get(hotkey)}
    return rv


@router.get("/unique_chute_history/{hotkey}")
async def unique_chute_history(hotkey: str, request: Request = None):
    if not await memcache_get(f"miner_exists:{hotkey}".encode()):
        async with get_session(readonly=True) as session:
            if not await get_miner_by_hotkey(hotkey, session):
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"Miner {hotkey} not found in metagraph.",
                )
        await memcache_set(f"miner_exists:{hotkey}".encode(), b"1", exptime=7200)

    cache_key = f"uqhist:{hotkey}".encode()
    cached = await memcache_get(cache_key)
    if cached:
        return json.loads(cached)
    raise HTTPException(
        status_code=status.HTTP_404_NOT_FOUND,
        detail=f"Miner {hotkey} not found in unique history cache (yet)",
    )


@router.get("/metagraph")
async def get_metagraph():
    async with get_session(readonly=True) as session:
        return (
            (
                await session.execute(
                    select(MetagraphNode).where(
                        MetagraphNode.netuid == settings.netuid, MetagraphNode.node_id >= 0
                    )
                )
            )
            .unique()
            .scalars()
            .all()
        )


@router.get("/utilization")
async def get_utilization(hotkey: Optional[str] = None, request: Request = None):
    cache_key = "utilscore".encode()
    result = None
    if request:
        cached = await memcache_get(cache_key)
        if cached:
            result = json.loads(cached)
    if not result:
        async with get_session(readonly=True) as session:
            result = {
                hotkey: float(utilization)
                for hotkey, utilization in (
                    await session.execute(text(UTILIZATION_RATIO_QUERY.format(interval="8 hours")))
                )
                .unique()
                .all()
            }
            await memcache_set(cache_key, json.dumps(result))
    if hotkey:
        return {hotkey: result.get(hotkey)}
    return result


@router.get("/utilization_instances")
async def get_utilization_instances(hotkey: Optional[str] = None, request: Request = None):
    query = """WITH instance_spans AS (
  SELECT
    miner_hotkey, instance_id, chute_id,
    MAX(completed_at) - MIN(started_at) as total_active_time,
    SUM(completed_at - started_at) AS total_processing_time
  FROM invocations
  WHERE started_at >= now() - INTERVAL '8 hours'
  AND error_message IS NULL AND completed_at IS NOT NULL
  GROUP BY miner_hotkey, instance_id, chute_id
),
instance_metrics AS (
  SELECT
    miner_hotkey, instance_id, chute_id,
    EXTRACT(EPOCH FROM total_active_time) AS total_active_seconds,
    EXTRACT(EPOCH FROM total_processing_time) AS total_processing_seconds,
    CASE
      WHEN EXTRACT(EPOCH FROM total_active_time) > 0
      THEN ROUND(
        (EXTRACT(EPOCH FROM total_processing_time) /
         EXTRACT(EPOCH FROM total_active_time))::numeric,
        2
      )
      ELSE 0
    END AS busy_ratio
  FROM instance_spans
  JOIN metagraph_nodes mn ON instance_spans.miner_hotkey = mn.hotkey
),
ranked_instances AS (
  SELECT
    miner_hotkey, instance_id, chute_id,
    total_active_seconds, total_processing_seconds, busy_ratio,
    ROW_NUMBER() OVER (PARTITION BY miner_hotkey ORDER BY busy_ratio DESC) AS rank
  FROM instance_metrics WHERE total_active_seconds >= 3600
),
top_instances AS (
  SELECT
    miner_hotkey, instance_id, chute_id,
    total_active_seconds, total_processing_seconds, busy_ratio
  FROM ranked_instances
  WHERE rank <= 6
)
SELECT * FROM top_instances ORDER BY miner_hotkey ASC;
"""
    cache_key = "utilinst".encode()
    result = None
    if request:
        cached = await memcache_get(cache_key)
        if cached:
            result = json.loads(cached)
    if not result:
        async with get_session(readonly=True) as session:
            result = {}
            raw_results = await session.execute(text(query))
            for hotkey, instance_id, chute_id, _, __, busy_ratio in raw_results:
                if hotkey not in result:
                    result[hotkey] = []
                result[hotkey].append(
                    {"instance_id": instance_id, "ratio": float(busy_ratio), "chute_id": chute_id}
                )
                await memcache_set(cache_key, json.dumps(result))
    if hotkey:
        return {hotkey: result.get(hotkey)}
    return result
