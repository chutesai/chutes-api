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
from api.chute.util import calculate_effective_compute_multiplier
from api.bounty.util import get_bounty_infos
from api.node.schemas import Node
from api.image.schemas import Image
from api.instance.schemas import Instance
from api.job.schemas import Job
from api.invocation.util import gather_metrics
from api.user.service import get_current_user
from api.database import get_session, get_db_session
from api.config import settings
from api.constants import HOTKEY_HEADER, THRASH_WINDOW_HOURS
from api.metasync import get_miner_by_hotkey, MetagraphNode
from api.util import semcomp
from metasync.shared import get_scoring_data

router = APIRouter()


async def model_to_dict(obj, bounty_info: Optional[dict] = None):
    """
    Helper to convert object to dict.
    """
    mapper = class_mapper(obj.__class__)
    data = {
        column.key: getattr(obj, column.key)
        for column in mapper.columns
        if column.key != "env_creation"
    }
    for name, value in vars(obj.__class__).items():
        if isinstance(getattr(value, "decorator_info", None), ComputedFieldInfo):
            data[name] = getattr(obj, name)
    if isinstance(obj, Chute):
        data["image"] = f"{obj.image.user.username}/{obj.image.name}:{obj.image.tag}".lower()
        if obj.image.patch_version not in (None, "initial"):
            data["image"] += f"-{obj.image.patch_version}"
        ns = NodeSelector(**obj.node_selector)
        data["node_selector"].update(
            {
                "compute_multiplier": ns.compute_multiplier,
                "supported_gpus": ns.supported_gpus,
            }
        )
        if semcomp(obj.chutes_version or "0.0.0", "0.3.61") >= 0:
            data["code"] = "print('legacy placeholder')"
        data["preemptible"] = obj.preemptible

        # Add effective compute multiplier and factors.
        effective_data = await calculate_effective_compute_multiplier(obj, bounty_info=bounty_info)
        data["effective_compute_multiplier"] = effective_data["effective_compute_multiplier"]
        data["compute_multiplier_factors"] = effective_data["compute_multiplier_factors"]
        data["bounty"] = effective_data["bounty"]

    if isinstance(obj, Image):
        data["username"] = obj.user.username.lower()
        data["name"] = obj.name.lower()
        data["tag"] = obj.tag.lower()
    if isinstance(data.get("seed"), Decimal):
        data["seed"] = int(data["seed"])
    data.pop("symmetric_key", None)
    data.pop("host", None)
    data.pop("inspecto", None)
    data.pop("port_mappings", None)
    data.pop("port", None)
    data.pop("env_creation", None)
    data.pop("package_hashes", None)
    for key in list(data.keys()):
        if key.startswith("rint_"):
            del data[key]
    return data


async def _stream_items(clazz: Any, selector: Any = None, explicit_null: bool = False):
    """
    Streaming results helper.
    """
    async with get_session() as db:
        query = selector if selector is not None else select(clazz)
        if clazz is Chute:
            result = await db.execute(query)
            items = result.unique().scalars().all()
            any_found = False
            if items:
                bounty_infos = await get_bounty_infos([item.chute_id for item in items])
                for item in items:
                    data = await model_to_dict(item, bounty_info=bounty_infos.get(item.chute_id))
                    yield f"data: {json.dumps(data).decode()}\n\n"
                    any_found = True
            if explicit_null and not any_found:
                yield "data: NO_ITEMS\n"
            return

        result = await db.stream(query)
        any_found = False
        async for row in result.unique():
            data = await model_to_dict(row[0])
            yield f"data: {json.dumps(data).decode()}\n\n"
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
    _: User = Depends(get_current_user(purpose="miner", registered_to=settings.netuid)),
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
):
    async def _stream():
        async for metric in gather_metrics():
            yield f"data: {json.dumps(metric).decode()}\n\n"

    return StreamingResponse(_stream())


@router.get("/active_instances/")
async def list_active_instances(
    _: User = Depends(get_current_user(purpose="miner", registered_to=settings.netuid)),
    session: AsyncSession = Depends(get_db_session),
):
    """
    Get all active instances across the platform.
    Used by miners to make informed preemption decisions based on global state.
    """
    query = text("""
        SELECT
            instance_id,
            miner_hotkey,
            chute_id,
            activated_at,
            COALESCE(compute_multiplier, 1.0) as compute_multiplier
        FROM instances
        WHERE active = true
        AND verified = true
    """)
    result = await session.execute(query)
    return [
        {
            "instance_id": row[0],
            "miner_hotkey": row[1],
            "chute_id": row[2],
            "activated_at": row[3].isoformat() if row[3] else None,
            "compute_multiplier": float(row[4]),
        }
        for row in result.fetchall()
    ]


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
        return await model_to_dict(chute)


@router.get("/stats")
async def get_stats(
    miner_hotkey: Optional[str] = None,
    session: AsyncSession = Depends(get_db_session),
    per_chute: Optional[bool] = False,
    request: Request = None,
) -> Response:
    """
    Get miner stats over different intervals based on instance data (matching actual scoring).

    Returns instance-based metrics (total_instances, compute_seconds, compute_units, bounty_count)
    which align with how miners are actually scored for validator weights.
    """
    cache_key = f"get_stats:{per_chute}"

    def _filter_by_key(mstats):
        if miner_hotkey:
            for _, data in mstats.items():
                for key in data:
                    if isinstance(data[key], list):
                        data[key] = [v for v in data[key] if v.get("miner_hotkey") == miner_hotkey]
        return mstats

    if request:
        cached = await settings.redis_client.get(cache_key)
        if cached:
            return _filter_by_key(json.loads(cached))

    if miner_hotkey and not re.match(r"^[a-zA-Z0-9]{48}$", miner_hotkey):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid hotkey parameter."
        )

    # Simple instance-based stats query - matches the scoring mechanism structure
    # but without the complex bounty decay formula (just counts for stats purposes).
    # Uses instance_compute_history for accurate time-weighted multipliers.
    # Startup bonus is included in history table (0.3x rate from created_at to activated_at).
    instance_stats_query = """
    WITH billed_instances AS (
        SELECT
            ia.instance_id,
            ia.miner_hotkey,
            ia.chute_id,
            ia.activated_at,
            ia.compute_multiplier,
            ia.bounty,
            GREATEST(ia.created_at, now() - interval '{interval}') as billing_start,
            LEAST(
                COALESCE(ia.stop_billing_at, now()),
                COALESCE(ia.deleted_at, now()),
                now()
            ) as billing_end
        FROM instance_audit ia
        WHERE ia.activated_at IS NOT NULL
          AND (
              (ia.billed_to IS NULL AND ia.deleted_at IS NOT NULL AND ia.deleted_at - ia.activated_at >= INTERVAL '1 hour')
              OR ia.valid_termination IS TRUE
              OR ia.deletion_reason in (
                  'job has been terminated due to insufficient user balance',
                  'user-defined/private chute instance has not been used since shutdown_after_seconds',
                  'user has zero/negative balance (private chute)'
              )
              OR ia.deletion_reason LIKE '%has an old version%'
              OR ia.deleted_at IS NULL
          )
          AND (ia.deleted_at IS NULL OR ia.deleted_at >= now() - interval '{interval}')
    ),
    instance_weighted AS (
        SELECT
            bi.instance_id,
            bi.miner_hotkey,
            bi.chute_id,
            bi.billing_start,
            bi.billing_end,
            bi.bounty,
            bi.compute_multiplier as fallback_multiplier,
            COALESCE(
                SUM(
                    EXTRACT(EPOCH FROM (
                        LEAST(COALESCE(ich.ended_at, now()), bi.billing_end)
                        - GREATEST(ich.started_at, bi.billing_start)
                    )) * ich.compute_multiplier
                ),
                EXTRACT(EPOCH FROM (bi.billing_end - bi.billing_start)) * COALESCE(bi.compute_multiplier, 1.0)
            ) AS weighted_compute_units
        FROM billed_instances bi
        LEFT JOIN instance_compute_history ich
               ON ich.instance_id = bi.instance_id
              AND ich.started_at < bi.billing_end
              AND (ich.ended_at IS NULL OR ich.ended_at > bi.billing_start)
        WHERE bi.billing_end > bi.billing_start
        GROUP BY bi.instance_id, bi.miner_hotkey, bi.chute_id,
                 bi.billing_start, bi.billing_end, bi.bounty, bi.compute_multiplier
    )
    SELECT
        iw.miner_hotkey,
        COUNT(*) AS total_instances,
        COUNT(CASE WHEN iw.bounty IS TRUE THEN 1 END) AS bounty_count,
        SUM(EXTRACT(EPOCH FROM (iw.billing_end - iw.billing_start))) AS compute_seconds,
        SUM(iw.weighted_compute_units) AS compute_units
    FROM instance_weighted iw
    JOIN metagraph_nodes mn ON iw.miner_hotkey = mn.hotkey AND mn.netuid = 64 AND mn.node_id >= 0
    GROUP BY iw.miner_hotkey
    HAVING SUM(iw.weighted_compute_units) > 0
    ORDER BY compute_units DESC
    """

    # Per-chute instance stats query
    per_chute_stats_query = """
    WITH billed_instances AS (
        SELECT
            ia.instance_id,
            ia.miner_hotkey,
            ia.chute_id,
            ia.activated_at,
            ia.compute_multiplier,
            ia.bounty,
            GREATEST(ia.created_at, now() - interval '{interval}') as billing_start,
            LEAST(
                COALESCE(ia.stop_billing_at, now()),
                COALESCE(ia.deleted_at, now()),
                now()
            ) as billing_end
        FROM instance_audit ia
        WHERE ia.activated_at IS NOT NULL
          AND (
              (ia.billed_to IS NULL AND ia.deleted_at IS NOT NULL AND ia.deleted_at - ia.activated_at >= INTERVAL '1 hour')
              OR ia.valid_termination IS TRUE
              OR ia.deletion_reason in (
                  'job has been terminated due to insufficient user balance',
                  'user-defined/private chute instance has not been used since shutdown_after_seconds',
                  'user has zero/negative balance (private chute)'
              )
              OR ia.deletion_reason LIKE '%has an old version%'
              OR ia.deleted_at IS NULL
          )
          AND (ia.deleted_at IS NULL OR ia.deleted_at >= now() - interval '{interval}')
    ),
    instance_weighted AS (
        SELECT
            bi.instance_id,
            bi.miner_hotkey,
            bi.chute_id,
            bi.billing_start,
            bi.billing_end,
            bi.bounty,
            bi.compute_multiplier as fallback_multiplier,
            COALESCE(
                SUM(
                    EXTRACT(EPOCH FROM (
                        LEAST(COALESCE(ich.ended_at, now()), bi.billing_end)
                        - GREATEST(ich.started_at, bi.billing_start)
                    )) * ich.compute_multiplier
                ),
                EXTRACT(EPOCH FROM (bi.billing_end - bi.billing_start)) * COALESCE(bi.compute_multiplier, 1.0)
            ) AS weighted_compute_units
        FROM billed_instances bi
        LEFT JOIN instance_compute_history ich
               ON ich.instance_id = bi.instance_id
              AND ich.started_at < bi.billing_end
              AND (ich.ended_at IS NULL OR ich.ended_at > bi.billing_start)
        WHERE bi.billing_end > bi.billing_start
        GROUP BY bi.instance_id, bi.miner_hotkey, bi.chute_id,
                 bi.billing_start, bi.billing_end, bi.bounty, bi.compute_multiplier
    )
    SELECT
        iw.miner_hotkey,
        iw.chute_id,
        COUNT(*) AS total_instances,
        COUNT(CASE WHEN iw.bounty IS TRUE THEN 1 END) AS bounty_count,
        SUM(EXTRACT(EPOCH FROM (iw.billing_end - iw.billing_start))) AS compute_seconds,
        SUM(iw.weighted_compute_units) AS compute_units
    FROM instance_weighted iw
    JOIN metagraph_nodes mn ON iw.miner_hotkey = mn.hotkey AND mn.netuid = 64 AND mn.node_id >= 0
    GROUP BY iw.miner_hotkey, iw.chute_id
    HAVING SUM(iw.weighted_compute_units) > 0
    ORDER BY compute_units DESC
    """

    results = {}
    for interval, label in (("1 hour", "past_hour"), ("1 day", "past_day"), ("7 days", "all")):
        if per_chute:
            result = await session.execute(text(per_chute_stats_query.format(interval=interval)))
            stats_data = [
                {
                    "miner_hotkey": row[0],
                    "chute_id": row[1],
                    "total_instances": int(row[2]),
                    "bounty_count": int(row[3]),
                    "compute_seconds": float(row[4]),
                    "compute_units": float(row[5]),
                    # Legacy field for backwards compatibility
                    "invocation_count": int(row[2]),
                }
                for row in result.fetchall()
            ]
        else:
            result = await session.execute(text(instance_stats_query.format(interval=interval)))
            stats_data = [
                {
                    "miner_hotkey": row[0],
                    "total_instances": int(row[1]),
                    "bounty_count": int(row[2]),
                    "compute_seconds": float(row[3]),
                    "compute_units": float(row[4]),
                    # Legacy fields for backwards compatibility
                    "invocation_count": int(row[1]),
                    "total_bounty": float(row[2]),
                }
                for row in result.fetchall()
            ]

        # Keep backwards-compatible structure
        results[label] = {
            "instance_stats": stats_data,
            # Legacy keys for backwards compatibility
            "bounties": [
                {"miner_hotkey": s["miner_hotkey"], "total_bounty": float(s["bounty_count"])}
                for s in stats_data
            ]
            if not per_chute
            else [],
            "compute_units": stats_data,
        }

    await settings.redis_client.set(cache_key, json.dumps(results))
    return _filter_by_key(results)


@router.get("/scores")
async def get_scores(hotkey: Optional[str] = None, request: Request = None):
    cache_key = "get_scores"
    rv = None
    if request:
        cached = await settings.redis_client.get(cache_key)
        if cached:
            rv = json.loads(cached)
        else:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Waiting for cache to populate.",
            )
    if not rv:
        rv = await get_scoring_data()
        await settings.redis_client.set(cache_key, json.dumps(rv))
    if hotkey:
        for key in rv:
            if key != "totals":
                rv[key] = {hotkey: rv[key].get(hotkey)}
    return rv


@router.get("/unique_chute_history/{hotkey}")
async def unique_chute_history(hotkey: str, request: Request = None):
    if not await settings.redis_client.get(f"miner_exists:{hotkey}"):
        async with get_session(readonly=True) as session:
            if not await get_miner_by_hotkey(hotkey, session):
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"Miner {hotkey} not found in metagraph.",
                )
        await settings.redis_client.set(f"miner_exists:{hotkey}", "1", ex=7200)

    cache_key = f"uqhist:{hotkey}"
    cached = await settings.redis_client.get(cache_key)
    if cached:
        return json.loads(cached)
    raise HTTPException(
        status_code=status.HTTP_404_NOT_FOUND,
        detail=f"Miner {hotkey} not found in unique history cache (yet)",
    )


@router.get("/thrash_cooldowns")
async def get_thrash_cooldowns(
    hotkey: str | None = Header(None, alias=HOTKEY_HEADER),
    session: AsyncSession = Depends(get_db_session),
    _: User = Depends(get_current_user(purpose="miner", registered_to=settings.netuid)),
):
    """
    Return all chutes where this miner is currently in a thrash cooldown,
    along with when the cooldown expires.
    """
    result = await session.execute(
        text(f"""
            SELECT
                ia.chute_id,
                c.name AS chute_name,
                ia.deleted_at,
                ia.deleted_at + INTERVAL '{THRASH_WINDOW_HOURS} hours' AS cooldown_expires_at
            FROM instance_audit ia
            JOIN chutes c ON c.chute_id = ia.chute_id
            WHERE ia.miner_hotkey = :hotkey
              AND ia.activated_at IS NOT NULL
              AND ia.deleted_at IS NOT NULL
              AND ia.deleted_at > NOW() - INTERVAL '{THRASH_WINDOW_HOURS} hours'
              AND ia.valid_termination IS NOT TRUE
            ORDER BY ia.deleted_at DESC
        """),
        {"hotkey": hotkey},
    )
    return [
        {
            "chute_id": row.chute_id,
            "chute_name": row.chute_name,
            "deleted_at": row.deleted_at.isoformat() if row.deleted_at else None,
            "cooldown_expires_at": row.cooldown_expires_at.isoformat()
            if row.cooldown_expires_at
            else None,
        }
        for row in result.fetchall()
    ]


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
