"""
Routes for chutes.
"""

import re
import random
import string
import uuid
import time
import orjson as json
import aiohttp
from loguru import logger
from slugify import slugify
from fastapi import APIRouter, Depends, HTTPException, status, Response, Request
from fastapi.responses import HTMLResponse, JSONResponse
from starlette.responses import StreamingResponse
from sqlalchemy import or_, exists, func, text
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from sqlalchemy.orm import selectinload
from sqlalchemy.dialects.postgresql import insert
from typing import Optional
from api.chute.schemas import (
    Chute,
    ChuteArgs,
    ChuteShare,
    ChuteShareArgs,
    NodeSelector,
    ChuteUpdateArgs,
    RollingUpdate,
)
from api.chute.codecheck import is_bad_code
from api.chute.templates import (
    VLLMChuteArgs,
    VLLMEngineArgs,
    DiffusionChuteArgs,
    TEIChuteArgs,
    build_vllm_code,
    build_diffusion_code,
    build_tei_code,
)
from api.gpu import SUPPORTED_GPUS
from api.chute.response import ChuteResponse
from api.chute.util import get_chute_by_id_or_name, selector_hourly_price, get_one, is_shared
from api.instance.schemas import Instance
from api.instance.util import get_chute_target_manager
from api.user.schemas import User
from api.user.service import get_current_user, chutes_user_id
from api.image.schemas import Image
from api.image.util import get_image_by_id_or_name
from api.permissions import Permissioning

# XXX from api.instance.util import discover_chute_targets
from api.database import get_db_session, get_session
from api.pagination import PaginatedResponse
from api.fmv.fetcher import get_fetcher
from api.config import settings
from api.constants import (
    LLM_MIN_PRICE_IN,
    LLM_MIN_PRICE_OUT,
    LLM_PRICE_MULT_PER_MILLION_IN,
    LLM_PRICE_MULT_PER_MILLION_OUT,
    DIFFUSION_PRICE_MULT_PER_STEP,
)
from api.util import (
    semcomp,
    ensure_is_developer,
    limit_deployments,
    get_current_hf_commit,
    is_affine_registered,
    notify_deleted,
)
from api.affine import check_affine_code
from api.util import memcache_get, memcache_set
from api.guesser import guesser
from api.graval_worker import handle_rolling_update

router = APIRouter()


async def _inject_current_estimated_price(chute: Chute, response: ChuteResponse):
    """
    Inject the current estimated price data into a response.
    """
    if chute.standard_template == "vllm":
        hourly = await selector_hourly_price(chute.node_selector)
        if chute.concurrency and chute.concurrency < 16:
            hourly *= 16 / chute.concurrency
        if chute.discount:
            hourly -= hourly * chute.discount
        per_million_in = max(hourly * LLM_PRICE_MULT_PER_MILLION_IN, LLM_MIN_PRICE_IN)
        per_million_out = max(hourly * LLM_PRICE_MULT_PER_MILLION_OUT, LLM_MIN_PRICE_OUT)
        response.current_estimated_price = {
            "per_million_tokens": {
                "input": {"usd": round(per_million_in, 2)},
                "output": {"usd": round(per_million_out, 2)},
            }
        }
        tao_usd = await get_fetcher().get_price("tao")
        if tao_usd:
            response.current_estimated_price["per_million_tokens"]["input"]["tao"] = (
                per_million_in / tao_usd
            )
            response.current_estimated_price["per_million_tokens"]["output"]["tao"] = (
                per_million_out / tao_usd
            )
    elif chute.standard_template == "diffusion":
        hourly = await selector_hourly_price(chute.node_selector)
        per_step = hourly * DIFFUSION_PRICE_MULT_PER_STEP
        if chute.discount:
            per_step -= per_step * chute.discount
        response.current_estimated_price = {"per_step": {"usd": per_step}}
        tao_usd = await get_fetcher().get_price("tao")
        if tao_usd:
            response.current_estimated_price["per_step"]["tao"] = per_step / tao_usd

    # Legacy/fallback, and discounts.
    if not response.current_estimated_price:
        response.current_estimated_price = {}
    node_selector = NodeSelector(**chute.node_selector)
    response.current_estimated_price.update(await node_selector.current_estimated_price())
    if chute.discount and response.current_estimated_price:
        for key in ("usd", "tao"):
            values = response.current_estimated_price.get(key)
            if values:
                for unit in values:
                    values[unit] -= values[unit] * chute.discount

    # Fix node selector return value.
    response.node_selector.update(
        {
            "compute_multiplier": node_selector.compute_multiplier,
            "supported_gpus": node_selector.supported_gpus,
        }
    )


@router.post("/share")
async def share_chute(
    args: ChuteShareArgs,
    db: AsyncSession = Depends(get_db_session),
    current_user: User = Depends(get_current_user()),
):
    """
    Share a chute with another user.
    """
    chute = (
        (
            await db.execute(
                select(Chute).where(
                    or_(
                        Chute.name.ilike(args.chute_id_or_name),
                        Chute.chute_id == args.chute_id_or_name,
                    ),
                    Chute.user_id == current_user.user_id,
                )
            )
        )
        .unique()
        .scalar_one_or_none()
    )
    if not chute:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Did not find target chute {str(args.chute_id_or_name)}, or it does not belong to you",
        )
    if chute.public:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Chute is public, no need to share.",
        )
    user = (
        (
            await db.execute(
                select(User).where(
                    or_(
                        User.username.ilike(args.user_id_or_name),
                        User.user_id == args.user_id_or_name,
                    )
                )
            )
        )
        .unique()
        .scalar_one_or_none()
    )
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Could not find target user to share with",
        )
    if user.user_id == current_user.user_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Cannot share a chute with yourself",
        )
    stmt = insert(ChuteShare).values(
        [
            {
                "chute_id": chute.chute_id,
                "shared_by": current_user.user_id,
                "shared_to": user.user_id,
                "shared_at": func.now(),
            }
        ]
    )
    stmt = stmt.on_conflict_do_nothing()
    await db.execute(stmt)
    await db.commit()
    return {"status": f"Successfully shared {chute.name=} with {user.username=}"}


@router.post("/unshare")
async def unshare_chute(
    args: ChuteShareArgs,
    db: AsyncSession = Depends(get_db_session),
    current_user: User = Depends(get_current_user()),
):
    """
    Unshare a chute with another user.
    """
    chute = (
        (
            await db.execute(
                select(Chute).where(
                    or_(
                        Chute.name.ilike(args.chute_id_or_name),
                        Chute.chute_id == args.chute_id_or_name,
                    ),
                    Chute.user_id == current_user.user_id,
                )
            )
        )
        .unique()
        .scalar_one_or_none()
    )
    if not chute:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Did not find target chute {str(args.chute_id_or_name)}, or it does not belong to you",
        )
    user = (
        (
            await db.execute(
                select(User).where(
                    or_(
                        User.username.ilike(args.user_id_or_name),
                        User.user_id == args.user_id_or_name,
                    )
                )
            )
        )
        .unique()
        .scalar_one_or_none()
    )
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Could not find target user to share with",
        )

    await db.execute(
        text(
            "delete from chute_shares where shared_by = :cuser_id and shared_to = :user_id and chute_id = :chute_id"
        ),
        {"cuser_id": current_user.user_id, "user_id": user.user_id, "chute_id": chute.chute_id},
    )
    await db.commit()
    return {
        "status": f"Successfully unshared {chute.name=} with {user.username=} (if share exists)"
    }


@router.get("/affine_available")
async def list_available_affine_chutes():
    """
    Get a list of affine chutes where the creator/user has a non-zero balance.
    """
    async with get_session() as session:
        query = text("""
            SELECT
                c.chute_id,
                c.user_id,
                c.name,
                n.hotkey,
                CASE WHEN COUNT(i.chute_id) > 0 THEN true ELSE false END as has_active_instance
            FROM chutes c
            JOIN users u ON c.user_id = u.user_id
            JOIN user_current_balance b ON b.user_id = u.user_id
            LEFT JOIN metagraph_nodes n ON n.hotkey = u.hotkey
            LEFT JOIN instances i ON i.chute_id = c.chute_id AND i.active = true
            WHERE
                c.name ILIKE '%affine%'
                AND b.effective_balance > 0
                AND n.netuid = 120
            GROUP BY c.chute_id, c.user_id, c.name, n.hotkey;
        """)
        result = await session.execute(query)
        rows = result.fetchall()
        return [
            {
                "chute_id": row.chute_id,
                "user_id": row.user_id,
                "name": row.name,
                "hotkey": row.hotkey,
                "has_active_instance": row.has_active_instance,
            }
            for row in rows
        ]


@router.get("/", response_model=PaginatedResponse)
async def list_chutes(
    include_public: Optional[bool] = False,
    template: Optional[str] = None,
    name: Optional[str] = None,
    exclude: Optional[str] = None,
    image: Optional[str] = None,
    slug: Optional[str] = None,
    page: Optional[int] = 0,
    limit: Optional[int] = 25,
    include_schemas: Optional[bool] = False,
    db: AsyncSession = Depends(get_db_session),
    current_user: User = Depends(get_current_user(purpose="chutes", raise_not_found=False)),
):
    """
    List (and optionally filter/paginate) chutes.
    """
    cache_key = str(
        uuid.uuid5(
            uuid.NAMESPACE_OID,
            ":".join(
                [
                    "chutes_list",
                    f"template:{template}",
                    f"image:{image}",
                    f"slug:{slug}",
                    f"page:{page}",
                    f"limit:{limit}",
                    f"name:{name}",
                    f"exclude:{exclude}",
                    f"include_public:{include_public}",
                    f"include_schemas:{include_schemas}",
                    f"user:{current_user.user_id if current_user else None}",
                ]
            ),
        )
    ).encode()
    cached = await memcache_get(cache_key)
    if cached:
        return json.loads(cached)
    query = select(Chute).options(selectinload(Chute.instances))

    # Filter by public and/or only the user's chutes.
    if current_user:
        if include_public:
            query = query.where(
                or_(
                    Chute.public.is_(True),
                    Chute.user_id == current_user.user_id,
                    Chute.name.ilike("%/affine%"),
                )
            )
        else:
            query = query.where(Chute.user_id == current_user.user_id)
    else:
        query = query.where(or_(Chute.public.is_(True), Chute.name.ilike("%/affine%")))

    # Filter by name/tag/etc.
    if name and name.strip():
        query = query.where(Chute.name.ilike(f"%{name}%"))
    if exclude and exclude.strip():
        query = query.where(~Chute.name.ilike(f"%{exclude}%"))
    if image and image.strip():
        query = query.where(
            or_(
                Image.name.ilike("%{image}%"),
                Image.tag.ilike("%{image}%"),
            )
        )
    if slug and slug.strip():
        query = query.where(Chute.slug.ilike(slug))

    # Standard template filtering.
    if template and template.strip() and template != "other":
        query = query.where(Chute.standard_template == template)
    elif template == "other":
        query = query.where(Chute.standard_template.is_(None))

    # Perform a count.
    total_query = select(func.count()).select_from(query.subquery())
    total_result = await db.execute(total_query)
    total = total_result.scalar() or 0

    # Pagination.
    query = (
        query.order_by(Chute.invocation_count.desc())
        .offset((page or 0) * (limit or 25))
        .limit((limit or 25))
    )

    result = await db.execute(query)
    responses = []
    cord_refs = {}
    for item in result.unique().scalars().all():
        chute_response = ChuteResponse.from_orm(item)
        cord_defs = json.dumps(item.cords).decode()
        if item.standard_template == "vllm":
            cord_defs = cord_defs.replace(f'"default":"{item.name}"', '"default":""')
        cord_ref_id = str(uuid.uuid5(uuid.NAMESPACE_OID, cord_defs))
        if cord_ref_id not in cord_refs:
            cord_refs[cord_ref_id] = item.cords
            if not include_schemas:
                for cord in cord_refs[cord_ref_id] or []:
                    cord.pop("input_schema", None)
                    cord.pop("minimal_input_schema", None)
                    cord.pop("output_schema", None)
        chute_response.cords = None
        chute_response.cord_ref_id = cord_ref_id
        responses.append(chute_response)
        await _inject_current_estimated_price(item, responses[-1])
    result = {
        "total": total,
        "page": page,
        "limit": limit,
        "items": [item.model_dump() for item in responses],
        "cord_refs": cord_refs,
    }
    await memcache_set(cache_key, json.dumps(result), exptime=60)
    return result


@router.get("/rolling_updates")
async def list_rolling_updates():
    async with get_session() as session:
        result = await session.execute(text("SELECT * FROM rolling_updates"))
        columns = result.keys()
        rows = result.fetchall()
        return [dict(zip(columns, row)) for row in rows]


@router.get("/gpu_count_history")
async def get_gpu_count_history():
    query = """
        SELECT DISTINCT ON (chute_id)
            chute_id,
            (node_selector->>'gpu_count')::integer AS gpu_count
        FROM chute_history
        WHERE
            node_selector ? 'gpu_count'
            AND jsonb_typeof(node_selector->'gpu_count') = 'number'
        ORDER BY
            chute_id, created_at DESC
    """
    async with get_session(readonly=True) as session:
        results = (await session.execute(text(query))).unique().all()
        return [dict(zip(["chute_id", "gpu_count"], row)) for row in results]


@router.get("/miner_means")
async def get_chute_miner_mean_index(db: AsyncSession = Depends(get_db_session)):
    query = """
        SELECT c.chute_id, c.name
        FROM chutes c
        WHERE c.standard_template = 'vllm'
        ORDER BY invocation_count DESC
    """
    result = await db.execute(text(query))
    chutes = result.fetchall()
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Chute LLM outlier index</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 20px; }
            h1 { color: #333; }
            ul { list-style-type: none; padding: 0; }
            li { margin: 10px 0; }
            a { text-decoration: none; color: #0066cc; }
            a:hover { text-decoration: underline; }
        </style>
    </head>
    <body>
        <h1>Metrics</h1>
        <ul>
    """
    for chute in chutes:
        link = f"https://api.{settings.base_domain}/chutes/miner_means/{chute.chute_id}"
        html_content += f'        <li><a href="{link}">{chute.name}</a></li>\n'
    html_content += """
        </ul>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)


@router.get("/miner_means/{chute_id}")
@router.get("/miner_means/{chute_id}.{ext}")
async def get_chute_miner_means(
    chute_id: str,
    ext: Optional[str] = None,
    db: AsyncSession = Depends(get_db_session),
):
    """
    Load a chute's mean TPS and output token count by miner ID.
    """
    query = """
        SELECT
            miner_hotkey,
            instance_id,
            avg_tps,
            avg_output_tokens
        FROM llm_means
        WHERE chute_id = :chute_id
        ORDER BY avg_output_tokens DESC
    """
    result = await db.execute(text(query), {"chute_id": chute_id})
    rows = result.fetchall()

    # JSON response.
    if ext == "json":
        miner_means = [
            {
                "miner_hotkey": row.miner_hotkey,
                "instance_id": row.instance_id,
                "avg_tps": float(row.avg_tps),
                "avg_output_tokens": float(row.avg_output_tokens),
            }
            for row in rows
        ]
        return JSONResponse(content=miner_means)

    # CSV response.
    if ext == "csv":
        csv_content = "instance_id,miner_hotkey,avg_tps,avg_output_tokens\n"
        for row in rows:
            csv_content += (
                f"{row.instance_id},{row.miner_hotkey},{row.avg_tps},{row.avg_output_tokens}\n"
            )
        return Response(content=csv_content, media_type="text/csv")

    # Default return an ugly hacky HTML page to make it easier to read.
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Chute metrics</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 20px; }
            table { border-collapse: collapse; width: 100%; }
            th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
            th { background-color: #4CAF50; color: white; }
            tr:nth-child(even) { background-color: #f2f2f2; }
            tr:hover { background-color: #ddd; }
            .number { text-align: right; }
        </style>
    </head>
    <body>
        <h1>Metrics</h1>
        <table>
            <thead>
                <tr>
                    <th>Hotkey</th>
                    <th>Instance ID</th>
                    <th class="number">Avg TPS</th>
                    <th class="number">Avg Output Tokens</th>
                </tr>
            </thead>
            <tbody>
    """
    for row in rows:
        html_content += f"""
                <tr>
                    <td>{row.miner_hotkey}</td>
                    <td>{row.instance_id}</td>
                    <td class="number">{row.avg_tps:.2f}</td>
                    <td class="number">{row.avg_output_tokens:.2f}</td>
                </tr>
        """
    html_content += """
            </tbody>
        </table>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)


@router.get("/code/{chute_id}")
async def get_chute_code(
    chute_id: str,
    db: AsyncSession = Depends(get_db_session),
    current_user: User = Depends(get_current_user(purpose="chutes", raise_not_found=False)),
):
    """
    Load a chute's code by ID or name.
    """
    chute = await get_one(chute_id)
    if not chute:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Chute not found, or does not belong to you",
        )
    authorized = False
    if (
        chute.public
        or (
            current_user
            and (
                current_user.user_id == chute.user_id
                or await is_shared(chute_id, current_user.user_id)
            )
        )
        or "/affine" in chute.name.lower()
    ):
        authorized = True
    if not authorized:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Chute not found, or does not belong to you",
        )
    return Response(content=chute.code, media_type="text/plain")


@router.get("/warmup/{chute_id_or_name:path}")
async def warm_up_chute(
    chute_id_or_name: str,
    current_user: User = Depends(get_current_user(purpose="chutes", raise_not_found=False)),
):
    """
    Warm up a chute.
    """
    chute = await get_one(chute_id_or_name)
    if not chute:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Chute not found, or does not belong to you",
        )
    if (
        not chute.public
        and not chute.user_id == current_user.user_id
        and not await is_shared(chute.chute_id, current_user.user_id)
    ):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Chute not found, or does not belong to you",
        )
    balance = (
        current_user.current_balance.effective_balance if current_user.current_balance else 0.0
    )
    if balance <= 0:
        raise HTTPException(
            status_code=status.HTTP_402_PAYMENT_REQUIRED,
            detail=f"Account balance is ${balance}, please top-up with fiat or send tao to {current_user.payment_address}",
        )

    started_at = time.time()

    async def _respond():
        max_wait = 0
        while time.time() - started_at < 600:
            if (
                tm := await get_chute_target_manager(chute=chute, max_wait=max_wait, dynonce=True)
            ) is None:
                yield 'data: {"status": "cold", "log": "waiting for instances ..."}\n\n'
                if not max_wait:
                    max_wait = 5.0
                continue
            yield f'data: {{"status": "hot", "log": "chute is hot, {len(tm.instances)} instances available"}}\n\n'
            return

    return StreamingResponse(
        _respond(),
        media_type="text/event-stream",
    )


@router.get("/utilization")
async def get_chute_utilization(request: Request):
    """
    Get chute utilization data from the most recent capacity log.
    """
    async with get_session(readonly=True) as session:
        query = text("""
            WITH latest_logs AS (
                SELECT DISTINCT ON (chute_id)
                    chute_id,
                    timestamp,
                    utilization_current,
                    utilization_5m,
                    utilization_15m,
                    utilization_1h,
                    rate_limit_ratio_5m,
                    rate_limit_ratio_15m,
                    rate_limit_ratio_1h,
                    total_requests_5m,
                    total_requests_15m,
                    total_requests_1h,
                    completed_requests_5m,
                    completed_requests_15m,
                    completed_requests_1h,
                    rate_limited_requests_5m,
                    rate_limited_requests_15m,
                    rate_limited_requests_1h,
                    instance_count,
                    action_taken,
                    target_count
                FROM capacity_log
                ORDER BY chute_id, timestamp DESC
            ),
            chute_details AS (
                SELECT
                    c.chute_id,
                    CASE WHEN c.public IS true THEN c.name ELSE '[private chute]' END AS name,
                    EXISTS(SELECT 1 FROM rolling_updates WHERE chute_id = c.chute_id) AS update_in_progress,
                    COUNT(i.instance_id) AS total_instance_count,
                    COUNT(CASE WHEN i.active = true AND i.verified = true THEN 1 END) AS active_instance_count
                FROM chutes c
                LEFT JOIN instances i ON c.chute_id = i.chute_id
                GROUP BY c.chute_id, c.name, c.public
            )
            SELECT
                ll.*,
                cd.name,
                cd.update_in_progress,
                cd.total_instance_count,
                cd.active_instance_count
            FROM latest_logs ll
            JOIN chute_details cd ON cd.chute_id = ll.chute_id
            ORDER BY ll.total_requests_1h DESC
        """)
        results = await session.execute(query)
        rows = results.mappings().all()
        utilization_data = []
        for row in rows:
            item = dict(row)
            scale_value = await settings.redis_client.get(f"scale:{item['chute_id']}")
            target_count = int(scale_value) if scale_value else item.get("target_count", 0)
            current_count = item.get("total_instance_count", 0)
            item["scalable"] = current_count < target_count
            item["scale_allowance"] = max(0, target_count - current_count)
            item["avg_busy_ratio"] = item.get("utilization_1h", 0)
            item["total_invocations"] = item.get("total_requests_1h", 0)
            item["total_rate_limit_errors"] = item.get("rate_limited_requests_1h", 0)
            utilization_data.append(item)
        return utilization_data


@router.get("/{chute_id_or_name:path}", response_model=ChuteResponse)
async def get_chute(
    chute_id_or_name: str,
    db: AsyncSession = Depends(get_db_session),
    current_user: User = Depends(get_current_user(purpose="chutes", raise_not_found=False)),
):
    """
    Load a chute by ID or name.
    """
    chute = await get_chute_by_id_or_name(chute_id_or_name, db, current_user, load_instances=True)
    if not chute:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Chute not found, or does not belong to you",
        )
    response = ChuteResponse.from_orm(chute)
    await _inject_current_estimated_price(chute, response)
    return response


@router.delete("/{chute_id_or_name:path}")
async def delete_chute(
    chute_id_or_name: str,
    db: AsyncSession = Depends(get_db_session),
    current_user: User = Depends(get_current_user(purpose="chutes")),
):
    """
    Delete a chute by ID or name.
    """
    chute = None
    if (
        current_user.has_role(Permissioning.affine_admin)
        and current_user.user_id != await chutes_user_id()
    ):
        try:
            uuid.UUID(chute_id_or_name)
        except ValueError:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Must use chute UUID to delete affine models.",
            )
        chute = await get_one(chute_id_or_name)
        if not chute:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Chute not found",
            )
        if chute.user_id != current_user.user_id and "affine" not in chute.name.lower():
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Cannot delete non-affine chutes not owned by you.",
            )
        logger.warning(
            f"AFFINE DEV DELETION TRIGGERED: {current_user.user_id=} "
            f"{current_user.username=} {chute.chute_id=} {chute.name=}"
        )
    else:
        chute = await get_chute_by_id_or_name(chute_id_or_name, db, current_user)
        if not chute or chute.user_id != current_user.user_id:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Chute not found, or does not belong to you",
            )

    chute_id = chute.chute_id
    version = chute.version
    await db.delete(chute)
    await db.commit()
    await settings.redis_client.publish(
        "miner_broadcast",
        json.dumps(
            {
                "reason": "chute_deleted",
                "data": {"chute_id": chute_id, "version": version},
            }
        ).decode(),
    )

    return {"chute_id": chute_id, "deleted": True}


async def _deploy_chute(
    chute_args: ChuteArgs,
    db: AsyncSession,
    current_user: User,
    use_rolling_update: bool = True,
    accept_fee: bool = False,
):
    """
    Deploy a chute!
    """
    if chute_args.public and not current_user.has_role(Permissioning.public_model_deployment):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                "Chutes no longer supports public chutes. You can instead "
                "deploy the chute as a private chute and share it with other users."
            ),
        )
    image = await get_image_by_id_or_name(chute_args.image, db, current_user)
    if not image:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Chute image not found, or does not belong to you",
        )
    if chute_args.public and not image.public:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Chute cannot be public when image is not public!",
        )
    version = str(uuid.uuid5(uuid.NAMESPACE_OID, f"{image.image_id}:{chute_args.code}"))
    chute = (
        (
            await db.execute(
                select(Chute)
                .where(Chute.name.ilike(chute_args.name))
                .where(Chute.user_id == current_user.user_id)
                .options(selectinload(Chute.instances))
            )
        )
        .unique()
        .scalar_one_or_none()
    )
    if chute and chute.version == version and chute.public == chute_args.public:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Chute with name={chute_args.name}, {version=} and public={chute_args.public} already exists",
        )

    # Limit h200 and b200 usage.
    if not chute_args.node_selector:
        chute_args.node_selector = {"gpu_count": 1}
    if isinstance(chute_args.node_selector, dict):
        chute_args.node_selector = NodeSelector(**chute_args.node_selector)
    if set(chute_args.node_selector.supported_gpus) == set(["5090"]):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You are not allowed to require 5090s",
        )

    # Fee estimate, as an error, if the user hasn't used the confirmed param.
    estimate = await chute_args.node_selector.current_estimated_price()
    deployment_fee = (
        chute_args.node_selector.gpu_count * estimate["usd"]["hour"] * 3
        if not current_user.has_role(Permissioning.free_account) and not chute
        else 0
    )
    if deployment_fee and not accept_fee:
        estimate = chute_args.node_selector.current_estimated_price
        raise HTTPException(
            status_code=status.HTTP_402_PAYMENT_REQUIRED,
            detail=(
                "DEPLOYMENT FEE NOTICE:\n===\nThere is a deployment fee of (hourly price per GPU * number of GPUs * 3), "
                f"which for this configuration is for this configuration is: ${round(deployment_fee, 2)}\n "
                "To acknowledge this fee, ensure you have chutes>=0.3.23 and re-run the deployment command with `--accept-fee`"
            ),
        )
    if current_user.balance <= deployment_fee and not current_user.has_role(
        Permissioning.free_account
    ):
        logger.warning(
            f"Payment required: attempted deployment of chute {chute_args.name} "
            f"from user {current_user.username} with balance < {deployment_fee=}"
        )
        raise HTTPException(
            status_code=status.HTTP_402_PAYMENT_REQUIRED,
            detail=(
                f"The deployment fee, based on your node selector, is ${deployment_fee}, "
                f"but you have a balance of {current_user.balance}.\n"
                f"Please top up your account with tao @ {current_user.payment_address} or via fiat."
            ),
        )

    affine_dev = await is_affine_registered(db, current_user)
    if (
        current_user.user_id != await chutes_user_id()
        and not current_user.has_role(Permissioning.unlimited_dev)
        and not affine_dev
    ):
        if (
            chute_args.node_selector
            and chute_args.node_selector.min_vram_gb_per_gpu
            and chute_args.node_selector.min_vram_gb_per_gpu > 80
        ):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="You are not allowed to require > 80gb VRAM per GPU at this time.",
            )
        if not chute_args.node_selector.exclude:
            chute_args.node_selector.exclude = []
        chute_args.node_selector.exclude = list(
            set(chute_args.node_selector.exclude or [] + ["h200", "b200", "mi300x"])
        )

        if not chute_args.node_selector.supported_gpus:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="No supported GPUs based on node selector!",
            )

        # Limit h/b 200 access for now.
        if not set(chute_args.node_selector.supported_gpus) - set(["b200", "h200", "mi300x"]):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="You are not allowed to require h200, b200 or mi300x at this time.",
            )

    # Disable non-chutes official images for affine.
    if (
        affine_dev
        and (
            image.user_id != await chutes_user_id() or not image.name.startswith(("sglang", "vllm"))
        )
    ) and not current_user.has_role(Permissioning.unlimited_dev):
        logger.error(
            f"Affine user tried to deploy unofficial vllm/sglang image: {image.name=} {image.tag=} {current_user.username=}"
        )
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Must use either sglang or vllm official chutes image for affine deployments.",
        )

    # Require revision for LLM templates.
    if chute_args.standard_template == "vllm" and not chute_args.revision:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Missing required revision parameter for vllm template.",
        )

    # Prevent deploying images with old chutes SDK versions.
    if not image.chutes_version or semcomp(image.chutes_version, "0.3.28") < 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                "Unable to deploy chutes with legacy images (chutes SDK < 0.3.28), please upgrade "
                f"(or ask chutes team to upgrade) {image.name=} {image.image_id=} currently {image.chutes_version}"
            ),
        )

    # Prevent enabling logging on a chute that had it disabled previously.
    if chute and not chute.logging_enabled and chute_args.logging_enabled:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You cannot update a chute's logging_enabled flag to true after it has been false.",
        )

    old_version = None
    if chute:
        # Create a rolling update object so we can gracefully restart/recreate.
        permitted = {}
        for inst in chute.instances:
            if inst.miner_hotkey not in permitted:
                permitted[inst.miner_hotkey] = 0
            permitted[inst.miner_hotkey] += 1
        await db.execute(
            text(
                "DELETE FROM rolling_updates WHERE chute_id = :chute_id",
            ),
            {"chute_id": chute.chute_id},
        )
        if use_rolling_update and chute.instances:
            rolling_update = RollingUpdate(
                chute_id=chute.chute_id,
                old_version=chute.version,
                new_version=version,
                permitted=permitted,
            )
            db.add(rolling_update)

        old_version = chute.version
        chute.image_id = image.image_id
        chute.tagline = chute_args.tagline
        chute.readme = chute_args.readme
        chute.code = chute_args.code
        chute.node_selector = chute_args.node_selector
        chute.tool_description = chute_args.tool_description
        chute.filename = chute_args.filename
        chute.ref_str = chute_args.ref_str
        chute.version = version
        chute.public = (
            chute_args.public
            if current_user.has_role(Permissioning.public_model_deployment)
            else False
        )
        chute.logo_id = (
            chute_args.logo_id if chute_args.logo_id and chute_args.logo_id.strip() else None
        )
        chute.chutes_version = image.chutes_version
        chute.cords = chute_args.cords
        chute.jobs = chute_args.jobs
        chute.concurrency = chute_args.concurrency
        chute.updated_at = func.now()
        chute.revision = chute_args.revision
        chute.logging_enabled = chute_args.logging_enabled
        chute.max_instances = (
            None
            if chute.public or chute.user_id == await chutes_user_id()
            else (chute_args.max_instances or 1)
        )
        chute.shutdown_after_seconds = (
            None
            if chute.public or chute.user_id == await chutes_user_id()
            else (chute_args.shutdown_after_seconds or 300)
        )
        chute.scaling_threshold = (
            None
            if chute.public or chute.user_id == await chutes_user_id()
            else (chute_args.scaling_threshold or 0.75)
        )
    else:
        try:
            is_public = (
                chute_args.public
                if current_user.has_role(Permissioning.public_model_deployment)
                else False
            )
            chute = Chute(
                chute_id=str(
                    uuid.uuid5(
                        uuid.NAMESPACE_OID, f"{current_user.username}::chute::{chute_args.name}"
                    )
                ),
                image_id=image.image_id,
                user_id=current_user.user_id,
                name=chute_args.name,
                tagline=chute_args.tagline,
                readme=chute_args.readme,
                tool_description=chute_args.tool_description,
                logo_id=chute_args.logo_id if chute_args.logo_id else None,
                code=chute_args.code,
                filename=chute_args.filename,
                ref_str=chute_args.ref_str,
                version=version,
                public=is_public,
                cords=chute_args.cords,
                jobs=chute_args.jobs,
                node_selector=chute_args.node_selector,
                standard_template=chute_args.standard_template,
                chutes_version=image.chutes_version,
                concurrency=chute_args.concurrency,
                revision=chute_args.revision,
                logging_enabled=chute_args.logging_enabled,
                scaling_threshold=None
                if is_public or current_user.user_id == await chutes_user_id()
                else (chute_args.scaling_threshold or 0.75),
                max_instances=None
                if is_public or current_user.user_id == await chutes_user_id()
                else (chute_args.max_instances or 1),
                shutdown_after_seconds=None
                if is_public or current_user.user_id == await chutes_user_id()
                else (chute_args.shutdown_after_seconds or 300),
            )
        except ValueError as exc:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Validation failure: {exc}",
            )

        # Generate a unique slug (subdomain).
        chute.slug = re.sub(
            r"[^a-z0-9-]+$",
            "-",
            slugify(f"{current_user.username}-{chute.name}", max_length=58).lower(),
        )
        base_slug = chute.slug
        already_exists = (
            await db.execute(select(exists().where(Chute.slug == chute.slug)))
        ).scalar()
        while already_exists:
            suffix = "".join(
                random.choice(string.ascii_lowercase + string.digits) for _ in range(5)
            )
            chute.slug = f"{base_slug}-{suffix}"
            already_exists = (
                await db.execute(select(exists().where(Chute.slug == chute.slug)))
            ).scalar()

        db.add(chute)

    # Make sure we have at least one cord or one job definition.
    if not chute.cords and not chute.jobs:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="A chute must define at least one cord() or job() function!",
        )
    elif chute.cords and chute.jobs:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="A chute can have jobs or cords not both.",
        )

    await db.commit()
    await db.refresh(chute)

    # Update in usage tracker, only after successful DB commit.
    if deployment_fee and not current_user.has_role(Permissioning.free_account):
        logger.info(
            f"DEPLOYMENTFEE: {deployment_fee} for {current_user.username=} with "
            f"{chute_args.node_selector=} of {chute_args.name=}, new balance={current_user.balance - deployment_fee}"
        )
        pipeline = settings.redis_client.pipeline()
        chute_id = str(
            uuid.uuid5(uuid.NAMESPACE_OID, f"{current_user.username}::chute::{chute_args.name}")
        )
        key = f"balance:{current_user.user_id}:{chute_id}"
        pipeline.hincrbyfloat(key, "amount", deployment_fee)
        pipeline.hset(key, "timestamp", int(time.time()))
        await pipeline.execute()

    if old_version:
        if use_rolling_update:
            await handle_rolling_update.kiq(chute.chute_id, chute.version)
            await settings.redis_client.publish(
                "miner_broadcast",
                json.dumps(
                    {
                        "reason": "chute_updated",
                        "data": {
                            "chute_id": chute.chute_id,
                            "version": chute.version,
                            "job_only": not chute.cords,
                        },
                    }
                ).decode(),
            )
        else:
            logger.warning(
                f"Chute deployed with rolling update disabled: {chute.chute_id=} {chute.name=}"
            )
            # Purge all instances immediately.
            instances = (
                (await db.execute(select(Instance).where(Instance.chute_id == chute.chute_id)))
                .unique()
                .scalars()
                .all()
            )
            for instance in instances:
                await db.delete(instance)
                await notify_deleted(instance, "Chute updated with use_rolling_update=False")
    else:
        await settings.redis_client.publish(
            "miner_broadcast",
            json.dumps(
                {
                    "reason": "chute_created",
                    "data": {
                        "chute_id": chute.chute_id,
                        "version": chute.version,
                        "job_only": not chute.cords,
                    },
                }
            ).decode(),
        )
    return await get_chute_by_id_or_name(chute.chute_id, db, current_user, load_instances=True)


@router.post("/", response_model=ChuteResponse)
async def deploy_chute(
    chute_args: ChuteArgs,
    accept_fee: Optional[bool] = False,
    db: AsyncSession = Depends(get_db_session),
    current_user: User = Depends(get_current_user()),
):
    """
    Standard deploy from the CDK.
    """
    is_affine_model = False
    http_exc = await ensure_is_developer(db, current_user, raise_=False)
    affine_checked = False
    if http_exc is not None:
        if not await is_affine_registered(db, current_user):
            logger.warning(
                f"Attempted chute creation from non-dev and non-affine user: {current_user.user_id=}"
            )
            raise http_exc
        else:
            if not re.match(r"[^/]+/affine.*", chute_args.name, re.I):
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="Affine miners may only deploy chutes named */affine*",
                )
        affine_checked = True

    # Affine special handling.
    if (
        "affine" in chute_args.name.lower()
        and not current_user.has_role(Permissioning.unlimited_dev)
        and current_user.username.lower() not in ("affine", "affine2", "unconst", "nonaffine")
    ):
        if not affine_checked and not await is_affine_registered(db, current_user):
            logger.warning(
                "Attempted affine deployment by unregistered hotkey: "
                f"{current_user.user_id=} {current_user.username=}"
            )
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Only users with hotkeys registered on affine may deploy chutes with 'affine' in the name",
            )
        valid, message = check_affine_code(chute_args.code)
        if not valid:
            logger.warning(
                f"Affine deployment attempted from {current_user.user_id=} "
                f"{current_user.hotkey=} with invalid code:\n{chute_args.code}"
            )
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=message,
            )

        # Sanity check the model.
        async with aiohttp.ClientSession() as hsession:
            try:
                guessed_config = await guesser.analyze_model(chute_args.name, hsession)
            except HTTPException as e:
                raise e
            except Exception as e:
                logger.error(
                    f"Affine user tried to deploy invalid model: {chute_args.name=} {current_user.username=}"
                )
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Unable to properly evaluate requested model {chute_args.name}: {str(e)}",
                )

        # Force exclude GPU types: 5090 because the inference engines don't support them,
        # mi300x because we don't have containers/miners that support them yet,
        # and b200s because we simply don't have sufficient quantity and need them for kimi-k2.
        chute_args.node_selector.exclude = list(
            set(chute_args.node_selector.exclude or [] + ["b200", "mi300x"])
        )

        # Check that our best guess for model config matches the node selector.
        min_vram_required = guessed_config.required_gpus * guessed_config.min_vram_per_gpu
        node_selector_min_vram = chute_args.node_selector.gpu_count * min(
            [
                SUPPORTED_GPUS[gpu]["memory"]
                for gpu in SUPPORTED_GPUS
                if gpu in chute_args.node_selector.supported_gpus
            ]
        )
        if min_vram_required < 8 * 140 and node_selector_min_vram < min_vram_required:
            logger.error(
                f"Affine user tried to deploy bad node_selector: {min_vram_required=} {node_selector_min_vram} {chute_args.name=} {current_user.username=}"
            )
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=(
                    "node_selector specs are insufficient to support the model: "
                    f"{min_vram_required=} {node_selector_min_vram=}, please fix and try again."
                ),
            )

        logger.success(
            f"Affine deployment initiated: {chute_args.name=} from {current_user.hotkey=}, "
            "code check and prelim model config/node selector config passed."
        )
        is_affine_model = True
        chute_args.logging_enabled = True

    # No-DoS-Plz.
    await limit_deployments(db, current_user)
    if current_user.user_id not in (
        await chutes_user_id(),
        "b167f56b-3e8d-5ffa-88bf-5cc6513bb6f4",
        "5bf8a979-ea71-54bf-8644-26a3411a3b58",
    ) and not current_user.has_role(Permissioning.unlimited_dev):
        bad, response = await is_bad_code(chute_args.code)
        if bad:
            logger.warning(
                f"CODECHECK FAIL: User {current_user.user_id} attempted to deploy bad code {response}\n{chute_args.code}"
            )
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=json.dumps(response).decode(),
            )
    chute = await _deploy_chute(
        chute_args,
        db,
        current_user,
        use_rolling_update=not is_affine_model,
        accept_fee=accept_fee,
    )

    # Auto-share affine chutes with affine admin users.
    if "/affine" in chute.name.lower():
        af_dev_user_ids = (
            (
                await db.execute(
                    select(User.user_id).where(
                        User.permissions_bitmask.op("&")(Permissioning.affine_admin.bitmask) != 0
                    )
                )
            )
            .unique()
            .scalars()
            .all()
        )
        if af_dev_user_ids:
            stmt = insert(ChuteShare).values(
                [
                    {
                        "chute_id": chute.chute_id,
                        "shared_by": current_user.user_id,
                        "shared_to": user_id,
                        "shared_at": func.now(),
                    }
                    for user_id in af_dev_user_ids
                    if user_id != current_user.user_id
                ]
            )
            stmt = stmt.on_conflict_do_nothing()
            await db.execute(stmt)

    return chute


async def _find_latest_image(db: AsyncSession, name: str) -> Image:
    """
    Find the latest vllm/diffusion image.
    """
    chute_user = (
        await db.execute(select(User).where(User.username == "chutes"))
    ).scalar_one_or_none()
    query = (
        select(Image)
        .where(Image.name == name)
        .where(Image.user_id == chute_user.user_id)
        .where(Image.tag != "0.8.3")
        .where(Image.status == "built and pushed")
        .where(~Image.tag.ilike("%nightly%"))
        .where(~Image.tag.ilike("%dev%"))
        .where(~Image.tag.ilike("%.rc%"))
        .order_by(Image.created_at.desc())
        .limit(1)
    )
    return (await db.execute(query)).scalar_one_or_none()


def chute_to_cords(chute: Chute):
    """
    Get all cords for a chute.
    """
    return [
        {
            "method": cord._method,
            "path": cord.path,
            "public_api_path": cord.public_api_path,
            "public_api_method": cord._public_api_method,
            "stream": cord._stream,
            "function": cord._func.__name__,
            "input_schema": cord.input_schema,
            "output_schema": cord.output_schema,
            "output_content_type": cord.output_content_type,
            "minimal_input_schema": cord.minimal_input_schema,
        }
        for cord in chute._cords
    ]


@router.post("/vllm", response_model=ChuteResponse)
async def easy_deploy_vllm_chute(
    args: VLLMChuteArgs,
    db: AsyncSession = Depends(get_db_session),
    current_user: User = Depends(get_current_user()),
):
    """
    Easy/templated vLLM deployment.
    """
    # XXX disabled
    raise HTTPException(
        status_code=status.HTTP_403_FORBIDDEN,
        detail="Easy deployment is currently disabled!",
    )

    if await is_affine_registered(db, current_user):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Easy vllm deployment method not supported for Affine currently.",
        )
    await ensure_is_developer(db, current_user)
    await limit_deployments(db, current_user)

    # Set revision to current main if not specified.
    if not args.revision:
        args.revision = get_current_hf_commit(args.model)
        if not args.revision:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Could not determine current revision from huggingface for {args.model}, and value was not provided",
            )
        logger.info(f"Set the revision automatically to {args.revision}")

    # Make sure we can download the model, set max model length.
    if not args.engine_args:
        args.engine_args = VLLMEngineArgs()
    gated_model = False
    llama_model = False
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(
                f"https://huggingface.co/{args.model}/resolve/main/config.json"
            ) as resp:
                if resp.status == 401:
                    gated_model = True
                resp.raise_for_status()
                try:
                    config = await resp.json()
                except Exception:
                    config = json.loads(await resp.text())
                length = config.get("max_position_embeddings", config.get("model_max_length"))
                if any(
                    [
                        arch.lower() == "llamaforcausallm"
                        for arch in config.get("architectures") or []
                    ]
                ):
                    llama_model = True
                if isinstance(length, str) and length.isidigit():
                    length = int(length)
                if isinstance(length, int):
                    if length <= 16384:
                        if (
                            not args.engine_args.max_model_len
                            or args.engine_args.max_model_len > length
                        ):
                            logger.info(
                                f"Setting max_model_len to {length} due to config.json, model={args.model}"
                            )
                            args.engine_args.max_model_len = length
                    elif not args.engine_args.max_model_len:
                        logger.info(
                            f"Setting max_model_len to 16384 due to excessively large context length in config.json, model={args.model}"
                        )
                        args.engine_args.max_model_len = 16384

        # Also check the tokenizer.
        if not args.engine_args.tokenizer:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"https://huggingface.co/{args.model}/resolve/main/tokenizer_config.json"
                ) as resp:
                    if resp.status == 404:
                        args.engine_args.tokenizer = "unsloth/Llama-3.2-1B-Instruct"
                    resp.raise_for_status()
                    try:
                        config = await resp.json()
                    except Exception:
                        config = json.loads(await resp.text())
                    if not config.get("chat_template"):
                        if config.get("tokenizer_class") == "tokenizer_class" and llama_model:
                            args.engine_args.tokenizer = "unsloth/Llama-3.2-1B-Instruct"
                            logger.warning(
                                f"Chat template not specified in {args.model}, defaulting to llama3"
                            )
                        elif config.get("tokenizer_class") == "LlamaTokenizer":
                            args.engine_args.tokenizer = "jondurbin/bagel-7b-v0.1"
                            logger.warning(
                                f"Chat template not specified in {args.model}, defaulting to llama2 (via bagel)"
                            )
    except Exception as exc:
        logger.warning(f"Error checking model tokenizer_config.json: {exc}")

    # Reject gaited models, e.g. meta-llama/*
    if gated_model:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Model {args.model} appears to have gated access, config.json could not be downloaded",
        )

    image = await _find_latest_image(db, "vllm")
    image = f"chutes/{image.name}:{image.tag}"
    if args.engine_args.max_model_len <= 0:
        args.engine_args.max_model_len = 16384
    code, chute = build_vllm_code(args, current_user.username, image)
    if (node_selector := args.node_selector) is None:
        async with aiohttp.ClientSession() as session:
            try:
                requirements = await guesser.analyze_model(args.model, session)
                node_selector = NodeSelector(
                    gpu_count=requirements.required_gpus,
                    min_vram_gb_per_gpu=requirements.min_vram_per_gpu,
                )
            except Exception:
                node_selector = NodeSelector(gpu_count=1, min_vram_gb_per_gpu=80)
    chute_args = ChuteArgs(
        name=args.model,
        image=image,
        tagline=args.tagline,
        readme=args.readme,
        tool_description=args.tool_description,
        logo_id=args.logo_id if args.logo_id and args.logo_id.strip() else None,
        public=args.public,
        code=code,
        filename="chute.py",
        ref_str="chute:chute",
        standard_template="vllm",
        node_selector=node_selector,
        cords=chute_to_cords(chute.chute),
        jobs=[],
        concurrency=args.concurrency,
        revision=args.revision,
    )
    return await _deploy_chute(chute_args, db, current_user)


@router.post("/diffusion", response_model=ChuteResponse)
async def easy_deploy_diffusion_chute(
    args: DiffusionChuteArgs,
    db: AsyncSession = Depends(get_db_session),
    current_user: User = Depends(get_current_user()),
):
    """
    Easy/templated diffusion deployment.
    """
    # XXX disabled right now.
    raise HTTPException(
        status_code=status.HTTP_403_FORBIDDEN,
        detail="Easy deployment is currently disabled!",
    )

    await ensure_is_developer(db, current_user)
    await limit_deployments(db, current_user)

    image = await _find_latest_image(db, "diffusion")
    image = f"chutes/{image.name}:{image.tag}"
    code, chute = build_diffusion_code(args, current_user.username, image)
    if (node_selector := args.node_selector) is None:
        node_selector = NodeSelector(
            gpu_count=1,
            min_vram_gb_per_gpu=24,
        )
    chute_args = ChuteArgs(
        name=args.name,
        image=image,
        tagline=args.tagline,
        readme=args.readme,
        tool_description=args.tool_description,
        logo_id=args.logo_id if args.logo_id and args.logo_id.strip() else None,
        public=args.public,
        code=code,
        filename="chute.py",
        ref_str="chute:chute",
        standard_template="diffusion",
        node_selector=node_selector,
        cords=chute_to_cords(chute.chute),
        jobs=[],
    )
    return await _deploy_chute(chute_args, db, current_user)


@router.post("/tei", response_model=ChuteResponse)
async def easy_deploy_tei_chute(
    args: TEIChuteArgs,
    db: AsyncSession = Depends(get_db_session),
    current_user: User = Depends(get_current_user()),
):
    """
    Easy/templated text-embeddings-inference deployment.
    """
    # XXX disabled
    raise HTTPException(
        status_code=status.HTTP_403_FORBIDDEN,
        detail="Easy deployment is currently disabled!",
    )

    await ensure_is_developer(db, current_user)
    await limit_deployments(db, current_user)

    image = await _find_latest_image(db, "tei")
    image = f"chutes/{image.name}:{image.tag}"
    code, chute = build_tei_code(args, current_user.username, image)
    if (node_selector := args.node_selector) is None:
        node_selector = NodeSelector(
            gpu_count=1,
            min_vram_gb_per_gpu=16,
        )
    node_selector.exclude = list(
        set(
            node_selector.exclude
            or [] + ["h200", "b200", "h100", "h100_sxm", "h100_nvl", "h800", "mi300x"]
        )
    )

    chute_args = ChuteArgs(
        name=args.model,
        image=image,
        tagline=args.tagline,
        readme=args.readme,
        tool_description=args.tool_description,
        logo_id=args.logo_id if args.logo_id and args.logo_id.strip() else None,
        public=args.public,
        code=code,
        filename="chute.py",
        ref_str="chute:chute",
        standard_template="tei",
        node_selector=node_selector,
        cords=chute_to_cords(chute.chute),
        jobs=[],
    )
    return await _deploy_chute(chute_args, db, current_user)


@router.put("/{chute_id_or_name:path}", response_model=ChuteResponse)
async def update_common_attributes(
    chute_id_or_name: str,
    args: ChuteUpdateArgs,
    db: AsyncSession = Depends(get_db_session),
    current_user: User = Depends(get_current_user()),
):
    """
    Update readme, tagline, etc. (but not code, image, etc.).
    """
    chute = await get_chute_by_id_or_name(chute_id_or_name, db, current_user, load_instances=True)
    if not chute or chute.user_id != current_user.user_id:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Chute not found, or does not belong to you",
        )
    if args.tagline and args.tagline.strip():
        chute.tagline = args.tagline
    if args.readme and args.readme.strip():
        chute.readme = args.readme
    if args.tool_description and args.tool_description.strip():
        chute.tool_description = args.tool_description
    if args.logo_id:
        chute.logo_id = args.logo_id
    await db.commit()
    await db.refresh(chute)
    return chute
