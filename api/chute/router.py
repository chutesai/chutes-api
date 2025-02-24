"""
Routes for chutes.
"""

import re
import random
import string
import uuid
import orjson as json
import aiohttp
from loguru import logger
from slugify import slugify
from fastapi import APIRouter, Depends, HTTPException, status, Request
from fastapi_cache.decorator import cache
from starlette.responses import StreamingResponse
from sqlalchemy import or_, exists, func
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from sqlalchemy.orm import selectinload
from typing import Optional
from api.chute.schemas import Chute, ChuteArgs, InvocationArgs, NodeSelector, ChuteUpdateArgs
from api.chute.templates import (
    VLLMChuteArgs,
    VLLMEngineArgs,
    DiffusionChuteArgs,
    TEIChuteArgs,
    build_vllm_code,
    build_diffusion_code,
    build_tei_code,
)
from api.chute.response import ChuteResponse
from api.chute.util import get_chute_by_id_or_name, invoke, selector_hourly_price
from api.user.schemas import User
from api.user.service import get_current_user, chutes_user_id
from api.image.schemas import Image
from api.image.util import get_image_by_id_or_name
from api.instance.util import discover_chute_targets
from api.database import get_db_session
from api.pagination import PaginatedResponse
from api.fmv.fetcher import get_fetcher
from api.config import settings
from api.constants import (
    LLM_PRICE_MULT_PER_MILLION,
    DIFFUSION_PRICE_MULT_PER_STEP,
)
from api.util import ensure_is_developer, rate_limit
from api.permissions import Permissioning
from api.guesser import guesser

router = APIRouter()


async def _inject_current_estimated_price(chute: Chute, response: ChuteResponse):
    """
    Inject the current estimated price data into a response.
    """
    if chute.standard_template == "vllm":
        hourly = await selector_hourly_price(chute.node_selector)
        per_million = hourly * LLM_PRICE_MULT_PER_MILLION
        if chute.discount:
            per_million -= per_million * chute.discount
        response.current_estimated_price = {"per_million_tokens": {"usd": per_million}}
        tao_usd = await get_fetcher().get_price("tao")
        if tao_usd:
            response.current_estimated_price["per_million_tokens"]["tao"] = per_million / tao_usd
    elif chute.standard_template == "diffusion":
        hourly = await selector_hourly_price(chute.node_selector)
        per_step = hourly * DIFFUSION_PRICE_MULT_PER_STEP
        if chute.discount:
            per_step -= per_step * chute.discount
        response.current_estimated_price = {"per_step": {"usd": per_step}}
        tao_usd = await get_fetcher().get_price("tao")
        if tao_usd:
            response.current_estimated_price["per_step"]["tao"] = per_step / tao_usd

    # Legacy/fallback.
    if not response.current_estimated_price:
        response.current_estimated_price = {}
    response.current_estimated_price.update(
        await NodeSelector(**chute.node_selector).current_estimated_price()
    )
    if chute.discount and response.current_estimated_price:
        for key in ("usd", "tao"):
            values = response.current_estimated_price.get(key)
            if values:
                for unit in values:
                    values[unit] -= values[unit] * chute.discount


@cache(expire=60)
@router.get("/", response_model=PaginatedResponse)
async def list_chutes(
    include_public: Optional[bool] = False,
    template: Optional[str] = None,
    name: Optional[str] = None,
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
                    f"include_public:{include_public}",
                    f"include_schemas:{include_schemas}",
                    f"user:{current_user.user_id if current_user else None}",
                ]
            ),
        )
    ).encode()
    cached = await settings.memcache.get(cache_key)
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
                )
            )
        else:
            query = query.where(Chute.user_id == current_user.user_id)
    else:
        query = query.where(Chute.public.is_(True))

    # Filter by name/tag/etc.
    if name and name.strip():
        query = query.where(Chute.name.ilike(f"%{name}%"))
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
    for item in result.scalars().all():
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
    await settings.memcache.set(cache_key, json.dumps(result), exptime=300)
    return result


@cache(expire=60)
@router.get("/{chute_id_or_name:path}", response_model=ChuteResponse)
async def get_chute(
    chute_id_or_name: str,
    db: AsyncSession = Depends(get_db_session),
    current_user: User = Depends(get_current_user(purpose="chutes")),
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
):
    """
    Deploy a chute!
    """
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
        await db.execute(
            select(Chute)
            .where(Chute.name.ilike(chute_args.name))
            .where(Chute.user_id == current_user.user_id)
            .options(selectinload(Chute.instances))
        )
    ).scalar_one_or_none()
    if chute and chute.version == version and chute.public == chute_args.public:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Chute with name={chute_args.name}, {version=} and public={chute_args.public} already exists",
        )

    # Prevent h200 usage for now.
    if not chute_args.node_selector:
        chute_args.node_selector = {"gpu_count": 1}
    if isinstance(chute_args.node_selector, dict):
        chute_args.node_selector = NodeSelector(**chute_args.node_selector)
    if current_user.user_id != await chutes_user_id():
        if (
            chute_args.node_selector
            and chute_args.node_selector.min_vram_gb_per_gpu
            and chute_args.node_selector.min_vram_gb_per_gpu > 80
        ):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="You are not allowed to require h200 at this time.",
            )
        if not chute_args.node_selector.exclude:
            chute_args.node_selector.exclude = []
        if "h200" not in chute_args.node_selector.exclude:
            chute_args.node_selector.exclude.append("h200")
        if not chute_args.node_selector.supported_gpus:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="No supported GPUs based on node selector!",
            )

    old_version = None
    if chute:
        # Make sure we delete the old instances.
        for instance in chute.instances:
            await db.delete(instance)
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
        chute.public = chute_args.public
        chute.logo_id = (
            chute_args.logo_id if chute_args.logo_id and chute_args.logo_id.strip() else None
        )
        chute.chutes_version = image.chutes_version
        chute.updated_at = func.now()
    else:
        chute = Chute(
            chute_id=str(
                uuid.uuid5(uuid.NAMESPACE_OID, f"{current_user.username}::chute::{chute_args.name}")
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
            public=chute_args.public,
            cords=chute_args.cords,
            node_selector=chute_args.node_selector,
            standard_template=chute_args.standard_template,
            chutes_version=image.chutes_version,
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

    # Limit h200 access for now.
    if (chute.node_selector or {}).get("supported_gpus", []) == [
        "h200"
    ] and chute.user_id != await chutes_user_id():
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You are not allowed to require h200 at this time.",
        )

    await db.commit()
    await db.refresh(chute)

    if old_version:
        await settings.redis_client.publish(
            "miner_broadcast",
            json.dumps(
                {
                    "reason": "chute_updated",
                    "data": {
                        "chute_id": chute.chute_id,
                        "version": chute.version,
                        "old_version": old_version,
                    },
                }
            ).decode(),
        )
    else:
        await settings.redis_client.publish(
            "miner_broadcast",
            json.dumps(
                {
                    "reason": "chute_created",
                    "data": {
                        "chute_id": chute.chute_id,
                        "version": chute.version,
                    },
                }
            ).decode(),
        )
    return await get_chute_by_id_or_name(chute.chute_id, db, current_user, load_instances=True)


@router.post("/", response_model=ChuteResponse)
async def deploy_chute(
    chute_args: ChuteArgs,
    db: AsyncSession = Depends(get_db_session),
    current_user: User = Depends(get_current_user()),
):
    """
    Standard deploy from the CDK.
    """
    await ensure_is_developer(db, current_user)
    return await _deploy_chute(chute_args, db, current_user)


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
    await ensure_is_developer(db, current_user)

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
    await ensure_is_developer(db, current_user)
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
    await ensure_is_developer(db, current_user)
    image = await _find_latest_image(db, "tei")
    image = f"chutes/{image.name}:{image.tag}"
    code, chute = build_tei_code(args, current_user.username, image)
    if (node_selector := args.node_selector) is None:
        node_selector = NodeSelector(
            gpu_count=1,
            min_vram_gb_per_gpu=16,
        )
    if not node_selector.include and not node_selector.exclude:
        node_selector.exclude = ["h200", "h100", "h100_sxm"]
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
    )
    return await _deploy_chute(chute_args, db, current_user)


@router.post("/{chute_id}/{path:path}")
async def invoke_(
    chute_id: str,
    path: str,
    invocation: InvocationArgs,
    request: Request,
    db: AsyncSession = Depends(get_db_session),
    current_user: User = Depends(get_current_user()),
):
    """
    Invoke a "chute" aka function.
    """
    logger.warning(f"INVOKE_VIA_SDK: {chute_id=} {path=}")
    return HTTPException(
        status_code=status.HTTP_400_BAD_REQUEST,
        detail="Please use the standard JSON API calls",
    )

    if current_user.balance <= 0 and not current_user.has_role(Permissioning.free_account):
        raise HTTPException(
            status_code=status.HTTP_402_PAYMENT_REQUIRED,
            detail=f"Account balance is ${current_user.balance}, please send tao to {current_user.payment_address}",
        )

    # Rate limit requests.
    await rate_limit(chute_id, current_user, settings.rate_limit_count, settings.rate_limit_window)

    args = invocation.args
    kwargs = invocation.kwargs
    query = (
        select(Chute)
        .join(User, Chute.user_id == User.user_id)
        .where(or_(Chute.public.is_(True), Chute.user_id == current_user.user_id))
        .where(Chute.chute_id == chute_id)
    )
    result = await db.execute(query)
    chute = result.scalar_one_or_none()
    if not chute:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Chute not found, or you do not have permission to use",
        )

    # Find a target to query.
    targets = await discover_chute_targets(db, chute_id, max_wait=60)
    if not targets:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"No instances available (yet) for {chute_id=}",
        )

    # Identify the upstream path to call.
    cord = None
    path = "/" + path.lstrip("/")
    identified = False
    stream = False
    function = None
    for cord in chute.cords:
        if cord["path"] == path:
            identified = True
            stream = cord["stream"]
            function = cord["function"]
    if not identified:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Chute has no cord matching your request",
        )

    # Initialize metrics.
    metrics = None
    if chute.standard_template == "vllm":
        metrics = {
            "ttft": None,
            "tps": 0.0,
            "tokens": 0,
        }

    # Do the deed.
    await db.close()
    parent_invocation_id = str(uuid.uuid4())
    return StreamingResponse(
        invoke(
            chute,
            current_user.user_id,
            path,
            function,
            stream,
            args,
            kwargs,
            targets,
            parent_invocation_id,
            metrics=metrics,
            request=request,
        ),
        headers={"X-Chutes-InvocationID": parent_invocation_id},
    )


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
    if not chute:
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
