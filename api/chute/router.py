"""
Routes for chutes.
"""

import re
import random
import string
import uuid
import orjson as json
import aiohttp
from slugify import slugify
from fastapi import APIRouter, Depends, HTTPException, status, Request
from starlette.responses import StreamingResponse
from sqlalchemy import or_, exists, func
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from sqlalchemy.orm import selectinload
from typing import Optional
from api.chute.schemas import Chute, ChuteArgs, InvocationArgs, NodeSelector
from api.chute.templates import (
    VLLMChuteArgs,
    DiffusionChuteArgs,
    TEIChuteArgs,
    build_vllm_code,
    build_diffusion_code,
    build_tei_code,
)
from api.chute.response import ChuteResponse
from api.chute.util import get_chute_by_id_or_name, invoke
from api.user.schemas import User
from api.user.service import get_current_user
from api.image.schemas import Image
from api.image.util import get_image_by_id_or_name
from api.instance.util import discover_chute_targets
from api.database import get_db_session
from api.pagination import PaginatedResponse
from api.config import settings
from api.util import ensure_is_developer
from api.permissions import Permissioning
from api.guesser import guesser

router = APIRouter()


async def _inject_current_estimated_price(chute: Chute, response: ChuteResponse):
    """
    Inject the current estimated price data into a response.
    """
    response.current_estimated_price = await NodeSelector(
        **chute.node_selector
    ).current_estimated_price()
    if not response.current_estimated_price:
        response.current_estimated_price = {"error": "pricing unavailable"}


@router.get("/", response_model=PaginatedResponse)
async def list_chutes(
    include_public: Optional[bool] = False,
    name: Optional[str] = None,
    image: Optional[str] = None,
    page: Optional[int] = 0,
    limit: Optional[int] = 25,
    db: AsyncSession = Depends(get_db_session),
    current_user: User = Depends(get_current_user(purpose="chutes")),
):
    """
    List (and optionally filter/paginate) chutes.
    """
    query = select(Chute).options(selectinload(Chute.instances))

    # Filter by public and/or only the user's chutes.
    if include_public:
        query = query.where(
            or_(
                Chute.public.is_(True),
                Chute.user_id == current_user.user_id,
            )
        )
    else:
        query = query.where(Chute.user_id == current_user.user_id)

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

    # Perform a count.
    total_query = select(func.count()).select_from(query.subquery())
    total_result = await db.execute(total_query)
    total = total_result.scalar() or 0

    # Pagination.
    query = (
        query.order_by(Chute.created_at.desc())
        .offset((page or 0) * (limit or 25))
        .limit((limit or 25))
    )

    result = await db.execute(query)
    responses = []
    for item in result.scalars().all():
        responses.append(ChuteResponse.from_orm(item))
        await _inject_current_estimated_price(item, responses[-1])
    return {
        "total": total,
        "page": page,
        "limit": limit,
        "items": responses,
    }


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
        )
    ).scalar_one_or_none()
    if chute and chute.version == version and chute.public == chute_args.public:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Chute with name={chute_args.name}, {version=} and public={chute_args.public} already exists",
        )
    old_version = None
    if chute:
        old_version = chute.version
        chute.image_id = image.image_id
        chute.readme = chute_args.readme
        chute.code = chute_args.code
        chute.filename = chute_args.filename
        chute.ref_str = chute_args.ref_str
        chute.version = version
        chute.public = chute_args.public
        chute.logo_id = chute_args.logo_id
    else:
        chute = Chute(
            chute_id=str(
                uuid.uuid5(uuid.NAMESPACE_OID, f"{current_user.username}::chute::{chute_args.name}")
            ),
            image_id=image.image_id,
            user_id=current_user.user_id,
            name=chute_args.name,
            readme=chute_args.readme,
            logo_id=chute_args.logo_id,
            code=chute_args.code,
            filename=chute_args.filename,
            ref_str=chute_args.ref_str,
            version=version,
            public=chute_args.public,
            cords=chute_args.cords,
            node_selector=chute_args.node_selector,
            standard_template=chute_args.standard_template,
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
    await db.commit()

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
        .order_by(func.version_numbers(Image.tag).desc())
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
    image = await _find_latest_image(db, "vllm")
    image = f"chutes/{image.name}:{image.tag}"
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
        readme=args.readme,
        logo_id=args.logo_id,
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
        readme=args.readme,
        logo_id=args.logo_id,
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
    chute_args = ChuteArgs(
        name=args.model,
        image=image,
        readme=args.readme,
        logo_id=args.logo_id,
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
    if current_user.balance <= 0 and not current_user.has_role(Permissioning.free_account):
        raise HTTPException(
            status_code=status.HTTP_402_PAYMENT_REQUIRED,
            detail=f"Account balance is ${current_user.balance}, please send tao to {current_user.payment_address}",
        )

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

    # Do the deed.
    await db.close()
    return StreamingResponse(
        invoke(chute, current_user.user_id, path, function, stream, args, kwargs, targets)
    )
