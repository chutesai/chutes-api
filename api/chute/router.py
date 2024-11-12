"""
Routes for chutes.
"""

import re
import random
import string
import orjson as json
import redis.asyncio as redis
from slugify import slugify
from fastapi import APIRouter, Depends, HTTPException, status, Request
from starlette.responses import StreamingResponse
from sqlalchemy import or_, exists, func
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from typing import Optional
from api.chute.schemas import Chute, ChuteArgs, InvocationArgs
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

router = APIRouter()


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
    query = select(Chute)

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
    query = query.offset((page or 0) * (limit or 25)).limit((limit or 25))

    result = await db.execute(query)
    return {
        "total": total,
        "page": page,
        "limit": limit,
        "items": [ChuteResponse.from_orm(item) for item in result.scalars().all()],
    }


@router.get("{chute_id_or_name:path}", response_model=ChuteResponse)
async def get_chute(
    chute_id_or_name: str,
    db: AsyncSession = Depends(get_db_session),
    current_user: User = Depends(get_current_user(purpose="chutes")),
):
    """
    Load a chute by ID or name.
    """
    chute = await get_chute_by_id_or_name(chute_id_or_name, db, current_user)
    if not chute:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Chute not found, or does not belong to you",
        )
    return chute


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
    await db.delete(chute)
    await db.commit()

    async with redis.from_url(settings.redis_url) as redis_client:
        await redis_client.publish(
            "miner_broadcast",
            json.dumps(
                {
                    "reason": "chute_deleted",
                    "data": {"chute_id": chute_id},
                }
            ).decode(),
        )

    return {"chute_id": chute_id, "deleted": True}


@router.post("/", response_model=ChuteResponse)
async def deploy_chute(
    chute_args: ChuteArgs,
    db: AsyncSession = Depends(get_db_session),
    current_user: User = Depends(get_current_user()),
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
    if (
        await db.execute(
            select(exists().where(Chute.name.ilike(chute_args.name)).where(Chute.user_id == current_user.user_id))
        )
    ).scalar():
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Chute with name={chute_args.name} already exists",
        )

    chute = Chute(
        image_id=image.image_id,
        user_id=current_user.user_id,
        name=chute_args.name,
        public=chute_args.public,
        cords=chute_args.cords,
        node_selector=chute_args.node_selector,
        standard_template=chute_args.standard_template,
    )

    # Generate a unique slug (subdomain).
    chute.slug = re.sub(r"[^a-z0-9-]+$", "-", slugify(f"{current_user.username}-{chute.name}").lower())
    base_slug = chute.slug
    already_exists = (await db.execute(select(exists().where(Chute.slug == chute.slug)))).scalar()
    while already_exists:
        suffix = "".join(random.choice(string.ascii_lowercase + string.digits) for _ in range(6))
        chute.slug = f"{base_slug}-{suffix}"
        already_exists = (await db.execute(select(exists().where(Chute.slug == chute.slug)))).scalar()

    db.add(chute)
    await db.commit()
    await db.refresh(chute)
    return chute


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
            detail="No instances available (yet) for {chute_id=}",
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
    return StreamingResponse(invoke(chute, current_user.user_id, path, function, stream, args, kwargs, targets))
