"""
Routes for chutes.
"""

import re
import random
import string
import uuid
import orjson as json
from slugify import slugify
from fastapi import APIRouter, Depends, HTTPException, status, Request
from starlette.responses import StreamingResponse
from sqlalchemy import or_, exists, func
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from sqlalchemy.orm import selectinload
from typing import Optional
from api.chute.schemas import Chute, ChuteArgs, InvocationArgs, NodeSelector
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
    query = query.offset((page or 0) * (limit or 25)).limit((limit or 25))

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


@router.post("/", response_model=ChuteResponse)
async def deploy_chute(
    chute_args: ChuteArgs,
    db: AsyncSession = Depends(get_db_session),
    current_user: User = Depends(get_current_user()),
):
    """
    Deploy a chute!
    """
    await ensure_is_developer(db, current_user)

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
    version = str(uuid.uuid5(uuid.NAMESPACE_OID, chute_args.code.encode()))
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
            slugify(f"{current_user.username}-{chute.name}").lower(),
        )
        base_slug = chute.slug
        already_exists = (
            await db.execute(select(exists().where(Chute.slug == chute.slug)))
        ).scalar()
        while already_exists:
            suffix = "".join(
                random.choice(string.ascii_lowercase + string.digits) for _ in range(6)
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
