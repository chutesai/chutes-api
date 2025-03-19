"""
Routes for images.
"""

import io
import uuid
import time
import asyncio
import orjson as json
from loguru import logger
from fastapi import APIRouter, Depends, HTTPException, status, File, Form, UploadFile
from fastapi_cache.decorator import cache
from starlette.responses import StreamingResponse
from sqlalchemy import and_, or_, exists, func, delete
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from typing import Optional
from api.image.schemas import Image
from api.chute.schemas import Chute
from api.user.schemas import User
from api.fs_challenge.schemas import FSChallenge
from api.user.service import get_current_user
from api.database import get_db_session
from api.config import settings
from api.image.response import ImageResponse
from api.image.util import get_image_by_id_or_name
from api.pagination import PaginatedResponse
from api.util import ensure_is_developer, limit_images

router = APIRouter()


@cache(expire=60)
@router.get("/", response_model=PaginatedResponse)
async def list_images(
    include_public: Optional[bool] = False,
    name: Optional[str] = None,
    tag: Optional[str] = None,
    page: Optional[int] = 0,
    limit: Optional[int] = 25,
    db: AsyncSession = Depends(get_db_session),
    current_user: User = Depends(get_current_user(purpose="images")),
):
    """
    List (and optionally filter/paginate) images.
    """
    logger.debug(f"Listing images for user {current_user.username}")
    query = select(Image)

    # Filter by public and/or only the user's images.
    if include_public:
        query = query.where(
            or_(
                Image.public.is_(True),
                Image.user_id == current_user.user_id,
            )
        )
    else:
        query = query.where(Image.user_id == current_user.user_id)

    # Filter by name/tag/etc.
    if name and name.strip():
        query = query.where(Image.name.ilike(f"%{name}%"))
    if tag and tag.strip():
        query = query.where(Image.tag.ilike(f"%{tag}%"))

    # Perform a count.
    total_query = select(func.count()).select_from(query.subquery())
    total_result = await db.execute(total_query)
    total = total_result.scalar() or 0

    # Pagination.
    query = (
        query.order_by(Image.created_at.desc())
        .offset((page or 0) * (limit or 25))
        .limit((limit or 25))
    )

    result = await db.execute(query)
    return {
        "total": total,
        "page": page,
        "limit": limit,
        "items": [ImageResponse.from_orm(item) for item in result.scalars().all()],
    }


@router.get("/{image_id_or_name:path}", response_model=ImageResponse)
async def get_image(
    image_id_or_name: str,
    db: AsyncSession = Depends(get_db_session),
    current_user: User = Depends(get_current_user(purpose="images")),
):
    """
    Load a single image by ID or name.
    """
    image = await get_image_by_id_or_name(image_id_or_name, db, current_user)
    if not image:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Image not found, or does not belong to you",
        )
    return image


@router.delete("/{image_id_or_name:path}")
async def delete_image(
    image_id_or_name: str,
    db: AsyncSession = Depends(get_db_session),
    current_user: User = Depends(get_current_user(purpose="images")),
):
    """
    Delete an image by ID or name:tag.
    """
    image = await get_image_by_id_or_name(image_id_or_name, db, current_user)
    if not image or image.user_id != current_user.user_id:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Image not found, or does not belong to you",
        )

    # No deleting images that have an associated chute.
    if (await db.execute(select(exists().where(Chute.image_id == image.image_id)))).scalar():
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Image is in use by one or more chutes",
        )
    image_id = image.image_id
    await db.delete(image)
    await db.commit()

    await settings.redis_client.publish(
        "miner_broadcast",
        json.dumps(
            {
                "reason": "image_deleted",
                "data": {
                    "image_id": image_id,
                },
            }
        ).decode(),
    )

    return {"image_id": image_id, "deleted": True}


@router.post("/", status_code=status.HTTP_202_ACCEPTED)
async def create_image(
    wait: bool = Form(...),
    build_context: UploadFile = File(...),
    username: str = Form(...),
    name: str = Form(...),
    readme: str = Form(...),
    logo_id: str = Form(...),
    tag: str = Form(...),
    dockerfile: str = Form(...),
    image: str = Form(...),
    public: bool = Form(...),
    db: AsyncSession = Depends(get_db_session),
    current_user: User = Depends(get_current_user()),
):
    """
    Create an image; really here we're just storing the metadata
    in the DB and kicking off the image build asynchronously.
    """
    if current_user.username != username:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Cannot make images for users other than yourself!",
        )
    await ensure_is_developer(db, current_user)
    await limit_images(db, current_user)

    image_id = str(uuid.uuid5(uuid.NAMESPACE_OID, f"{username}/{name}:{tag}"))
    query = select(
        exists().where(
            or_(
                Image.image_id == image_id,
                and_(
                    Image.user_id == current_user.user_id,
                    Image.name == name,
                    Image.tag == tag,
                ),
            )
        )
    )
    if (await db.execute(query)).scalar():
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Image with {name=} and {tag=} aready exists",
        )

    # Double check we've purged all FS challenges.
    await db.execute(delete(FSChallenge).where(FSChallenge.image_id == image_id))

    # Force installation of chutes with the specified version.
    dockerfile += f"\n\nRUN pip install --upgrade chutes=={settings.chutes_version}"

    # Upload the build context to our S3-compatible storage backend.
    for obj, destination in (
        (build_context, f"forge/{current_user.user_id}/{image_id}.zip"),
        (
            io.BytesIO(dockerfile.encode()),
            f"forge/{current_user.user_id}/{image_id}.Dockerfile",
        ),
        (
            io.BytesIO(image.encode()),
            f"forge/{current_user.user_id}/{image_id}.pickle",
        ),
    ):
        logger.info(f"Trying to upload: {destination}")
        async with settings.s3_client() as s3:
            await s3.upload_fileobj(obj, settings.storage_bucket, destination)
        logger.success(
            f"Uploaded build context component {image_id=} to {settings.storage_bucket}/{destination}"
        )

    # Create the image once we've persisted the context, which will trigger the build via events.
    image = Image(
        image_id=image_id,
        user_id=current_user.user_id,
        name=name,
        readme=readme,
        logo_id=logo_id if logo_id and logo_id != "__none__" else None,
        tag=tag,
        public=public,
        chutes_version=settings.chutes_version,
    )
    db.add(image)
    await db.commit()
    await db.refresh(image)

    # Clean up any previous streams, just in case of retry.
    await settings.redis_client.delete(f"forge:{image_id}:stream")

    # Stream logs for clients who set the "wait" flag.
    async def _stream():
        started_at = time.time()
        last_offset = None
        while True:
            stream_result = None
            try:
                stream_result = await settings.redis_client.xrange(
                    f"forge:{image_id}:stream", last_offset or "-", "+"
                )
            except Exception as exc:
                logger.error(f"Error fetching stream result: {exc}")
                yield f"data: ERROR: {exc}"
                return
            if not stream_result:
                await asyncio.sleep(1.0)
                continue
            for offset, data in stream_result:
                last_offset = offset.decode()
                parts = last_offset.split("-")
                last_offset = parts[0] + "-" + str(int(parts[1]) + 1)
                if data[b"data"] == b"DONE":
                    await settings.redis_client.delete(f"forge:{image_id}:stream")
                    yield "DONE\n"
                    break
                yield f"data: {data[b'data'].decode()}\n\n"
        delta = time.time() - started_at
        logger.success(
            "\N{HAMMER AND WRENCH} "
            + f"finished building image {image_id} in {round(delta, 5)} seconds"
        )

    if wait:
        return StreamingResponse(_stream())
    return image
