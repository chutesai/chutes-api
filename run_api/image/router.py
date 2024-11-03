"""
Routes for images.
"""

import io
import uuid
import time
import asyncio
from loguru import logger
import redis.asyncio as redis
from fastapi import APIRouter, Depends, HTTPException, status, File, Form, UploadFile
from starlette.responses import StreamingResponse
from sqlalchemy import or_, exists, func
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from typing import Optional
from run_api.image.schemas import Image
from run_api.chute.schemas import Chute
from run_api.user.schemas import User
from run_api.user.service import get_current_user
from run_api.database import get_db_session
from run_api.config import settings
from run_api.image.response import ImageResponse
from run_api.image.util import get_image_by_id_or_name
from run_api.pagination import PaginatedResponse

router = APIRouter()


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
        query = query.owhere(Image.tag.ilike(f"%{tag}%"))

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
    if (
        await db.execute(select(exists().where(Chute.image_id == image.image_id)))
    ).scalar():
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Image is in use by one or more chutes",
        )
    image_id = image.image_id
    await db.delete(image)
    await db.commit()
    return {"image_id": image_id, "deleted": True}


@router.post("/", status_code=status.HTTP_202_ACCEPTED)
async def create_image(
    wait: bool = Form(...),
    build_context: UploadFile = File(...),
    name: str = Form(...),
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
    image_id = str(
        uuid.uuid5(uuid.NAMESPACE_OID, f"{current_user.user_id}/{name}:{tag}")
    )
    if (await db.execute(select(exists().where(Image.image_id == image_id)))).scalar():
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Image with {name=} and {tag=} aready exists",
        )

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
        result = await settings.storage_client.put_object(
            settings.storage_bucket,
            destination,
            obj,
            length=-1,
            part_size=10 * 1024 * 1024,
        )
        logger.success(
            f"Uploaded build context component {image_id=} to {settings.storage_bucket}/{destination} etag={result.etag}"
        )

    # Create the image once we've persisted the context, which will trigger the build via events.
    image = Image(
        image_id=image_id,
        user_id=current_user.user_id,
        name=name,
        tag=tag,
        public=public,
    )
    db.add(image)
    await db.commit()
    await db.refresh(image)

    # Clean up any previous streams, just in case of retry.
    redis_client = redis.Redis.from_url(settings.redis_url)
    await redis_client.delete(f"forge:{image_id}:stream")

    # Stream logs for clients who set the "wait" flag.
    async def _stream():
        started_at = time.time()
        last_offset = None
        while True:
            stream_result = None
            try:
                stream_result = await redis_client.xrange(
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
                    await redis_client.delete(f"forge:{image_id}:stream")
                    yield "DONE\n"
                    break
                yield f"data: {data[b'data'].decode()}\n\n"
        delta = time.time() - started_at
        logger.success(
            "\N{hammer and wrench} "
            + f"finished building image {image_id} in {round(delta, 5)} seconds"
        )

    if wait:
        return StreamingResponse(_stream())
    return image
