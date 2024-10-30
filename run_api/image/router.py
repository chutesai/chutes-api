"""
Routes for images.
"""

import re
import uuid
from loguru import logger
from fastapi import APIRouter, Depends, HTTPException, status, File, Form, UploadFile
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
    current_user: User = Depends(get_current_user),
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


@router.get("/{image_id}", response_model=ImageResponse)
async def get_image(
    image_id: str,
    db: AsyncSession = Depends(get_db_session),
    current_user: User = Depends(get_current_user),
):
    """
    Load a single image by ID.
    """
    query = (
        select(Image)
        .where(or_(Image.public.is_(True), Image.user_id == current_user.user_id))
        .where(Image.image_id == image_id)
    )
    result = await db.execute(query)
    image = result.scalar_one_or_none()
    if not image:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Image not found, or does not belong to you",
        )
    return image


@router.get("/{image_name:path}", response_model=ImageResponse)
async def get_image_by_name(
    image_name: str,
    db: AsyncSession = Depends(get_db_session),
    current_user: User = Depends(get_current_user),
):
    """
    Load a single image by ID.
    """
    name_match = re.match(
        r"([a-z0-9][a-z0-9_-]*)/([a-z0-9][a-z0-9_-]*):([a-z0-9][a-z0-9_\.-]*)$",
        image_name,
        re.I,
    )
    if not name_match:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Image not found, or does not belong to you (re): {image_name}",
        )
    username = name_match.group(1)
    image_name = name_match.group(2)
    tag = name_match.group(3)
    query = (
        select(Image)
        .join(User, Image.user_id == User.user_id)
        .where(or_(Image.public.is_(True), Image.user_id == current_user.user_id))
        .where(User.username == username)
        .where(Image.name == image_name)
        .where(Image.tag == tag)
    )
    result = await db.execute(query)
    image = result.scalar_one_or_none()
    if not image:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Image not found, or does not belong to you",
        )
    return image


@router.delete("/{image_id}")
async def delete_image(
    image_id: str,
    db: AsyncSession = Depends(get_db_session),
    current_user: User = Depends(get_current_user),
):
    """
    Delete an image by ID.
    """
    query = (
        select(Image)
        .where(Image.user_id == current_user.user_id)
        .where(Image.image_id == image_id)
    )
    result = await db.execute(query)
    image = result.scalar_one_or_none()
    if not image:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Image not found, or does not belong to you",
        )

    # No deleting images that have an associated chute.
    if await db.query(exists().where(Chute.image_id == image_id)).scalar():
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Image is in use by one or more chutes",
        )
    await db.delete(image)
    await db.commit()
    return {"image_id": image_id, "deleted": True}


@router.post("/", status_code=status.HTTP_202_ACCEPTED)
async def create_image(
    build_context: UploadFile = File(...),
    name: str = Form(...),
    tag: str = Form(...),
    dockerfile: str = Form(...),
    image: str = Form(...),
    public: bool = Form(...),
    db: AsyncSession = Depends(get_db_session),
    current_user: User = Depends(get_current_user),
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
    destination = f"build-contexts/{current_user.user_id}/{image_id}.zip"
    result = await settings.storage_client.put_object(
        settings.storage_bucket,
        destination,
        build_context,
        length=-1,
        part_size=10 * 1024 * 1024,
    )
    logger.success(
        f"Uploaded build context for {image_id=} to {settings.storage_bucket}/{destination} etag={result.etag}"
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
    return image
