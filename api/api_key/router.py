"""
API keys router.
"""

from typing import Optional
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select, func
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession
from api.api_key.schemas import APIKeyArgs, APIKey
from api.api_key.response import APIKeyCreationResponse, APIKeyResponse
from api.user.schemas import User
from api.user.service import get_current_user
from api.database import get_db_session
from api.pagination import PaginatedResponse
from loguru import logger
from api.api_key.schemas import APIKeyArgs, APIKey
from api.api_key.response import APIKeyCreationResponse, APIKeyResponse
from api.user.schemas import User
from api.user.service import get_current_user
from api.database import get_db_session
from api.pagination import PaginatedResponse

router = APIRouter()


@router.get("/", response_model=PaginatedResponse)
async def list_images(
    page: Optional[int] = 0,
    limit: Optional[int] = 25,
    db: AsyncSession = Depends(get_db_session),
    current_user: User = Depends(get_current_user(purpose="api_keys")),
):
    """
    List (and optionally filter/paginate) images.
    """

    query = select(APIKey).where(APIKey.user_id == current_user.user_id)

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
        "items": [
            APIKeyResponse.from_orm(item) for item in result.scalars().unique().all()
        ],
    }


@router.delete("/{api_key_id}")
async def delete_api_key(
    api_key_id: str,
    db: AsyncSession = Depends(get_db_session),
    current_user: User = Depends(get_current_user(purpose="api_keys")),
):
    """
    Delete an API key by ID.
    """
    query = select(APIKey).where(APIKey.api_key_id == api_key_id)
    api_key = (await db.execute(query)).unique().scalar_one_or_none()
    if not api_key:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Image not found, or does not belong to you",
        )
    await db.delete(api_key)
    await db.commit()
    return {"api_key_id": api_key_id, "deleted": True}


@router.post("/", response_model=APIKeyCreationResponse)
async def create_api_key(
    key_args: APIKeyArgs,
    db: AsyncSession = Depends(get_db_session),
    current_user: User = Depends(get_current_user()),
):
    """
    Create a new API key.
    """
    api_key, one_time_secret = APIKey.create(current_user.user_id, key_args)
    try:
        db.add(api_key)
        await db.commit()
        await db.refresh(api_key)
    except IntegrityError as exc:
        if "unique constraint" in str(exc):
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="An API key already exists with this name",
            )
        raise
    result = APIKeyCreationResponse.model_validate(api_key)
    result.secret_key = one_time_secret
    return result
