"""
CRUD endpoints for user model aliases.
"""

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select, delete, func
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import AsyncSession
from api.database import get_db_session
from api.user.schemas import User
from api.user.service import get_current_user
from api.chute.util import get_one
from api.config import settings
from api.model_alias.schemas import ModelAlias, ModelAliasCreate, ModelAliasResponse

router = APIRouter()


@router.get("/", response_model=list[ModelAliasResponse])
async def list_aliases(
    db: AsyncSession = Depends(get_db_session),
    current_user: User = Depends(get_current_user()),
):
    result = await db.execute(select(ModelAlias).where(ModelAlias.user_id == current_user.user_id))
    return result.scalars().all()


@router.post("/", response_model=ModelAliasResponse, status_code=status.HTTP_201_CREATED)
async def create_or_update_alias(
    body: ModelAliasCreate,
    db: AsyncSession = Depends(get_db_session),
    current_user: User = Depends(get_current_user()),
):
    # Validate all chute_ids exist.
    for cid in body.chute_ids:
        chute = await get_one(cid)
        if chute is None:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"chute not found: {cid}",
            )

    stmt = (
        insert(ModelAlias)
        .values(
            user_id=current_user.user_id,
            alias=body.alias,
            chute_ids=body.chute_ids,
        )
        .on_conflict_do_update(
            index_elements=["user_id", "alias"],
            set_={"chute_ids": body.chute_ids, "updated_at": func.now()},
        )
    )
    await db.execute(stmt)
    await db.commit()

    # Invalidate cache.
    cache_key = f"malias:v2:{current_user.user_id}:{body.alias.lower()}"
    await settings.redis_client.delete(cache_key)

    # Re-fetch the row.
    result = await db.execute(
        select(ModelAlias).where(
            ModelAlias.user_id == current_user.user_id,
            ModelAlias.alias == body.alias,
        )
    )
    return result.scalar_one()


@router.delete("/{alias}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_alias(
    alias: str,
    db: AsyncSession = Depends(get_db_session),
    current_user: User = Depends(get_current_user()),
):
    result = await db.execute(
        delete(ModelAlias).where(
            ModelAlias.user_id == current_user.user_id,
            func.lower(ModelAlias.alias) == alias.lower(),
        )
    )
    if result.rowcount == 0:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="alias not found",
        )
    await db.commit()

    cache_key = f"malias:v2:{current_user.user_id}:{alias.lower()}"
    await settings.redis_client.delete(cache_key)
