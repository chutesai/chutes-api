"""
Routes for chutes.
"""

import uuid
import asyncpg
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import or_, func
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from typing import Optional
from api.chute.schemas import Chute
from api.secret.schemas import Secret, SecretArgs
from api.secret.response import SecretResponse
from api.user.schemas import User
from api.user.service import get_current_user
from api.payment.util import encrypt_secret

# XXX from api.instance.util import discover_chute_targets
from api.database import get_db_session
from api.pagination import PaginatedResponse

router = APIRouter()


@router.get("/", response_model=PaginatedResponse)
async def list_secrets(
    page: Optional[int] = 0,
    limit: Optional[int] = 25,
    db: AsyncSession = Depends(get_db_session),
    current_user: User = Depends(get_current_user(purpose="secrets")),
):
    """
    List secrets.
    """
    query = select(Secret).where(Secret.user_id == current_user.user_id)
    total_query = select(func.count()).select_from(query.subquery())
    total_result = await db.execute(total_query)
    total = total_result.scalar() or 0
    query = (
        query.order_by(Secret.created_at.desc())
        .offset((page or 0) * (limit or 25))
        .limit((limit or 25))
    )
    result = await db.execute(query)
    return {
        "total": total,
        "page": page,
        "limit": limit,
        "items": [SecretResponse.from_orm(item) for item in result.scalars().all()],
    }


@router.get("/{secret_id}", response_model=SecretResponse)
async def get_secret(
    secret_id: str,
    db: AsyncSession = Depends(get_db_session),
    current_user: User = Depends(get_current_user(purpose="secrets")),
):
    """
    Load a single secret by ID.
    """
    secret = (
        (
            await db.execute(
                select(Secret).where(
                    Secret.user_id == current_user.user_id, Secret.secret_id == secret_id
                )
            )
        )
        .unique()
        .scalar_one_or_none()
    )
    if not secret:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Secret not found, or does not belong to you",
        )
    return secret


@router.delete("/{secret_id}")
async def delete_secret(
    secret_id: str,
    db: AsyncSession = Depends(get_db_session),
    current_user: User = Depends(get_current_user(purpose="secrets")),
):
    """
    Delete a secret by ID.
    """
    secret = (
        (
            await db.execute(
                select(Secret).where(
                    Secret.user_id == current_user.user_id, Secret.secret_id == secret_id
                )
            )
        )
        .unique()
        .scalar_one_or_none()
    )
    if not secret:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Secret not found, or does not belong to you",
        )
    await db.delete(secret)
    await db.commit()
    return {"status": f"Successfully deleted secret {secret_id}"}


@router.post("/")
async def create_secret(
    args: SecretArgs,
    db: AsyncSession = Depends(get_db_session),
    current_user: User = Depends(get_current_user()),
):
    """
    Create a secret (e.g. private HF token).
    """
    # The secret may be associated with a chute (for now, must be).
    chute = (
        (
            await db.execute(
                select(Chute).where(
                    or_(
                        Chute.name.ilike(args.purpose),
                        Chute.chute_id == args.purpose,
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
            detail=f"Did not find target chute {str(args.purpose)}, or it does not belong to you",
        )
    encrypted_value = await encrypt_secret(args.value)
    secret = Secret(
        secret_id=str(
            uuid.uuid5(uuid.NAMESPACE_OID, f"{current_user.user_id}:{chute.chute_id}:{args.key}")
        ),
        purpose=chute.chute_id,
        key=args.key,
        value=encrypted_value,
        user_id=current_user.user_id,
    )
    secret.created_at = func.now()
    db.add(secret)
    try:
        await db.commit()
    except (IntegrityError, asyncpg.exceptions.UniqueViolationError):
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="A secret already exists for this purpose with this key.",
        )
    await db.refresh(secret)
    return secret
