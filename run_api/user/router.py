"""
User routes.
"""

from fastapi import APIRouter, Depends, HTTPException, status, Request
from run_api.database import get_db_session
from run_api.user.schemas import UserRequest, User
from run_api.user.response import RegistrationResponse
from run_api.user.service import get_current_user
from sqlalchemy.ext.asyncio import AsyncSession


router = APIRouter()


@router.post("/register", response_model=RegistrationResponse)
async def register(
    request: Request,
    user_args: UserRequest,
    db: AsyncSession = Depends(get_db_session),
    current_user: User = Depends(get_current_user(raise_not_found=False)),
):
    """
    Register a user.
    """
    if current_user:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="User with the provided hotkey has already registered!",
        )
    user, fingerprint = User.create(
        username=user_args.username,
        coldkey=user_args.coldkey,
        hotkey=request.headers["X-Parachutes-Hotkey"],
    )
    db.add(user)
    await db.commit()
    await db.refresh(user)
    return RegistrationResponse(
        username=user.username,
        user_id=user.user_id,
        created_at=user.created_at,
        hotkey=user.hotkey,
        coldkey=user.coldkey,
        payment_address=user.payment_address,
        fingerprint=fingerprint,
    )
