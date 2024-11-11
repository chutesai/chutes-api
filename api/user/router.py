"""
User routes.
"""

from fastapi import APIRouter, Depends, HTTPException, Header, status, Request
from api.database import get_db_session
from api.user.schemas import UserRequest, User
from api.user.response import RegistrationResponse
from api.user.service import get_current_user
from sqlalchemy.ext.asyncio import AsyncSession
from api.constants import HOTKEY_HEADER
from api.user.util import validate_the_username
from sqlalchemy import select

router = APIRouter()


# NOTE: Allow registertation without a hotkey and coldkey, for normal plebs?
@router.post(
    "/register",
    response_model=RegistrationResponse,
)
async def register(
    user_args: UserRequest,
    db: AsyncSession = Depends(get_db_session),
    current_user: User = Depends(get_current_user(raise_not_found=False)),
    hotkey: str = Header(
        ..., description="The hotkey of the user", alias=HOTKEY_HEADER
    ),
):
    """
    Register a user.
    """
    if current_user:
        # NOTE: Change when we allow register without a hotkey
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="This hotkey is already registered to a user!", 
        )

    # Validate the username
    try:
        validate_the_username(user_args.username)
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        ) from e

    # Check if the username exists already.
    existing_user = await db.execute(
        select(User).where(User.username == user_args.username)
    )
    if existing_user.first() is not None:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Username {user_args.username} already exists, sorry! Please choose another.",
        )
    user, fingerprint = User.create(
        username=user_args.username,
        coldkey=user_args.coldkey,
        hotkey=hotkey,
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
