"""
User routes.
"""

from pydantic import BaseModel
from datetime import datetime, timedelta, timezone
from fastapi import APIRouter, Depends, HTTPException, Header, status
from api.database import get_db_session
from api.user.schemas import UserRequest, User
from api.user.response import RegistrationResponse, SelfResponse
from api.user.service import get_current_user
from sqlalchemy import desc
from sqlalchemy.ext.asyncio import AsyncSession
from api.constants import HOTKEY_HEADER
from api.user.util import validate_the_username, generate_payment_address, refund_deposit
from api.payment.schemas import Payment
from sqlalchemy import select

router = APIRouter()


class ReturnDepositArgs(BaseModel):
    address: str


@router.get("/me", response_model=SelfResponse)
async def me(
    db: AsyncSession = Depends(get_db_session),
    current_user: User = Depends(get_current_user()),
):
    return current_user


# NOTE: Allow registertation without a hotkey and coldkey, for normal plebs?
@router.post(
    "/register",
    response_model=RegistrationResponse,
)
async def register(
    user_args: UserRequest,
    db: AsyncSession = Depends(get_db_session),
    current_user: User = Depends(get_current_user(raise_not_found=False)),
    hotkey: str = Header(..., description="The hotkey of the user", alias=HOTKEY_HEADER),
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
    existing_user = await db.execute(select(User).where(User.username == user_args.username))
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
    user.payment_address, user.wallet_secret = await generate_payment_address()
    user.developer_payment_address, user.developer_wallet_secret = await generate_payment_address()
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
        developer_payment_address=user.developer_payment_address,
        fingerprint=fingerprint,
    )


@router.post("/return_developer_deposit")
async def return_developer_deposit(
    args: ReturnDepositArgs,
    db: AsyncSession = Depends(get_db_session),
    current_user: User = Depends(get_current_user()),
):
    query = (
        select(Payment)
        .where(Payment.user_id == current_user.user_id)
        .order_by(desc(Payment.created_at))
        .limit(1)
    )
    recent_payment = (await db.execute(query)).scalar_one_or_none()
    if not recent_payment:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"You have not made any payments to the developer deposit address: {current_user.developer_deposit_address}",
        )
    if datetime.now(timezone.utc) - recent_payment.created_at <= timedelta(days=7):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"You must wait at least 7 days between payment and cancellation, most recent payment: {recent_payment.created_at}",
        )
    result, message = await refund_deposit(current_user.user_id, args.address)
    if not result:
        return HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=message,
        )
    return {
        "status": "transferred",
        "message": message,
    }
