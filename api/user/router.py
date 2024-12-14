"""
User routes.
"""

from fastapi import APIRouter, Depends, HTTPException, Header, status
from api.database import get_db_session
from api.user.schemas import UserRequest, User
from api.user.response import RegistrationResponse, SelfResponse
from api.user.service import get_current_user
from sqlalchemy.ext.asyncio import AsyncSession
from api.constants import HOTKEY_HEADER
from api.permissions import Permissioning
from api.config import settings
from api.user.util import validate_the_username, generate_payment_address
from substrateinterface import Keypair, KeypairType
from sqlalchemy import select

router = APIRouter()


async def _link_hotkey(
    hotkey: str,
    signature: str,
    attribute: str,
    allow_list: list[str],
    db: AsyncSession,
    current_user: User,
):
    """
    Link a validator or subnet owner hotkey to this account, allowing free usage and developer access.
    """
    signature_string = f"{hotkey}:{current_user.username}"
    if hotkey in allow_list:
        if Keypair(ss58_address=hotkey, crypto_type=KeypairType.SR25519).verify(
            signature_string, bytes.fromhex(signature)
        ):
            # Any other accounts already associated?
            existing = (
                await db.execute(
                    select(User)
                    .where(getattr(User, attribute) == hotkey)
                    .where(User.user_id != current_user.user_id)
                )
            ).scalar_one_or_none()
            if existing:
                raise HTTPException(
                    status_code=status.HTTP_409_CONFLICT,
                    detail=f"Hotkey already associated with user {existing.username}",
                )

            # Reload the user since current_user isn't bound to a session, then update.
            user = (
                await db.execute(select(User).where(User.user_id == current_user.user_id))
            ).scalar_one_or_none()
            setattr(user, attribute, hotkey)
            Permissioning.enable(user, Permissioning.free_account)
            Permissioning.enable(user, Permissioning.developer)
            await db.commit()
            await db.refresh(user)
            return user
    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Invalid hotkey or signature.",
    )


@router.get("/me", response_model=SelfResponse)
async def me(
    db: AsyncSession = Depends(get_db_session),
    current_user: User = Depends(get_current_user()),
):
    """
    Get a detailed response for the current user.
    """
    return current_user


@router.get("/link_validator", response_model=SelfResponse)
async def link_validator(
    hotkey: str,
    signature: str,
    db: AsyncSession = Depends(get_db_session),
    current_user: User = Depends(get_current_user(purpose="link_account")),
):
    """
    Link a validator hotkey to this account, allowing free usage.
    """
    return await _link_hotkey(
        hotkey, signature, "validator_hotkey", settings.validators, db, current_user
    )


@router.get("/link_subnet_owner", response_model=SelfResponse)
async def link_subnet_owner(
    hotkey: str,
    signature: str,
    db: AsyncSession = Depends(get_db_session),
    current_user: User = Depends(get_current_user(purpose="link_account")),
):
    """
    Link a subnet owner hotkey to this account, allowing free usage.
    """
    return await _link_hotkey(
        hotkey, signature, "subnet_owner_hotkey", settings.subnet_owners, db, current_user
    )


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
    if settings.all_accounts_free:
        Permissioning.enable(user, Permissioning.free_account)
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
