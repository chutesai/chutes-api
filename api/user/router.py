"""
User routes.
"""

import secrets
from fastapi import APIRouter, Depends, HTTPException, Header, status
from api.database import get_db_session
from api.user.schemas import UserRequest, User, AdminUserRequest
from api.user.response import RegistrationResponse, SelfResponse
from api.user.service import get_current_user
from api.user.events import generate_uid as generate_user_uid
from sqlalchemy.ext.asyncio import AsyncSession
from api.constants import HOTKEY_HEADER
from api.permissions import Permissioning
from api.config import settings
from api.api_key.schemas import APIKey, APIKeyArgs
from api.api_key.response import APIKeyCreationResponse
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


async def _validate_username(db, username):
    """
    Check validity and availability of a username.
    """
    try:
        validate_the_username(username)
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        ) from e
    existing_user = await db.execute(select(User).where(User.username == username))
    if existing_user.first() is not None:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Username {username} already exists, sorry! Please choose another.",
        )


def _registration_response(user, fingerprint):
    """
    Generate a response for a newly registered user.
    """
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
    await _validate_username(db, user_args.username)

    # Create.
    user, fingerprint = User.create(
        username=user_args.username,
        coldkey=user_args.coldkey,
        hotkey=hotkey,
    )
    user.payment_address, user.wallet_secret = await generate_payment_address()
    user.developer_payment_address, user.developer_wallet_secret = await generate_payment_address()
    if settings.all_accounts_free:
        user.permissions_bitmask = 0
        Permissioning.enable(user, Permissioning.free_account)
    db.add(user)
    await db.commit()
    await db.refresh(user)
    return _registration_response(user, fingerprint)


@router.post(
    "/create_user",
    response_model=RegistrationResponse,
)
async def admin_create_user(
    user_args: AdminUserRequest,
    db: AsyncSession = Depends(get_db_session),
    current_user: User = Depends(get_current_user()),
):
    """
    Create a new user manually from an admin account, no bittensor stuff necessary.
    """
    if not current_user.has_role(Permissioning.create_user):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="This action can only be performed by user admin accounts.",
        )

    # Validate the username
    await _validate_username(db, user_args.username)

    # Create the user, faking the hotkey and using the payment address as the coldkey, since this
    # user is API/APP only and not really cognisant of bittensor.
    user, fingerprint = User.create(
        username=user_args.username,
        coldkey=secrets.token_hex(24),
        hotkey=secrets.token_hex(24),
    )
    generate_user_uid(None, None, user)
    user.payment_address, user.wallet_secret = await generate_payment_address()
    user.coldkey = user.payment_address
    user.developer_payment_address, user.developer_wallet_secret = await generate_payment_address()
    if settings.all_accounts_free:
        user.permissions_bitmask = 0
        Permissioning.enable(user, Permissioning.free_account)
    db.add(user)

    # Automatically create an API key for the user as well.
    api_key, one_time_secret = APIKey.create(user.user_id, APIKeyArgs(name="default", admin=True))
    db.add(api_key)
    await db.commit()
    await db.refresh(user)
    await db.refresh(api_key)
    key_response = APIKeyCreationResponse.model_validate(api_key)
    key_response.secret_key = one_time_secret
    response = _registration_response(user, fingerprint)
    response.api_key = key_response

    return response
