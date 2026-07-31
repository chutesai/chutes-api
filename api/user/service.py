"""
User logic/code.
"""

from typing import Optional
from dataclasses import dataclass, field
from loguru import logger
from sqlalchemy import exists, or_, delete, text
from sqlalchemy.future import select
from fastapi import APIRouter, Depends, Header, Request, HTTPException, Security, status
from bittensor_wallet.keypair import Keypair
from api.config import settings
from api.metagraph import MetagraphNode
from api.database import get_session
from api.user.schemas import User
from api.chute.schemas import Chute
from api.image.schemas import Image
from api.secret.schemas import Secret
from api.instance.schemas import Instance
from api.api_key.schemas import APIKey
from api.api_key.util import get_and_check_api_key
from api.user.tokens import get_user_from_token
from fastapi.security import APIKeyHeader
from api.constants import HOTKEY_HEADER, SIGNATURE_HEADER, AUTHORIZATION_HEADER
from api.constants import NONCE_HEADER, INTEGRATED_SUBNETS
from api.util import nonce_is_valid, get_signing_message
from api.permissions import Permissioning, Role
from sqlalchemy.ext.asyncio import AsyncSession

router = APIRouter()
api_key_header = APIKeyHeader(name="Authorization", auto_error=False)


@dataclass
class ChuteRef:
    chute_id: str
    name: Optional[str] = None
    version: Optional[str] = None


@dataclass
class ImageRef:
    image_id: str
    name: str
    tag: str


@dataclass
class SecretRef:
    secret_id: str
    key: str


@dataclass
class UserResources:
    """
    A user's resources whose ``NO ACTION`` foreign keys block a hard delete: chutes, images
    and secrets. Internal domain object -- everything else CASCADEs or has no FK to ``users``.
    """

    chutes: list[ChuteRef] = field(default_factory=list)
    images: list[ImageRef] = field(default_factory=list)
    secrets: list[SecretRef] = field(default_factory=list)

    @property
    def is_empty(self) -> bool:
        return not (self.chutes or self.images or self.secrets)


@dataclass
class UserDeletionResult:
    """
    Side effects the caller must run AFTER committing a user hard-delete: they touch Redis
    and external caches, so they're only safe once the delete is durable.
    """

    terminated_instances: list[tuple[str, str]] = field(default_factory=list)  # (instance, chute)
    broadcasts: list[dict] = field(default_factory=list)  # miner_broadcast payloads
    api_key_ids: list[str] = field(default_factory=list)
    chutes_deleted: int = 0
    images_deleted: int = 0
    secrets_deleted: int = 0


def get_current_user(
    purpose: str = None,
    registered_to: int = None,
    raise_not_found: bool = True,
    allow_api_key=False,
):
    """
    Authentication dependency builder.
    """

    async def _authenticate(
        request: Request,
        api_key: Optional[str] = Security(api_key_header),
        hotkey: str | None = Header(None, alias=HOTKEY_HEADER),
        signature: str | None = Header(None, alias=SIGNATURE_HEADER),
        nonce: str | None = Header(None, alias=NONCE_HEADER),
        authorization: str | None = Header(None, alias=AUTHORIZATION_HEADER),
    ):
        """
        Helper to authenticate requests.
        """

        if (hotkey or signature or nonce) and (not hotkey or not signature or not nonce):
            hotkey, signature, nonce = None, None, None
        use_hotkey_auth = registered_to is not None or (hotkey and signature)
        if registered_to is not None and raise_not_found:
            if not hotkey or not signature or not nonce:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Invalid BT Auth.",
                )

        # If not using hotkey auth, then just use the API key
        if not use_hotkey_auth:
            # API key validation.
            user = None
            if authorization:
                token = authorization.split(" ")[-1]

                # JWT auth (standard JWTs, not OAuth access tokens).
                if (
                    token
                    and authorization.lower().lstrip().startswith("bearer ")
                    and not token.strip().startswith("cpk_")
                    and not token.strip().startswith("cak_")
                ):
                    user = await get_user_from_token(token, request)

                # API key auth (supports both cpk_ API keys and cak_ OAuth tokens).
                if not user and token:
                    api_key = await get_and_check_api_key(token, request)
                    if api_key:
                        request.state.api_key = api_key
                        user = api_key.user
            if user:
                return user
            if raise_not_found:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Can't find a user with that api key in our db :(",
                )
            return None

        # Otherwise we are using hotkey auth, so need to check the nonce
        # and check the message was signed correctly
        if not nonce_is_valid(nonce):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid nonce!",
            )

        # Custom whitelist from potentially compromised hotkey.
        # Request message:
        #   btcli w sign --message "whitelist IP 207.246.94.14"
        #   4063dc072f57f6ce77ad1208dc002373c7c491c5f3d75248a51b856b061f6838656a9052b0717bbed20e437cd6e168c784605cd9adb8fb2f90b9a1b25e94528a
        #   My cold key is 5C5zpdLSSxFeFkLFw9tAc7DdxdK82GCAjnoe5pub73GMvKLt
        # miner hotkey 5FhMaRd59y5nyDEtCz1JMMEMZzAGimtmC8m5AfCeXVE3vzCx
        if purpose not in ("sockets", "registry"):
            client_ip = request.state.client_ip
            if hotkey == "5FhMaRd59y5nyDEtCz1JMMEMZzAGimtmC8m5AfCeXVE3vzCx":
                if client_ip != "207.246.94.14":
                    raise HTTPException(
                        status_code=status.HTTP_401_UNAUTHORIZED,
                        detail=f"Unauthorized IP address: {client_ip=}",
                    )

        # Now get the Signing message
        body_sha256 = getattr(request.state, "body_sha256", None)

        signing_message = get_signing_message(
            hotkey=hotkey,
            nonce=nonce,
            payload_hash=body_sha256,
            purpose=purpose,
            payload_str=None,
        )

        if not signing_message:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail=f"Bad signing message: {signing_message}",
            )

        # Verify the signature
        try:
            signature_hex = bytes.fromhex(signature)
        except ValueError as e:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail=f"Invalid signature: {signature}, with error: {e}",
            )
        try:
            keypair = Keypair(hotkey)
            if not keypair.verify(signing_message, signature_hex):
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail=f"Invalid request signature for hotkey {hotkey}. Message: {signing_message}",
                )
        except ValueError as e:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail=f"Invalid request signature for hotkey {hotkey}. Message: {signing_message}",
            ) from e

        # Requires a hotkey registered to a netuid?
        if registered_to is not None:
            async with get_session(readonly=True) as session:
                if not (
                    await session.execute(
                        select(
                            exists()
                            .where(MetagraphNode.hotkey == hotkey)
                            .where(MetagraphNode.netuid == registered_to)
                        )
                    )
                ).scalar():
                    raise HTTPException(
                        status_code=status.HTTP_401_UNAUTHORIZED,
                        detail=f"Hotkey is not registered on netuid {settings.netuid}",
                    )

        # Fetch the actual user.
        # NOTE: We should have a standard way to get this session
        async with get_session(readonly=True) as session:
            session: AsyncSession  # For nice type hinting for IDE's
            result = await session.execute(select(User).where(User.hotkey == hotkey))

            user = result.scalar_one_or_none()
            if not user and raise_not_found and not registered_to:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail=f"Could not find user with hotkey: {hotkey}",
                )
            return user

    return _authenticate


async def resolve_user(session: AsyncSession, identifier: str) -> Optional[User]:
    """Resolve a user by ``user_id`` or ``username``; return ``None`` if there's no match.

    A pure lookup with no HTTP concerns -- the caller decides what a missing user means.
    """
    return (
        (
            await session.execute(
                select(User).where(or_(User.username == identifier, User.user_id == identifier))
            )
        )
        .unique()
        .scalar_one_or_none()
    )


async def get_user_resources(session: AsyncSession, user_id: str) -> UserResources:
    """Return the user-owned resources whose ``NO ACTION`` FKs block a hard delete.

    These (chutes, images, secrets) must be torn down before the ``users`` row can be
    deleted; everything else either CASCADEs or has no FK to ``users``.
    """
    chutes = (
        await session.execute(
            select(Chute.chute_id, Chute.name, Chute.version).where(Chute.user_id == user_id)
        )
    ).all()
    images = (
        await session.execute(
            select(Image.image_id, Image.name, Image.tag).where(Image.user_id == user_id)
        )
    ).all()
    secrets = (
        await session.execute(select(Secret.secret_id, Secret.key).where(Secret.user_id == user_id))
    ).all()
    return UserResources(
        chutes=[ChuteRef(chute_id=c.chute_id, name=c.name, version=c.version) for c in chutes],
        images=[ImageRef(image_id=i.image_id, name=i.name, tag=i.tag) for i in images],
        secrets=[SecretRef(secret_id=s.secret_id, key=s.key) for s in secrets],
    )


async def delete_user_and_resources(
    session: AsyncSession, user: User, resources: UserResources
) -> UserDeletionResult:
    """Hard-delete a user and their blocking resources within ``session`` (does NOT commit).

    Teardown order respects the ``NO ACTION`` FKs and the instance billing trigger:
    instances -> chutes -> images -> secrets -> user. Instances go first so the
    ``BEFORE DELETE`` billing trigger can still resolve the (not-yet-deleted) chute + user.
    Deleting the ``users`` row fires the ``on_user_delete`` trigger (archives to
    ``deleted_users``) and CASCADEs api_keys/jobs/logos/chute_shares/oauth_*.

    Returns the side effects the caller must run *after* commit (Redis broadcasts,
    conn-tracking cleanup, api-key cache invalidation).
    """
    # Capture api key ids before the users delete cascades them away.
    api_key_ids = list(
        (await session.execute(select(APIKey.api_key_id).where(APIKey.user_id == user.user_id)))
        .scalars()
        .all()
    )

    chute_ids = [c.chute_id for c in resources.chutes]
    terminated_instances = []
    if chute_ids:
        rows = (
            await session.execute(
                select(Instance.instance_id, Instance.chute_id).where(
                    Instance.chute_id.in_(chute_ids)
                )
            )
        ).all()
        terminated_instances = [(row.instance_id, row.chute_id) for row in rows]
        instance_ids = [instance_id for instance_id, _ in terminated_instances]
        if instance_ids:
            await session.execute(delete(Instance).where(Instance.instance_id.in_(instance_ids)))
            # instance_audit has no ORM model and is otherwise trigger-managed; the app only
            # owns valid_termination (so miners aren't penalized for the termination).
            await session.execute(
                text(
                    "UPDATE instance_audit SET valid_termination = true, "
                    "deletion_reason = 'account deleted' WHERE instance_id = ANY(:instance_ids)"
                ),
                {"instance_ids": instance_ids},
            )
            # One line per instance with its single id bound, so a search for that instance_id
            # matches (a list field would not) -- same for chutes/images below.
            for terminated_id, terminated_chute_id in terminated_instances:
                logger.bind(instance_id=terminated_id, chute_id=terminated_chute_id).info(
                    "terminated instance (account deleted)"
                )

    broadcasts = []
    if chute_ids:
        await session.execute(delete(Chute).where(Chute.chute_id.in_(chute_ids)))
        for c in resources.chutes:
            logger.bind(chute_id=c.chute_id).info("deleted chute (account deleted)")
        broadcasts.extend(
            {"reason": "chute_deleted", "data": {"chute_id": c.chute_id, "version": c.version}}
            for c in resources.chutes
        )
    image_ids = [i.image_id for i in resources.images]
    if image_ids:
        await session.execute(delete(Image).where(Image.image_id.in_(image_ids)))
        for i in resources.images:
            logger.bind(image_id=i.image_id).info("deleted image (account deleted)")
        broadcasts.extend(
            {"reason": "image_deleted", "data": {"image_id": i.image_id}} for i in resources.images
        )
    if resources.secrets:
        await session.execute(delete(Secret).where(Secret.user_id == user.user_id))

    # Finally the user row: fires on_user_delete (-> deleted_users) and CASCADEs the rest.
    await session.execute(delete(User).where(User.user_id == user.user_id))

    return UserDeletionResult(
        terminated_instances=terminated_instances,
        broadcasts=broadcasts,
        api_key_ids=api_key_ids,
        chutes_deleted=len(resources.chutes),
        images_deleted=len(resources.images),
        secrets_deleted=len(resources.secrets),
    )


def require_role(role: Role, purpose: str = None):
    """FastAPI dependency: authenticate the caller AND require that they hold ``role``.

    Declarative role enforcement — attach it to the endpoint signature (or a router's
    ``dependencies=``) so the required role is visible in the route and checked *before*
    the handler runs. Prefer this over an in-body ``current_user.has_role(...)`` check,
    which a new endpoint can silently omit and which is far harder to audit than a role
    gate that lives in the signature. Returns the authenticated ``User``.

    ``purpose`` is a pass-through to the **hotkey-signature** auth path only — it binds the
    signed message (``get_signing_message``). API-key / JWT / OAuth auth ignores it, so
    leave it ``None`` for endpoints reached with a bearer token.
    """
    authenticate = get_current_user(purpose=purpose)

    async def _dep(current_user: Optional[User] = Depends(authenticate)) -> User:
        if current_user is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Authentication required.",
            )
        if not current_user.has_role(role):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="You do not have permission to perform this action.",
            )
        return current_user

    return _dep


async def chutes_user_id():
    if (user_id := getattr(router, "_chutes_user_id", None)) is not None:
        return user_id
    async with get_session(readonly=True) as session:
        router._chutes_user_id = (
            (await session.execute(select(User.user_id).where(User.username == "chutes")))
            .unique()
            .scalar_one_or_none()
        )
    return router._chutes_user_id


async def chutes_user():
    if (user := getattr(router, "_chutes_user", None)) is not None:
        return user
    async with get_session(readonly=True) as session:
        router._chutes_user = (
            (await session.execute(select(User).where(User.username == "chutes")))
            .unique()
            .scalar_one_or_none()
        )
    return router._chutes_user


def subnet_role_accessible(chute, user, admin: bool = False):
    netuid = None
    for subnet, info in INTEGRATED_SUBNETS.items():
        if info["model_substring"] in chute.name.lower():
            netuid = info["netuid"]
            break
    if not netuid:
        return False
    perms = [Permissioning.subnet_admin]
    if not admin:
        perms.append(Permissioning.subnet_invoke)
    return user.netuids and netuid in user.netuids and any(user.has_role(perm) for perm in perms)


async def bt_user_exists(session, hotkey: str) -> bool:
    user = (
        (await session.execute(select(User).where(User.hotkey == hotkey)))
        .unique()
        .scalar_one_or_none()
    )
    return user is not None
