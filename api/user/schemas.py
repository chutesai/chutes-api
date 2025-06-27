"""
ORM definitions for users.
"""

import datetime
from typing import Self, Optional
from pydantic import BaseModel
from sqlalchemy import (
    func,
    Column,
    String,
    DateTime,
    Double,
    Boolean,
    BigInteger,
    ForeignKey,
    select,
    case,
)
from sqlalchemy.orm import relationship, validates
from api.database import Base
import hashlib
from api.database import get_session
from api.config import settings
from api.util import gen_random_token
from api.user.util import validate_the_username
from api.permissions import Permissioning, Role


# Other fields are populated by listeners
# Except hotkey which is added in the header
# NOTE: Can we add hotkey here instead?
class UserRequest(BaseModel):
    username: str
    coldkey: str
    logo_id: Optional[str] = None


class AdminUserRequest(BaseModel):
    username: str
    logo_id: Optional[str] = None


class User(Base):
    __tablename__ = "users"

    # Populated in user/events based on fingerprint_hash
    user_id = Column(String, primary_key=True)

    # An alternative to an API key.
    # Must be nullable for not all users have a hotkey, and the unique
    # constraint prevents us using a default hotkey.
    hotkey = Column(String, nullable=True, unique=True)

    # To receive commission payments
    coldkey = Column(String, nullable=False)

    # Users should send to this address to top up
    payment_address = Column(String)

    # Secret (encrypted)
    wallet_secret = Column(String)

    # Developer program/anti-turdnugget deposit address.
    developer_payment_address = Column(String)
    developer_wallet_secret = Column(String)

    # Balance in USD.
    balance = Column(Double, default=0.0)

    # Flag indicating if the first payment bonus has been credited.
    bonus_used = Column(Boolean, default=False)

    # Friendly name for the frontend for chute creators
    username = Column(String, unique=True)

    # Gets populated in user/events.py to be a 16 digit alphanumeric which acts as an account id
    fingerprint_hash = Column(String, nullable=False, unique=True)

    # Extra permissions/roles bitmask.
    permissions_bitmask = Column(BigInteger, default=0)

    # Validator association (for free accounts).
    validator_hotkey = Column(String, nullable=True)

    # Subnet owner association (for free accounts).
    subnet_owner_hotkey = Column(String, nullable=True)

    # Squad enabled.
    squad_enabled = Column(Boolean, default=False)

    # Job limits.
    job_limits = Column(JSONB, default=None)

    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now())

    # Logo/avatar.
    logo_id = Column(String, ForeignKey("logos.logo_id", ondelete="SET NULL"), nullable=True)

    chutes = relationship("Chute", back_populates="user")
    images = relationship("Image", back_populates="user")
    api_keys = relationship("APIKey", back_populates="user", cascade="all, delete-orphan")
    jobs = relationship("Job", back_populates="user")

    @validates("username")
    def validate_username(self, _, value):
        """
        Simple username validation.
        """
        return validate_the_username(value)

    @classmethod
    def create(
        cls, username: str, coldkey: str | None = None, hotkey: str | None = None
    ) -> tuple[Self, str]:
        """
        Create a new user.
        """
        fingerprint = gen_random_token(k=32)
        fingerprint_hash = hashlib.blake2b(fingerprint.encode()).hexdigest()
        user = cls(
            username=username,
            coldkey=coldkey,
            hotkey=hotkey,
            fingerprint_hash=fingerprint_hash,
        )
        return user, fingerprint

    def __repr__(self):
        """
        String representation.
        """
        return f"<User(user_id={self.user_id}, username={self.username})>"

    def has_role(self, role: Role):
        """
        Check if a user has a role/permission.
        """
        return Permissioning.enabled(self, role)


class InvocationQuota(Base):
    __tablename__ = "invocation_quotas"
    user_id = Column(String, primary_key=True)
    chute_id = Column(String, primary_key=True)
    quota = Column(BigInteger, nullable=False, default=settings.default_quotas.get("*", 200))

    @staticmethod
    async def get(user_id: str, chute_id: str):
        key = f"quota:{user_id}:{chute_id}".encode()
        cached = (await settings.memcache.get(key) or b"").decode()
        if cached and cached.isdigit():
            return int(cached)
        async with get_session() as session:
            result = await session.execute(
                select(InvocationQuota.quota)
                .where(InvocationQuota.user_id == user_id)
                .where(InvocationQuota.chute_id.in_([chute_id, "*"]))
                .order_by(case((InvocationQuota.chute_id == chute_id, 0), else_=1))
                .limit(1)
            )
            quota = result.scalar()
            if quota is not None:
                await settings.memcache.set(key, str(quota).encode(), exptime=3600)
                return quota
            default_quota = settings.default_quotas.get(
                chute_id, settings.default_quotas.get("*", 200)
            )
            await settings.memcache.set(key, str(default_quota).encode(), exptime=3600)
            return default_quota

    @staticmethod
    async def quota_key(user_id: str, chute_id: str):
        """
        Get the quota (redis) key for a user and chute.
        """
        date = datetime.datetime.now().strftime("%Y%m%d")
        cache_key = f"quota_type:{user_id}:{chute_id}".encode()
        cached = await settings.memcache.get(cache_key)
        if cached is not None:
            quota_type = cached.decode()
        else:
            async with get_session() as session:
                result = await session.execute(
                    select(InvocationQuota.chute_id)
                    .where(InvocationQuota.user_id == user_id)
                    .where(InvocationQuota.chute_id.in_([chute_id, "*"]))
                    .order_by(case((InvocationQuota.chute_id == chute_id, 0), else_=1))
                    .limit(1)
                )
                db_chute_id = result.scalar()
            if db_chute_id == chute_id:
                quota_type = "specific"
            elif db_chute_id == "*":
                quota_type = "wildcard"
            elif chute_id in settings.default_quotas:
                quota_type = "default_specific"
            elif "*" in settings.default_quotas:
                quota_type = "default_wildcard"
            else:
                quota_type = "none"
            await settings.memcache.set(cache_key, quota_type.encode(), exptime=3600)

        if quota_type in ["specific", "default_specific"]:
            return f"q:{date}:{user_id}:{chute_id}"
        else:
            return f"q:{date}:{user_id}:__default__"
