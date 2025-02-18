"""
ORM definitions for users.
"""

from typing import Self
from pydantic import BaseModel
from sqlalchemy import func, Column, String, DateTime, Double, Boolean, BigInteger
from sqlalchemy.orm import relationship, validates
from sqlalchemy.dialects.postgresql import JSONB
from api.database import Base
import hashlib
from api.util import gen_random_token
from api.user.util import validate_the_username
from api.permissions import Permissioning, Role


# Other fields are populated by listeners
# Except hotkey which is added in the header
# NOTE: Can we add hotkey here instead?
class UserRequest(BaseModel):
    username: str
    coldkey: str


class AdminUserRequest(BaseModel):
    username: str


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

    # Rate limit overrides.
    rate_limit_overrides = Column(JSONB, default=None)

    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now())

    chutes = relationship("Chute", back_populates="user")
    images = relationship("Image", back_populates="user")
    api_keys = relationship("APIKey", back_populates="user", cascade="all, delete-orphan")

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
