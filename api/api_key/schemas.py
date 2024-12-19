"""
ORM for API keys/scopes.
"""

import re
import string
from sqlalchemy import (
    Column,
    String,
    ForeignKey,
    DateTime,
    func,
    Enum,
    Boolean,
    UniqueConstraint,
)
from sqlalchemy.orm import relationship, validates
import secrets
from passlib.hash import argon2
import enum
from typing import List, Optional, Self
from pydantic import BaseModel
from api.database import Base, generate_uuid


class Action(enum.Enum):
    READ = "read"
    WRITE = "write"
    DELETE = "delete"
    INVOKE = "invoke"


class ScopeArgs(BaseModel):
    object_type: str
    object_id: Optional[str] = None
    action: Optional[Action] = None


class APIKeyArgs(BaseModel):
    admin: bool
    name: str
    scopes: Optional[List[ScopeArgs]] = []


class APIKeyScope(Base):
    __tablename__ = "api_key_scopes"

    scope_id = Column(String, primary_key=True)
    api_key_id = Column(
        String, ForeignKey("api_keys.api_key_id", ondelete="CASCADE"), nullable=False
    )
    object_type = Column(String, nullable=False)
    object_id = Column(String)
    action = Column(Enum(Action))

    # Relationships
    api_key = relationship("APIKey", back_populates="scopes")

    @validates("object_type")
    def validate_object_type(_, __, type_):
        """
        Limit which types of objects we can manipulate with API keys.
        """
        if type_ not in ("images", "chutes", "invocations"):
            raise ValueError("Invalid object_type, must be one of images, chutes, invocations")
        return type_


class APIKey(Base):
    __tablename__ = "api_keys"

    api_key_id = Column(String, primary_key=True, default=generate_uuid)
    key_hash = Column(String, nullable=False)
    user_id = Column(
        String,
        ForeignKey("users.user_id", ondelete="CASCADE"),
        nullable=False,
    )
    admin = Column(Boolean, default=True)
    name = Column(String, nullable=False)
    created_at = Column(DateTime, server_default=func.now())
    last_used_at = Column(DateTime, nullable=True)

    # Relationships
    user = relationship("User", back_populates="api_keys", lazy="joined")
    scopes = relationship(
        "APIKeyScope",
        back_populates="api_key",
        cascade="all, delete-orphan",
        lazy="joined",
    )

    __table_args__ = (UniqueConstraint("user_id", "name", name="constraint_api_key_user_name"),)

    @validates("name")
    def validate_key_name(_, __, name):
        """
        Keep the API key names simple, please...
        """
        if not re.match(r"[a-zA-Z0-9][a-zA-Z0-9_ -]{2,31}$", name):
            raise ValueError(
                "API key name should start with alphanumeric character and contain up to 32 total allowed characters (letters, numbers, dashes, underscores, or spaces)"
            )
        return name

    @classmethod
    def generate_key(cls, user_id: str, api_key_id: str):
        """
        Generate a new API key with format: prefix_base64chars
        """
        secret = "".join(secrets.choice(string.ascii_letters + string.digits) for _ in range(32))
        return f"cpk_{user_id.replace('-', '')}.{api_key_id.replace('-', '')}.{secret}"

    @classmethod
    def create(cls, user_id: str, args: APIKeyArgs) -> tuple[Self, str]:
        """
        Helper to create a new API key with scopes.
        """
        # We need to return the plain text key when initially created.
        api_key_id = generate_uuid()
        secret = cls.generate_key(api_key_id, user_id)
        instance = cls(
            api_key_id=api_key_id,
            key_hash=argon2.hash(secret),
            name=args.name,
            user_id=user_id,
            admin=args.admin,
        )
        if not args.admin:
            for scope in args.scopes:
                instance.scopes.append(
                    APIKeyScope(
                        scope_id=generate_uuid(),
                        api_key_id=instance.api_key_id,
                        object_type=scope.object_type,
                        object_id=scope.object_id,
                        action=scope.action,
                    )
                )

        return instance, secret

    @staticmethod
    def could_be_valid(key: str) -> bool:
        """
        Fast check for token validity.
        """
        if (
            not key.startswith("cpk_")
            or len(key) != 102
            or not re.match(r"^cpk_[a-f0-9]{32}\.[a-f0-9]{32}\.[a-zA-Z0-9]{32}$", key)
        ):
            return False
        return True

    def verify(self, key: str) -> bool:
        """
        Verify if provided key matches stored key
        """
        if not self.could_be_valid(key):
            return False
        return argon2.verify(key, self.key_hash)

    def has_access(self, object_type: str, object_id: str, action: str) -> bool:
        """
        Check if the user's API key has access to the specified thing.
        """
        if self.admin:
            return True
        for scope in self.scopes:
            if scope.object_type != object_type:
                continue
            # Special handler for llm.chutes.ai endpoint.
            if (
                object_type == "chutes"
                and object_id == "__megallm__"
                and (not scope.action or scope.action.value == action)
            ):
                return True
            if scope.object_id in (None, object_id) and (
                not scope.action or scope.action.value == action
            ):
                return True
        return False
