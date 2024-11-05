"""
ORM for API keys/scopes.
"""

from sqlalchemy import (
    Column,
    String,
    ForeignKey,
    DateTime,
    func,
    Enum,
    Boolean,
)
from sqlalchemy.orm import relationship
import secrets
from passlib.hash import argon2
import enum
from typing import List, Optional
from pydantic import BaseModel
from run_api.database import Base, generate_uuid


class Method(enum.Enum):
    CREATE = "create"
    READ = "read"
    WRITE = "write"
    DELETE = "delete"
    IVOKE = "invoke"


class APIKeyScope(Base):
    __tablename__ = "api_key_scopes"

    scope_id = Column(String, primary_key=True)
    api_key_id = Column(
        String, ForeignKey("api_keys.api_key_id", ondelete="CASCADE"), nullable=False
    )
    object_type = Column(String, nullable=False)
    object_id = Column(String)
    method = Column(Enum(Method))

    # Relationships
    api_key = relationship("APIKey", back_populates="scopes")


class ScopeArgs(BaseModel):
    object_type: str
    object_id: Optional[str] = None
    method: Optional[str] = None


class APIKey(Base):
    __tablename__ = "api_keys"

    api_key_id = Column(String, primary_key=True, default=generate_uuid)
    key_hash = Column(String, nullable=False)
    user_id = Column(
        String, ForeignKey("users.user_id", ondelete="CASCADE"), nullable=False
    )
    admin = Column(Boolean, default=True)
    name = Column(String, nullable=False)
    created_at = Column(DateTime, server_default=func.now())
    last_used_at = Column(DateTime, nullable=True)

    # Relationships
    user = relationship("User", back_populates="api_keys")
    scopes = relationship(
        "APIKeyScope",
        back_populates="api_key",
        cascade="all, delete-orphan",
        lazy="joined",
    )

    @classmethod
    def generate_key(cls):
        """
        Generate a new API key with format: prefix_base64chars
        """
        return f"cpk_{secrets.token_urlsafe(32)}"

    @classmethod
    def create(
        cls, name: str, user_id: int, admin: bool = False, scopes: List[ScopeArgs] = []
    ):
        """
        Helper to create a new API key with scopes.

        scopes format: [
            {"object_type": "chutes", "object_id": "000"},   # any action on chute.000
            {"object_type": "chutes", "method": Method.READ} # read any chute
        ]
        """
        # We need to return the plain text key when initially created.
        api_key = cls.generate_key()
        instance = cls(
            key_hash=argon2.hash(api_key),
            name=name,
            user_id=user_id,
            admin=admin,
        )
        if not admin:
            for scope in scopes:
                instance.scopes.append(
                    APIKeyScope(chute_id=scope.get("chute_id"), method=scope["method"])
                )
        return instance, api_key

    def verify_key(self, key: str) -> bool:
        """
        Verify if provided key matches stored key
        """
        if not key.startswith(self.prefix) or len(key) != 36:
            return False
        return argon2.verify(key, self.key_hash)

    def has_access(self, object_type: str, object_id: str, method: str) -> bool:
        """
        Check if the user's API key has access to the specified thing.
        """
        if self.admin:
            return True
        for scope in self.scopes:
            if scope.object_type != object_type:
                continue
            if scope.object_id in (None, object_id) and (
                not scope.method or scope.method == method
            ):
                return True
        return False
