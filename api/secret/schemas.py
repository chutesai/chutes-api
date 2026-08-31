"""
ORM definitions for secrets.
"""

from sqlalchemy.sql import func
from sqlalchemy import (
    CheckConstraint,
    Column,
    String,
    DateTime,
    ForeignKey,
)
from sqlalchemy.orm import relationship
from api.database import Base
from pydantic import BaseModel


class Secret(Base):
    __tablename__ = "secrets"
    __table_args__ = (
        CheckConstraint(
            "kind IN ('chute', 'external_backend')",
            name="ck_secrets_kind",
        ),
    )

    secret_id = Column(String, primary_key=True)
    user_id = Column(String, ForeignKey("users.user_id"), nullable=False)
    # This discriminator is a security boundary: only ``chute`` secrets may be
    # delivered to untrusted hosted instances. External-backend credentials are
    # consumed exclusively by the control-plane relay.
    kind = Column(String(32), nullable=False, default="chute", server_default="chute")
    purpose = Column(String, nullable=True)
    key = Column(String, nullable=False)
    value = Column(String, nullable=False)
    created_at = Column(DateTime, server_default=func.now())

    user = relationship("User", back_populates="secrets", lazy="select")


class SecretArgs(BaseModel):
    purpose: str
    key: str
    value: str
