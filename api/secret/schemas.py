"""
ORM definitions for secrets.
"""

from sqlalchemy.sql import func
from sqlalchemy import (
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
    secret_id = Column(String, primary_key=True)
    user_id = Column(String, ForeignKey("users.user_id"), nullable=False)
    purpose = Column(String, nullable=True)
    key = Column(String, nullable=False)
    value = Column(String, nullable=False)
    created_at = Column(DateTime, server_default=func.now())

    user = relationship("User", back_populates="secrets", lazy="select")


class SecretArgs(BaseModel):
    purpose: str
    key: str
    value: str
