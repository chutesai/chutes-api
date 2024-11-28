"""
ORM definitions for logos.
"""

from sqlalchemy.sql import func
from sqlalchemy.orm import relationship
from sqlalchemy import (
    Column,
    String,
    DateTime,
    ForeignKey,
)
from api.database import Base, generate_uuid


class Logo(Base):
    __tablename__ = "logos"
    logo_id = Column(String, primary_key=True, default=generate_uuid)
    path = Column(String, nullable=False)
    user_id = Column(String, ForeignKey("users.user_id", ondelete="CASCADE"), nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    chutes = relationship("Chute", back_populates="logo")
    images = relationship("Image", back_populates="logo")
