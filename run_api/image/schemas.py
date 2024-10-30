"""
ORM definitions for images.
"""

from sqlalchemy.sql import func
from sqlalchemy.orm import relationship
from sqlalchemy import (
    Column,
    String,
    DateTime,
    Boolean,
    Index,
    ForeignKey,
    UniqueConstraint,
)
from run_api.database import Base


class Image(Base):
    __tablename__ = "images"
    image_id = Column(String, primary_key=True, default="replaceme")
    user_id = Column(String, ForeignKey("users.user_id"))
    name = Column(String)
    tag = Column(String)
    public = Column(Boolean, default=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    chutes = relationship("Chute", back_populates="image")
    user = relationship("User", back_populates="images", lazy="joined")

    __table_args__ = (
        Index("idx_image_name_tag", "name", "tag"),
        Index("idx_name_public", "name", "public"),
        Index("idx_name_created_at", "name", "created_at"),
        UniqueConstraint("name", "tag", name="constraint_image_name_tag"),
    )
