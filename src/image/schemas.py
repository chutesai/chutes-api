"""
ORM definitions for tracking images.
"""
from sqlalchemy.sql import func
from sqlalchemy import Column, String, DateTime, Boolean, Index, ForeignKey
from database import Base


class Image(Base):
    __tablename__ = "images"
    image_id = Column(String, primary_key=True)
    user_id = Column(String, ForeignKey("users.user_id"))
    name = Column(String)
    tag = Column(String)
    public = Column(Boolean, default=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    __table_args__ = (
        Index("idx_image_name_tag", "name", "tag"),
        Index("idx_name_public", "name", "public"),
        Index("idx_name_created_at", "name", "created_at"),
    )
