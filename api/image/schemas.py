"""
ORM definitions for images.
"""

import re
from sqlalchemy.sql import func
from sqlalchemy.orm import relationship, validates
from sqlalchemy import (
    Column,
    String,
    DateTime,
    Boolean,
    Index,
    ForeignKey,
    UniqueConstraint,
)
from api.database import Base


class Image(Base):
    __tablename__ = "images"
    image_id = Column(String, primary_key=True, default="replaceme")
    user_id = Column(String, ForeignKey("users.user_id"), nullable=False)
    name = Column(String, nullable=False)
    tag = Column(String, nullable=False)
    readme = Column(String, nullable=True, default="")
    logo_id = Column(String, ForeignKey("logos.logo_id", ondelete="SET NULL"), nullable=True)
    public = Column(Boolean, default=False)
    status = Column(String, default="pending build")
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    chutes_version = Column(String, nullable=True)
    build_started_at = Column(DateTime(timezone=True))
    build_completed_at = Column(DateTime(timezone=True))

    chutes = relationship("Chute", back_populates="image")
    logo = relationship("Logo", back_populates="images", lazy="joined")
    user = relationship("User", back_populates="images", lazy="joined")
    fs_challenges = relationship(
        "FSChallenge", back_populates="image", cascade="all, delete-orphan"
    )

    __table_args__ = (
        Index("idx_image_name_tag", "name", "tag"),
        Index("idx_name_public", "name", "public"),
        Index("idx_name_created_at", "name", "created_at"),
        UniqueConstraint("user_id", "name", "tag", name="constraint_user_id_image_name_tag"),
    )

    @validates("name")
    def validate_name(self, _, name):
        """
        Basic validation on image name.
        """
        if not isinstance(name, str) or not re.match(r"^[a-z0-9][a-z0-9_\.\/-]{2,64}$", name, re.I):
            raise ValueError(f"Invalid image name: {name}")
        return name

    @validates("tag")
    def validate_tag(self, _, tag):
        """
        Basic validation on image tag.
        """
        if not isinstance(tag, str) or not re.match(r"^[a-z0-9][a-z0-9_\.\/-]{1,32}$", tag, re.I):
            raise ValueError(f"Invalid image tag: {tag}")
        return tag
