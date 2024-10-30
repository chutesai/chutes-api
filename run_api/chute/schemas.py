"""
ORM definitions for Chutes.
"""

import re
import uuid
from sqlalchemy.sql import func
from sqlalchemy.orm import relationship, validates
from sqlalchemy import event, Column, String, DateTime, Boolean, ForeignKey
from sqlalchemy.dialects.postgresql import JSONB
from run_api.database import Base


class Chute(Base):
    __tablename__ = "chutes"
    chute_id = Column(String, primary_key=True, default="replaceme")
    user_id = Column(String, ForeignKey("users.user_id"), nullable=False)
    name = Column(String)
    image_id = Column(String, ForeignKey("images.image_id"))
    public = Column(Boolean, default=False)
    cords = Column(JSONB, nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now())

    image = relationship("Image", back_populates="chutes", lazy="joined")
    user = relationship("User", back_populates="chutes", lazy="joined")

    @validates("cords")
    def validate_cords(self, _, cords):
        """
        Basic validation on the cords, i.e. methods that can be called for this chute.
        """
        if not cords or not isinstance(cords, list):
            raise ValueError(
                "Must include between 1 and 25 valid cords to create a chute"
            )
        for cord in cords:
            path = cord.get("path")
            if not isinstance(path, str) or not re.match(
                r"^(/[a-z][a-z0-9_]*)+$", path, re.I
            ):
                raise ValueError("Invalid cord path: {path}")
            stream = cord.get("stream")
            if stream not in (None, True, False):
                raise ValueError(f"Invalid cord stream value: {stream}")
        if set(cord) - {"path", "stream"}:
            raise ValueError("Extraneous parameters passed to cord")
        return cord


@event.listens_for(Chute, "before_insert")
def generate_uid(_, __, chute):
    """
    Set the chute_id deterministically.
    """
    chute.chute_id = str(
        uuid.uuid5(uuid.NAMESPACE_OID, f"{chute.user_id}::chute::{chute.name}")
    )
