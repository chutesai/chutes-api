"""
ORM definitions for Chutes.
"""

import re
from sqlalchemy.sql import func
from sqlalchemy.orm import relationship, validates
from sqlalchemy import Column, String, DateTime, Boolean, ForeignKey
from sqlalchemy.dialects.postgresql import JSONB
from run_api.database import Base
from pydantic import BaseModel, Field
from typing import List, Optional


class Cord(BaseModel):
    method: str
    path: str
    function: str
    stream: bool
    public_api_path: Optional[str] = None
    public_api_method: Optional[str] = None


class NodeSelector(BaseModel):
    gpu_count: Optional[int] = Field(1, ge=1, le=8)
    min_vram_gb_per_gpu: Optional[int] = Field(16, ge=16, le=80)
    exclude: Optional[List[str]] = []
    include: Optional[List[str]] = None
    require_sxm: Optional[bool] = False


class ChuteArgs(BaseModel):
    name: str
    image: str
    public: bool
    standard_template: Optional[str] = None
    node_selector: NodeSelector
    cords: List[Cord]


class InvocationArgs(BaseModel):
    args: str
    kwargs: str


class Chute(Base):
    __tablename__ = "chutes"
    chute_id = Column(String, primary_key=True, default="replaceme")
    user_id = Column(String, ForeignKey("users.user_id"), nullable=False)
    name = Column(String)
    image_id = Column(String, ForeignKey("images.image_id"))
    public = Column(Boolean, default=False)
    standard_template = Column(String)
    cords = Column(JSONB, nullable=False)
    node_selector = Column(JSONB, nullable=False)
    slug = Column(String)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now())

    image = relationship("Image", back_populates="chutes", lazy="joined")
    user = relationship("User", back_populates="chutes", lazy="joined")
    instances = relationship("Instance", back_populates="chute", lazy="dynamic")

    @validates("standard_template")
    def validate_standard_template(self, _, template):
        """
        Basic validation on standard templates, which for now requires either None or vllm.
        """
        if template not in (None, "vllm"):
            raise ValueError(f"Invalid standard template: {template}")
        return template

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
            path = cord.path
            if not isinstance(path, str) or not re.match(
                r"^(/[a-z][a-z0-9_]*)+$", path, re.I
            ):
                raise ValueError(f"Invalid cord path: {path}")
            public_path = cord.public_api_path
            if public_path:
                if not isinstance(public_path, str) or not re.match(
                    r"^(/[a-z][a-z0-9_]*)+$", public_path, re.I
                ):
                    raise ValueError(f"Invalid cord public path: {public_path}")
                if cord.public_api_method not in ("GET", "POST"):
                    raise ValueError(
                        f"Unsupported public API method: {cord.public_api_method}"
                    )
            stream = cord.stream
            if stream not in (None, True, False):
                raise ValueError(f"Invalid cord stream value: {stream}")
        return [cord.dict() for cord in cords]

    @validates("node_selector")
    def validate_node_selector(self, _, node_selector):
        """
        Convert back to dict.
        """
        return node_selector.dict()
