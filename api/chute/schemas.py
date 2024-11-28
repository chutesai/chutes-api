"""
ORM definitions for Chutes.
"""

import re
import ast
from sqlalchemy.sql import func
from sqlalchemy.orm import relationship, validates
from sqlalchemy import Column, String, DateTime, Boolean, ForeignKey
from sqlalchemy.dialects.postgresql import JSONB
from api.database import Base
from api.gpu import SUPPORTED_GPUS, COMPUTE_MULTIPLIER, ALLOWED_INCLUDE
from api.fmv.fetcher import get_fetcher
from api.payment.constants import COMPUTE_UNIT_PRICE_BASIS
from pydantic import BaseModel, Field, computed_field, validator
from typing import List, Optional, Dict, Any


class Cord(BaseModel):
    method: str
    path: str
    function: str
    stream: bool
    public_api_path: Optional[str] = None
    public_api_method: Optional[str] = None
    input_schema: Optional[Dict[str, Any]] = {}
    output_schema: Optional[Dict[str, Any]] = {}
    minimal_input_schema: Optional[Dict[str, Any]] = {}


class NodeSelector(BaseModel):
    gpu_count: Optional[int] = Field(1, ge=1, le=8)
    min_vram_gb_per_gpu: Optional[int] = Field(16, ge=16, le=80)
    exclude: Optional[List[str]] = None
    include: Optional[List[str]] = None
    require_sxm: Optional[bool] = False

    @validator("include")
    def include_supported_gpus(cls, gpus):
        """
        Simple validation for including specific GPUs in the filter.  We're currently
        only allowing high availability and likely-to-be-selected GPUs in this list.
        """
        if not gpus:
            return gpus
        if set(map(lambda s: s.lower(), gpus)) - ALLOWED_INCLUDE:
            raise ValueError(
                f"include must contain only the following GPUs: {list(ALLOWED_INCLUDE)}"
            )
        return gpus

    @validator("exclude")
    def validate_exclude(cls, gpus):
        """
        Make sure people don't try to be sneaky with the exclude flag to XOR
        the list of allowed GPUs.
        """
        if not gpus:
            return gpus
        remaining = set(SUPPORTED_GPUS) - set(gpus)
        if not remaining & ALLOWED_INCLUDE:
            raise ValueError(
                f"exclude must allow for at least one the following GPUs: {list(ALLOWED_INCLUDE)}"
            )
        return gpus

    @computed_field
    @property
    def compute_multiplier(self) -> float:
        """
        Determine a multiplier to use when calculating incentive and such,
        e.g. a6000 < l40s < a100 < h100, 2 GPUs > 1 GPU, etc.

        This operates on the MINIMUM value specified by the node multiplier.
        """
        supported_gpus = self.supported_gpus
        if not supported_gpus:
            raise ValueError("No GPUs match specified node_selector criteria")

        # Always use the minimum boost value, since miners should try to optimize
        # to run as cheaply as possible while satisfying the requirements.
        multiplier = min([COMPUTE_MULTIPLIER[gpu] for gpu in supported_gpus])
        return self.gpu_count * multiplier

    async def current_estimated_price(self):
        """
        Calculate estimated price to use this chute, per second.
        """
        current_tao_price = await get_fetcher().get_price("tao")
        if current_tao_price is None:
            return None
        usd_price = COMPUTE_UNIT_PRICE_BASIS * self.compute_multiplier
        tao_price = usd_price / current_tao_price
        return {
            "usd": {
                "hour": usd_price,
                "second": usd_price / 3600,
            },
            "tao": {
                "hour": tao_price,
                "second": tao_price / 3600,
            },
        }

    @computed_field
    @property
    def supported_gpus(self) -> List[str]:
        """
        Generate the list of all supported GPUs (short ref string).
        """
        allowed_gpus = set(SUPPORTED_GPUS)
        if self.include:
            allowed_gpus = set(self.include)
        if self.exclude:
            allowed_gpus -= set(self.exclude)
        if self.min_vram_gb_per_gpu:
            allowed_gpus = set(
                [
                    gpu
                    for gpu in allowed_gpus
                    if SUPPORTED_GPUS[gpu]["memory"] >= self.min_vram_gb_per_gpu
                ]
            )
        if self.require_sxm:
            allowed_gpus = set([gpu for gpu in allowed_gpus if SUPPORTED_GPUS[gpu]["sxm"]])
        return list(allowed_gpus)


class ChuteArgs(BaseModel):
    name: str = Field(min_length=3, max_length=128)
    readme: Optional[str] = Field(default="", max_length=16384)
    logo_id: Optional[str] = None
    image: str
    public: bool
    code: str
    filename: str
    ref_str: str
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
    readme = Column(String, default="")
    image_id = Column(String, ForeignKey("images.image_id"))
    logo_id = Column(String, ForeignKey("logos.logo_id", ondelete="SET NULL"), nullable=True)
    public = Column(Boolean, default=False)
    standard_template = Column(String)
    cords = Column(JSONB, nullable=False)
    node_selector = Column(JSONB, nullable=False)
    slug = Column(String)
    code = Column(String, nullable=False)
    filename = Column(String, nullable=False)
    ref_str = Column(String, nullable=False)
    version = Column(String)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now())

    image = relationship("Image", back_populates="chutes", lazy="joined")
    user = relationship("User", back_populates="chutes", lazy="joined")
    logo = relationship("Logo", back_populates="chutes", lazy="joined")
    instances = relationship(
        "Instance", back_populates="chute", lazy="select", cascade="all, delete-orphan"
    )

    @validates("name")
    def validate_name(self, _, name):
        """
        Basic validation on chute name.
        """
        if (
            not isinstance(name, str)
            or not re.match(r"^(?:([a-zA-Z0-9_\.-]+)/)*([a-z0-9][a-z0-9_\.\/-]*)$", name, re.I)
            or len(name) >= 128
        ):
            raise ValueError(f"Invalid chute name: {name}")
        return name

    @validates("standard_template")
    def validate_standard_template(self, _, template):
        """
        Basic validation on standard templates, which for now requires either None or vllm.
        """
        if template not in (None, "vllm"):
            raise ValueError(f"Invalid standard template: {template}")
        return template

    @validates("filename")
    def validate_filename(self, _, filename):
        """
        Validate the filename (the entrypoint for chutes run ...)
        """
        if not isinstance(filename, str) or not re.match(r"^[a-z][a-z0-9_]*\.py$", filename):
            raise ValueError(f"Invalid entrypoint filename: '{filename}'")
        return filename

    @validates("ref_str")
    def validate_ref_str(self, _, ref_str):
        """
        Validate the reference string, which should be {filename (no .py ext)}:{chute var name}
        """
        if not isinstance(ref_str, str) or not re.match(
            r"^[a-z][a-z0-9_]*:[a-z][a-z0-9_]*$", ref_str
        ):
            raise ValueError(f"Invalid reference string: '{ref_str}'")
        return ref_str

    @validates("code")
    def validate_code(self, _, code):
        """
        Syntax check on the code.
        """
        try:
            ast.parse(code)
        except SyntaxError as exc:
            raise ValueError(f"Invalid code submited: {exc}")
        return code

    @validates("cords")
    def validate_cords(self, _, cords):
        """
        Basic validation on the cords, i.e. methods that can be called for this chute.
        """
        if not cords or not isinstance(cords, list):
            raise ValueError("Must include between 1 and 25 valid cords to create a chute")
        for cord in cords:
            path = cord.path
            if not isinstance(path, str) or not re.match(r"^(/[a-z][a-z0-9_]*)+$", path, re.I):
                raise ValueError(f"Invalid cord path: {path}")
            public_path = cord.public_api_path
            if public_path:
                if not isinstance(public_path, str) or not re.match(
                    r"^(/[a-z][a-z0-9_]*)+$", public_path, re.I
                ):
                    raise ValueError(f"Invalid cord public path: {public_path}")
                if cord.public_api_method not in ("GET", "POST"):
                    raise ValueError(f"Unsupported public API method: {cord.public_api_method}")
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

    @computed_field
    @property
    def supported_gpus(self) -> List[str]:
        return NodeSelector(**self.node_selector).supported_gpus
