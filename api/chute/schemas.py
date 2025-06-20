"""
ORM definitions for Chutes.
"""

import re
import ast
from sqlalchemy.sql import func
from sqlalchemy.orm import relationship, validates
from sqlalchemy import Column, Float, String, DateTime, Boolean, ForeignKey, BigInteger
from sqlalchemy.dialects.postgresql import JSONB
from api.database import Base
from api.gpu import SUPPORTED_GPUS, COMPUTE_MULTIPLIER, COMPUTE_UNIT_PRICE_BASIS
from api.fmv.fetcher import get_fetcher
from pydantic import BaseModel, Field, computed_field, validator, constr
from typing import List, Optional, Dict, Any


class ChuteUpdateArgs(BaseModel):
    tagline: Optional[str] = Field(default="", max_length=1024)
    readme: Optional[str] = Field(default="", max_length=16384)
    tool_description: Optional[str] = Field(default="", max_length=16384)
    logo_id: Optional[str] = None


class Cord(BaseModel):
    method: str
    path: str
    function: str
    stream: bool
    passthrough: Optional[bool] = False
    public_api_path: Optional[str] = None
    public_api_method: Optional[str] = None
    input_schema: Optional[Dict[str, Any]] = {}
    output_schema: Optional[Dict[str, Any]] = {}
    output_content_type: Optional[str] = None
    minimal_input_schema: Optional[Dict[str, Any]] = {}


class Port(BaseModel):
    name: str
    port: int = Field(..., ge=22, le=65535)
    proto: str = constr(pattern=r"^(tcp|udp|http)$")


class Job(BaseModel):
    name: str
    upload: Optional[bool] = False
    timeout: Optional[int] = None
    ports: list[Port] = []


class NodeSelector(BaseModel):
    gpu_count: Optional[int] = Field(1, ge=1, le=8)
    min_vram_gb_per_gpu: Optional[int] = Field(16, ge=16, le=140)
    exclude: Optional[List[str]] = None
    include: Optional[List[str]] = None

    def __init__(self, **data):
        """
        Override the constructor to remove computed fields.
        """
        super().__init__(
            **{k: v for k, v in data.items() if k not in ("compute_multiplier", "supported_gpus")}
        )

    @validator("include")
    def include_supported_gpus(cls, gpus):
        """
        Simple validation for including specific GPUs in the filter.
        """
        if not gpus:
            return gpus
        if extra := set(map(lambda s: s.lower(), gpus)) - set(SUPPORTED_GPUS):
            raise ValueError(f"Invalid GPU identifiers `include`: {extra}")
        return gpus

    @validator("exclude")
    def validate_exclude(cls, gpus):
        """
        Simpe validation for excluding specific GPUs.
        """
        if not gpus:
            return gpus
        if extra := set(map(lambda s: s.lower(), gpus)) - set(SUPPORTED_GPUS):
            raise ValueError(f"Invalid GPU identifiers `exclude`: {extra}")
        return gpus

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
        return list(allowed_gpus)


class ChuteArgs(BaseModel):
    name: str = Field(min_length=3, max_length=128)
    tagline: Optional[str] = Field(default="", max_length=1024)
    readme: Optional[str] = Field(default="", max_length=16384)
    tool_description: Optional[str] = Field(None, max_length=16384)
    logo_id: Optional[str] = None
    image: str
    public: bool
    code: str
    filename: str
    ref_str: str
    standard_template: Optional[str] = None
    node_selector: NodeSelector
    cords: Optional[List[Cord]] = []
    jobs: Optional[List[Job]] = []


class InvocationArgs(BaseModel):
    args: str
    kwargs: str


class Chute(Base):
    __tablename__ = "chutes"
    chute_id = Column(String, primary_key=True, default="replaceme")
    user_id = Column(String, ForeignKey("users.user_id"), nullable=False)
    name = Column(String)
    tagline = Column(String, default="")
    readme = Column(String, default="")
    tool_description = Column(String, nullable=True)
    image_id = Column(String, ForeignKey("images.image_id"))
    logo_id = Column(String, ForeignKey("logos.logo_id", ondelete="SET NULL"), nullable=True)
    public = Column(Boolean, default=False)
    standard_template = Column(String)
    cords = Column(JSONB, nullable=False)
    jobs = Column(JSONB, nullable=True)
    node_selector = Column(JSONB, nullable=False)
    slug = Column(String)
    code = Column(String, nullable=False)
    filename = Column(String, nullable=False)
    ref_str = Column(String, nullable=False)
    version = Column(String)
    chutes_version = Column(String, nullable=True)
    openrouter = Column(Boolean, default=False)
    discount = Column(Float, nullable=True, default=0.0)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now())

    # Stats for sorting.
    invocation_count = Column(BigInteger, default=0)

    image = relationship("Image", back_populates="chutes", lazy="joined")
    user = relationship("User", back_populates="chutes", lazy="joined")
    logo = relationship("Logo", back_populates="chutes", lazy="joined")
    rolling_update = relationship(
        "RollingUpdate", back_populates="chute", lazy="joined", uselist=False
    )
    instances = relationship(
        "Instance", back_populates="chute", lazy="select", cascade="all, delete-orphan"
    )
    shares = relationship(
        "ChuteShare", back_populates="chute", lazy="select", cascade="all, delete-orphan"
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
        if template not in (None, "vllm", "diffusion", "tei"):
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
            return []
        for cord in cords:
            path = cord.path
            if not isinstance(path, str) or not re.match(r"^(/[a-z][a-z0-9_]*)+$", path, re.I):
                raise ValueError(f"Invalid cord path: {path}")
            public_path = cord.public_api_path
            if public_path:
                if not isinstance(public_path, str) or not re.match(
                    r"^(/[a-z][a-z0-9_-]*)+$", public_path, re.I
                ):
                    raise ValueError(f"Invalid cord public path: {public_path}")
                if cord.public_api_method not in ("GET", "POST"):
                    raise ValueError(f"Unsupported public API method: {cord.public_api_method}")
            stream = cord.stream
            if stream not in (None, True, False):
                raise ValueError(f"Invalid cord stream value: {stream}")
        return [cord.dict() for cord in cords]

    @validates("jobs")
    def validate_jobs(self, _, jobs):
        """
        Basic validation of job definitions.
        """
        if not jobs:
            return []
        job_dicts = []
        for job in jobs:
            if isinstance(job, dict):
                job_object = Job(**job)
            job_dicts.append(job_object.model_dump())
        return job_dicts

    @validates("node_selector")
    def validate_node_selector(self, _, node_selector):
        """
        Convert back to dict.
        """
        as_dict = node_selector.dict()
        as_dict.pop("compute_multiplier", None)
        as_dict.pop("supported_gpus", None)
        return as_dict

    @computed_field
    @property
    def supported_gpus(self) -> List[str]:
        return NodeSelector(**self.node_selector).supported_gpus


class ChuteHistory(Base):
    __tablename__ = "chute_history"
    entry_id = Column(String, primary_key=True)
    chute_id = Column(String, nullable=False)
    user_id = Column(String, nullable=False)
    version = Column(String, nullable=False)
    name = Column(String)
    tagline = Column(String, default="")
    readme = Column(String, default="")
    tool_description = Column(String, nullable=True)
    image_id = Column(String, nullable=False)
    logo_id = Column(String)
    public = Column(Boolean, default=False)
    standard_template = Column(String)
    cords = Column(JSONB, nullable=False)
    node_selector = Column(JSONB, nullable=False)
    slug = Column(String)
    code = Column(String, nullable=False)
    filename = Column(String, nullable=False)
    ref_str = Column(String, nullable=False)
    version = Column(String)
    chutes_version = Column(String, nullable=True)
    openrouter = Column(Boolean, default=False)
    discount = Column(Float, nullable=True, default=0.0)
    created_at = Column(DateTime, server_default=func.now())
    updated_at = Column(DateTime, server_default=func.now())
    deleted_at = Column(DateTime, server_default=func.now())


class RollingUpdate(Base):
    __tablename__ = "rolling_updates"
    chute_id = Column(
        String, ForeignKey("chutes.chute_id", ondelete="CASCADE"), nullable=False, primary_key=True
    )
    old_version = Column(String, nullable=False)
    new_version = Column(String, nullable=False)
    started_at = Column(DateTime, server_default=func.now())
    permitted = Column(JSONB, nullable=False)

    chute = relationship("Chute", back_populates="rolling_update", uselist=False)


class ChuteShare(Base):
    __tablename__ = "chute_shares"
    chute_id = Column(
        String, ForeignKey("chutes.chute_id", ondelete="CASCADE"), nullable=False, primary_key=True
    )
    shared_by = Column(
        String, ForeignKey("users.user_id", ondelete="CASCADE"), nullable=False, primary_key=True
    )
    shared_to = Column(
        String, ForeignKey("users.user_id", ondelete="CASCADE"), nullable=False, primary_key=True
    )
    shared_at = Column(DateTime, server_default=func.now())

    chute = relationship("Chute", back_populates="shares", uselist=False)
