"""
ORM definitions for nodes (single GPUs in miner infra).
"""

import re
from pydantic import BaseModel, Field
from typing import List
from sqlalchemy import (
    Column,
    String,
    Integer,
    BigInteger,
    Boolean,
    Float,
    ForeignKey,
    DateTime,
    func,
)
from sqlalchemy.orm import validates, relationship
from api.gpu import SUPPORTED_GPUS
from api.database import Base
from api.instance.schemas import instance_nodes
from api.chute.schemas import Chute, NodeSelector


class NodeArgs(BaseModel):
    uuid: str
    name: str
    memory: int
    major: int
    minor: int
    processors: int
    sxm: bool
    clock_rate: float
    max_threads_per_processor: int
    concurrent_kernels: bool
    ecc: bool
    device_index: int = Field(gte=0, lt=8)
    gpu_identifier: str
    verification_host: str
    verification_port: int


class MultiNodeArgs(BaseModel):
    nodes: List[NodeArgs]


class Node(Base):
    __tablename__ = "nodes"
    # Normal device info fields.
    uuid = Column(String, primary_key=True)
    name = Column(String, nullable=False)
    memory = Column(BigInteger, nullable=False)
    major = Column(Integer, nullable=False)
    minor = Column(Integer, nullable=False)
    processors = Column(Integer, nullable=False)
    sxm = Column(Boolean, nullable=False)
    clock_rate = Column(Float, nullable=False)
    max_threads_per_processor = Column(Integer, nullable=False)
    concurrent_kernels = Column(Boolean, nullable=False)
    ecc = Column(Boolean, nullable=False)
    seed = Column(BigInteger, nullable=False)

    # Meta/app fields.
    miner_hotkey = Column(
        String, ForeignKey("metagraph_nodes.hotkey", ondelete="CASCADE"), nullable=False
    )
    gpu_identifier = Column(String, nullable=False)
    device_index = Column(Integer, nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    verification_host = Column(String, nullable=False)
    verification_port = Column(Integer, nullable=False)
    verification_error = Column(String)
    verified_at = Column(DateTime(timezone=True))

    _gpu_specs = None
    _gpu_key = None

    instance = relationship(
        "Instance",
        back_populates="nodes",
        secondary=instance_nodes,
        lazy="joined",
        uselist=False,
    )
    miner = relationship("MetagraphNode", back_populates="nodes", lazy="joined")
    challenges = relationship("Challenge", back_populates="node", cascade="all, delete-orphan")
    challenge_results = relationship(
        "ChallengeResult", back_populates="node", cascade="all, delete-orphan"
    )

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._validate_all()

    def graval_dict(self):
        """
        Dict representation as expected by graval.
        """
        return {
            key: getattr(self, key, None)
            for key in [
                "uuid",
                "name",
                "memory",
                "major",
                "minor",
                "processors",
                "sxm",
                "clock_rate",
                "max_threads_per_processor",
                "concurrent_kernels",
                "ecc",
            ]
        }

    def _validate_all(self):
        """
        Check the miner-specified specs against expected values.
        """
        self.validate_gpu_model(None, self.name)
        self.validate_memory(None, self.memory)
        self.validate_compute_capability("major", self.major)
        self.validate_compute_capability("minor", self.minor)
        self.validate_processors(None, self.processors)
        self.validate_clock_rate(None, self.clock_rate)
        self.validate_max_threads(None, self.max_threads_per_processor)
        self.validate_boolean_features("concurrent_kernels", self.concurrent_kernels)
        self.validate_boolean_features("ecc", self.ecc)
        self.validate_boolean_features("sxm", self.sxm)

    @validates("verification_port")
    def validate_port(self, _, port: int) -> int:
        if 80 <= port <= 65535:
            return port
        raise ValueError(f"Invalid verification_port: {port}")

    @validates("name")
    def validate_gpu_model(self, _, name: str) -> str:
        for gpu_key, specs in SUPPORTED_GPUS.items():
            if re.search(specs["model_name_check"], name):
                self._gpu_specs = specs
                self._gpu_key = gpu_key
                return name
        raise ValueError(f"GPU model in name '{name}' not found in supported GPUs")

    @validates("memory")
    def validate_memory(self, _, memory: int) -> int:
        if not self._gpu_specs:
            return memory
        memory_gb = int(memory / (1000 * 1000 * 1000))
        expected_memory = self._gpu_specs["memory"]
        if not (expected_memory - 1 <= memory_gb <= expected_memory + 1):
            raise ValueError(
                f"Memory {memory_gb}GB does not match expected {expected_memory}GB for {self._gpu_key}"
            )
        return memory

    @validates("major", "minor")
    def validate_compute_capability(self, key: str, value: int) -> int:
        if not self._gpu_specs:
            return value
        expected = self._gpu_specs[key]
        if value != expected:
            raise ValueError(
                f"Compute capability {key} {value} does not match expected {expected} for {self._gpu_key}"
            )
        return value

    @validates("processors")
    def validate_processors(self, _, processors: int) -> int:
        if not self._gpu_specs:
            return processors
        expected = self._gpu_specs["processors"]
        if processors != expected:
            raise ValueError(
                f"Processor count {processors} does not match expected {expected} for {self._gpu_key}"
            )
        return processors

    @validates("clock_rate")
    def validate_clock_rate(self, _, clock_rate: float) -> float:
        if not self._gpu_specs:
            return clock_rate
        base_clock = self._gpu_specs["clock_rate"]["base"]
        boost_clock = self._gpu_specs["clock_rate"]["boost"]
        clock_mhz = clock_rate / 1000
        if not (base_clock <= clock_mhz <= boost_clock * 1.1):
            raise ValueError(
                f"Clock rate {clock_mhz:.0f}MHz not within expected range {base_clock}-{boost_clock}MHz for {self._gpu_key}"
            )
        return clock_rate

    @validates("max_threads_per_processor")
    def validate_max_threads(self, _, max_threads: int) -> int:
        if not self._gpu_specs:
            return max_threads
        expected = self._gpu_specs["max_threads_per_processor"]
        if max_threads != expected:
            raise ValueError(
                f"Max threads per processor {max_threads} does not match expected {expected} for {self._gpu_key}"
            )
        return max_threads

    @validates("concurrent_kernels", "ecc", "sxm")
    def validate_boolean_features(self, key: str, value: bool) -> bool:
        if not self._gpu_specs:
            return value
        expected = self._gpu_specs[key]
        if value != expected:
            raise ValueError(
                f"{key} setting {value} does not match expected {expected} for {self._gpu_key}"
            )
        return value

    def is_suitable(self, chute: Chute) -> bool:
        """
        Check if a node fits all requirements for a particular chute (via node selector).
        """
        if not self.verified_at:
            return False
        allowed_gpus = set(SUPPORTED_GPUS)
        node_selector = NodeSelector(**chute.node_selector)
        if node_selector.include:
            allowed_gpus = set(node_selector.include)
        if node_selector.exclude:
            allowed_gpus -= set(node_selector.exclude)
        if node_selector.min_vram_gb_per_gpu:
            allowed_gpus = set(
                [
                    gpu
                    for gpu in allowed_gpus
                    if SUPPORTED_GPUS[gpu]["memory"] >= node_selector.min_vram_gb_per_gpu
                ]
            )
        return self.gpu_identifier in allowed_gpus
