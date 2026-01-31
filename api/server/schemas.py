"""
ORM definitions for servers and TDX attestations.
"""

from pydantic import BaseModel, Field
from sqlalchemy.sql import func
from sqlalchemy.orm import relationship
from sqlalchemy import (
    Column,
    Integer,
    String,
    DateTime,
    Boolean,
    ForeignKey,
    Text,
    Index,
    ForeignKeyConstraint,
)
from typing import Dict, Any
from api.database import Base, generate_uuid
from api.node.schemas import NodeArgs


class NonceResponse(BaseModel):
    """Response model for nonce generation."""

    nonce: str
    expires_at: str


class BootAttestationArgs(BaseModel):
    """Request model for boot attestation."""

    quote: str = Field(..., description="Base64 encoded TDX quote")
    miner_hotkey: str = Field(..., description="Miner hotkey that owns this VM")
    vm_name: str = Field(..., description="VM name/identifier")


class BootAttestationResponse(BaseModel):
    """Response model for successful boot attestation."""

    key: str
    boot_token: str


class RuntimeAttestationArgs(BaseModel):
    """Request model for runtime attestation."""

    quote: str = Field(..., description="Base64 encoded TDX quote")


class RuntimeAttestationResponse(BaseModel):
    """Response model for runtime attestation."""

    attestation_id: str
    verified_at: str
    status: str


class CacheLuksPassphraseResponse(BaseModel):
    """Response model for cache LUKS passphrase."""

    passphrase: str


class GpuAttestationArgs(BaseModel):
    evidence: str = Field(..., description="Base64 encoded GPU evidence")


class GpuAttestationResponse(BaseModel):
    attestation_id: str
    verified_at: str
    gpu_info: Dict[str, Any]  # GPU details from evidence


class ServerArgs(BaseModel):
    """Request model for server registration."""

    id: str = Field(..., description="Server ID, should come from the k8s node uid.")
    host: str = Field(..., descriptiopn="Public IP address or DNS Name of the server")
    gpus: list[NodeArgs] = Field(..., description="GPU info for this server")


class BootAttestation(Base):
    """Track anonymous boot attestations (pre-registration)."""

    __tablename__ = "boot_attestations"

    attestation_id = Column(String, primary_key=True, default=generate_uuid)
    quote_data = Column(Text, nullable=False)  # Base64 encoded quote
    server_ip = Column(String, nullable=True)  # For later linking to server
    verification_error = Column(String, nullable=True)
    measurement_version = Column(String, nullable=True)  # Matched TEE measurement config version (audit trail)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    verified_at = Column(DateTime(timezone=True), nullable=True)

    __table_args__ = (
        Index("idx_boot_server_id", "server_ip"),
        Index("idx_boot_created", "created_at"),
        Index("idx_boot_verified", "verified_at"),
    )


class Server(Base):
    """Main server entity (created after boot via CLI)."""

    __tablename__ = "servers"

    server_id = Column(String, primary_key=True)
    ip = Column(String, nullable=False)  # Links to boot attestations
    miner_hotkey = Column(String, nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    netuid = Column(Integer, nullable=False, default=64, server_default="64")

    is_tee = Column(Boolean, default=False, server_default="false")

    # Relationships
    nodes = relationship("Node", back_populates="server", cascade="all, delete-orphan")
    runtime_attestations = relationship(
        "ServerAttestation", back_populates="server", cascade="all, delete-orphan"
    )
    miner = relationship("MetagraphNode", back_populates="servers")

    __table_args__ = (
        Index("idx_server_miner", "miner_hotkey"),
        ForeignKeyConstraint(
            ["netuid", "miner_hotkey"], ["metagraph_nodes.netuid", "metagraph_nodes.hotkey"]
        ),
    )


class ServerAttestation(Base):
    """Track runtime attestations (post-registration)."""

    __tablename__ = "server_attestations"

    attestation_id = Column(String, primary_key=True, default=generate_uuid)
    server_id = Column(String, ForeignKey("servers.server_id", ondelete="CASCADE"), nullable=False)
    quote_data = Column(Text, nullable=True)  # Base64 encoded quote
    verification_error = Column(String, nullable=True)
    measurement_version = Column(String, nullable=True)  # Matched TEE measurement config version (audit trail)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    verified_at = Column(DateTime(timezone=True), nullable=True)

    server = relationship("Server", back_populates="runtime_attestations")

    __table_args__ = (
        Index("idx_attestation_server", "server_id"),
        Index("idx_attestation_created", "created_at"),
        Index("idx_attestation_verified", "verified_at"),
    )


class VmCacheConfig(Base):
    """Track cache volume encryption passphrases by VM configuration."""

    __tablename__ = "vm_cache_configs"

    miner_hotkey = Column(String, primary_key=True)
    vm_name = Column(String, primary_key=True)
    encrypted_passphrase = Column(String, nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    last_boot_at = Column(DateTime(timezone=True), nullable=True)

    __table_args__ = (
        Index("idx_vm_cache_miner", "miner_hotkey"),
        Index("idx_vm_cache_last_boot", "last_boot_at"),
    )
