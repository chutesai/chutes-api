"""
ORM definitions for servers and TDX attestations.
"""

import secrets
from pydantic import BaseModel, Field
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.sql import func
from sqlalchemy.orm import relationship
from sqlalchemy import (
    Column,
    String,
    DateTime,
    Boolean,
    ForeignKey,
    Integer,
    Text,
    Index,
    UniqueConstraint,
)
from typing import Optional, Dict, Any
from api.database import Base, generate_uuid


class BootAttestationArgs(BaseModel):
    """Request model for boot attestation."""
    quote: str = Field(..., description="Base64 encoded TDX quote")
    nonce: str = Field(..., description="Nonce for replay protection")
    hardware_id: Optional[str] = Field(None, description="Hardware identifier for linking")


class RuntimeAttestationArgs(BaseModel):
    """Request model for runtime attestation."""
    quote: str = Field(..., description="Base64 encoded TDX quote")
    nonce: str = Field(..., description="Nonce for replay protection")


class ServerArgs(BaseModel):
    """Request model for server registration."""
    hardware_id: Optional[str] = Field(None, description="Hardware identifier to link boot attestations")
    name: str = Field(..., description="Server name/identifier")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional server metadata")


class NonceResponse(BaseModel):
    """Response model for nonce generation."""
    nonce: str
    expires_at: str


class BootAttestationResponse(BaseModel):
    """Response model for successful boot attestation."""
    luks_passphrase: str
    attestation_id: str
    verified_at: str


class RuntimeAttestationResponse(BaseModel):
    """Response model for runtime attestation."""
    attestation_id: str
    verified_at: str
    status: str


# AttestationNonce removed - using Redis instead


class BootAttestation(Base):
    """Track anonymous boot attestations (pre-registration)."""
    __tablename__ = "boot_attestations"
    
    attestation_id = Column(String, primary_key=True, default=generate_uuid)
    quote_data = Column(Text, nullable=False)  # Base64 encoded quote
    hardware_id = Column(String, nullable=True)  # For later linking to server
    mrtd = Column(String, nullable=True)  # Extracted MRTD from quote
    verification_result = Column(JSONB, nullable=True)  # Detailed verification results
    verified = Column(Boolean, default=False)
    verification_error = Column(String, nullable=True)
    nonce_used = Column(String, nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    verified_at = Column(DateTime(timezone=True), nullable=True)

    __table_args__ = (
        Index("idx_boot_hardware_id", "hardware_id"),
        Index("idx_boot_created", "created_at"),
        Index("idx_boot_verified", "verified"),
    )


class Server(Base):
    """Main server entity (created after boot via CLI)."""
    __tablename__ = "servers"
    
    server_id = Column(String, primary_key=True, default=generate_uuid)
    name = Column(String, nullable=False)
    hardware_id = Column(String, nullable=True, unique=True)  # Links to boot attestations
    miner_hotkey = Column(
        String, ForeignKey("metagraph_nodes.hotkey", ondelete="CASCADE"), nullable=False
    )
    metadata = Column(JSONB, nullable=True)  # Additional server info
    expected_measurements = Column(JSONB, nullable=True)  # Runtime MRTD/RTMRs or K8s secret refs
    active = Column(Boolean, default=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())

    # Relationships
    nodes = relationship("Node", back_populates="server")
    runtime_attestations = relationship("ServerAttestation", back_populates="server", cascade="all, delete-orphan")
    miner = relationship("MetagraphNode", back_populates="servers")

    __table_args__ = (
        Index("idx_server_miner", "miner_hotkey"),
        Index("idx_server_active", "active"),
    )


class ServerAttestation(Base):
    """Track runtime attestations (post-registration)."""
    __tablename__ = "server_attestations"
    
    attestation_id = Column(String, primary_key=True, default=generate_uuid)
    server_id = Column(String, ForeignKey("servers.server_id", ondelete="CASCADE"), nullable=False)
    quote_data = Column(Text, nullable=False)  # Base64 encoded quote
    mrtd = Column(String, nullable=True)  # Extracted MRTD
    rtmrs = Column(JSONB, nullable=True)  # Extracted RTMRs
    verification_result = Column(JSONB, nullable=True)  # Detailed verification results
    verified = Column(Boolean, default=False)
    verification_error = Column(String, nullable=True)
    nonce_used = Column(String, nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    verified_at = Column(DateTime(timezone=True), nullable=True)

    server = relationship("Server", back_populates="runtime_attestations")

    __table_args__ = (
        Index("idx_attestation_server", "server_id"),
        Index("idx_attestation_created", "created_at"),
        Index("idx_attestation_verified", "verified"),
    )