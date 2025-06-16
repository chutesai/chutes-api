"""
Async jobs (or rentals, etc.)
"""

from sqlalchemy.sql import func
from sqlalchemy.orm import relationship
from sqlalchemy import (
    Column,
    String,
    DateTime,
    Boolean,
    ForeignKey,
    Integer,
)
from sqlalchemy.dialects.postgresql import JSONB
from api.database import Base, generate_uuid


class Job(Base):
    __tablename__ = "jobs"

    # Base metadata.
    job_id = Column(String, primary_key=True, default=generate_uuid)
    user_id = Column(String, ForeignKey("users.user_id", ondelete="SET NULL"), nullable=False)
    chute_id = Column(String, ForeignKey("chutes.chute_id", ondelete="SET NULL"), nullable=False)
    version = Column(String, nullable=False)
    chutes_version = Column(String, nullable=True)
    method = Column(String, nullable=False)

    # Miner info, when claimed.
    miner_uid = Column(Integer, nullable=True)
    miner_hotkey = Column(String, nullable=True)
    miner_coldkey = Column(String, nullable=True)
    instance_id = Column(String, nullable=True)

    # State info.
    active = Column(Boolean, default=False)
    verified = Column(Boolean, default=False)
    last_queried_at = Column(DateTime)
    created_at = Column(DateTime, server_default=func.now())
    updated_at = Column(DateTime)
    started_at = Column(DateTime)
    finished_at = Column(DateTime)

    # Job args, e.g. ports, timeout, etc.
    job_args = Column(JSONB, nullable=False)

    # Port mappings.
    port_mappings = Column(JSONB, nullable=True)

    # Relationships.
    chute = relationship("Chute", back_populates="jobs", lazy="joined")
    user = relationship("User", back_populates="jobs", lazy="joined")
    instance = relationship("Instance", back_populates="job", lazy="joined")
