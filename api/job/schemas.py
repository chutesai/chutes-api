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
    Double,
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
    instance_id = Column(
        String, ForeignKey("instances.instance_id", ondelete="SET NULL"), nullable=True, unique=True
    )

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

    # Final result.
    status = Column(String, nullable=False, default="pending")
    result = Column(JSONB, nullable=True)
    error_detail = Column(String, nullable=True)
    output_files = Column(JSONB, nullable=True)
    miner_terminated = Column(Boolean, nullable=True, default=False)

    # Port mappings.
    port_mappings = Column(JSONB, nullable=True)

    # Track the hotkeys that have attempted a job.
    miner_history = Column(JSONB, nullable=False, default=[])

    # Track the compute multiplier, which we could manually tweak to prioritize jobs.
    compute_multiplier = Column(Double, nullable=False)

    # Relationships.
    chute = relationship("Chute", back_populates="running_jobs", lazy="joined")
    user = relationship("User", back_populates="jobs", lazy="joined")
    instance = relationship("Instance", back_populates="job", lazy="joined", uselist=False)
    launch_config = relationship("LaunchConfig", back_populates="job", uselist=False)
