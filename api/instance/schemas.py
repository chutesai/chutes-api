"""
ORM definitions for instances (deployments of chutes and/or inventory announcements).
"""

import secrets
from pydantic import BaseModel
from sqlalchemy.sql import func
from sqlalchemy.orm import relationship
from sqlalchemy import (
    Column,
    String,
    DateTime,
    Boolean,
    ForeignKey,
    Integer,
    Index,
    Table,
    UniqueConstraint,
)
from api.database import Base, generate_uuid

# Association table.
instance_nodes = Table(
    "instance_nodes",
    Base.metadata,
    Column("instance_id", String, ForeignKey("instances.instance_id", ondelete="CASCADE")),
    Column("node_id", String, ForeignKey("nodes.uuid", ondelete="NO ACTION")),
    UniqueConstraint("instance_id", "node_id", name="uq_instance_node"),
    UniqueConstraint("node_id", name="uq_inode"),
)


class InstanceArgs(BaseModel):
    node_ids: list[str]
    host: str
    port: int


class ActivateArgs(BaseModel):
    active: bool


class Instance(Base):
    __tablename__ = "instances"
    instance_id = Column(String, primary_key=True, default=generate_uuid)
    host = Column(String, nullable=False)
    port = Column(Integer, nullable=False)
    chute_id = Column(String, ForeignKey("chutes.chute_id", ondelete="CASCADE"), nullable=False)
    version = Column(String, nullable=False)
    miner_uid = Column(Integer, nullable=False)
    miner_hotkey = Column(String, nullable=False)
    miner_coldkey = Column(String, nullable=False)
    region = Column(String)
    active = Column(Boolean, default=False)
    verified = Column(Boolean, default=False)
    last_queried_at = Column(DateTime(timezone=True))
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True))
    last_verified_at = Column(DateTime(timezone=True))
    verification_error = Column(String, nullable=True)
    consecutive_failures = Column(Integer, default=0)
    chutes_version = Column(String, nullable=True)
    symmetric_key = Column(String, default=lambda: secrets.token_bytes(16).hex())
    job_id = Column(String, ForeignKey("jobs.job_id", ondelete="SET NULL"), nullable=True)
    config_id = Column(String, ForeignKey("launch_configs.config_id", ondelete="SET NULL"), nullable=True)

    nodes = relationship("Node", secondary=instance_nodes, back_populates="instance")
    chute = relationship("Chute", back_populates="instances")
    job = relationship("Job", back_populates="instance")

    __table_args__ = (
        Index(
            "idx_chute_active_lastq",
            "chute_id",
            "active",
            "verified",
            "last_queried_at",
        ),
        UniqueConstraint("host", "port", name="unique_host_port"),
    )


class LaunchConfig(Base):
    __tablename__ = "launch_configs"
    config_id =  Column(String, primary_key=True, default=generate_uuid)
    seed = Column(BigInteger, nullable=False)
    env_key = Column(String, nullable=False)
    chute_id = Column(String, ForeignKey("chutes.chute_id", ondelete="CASCADE"), nullable=False)
    job_id = Column(String, ForeignKey("jobs.job_id", ondelete="CASCADE"), nullable=True)
