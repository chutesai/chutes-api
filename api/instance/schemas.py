"""
ORM definitions for instances (deployments of chutes and/or inventory announcements).
"""

from pydantic import BaseModel
from sqlalchemy.sql import func
from sqlalchemy.orm import relationship, validates
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
from api.utils import is_valid_host


# Association table.
instance_nodes = Table(
    "instance_nodes",
    Base.metadata,
    Column(
        "instance_id", String, ForeignKey("instances.instance_id", ondelete="CASCADE")
    ),
    Column("node_id", String, ForeignKey("nodes.uuid", ondelete="CASCADE")),
    UniqueConstraint("instance_id", "node_id", name="uq_instance_node"),
)


class InstanceArgs(BaseModel):
    node_ids: list[str]
    host: str
    port: int
from api.node.schemas import Node  # noqa

class Instance(Base):
    __tablename__ = "instances"
    instance_id = Column(String, primary_key=True, default=generate_uuid)
    host = Column(String, nullable=False)
    port = Column(Integer, nullable=False)
    chute_id = Column(
        String, ForeignKey("chutes.chute_id", ondelete="CASCADE"), nullable=False
    )
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

    nodes = relationship(
        "Node", secondary=instance_nodes, back_populates="instance", lazy="joined"
    )
    chute = relationship("Chute", back_populates="instances", lazy="joined")

    __table_args__ = (
        Index(
            "idx_chute_active_lastq",
            "chute_id",
            "active",
            "verified",
            "last_queried_at",
        ),
    )

    @validates("host")
    async def validate_host(self, host: str) -> str:
        if await is_valid_host(host):
            return host
        raise ValueError(f"Invalid verification_host: {host}")
