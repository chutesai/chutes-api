"""
Challenge schemas.
"""

from sqlalchemy import (
    Column,
    String,
    DateTime,
    Boolean,
    ForeignKey,
    Index,
    func,
)
from sqlalchemy.orm import relationship
from sqlalchemy.dialects.postgresql import JSONB
from api.database import Base, generate_uuid


class Challenge(Base):
    __tablename__ = "miner_challenges"
    uuid = Column(
        String, ForeignKey("nodes.uuid", ondelete="CASCADE"), primary_key=True
    )
    challenge = Column(String, nullable=False)
    challenge_type = Column(String, default="graval", primary_key=True)
    created_at = Column(DateTime, server_default=func.now())

    node = relationship("Node", back_populates="challenges")


class ChallengeResult(Base):
    __tablename__ = "miner_challenge_results"
    result_id = Column(String, primary_key=True, default=generate_uuid)
    device_uuid = Column(String, ForeignKey("nodes.uuid", ondelete="CASCADE"))
    miner_hotkey = Column(String, ForeignKey("metagraph_nodes.hotkey"), nullable=False)
    challenge_type = Column(String, nullable=False)
    challenge_args = Column(JSONB, nullable=False)
    challenge_result = Column(JSONB)
    success = Column(Boolean)
    challenge_started_at = Column(DateTime, server_default=func.now())
    challenge_completed_at = Column(DateTime)

    node = relationship("Node", back_populates="challenge_results")
    miner = relationship("MetagraphNode", back_populates="challenge_results")

    __table_args__ = (
        Index("idx_hotkey_date", "miner_hotkey", "challenge_started_at"),
        Index("idx_node_date", "device_uuid", "challenge_started_at"),
    )
