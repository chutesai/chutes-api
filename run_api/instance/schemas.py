"""
ORM definitions for instances (deployments of chutes and/or inventory announcements).
"""

from sqlalchemy.sql import func
from sqlalchemy.orm import relationship
from sqlalchemy import Column, String, DateTime, Boolean, ForeignKey, Integer, Index
from sqlalchemy.dialects.postgresql import JSONB
from run_api.database import Base


class Instance(Base):
    __tablename__ = "instances"
    instance_id = Column(String, primary_key=True, default="replaceme")
    ip = Column(String, nullable=False)
    port = Column(Integer, nullable=False)
    chute_id = Column(String, ForeignKey("chutes.chute_id"))
    gpus = Column(JSONB, nullable=False)
    miner_uid = Column(Integer, nullable=False)
    miner_hotkey = Column(String, nullable=False)
    region = Column(String)
    active = Column(Boolean, default=False)
    verified = Column(Boolean, default=False)
    last_queried_at = Column(DateTime(timezone=True))
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True))
    last_verified_at = Column(DateTime(timezone=True))

    chute = relationship("Chute", back_populates="instances", lazy="joined")

    __table_args__ = (
        Index("idx_chute_active_lastq", "chute_id", "active", "last_queried_at"),
    )
