"""
ORM definitions for metagraph nodes.
"""

from sqlalchemy.sql import func
from sqlalchemy.orm import relationship
from sqlalchemy import Column, String, DateTime, Integer, Float


def create_metagraph_node_class(base):
    """
    Instantiate our metagraph node class from a dynamic declarative base.
    """

    class MetagraphNode(base):
        __tablename__ = "metagraph_nodes"
        hotkey = Column(String, primary_key=True)
        checksum = Column(String, nullable=False)
        coldkey = Column(String, nullable=False)
        node_id = Column(Integer)
        incentive = Column(Float)
        netuid = Column(Integer)
        stake = Column(Float)
        trust = Column(Float)
        vtrust = Column(Float)
        last_updated = Column(Integer)
        ip = Column(String)
        ip_type = Column(Integer)
        port = Column(Integer)
        protocol = Column(Integer)
        real_host = Column(String)
        real_port = Column(Integer)
        synced_at = Column(DateTime, server_default=func.now())

        nodes = relationship(
            "Node",
            back_populates="miner",
            cascade="all, delete-orphan",
        )

    return MetagraphNode
