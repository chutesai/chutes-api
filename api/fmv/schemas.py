"""
ORM for fair-market value tracking.
"""

from sqlalchemy import Column, String, Float, DateTime, func
from api.database import Base


class FMV(Base):
    __tablename__ = "fmv_history"
    ticker = Column(String, nullable=False, primary_key=True)
    timestamp = Column(
        DateTime(timezone=True),
        nullable=False,
        primary_key=True,
        server_default=func.now(),
    )
    price = Column(Float, nullable=False)
