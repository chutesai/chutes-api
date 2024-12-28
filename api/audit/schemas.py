"""
Audit entry schemas.
"""

from sqlalchemy import (
    Column,
    String,
    DateTime,
    BigInteger,
    func,
)
from api.database import Base


class AuditEntry(Base):
    __tablename__ = "audit_entries"
    entry_id = Column(String, primary_key=True)
    hotkey = Column(String, nullable=False)
    block = Column(BigInteger, nullable=False)
    path = Column(String, nullable=False)
    created_at = Column(DateTime, server_default=func.now())
    start_time = Column(DateTime, nullable=False)
    end_time = Column(DateTime, nullable=False)
