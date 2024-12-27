"""
ORM for invocation reports.
"""

from pydantic import BaseModel
from sqlalchemy import Column, String, DateTime, func
from api.database import Base


class Report(Base):
    __tablename__ = "reports"
    invocation_id = Column(String, nullable=False, primary_key=True)
    user_id = Column(String, nullable=False)
    timestamp = Column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )
    confirmed_at = Column(DateTime(timezone=True))
    confirmed_by = Column(String)
    reason = Column(String, nullable=False)


class ReportArgs(BaseModel):
    reason: str
