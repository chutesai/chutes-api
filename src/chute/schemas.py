from sqlalchemy.sql import func
from sqlalchemy import Column, String, DateTime, Boolean, ForeignKey
from database import Base


class Chute(Base):
    __tablename__ = "chutes"
    user_id = Column(String, ForeignKey("users.user_id"), nullable=False)
    name = Column(String)
    uid = Column(String, primary=True)
    image = Column(String, primary_key=True)
    public = Column(Boolean, default=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now())
