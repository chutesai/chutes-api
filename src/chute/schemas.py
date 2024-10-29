from sqlalchemy.sql import func
from sqlalchemy.orm import relationship
from sqlalchemy import Column, String, DateTime, Boolean, ForeignKey
from database import Base, generate_uuid


class Chute(Base):
    __tablename__ = "chutes"
    chute_id = Column(String, primary_key=True, default=generate_uuid)
    user_id = Column(String, ForeignKey("users.user_id"), nullable=False)
    name = Column(String)
    image_id = Column(String, ForeignKey("images.image_id"))
    public = Column(Boolean, default=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now())

    image = relationship("Image", back_populates="chutes", lazy="joined")
    user = relationship("User", back_populates="chutes")
