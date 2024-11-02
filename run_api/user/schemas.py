"""
ORM definitions for users.
"""

import re
from pydantic import BaseModel
from sqlalchemy import func, Column, String, DateTime
from sqlalchemy.orm import relationship, validates
from run_api.database import Base


class UserRequest(BaseModel):
    username: str
    commission_address: str


class User(Base):
    __tablename__ = "users"

    user_id = Column(String, primary_key=True)
    hotkey = Column(String, nullable=False)
    commission_address = Column(String, nullable=False)
    username = Column(String, unique=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now())

    chutes = relationship("Chute", back_populates="user")
    images = relationship("Image", back_populates="user")

    @validates("username")
    def validate_username(self, _, value):
        """
        Simple username validation.
        """
        if not re.match(r"^[a-zA-Z0-9_]{3,15}$", value):
            raise ValueError(
                "Username must be 3-15 characters and contain only alphanumeric/underscore characters"
            )
        return value

    def __repr__(self):
        """
        String representation.
        """
        return f"<User(user_id={self.user_id}>"
