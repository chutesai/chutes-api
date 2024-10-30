"""
ORM definitions for users.
"""

import re
from sqlalchemy import func, Column, String, DateTime
from sqlalchemy.orm import relationship, validates
from passlib.context import CryptContext
from run_api.database import generate_uuid, Base

api_key_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


class User(Base):
    __tablename__ = "users"

    user_id = Column(String, primary_key=True, default=generate_uuid)
    username = Column(String, unique=True)
    api_key_hash = Column(String, nullable=False)
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

    def set_api_key(self, api_key: str):
        """
        Hash API keys before storing.
        """
        self.api_key_hash = api_key_context.hash(api_key)

    def verify_api_key(self, api_key: str) -> bool:
        """
        Verify the hash of an API key.
        """
        # TODO: integrate with Nam's auth stuff or whatever we want here.
        return api_key_context.verify(api_key, self.api_key_hash)

    def __repr__(self):
        """
        String representation.
        """
        return f"<User(user_id={self.user_id}>"
