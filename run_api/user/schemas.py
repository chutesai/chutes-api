"""
ORM definitions for users.
"""

import re
from fastapi import Depends, Header, HTTPException, status
from sqlalchemy import func, Column, String, DateTime
from sqlalchemy.orm import relationship, validates
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from passlib.context import CryptContext
from run_api.database import Base, get_db_session, generate_uuid

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
        # TODO: integrate with Nam's auth stuff or replace.
        # XXX: actual code should be something like this:
        # return api_key_context.verify(api_key, self.api_key_hash)
        return api_key == "TEST"

    def __repr__(self):
        return f"<User(user_id={self.user_id}>"


async def get_current_user(
    user_id: str = Header(..., alias="X-Parachutes-UserID"),
    authorization: str = Header(..., alias="Authorization"),
    db: AsyncSession = Depends(get_db_session),
) -> User:
    """
    Load the current user from the database.
    """
    token = authorization.split(" ")[1] if " " in authorization else None
    if not token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing authorization token",
        )
    query = select(User).where(User.user_id == user_id)
    result = await db.execute(query)
    user = result.scalar_one_or_none()
    if not user or not user.verify_api_key(token):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token or user not found",
        )
    return user
