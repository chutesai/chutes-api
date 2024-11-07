"""
ORM definitions for users.
"""

import re
from pydantic import BaseModel
from sqlalchemy import func, Column, String, DateTime, BigInteger
from sqlalchemy.orm import relationship, validates
from substrateinterface import SubstrateInterface
from run_api.database import Base
from run_api.config import settings


# Other fields are populated by listeners
# Except hotkey which is added in the header
# NOTE: Can we add hotkey here instead?
class UserRequest(BaseModel):
    username: str
    coldkey: str


class User(Base):
    __tablename__ = "users"

    # Populated in user/events based on fingerprint
    user_id = Column(String, primary_key=True)

    # An alternative to an API key.
    hotkey = Column(String, nullable=False)

    # To receive commission payments
    coldkey = Column(String, nullable=False)

    # Users should send to this address to top up
    payment_address = Column(String)

    # Current balance in Tao
    balance = Column(BigInteger, default=settings.signup_bonus_balance)

    # Friendly name for the frontend for chute creators
    username = Column(String, unique=True)

    # Gets populated in user/events.py to be a 16 digit alphanumeric which acts as an account id
    fingerprint = Column(String, nullable=False, unique=True)
    
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now())

    chutes = relationship("Chute", back_populates="user")
    images = relationship("Image", back_populates="user")
    api_keys = relationship(
        "APIKey", back_populates="user", cascade="all, delete-orphan"
    )

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

    # NOTE: The below can be deleted as coldkey is just for receiving payments.
    # @validates("coldkey")
    # def validate_coldkey(self, _, value):
    #     """
    #     Ensure the coldkey has a balance before allowing registration.
    #     """
    #     if not re.match(r"^[a-zA-Z0-9]{48}$", value):
    #         raise ValueError("Invalid coldkey address")
    #     balance = 0
    #     try:
    #         substrate = SubstrateInterface(settings.subtensor, ss58_format=42)
    #         result = substrate.query(
    #             module="System", storage_function="Account", params=[value]
    #         )
    #         balance = float(result.value["data"]["free"]) / 1e9
    #     except Exception as e:
    #         raise ValueError(f"Error checking tao balance for {value}: {e}")
    #     if not balance or balance < settings.registration_minimum_balance:
    #         raise ValueError(
    #             f"Free tao balance for coldkey={value} too low [{balance}], minimum: {settings.registration_minimum_balance}"
    #         )
    #     return value

    def __repr__(self):
        """
        String representation.
        """
        return f"<User(user_id={self.user_id}, username={self.username})>"
