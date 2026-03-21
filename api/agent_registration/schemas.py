"""
ORM and Pydantic models for agent registration.
"""

from typing import Optional
from pydantic import BaseModel
from sqlalchemy import Column, String, Double, BigInteger, DateTime, func
from api.database import Base, generate_uuid


class AgentRegistration(Base):
    __tablename__ = "agent_registrations"

    registration_id = Column(String, primary_key=True, default=generate_uuid)
    user_id = Column(String, nullable=False)
    hotkey = Column(String, nullable=False, unique=True)
    coldkey = Column(String, nullable=False)
    username = Column(String, nullable=False)
    payment_address = Column(String, nullable=False)
    wallet_secret = Column(String, nullable=False)
    received_amount = Column(Double, default=0.0)
    received_rao = Column(BigInteger, default=0)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
    deleted_at = Column(DateTime(timezone=True), nullable=True)


class AgentRegistrationRequest(BaseModel):
    hotkey: str
    coldkey: str
    signature: str
    username: Optional[str] = None


class AgentRegistrationResponse(BaseModel):
    registration_id: str
    user_id: str
    hotkey: str
    coldkey: str
    payment_address: str
    required_amount: float
    message: str

    class Config:
        from_attributes = True


class AgentRegistrationStatusResponse(BaseModel):
    registration_id: Optional[str] = None
    user_id: str
    hotkey: str
    coldkey: str
    payment_address: str
    received_amount: float
    required_amount: float
    status: str  # "pending_payment", "completed", "expired"
    message: str

    class Config:
        from_attributes = True


class AgentSetupRequest(BaseModel):
    hotkey: str
    signature: str


class AgentSetupResponse(BaseModel):
    user_id: str
    api_key: str
    hotkey_ss58address: str
    payment_address: str
    username: str
    config_ini: str
    setup_instructions: str
