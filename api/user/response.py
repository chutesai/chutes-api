"""
Safer response class for user.
"""

from pydantic import BaseModel
from datetime import datetime


class UserResponse(BaseModel):
    username: str
    user_id: str
    created_at: datetime

    class Config:
        from_attributes = True


class RegistrationResponse(UserResponse):
    hotkey: str
    coldkey: str
    payment_address: str
    developer_payment_address: str
    fingerprint: str
