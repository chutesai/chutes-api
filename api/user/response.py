"""
Safer response class for user.
"""

from pydantic import BaseModel, computed_field
from datetime import datetime
from api.permissions import Permissioning, Role


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


class SelfResponse(RegistrationResponse):
    balance: str
    bonus_used: bool
    permissions_bitmask: int

    @computed_field
    @property
    def permissions(self) -> list[str]:
        return [
            role
            for role in dir(Permissioning)
            if isinstance(getattr(Permissioning, role, None), Role)
            and self.permissions_bitmask & role.bitmask == role.bitmask
        ]
