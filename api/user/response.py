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


class SelfResponse(UserResponse):
    hotkey: str
    coldkey: str
    payment_address: str
    developer_payment_address: str
    balance: float
    bonus_used: bool
    permissions_bitmask: int

    @computed_field
    @property
    def permissions(self) -> list[str]:
        permissions = []
        for role_str in dir(Permissioning):
            if isinstance(role := getattr(Permissioning, role_str, None), Role):
                if self.permissions_bitmask & role.bitmask == role.bitmask:
                    permissions.append(role.description)
        return permissions
