"""
Safer response class for user.
"""

from typing import Optional
from pydantic import BaseModel, computed_field
from datetime import datetime
from api.api_key.response import APIKeyCreationResponse
from api.permissions import Permissioning, Role


class UserResponse(BaseModel):
    username: str
    user_id: str
    logo_id: Optional[str] = None
    created_at: datetime

    class Config:
        from_attributes = True

    @computed_field
    @property
    def logo(self) -> Optional[str]:
        return f"https://logos.chutes.ai/logos/{self.logo_id}.webp" if self.logo_id else None


class RegistrationResponse(UserResponse):
    hotkey: str
    coldkey: str
    payment_address: str
    developer_payment_address: str
    fingerprint: str
    api_key: Optional[APIKeyCreationResponse] = None


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
