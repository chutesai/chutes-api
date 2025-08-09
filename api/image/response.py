"""
Response object representation, to hide some fields if desired.
"""

from pydantic import BaseModel, computed_field
from datetime import datetime
from typing import Optional
from api.user.response import UserResponse
from api.config import settings


class MinimalImageResponse(BaseModel):
    image_id: str
    name: str
    tag: str
    public: bool
    patch_version: Optional[str]

    class Config:
        from_attributes = True


class ImageResponse(BaseModel):
    image_id: str
    name: str
    readme: str
    tag: str
    public: bool
    status: str
    created_at: datetime
    build_started_at: Optional[datetime]
    build_completed_at: Optional[datetime]
    user: UserResponse
    logo_id: Optional[str]
    patch_version: Optional[str]

    class Config:
        from_attributes = True

    @computed_field
    @property
    def logo(self) -> Optional[str]:
        return f"{settings.logo_cdn}logos/{self.logo_id}.webp" if self.logo_id else None
