"""
Response object representation, to hide some fields if desired.
"""

from pydantic import BaseModel
from datetime import datetime
from typing import Optional
from api.user.response import UserResponse


class ImageResponse(BaseModel):
    image_id: str
    name: str
    tag: str
    public: bool
    status: str
    created_at: datetime
    build_started_at: Optional[datetime]
    build_completed_at: Optional[datetime]
    user: UserResponse

    class Config:
        from_attributes = True
