"""
Response class for Chutes, to hide sensitive data.
"""

from pydantic import BaseModel
from typing import Dict, Any
from datetime import datetime
from run_api.user.response import UserResponse
from run_api.image.response import ImageResponse


class ChuteResponse(BaseModel):
    chute_id: str
    name: str
    public: bool
    cords: Dict[str, Any]
    created_at: datetime
    updated_at: datetime
    image: ImageResponse
    user: UserResponse

    class Config:
        orm_mode = True
        from_attributes = True
