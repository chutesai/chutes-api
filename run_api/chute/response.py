"""
Response class for Chutes, to hide sensitive data.
"""

from pydantic import BaseModel
from typing import List
from datetime import datetime
from run_api.user.response import UserResponse
from run_api.image.response import ImageResponse
from run_api.chute.schemas import Cord


class ChuteResponse(BaseModel):
    chute_id: str
    name: str
    public: bool
    cords: List[Cord]
    created_at: datetime
    updated_at: datetime
    image: ImageResponse
    user: UserResponse

    class Config:
        orm_mode = True
        from_attributes = True
