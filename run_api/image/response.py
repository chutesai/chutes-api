"""
Response object representation, to hide some fields if desired.
"""

from pydantic import BaseModel
from datetime import datetime
from run_api.user.response import UserResponse


class ImageResponse(BaseModel):
    image_id: str
    name: str
    tag: str
    public: bool
    created_at: datetime
    user: UserResponse

    class Config:
        orm_mode = True
        from_attributes = True
