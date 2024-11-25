"""
Response class for Chutes, to hide sensitive data.
"""

from pydantic import BaseModel
from typing import List, Optional, Dict, Any
from datetime import datetime
from api.user.response import UserResponse
from api.image.response import ImageResponse
from api.instance.response import MinimalInstanceResponse
from api.chute.schemas import Cord


class ChuteResponse(BaseModel):
    chute_id: str
    name: str
    public: bool
    version: str
    slug: Optional[str]
    cords: List[Cord]
    created_at: datetime
    updated_at: datetime
    image: ImageResponse
    user: UserResponse
    supported_gpus: List[str]
    current_estimated_price: Optional[Dict[str, Any]] = None
    instances: Optional[List[MinimalInstanceResponse]] = []

    class Config:
        from_attributes = True
