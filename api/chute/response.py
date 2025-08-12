"""
Response class for Chutes, to hide sensitive data.
"""

from pydantic import BaseModel, computed_field
from typing import List, Optional, Dict, Any
from datetime import datetime
from api.user.response import UserResponse
from api.image.response import MinimalImageResponse, ImageResponse
from api.instance.response import MinimalInstanceResponse
from api.chute.schemas import Cord, Job
from api.config import settings


class MinimalChuteResponse(BaseModel):
    chute_id: str
    name: str
    public: bool
    version: str
    slug: str
    chutes_version: str
    image: MinimalImageResponse

    class Config:
        from_attributes = True


class ChuteResponse(BaseModel):
    chute_id: str
    name: str
    tagline: Optional[str]
    readme: str
    public: bool
    version: str
    tool_description: Optional[str]
    slug: Optional[str]
    standard_template: Optional[str]
    cords: Optional[List[Cord]] = []
    jobs: Optional[List[Job]] = []
    cord_ref_id: Optional[str] = None
    created_at: datetime
    updated_at: datetime
    image: ImageResponse
    user: UserResponse
    supported_gpus: List[str]
    node_selector: dict
    invocation_count: Optional[int] = 0
    current_estimated_price: Optional[Dict[str, Any]] = None
    instances: Optional[List[MinimalInstanceResponse]] = []
    logo_id: Optional[str]
    openrouter: Optional[bool] = False
    chutes_version: Optional[str] = None
    revision: Optional[str] = None
    concurrency: Optional[int] = None
    boost: Optional[float] = None
    logging_enabled: Optional[bool] = False

    class Config:
        from_attributes = True

    @computed_field
    @property
    def logo(self) -> Optional[str]:
        return f"{settings.logo_cdn}/logos/{self.logo_id}.webp" if self.logo_id else None

    @computed_field
    @property
    def hot(self) -> bool:
        return any([instance.active and instance.verified for instance in self.instances])
