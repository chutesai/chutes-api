"""
Response class for instances, to hide sensitive data.
"""

from pydantic import BaseModel
from typing import List, Dict, Any, Optional
from datetime import datetime
from api.chute.response import ChuteResponse


class InstanceResponse(BaseModel):
    instance_id: str
    gpus: List[Dict[str, Any]]
    miner_uid: int
    miner_hotkey: str
    miner_coldkey: str
    region: str
    active: bool
    verified: bool
    created_at: datetime
    updated_at: Optional[datetime] = None
    last_verified_at: Optional[datetime] = None
    chute: ChuteResponse

    class Config:
        from_attributes = True
