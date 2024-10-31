"""
Response class for instances, to hide sensitive data.
"""

from pydantic import BaseModel
from typing import List, Dict, Any
from datetime import datetime
from run_api.chute.response import ChuteResponse


class InstanceResponse(BaseModel):
    instance_id: str
    gpus: List[Dict[str, Any]]
    miner_uid: str
    miner_hotkey: str
    region: str
    active: bool
    verified: bool
    created_at: datetime
    updated_at: datetime
    last_verified_at: datetime
    chute: ChuteResponse

    class Config:
        orm_mode = True
        from_attributes = True
