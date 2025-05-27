"""
Response class for instances, to hide sensitive data.
"""

from pydantic import BaseModel
from typing import List, Dict, Any, Optional
from datetime import datetime


class InstanceResponse(BaseModel):
    instance_id: str
    chute_id: str
    gpus: List[Dict[str, Any]]
    miner_uid: int
    miner_hotkey: str
    miner_coldkey: str
    region: str
    active: bool
    verified: bool
    chutes_version: Optional[str] = None
    created_at: datetime
    updated_at: Optional[datetime] = None
    last_verified_at: Optional[datetime] = None

    class Config:
        from_attributes = True


class MinimalInstanceResponse(BaseModel):
    instance_id: str
    region: str
    active: bool
    verified: bool
    last_verified_at: Optional[datetime] = None

    class Config:
        from_attributes = True
