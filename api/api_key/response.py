"""
API key response classes.
"""

from datetime import datetime
from pydantic import BaseModel
from typing import List, Optional


class APIKeyScopeResponse(BaseModel):
    scope_id: str
    object_type: Optional[str] = None
    object_id: Optional[str] = None
    method: Optional[str] = None

    class Config:
        from_attributes = True


class APIKeyResponse(BaseModel):
    """
    Normal representation of API keys.
    """

    api_key_id: str
    user_id: str
    admin: bool
    name: str
    created_at: datetime
    last_used_at: Optional[datetime]
    scopes: Optional[List[APIKeyScopeResponse]]

    class Config:
        from_attributes = True


class APIKeyCreationResponse(APIKeyResponse):
    """
    Representation of an API key when it's initially created.
    """

    secret_key: Optional[str] = None
