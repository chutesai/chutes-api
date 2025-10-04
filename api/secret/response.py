"""
Response object representation, to hide some fields if desired.
"""

from pydantic import BaseModel
from datetime import datetime


class SecretResponse(BaseModel):
    secret_id: str
    key: str
    purpose: str
    created_at: datetime

    class Config:
        from_attributes = True
