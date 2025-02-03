from pydantic import BaseModel
from datetime import datetime


class AuditEntryResponse(BaseModel):
    entry_id: str
    hotkey: str
    block: int
    path: str
    created_at: datetime
    start_time: datetime
    end_time: datetime

    class Config:
        from_attributes = True
