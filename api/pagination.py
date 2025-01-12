"""
Helper to paginate list endpoints.
"""

from pydantic import BaseModel
from typing import List, Any, Optional


class PaginatedResponse(BaseModel):
    total: int
    page: int
    limit: int
    items: List[Any]
    cord_refs: Optional[dict] = None
