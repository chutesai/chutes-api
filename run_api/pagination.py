"""
Helper to paginate list endpoints.
"""

from pydantic import BaseModel
from typing import List, Any


class PaginatedResponse(BaseModel):
    total: int
    page: int
    limit: int
    items: List[Any]
