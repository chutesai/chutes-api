"""
Schemas/definitions for invocations.
"""

from pydantic import BaseModel


class Report(BaseModel):
    reason: str
