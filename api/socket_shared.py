"""
Socket.IO shared methods and classes.
"""

from fastapi import Request
import api.database.orms  # noqa
from typing import Dict


class SyntheticRequest(Request):
    """
    Synthetic requests to allow re-using our existing
    authentication logic within socket.io.
    """

    def __init__(self, headers: Dict[str, str]):
        scope = {
            "type": "http",
            "method": "GET",
            "headers": [(k.lower().encode(), v.encode()) for k, v in headers.items()],
            "query_string": b"",
        }
        super().__init__(scope)
