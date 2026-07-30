"""Chute-log-shipper domain exceptions.

Paradigm (mirrors ``api/server/exceptions.py``): the service business logic raises these pure
domain errors — no HTTP knowledge, no ``HTTPException``. The request-handling boundary maps them
to a response via ``http_status`` + ``message``. For shipment auth that boundary is the
``resolve_log_context`` FastAPI dependency (``api/chute_logs/dependencies.py``) — a dependency
handles the request directly, so it owns the HTTP mapping; the business logic in ``service.py``
stays transport-free.
"""

from typing import Optional

from fastapi import status


class LogCaptureError(Exception):
    """Base for chute-log-shipper domain errors.

    Not an ``HTTPException`` — the transport boundary decides the response, mapping
    ``http_status`` + ``message``.
    """

    http_status: int = status.HTTP_400_BAD_REQUEST
    default_message: str = "Log capture error."

    def __init__(self, message: Optional[str] = None):
        self.message = message or self.default_message
        super().__init__(self.message)


class UnknownLaunchConfig(LogCaptureError):
    """No launch config (or its chute) for the ``config_id`` — the guest treats this as terminal."""

    http_status = status.HTTP_404_NOT_FOUND
    default_message = "Unknown launch config."


class LogCaptureNotAuthorized(LogCaptureError):
    """The mTLS leaf did not verify against any registered VM CA for this launch config."""

    http_status = status.HTTP_403_FORBIDDEN
    default_message = (
        "Log shipment mTLS leaf does not match a registered VM CA for this launch config."
    )
