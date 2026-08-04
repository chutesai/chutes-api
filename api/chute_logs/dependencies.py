"""FastAPI dependencies for the chute log shipper — the transport boundary.

These are strictly request-handling wiring, not business logic, so they live outside
``service.py``. A dependency handles the request directly, so it maps the service's pure domain
errors (``exceptions.py``) to HTTP here; the business logic underneath stays transport-free.
"""

from cryptography.x509 import Certificate
from fastapi import Depends, HTTPException

from api.chute_logs.exceptions import LogCaptureError
from api.chute_logs.service import LogCaptureContext, authenticate_shipment
from api.server.util import extract_client_cert


def resolve_log_context():
    """FastAPI dependency: authenticate an incoming log shipment → ``LogCaptureContext``.

    Composes ``extract_client_cert`` with the service's (Redis-cached) shipment auth, and maps
    the service's domain error to an HTTP status at this boundary — the service itself never
    raises ``HTTPException``.
    """

    async def _dep(
        config_id: str,
        client_cert: Certificate = Depends(extract_client_cert()),
    ) -> LogCaptureContext:
        try:
            return await authenticate_shipment(config_id, client_cert)
        except LogCaptureError as exc:
            raise HTTPException(status_code=exc.http_status, detail=exc.message)

    return _dep
