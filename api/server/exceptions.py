"""
Server and attestation-specific exceptions.

Paradigm (server router): the *router boundary* is the only layer that speaks HTTP.
Internal service/util layers raise the domain exceptions below; they never build an
HTTP response. A single app-level handler (registered in api/main.py for
``AttestationError``) maps each domain exception to an HTTP response, logs the private
detail server-side, and returns only safe fields to the client.

Each ``AttestationError`` carries:
  - ``message``     -> safe, client-facing text (also exposed as ``detail`` for back-compat)
  - ``code``        -> stable machine-readable slug (e.g. "invalid_gpu_evidence")
  - ``http_status`` -> the status the boundary maps it to (also exposed as ``status_code``)
  - ``context``     -> safe structured fields to return to the client (e.g. failing GPU indices)
  - ``log_detail``  -> PRIVATE detail for server logs only (hashes, nonces, firmware, raw output)

``context`` and ``log_detail`` are the security boundary: never put reference measurement
values, nonces, tokens, cert hashes, or raw verifier output in ``message``/``context``.
"""

from typing import Any, Dict, Optional

from fastapi import HTTPException, status


class AttestationError(Exception):
    """
    Base for attestation/verification domain errors raised by server service/util layers.

    Not an HTTPException — the router boundary decides the HTTP response. Exposes
    ``status_code`` and ``detail`` as aliases so existing callers that read those
    attributes (and re-raise as HTTPException) keep working unchanged.
    """

    http_status: int = status.HTTP_403_FORBIDDEN
    code: str = "attestation_error"
    default_message: str = "Attestation failed."

    def __init__(
        self,
        detail: Optional[str] = None,
        *,
        status_code: Optional[int] = None,
        context: Optional[Dict[str, Any]] = None,
        log_detail: Optional[str] = None,
        log_fields: Optional[Dict[str, Any]] = None,
        code: Optional[str] = None,
    ):
        self.message = detail if detail is not None else self.default_message
        self.context: Dict[str, Any] = dict(context) if context else {}
        self.log_detail = log_detail
        # Structured tracing fields (server_id, ip, miner_hotkey, ...) bound onto the
        # server-side log line by the boundary handler. Populated by whichever layer has
        # the server context; safe to leave empty for deep raise sites that don't.
        self.log_fields: Dict[str, Any] = dict(log_fields) if log_fields else {}
        if status_code is not None:
            self.http_status = status_code
        if code is not None:
            self.code = code
        super().__init__(self.message)

    def trace(self, **fields: Any) -> "AttestationError":
        """Attach structured tracing fields (dropping None) and return self, so a catch
        site with server context can enrich then re-raise: ``raise e.trace(server_id=...)``."""
        self.log_fields.update({k: v for k, v in fields.items() if v is not None})
        return self

    # Back-compat aliases: callers/logs read e.detail; a couple of catch sites read
    # e.status_code before re-raising as HTTPException.
    @property
    def detail(self) -> str:
        return self.message

    @property
    def status_code(self) -> int:
        return self.http_status


class NoClientCertError(AttestationError):
    """Raised when attestation is performed without mTLS."""

    http_status = status.HTTP_403_FORBIDDEN
    code = "no_client_cert"
    default_message = "No client certificate found."


class InvalidClientCertError(AttestationError):
    """Raised when the client certificate does not match the expected hash."""

    http_status = status.HTTP_403_FORBIDDEN
    code = "invalid_client_cert"
    default_message = "Invalid client certificate provided."


class InvalidQuoteError(AttestationError):
    """Raised when TDX quote is invalid or malformed."""

    http_status = status.HTTP_403_FORBIDDEN
    code = "invalid_quote"
    default_message = "Invalid TDX quote."


class InvalidSignatureError(AttestationError):
    """Raised when TDX quote signature verification fails."""

    http_status = status.HTTP_403_FORBIDDEN
    code = "invalid_quote_signature"
    default_message = (
        "Invalid TDX quote signature. The attestation quote could not be verified."
    )


class MeasurementMismatchError(AttestationError):
    """Raised when measurements don't match expected values."""

    http_status = status.HTTP_403_FORBIDDEN
    code = "measurement_mismatch"
    default_message = (
        "Measurement verification failed. Please ensure your server is running the most recent VM."
    )


class GpuEvidenceError(AttestationError):
    """Raised for an unexpected error during GPU evidence verification."""

    http_status = status.HTTP_403_FORBIDDEN
    code = "gpu_evidence_error"
    default_message = (
        "Failed to verify GPU evidence. Please check your GPU attestation configuration."
    )


class InvalidGpuEvidenceError(AttestationError):
    """Raised for GPU evidence that fails verification."""

    http_status = status.HTTP_403_FORBIDDEN
    code = "invalid_gpu_evidence"
    default_message = (
        "Invalid GPU evidence. Please ensure your GPUs support attestation and are properly configured."
    )


class NonceError(AttestationError):
    """Raised when nonce validation fails."""

    http_status = status.HTTP_400_BAD_REQUEST
    code = "nonce_error"
    default_message = "Invalid or expired nonce"


class GetEvidenceError(AttestationError):
    """Raised when unable to retrieve attestation evidence from the server."""

    http_status = status.HTTP_403_FORBIDDEN
    code = "get_evidence_error"
    default_message = (
        "Failed to get evidence for attestation. Please ensure the server is accessible and the "
        "attestation service is running."
    )


class NoServerCertError(Exception):
    """Raised when attestation is performed without TLS (server side)."""

    def __init__(self, detail: str = "No server certificate found."):
        super().__init__(detail)


# --- Non-attestation server errors -------------------------------------------------
# These remain HTTPException subclasses for now: they are simple, already produce the
# right status, and (ChuteNotTeeError / InstanceNotFoundError) are consumed by the
# instance/chute routers. They are out of scope for the attestation-first migration.


class ServerNotFoundError(HTTPException):
    """Raised when server is not found."""

    def __init__(self, server_id: str):
        super().__init__(
            status_code=status.HTTP_404_NOT_FOUND, detail=f"Server {server_id} not found"
        )


class ServerRegistrationError(HTTPException):
    """Raised when server registration fails for non-attestation reasons (DB, unexpected)."""

    def __init__(self, detail: str = "Server registration failed"):
        super().__init__(status_code=status.HTTP_400_BAD_REQUEST, detail=detail)


class InvalidTdxConfiguration(HTTPException):
    """Raised if invalid configuration is encountered during TDX validation."""

    def __init__(self, detail: str = "Missing or invalid configuration for TDX verification."):
        super().__init__(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=detail)


class ChuteNotTeeError(HTTPException):
    """Raised when a chute is not TEE-enabled."""

    def __init__(self, chute_id: str):
        super().__init__(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Chute {chute_id} is not TEE-enabled",
        )


class InstanceNotFoundError(HTTPException):
    """Raised when an instance is not found."""

    def __init__(self, instance_id: str):
        super().__init__(
            status_code=status.HTTP_404_NOT_FOUND, detail=f"Instance {instance_id} not found"
        )
