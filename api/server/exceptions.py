"""
Server and attestation-specific exceptions.
"""

from fastapi import HTTPException, status


class AttestationError(HTTPException):
    """Base exception for attestation failures."""

    def __init__(self, detail: str, status_code: int = status.HTTP_403_FORBIDDEN):
        super().__init__(status_code=status_code, detail=detail)


class NoClientCertError(AttestationError):
    """Raised when attestation is performed without mTLS."""

    def __init__(self, detail: str = "No client certificate found."):
        super().__init__(detail=detail)


class NoServerCertError(Exception):
    """Raised when attestation is performed without TLS."""

    def __init__(self, detail: str = "No server certificate found."):
        super().__init__(detail=detail)


class InvalidClientCertError(AttestationError):
    """Raised when attestation is performed without mTLS."""

    def __init__(self, detail: str = "Invalid client certificate provided."):
        super().__init__(detail=detail)


class InvalidQuoteError(AttestationError):
    """Raised when TDX quote is invalid or malformed."""

    def __init__(self, detail: str = "Invalid TDX quote."):
        super().__init__(detail=detail)


class InvalidSignatureError(AttestationError):
    """Raised when TDX quote signature verification fails."""

    def __init__(
        self,
        detail: str = "Invalid TDX quote signature. The attestation quote could not be verified.",
    ):
        super().__init__(detail=detail)


class MeasurementMismatchError(AttestationError):
    """Raised when measurements don't match expected values."""

    def __init__(
        self,
        detail: str = "Measurement verification failed. Please ensure your server is running the most recent VM.",
    ):
        super().__init__(detail=detail)


class GpuEvidenceError(AttestationError):
    """Raised for an unexpected error during GPU evidence verification."""

    def __init__(
        self,
        detail: str = "Failed to verify GPU evidence. Please check your GPU attestation configuration.",
    ):
        super().__init__(detail=detail)


class InvalidGpuEvidenceError(AttestationError):
    """Raised for invalid GPU evidence."""

    def __init__(
        self,
        detail: str = "Invalid GPU evidence. Please ensure your GPUs support attestation and are properly configured.",
    ):
        super().__init__(detail=detail)


class NonceError(AttestationError):
    """Raised when nonce validation fails."""

    def __init__(self, detail: str = "Invalid or expired nonce"):
        super().__init__(detail=detail, status_code=status.HTTP_400_BAD_REQUEST)


class GetEvidenceError(AttestationError):
    """Raised when unable to retrieve attestation evidence from the server."""

    def __init__(
        self,
        detail: str = "Failed to get evidence for attestation. Please ensure the server is accessible and the attestation service is running.",
    ):
        super().__init__(detail=detail)


class ServerNotFoundError(HTTPException):
    """Raised when server is not found."""

    def __init__(self, server_id: str):
        super().__init__(
            status_code=status.HTTP_404_NOT_FOUND, detail=f"Server {server_id} not found"
        )


class ServerRegistrationError(HTTPException):
    """Raised when server registration fails."""

    def __init__(self, detail: str = "Server registration failed"):
        super().__init__(status_code=status.HTTP_400_BAD_REQUEST, detail=detail)


class InvalidTdxConfiguration(HTTPException):
    """Raised if invalid configuration is encouted during TDX validation."""

    def __init__(self, detail: str = "Missing or invalid configuration for TDX verification."):
        super().__init__(status_code=status.HTTP_400_BAD_REQUEST, detail=detail)


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
