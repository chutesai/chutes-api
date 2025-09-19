"""
Server and attestation-specific exceptions.
"""

from fastapi import HTTPException, status


class AttestationError(HTTPException):
    """Base exception for attestation failures."""
    def __init__(self, detail: str, status_code: int = status.HTTP_403_FORBIDDEN):
        super().__init__(status_code=status_code, detail=detail)


class InvalidQuoteError(AttestationError):
    """Raised when TDX quote is invalid or malformed."""
    def __init__(self, detail: str = "Invalid TDX quote"):
        super().__init__(detail=detail)

class InvalidSignatureError(AttestationError):
    """Raised when TDX quote signature verification fails."""
    def __init__(self, detail: str = "Invalid TDX quote signature"):
        super().__init__(detail=detail)


class MeasurementMismatchError(AttestationError):
    """Raised when measurements don't match expected values."""
    def __init__(self, detail: str = "Measurement verification failed"):
        super().__init__(detail=detail)

class GpuEvidenceError(AttestationError):
    "Raised for an unexpected error during GPU evidence verification"
    def __init__(self, detail: str = "Failed to verify GPU evidence"):
        super().__init__(detail=detail)

class InvalidGpuEvidenceError(AttestationError):
    "Raised for invalid GPU evidence"
    def __init__(self, detail: str = "Invalid GPU evidence"):
        super().__init__(detail=detail)

class NonceError(AttestationError):
    """Raised when nonce validation fails."""
    def __init__(self, detail: str = "Invalid or expired nonce"):
        super().__init__(detail=detail, status_code=status.HTTP_400_BAD_REQUEST)


class ServerNotFoundError(HTTPException):
    """Raised when server is not found."""
    def __init__(self, server_id: str):
        super().__init__(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Server {server_id} not found"
        )


class ServerRegistrationError(HTTPException):
    """Raised when server registration fails."""
    def __init__(self, detail: str = "Server registration failed"):
        super().__init__(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=detail
        )