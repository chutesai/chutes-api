"""
Server and attestation-specific exceptions.

Paradigm: internal service/util layers raise these pure domain exceptions; they carry
no HTTP or logging knowledge. The layer that detects a failure logs the private reason
(raw verifier output, struct errors, IPs, ...) right at the raise site -- where ambient
request identity is already bound (see api.log) -- and then raises with only a safe,
client-facing message. Each HTTP boundary that surfaces one maps it to an
``HTTPException(e.http_status, e.message)``; it does not re-log.

Each ``AttestationError`` carries only:
  - ``message``     -> safe, client-facing text (also exposed as ``detail`` for back-compat)
  - ``code``        -> stable machine-readable slug for server-side log filtering
  - ``http_status`` -> the status a boundary maps it to (also exposed as ``status_code``)

Security boundary: never put reference measurement values, nonces, tokens, cert hashes,
or raw verifier output in ``message`` -- log those at the raise site instead.
"""

from typing import Optional

from fastapi import HTTPException, status


class AttestationError(Exception):
    """
    Base for attestation/verification domain errors raised by server service/util layers.

    Not an HTTPException -- the surfacing boundary decides the HTTP response. Exposes
    ``status_code`` and ``detail`` as aliases so callers/logs that read those attributes
    keep working unchanged.
    """

    http_status: int = status.HTTP_403_FORBIDDEN
    code: str = "attestation_error"
    default_message: str = "Attestation failed."

    def __init__(
        self,
        detail: Optional[str] = None,
        *,
        status_code: Optional[int] = None,
        code: Optional[str] = None,
    ):
        self.message = detail if detail is not None else self.default_message
        if status_code is not None:
            self.http_status = status_code
        if code is not None:
            self.code = code
        super().__init__(self.message)

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
    default_message = (
        "No mTLS client certificate presented. Attestation must be performed through the CVM "
        "proxy (cvm.chutes.ai) so the VM's per-boot certificate is attached."
    )


class InvalidClientCertError(AttestationError):
    """Raised when the client certificate does not match the expected hash."""

    http_status = status.HTTP_403_FORBIDDEN
    code = "invalid_client_cert"
    default_message = (
        "The mTLS client certificate does not match the one bound to this attestation. The VM "
        "must present the same per-boot certificate whose hash is in the quote's report data."
    )


class InvalidQuoteError(AttestationError):
    """Raised when TDX quote is invalid or malformed."""

    http_status = status.HTTP_403_FORBIDDEN
    code = "invalid_quote"
    default_message = (
        "The TDX quote could not be parsed. Check that the guest is running on TDX hardware and "
        "that the attestation service produced a complete quote."
    )


class InvalidSignatureError(AttestationError):
    """Raised when TDX quote signature verification fails."""

    http_status = status.HTTP_403_FORBIDDEN
    code = "invalid_quote_signature"
    default_message = (
        "TDX quote signature verification failed. The quote could not be verified against Intel's "
        "collateral -- check the host's TDX module and PCCS/collateral freshness."
    )


class MeasurementMismatchError(AttestationError):
    """Raised when measurements don't match expected values.

    The default message is the generic "no recognized measurement" text used for every
    non-informative rejection -- a quote matching nothing, and (deliberately identical, so it leaks
    nothing) a release-candidate measurement the caller isn't authorized for. Raise it bare
    (``MeasurementMismatchError()``) for those; pass an explicit message only where revealing the
    reason to the operator is intended and safe (e.g. GPU-count or version-outdated at registration).
    """

    http_status = status.HTTP_403_FORBIDDEN
    code = "measurement_mismatch"
    # Read aloud by the sek8s initramfs on a failed boot, so it says what to DO. The wording is
    # identical for "nothing matched" and "rc you aren't authorized for" -- that is the point: the
    # two must stay indistinguishable, so the message may not name which one happened.
    default_message = (
        "No registered measurement matches this host's topology x QEMU version. If this hardware "
        "is new, run `chutes-cvm discover-profile` and submit the profile so measurements can be "
        "generated; otherwise check that the VM image and QEMU version are supported."
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
    default_message = "Invalid GPU evidence. Please ensure your GPUs support attestation and are properly configured."


class NonceError(AttestationError):
    """Raised when nonce validation fails."""

    http_status = status.HTTP_400_BAD_REQUEST
    code = "nonce_error"
    default_message = (
        "Nonce is invalid, expired, or already used. Fetch a fresh one from GET /servers/nonce "
        "immediately before attesting; nonces are single-use and short-lived."
    )


class UnauthorizedError(AttestationError):
    """Raised when the caller failed to prove it holds the hotkey it is acting for."""

    http_status = status.HTTP_401_UNAUTHORIZED
    code = "unauthorized"
    default_message = (
        "Hotkey authentication is missing or invalid. Send X-Chutes-Hotkey, X-Chutes-Nonce (the "
        "server-issued nonce used for this call), and X-Chutes-Signature -- a hex sr25519 "
        "signature over '{hotkey}:{nonce}:{sha256(body)}'."
    )


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
