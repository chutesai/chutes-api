"""
TDX quote parsing, crypto operations, and server helper functions.
"""

import secrets
from typing import Dict, Any, Optional
from urllib.parse import unquote
from aiohttp import ClientResponse
from fastapi import Request, status
from loguru import logger
from dcap_qvl import get_collateral_and_verify
from api.config import settings
from cryptography import x509
from cryptography.x509 import Certificate
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.backends import default_backend
from api.server.exceptions import InvalidSignatureError, InvalidTdxConfiguration, MeasurementMismatchError, NoClientCertError, NoServerCertError
from api.server.quote import TdxQuote, TdxVerificationResult
import hashlib

def generate_nonce() -> str:
    """Generate a cryptographically secure nonce."""
    return secrets.token_hex(32)


def get_nonce_expiry_seconds(minutes: int = 10) -> int:
    """Get expiry time for a nonce in seconds."""
    return minutes * 60

def extract_client_cert_hash():
    async def _extract_request_client_cert(
        request: Request
    ):
        try:
            cert = _get_client_certificate(request)
            cert_hash = _get_public_key_hash(cert)

            return cert_hash
        except Exception as e:
            logger.error(f"Boot attestation failed, no client cert provided:\n{e}")
            raise NoClientCertError(status_code=status.HTTP_401_UNAUTHORIZED, detail=str(e))

    return _extract_request_client_cert

def extract_server_cert_hash(response: ClientResponse):
    try:
        cert = _get_server_certificate(response)
        cert_hash = _get_public_key_hash(cert)

        return cert_hash
    except Exception as e:
        logger.error(f"Exception trying to extract cert hash from server cert:\n{e}")
        raise NoServerCertError(detail=str(e))

def _get_server_certificate(response: ClientResponse) -> bytes:
    """
    Extract client certificate from Uvicorn request.
    Simplified for FastAPI-to-FastAPI communication.
    """
    # Get the server certificate from the connection
    # The transport contains the SSL object with peer certificate info
    transport = response.connection.transport
    ssl_object = transport.get_extra_info('ssl_object')
    
    if ssl_object is None:
        raise ValueError("No SSL connection established")
    
    # Get the peer certificate in DER format
    cert_der = ssl_object.getpeercert(binary_form=True)
    
    if cert_der is None:
        raise ValueError("No peer certificate available")
    
    # Load the DER certificate
    cert = x509.load_der_x509_certificate(cert_der, default_backend())

    return cert

def _get_public_key_hash(cert: Certificate) -> str:
    """
    Compute SHA-256 hash of certificate's public key in DER format.
    This matches the bash snippet's logic:
    openssl x509 -pubkey -noout | openssl pkey -pubin -outform der | sha256sum
    """
    # Extract the public key
    public_key = cert.public_key()
    
    # Serialize public key to DER format (matching openssl pkey -outform der)
    public_key_der = public_key.public_bytes(
        encoding=serialization.Encoding.DER,
        format=serialization.PublicFormat.SubjectPublicKeyInfo
    )
    
    # Compute SHA-256 hash
    hash_digest = hashlib.sha256(public_key_der).hexdigest()
    
    return hash_digest

def _get_client_certificate(request: Request) -> bytes:
    """
    Extract client certificate from Uvicorn request.
    Simplified for FastAPI-to-FastAPI communication.
    """
    cert_header = request.headers.get("X-Client-Cert")
    if not cert_header:
        raise NoClientCertError(detail="No client certificate provided")
    
    # Decode the URL-encoded PEM cert from nginx
    cert_pem = unquote(cert_header).encode()

    # Parse the certificate
    cert = x509.load_pem_x509_certificate(cert_pem, default_backend())
    
    return cert


def extract_nonce(quote: TdxQuote):
    # Extract nonce from report_data (first printable ASCII portion)
    # nonce = ""
    # _bytes = bytes.fromhex(quote.report_data[:64])
    # for i, b in enumerate(_bytes):
    #     if b == 0 or not (32 <= b <= 126):  # Stop at null or non-printable
    #         break
    #     nonce += chr(b)

    # return nonce
    return quote.report_data[:64].lower()

def extract_cert_hash(quote: TdxQuote):
    return quote.report_data[64:128].lower()

def extract_report_data(quote: TdxQuote):
    # Extract nonce from report_data (first printable ASCII portion)
    nonce = extract_nonce(quote)
    cert_hash = extract_cert_hash(quote)

    return nonce, cert_hash


def _bytes_to_hex(data: Any) -> str:
    """Convert bytes to uppercase hex string, handling various input types."""
    if isinstance(data, bytes):
        return data.hex().upper()
    elif isinstance(data, str):
        return data.upper()
    else:
        return str(data).upper()


def _extract_user_data_from_bytes(reportdata_bytes: bytes) -> Optional[str]:
    """Extract user data from report data bytes."""
    if not reportdata_bytes or not any(reportdata_bytes):
        return None

    try:
        # Remove trailing null bytes from the 64-byte field
        user_data_trimmed = reportdata_bytes.rstrip(b"\x00")

        # Decode as UTF-8 to get the original nonce
        user_data = user_data_trimmed.decode("utf-8")
        logger.debug(f"Extracted nonce from reportdata: {user_data}")
        return user_data

    except UnicodeDecodeError as e:
        logger.warning(f"Reportdata is not valid UTF-8, using hex representation: {e}")
        # Fallback: use the hex representation
        user_data = user_data_trimmed.hex()
        return user_data
    except Exception as e:
        logger.error(f"Failed to process reportdata: {e}")
        # Final fallback: use the raw hex representation
        return reportdata_bytes.rstrip(b"\x00").hex()


async def verify_quote_signature(quote: TdxQuote) -> TdxVerificationResult:
    """
    Verify the cryptographic signature of a TDX quote using dcap-qvl.

    Args:
        quote_bytes: Raw TDX quote bytes
        verify_collateral: Whether to verify against Intel's collateral (requires PCCS)

    Returns:
        True if signature is valid, False otherwise
    """

    logger.info("Verifying TDX quote signature using dcap-qvl")

    # Perform quote verification
    verified_report = await get_collateral_and_verify(quote.raw_bytes)

    result = TdxVerificationResult.from_report(verified_report)

    if result.is_valid:
        logger.success("TDX quote signature verification successful")
    else:
        error_msg = verified_report.get("error", "Unknown verification error")
        logger.error(f"TDX quote signature verification failed: {error_msg}")
        raise InvalidSignatureError("TDX quote signature verification failed")

    return result


def verify_measurements(quote: TdxQuote) -> bool:
    """
    Verify quote measurements against expected values.

    Args:
        quote: Parsed TDX quote

    Returns:
        True if all measurements match

    Raises:
        MeasurementMismatchError: If any measurements don't match
    """
    expected_rtmrs = (
        settings.expected_boot_rmtrs
        if quote.quote_type == "boot"
        else settings.expected_runtime_rmtrs
    )
    logger.info("Verifying quote against configured MRTD and RTMRS.")
    return _verify_measurements(quote, expected_rtmrs)

def verify_result(quote: TdxQuote, result: TdxVerificationResult) -> bool:
    """
    Verify quote measurements against verification result values.

    Args:
        quote: Parsed TDX quote
        result: The verification result from DCAP

    Returns:
        True if all measurements match

    Raises:
        MeasurementMismatchError: If any measurements don't match
    """
    logger.info("Verifying quote against verification result MRTD and RTMRS.")
    return _verify_measurements(quote, result.rtmrs)


def _verify_measurements(quote: TdxQuote, expected_rtmrs: Dict[str, str]) -> bool:
    try:
        # Verify MRTD
        expected_mrtd = settings.expected_mrtd
        if quote.mrtd.upper() != expected_mrtd.upper():
            logger.error(f"MRTD mismatch: expected {expected_mrtd}, got {quote.mrtd}")
            raise MeasurementMismatchError("MRTD verification failed")

        # Verify RTMRs
        for rtmr_name, expected_value in expected_rtmrs.items():
            actual_value = quote.rtmrs.get(rtmr_name)
            if not actual_value:
                raise MeasurementMismatchError(f"Quote missing excepted RTMR[{rtmr_name}]")

            if actual_value.upper() != expected_value.upper():
                logger.error(
                    f"RTMR {rtmr_name} mismatch: expected {expected_value}, got {actual_value}"
                )
                raise MeasurementMismatchError(f"RTMR {rtmr_name} verification failed")

        logger.info("Measurements verified successfully")
        return True

    except MeasurementMismatchError:
        raise
    except Exception as e:
        logger.error(f"Runtime measurement verification failed: {e}")
        raise MeasurementMismatchError(f"Runtime measurement verification error: {str(e)}")


def get_luks_passphrase() -> str:
    """
    Get the LUKS passphrase for disk decryption.

    Returns:
        LUKS passphrase string
    """

    passphrase = settings.luks_passphrase
    if not passphrase:
        logger.error("No LUKS passphrase configured")
        raise InvalidTdxConfiguration("Missing LUKS phassphrase configuration")

    return passphrase
