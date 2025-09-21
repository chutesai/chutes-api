"""
TDX quote parsing, crypto operations, and server helper functions.
"""

import secrets
from typing import Dict, Any, Optional
from loguru import logger
from dcap_qvl import get_collateral_and_verify
from api.config import settings
from api.server.exceptions import InvalidSignatureError, InvalidTdxConfiguration, MeasurementMismatchError
from api.server.quote import TdxQuote, TdxVerificationResult


def generate_nonce() -> str:
    """Generate a cryptographically secure nonce."""
    return secrets.token_hex(32)


def get_nonce_expiry_seconds(minutes: int = 10) -> int:
    """Get expiry time for a nonce in seconds."""
    return minutes * 60


def extract_nonce(quote: TdxQuote):
    # Extract nonce from report_data (first printable ASCII portion)
    nonce = ""
    _bytes = bytes.fromhex(quote.user_data)
    for i, b in enumerate(_bytes):
        if b == 0 or not (32 <= b <= 126):  # Stop at null or non-printable
            break
        nonce += chr(b)

    return nonce


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
