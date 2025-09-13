"""
TDX quote parsing, crypto operations, and server helper functions.
"""

import secrets
import struct
from datetime import datetime, timezone
from typing import Dict, Any, Optional
from dataclasses import dataclass
from loguru import logger
from dcap_qvl import get_collateral_and_verify
from api.config import settings
from api.server.exceptions import InvalidQuoteError, MeasurementMismatchError


@dataclass
class TdxQuote:
    """
    Parsed TDX quote with extracted measurements.
    """
    version: int
    att_key_type: int
    tee_type: int
    mrtd: str
    rtmr0: str
    rtmr1: str
    rtmr2: str
    rtmr3: str
    user_data: Optional[str]
    raw_quote_size: int
    parsed_at: str
    verification_result: Optional[Dict[str, Any]] = None
    
    @property
    def rtmrs(self) -> Dict[str, str]:
        """Get RTMRs as a dictionary."""
        return {
            "rtmr0": self.rtmr0,
            "rtmr1": self.rtmr1,
            "rtmr2": self.rtmr2,
            "rtmr3": self.rtmr3,
        }
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format for compatibility."""
        return {
            "quote_version": str(self.version),
            "mrtd": self.mrtd,
            "rtmrs": self.rtmrs,
            "user_data": self.user_data,
            "raw_quote_size": self.raw_quote_size,
            "parsed_at": self.parsed_at,
            "tcb_level": self.tcb_level,
            "verification_result": self.verification_result,
            "header": {
                "version": self.version,
                "att_key_type": self.att_key_type,
                "tee_type": f"0x{self.tee_type:02x}",
            }
        }


def generate_nonce() -> str:
    """Generate a cryptographically secure nonce."""
    return secrets.token_hex(32)


def get_nonce_expiry_seconds(minutes: int = 10) -> int:
    """Get expiry time for a nonce in seconds."""
    return minutes * 60


def parse_tdx_quote(quote_bytes: bytes) -> TdxQuote:
    """
    Parse TDX quote using manual byte parsing based on TDX quote structure.
    
    Args:
        quote_bytes: Raw quote bytes
        
    Returns:
        TdxQuote object with parsed data
        
    Raises:
        InvalidQuoteError: If parsing fails
    """
    try:
        # Validate minimum size (header + TD report = 48 + 584)
        if len(quote_bytes) < 632:
            raise InvalidQuoteError(f"Quote too short: {len(quote_bytes)} bytes")

        # Parse header (48 bytes, little-endian)
        header_format = "<HHI16s20s"  # uint16 version, uint16 att_key_type, uint32 tee_type, 16s QE Vendor ID, 20s User Data
        header = struct.unpack_from(header_format, quote_bytes, 0)
        version, att_key_type, tee_type, qe_vendor_id, header_user_data = header

        # Validate header
        if version != 4:
            raise InvalidQuoteError(f"Invalid quote version: {version} (expected 4)")
        if tee_type != 0x81:
            raise InvalidQuoteError(f"Invalid TEE type: {tee_type:08x} (expected 0x81 for TDX)")
        if att_key_type not in (2, 3):  # ECDSA-256 or ECDSA-384
            raise InvalidQuoteError(f"Invalid attestation key type: {att_key_type}")

        # TD report starts at offset 48
        td_report = quote_bytes[48:]

        # Extract fields using corrected offsets from Intel API verification
        mrtd = td_report[136:184].hex().upper()
        rtmr0 = td_report[328:376].hex().upper()
        rtmr1 = td_report[376:424].hex().upper()
        rtmr2 = td_report[424:472].hex().upper()
        rtmr3 = td_report[472:520].hex().upper()
        report_data = td_report[520:584]

        # Extract nonce from report_data (first printable ASCII portion)
        user_data = ""
        for i, b in enumerate(report_data):
            if b == 0 or not (32 <= b <= 126):  # Stop at null or non-printable
                break
            user_data += chr(b)

        # Create TdxQuote object
        quote = TdxQuote(
            version=version,
            att_key_type=att_key_type,
            tee_type=tee_type,
            mrtd=mrtd,
            rtmr0=rtmr0,
            rtmr1=rtmr1,
            rtmr2=rtmr2,
            rtmr3=rtmr3,
            user_data=user_data,
            raw_quote_size=len(quote_bytes),
            parsed_at=datetime.now(timezone.utc).isoformat()
        )

        logger.success(f"Successfully parsed TDX quote: MRTD={quote.mrtd[:16]}...")
        return quote

    except Exception as e:
        logger.error(f"Failed to parse quote: {e}")
        raise InvalidQuoteError(f"Failed to parse quote: {str(e)}")


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
        user_data_trimmed = reportdata_bytes.rstrip(b'\x00')
        
        # Decode as UTF-8 to get the original nonce
        user_data = user_data_trimmed.decode('utf-8')
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
        return reportdata_bytes.rstrip(b'\x00').hex()


def _validate_measurements(quote: TdxQuote) -> None:
    """Validate that extracted measurements have correct format."""
    if len(quote.mrtd) != 96:  # 48 bytes = 96 hex chars
        raise InvalidQuoteError(f"Invalid MRTD length: {len(quote.mrtd)} (expected 96)")
    
    for rtmr_name, rtmr_value in quote.rtmrs.items():
        if len(rtmr_value) != 96:  # 48 bytes = 96 hex chars
            raise InvalidQuoteError(f"Invalid {rtmr_name} length: {len(rtmr_value)} (expected 96)")


async def verify_quote(quote_bytes: bytes) -> bool:
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
    verification_result = await get_collateral_and_verify(
        quote_bytes, 
        # "https://localhost:8081/sgx/certification/v4"
    )
    
    # Check if verification was successful
    is_valid = verification_result.get('status') == 'OK'
    
    if is_valid:
        logger.success("TDX quote signature verification successful")
    else:
        error_msg = verification_result.get('error', 'Unknown verification error')
        logger.error(f"TDX quote signature verification failed: {error_msg}")
    
    return is_valid


# def verify_quote_with_collateral(quote_bytes: bytes) -> Dict[str, Any]:
#     """
#     Verify quote against Intel's collateral and return detailed results.
    
#     Args:
#         quote_bytes: Raw TDX quote bytes
        
#     Returns:
#         Dictionary containing verification results
        
#     Raises:
#         InvalidQuoteError: If verification fails or library not available
#     """

#     try:
#         logger.info("Performing full TDX quote verification with collateral")
        
#         # Perform comprehensive verification
#         verification_result = quote_verification.verify_quote_with_collateral(quote_bytes)
        
#         logger.info(f"Quote verification completed with status: {verification_result.get('status')}")
#         return verification_result
        
#     except Exception as e:
#         logger.error(f"Quote collateral verification failed: {e}")
#         raise InvalidQuoteError(f"Quote verification failed: {str(e)}")


def verify_boot_measurements(quote: TdxQuote) -> bool:
    """
    Verify boot-time measurements against expected values.
    
    Args:
        quote: Parsed TDX quote
        
    Returns:
        True if measurements match expected values
        
    Raises:
        MeasurementMismatchError: If measurements don't match
    """
    try:
        expected_mrtd = settings.boot_expected_mrtd
        if not expected_mrtd:
            logger.warning("No expected boot MRTD configured")
            return True  # Skip verification if not configured
            
        if quote.mrtd.upper() != expected_mrtd.upper():
            logger.error(f"MRTD mismatch: expected {expected_mrtd}, got {quote.mrtd}")
            raise MeasurementMismatchError(f"MRTD verification failed")
            
        logger.info("Boot measurements verified successfully")
        return True
        
    except MeasurementMismatchError:
        raise
    except Exception as e:
        logger.error(f"Boot measurement verification failed: {e}")
        raise MeasurementMismatchError(f"Measurement verification error: {str(e)}")


def verify_runtime_measurements(quote: TdxQuote, expected_measurements: Dict[str, Any]) -> bool:
    """
    Verify runtime measurements against expected values.
    
    Args:
        quote: Parsed TDX quote
        expected_measurements: Expected MRTD and RTMRs
        
    Returns:
        True if all measurements match
        
    Raises:
        MeasurementMismatchError: If any measurements don't match
    """
    try:
        # Verify MRTD
        expected_mrtd = expected_measurements.get('mrtd')
        if expected_mrtd:
            if quote.mrtd.upper() != expected_mrtd.upper():
                logger.error(f"Runtime MRTD mismatch: expected {expected_mrtd}, got {quote.mrtd}")
                raise MeasurementMismatchError("Runtime MRTD verification failed")
        
        # Verify RTMRs
        expected_rtmrs = expected_measurements.get('rtmrs', {})
        
        for rtmr_name, expected_value in expected_rtmrs.items():
            actual_value = quote.rtmrs.get(rtmr_name)
            if actual_value and actual_value.upper() != expected_value.upper():
                logger.error(f"RTMR {rtmr_name} mismatch: expected {expected_value}, got {actual_value}")
                raise MeasurementMismatchError(f"RTMR {rtmr_name} verification failed")
        
        logger.info("Runtime measurements verified successfully")
        return True
        
    except MeasurementMismatchError:
        raise
    except Exception as e:
        logger.error(f"Runtime measurement verification failed: {e}")
        raise MeasurementMismatchError(f"Runtime measurement verification error: {str(e)}")


def extract_nonce_from_quote(quote: TdxQuote) -> Optional[str]:
    """
    Extract the nonce from the quote user data.
    
    Args:
        quote: Parsed TDX quote
        
    Returns:
        Extracted nonce string or None if not found
    """
    try:
        if quote.user_data is None:
            return None
            
        # The user data already contains the decoded nonce string
        nonce = quote.user_data.strip()
        
        # Validate it looks like a nonce (64 hex characters)
        if len(nonce) == 64 and all(c in '0123456789abcdef' for c in nonce.lower()):
            return nonce
        else:
            logger.warning(f"User data doesn't look like a nonce: '{nonce}' (length: {len(nonce)})")
            return nonce  # Return anyway, let validation decide
            
    except Exception as e:
        logger.error(f"Failed to extract nonce from user data: {e}")
        return None


def get_luks_passphrase() -> str:
    """
    Get the LUKS passphrase for disk decryption.
    
    Returns:
        LUKS passphrase string
    """
    # TODO: Implement secure passphrase retrieval
    # This could come from:
    # - Environment variable
    # - K8s secret
    # - Secure key management service
    
    passphrase = settings.luks_passphrase
    if not passphrase:
        logger.warning("No LUKS passphrase configured")
        # Return a placeholder for now
        passphrase = "placeholder_luks_passphrase"
    
    return passphrase