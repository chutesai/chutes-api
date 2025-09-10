"""
TDX quote parsing, crypto operations, and server helper functions.
"""

import base64
import secrets
import struct
from datetime import datetime, timedelta, timezone
from typing import Dict, Any, Optional
from dataclasses import dataclass
from loguru import logger
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


def is_nonce_expired(expires_at: datetime) -> bool:
    """Check if a nonce has expired (deprecated - Redis TTL handles expiration)."""
    return datetime.now(timezone.utc) > expires_at


def parse_tdx_quote(quote_b64: str) -> TdxQuote:
    """
    Parse a TDX quote and extract relevant measurements.
    
    Args:
        quote_b64: Base64 encoded TDX quote
        
    Returns:
        TdxQuote object containing parsed quote data
        
    Raises:
        InvalidQuoteError: If quote is malformed or invalid
    """
    try:
        quote_bytes = base64.b64decode(quote_b64)
        
        logger.info(f"Parsing TDX quote of {len(quote_bytes)} bytes")
        
        # Validate minimum size (header + TD report = 16 + 584 = 600 bytes minimum)
        if len(quote_bytes) < 600:
            raise InvalidQuoteError(f"Quote too small: {len(quote_bytes)} bytes (minimum 600)")
        
        # Parse TDX Quote Header (first 16 bytes)
        if len(quote_bytes) < 16:
            raise InvalidQuoteError("Quote too small for header")
            
        # Unpack header fields (little endian)
        header_data = struct.unpack('<HHIIHH', quote_bytes[:16])
        version, att_key_type, att_key_data_0, att_key_data_1, tee_type, reserved = header_data
        
        # Validate TDX quote
        if version != 4:
            raise InvalidQuoteError(f"Invalid quote version: {version} (expected 4)")
        
        if tee_type != 0x81:  # TDX TEE type
            raise InvalidQuoteError(f"Invalid TEE type: 0x{tee_type:02x} (expected 0x81 for TDX)")
        
        # Parse TD Report (starts at offset 16)
        td_report_offset = 16
        if len(quote_bytes) < td_report_offset + 584:  # TD Report is 584 bytes
            raise InvalidQuoteError("Quote too small for TD Report")
        
        td_report = quote_bytes[td_report_offset:]
        
        # Extract fields from TD Report based on C struct layout:
        # uint8_t cpusvn[16];      // CPU Security Version Number (offset 0)
        # uint8_t tee_tcb_svn[16]; // TEE TCB SVN (offset 16) 
        # uint8_t mrseam[48];      // MRTD (Measurement of SEAM module) (offset 32)
        # uint8_t mrsigner_seam[48]; // Signer of SEAM module (offset 80)
        # uint8_t attributes[8];   // TDX attributes (offset 128)
        # uint8_t rtmrs[192];      // RTMR0-RTMR3 (4 x 48 bytes) (offset 136)
        # uint8_t user_data[64];   // User data field (offset 328)
        
        # Extract MRTD (offset 32 from start of TD Report)
        mrtd_offset = 32
        mrtd_bytes = td_report[mrtd_offset:mrtd_offset + 48]
        mrtd = mrtd_bytes.hex()
        
        # Extract RTMRs (offset 136 from start of TD Report) 
        # cpusvn(16) + tee_tcb_svn(16) + mrseam(48) + mrsigner_seam(48) + attributes(8) = 136
        rtmr_offset = 136
        rtmr_bytes = td_report[rtmr_offset:rtmr_offset + 192]  # 4 RTMRs × 48 bytes each
        
        # Extract individual RTMRs (48 bytes each)
        rtmr0 = rtmr_bytes[0:48].hex()
        rtmr1 = rtmr_bytes[48:96].hex()
        rtmr2 = rtmr_bytes[96:144].hex()
        rtmr3 = rtmr_bytes[144:192].hex()
        
        # Extract user data (offset 328 from start of TD Report)
        # cpusvn(16) + tee_tcb_svn(16) + mrseam(48) + mrsigner_seam(48) + attributes(8) + rtmrs(192) = 328
        user_data_offset = 328
        if len(td_report) >= user_data_offset + 64:
            user_data_bytes = td_report[user_data_offset:user_data_offset + 64]
            
            # The client script hex-encodes the nonce and pads it to 64 bytes
            # We need to decode this back to the original string to match Redis
            if any(user_data_bytes):
                try:
                    # Remove trailing null bytes from the 64-byte field
                    user_data_trimmed = user_data_bytes.rstrip(b'\x00')
                    
                    # The trimmed data should be the hex-encoded nonce
                    # Decode it back to the original string
                    user_data = user_data_trimmed.decode('utf-8')
                    logger.debug(f"Extracted user data from quote: {user_data[:16]}...")
                except UnicodeDecodeError as e:
                    logger.error(f"Failed to decode user data as UTF-8: {e}")
                    user_data = None
            else:
                user_data = None
        else:
            user_data = None
        
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
        
        # Validate extracted measurements
        if len(quote.mrtd) != 96:  # 48 bytes = 96 hex chars
            raise InvalidQuoteError(f"Invalid MRTD length: {len(quote.mrtd)} (expected 96)")
        
        for rtmr_name, rtmr_value in quote.rtmrs.items():
            if len(rtmr_value) != 96:  # 48 bytes = 96 hex chars
                raise InvalidQuoteError(f"Invalid {rtmr_name} length: {len(rtmr_value)} (expected 96)")
        
        logger.success(f"Successfully parsed TDX quote: MRTD={quote.mrtd[:16]}...")
        return quote
        
    except base64.binascii.Error as e:
        logger.error(f"Failed to decode base64 quote: {e}")
        raise InvalidQuoteError("Quote is not valid base64")
    except struct.error as e:
        logger.error(f"Failed to unpack quote header: {e}")
        raise InvalidQuoteError(f"Invalid quote structure: {e}")
    except Exception as e:
        logger.error(f"Failed to parse TDX quote: {e}")
        raise InvalidQuoteError(f"Failed to parse TDX quote: {str(e)}")


def verify_quote_signature(quote_bytes: bytes) -> bool:
    """
    Verify the cryptographic signature of a TDX quote.
    
    Args:
        quote_bytes: Raw TDX quote bytes
        
    Returns:
        True if signature is valid, False otherwise
    """
    try:
        # TODO: Implement actual signature verification
        # This would involve:
        # - Extracting the signature from the quote
        # - Verifying against Intel's certificate chain
        # - Checking certificate validity and trust chain
        
        logger.info("Verifying TDX quote signature (placeholder)")
        
        # Placeholder verification - always returns True for now
        return True
        
    except Exception as e:
        logger.error(f"Quote signature verification failed: {e}")
        return False


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
        expected_mrtd = getattr(settings, 'boot_expected_mrtd', None)
        if not expected_mrtd:
            logger.warning("No expected boot MRTD configured")
            return True  # Skip verification if not configured
            
        if quote.mrtd != expected_mrtd:
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
            if quote.mrtd != expected_mrtd:
                logger.error(f"Runtime MRTD mismatch: expected {expected_mrtd}, got {quote.mrtd}")
                raise MeasurementMismatchError("Runtime MRTD verification failed")
        
        # Verify RTMRs
        expected_rtmrs = expected_measurements.get('rtmrs', {})
        
        for rtmr_name, expected_value in expected_rtmrs.items():
            actual_value = quote.rtmrs.get(rtmr_name)
            if actual_value != expected_value:
                logger.error(f"RTMR {rtmr_name} mismatch: expected {expected_value}, got {actual_value}")
                raise MeasurementMismatchError(f"RTMR {rtmr_name} verification failed")
        
        logger.info("Runtime measurements verified successfully")
        return True
        
    except MeasurementMismatchError:
        raise
    except Exception as e:
        logger.error(f"Runtime measurement verification failed: {e}")
        raise MeasurementMismatchError(f"Runtime measurement verification error: {str(e)}")


def validate_nonce_in_quote(quote: TdxQuote, expected_nonce: str) -> bool:
    """
    Validate that the nonce is embedded in the quote user data.
    
    Args:
        quote: Parsed TDX quote
        expected_nonce: Expected nonce value
        
    Returns:
        True if nonce is valid
    """
    try:
        if quote.user_data is None:
            logger.error("No user data in quote for nonce validation")
            return False
            
        # The user data should contain the original nonce string
        # (already decoded from hex in parse_tdx_quote)
        embedded_nonce = quote.user_data.strip()
        
        if embedded_nonce != expected_nonce:
            logger.error(f"Nonce mismatch in quote user data: expected '{expected_nonce}', got '{embedded_nonce}'")
            return False
            
        logger.info(f"Successfully validated nonce in quote user data: {expected_nonce[:8]}...")
        return True
        
    except Exception as e:
        logger.error(f"Nonce validation failed: {e}")
        return False


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
    
    passphrase = getattr(settings, 'luks_passphrase', None)
    if not passphrase:
        logger.warning("No LUKS passphrase configured")
        # Return a placeholder for now
        passphrase = "placeholder_luks_passphrase"
    
    return passphrase