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
        
        # Validate minimum size (header + TD quote body = 48 + 584 = 632 bytes minimum)
        if len(quote_bytes) < 632:
            raise InvalidQuoteError(f"Quote too small: {len(quote_bytes)} bytes (minimum 632)")
        
        # Parse TDX Quote Header (first 48 bytes)
        if len(quote_bytes) < 48:
            raise InvalidQuoteError("Quote too small for header")
            
        # Unpack header fields (little endian) - first 16 bytes of 48-byte header
        header_data = struct.unpack('<HHIIHH', quote_bytes[:16])
        version, att_key_type, att_key_data_0, att_key_data_1, tee_type, reserved = header_data
        
        # Validate TDX quote
        if version != 4:
            raise InvalidQuoteError(f"Invalid quote version: {version} (expected 4)")
        
        # TEE type can be 0x81 (standard TDX) or 0x9a93 (as seen in your quote)
        if tee_type not in [0x81, 0x9a93]:
            logger.warning(f"Unexpected TEE type: 0x{tee_type:04x}")
        
        # Parse TD Quote Body (starts at offset 48)
        td_quote_body_offset = 48
        if len(quote_bytes) < td_quote_body_offset + 584:  # TD Quote Body is 584 bytes
            raise InvalidQuoteError("Quote too small for TD Quote Body")
        
        td_quote_body = quote_bytes[td_quote_body_offset:]
        
        # Extract fields from TD Quote Body based on the C struct layout:
        # typedef struct {
        #     uint8_t tee_tcb_svn[16];    // TEE TCB SVN (offset 0)
        #     uint8_t mrseam[48];         // MRSEAM (offset 16) 
        #     uint8_t mrsigner_seam[48];  // MRSIGNERSEAM (offset 64)
        #     uint8_t seamattributes[8];  // SEAM attributes (offset 112)
        #     uint8_t tdattributes[8];    // TD attributes (offset 120)
        #     uint8_t xfam[8];            // XFAM (offset 128)
        #     uint8_t mrtd[48];           // MRTD (offset 136)
        #     uint8_t mrconfigid[48];     // MRCONFIGID (offset 184)
        #     uint8_t mrowner[48];        // MROWNER (offset 232)
        #     uint8_t mrownerconfig[48];  // MROWNERCONFIG (offset 280)
        #     uint8_t rtmrs[192];         // RTMR0-3 (4 x 48 bytes) (offset 328)
        #     uint8_t reportdata[64];     // User data (nonce) (offset 520)
        # } tdx_td_quote_body_t;
        
        # Extract MRTD (offset 136 from start of TD Quote Body)
        mrtd_offset = 136
        mrtd_bytes = td_quote_body[mrtd_offset:mrtd_offset + 48]
        mrtd = mrtd_bytes.hex()
        
        # Extract RTMRs (offset 328 from start of TD Quote Body)
        rtmr_offset = 328
        rtmr_bytes = td_quote_body[rtmr_offset:rtmr_offset + 192]  # 4 RTMRs × 48 bytes each
        
        # Extract individual RTMRs (48 bytes each)
        rtmr0 = rtmr_bytes[0:48].hex()
        rtmr1 = rtmr_bytes[48:96].hex()
        rtmr2 = rtmr_bytes[96:144].hex()
        rtmr3 = rtmr_bytes[144:192].hex()
        
        # Extract user data / reportdata (offset 520 from start of TD Quote Body)
        reportdata_offset = 520
        if len(td_quote_body) >= reportdata_offset + 64:
            reportdata_bytes = td_quote_body[reportdata_offset:reportdata_offset + 64]
            
            # Based on your C program output:
            # The C program now stores raw text directly: memcpy(req.reportdata, user_data, len)
            # So for nonce "abc123", we get bytes: [0x61, 0x62, 0x63, 0x31, 0x32, 0x33, 0x00, ...]
            # This is the original nonce as UTF-8 bytes, null-padded to 64 bytes
            
            if any(reportdata_bytes):
                try:
                    # Remove trailing null bytes from the 64-byte field
                    user_data_trimmed = reportdata_bytes.rstrip(b'\x00')
                    
                    # Decode as UTF-8 to get the original nonce
                    user_data = user_data_trimmed.decode('utf-8')
                    logger.debug(f"Extracted nonce from reportdata: {user_data}")
                    
                except UnicodeDecodeError as e:
                    logger.warning(f"Reportdata is not valid UTF-8, using hex representation: {e}")
                    # Fallback: use the hex representation
                    user_data = user_data_trimmed.hex()
                except Exception as e:
                    logger.error(f"Failed to process reportdata: {e}")
                    # Final fallback: use the raw hex representation
                    user_data = reportdata_bytes.rstrip(b'\x00').hex()
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