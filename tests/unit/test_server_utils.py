"""
Unit tests for api/server/util module.
Tests TDX quote parsing, validation, and utility functions.
"""

import base64
import pytest
import secrets
from datetime import datetime, timezone, timedelta
from pathlib import Path
from unittest.mock import patch, Mock

from api.server.util import (
    TdxQuote,
    generate_nonce,
    get_nonce_expiry_seconds,
    parse_tdx_quote,
    verify_quote_signature,
    verify_boot_measurements,
    verify_runtime_measurements,
    validate_nonce_in_quote,
    extract_nonce_from_quote,
    get_luks_passphrase,
)
from api.server.exceptions import (
    InvalidQuoteError,
    MeasurementMismatchError,
)


# Test fixtures
@pytest.fixture
def valid_quote_bin():
    """Load the valid quote.bin file from test assets."""
    quote_path = Path("tests/assets/quote.bin")
    if not quote_path.exists():
        pytest.skip(f"Valid quote file not found at {quote_path}")
    
    with open(quote_path, "rb") as f:
        quote_bytes = f.read()
    
    return base64.b64encode(quote_bytes).decode('utf-8')


@pytest.fixture
def valid_quote_bytes(valid_quote_bin):
    """Return the decoded bytes of the valid quote."""
    return base64.b64decode(valid_quote_bin)


@pytest.fixture
def test_nonce():
    """Generate a test nonce."""
    return "test_nonce_123"


@pytest.fixture
def mock_settings():
    """Mock settings for testing."""
    settings = Mock()
    settings.boot_expected_mrtd = "a" * 96  # 48 bytes = 96 hex chars
    settings.luks_passphrase = "test_luks_passphrase"
    return settings


# TdxQuote class tests
def test_tdx_quote_creation():
    """Test TdxQuote dataclass creation and properties."""
    quote = TdxQuote(
        version=4,
        att_key_type=2,
        tee_type=0x9a93,
        mrtd="a" * 96,
        rtmr0="b" * 96,
        rtmr1="c" * 96,
        rtmr2="d" * 96,
        rtmr3="e" * 96,
        user_data="test_nonce_data",
        raw_quote_size=4096,
        parsed_at="2025-01-01T00:00:00Z"
    )
    
    assert quote.version == 4
    assert quote.tee_type == 0x9a93
    assert quote.mrtd == "a" * 96
    assert quote.user_data == "test_nonce_data"
    
    # Test rtmrs property
    expected_rtmrs = {
        "rtmr0": "b" * 96,
        "rtmr1": "c" * 96,
        "rtmr2": "d" * 96,
        "rtmr3": "e" * 96,
    }
    assert quote.rtmrs == expected_rtmrs


def test_tdx_quote_to_dict():
    """Test TdxQuote.to_dict() method."""
    quote = TdxQuote(
        version=4,
        att_key_type=2,
        tee_type=0x9a93,
        mrtd="a" * 96,
        rtmr0="b" * 96,
        rtmr1="c" * 96,
        rtmr2="d" * 96,
        rtmr3="e" * 96,
        user_data="test_data",
        raw_quote_size=4096,
        parsed_at="2025-01-01T00:00:00Z"
    )
    
    result = quote.to_dict()
    
    assert result["quote_version"] == "4"
    assert result["mrtd"] == "a" * 96
    assert result["user_data"] == "test_data"
    assert result["raw_quote_size"] == 4096
    assert result["parsed_at"] == "2025-01-01T00:00:00Z"
    assert result["header"]["version"] == 4
    assert result["header"]["tee_type"] == "0x9a93"
    assert "rtmrs" in result
    assert result["rtmrs"]["rtmr0"] == "b" * 96


def test_tdx_quote_rtmrs_property():
    """Test that RTMRs property returns correct dictionary."""
    quote = TdxQuote(
        version=4, att_key_type=2, tee_type=0x9a93,
        mrtd="a" * 96, 
        rtmr0="0" * 96, 
        rtmr1="1" * 96, 
        rtmr2="2" * 96, 
        rtmr3="3" * 96,
        user_data=None, raw_quote_size=4096,
        parsed_at="2025-01-01T00:00:00Z"
    )
    
    rtmrs = quote.rtmrs
    assert len(rtmrs) == 4
    assert rtmrs["rtmr0"] == "0" * 96
    assert rtmrs["rtmr1"] == "1" * 96
    assert rtmrs["rtmr2"] == "2" * 96
    assert rtmrs["rtmr3"] == "3" * 96


# Utility function tests
def test_generate_nonce():
    """Test nonce generation."""
    nonce1 = generate_nonce()
    nonce2 = generate_nonce()
    
    # Should be 64 characters (32 bytes as hex)
    assert len(nonce1) == 64
    assert len(nonce2) == 64
    
    # Should be different
    assert nonce1 != nonce2
    
    # Should be valid hex
    int(nonce1, 16)  # Will raise if not valid hex
    int(nonce2, 16)


def test_get_nonce_expiry_seconds():
    """Test nonce expiry calculation."""
    # Default (10 minutes)
    assert get_nonce_expiry_seconds() == 600



@pytest.mark.parametrize("minutes,expected_seconds", [
    (1, 60),
    (5, 300),
    (10, 600),
    (30, 1800),
    (60, 3600),
])
def test_get_nonce_expiry_seconds_parametrized(minutes, expected_seconds):
    """Test nonce expiry calculation with various inputs."""
    assert get_nonce_expiry_seconds(minutes) == expected_seconds


# Quote parsing tests - Valid cases
def test_parse_tdx_quote_valid(valid_quote_bin):
    """Test parsing a valid TDX quote."""
    quote = parse_tdx_quote(valid_quote_bin)
    
    assert isinstance(quote, TdxQuote)
    assert quote.version == 4
    assert quote.tee_type in [0x81, 0x9a93]  # Allow both TEE types
    assert len(quote.mrtd) == 96  # 48 bytes as hex
    assert len(quote.rtmr0) == 96
    assert len(quote.rtmr1) == 96
    assert len(quote.rtmr2) == 96
    assert len(quote.rtmr3) == 96
    assert quote.raw_quote_size > 0
    assert quote.parsed_at is not None


def test_parse_actual_quote_bin_if_exists():
    """Test parsing the actual quote.bin file if it exists (with abc123 nonce)."""
    quote_path = Path("tests/assets/quote.bin")
    if not quote_path.exists():
        pytest.skip("Actual quote.bin not found")
    
    with open(quote_path, "rb") as f:
        quote_bytes = f.read()
    
    quote_b64 = base64.b64encode(quote_bytes).decode('utf-8')
    quote = parse_tdx_quote(quote_b64)
    
    # Verify basic structure
    assert quote.version == 4
    assert quote.tee_type in [0x81, 0x9a93]
    assert len(quote.mrtd) == 96
    assert len(quote.rtmr0) == 96
    
    # If this quote was generated with "abc123" nonce, verify it
    if quote.user_data == "abc123":
        print("✓ Successfully extracted 'abc123' nonce from actual quote.bin")
    else:
        print(f"Note: Quote user data is: {quote.user_data!r}")


def test_parse_tdx_quote_with_user_data(valid_quote_bytes, test_nonce):
    """Test parsing quote with embedded nonce in user data."""
    # Modify the valid quote to include our test nonce in user data
    quote_bytes = bytearray(valid_quote_bytes)
    
    # User data is at offset 48 (header) + 520 (TD quote body offset) = 568
    user_data_offset = 48 + 520
    
    # Ensure we have enough space
    if len(quote_bytes) < user_data_offset + 64:
        # Extend the quote if needed
        quote_bytes.extend(b'\x00' * (user_data_offset + 64 - len(quote_bytes)))
    
    # Clear user data field and insert our nonce
    for i in range(64):
        quote_bytes[user_data_offset + i] = 0
    
    # Store the nonce as UTF-8 bytes (what the updated C program does)
    nonce_bytes = test_nonce.encode('utf-8')
    for i, byte in enumerate(nonce_bytes[:64]):  # Truncate if too long
        quote_bytes[user_data_offset + i] = byte
    
    # Encode and parse
    modified_quote_b64 = base64.b64encode(bytes(quote_bytes)).decode('utf-8')
    quote = parse_tdx_quote(modified_quote_b64)
    
    assert quote.user_data == test_nonce


def test_parse_tdx_quote_c_program_style_nonce(valid_quote_bytes):
    """Test parsing quote with nonce stored exactly like the C program does."""
    quote_bytes = bytearray(valid_quote_bytes)
    
    # User data is at offset 48 (header) + 520 (TD quote body offset) = 568
    user_data_offset = 48 + 520
    
    # Ensure we have enough space
    if len(quote_bytes) < user_data_offset + 64:
        quote_bytes.extend(b'\x00' * (user_data_offset + 64 - len(quote_bytes)))
    
    # Clear user data field
    for i in range(64):
        quote_bytes[user_data_offset + i] = 0
    
    # Simulate the exact C program behavior:
    # ./tdx-quote-generator -d "abc123"
    # memcpy(req.reportdata, user_data, strlen(user_data))
    
    original_nonce = "abc123"
    # Store as UTF-8 bytes (what C program actually does)
    nonce_bytes = original_nonce.encode('utf-8')
    
    # Store the bytes in user data (this is what the C program does)
    for i, byte in enumerate(nonce_bytes):
        quote_bytes[user_data_offset + i] = byte
    
    # Encode and parse
    modified_quote_b64 = base64.b64encode(bytes(quote_bytes)).decode('utf-8')
    quote = parse_tdx_quote(modified_quote_b64)
    
    # The parser should extract the original nonce
    assert quote.user_data == original_nonce


def test_parse_tdx_quote_long_nonce(valid_quote_bytes):
    """Test parsing quote with a longer nonce."""
    quote_bytes = bytearray(valid_quote_bytes)
    
    user_data_offset = 48 + 520
    
    if len(quote_bytes) < user_data_offset + 64:
        quote_bytes.extend(b'\x00' * (user_data_offset + 64 - len(quote_bytes)))
    
    # Clear user data field
    for i in range(64):
        quote_bytes[user_data_offset + i] = 0
    
    # Use a longer nonce that fills more of the 64-byte field
    long_nonce = "this_is_a_longer_test_nonce_with_more_characters_123456"
    nonce_bytes = long_nonce.encode('utf-8')
    
    # Store up to 64 bytes
    for i, byte in enumerate(nonce_bytes[:64]):
        quote_bytes[user_data_offset + i] = byte
    
    modified_quote_b64 = base64.b64encode(bytes(quote_bytes)).decode('utf-8')
    quote = parse_tdx_quote(modified_quote_b64)
    
    # Should extract the nonce (possibly truncated if over 64 bytes)
    expected_nonce = long_nonce if len(nonce_bytes) <= 64 else long_nonce[:64]
    assert quote.user_data == expected_nonce


def test_parse_tdx_quote_random_binary_user_data(valid_quote_bytes):
    """Test parsing quote with random binary data in user data."""
    quote_bytes = bytearray(valid_quote_bytes)
    
    user_data_offset = 48 + 520
    
    if len(quote_bytes) < user_data_offset + 64:
        quote_bytes.extend(b'\x00' * (user_data_offset + 64 - len(quote_bytes)))
    
    # Clear user data field
    for i in range(64):
        quote_bytes[user_data_offset + i] = 0
    
    # Store random binary data that's not valid UTF-8
    random_bytes = secrets.token_bytes(32)
    
    for i, byte in enumerate(random_bytes):
        quote_bytes[user_data_offset + i] = byte
    
    modified_quote_b64 = base64.b64encode(bytes(quote_bytes)).decode('utf-8')
    quote = parse_tdx_quote(modified_quote_b64)
    
    # Should fallback to hex representation
    expected_hex = random_bytes.hex()
    assert quote.user_data == expected_hex


def test_parse_tdx_quote_empty_user_data(valid_quote_bytes):
    """Test parsing quote with empty user data."""
    quote_bytes = bytearray(valid_quote_bytes)
    
    user_data_offset = 48 + 520
    
    if len(quote_bytes) < user_data_offset + 64:
        quote_bytes.extend(b'\x00' * (user_data_offset + 64 - len(quote_bytes)))
    
    # Clear user data field (all zeros)
    for i in range(64):
        quote_bytes[user_data_offset + i] = 0
    
    modified_quote_b64 = base64.b64encode(bytes(quote_bytes)).decode('utf-8')
    quote = parse_tdx_quote(modified_quote_b64)
    
    # Should be None for empty user data
    assert quote.user_data is None


def test_parse_tdx_quote_user_data_exactly_64_bytes(valid_quote_bytes):
    """Test user data that exactly fills the 64-byte field."""
    quote_bytes = bytearray(valid_quote_bytes)
    
    user_data_offset = 48 + 520
    
    if len(quote_bytes) < user_data_offset + 64:
        quote_bytes.extend(b'\x00' * (user_data_offset + 64 - len(quote_bytes)))
    
    # Create exactly 64 bytes of user data (no null terminator)
    nonce_64_bytes = "A" * 64
    nonce_bytes = nonce_64_bytes.encode('utf-8')
    
    for i, byte in enumerate(nonce_bytes):
        quote_bytes[user_data_offset + i] = byte
    
    modified_quote_b64 = base64.b64encode(bytes(quote_bytes)).decode('utf-8')
    quote = parse_tdx_quote(modified_quote_b64)
    
    assert quote.user_data == nonce_64_bytes


def test_parse_tdx_quote_unicode_user_data(valid_quote_bytes):
    """Test user data with unicode characters."""
    quote_bytes = bytearray(valid_quote_bytes)
    
    user_data_offset = 48 + 520
    
    if len(quote_bytes) < user_data_offset + 64:
        quote_bytes.extend(b'\x00' * (user_data_offset + 64 - len(quote_bytes)))
    
    # Clear user data field
    for i in range(64):
        quote_bytes[user_data_offset + i] = 0
    
    # Use unicode nonce
    unicode_nonce = "test_αβγ_123_🔒"
    nonce_bytes = unicode_nonce.encode('utf-8')
    
    # Only store if it fits in 64 bytes
    if len(nonce_bytes) <= 64:
        for i, byte in enumerate(nonce_bytes):
            quote_bytes[user_data_offset + i] = byte
        
        modified_quote_b64 = base64.b64encode(bytes(quote_bytes)).decode('utf-8')
        quote = parse_tdx_quote(modified_quote_b64)
        
        assert quote.user_data == unicode_nonce


# Quote parsing tests - Invalid cases
def test_parse_tdx_quote_invalid_base64():
    """Test parsing with invalid base64."""
    with pytest.raises(InvalidQuoteError, match="Quote is not valid base64"):
        parse_tdx_quote("invalid_base64!")


def test_parse_tdx_quote_too_small():
    """Test parsing with quote that's too small."""
    small_quote = base64.b64encode(b"small").decode('utf-8')
    
    with pytest.raises(InvalidQuoteError, match="Quote too small"):
        parse_tdx_quote(small_quote)


def test_parse_tdx_quote_one_byte_too_small():
    """Test parsing with one byte less than minimum."""
    quote_bytes = bytearray(631)  # One byte too small
    
    quote_b64 = base64.b64encode(bytes(quote_bytes)).decode('utf-8')
    
    with pytest.raises(InvalidQuoteError, match="Quote too small.*minimum 632"):
        parse_tdx_quote(quote_b64)


def test_parse_tdx_quote_exact_minimum_size():
    """Test parsing with exactly minimum size (632 bytes)."""
    quote_bytes = bytearray(632)  # Exact minimum
    
    # Set valid header
    quote_bytes[0] = 4  # version
    quote_bytes[12] = 0x93  # TEE type
    quote_bytes[13] = 0x9a
    
    quote_b64 = base64.b64encode(bytes(quote_bytes)).decode('utf-8')
    # Should not raise an error
    quote = parse_tdx_quote(quote_b64)
    assert quote.version == 4


def test_parse_tdx_quote_invalid_version(valid_quote_bytes):
    """Test parsing with invalid quote version."""
    quote_bytes = bytearray(valid_quote_bytes)
    
    # Modify version (first 2 bytes, little endian)
    quote_bytes[0] = 5  # Invalid version
    quote_bytes[1] = 0
    
    invalid_quote_b64 = base64.b64encode(bytes(quote_bytes)).decode('utf-8')
    
    with pytest.raises(InvalidQuoteError, match="Invalid quote version"):
        parse_tdx_quote(invalid_quote_b64)


def test_parse_tdx_quote_min_size_header():
    """Test parsing with quote smaller than header size."""
    small_quote_bytes = b"x" * 40  # Less than 48 bytes
    invalid_quote_b64 = base64.b64encode(small_quote_bytes).decode('utf-8')
    
    with pytest.raises(InvalidQuoteError, match="Quote too small"):
        parse_tdx_quote(invalid_quote_b64)


def test_parse_tdx_quote_truncated_td_quote_body(valid_quote_bytes):
    """Test parsing with truncated TD quote body."""
    quote_bytes = valid_quote_bytes[:100]  # Truncate to 100 bytes
    
    invalid_quote_b64 = base64.b64encode(quote_bytes).decode('utf-8')
    
    with pytest.raises(InvalidQuoteError, match="Quote too small"):
        parse_tdx_quote(invalid_quote_b64)


def test_parse_tdx_quote_malformed_struct(valid_quote_bytes):
    """Test parsing with malformed structure that causes struct.error."""
    # Create a quote that's long enough but has invalid header structure
    quote_bytes = b"x" * 700  # Valid length but invalid content
    
    invalid_quote_b64 = base64.b64encode(quote_bytes).decode('utf-8')
    
    with pytest.raises(InvalidQuoteError, match="Failed to parse TDX quote"):
        parse_tdx_quote(invalid_quote_b64)


def test_parse_tdx_quote_different_tee_types(valid_quote_bytes):
    """Test parsing quotes with different TEE types."""
    # Test with 0x81 (standard TDX)
    quote_bytes = bytearray(valid_quote_bytes)
    # Set TEE type to 0x81 (offset 12-13 in header)
    quote_bytes[12] = 0x81
    quote_bytes[13] = 0x00
    
    quote_b64 = base64.b64encode(bytes(quote_bytes)).decode('utf-8')
    quote = parse_tdx_quote(quote_b64)
    assert quote.tee_type == 0x81
    
    # Test with 0x9a93 (your actual quote type)
    quote_bytes[12] = 0x93
    quote_bytes[13] = 0x9a
    
    quote_b64 = base64.b64encode(bytes(quote_bytes)).decode('utf-8')
    quote = parse_tdx_quote(quote_b64)
    assert quote.tee_type == 0x9a93


def test_parse_tdx_quote_unexpected_tee_type_warning(valid_quote_bytes):
    """Test that unexpected TEE types generate warnings but don't fail."""
    quote_bytes = bytearray(valid_quote_bytes)
    
    # Set an unexpected TEE type
    quote_bytes[12] = 0xFF  
    quote_bytes[13] = 0xFF
    
    quote_b64 = base64.b64encode(bytes(quote_bytes)).decode('utf-8')
    
    # Should parse successfully despite unexpected TEE type
    quote = parse_tdx_quote(quote_b64)
    assert quote.tee_type == 0xFFFF
    assert quote.version == 4


# Test structure offset validation
def test_parse_tdx_quote_structure_offsets():
    """Test that we're reading from the correct offsets."""
    # Create a minimal valid quote structure
    quote_bytes = bytearray(4096)  # 4KB like your actual quote
    
    # Header (48 bytes)
    # Version at offset 0-1
    quote_bytes[0] = 4
    quote_bytes[1] = 0
    # TEE type at offset 12-13  
    quote_bytes[12] = 0x93
    quote_bytes[13] = 0x9a
    
    # TD Quote Body starts at offset 48
    # MRTD at offset 48 + 136 = 184
    mrtd_offset = 48 + 136
    test_mrtd = secrets.token_bytes(48)
    quote_bytes[mrtd_offset:mrtd_offset + 48] = test_mrtd
    
    # RTMRs at offset 48 + 328 = 376  
    rtmr_offset = 48 + 328
    test_rtmrs = secrets.token_bytes(192)  # 4 x 48 bytes
    quote_bytes[rtmr_offset:rtmr_offset + 192] = test_rtmrs
    
    # User data at offset 48 + 520 = 568
    user_data_offset = 48 + 520
    test_nonce = "structure_test_nonce"
    nonce_bytes = test_nonce.encode('utf-8')
    quote_bytes[user_data_offset:user_data_offset + len(nonce_bytes)] = nonce_bytes
    
    # Parse and verify
    quote_b64 = base64.b64encode(bytes(quote_bytes)).decode('utf-8')
    quote = parse_tdx_quote(quote_b64)
    
    assert quote.version == 4
    assert quote.tee_type == 0x9a93
    assert quote.mrtd == test_mrtd.hex()
    assert quote.rtmr0 == test_rtmrs[0:48].hex()
    assert quote.rtmr1 == test_rtmrs[48:96].hex()
    assert quote.rtmr2 == test_rtmrs[96:144].hex()
    assert quote.rtmr3 == test_rtmrs[144:192].hex()
    assert quote.user_data == test_nonce


def test_parse_tdx_quote_measurement_accuracy(valid_quote_bytes):
    """Test that measurements are extracted from correct locations."""
    quote_bytes = bytearray(valid_quote_bytes)
    
    # Create known test data for each measurement
    known_mrtd = secrets.token_bytes(48)
    known_rtmr0 = secrets.token_bytes(48)
    known_rtmr1 = secrets.token_bytes(48)
    known_rtmr2 = secrets.token_bytes(48)
    known_rtmr3 = secrets.token_bytes(48)
    
    # Write to specific offsets
    # MRTD at 48 + 136 = 184
    quote_bytes[184:184+48] = known_mrtd
    
    # RTMRs at 48 + 328 = 376
    quote_bytes[376:376+48] = known_rtmr0      # RTMR0
    quote_bytes[424:424+48] = known_rtmr1      # RTMR1  
    quote_bytes[472:472+48] = known_rtmr2      # RTMR2
    quote_bytes[520:520+48] = known_rtmr3      # RTMR3
    
    modified_quote_b64 = base64.b64encode(bytes(quote_bytes)).decode('utf-8')
    quote = parse_tdx_quote(modified_quote_b64)
    
    # Verify each measurement matches what we wrote
    assert quote.mrtd == known_mrtd.hex()
    assert quote.rtmr0 == known_rtmr0.hex()
    assert quote.rtmr1 == known_rtmr1.hex()
    assert quote.rtmr2 == known_rtmr2.hex()
    assert quote.rtmr3 == known_rtmr3.hex()


# Quote signature verification tests
def test_verify_quote_signature_placeholder():
    """Test quote signature verification (placeholder implementation)."""
    # Since this is a placeholder, it should always return True
    quote_bytes = b"dummy_quote_data"
    
    result = verify_quote_signature(quote_bytes)
    assert result is True


# Boot measurement verification tests
@patch('api.server.util.settings')
def test_verify_boot_measurements_success(mock_settings, valid_quote_bin):
    """Test successful boot measurement verification."""
    quote = parse_tdx_quote(valid_quote_bin)
    
    # Set expected MRTD to match the quote
    mock_settings.boot_expected_mrtd = quote.mrtd
    
    result = verify_boot_measurements(quote)
    assert result is True


@patch('api.server.util.settings')
def test_verify_boot_measurements_no_expected(mock_settings, valid_quote_bin):
    """Test boot measurement verification with no expected MRTD configured."""
    quote = parse_tdx_quote(valid_quote_bin)
    
    # No expected MRTD configured
    mock_settings.boot_expected_mrtd = None
    
    result = verify_boot_measurements(quote)
    assert result is True  # Should pass when not configured


@patch('api.server.util.settings')
def test_verify_boot_measurements_mismatch(mock_settings, valid_quote_bin):
    """Test boot measurement verification failure."""
    quote = parse_tdx_quote(valid_quote_bin)
    
    # Set different expected MRTD
    mock_settings.boot_expected_mrtd = "different" + "0" * 88
    
    with pytest.raises(MeasurementMismatchError, match="MRTD verification failed"):
        verify_boot_measurements(quote)


# Runtime measurement verification tests
def test_verify_runtime_measurements_success(valid_quote_bin):
    """Test successful runtime measurement verification."""
    quote = parse_tdx_quote(valid_quote_bin)
    
    expected_measurements = {
        "mrtd": quote.mrtd,
        "rtmrs": {
            "rtmr0": quote.rtmr0,
            "rtmr1": quote.rtmr1,
        }
    }
    
    result = verify_runtime_measurements(quote, expected_measurements)
    assert result is True


def test_verify_runtime_measurements_mrtd_mismatch(valid_quote_bin):
    """Test runtime measurement verification with MRTD mismatch."""
    quote = parse_tdx_quote(valid_quote_bin)
    
    expected_measurements = {
        "mrtd": "different" + "0" * 88,
    }
    
    with pytest.raises(MeasurementMismatchError, match="Runtime MRTD verification failed"):
        verify_runtime_measurements(quote, expected_measurements)


def test_verify_runtime_measurements_rtmr_mismatch(valid_quote_bin):
    """Test runtime measurement verification with RTMR mismatch."""
    quote = parse_tdx_quote(valid_quote_bin)
    
    expected_measurements = {
        "rtmrs": {
            "rtmr0": "different" + "0" * 88,
        }
    }
    
    with pytest.raises(MeasurementMismatchError, match="RTMR rtmr0 verification failed"):
        verify_runtime_measurements(quote, expected_measurements)


def test_verify_runtime_measurements_partial_rtmrs(valid_quote_bin):
    """Test runtime measurement verification with partial RTMR checking."""
    quote = parse_tdx_quote(valid_quote_bin)
    
    # Only check RTMR1, ignore others
    expected_measurements = {
        "rtmrs": {
            "rtmr1": quote.rtmr1,
        }
    }
    
    result = verify_runtime_measurements(quote, expected_measurements)
    assert result is True


def test_verify_runtime_measurements_empty():
    """Test runtime measurement verification with empty expected measurements."""
    quote = TdxQuote(
        version=4, att_key_type=2, tee_type=0x9a93,
        mrtd="a" * 96, rtmr0="b" * 96, rtmr1="c" * 96, 
        rtmr2="d" * 96, rtmr3="e" * 96,
        user_data=None, raw_quote_size=4096,
        parsed_at="2025-01-01T00:00:00Z"
    )
    
    result = verify_runtime_measurements(quote, {})
    assert result is True


# Nonce validation tests
def test_validate_nonce_in_quote_success(test_nonce):
    """Test successful nonce validation in quote."""
    quote = TdxQuote(
        version=4, att_key_type=2, tee_type=0x9a93,
        mrtd="a" * 96, rtmr0="b" * 96, rtmr1="c" * 96, 
        rtmr2="d" * 96, rtmr3="e" * 96,
        user_data=test_nonce, raw_quote_size=4096,
        parsed_at="2025-01-01T00:00:00Z"
    )
    
    result = validate_nonce_in_quote(quote, test_nonce)
    assert result is True


def test_validate_nonce_in_quote_no_user_data():
    """Test nonce validation with no user data in quote."""
    quote = TdxQuote(
        version=4, att_key_type=2, tee_type=0x9a93,
        mrtd="a" * 96, rtmr0="b" * 96, rtmr1="c" * 96, 
        rtmr2="d" * 96, rtmr3="e" * 96,
        user_data=None, raw_quote_size=4096,
        parsed_at="2025-01-01T00:00:00Z"
    )
    
    result = validate_nonce_in_quote(quote, "any_nonce")
    assert result is False


def test_validate_nonce_in_quote_mismatch():
    """Test nonce validation with mismatched nonce."""
    quote = TdxQuote(
        version=4, att_key_type=2, tee_type=0x9a93,
        mrtd="a" * 96, rtmr0="b" * 96, rtmr1="c" * 96, 
        rtmr2="d" * 96, rtmr3="e" * 96,
        user_data="wrong_nonce", raw_quote_size=4096,
        parsed_at="2025-01-01T00:00:00Z"
    )
    
    result = validate_nonce_in_quote(quote, "expected_nonce")
    assert result is False


def test_validate_nonce_in_quote_with_whitespace():
    """Test nonce validation with whitespace in user data."""
    quote = TdxQuote(
        version=4, att_key_type=2, tee_type=0x9a93,
        mrtd="a" * 96, rtmr0="b" * 96, rtmr1="c" * 96, 
        rtmr2="d" * 96, rtmr3="e" * 96,
        user_data="  test_nonce  ", raw_quote_size=4096,
        parsed_at="2025-01-01T00:00:00Z"
    )
    
    result = validate_nonce_in_quote(quote, "test_nonce")
    assert result is True


# Nonce extraction tests
def test_extract_nonce_from_quote_valid():
    """Test extracting valid nonce from quote."""
    test_nonce = "extracted_nonce_test"
    quote = TdxQuote(
        version=4, att_key_type=2, tee_type=0x9a93,
        mrtd="a" * 96, rtmr0="b" * 96, rtmr1="c" * 96, 
        rtmr2="d" * 96, rtmr3="e" * 96,
        user_data=test_nonce, raw_quote_size=4096,
        parsed_at="2025-01-01T00:00:00Z"
    )
    
    extracted = extract_nonce_from_quote(quote)
    assert extracted == test_nonce


def test_extract_nonce_from_quote_no_user_data():
    """Test extracting nonce when no user data."""
    quote = TdxQuote(
        version=4, att_key_type=2, tee_type=0x9a93,
        mrtd="a" * 96, rtmr0="b" * 96, rtmr1="c" * 96, 
        rtmr2="d" * 96, rtmr3="e" * 96,
        user_data=None, raw_quote_size=4096,
        parsed_at="2025-01-01T00:00:00Z"
    )
    
    extracted = extract_nonce_from_quote(quote)
    assert extracted is None


def test_extract_nonce_from_quote_hex_format():
    """Test extracting nonce that looks like hex."""
    hex_nonce = secrets.token_hex(32)  # 64 hex characters
    quote = TdxQuote(
        version=4, att_key_type=2, tee_type=0x9a93,
        mrtd="a" * 96, rtmr0="b" * 96, rtmr1="c" * 96, 
        rtmr2="d" * 96, rtmr3="e" * 96,
        user_data=hex_nonce, raw_quote_size=4096,
        parsed_at="2025-01-01T00:00:00Z"
    )
    
    extracted = extract_nonce_from_quote(quote)
    assert extracted == hex_nonce


def test_extract_nonce_from_quote_with_whitespace():
    """Test extracting nonce with whitespace."""
    test_nonce = "test_nonce_with_spaces"
    quote = TdxQuote(
        version=4, att_key_type=2, tee_type=0x9a93,
        mrtd="a" * 96, rtmr0="b" * 96, rtmr1="c" * 96, 
        rtmr2="d" * 96, rtmr3="e" * 96,
        user_data=f"  {test_nonce}  ", raw_quote_size=4096,
        parsed_at="2025-01-01T00:00:00Z"
    )
    
    extracted = extract_nonce_from_quote(quote)
    assert extracted == test_nonce


# LUKS passphrase tests
@patch('api.server.util.settings')
def test_get_luks_passphrase_configured(mock_settings):
    """Test getting LUKS passphrase when configured."""
    mock_settings.luks_passphrase = "configured_passphrase"
    
    passphrase = get_luks_passphrase()
    assert passphrase == "configured_passphrase"


@patch('api.server.util.settings')
def test_get_luks_passphrase_not_configured(mock_settings):
    """Test getting LUKS passphrase when not configured."""
    mock_settings.luks_passphrase = None
    
    passphrase = get_luks_passphrase()
    assert passphrase == "placeholder_luks_passphrase"