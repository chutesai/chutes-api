"""
Unit tests for api/server/util module.
Tests TDX quote parsing, validation, and utility functions.
"""

import base64
import pytest
import secrets
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch, Mock

from api.server.util import (
    generate_nonce,
    get_nonce_expiry_seconds,
    verify_quote_signature,
    verify_measurements,
    extract_nonce,
    get_luks_passphrase,
)
from api.server.quote import (
    TdxQuote,
    BootTdxQuote,
    RuntimeTdxQuote,
    TdxVerificationResult,
)
from api.server.exceptions import (
    InvalidQuoteError,
    MeasurementMismatchError,
    InvalidSignatureError,
)


# Test fixtures
@pytest.fixture
def test_nonce():
    """Generate a test nonce."""
    return "test_nonce_123"


@pytest.fixture
def mock_settings():
    """Mock settings for testing."""
    settings = Mock()
    settings.expected_mrtd = "a" * 96  # 48 bytes = 96 hex chars
    settings.expected_boot_rmtrs = {
        "rtmr0": "b" * 96,
        "rtmr1": "c" * 96,
    }
    settings.exepected_runtime_rmtrs = {
        "rtmr0": "d" * 96,
        "rtmr1": "e" * 96,
    }
    settings.luks_passphrase = "test_luks_passphrase"
    return settings


@pytest.fixture
def sample_boot_quote():
    """Create a sample BootTdxQuote for testing."""

    return BootTdxQuote(
        version=4,
        att_key_type=2,
        tee_type=0x81,
        mrtd="a" * 96,
        rtmr0="b" * 96,
        rtmr1="c" * 96,
        rtmr2="d" * 96,
        rtmr3="e" * 96,
        user_data="746573745f6e6f6e63655f31323300000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000",  # test_nonce_123 as hex
        raw_quote_size=4096,
        parsed_at=datetime.now(timezone.utc).isoformat(),
        raw_bytes=b"dummy_quote_bytes",
    )


@pytest.fixture
def sample_runtime_quote():
    """Create a sample RuntimeTdxQuote for testing."""

    return RuntimeTdxQuote(
        version=4,
        att_key_type=2,
        tee_type=0x81,
        mrtd="a" * 96,
        rtmr0="d" * 96,  # Different from boot
        rtmr1="e" * 96,  # Different from boot
        rtmr2="f" * 96,
        rtmr3="0" * 96,
        user_data="72756e74696d655f6e6f6e63655f34353600000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000",  # runtime_nonce_456 as hex
        raw_quote_size=4096,
        parsed_at=datetime.now(timezone.utc).isoformat(),
        raw_bytes=b"dummy_runtime_quote_bytes",
    )


# TdxQuote class tests
def test_tdx_quote_abstract_quote_type():
    """Test that TdxQuote has abstract quote_type property."""

    # Can't instantiate TdxQuote directly due to abstract method
    with pytest.raises(TypeError):
        TdxQuote(
            version=4,
            att_key_type=2,
            tee_type=0x81,
            mrtd="a" * 96,
            rtmr0="b" * 96,
            rtmr1="c" * 96,
            rtmr2="d" * 96,
            rtmr3="e" * 96,
            user_data=None,
            raw_quote_size=4096,
            parsed_at=datetime.now(timezone.utc).isoformat(),
            raw_bytes=b"test",
        )


def test_boot_tdx_quote_type(sample_boot_quote):
    """Test BootTdxQuote quote_type property."""
    assert sample_boot_quote.quote_type == "boot"


def test_runtime_tdx_quote_type(sample_runtime_quote):
    """Test RuntimeTdxQuote quote_type property."""
    assert sample_runtime_quote.quote_type == "runtime"


def test_tdx_quote_rtmrs_property(sample_boot_quote):
    """Test that RTMRs property returns correct dictionary."""
    rtmrs = sample_boot_quote.rtmrs
    assert len(rtmrs) == 4
    assert rtmrs["rtmr0"] == "b" * 96
    assert rtmrs["rtmr1"] == "c" * 96
    assert rtmrs["rtmr2"] == "d" * 96
    assert rtmrs["rtmr3"] == "e" * 96


def test_tdx_quote_to_dict(sample_boot_quote):
    """Test TdxQuote.to_dict() method."""
    result = sample_boot_quote.to_dict()

    assert result["quote_version"] == "4"
    assert result["mrtd"] == "a" * 96
    assert result["raw_quote_size"] == 4096
    assert result["header"]["version"] == 4
    assert result["header"]["tee_type"] == "0x81"
    assert "rtmrs" in result
    assert result["rtmrs"]["rtmr0"] == "b" * 96


# TdxVerificationResult tests
def test_tdx_verification_result_creation():
    """Test TdxVerificationResult creation and properties."""
    result = TdxVerificationResult(
        mrtd="a" * 96,
        rtmr0="b" * 96,
        rtmr1="c" * 96,
        rtmr2="d" * 96,
        rtmr3="e" * 96,
        user_data="test_data",
        parsed_at=datetime.now(timezone.utc),
        is_valid=True,
    )

    assert result.mrtd == "a" * 96
    assert result.is_valid is True
    assert result.user_data == "test_data"
    assert isinstance(result.parsed_at, datetime)


def test_tdx_verification_result_rtmrs_property():
    """Test TdxVerificationResult rtmrs property."""
    result = TdxVerificationResult(
        mrtd="a" * 96,
        rtmr0="0" * 96,
        rtmr1="1" * 96,
        rtmr2="2" * 96,
        rtmr3="3" * 96,
        user_data=None,
        parsed_at=datetime.now(timezone.utc),
        is_valid=True,
    )

    rtmrs = result.rtmrs
    assert len(rtmrs) == 4
    assert rtmrs["rtmr0"] == "0" * 96
    assert rtmrs["rtmr1"] == "1" * 96
    assert rtmrs["rtmr2"] == "2" * 96
    assert rtmrs["rtmr3"] == "3" * 96


def test_tdx_verification_result_to_dict():
    """Test TdxVerificationResult.to_dict() method."""
    now = datetime.now(timezone.utc)
    result = TdxVerificationResult(
        mrtd="a" * 96,
        rtmr0="b" * 96,
        rtmr1="c" * 96,
        rtmr2="d" * 96,
        rtmr3="e" * 96,
        user_data="test",
        parsed_at=now,
        is_valid=True,
    )

    dict_result = result.to_dict()

    assert dict_result["mrtd"] == "a" * 96
    assert dict_result["is_valid"] is True
    assert dict_result["user_data"] == "test"
    assert dict_result["parsed_at"] == now
    assert "rtmrs" in dict_result
    assert dict_result["rtmrs"]["rtmr0"] == "b" * 96


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


@pytest.mark.parametrize(
    "minutes,expected_seconds",
    [
        (1, 60),
        (5, 300),
        (10, 600),
        (30, 1800),
        (60, 3600),
    ],
)
def test_get_nonce_expiry_seconds_parametrized(minutes, expected_seconds):
    """Test nonce expiry calculation with various inputs."""
    assert get_nonce_expiry_seconds(minutes) == expected_seconds


# Quote parsing tests - Valid cases
def test_boot_quote_from_base64(valid_quote_base64):
    """Test parsing a valid TDX boot quote from base64."""
    quote = BootTdxQuote.from_base64(valid_quote_base64)

    assert isinstance(quote, BootTdxQuote)
    assert quote.quote_type == "boot"
    assert quote.version == 4
    assert quote.tee_type in [0x81, 0x9A93]  # Allow both TEE types
    assert len(quote.mrtd) == 96  # 48 bytes as hex
    assert len(quote.rtmr0) == 96
    assert quote.raw_quote_size > 0
    assert quote.parsed_at is not None


def test_runtime_quote_from_base64(valid_quote_base64):
    """Test parsing a valid TDX runtime quote from base64."""
    quote = RuntimeTdxQuote.from_base64(valid_quote_base64)

    assert isinstance(quote, RuntimeTdxQuote)
    assert quote.quote_type == "runtime"
    assert quote.version == 4
    assert len(quote.mrtd) == 96
    assert len(quote.rtmr0) == 96


def test_quote_from_bytes(valid_quote_bytes):
    """Test parsing quote from bytes."""
    boot_quote = BootTdxQuote.from_bytes(valid_quote_bytes)
    runtime_quote = RuntimeTdxQuote.from_bytes(valid_quote_bytes)

    assert boot_quote.quote_type == "boot"
    assert runtime_quote.quote_type == "runtime"
    assert boot_quote.mrtd == runtime_quote.mrtd  # Same underlying data


def test_parse_quote_with_user_data(valid_quote_bytes, test_nonce):
    """Test parsing quote with embedded nonce in user data."""
    # Modify the valid quote to include our test nonce in user data
    quote_bytes = bytearray(valid_quote_bytes)

    # User data is at offset 48 (header) + 520 (TD quote body offset) = 568
    user_data_offset = 48 + 520

    # Ensure we have enough space
    if len(quote_bytes) < user_data_offset + 64:
        quote_bytes.extend(b"\x00" * (user_data_offset + 64 - len(quote_bytes)))

    # Clear user data field and insert our nonce
    for i in range(64):
        quote_bytes[user_data_offset + i] = 0

    # Store the nonce as UTF-8 bytes
    nonce_bytes = test_nonce.encode("utf-8")
    for i, byte in enumerate(nonce_bytes[:64]):
        quote_bytes[user_data_offset + i] = byte

    # Parse both types
    boot_quote = BootTdxQuote.from_bytes(bytes(quote_bytes))
    runtime_quote = RuntimeTdxQuote.from_bytes(bytes(quote_bytes))

    # Both should extract the same nonce
    assert extract_nonce(boot_quote) == test_nonce
    assert extract_nonce(runtime_quote) == test_nonce


# Quote parsing tests - Invalid cases
def test_parse_invalid_base64():
    """Test parsing with invalid base64."""
    with pytest.raises(InvalidQuoteError):
        BootTdxQuote.from_base64("invalid_base64!")


def test_parse_quote_too_small():
    """Test parsing with quote that's too small."""
    small_quote = base64.b64encode(b"small").decode("utf-8")

    with pytest.raises(InvalidQuoteError, match="Quote too short"):
        BootTdxQuote.from_base64(small_quote)


def test_parse_invalid_version(valid_quote_bytes):
    """Test parsing with invalid quote version."""
    quote_bytes = bytearray(valid_quote_bytes)

    # Modify version (first 2 bytes, little endian)
    quote_bytes[0] = 5  # Invalid version
    quote_bytes[1] = 0

    with pytest.raises(InvalidQuoteError, match="Invalid quote version"):
        BootTdxQuote.from_bytes(bytes(quote_bytes))


def test_parse_invalid_tee_type(valid_quote_bytes):
    """Test parsing with invalid TEE type."""
    quote_bytes = bytearray(valid_quote_bytes)

    # Set invalid TEE type (offset 8-11 in header, uint32 little endian)
    quote_bytes[4] = 0x99  # Invalid TEE type
    quote_bytes[5] = 0x99
    quote_bytes[6] = 0x99
    quote_bytes[7] = 0x99

    with pytest.raises(InvalidQuoteError, match="Invalid TEE type"):
        BootTdxQuote.from_bytes(bytes(quote_bytes))


def test_parse_invalid_att_key_type(valid_quote_bytes):
    """Test parsing with invalid attestation key type."""
    quote_bytes = bytearray(valid_quote_bytes)

    # Set invalid att_key_type (offset 2-3 in header, uint16 little endian)
    quote_bytes[2] = 99  # Invalid att_key_type
    quote_bytes[3] = 0

    with pytest.raises(InvalidQuoteError, match="Invalid attestation key type"):
        BootTdxQuote.from_bytes(bytes(quote_bytes))


# Extract nonce tests
def test_extract_nonce_valid(sample_boot_quote):
    """Test extracting valid nonce from quote."""
    # The sample quote has test_nonce_123 encoded in user_data
    extracted = extract_nonce(sample_boot_quote)
    assert extracted == "test_nonce_123"


def test_extract_nonce_runtime_quote(sample_runtime_quote):
    """Test extracting nonce from runtime quote."""
    extracted = extract_nonce(sample_runtime_quote)
    assert extracted == "runtime_nonce_456"


def test_extract_nonce_empty_user_data():
    """Test extracting nonce when user data is empty."""

    quote = BootTdxQuote(
        version=4,
        att_key_type=2,
        tee_type=0x81,
        mrtd="a" * 96,
        rtmr0="b" * 96,
        rtmr1="c" * 96,
        rtmr2="d" * 96,
        rtmr3="e" * 96,
        user_data="00000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000",  # All zeros
        raw_quote_size=4096,
        parsed_at=datetime.now(timezone.utc).isoformat(),
        raw_bytes=b"test",
    )

    extracted = extract_nonce(quote)
    assert extracted == ""


# Quote signature verification tests
@pytest.mark.asyncio
async def test_verify_quote_signature_success(sample_boot_quote):
    """Test successful quote signature verification."""
    mock_verified_report = Mock()
    mock_verified_report.status = "UpToDate"
    mock_verified_report.to_json.return_value = '{"report": {"TD10": {"mr_td": "a", "rt_mr0": "b", "rt_mr1": "c", "rt_mr2": "d", "rt_mr3": "e", "report_data": "test"}}}'

    with patch(
        "api.server.util.get_collateral_and_verify", return_value=mock_verified_report
    ) as mock_verify:
        result = await verify_quote_signature(sample_boot_quote)

        assert isinstance(result, TdxVerificationResult)
        assert result.is_valid is True
        mock_verify.assert_called_once_with(sample_boot_quote.raw_bytes)


@pytest.mark.asyncio
async def test_verify_quote_signature_failure(sample_boot_quote):
    """Test failed quote signature verification."""
    mock_verified_report = Mock()
    mock_verified_report.status = "Invalid"
    mock_verified_report.to_json.return_value = '{"report": {"TD10": {"mr_td": "a", "rt_mr0": "b", "rt_mr1": "c", "rt_mr2": "d", "rt_mr3": "e", "report_data": "test"}}}'

    with patch("api.server.util.get_collateral_and_verify", return_value=mock_verified_report):
        with pytest.raises(InvalidSignatureError):
            await verify_quote_signature(sample_boot_quote)


# Measurement verification tests
@patch("api.server.util.settings")
def test_verify_measurements_boot_success(mock_settings, sample_boot_quote):
    """Test successful boot measurement verification."""
    mock_settings.expected_mrtd = sample_boot_quote.mrtd
    mock_settings.expected_boot_rmtrs = {
        "rtmr0": sample_boot_quote.rtmr0,
        "rtmr1": sample_boot_quote.rtmr1,
    }

    result = verify_measurements(sample_boot_quote)
    assert result is True


@patch("api.server.util.settings")
def test_verify_measurements_runtime_success(mock_settings, sample_runtime_quote):
    """Test successful runtime measurement verification."""
    mock_settings.expected_mrtd = sample_runtime_quote.mrtd
    mock_settings.exepected_runtime_rmtrs = {
        "rtmr0": sample_runtime_quote.rtmr0,
        "rtmr1": sample_runtime_quote.rtmr1,
    }

    result = verify_measurements(sample_runtime_quote)
    assert result is True


@patch("api.server.util.settings")
def test_verify_measurements_mrtd_mismatch(mock_settings, sample_boot_quote):
    """Test measurement verification with MRTD mismatch."""
    mock_settings.expected_mrtd = "different" + "0" * 88
    mock_settings.expected_boot_rmtrs = {}

    with pytest.raises(MeasurementMismatchError, match="MRTD verification failed"):
        verify_measurements(sample_boot_quote)


@patch("api.server.util.settings")
def test_verify_measurements_rtmr_mismatch(mock_settings, sample_boot_quote):
    """Test measurement verification with RTMR mismatch."""
    mock_settings.expected_mrtd = sample_boot_quote.mrtd
    mock_settings.expected_boot_rmtrs = {
        "rtmr0": "different" + "0" * 88,
    }

    with pytest.raises(MeasurementMismatchError, match="RTMR rtmr0 verification failed"):
        verify_measurements(sample_boot_quote)


@patch("api.server.util.settings")
def test_verify_measurements_no_expected_mrtd(mock_settings, sample_boot_quote):
    """Test measurement verification with no expected MRTD configured."""
    mock_settings.expected_mrtd = None
    mock_settings.expected_boot_rmtrs = {}

    result = verify_measurements(sample_boot_quote)
    assert result is True


@patch("api.server.util.settings")
def test_verify_measurements_partial_rtmrs(mock_settings, sample_runtime_quote):
    """Test measurement verification with partial RTMR checking."""
    mock_settings.expected_mrtd = sample_runtime_quote.mrtd
    mock_settings.exepected_runtime_rmtrs = {
        "rtmr1": sample_runtime_quote.rtmr1,  # Only check RTMR1
    }

    result = verify_measurements(sample_runtime_quote)
    assert result is True


# LUKS passphrase tests
@patch("api.server.util.settings")
def test_get_luks_passphrase_configured(mock_settings):
    """Test getting LUKS passphrase when configured."""
    mock_settings.luks_passphrase = "configured_passphrase"

    passphrase = get_luks_passphrase()
    assert passphrase == "configured_passphrase"


@patch("api.server.util.settings")
def test_get_luks_passphrase_not_configured(mock_settings):
    """Test getting LUKS passphrase when not configured."""
    mock_settings.luks_passphrase = None

    passphrase = get_luks_passphrase()
    assert passphrase == "placeholder_luks_passphrase"


# Test different quote types with different RTMRs
def test_boot_vs_runtime_quote_differences():
    """Test that boot and runtime quotes can have different RTMRs."""

    boot_quote = BootTdxQuote(
        version=4,
        att_key_type=2,
        tee_type=0x81,
        mrtd="a" * 96,
        rtmr0="boot_rtmr0",
        rtmr1="boot_rtmr1",
        rtmr2="d" * 96,
        rtmr3="e" * 96,
        user_data=None,
        raw_quote_size=4096,
        parsed_at=datetime.now(timezone.utc).isoformat(),
        raw_bytes=b"boot",
    )

    runtime_quote = RuntimeTdxQuote(
        version=4,
        att_key_type=2,
        tee_type=0x81,
        mrtd="a" * 96,
        rtmr0="runtime_rtmr0",
        rtmr1="runtime_rtmr1",
        rtmr2="d" * 96,
        rtmr3="e" * 96,
        user_data=None,
        raw_quote_size=4096,
        parsed_at=datetime.now(timezone.utc).isoformat(),
        raw_bytes=b"runtime",
    )

    assert boot_quote.quote_type == "boot"
    assert runtime_quote.quote_type == "runtime"
    assert boot_quote.rtmr0 == "boot_rtmr0"
    assert runtime_quote.rtmr0 == "runtime_rtmr0"
    assert boot_quote.rtmr1 == "boot_rtmr1"
    assert runtime_quote.rtmr1 == "runtime_rtmr1"


# Edge cases and error handling
def test_quote_with_mixed_case_hex():
    """Test that hex values are handled consistently regardless of case."""

    quote = BootTdxQuote(
        version=4,
        att_key_type=2,
        tee_type=0x81,
        mrtd="AbCdEf" * 16,
        rtmr0="123456" * 16,
        rtmr1="fedcba" * 16,
        rtmr2="d" * 96,
        rtmr3="e" * 96,
        user_data=None,
        raw_quote_size=4096,
        parsed_at=datetime.now(timezone.utc).isoformat(),
        raw_bytes=b"test",
    )

    # Should handle mixed case properly
    assert quote.mrtd == "AbCdEf" * 16
    assert quote.rtmr0 == "123456" * 16


def test_quote_parsing_with_different_att_key_types(valid_quote_bytes):
    """Test parsing quotes with different attestation key types."""
    quote_bytes = bytearray(valid_quote_bytes)

    # Test ECDSA-256 (type 2)
    quote_bytes[2] = 2
    quote_bytes[3] = 0
    quote = BootTdxQuote.from_bytes(bytes(quote_bytes))
    assert quote.att_key_type == 2

    # Test ECDSA-384 (type 3)
    quote_bytes[2] = 3
    quote_bytes[3] = 0
    quote = RuntimeTdxQuote.from_bytes(bytes(quote_bytes))
    assert quote.att_key_type == 3


def test_extract_nonce_with_binary_data():
    """Test extracting nonce from quote with binary user data."""
    # Create user_data with binary content that should fallback to hex
    binary_data = bytes([0x01, 0x02, 0x03, 0xFF, 0xFE, 0xFD]) + b"\x00" * 58
    user_data_hex = binary_data.hex().upper()

    quote = BootTdxQuote(
        version=4,
        att_key_type=2,
        tee_type=0x81,
        mrtd="a" * 96,
        rtmr0="b" * 96,
        rtmr1="c" * 96,
        rtmr2="d" * 96,
        rtmr3="e" * 96,
        user_data=user_data_hex,
        raw_quote_size=4096,
        parsed_at=datetime.now(timezone.utc).isoformat(),
        raw_bytes=b"test",
    )

    extracted = extract_nonce(quote)
    # Should extract the printable portion or handle binary gracefully
    assert extracted is not None


def test_verification_result_from_report():
    """Test creating TdxVerificationResult from dcap report."""
    from unittest.mock import Mock

    mock_report = Mock()
    mock_report.status = "UpToDate"
    mock_report.to_json.return_value = """
    {
        "report": {
            "TD10": {
                "mr_td": "test_mrtd",
                "rt_mr0": "test_rtmr0",
                "rt_mr1": "test_rtmr1",
                "rt_mr2": "test_rtmr2",
                "rt_mr3": "test_rtmr3",
                "report_data": "test_user_data"
            }
        }
    }
    """

    result = TdxVerificationResult.from_report(mock_report)

    assert result.mrtd == "test_mrtd"
    assert result.rtmr0 == "test_rtmr0"
    assert result.rtmr1 == "test_rtmr1"
    assert result.rtmr2 == "test_rtmr2"
    assert result.rtmr3 == "test_rtmr3"
    assert result.user_data == "test_user_data"
    assert result.is_valid is True
    assert isinstance(result.parsed_at, datetime)


def test_verification_result_from_report_invalid():
    """Test creating TdxVerificationResult from invalid dcap report."""
    from unittest.mock import Mock

    mock_report = Mock()
    mock_report.status = "Invalid"
    mock_report.to_json.return_value = """
    {
        "report": {
            "TD10": {
                "mr_td": "test_mrtd",
                "rt_mr0": "test_rtmr0",
                "rt_mr1": "test_rtmr1",
                "rt_mr2": "test_rtmr2",
                "rt_mr3": "test_rtmr3",
                "report_data": "test_user_data"
            }
        }
    }
    """

    result = TdxVerificationResult.from_report(mock_report)

    assert result.is_valid is False


# Performance and robustness tests
def test_large_quote_parsing():
    """Test parsing a large quote doesn't cause issues."""
    # Create a large quote (e.g., 8KB)
    large_quote_bytes = bytearray(8192)

    # Set valid header
    large_quote_bytes[0] = 4  # version
    large_quote_bytes[1] = 0
    large_quote_bytes[2] = 2  # att_key_type
    large_quote_bytes[3] = 0
    large_quote_bytes[4] = 0x81  # tee_type
    large_quote_bytes[5] = 0
    large_quote_bytes[6] = 0
    large_quote_bytes[7] = 0

    # Add some MRTD and RTMR data
    mrtd_offset = 48 + 136
    large_quote_bytes[mrtd_offset : mrtd_offset + 48] = secrets.token_bytes(48)

    rtmr_offset = 48 + 328
    large_quote_bytes[rtmr_offset : rtmr_offset + 192] = secrets.token_bytes(192)

    # Should parse without issues
    large_quote_b64 = base64.b64encode(bytes(large_quote_bytes)).decode("utf-8")
    quote = BootTdxQuote.from_base64(large_quote_b64)

    assert quote.raw_quote_size == 8192
    assert quote.version == 4


def test_multiple_quote_parsing():
    """Test parsing multiple quotes in sequence."""
    if not Path("tests/assets/quote.bin").exists():
        pytest.skip("Quote file not available")

    with open("tests/assets/quote.bin", "rb") as f:
        quote_bytes = f.read()

    quote_b64 = base64.b64encode(quote_bytes).decode("utf-8")

    # Parse multiple times to ensure no state contamination
    quotes = []
    for i in range(5):
        boot_quote = BootTdxQuote.from_base64(quote_b64)
        runtime_quote = RuntimeTdxQuote.from_base64(quote_b64)
        quotes.extend([boot_quote, runtime_quote])

    # All quotes should have same MRTD but different types
    for i in range(0, len(quotes), 2):
        boot_quote = quotes[i]
        runtime_quote = quotes[i + 1]

        assert boot_quote.quote_type == "boot"
        assert runtime_quote.quote_type == "runtime"
        assert boot_quote.mrtd == runtime_quote.mrtd  # Same underlying data


# Integration with settings tests
@patch("api.server.util.settings")
def test_measurement_verification_with_actual_config(mock_settings):
    """Test measurement verification with realistic configuration."""
    from tests.fixtures.tdx import (
        EXPECTED_MRTD,
        EXPECTED_RMTR0,
        EXPECTED_RMTR1,
        EXPECTED_RMTR2,
        EXPECTED_RMTR3,
    )

    # Configure settings with test fixture values
    mock_settings.expected_mrtd = EXPECTED_MRTD
    mock_settings.expected_boot_rmtrs = {
        "rtmr0": EXPECTED_RMTR0,
        "rtmr1": EXPECTED_RMTR1,
        "rtmr2": EXPECTED_RMTR2,
        "rtmr3": EXPECTED_RMTR3,
    }
    mock_settings.exepected_runtime_rmtrs = {
        "rtmr0": EXPECTED_RMTR0,
        "rtmr1": EXPECTED_RMTR1,
        # Runtime might have different RTMR2/3
        "rtmr2": "0" * 96,  # Different from boot
        "rtmr3": "0" * 96,
    }

    boot_quote = BootTdxQuote(
        version=4,
        att_key_type=2,
        tee_type=0x81,
        mrtd=EXPECTED_MRTD,
        rtmr0=EXPECTED_RMTR0,
        rtmr1=EXPECTED_RMTR1,
        rtmr2=EXPECTED_RMTR2,
        rtmr3=EXPECTED_RMTR3,
        user_data=None,
        raw_quote_size=4096,
        parsed_at=datetime.now(timezone.utc).isoformat(),
        raw_bytes=b"boot",
    )

    runtime_quote = RuntimeTdxQuote(
        version=4,
        att_key_type=2,
        tee_type=0x81,
        mrtd=EXPECTED_MRTD,
        rtmr0=EXPECTED_RMTR0,
        rtmr1=EXPECTED_RMTR1,
        rtmr2="0" * 96,
        rtmr3="0" * 96,  # Different runtime values
        user_data=None,
        raw_quote_size=4096,
        parsed_at=datetime.now(timezone.utc).isoformat(),
        raw_bytes=b"runtime",
    )

    # Boot quote should verify successfully
    assert verify_measurements(boot_quote) is True

    # Runtime quote should also verify (only checks configured RTMRs)
    assert verify_measurements(runtime_quote) is True


# Comprehensive error case coverage
def test_all_quote_validation_errors():
    """Test all possible quote validation errors."""

    # Invalid base64
    with pytest.raises(InvalidQuoteError):
        BootTdxQuote.from_base64("not_base64!")

    # Too small
    with pytest.raises(InvalidQuoteError, match="Quote too short"):
        BootTdxQuote.from_bytes(b"tiny")

    # Invalid version
    invalid_quote = bytearray(1000)
    invalid_quote[0] = 99  # Invalid version
    with pytest.raises(InvalidQuoteError, match="Invalid quote version"):
        BootTdxQuote.from_bytes(bytes(invalid_quote))

    # Invalid TEE type
    invalid_quote = bytearray(1000)
    invalid_quote[0] = 4  # Valid version
    invalid_quote[8] = 0x99  # Invalid TEE type
    with pytest.raises(InvalidQuoteError, match="Invalid TEE type"):
        BootTdxQuote.from_bytes(bytes(invalid_quote))

    # Invalid att_key_type
    invalid_quote = bytearray(1000)
    invalid_quote[0] = 4  # Valid version
    invalid_quote[4] = 0x81  # Valid TEE type
    invalid_quote[2] = 99  # Invalid att_key_type
    with pytest.raises(InvalidQuoteError, match="Invalid attestation key type"):
        BootTdxQuote.from_bytes(bytes(invalid_quote))


def test_nonce_edge_cases():
    """Test nonce extraction edge cases."""

    # Nonce with only null bytes
    quote_null = BootTdxQuote(
        version=4,
        att_key_type=2,
        tee_type=0x81,
        mrtd="a" * 96,
        rtmr0="b" * 96,
        rtmr1="c" * 96,
        rtmr2="d" * 96,
        rtmr3="e" * 96,
        user_data="0" * 128,  # All null bytes
        raw_quote_size=4096,
        parsed_at=datetime.now(timezone.utc).isoformat(),
        raw_bytes=b"test",
    )

    extracted = extract_nonce(quote_null)
    assert extracted == ""

    # Nonce with non-printable characters
    non_printable_hex = "01020304050607080910111213141516" + "0" * 96
    quote_binary = BootTdxQuote(
        version=4,
        att_key_type=2,
        tee_type=0x81,
        mrtd="a" * 96,
        rtmr0="b" * 96,
        rtmr1="c" * 96,
        rtmr2="d" * 96,
        rtmr3="e" * 96,
        user_data=non_printable_hex,
        raw_quote_size=4096,
        parsed_at=datetime.now(timezone.utc).isoformat(),
        raw_bytes=b"test",
    )

    extracted = extract_nonce(quote_binary)
    assert extracted == ""  # Should stop at first non-printable

    # Nonce exactly 64 bytes
    max_nonce = "A" * 64
    max_nonce_hex = max_nonce.encode("utf-8").hex().upper()
    quote_max = BootTdxQuote(
        version=4,
        att_key_type=2,
        tee_type=0x81,
        mrtd="a" * 96,
        rtmr0="b" * 96,
        rtmr1="c" * 96,
        rtmr2="d" * 96,
        rtmr3="e" * 96,
        user_data=max_nonce_hex,
        raw_quote_size=4096,
        parsed_at=datetime.now(timezone.utc).isoformat(),
        raw_bytes=b"test",
    )

    extracted = extract_nonce(quote_max)
    assert extracted == max_nonce


# Cross-type verification tests
def test_boot_vs_runtime_verification_differences():
    """Test that boot and runtime quotes use different verification settings."""

    boot_quote = BootTdxQuote(
        version=4,
        att_key_type=2,
        tee_type=0x81,
        mrtd="a" * 96,
        rtmr0="boot_specific_rtmr0",
        rtmr1="boot_specific_rtmr1",
        rtmr2="d" * 96,
        rtmr3="e" * 96,
        user_data=None,
        raw_quote_size=4096,
        parsed_at=datetime.now(timezone.utc).isoformat(),
        raw_bytes=b"boot",
    )

    runtime_quote = RuntimeTdxQuote(
        version=4,
        att_key_type=2,
        tee_type=0x81,
        mrtd="a" * 96,
        rtmr0="runtime_specific_rtmr0",
        rtmr1="runtime_specific_rtmr1",
        rtmr2="f" * 96,
        rtmr3="0" * 96,
        user_data=None,
        raw_quote_size=4096,
        parsed_at=datetime.now(timezone.utc).isoformat(),
        raw_bytes=b"runtime",
    )

    with patch("api.server.util.settings") as mock_settings:
        # Configure different expectations for boot vs runtime
        mock_settings.expected_mrtd = "a" * 96
        mock_settings.expected_boot_rmtrs = {
            "rtmr0": "boot_specific_rtmr0",
            "rtmr1": "boot_specific_rtmr1",
        }
        mock_settings.exepected_runtime_rmtrs = {
            "rtmr0": "runtime_specific_rtmr0",
            "rtmr1": "runtime_specific_rtmr1",
        }

        # Both should verify successfully with their respective settings
        assert verify_measurements(boot_quote) is True
        assert verify_measurements(runtime_quote) is True

        # Boot quote should fail with wrong expectations
        boot_quote_copy = BootTdxQuote(
            version=4,
            att_key_type=2,
            tee_type=0x81,
            mrtd="a" * 96,
            rtmr0="runtime_specific_rtmr0",
            rtmr1="runtime_specific_rtmr1",
            rtmr2="f" * 96,
            rtmr3="0" * 96,
            user_data=None,
            raw_quote_size=4096,
            parsed_at=datetime.now(timezone.utc).isoformat(),
            raw_bytes=b"boot",
        )

        mock_settings.expected_boot_rmtrs = {
            "rtmr0": "different_boot_rtmr0",
        }

        with pytest.raises(MeasurementMismatchError):
            verify_measurements(boot_quote_copy)
