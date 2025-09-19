"""
Unit tests for api/server/service module.
Tests nonce management, attestation processing, server registration, and management operations.
"""

import base64
import json
import os
import pytest
import secrets
from datetime import datetime, timezone
from unittest.mock import AsyncMock, Mock, patch
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.exc import IntegrityError

from api.server.service import (
    create_nonce,
    validate_and_consume_nonce,
    verify_gpu_evidence,
    verify_quote,
    process_boot_attestation,
    process_runtime_attestation,
    register_server,
    check_server_ownership,
    get_server_attestation_status,
    list_servers,
    delete_server,
)
from api.server.schemas import (
    Server,
    ServerAttestation,
    BootAttestation,
    BootAttestationArgs,
    RuntimeAttestationArgs,
    ServerArgs,
)
from api.server.quote import BootTdxQuote, RuntimeTdxQuote, TdxVerificationResult
from api.server.exceptions import (
    InvalidQuoteError,
    MeasurementMismatchError,
    NonceError,
    ServerNotFoundError,
    ServerRegistrationError,
    InvalidSignatureError,
)
from tests.fixtures.gpus import TEST_GPU_NONCE
from tests.fixtures.tdx import EXPECTED_MRTD, TDX_BOOT_RMTRS, TDX_RUNTIME_RMTRS

TEST_SERVER_IP = "127.0.0.1"
TEST_NONCE = TEST_GPU_NONCE

@pytest.fixture
def mock_redis_client():
    """Mock Redis client for nonce operations."""
    redis_mock = AsyncMock()
    redis_mock.setex = AsyncMock(return_value=True)
    redis_mock.get = AsyncMock()
    redis_mock.delete = AsyncMock(return_value=1)
    return redis_mock


@pytest.fixture(autouse=True)
def mock_settings(mock_redis_client):
    """Mock settings with Redis client - auto-applied to all tests."""
    settings = Mock()
    settings.redis_client = mock_redis_client
    settings.expected_mrtd = EXPECTED_MRTD
    settings.expected_boot_rmtrs = {pair.split("=")[0]: pair.split("=")[1] for pair in TDX_BOOT_RMTRS.split(",")}
    settings.expected_runtime_rmtrs = {pair.split("=")[0]: pair.split("=")[1] for pair in TDX_RUNTIME_RMTRS.split(",")}
    settings.luks_passphrase = "test_luks_passphrase"
    
    with patch('api.server.service.settings', settings), \
        patch('api.server.util.settings', settings):
            yield settings


@pytest.fixture(autouse=True)
def mock_util_functions():
    """Mock utility functions that are consistently used."""
    with patch('api.server.service.generate_nonce', return_value=TEST_GPU_NONCE) as mock_gen, \
         patch('api.server.service.get_nonce_expiry_seconds', return_value=600) as mock_exp, \
         patch('api.server.service.extract_nonce', return_value=TEST_GPU_NONCE) as mock_extract, \
         patch('api.server.service.get_luks_passphrase', return_value="test_passphrase") as mock_luks, \
         patch('api.server.service.verify_gpu_evidence') as mock_verify_gpu:
            yield {
                'generate_nonce': mock_gen,
                'get_nonce_expiry_seconds': mock_exp,
                'extract_nonce': mock_extract,
                'get_luks_passphrase': mock_luks,
                'mock_verify_gpu': mock_verify_gpu
            }


@pytest.fixture(autouse=True)
def mock_sqlalchemy_func():
    """Mock SQLAlchemy func.now() - auto-applied to all tests."""
    with patch('api.server.service.func') as mock_func:
        mock_func.now.return_value = datetime.now(timezone.utc)
        yield mock_func


@pytest.fixture
def mock_db_session():
    """Mock database session."""
    session = AsyncMock(spec=AsyncSession)
    session.add = Mock()
    session.commit = AsyncMock()
    session.rollback = AsyncMock()
    session.refresh = AsyncMock()
    session.execute = AsyncMock()
    return session


# Test data fixtures

@pytest.fixture
def sample_boot_quote():
    """Sample BootTdxQuote for testing."""
    return BootTdxQuote(
        version=4,
        att_key_type=2,
        tee_type=0x81,
        mrtd="a" * 96,
        rtmr0="b" * 96,
        rtmr1="c" * 96,
        rtmr2="d" * 96,
        rtmr3="e" * 96,
        user_data="746573745f6e6f6e63655f31323300000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000",  # TEST_NONCE
        raw_quote_size=4096,
        parsed_at=datetime.now(timezone.utc).isoformat(),
        raw_bytes=b"dummy_boot_quote_bytes"
    )


@pytest.fixture
def sample_runtime_quote():
    """Sample RuntimeTdxQuote for testing."""
    return RuntimeTdxQuote(
        version=4,
        att_key_type=2,
        tee_type=0x81,
        mrtd="a" * 96,
        rtmr0="d" * 96,
        rtmr1="e" * 96,
        rtmr2="f" * 96,
        rtmr3="0" * 96,
        user_data="72756e74696d655f6e6f6e63655f34353600000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000",  # runtime_nonce_456
        raw_quote_size=4096,
        parsed_at=datetime.now(timezone.utc).isoformat(),
        raw_bytes=b"dummy_runtime_quote_bytes"
    )


@pytest.fixture
def sample_verification_result():
    """Sample TdxVerificationResult for testing."""
    return TdxVerificationResult(
        mrtd="a" * 96,
        rtmr0="b" * 96,
        rtmr1="c" * 96,
        rtmr2="d" * 96,
        rtmr3="e" * 96,
        user_data="test_data",
        parsed_at=datetime.now(timezone.utc),
        is_valid=True
    )


@pytest.fixture
def boot_attestation_args(valid_quote_base64):
    """Sample BootAttestationArgs for testing."""
    return BootAttestationArgs(
        quote=valid_quote_base64,  # base64 encoded "test_quote_data"
    )


@pytest.fixture
def runtime_attestation_args(valid_quote_base64):
    """Sample RuntimeAttestationArgs for testing."""
    return RuntimeAttestationArgs(
        quote=valid_quote_base64  # base64 encoded "runtime_quote_data"
    )


@pytest.fixture
def server_args(valid_quote_base64, sample_gpu_evidence_base64):
    """Sample ServerArgs for testing."""
    return ServerArgs(
        name="test-server",
        quote=valid_quote_base64,
        evidence=sample_gpu_evidence_base64
    )


@pytest.fixture
def sample_server():
    """Sample Server object for testing."""
    server = Server(
        server_id="test-server-123",
        name="test-server",
        ip=TEST_SERVER_IP,
        miner_hotkey="5FTestHotkey123",
        created_at=datetime.now(timezone.utc),
        updated_at=None
    )
    return server


@pytest.fixture
def sample_server_attestation():
    """Sample ServerAttestation object for testing."""
    return ServerAttestation(
        attestation_id="server-attest-123",
        server_id="test-server-123",
        quote_data="cnVudGltZV9xdW90ZV9kYXRh",
        mrtd="a" * 96,
        rtmrs={"rtmr0": "d" * 96, "rtmr1": "e" * 96},
        verification_result={"is_valid": True},
        verified=True,
        nonce_used="runtime_nonce_456",
        created_at=datetime.now(timezone.utc),
        verified_at=datetime.now(timezone.utc)
    )


# Mock verification functions as fixtures

@pytest.fixture
def mock_verify_quote_signature(sample_verification_result):
    """Mock verify_quote_signature function."""
    with patch('api.server.service.verify_quote_signature', return_value=sample_verification_result) as mock:
        yield mock


@pytest.fixture
def mock_verify_measurements():
    """Mock verify_measurements function."""
    with patch('api.server.service.verify_measurements', return_value=True) as mock:
        yield mock


@pytest.fixture
def mock_validate_nonce():
    """Mock validate_and_consume_nonce function."""
    with patch('api.server.service.validate_and_consume_nonce') as mock:
        yield mock


@pytest.fixture
def mock_quote_parsing(sample_boot_quote, sample_runtime_quote):
    """Mock quote parsing functions."""
    with patch('api.server.service.BootTdxQuote.from_base64', return_value=sample_boot_quote) as mock_boot:
        with patch('api.server.service.RuntimeTdxQuote.from_base64', return_value=sample_runtime_quote) as mock_runtime:
            yield {'boot': mock_boot, 'runtime': mock_runtime}


# Nonce Management Tests

@pytest.mark.asyncio
async def test_create_nonce(mock_settings):
    """Test creating a boot nonce."""
    result = await create_nonce(TEST_SERVER_IP)
    
    assert result["nonce"] == TEST_NONCE
    assert "expires_at" in result
    
    # Verify Redis operations
    mock_settings.redis_client.setex.assert_called_once_with(
        f"nonce:{TEST_NONCE}", 600, TEST_SERVER_IP
    )

@pytest.mark.asyncio
async def test_validate_and_consume_nonce_success(mock_settings):
    """Test successful nonce validation and consumption."""
    mock_settings.redis_client.get.return_value = f"{TEST_SERVER_IP}".encode()
    x = b'TEST'
    # Should not raise any exception
    await validate_and_consume_nonce(TEST_GPU_NONCE, TEST_SERVER_IP)
    
    mock_settings.redis_client.get.assert_called_once_with(f"nonce:{TEST_NONCE}")
    mock_settings.redis_client.delete.assert_called_once_with(f"nonce:{TEST_NONCE}")


@pytest.mark.asyncio
async def test_validate_and_consume_nonce_not_found(mock_settings):
    """Test nonce validation when nonce doesn't exist."""
    mock_settings.redis_client.get.return_value = None
    
    with pytest.raises(NonceError, match="Nonce not found or expired"):
        await validate_and_consume_nonce("invalid_nonce", "boot")


@pytest.mark.asyncio
async def test_validate_and_consume_nonce_server_mismatch(mock_settings):
    """Test nonce validation with wrong server ID."""
    mock_settings.redis_client.get.return_value = TEST_SERVER_IP.encode()
    
    with pytest.raises(NonceError, match="Nonce server mismatch"):
        await validate_and_consume_nonce(TEST_GPU_NONCE, "192.168.0.1")


@pytest.mark.asyncio
async def test_validate_and_consume_nonce_already_consumed(mock_settings):
    """Test nonce validation when nonce was already consumed."""
    mock_settings.redis_client.get.return_value = TEST_SERVER_IP.encode()
    mock_settings.redis_client.delete.return_value = 0  # Nothing deleted (already consumed)
    
    with pytest.raises(NonceError, match="Nonce was already consumed"):
        await validate_and_consume_nonce(TEST_GPU_NONCE, TEST_SERVER_IP)


# Quote Verification Tests

@pytest.mark.asyncio
async def test_verify_quote_success(sample_boot_quote, mock_validate_nonce, mock_verify_quote_signature, mock_verify_measurements):
    """Test successful quote verification."""
    result = await verify_quote(sample_boot_quote, TEST_NONCE)
    
    assert isinstance(result, TdxVerificationResult)
    mock_verify_quote_signature.assert_called_once_with(sample_boot_quote)
    mock_verify_measurements.assert_called_once_with(sample_boot_quote)


@pytest.mark.asyncio
async def test_verify_quote_nonce_failure(sample_boot_quote, mock_validate_nonce):
    """Test quote verification with nonce failure."""
    mock_validate_nonce.side_effect = NonceError("Invalid nonce")
    
    with pytest.raises(NonceError):
        await verify_quote(sample_boot_quote, "INVALID_NONCE")


@pytest.mark.asyncio
async def test_verify_quote_signature_failure(sample_boot_quote, mock_validate_nonce, mock_verify_quote_signature):
    """Test quote verification with signature failure."""
    mock_verify_quote_signature.side_effect = InvalidSignatureError("Invalid signature")
    
    with pytest.raises(InvalidSignatureError):
        await verify_quote(sample_boot_quote, TEST_NONCE)


@pytest.mark.asyncio
async def test_verify_quote_measurement_failure(sample_boot_quote, mock_validate_nonce, mock_verify_quote_signature, mock_verify_measurements):
    """Test quote verification with measurement failure."""
    mock_verify_measurements.side_effect = MeasurementMismatchError("MRTD mismatch")
    
    with pytest.raises(MeasurementMismatchError):
        await verify_quote(sample_boot_quote, TEST_NONCE)


# Boot Attestation Tests

@pytest.mark.asyncio
async def test_process_boot_attestation_success(mock_db_session, boot_attestation_args, mock_quote_parsing, mock_verify_quote_signature, mock_verify_measurements, mock_validate_nonce):
    """Test successful boot attestation processing."""
    # Setup mocks for verification success
    with patch('api.server.service.verify_quote') as mock_verify:
        mock_verify.return_value = TdxVerificationResult(
            mrtd="a" * 96, rtmr0="b" * 96, rtmr1="c" * 96, rtmr2="d" * 96, rtmr3="e" * 96,
            user_data="test", parsed_at=datetime.now(timezone.utc), is_valid=True
        )
        
        # Mock database refresh to set attestation_id
        def mock_refresh(obj):
            obj.attestation_id = "boot-attest-123"
            obj.verified_at = datetime.now(timezone.utc)
        mock_db_session.refresh.side_effect = mock_refresh
        
        result = await process_boot_attestation(mock_db_session, TEST_SERVER_IP, boot_attestation_args, TEST_NONCE)
        
        assert result["luks_passphrase"] == "test_passphrase"
        assert result["attestation_id"] == "boot-attest-123"
        assert "verified_at" in result
        
        # Verify database operations
        mock_db_session.add.assert_called_once()
        mock_db_session.commit.assert_called_once()


@pytest.mark.asyncio
async def test_process_boot_attestation_quote_failure(mock_db_session, boot_attestation_args):
    """Test boot attestation with quote parsing failure."""
    with patch('api.server.service.BootTdxQuote.from_base64', side_effect=InvalidQuoteError("Invalid quote")):
        with pytest.raises(InvalidQuoteError):
            await process_boot_attestation(mock_db_session, TEST_SERVER_IP, boot_attestation_args, TEST_NONCE)


@pytest.mark.asyncio
async def test_process_boot_attestation_verification_failure(mock_db_session, boot_attestation_args, sample_boot_quote):
    """Test boot attestation with verification failure."""
    with patch('api.server.service.BootTdxQuote.from_base64', return_value=sample_boot_quote):
        with patch('api.server.service.verify_quote', side_effect=MeasurementMismatchError("Measurement failed")):
            
            with pytest.raises(MeasurementMismatchError):
                await process_boot_attestation(mock_db_session, TEST_SERVER_IP, boot_attestation_args, TEST_NONCE)
            
            # Should still create failed attestation record
            mock_db_session.add.assert_called_once()
            mock_db_session.commit.assert_called_once()


# Runtime Attestation Tests

@pytest.mark.asyncio
async def test_process_runtime_attestation_success(mock_db_session, runtime_attestation_args, sample_server):
    """Test successful runtime attestation processing."""
    server_id = "test-server-123"
    miner_hotkey = "5FTestHotkey123"
    
    with patch('api.server.service.check_server_ownership', return_value=sample_server):
        with patch('api.server.service.verify_quote') as mock_verify:
            mock_verify.return_value = TdxVerificationResult(
                mrtd="a" * 96, rtmr0="d" * 96, rtmr1="e" * 96, rtmr2="f" * 96, rtmr3="0" * 96,
                user_data="test", parsed_at=datetime.now(timezone.utc), is_valid=True
            )
            
            def mock_refresh(obj):
                obj.attestation_id = "runtime-attest-123"
                obj.verified_at = datetime.now(timezone.utc)
            mock_db_session.refresh.side_effect = mock_refresh
            
            result = await process_runtime_attestation(
                mock_db_session, server_id, TEST_SERVER_IP, runtime_attestation_args, \
                miner_hotkey, TEST_NONCE
            )
            
            assert result["attestation_id"] == "runtime-attest-123"
            assert result["status"] == "verified"
            assert "verified_at" in result
            
            mock_db_session.add.assert_called_once()
            mock_db_session.commit.assert_called_once()


@pytest.mark.asyncio
async def test_process_runtime_attestation_server_not_found(mock_db_session, runtime_attestation_args):
    """Test runtime attestation when server is not found."""
    server_id = "nonexistent-server"
    miner_hotkey = "5FTestHotkey123"
    
    with patch('api.server.service.check_server_ownership', side_effect=ServerNotFoundError(server_id)):
        with pytest.raises(ServerNotFoundError):
            await process_runtime_attestation(
                mock_db_session, server_id, TEST_SERVER_IP, runtime_attestation_args, miner_hotkey, TEST_NONCE
            )


# Server Registration Tests

@pytest.mark.asyncio
async def test_register_server_success(mock_db_session, server_args):
    """Test successful server registration."""
    miner_hotkey = "5FTestHotkey123"
    
    def mock_refresh(obj):
        obj.server_id = "test-server-123"
    mock_db_session.refresh.side_effect = mock_refresh
    
    await register_server(mock_db_session, TEST_SERVER_IP, server_args, miner_hotkey, TEST_GPU_NONCE)
    
    # Verify database operations
    mock_db_session.add.assert_called_once()
    mock_db_session.commit.assert_called_once()
    mock_db_session.refresh.assert_called_once()


@pytest.mark.asyncio
async def test_register_server_integrity_error(mock_db_session, server_args):
    """Test server registration with database integrity error."""
    miner_hotkey = "5FTestHotkey123"
    
    mock_db_session.commit.side_effect = IntegrityError("Duplicate key", None, None)
    
    with pytest.raises(ServerRegistrationError, match="constraint violation"):
        await register_server(mock_db_session, TEST_SERVER_IP, server_args, miner_hotkey, TEST_NONCE)
    
    mock_db_session.rollback.assert_called_once()


# Server Ownership Tests

@pytest.mark.asyncio
async def test_check_server_ownership_success(mock_db_session, sample_server):
    """Test successful server ownership check."""
    server_id = "test-server-123"
    miner_hotkey = "5FTestHotkey123"
    
    # Mock database query result
    mock_result = Mock()
    mock_result.scalar_one_or_none.return_value = sample_server
    mock_db_session.execute.return_value = mock_result
    
    result = await check_server_ownership(mock_db_session, server_id, miner_hotkey)
    
    assert result == sample_server
    mock_db_session.execute.assert_called_once()


@pytest.mark.asyncio
async def test_check_server_ownership_not_found(mock_db_session):
    """Test server ownership check when server not found."""
    server_id = "nonexistent-server"
    miner_hotkey = "5FTestHotkey123"
    
    mock_result = Mock()
    mock_result.scalar_one_or_none.return_value = None
    mock_db_session.execute.return_value = mock_result
    
    with pytest.raises(ServerNotFoundError):
        await check_server_ownership(mock_db_session, server_id, miner_hotkey)


# Server Attestation Status Tests

@pytest.mark.asyncio
async def test_get_server_attestation_status_with_attestation(mock_db_session, sample_server, sample_server_attestation):
    """Test getting server attestation status with existing attestation."""
    server_id = "test-server-123"
    miner_hotkey = "5FTestHotkey123"
    
    with patch('api.server.service.check_server_ownership', return_value=sample_server):
        mock_result = Mock()
        mock_result.scalar_one_or_none.return_value = sample_server_attestation
        mock_db_session.execute.return_value = mock_result
        
        result = await get_server_attestation_status(mock_db_session, server_id, miner_hotkey)
        
        assert result["server_id"] == server_id
        assert result["server_name"] == sample_server.name
        assert result["attestation_status"] == "verified"
        assert result["last_attestation"]["attestation_id"] == sample_server_attestation.attestation_id


@pytest.mark.asyncio
async def test_get_server_attestation_status_no_attestation(mock_db_session, sample_server):
    """Test getting server attestation status with no attestations."""
    server_id = "test-server-123"
    miner_hotkey = "5FTestHotkey123"
    
    with patch('api.server.service.check_server_ownership', return_value=sample_server):
        mock_result = Mock()
        mock_result.scalar_one_or_none.return_value = None
        mock_db_session.execute.return_value = mock_result
        
        result = await get_server_attestation_status(mock_db_session, server_id, miner_hotkey)
        
        assert result["server_id"] == server_id
        assert result["attestation_status"] == "never_attested"
        assert result["last_attestation"] is None


# Server Listing Tests

@pytest.mark.asyncio
async def test_list_servers_success(mock_db_session):
    """Test successful server listing."""
    miner_hotkey = "5FTestHotkey123"
    
    servers = [
        Server(server_id="server-1", name="server-1", miner_hotkey=miner_hotkey, created_at=datetime.now(timezone.utc)),
        Server(server_id="server-2", name="server-2", miner_hotkey=miner_hotkey, created_at=datetime.now(timezone.utc))
    ]
    
    mock_result = Mock()
    mock_result.scalars.return_value.all.return_value = servers
    mock_db_session.execute.return_value = mock_result
    
    result = await list_servers(mock_db_session, miner_hotkey)
    
    assert len(result) == 2
    assert result == servers


@pytest.mark.asyncio
async def test_list_servers_empty(mock_db_session):
    """Test server listing with no servers."""
    miner_hotkey = "5FTestHotkey123"
    
    mock_result = Mock()
    mock_result.scalars.return_value.all.return_value = []
    mock_db_session.execute.return_value = mock_result
    
    result = await list_servers(mock_db_session, miner_hotkey)
    
    assert len(result) == 0


# Server Deletion Tests

@pytest.mark.asyncio
async def test_delete_server_success(mock_db_session, sample_server):
    """Test successful server deletion (marking as inactive)."""
    server_id = "test-server-123"
    miner_hotkey = "5FTestHotkey123"
    
    with patch('api.server.service.check_server_ownership', return_value=sample_server):
        result = await delete_server(mock_db_session, server_id, miner_hotkey)
        
        assert result is True
        mock_db_session.delete.assert_called_once()
        mock_db_session.commit.assert_called_once()


@pytest.mark.asyncio
async def test_delete_server_not_found(mock_db_session):
    """Test server deletion when server not found."""
    server_id = "nonexistent-server"
    miner_hotkey = "5FTestHotkey123"
    
    with patch('api.server.service.check_server_ownership', side_effect=ServerNotFoundError(server_id)):
        with pytest.raises(ServerNotFoundError):
            await delete_server(mock_db_session, server_id, miner_hotkey)


# Edge Cases and Error Handling Tests

@pytest.mark.asyncio
async def test_create_nonce_redis_failure(mock_settings):
    """Test nonce creation when Redis fails."""
    mock_settings.redis_client.setex.side_effect = Exception("Redis connection failed")
    
    with pytest.raises(Exception):
        await create_nonce("boot")


@pytest.mark.asyncio
async def test_validate_nonce_invalid_format(mock_settings):
    """Test nonce validation when Redis value can't be decoded properly."""
    mock_settings.redis_client.get.return_value = b'\xff\xfe\xfd'
    
    with pytest.raises(NonceError, match="Invalid nonce format"):
        await validate_and_consume_nonce(TEST_GPU_NONCE, "boot")


@pytest.mark.asyncio
async def test_register_server_general_exception(mock_db_session, server_args):
    """Test server registration with general exception."""
    miner_hotkey = "5FTestHotkey123"
    
    mock_db_session.commit.side_effect = Exception("Database error")
    
    with pytest.raises(ServerRegistrationError, match="Server registration failed"):
        await register_server(mock_db_session, TEST_SERVER_IP, server_args, miner_hotkey, TEST_NONCE)
    
    mock_db_session.rollback.assert_called_once()


# Parameterized Tests
@pytest.mark.parametrize("redis_value,expected_error", [
    (None, "Nonce not found or expired"),
    (TEST_SERVER_IP, "Invalid nonce format"),
    (b"192.168.0.1", "Nonce server mismatch")
])
@pytest.mark.asyncio
async def test_nonce_validation_error_cases(mock_settings, redis_value, expected_error):
    """Test various nonce validation error scenarios."""
    mock_settings.redis_client.get.return_value = redis_value
    
    with pytest.raises(NonceError, match=expected_error):
        await validate_and_consume_nonce(TEST_GPU_NONCE, TEST_SERVER_IP)


# Integration-style Tests (Testing Multiple Functions Together)

@pytest.mark.asyncio
async def test_full_boot_flow_end_to_end(mock_db_session, mock_settings, mock_verify_measurements):
    """Test complete boot attestation flow."""
    # Step 1: Create nonce
    mock_settings.redis_client.get.return_value = b"boot:boot"
    
    nonce_result = await create_nonce("boot")
    assert nonce_result["nonce"] == TEST_GPU_NONCE
    
    # Step 2: Create quote with nonce
    boot_quote = BootTdxQuote(
        version=4, att_key_type=2, tee_type=0x81,
        mrtd="a" * 96, rtmr0="b" * 96, rtmr1="c" * 96, rtmr2="d" * 96, rtmr3="e" * 96,
        user_data="626f6f745f6e6f6e63655f31323300000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000",  # boot_nonce_123
        raw_quote_size=4096, parsed_at=datetime.now(timezone.utc).isoformat(),
        raw_bytes=b"boot_quote"
    )
    
    # Step 3: Process attestation
    args = BootAttestationArgs(quote="dGVzdF9xdW90ZV9kYXRh")
    
    with patch('api.server.service.BootTdxQuote.from_base64', return_value=boot_quote):
        with patch('api.server.service.verify_quote_signature') as mock_verify:
            mock_verify.return_value = TdxVerificationResult(
                mrtd="a" * 96, rtmr0="b" * 96, rtmr1="c" * 96, rtmr2="d" * 96, rtmr3="e" * 96,
                user_data="test", parsed_at=datetime.now(timezone.utc), is_valid=True
            )
            
            def mock_refresh(obj):
                obj.attestation_id = "boot-attest-123"
                obj.verified_at = datetime.now(timezone.utc)
            mock_db_session.refresh.side_effect = mock_refresh
            
            result = await process_boot_attestation(mock_db_session, TEST_SERVER_IP, args, TEST_NONCE)
            
            assert result["luks_passphrase"] == "test_passphrase"
            assert result["attestation_id"] == "boot-attest-123"


@pytest.mark.asyncio
async def test_full_runtime_flow_end_to_end(mock_db_session, mock_settings, sample_server, mock_verify_measurements):
    """Test complete runtime attestation flow."""
    server_id = "test-server-123"
    miner_hotkey = "5FTestHotkey123"
    
    # Step 1: Create runtime nonce
    mock_settings.redis_client.get.return_value = f"runtime:{server_id}".encode()
    
    nonce_result = await create_nonce(TEST_SERVER_IP)
    assert nonce_result["nonce"] == TEST_NONCE
    
    # Step 2: Process runtime attestation
    runtime_quote = RuntimeTdxQuote(
        version=4, att_key_type=2, tee_type=0x81,
        mrtd="a" * 96, rtmr0="d" * 96, rtmr1="e" * 96, rtmr2="f" * 96, rtmr3="0" * 96,
        user_data="72756e74696d655f6e6f6e63655f34353600000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000",  # runtime_nonce_456
        raw_quote_size=4096, parsed_at=datetime.now(timezone.utc).isoformat(),
        raw_bytes=b"runtime_quote"
    )
    
    args = RuntimeAttestationArgs(quote="cnVudGltZV9xdW90ZV9kYXRh")
    
    with patch('api.server.service.check_server_ownership', return_value=sample_server):
        with patch('api.server.service.RuntimeTdxQuote.from_base64', return_value=runtime_quote):
            with patch('api.server.service.verify_quote_signature') as mock_verify:
                mock_verify.return_value = TdxVerificationResult(
                    mrtd="a" * 96, rtmr0="d" * 96, rtmr1="e" * 96, rtmr2="f" * 96, rtmr3="0" * 96,
                    user_data="test", parsed_at=datetime.now(timezone.utc), is_valid=True
                )
                
                def mock_refresh(obj):
                    obj.attestation_id = "runtime-attest-123"
                    obj.verified_at = datetime.now(timezone.utc)
                mock_db_session.refresh.side_effect = mock_refresh
                
                result = await process_runtime_attestation(
                    mock_db_session, server_id, TEST_SERVER_IP, 
                    args, miner_hotkey, TEST_NONCE
                )
                
                assert result["status"] == "verified"
                assert result["attestation_id"] == "runtime-attest-123"


@pytest.mark.asyncio
async def test_server_lifecycle_flow(mock_db_session, sample_server, valid_quote_base64, sample_gpu_evidence_base64):
    """Test complete server lifecycle: register -> list -> delete."""
    miner_hotkey = "5FTestHotkey123"
    
    # Step 1: Register server
    server_args = ServerArgs(name="test-server", quote=valid_quote_base64, evidence=sample_gpu_evidence_base64)
    
    def mock_refresh(obj):
        obj.server_id = "test-server-123"
    mock_db_session.refresh.side_effect = mock_refresh
    
    await register_server(mock_db_session, TEST_SERVER_IP, server_args, miner_hotkey, TEST_NONCE)
    mock_db_session.add.assert_called()
    mock_db_session.commit.assert_called()
    
    # Step 2: List servers
    mock_result = Mock()
    mock_result.scalars.return_value.all.return_value = [sample_server]
    mock_db_session.execute.return_value = mock_result
    
    servers = await list_servers(mock_db_session, miner_hotkey)
    assert len(servers) == 1
    assert servers[0] == sample_server
    
    # Step 3: Check ownership
    mock_ownership_result = Mock()
    mock_ownership_result.scalar_one_or_none.return_value = sample_server
    mock_db_session.execute.return_value = mock_ownership_result
    
    owned_server = await check_server_ownership(mock_db_session, "test-server-123", miner_hotkey)
    assert owned_server == sample_server
    
    # Step 4: Delete server
    with patch('api.server.service.check_server_ownership', return_value=sample_server):
        deleted = await delete_server(mock_db_session, "test-server-123", miner_hotkey)
        assert deleted is True


# Error Recovery Tests

@pytest.mark.asyncio
async def test_boot_attestation_partial_failure_recovery(mock_db_session, boot_attestation_args, sample_boot_quote):
    """Test boot attestation handles partial failures gracefully."""
    # Simulate verification failure but ensure failed record is still created
    with patch('api.server.service.BootTdxQuote.from_base64', return_value=sample_boot_quote):
        with patch('api.server.service.verify_quote', side_effect=MeasurementMismatchError("MRTD mismatch")):
            
            with pytest.raises(MeasurementMismatchError):
                await process_boot_attestation(mock_db_session, TEST_SERVER_IP, \
                    boot_attestation_args, TEST_NONCE)
            
            # Should still create failed attestation record
            mock_db_session.add.assert_called_once()
            mock_db_session.commit.assert_called_once()
            
            # Verify the failed record has correct fields
            call_args = mock_db_session.add.call_args[0][0]
            assert isinstance(call_args, BootAttestation)
            assert call_args.verified is False
            assert call_args.verification_error == "MRTD mismatch"


@pytest.mark.asyncio
async def test_runtime_attestation_partial_failure_recovery(mock_db_session, runtime_attestation_args, sample_runtime_quote, sample_server):
    """Test runtime attestation handles partial failures gracefully."""
    server_id = "test-server-123"
    miner_hotkey = "5FTestHotkey123"
    
    with patch('api.server.service.check_server_ownership', return_value=sample_server):
        with patch('api.server.service.RuntimeTdxQuote.from_base64', return_value=sample_runtime_quote):
            with patch('api.server.service.verify_quote', side_effect=InvalidQuoteError("Invalid quote")):
                
                with pytest.raises(InvalidQuoteError):
                    await process_runtime_attestation(
                        mock_db_session, server_id, TEST_SERVER_IP, \
                        runtime_attestation_args, miner_hotkey, TEST_NONCE
                    )
                
                # Should still create failed attestation record
                mock_db_session.add.assert_called_once()
                mock_db_session.commit.assert_called_once()
                
                # Verify the failed record has correct fields
                call_args = mock_db_session.add.call_args[0][0]
                assert isinstance(call_args, ServerAttestation)
                assert call_args.verified is False
                assert call_args.verification_error == "Invalid quote"


# Performance and Concurrency Tests

@pytest.mark.asyncio
async def test_multiple_nonce_operations_concurrent(mock_settings):
    """Test concurrent nonce operations don't interfere."""
    # Override the generate_nonce mock to return unique values for each call
    with patch('api.server.service.generate_nonce', side_effect=lambda: secrets.token_hex(16)):
        # Create multiple nonces concurrently
        import asyncio
        tasks = [create_nonce("boot") for _ in range(5)]
        results = await asyncio.gather(*tasks)
        
        # All should succeed
        assert len(results) == 5
        for result in results:
            assert "nonce" in result
            assert "expires_at" in result
        
        # Redis should have been called 5 times
        assert mock_settings.redis_client.setex.call_count == 5


# Quote Type Specific Tests

@pytest.mark.asyncio
async def test_verify_quote_boot_vs_runtime_different_settings(mock_verify_quote_signature, mock_settings):
    """Test that boot and runtime quotes use different verification settings."""
    boot_quote = BootTdxQuote(
        version=4, att_key_type=2, tee_type=0x81,
        mrtd="a" * 96, rtmr0="boot_specific_rtmr0", rtmr1="boot_specific_rtmr1",
        rtmr2="d" * 96, rtmr3="e" * 96,
        user_data="test", raw_quote_size=4096,
        parsed_at=datetime.now(timezone.utc).isoformat(), raw_bytes=b"boot"
    )
    
    runtime_quote = RuntimeTdxQuote(
        version=4, att_key_type=2, tee_type=0x81,
        mrtd="a" * 96, rtmr0="runtime_specific_rtmr0", rtmr1="runtime_specific_rtmr1",
        rtmr2="h" * 96, rtmr3="i" * 96,
        user_data="test", raw_quote_size=4096,
        parsed_at=datetime.now(timezone.utc).isoformat(), raw_bytes=b"runtime"
    )

    mock_settings.expected_mrtd = "a" * 96

    mock_settings.expected_boot_rmtrs["rtmr0"] = "boot_specific_rtmr0"
    mock_settings.expected_boot_rmtrs["rtmr1"] = "boot_specific_rtmr1"
    mock_settings.expected_boot_rmtrs["rtmr2"] = "d" * 96
    mock_settings.expected_boot_rmtrs["rtmr3"] = "e" * 96

    mock_settings.expected_runtime_rmtrs["rtmr0"] = "runtime_specific_rtmr0"
    mock_settings.expected_runtime_rmtrs["rtmr1"] = "runtime_specific_rtmr1"
    mock_settings.expected_runtime_rmtrs["rtmr2"] = "h" * 96
    mock_settings.expected_runtime_rmtrs["rtmr3"] = "i" * 96
    
    # Both should call their respective verification functions
    await verify_quote(boot_quote, TEST_NONCE)
    await verify_quote(runtime_quote, TEST_NONCE)
    
    # Both quotes should have been verified
    assert mock_verify_quote_signature.call_count == 2


# Special Edge Cases

@pytest.mark.asyncio
async def test_get_server_attestation_status_failed_attestation(mock_db_session, sample_server):
    """Test getting server attestation status with failed attestation."""
    server_id = "test-server-123"
    miner_hotkey = "5FTestHotkey123"
    
    # Create failed attestation
    failed_attestation = ServerAttestation(
        attestation_id="failed-attest-123",
        server_id=server_id,
        verified=False,
        verification_error="Measurement mismatch",
        created_at=datetime.now(timezone.utc),
        verified_at=None  # No verified_at for failed attestations
    )
    
    with patch('api.server.service.check_server_ownership', return_value=sample_server):
        mock_result = Mock()
        mock_result.scalar_one_or_none.return_value = failed_attestation
        mock_db_session.execute.return_value = mock_result
        
        result = await get_server_attestation_status(mock_db_session, server_id, miner_hotkey)
        
        assert result["attestation_status"] == "failed"
        assert result["last_attestation"]["verified"] is False
        assert result["last_attestation"]["verification_error"] == "Measurement mismatch"
        assert result["last_attestation"]["verified_at"] is None


# Database Transaction Tests

@pytest.mark.asyncio
async def test_boot_attestation_database_rollback_on_error(mock_db_session, boot_attestation_args, sample_boot_quote):
    """Test that database operations are rolled back on errors."""
    with patch('api.server.service.BootTdxQuote.from_base64', return_value=sample_boot_quote):
        with patch('api.server.service.verify_quote') as mock_verify:
            mock_verify.return_value = TdxVerificationResult(
                mrtd="a" * 96, rtmr0="b" * 96, rtmr1="c" * 96, rtmr2="d" * 96, rtmr3="e" * 96,
                user_data="test", parsed_at=datetime.now(timezone.utc), is_valid=True
            )
            
            # Mock commit to fail after add
            mock_db_session.commit.side_effect = Exception("Database connection lost")
            
            with pytest.raises(Exception, match="Database connection lost"):
                await process_boot_attestation(mock_db_session, TEST_SERVER_IP, boot_attestation_args, TEST_NONCE)
            
            # Verify add was called but rollback should not be called
            # (since we're not explicitly handling this exception)
            mock_db_session.add.assert_called_once()
            mock_db_session.commit.assert_called_once()


@pytest.mark.asyncio
async def test_runtime_attestation_database_rollback_on_error(mock_db_session, runtime_attestation_args, sample_runtime_quote, sample_server):
    """Test that runtime attestation database operations handle errors."""
    server_id = "test-server-123"
    miner_hotkey = "5FTestHotkey123"
    
    with patch('api.server.service.check_server_ownership', return_value=sample_server):
        with patch('api.server.service.RuntimeTdxQuote.from_base64', return_value=sample_runtime_quote):
            with patch('api.server.service.verify_quote') as mock_verify:
                mock_verify.return_value = TdxVerificationResult(
                    mrtd="a" * 96, rtmr0="d" * 96, rtmr1="e" * 96, rtmr2="f" * 96, rtmr3="0" * 96,
                    user_data="test", parsed_at=datetime.now(timezone.utc), is_valid=True
                )
                
                # Mock refresh to fail
                mock_db_session.refresh.side_effect = Exception("Database error during refresh")
                
                with pytest.raises(Exception, match="Database error during refresh"):
                    await process_runtime_attestation(
                        mock_db_session, server_id, TEST_SERVER_IP, \
                        runtime_attestation_args, miner_hotkey, TEST_NONCE
                    )
                
                mock_db_session.add.assert_called_once()
                mock_db_session.commit.assert_called_once()


# Comprehensive Quote Validation Tests

@pytest.mark.asyncio
async def test_verify_quote_with_different_quote_types(mock_verify_quote_signature, mock_verify_measurements):
    """Test quote verification with different quote implementations."""
    
    # Test with BootTdxQuote
    boot_quote = BootTdxQuote(
        version=4, att_key_type=2, tee_type=0x81,
        mrtd="a" * 96, rtmr0="b" * 96, rtmr1="c" * 96, rtmr2="d" * 96, rtmr3="e" * 96,
        user_data="test", raw_quote_size=4096,
        parsed_at=datetime.now(timezone.utc).isoformat(), raw_bytes=b"boot"
    )
    
    # Test with RuntimeTdxQuote  
    runtime_quote = RuntimeTdxQuote(
        version=4, att_key_type=2, tee_type=0x81,
        mrtd="a" * 96, rtmr0="d" * 96, rtmr1="e" * 96, rtmr2="f" * 96, rtmr3="0" * 96,
        user_data="test", raw_quote_size=4096,
        parsed_at=datetime.now(timezone.utc).isoformat(), raw_bytes=b"runtime"
    )

    # Both should verify successfully
    boot_result = await verify_quote(boot_quote, TEST_NONCE)
    runtime_result = await verify_quote(runtime_quote, TEST_NONCE)
    
    assert isinstance(boot_result, TdxVerificationResult)
    assert isinstance(runtime_result, TdxVerificationResult)
    
    # Verify both were processed
    assert mock_verify_quote_signature.call_count == 2
    assert mock_verify_measurements.call_count == 2