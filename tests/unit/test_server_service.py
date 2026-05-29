"""
Unit tests for api/server/service module.
Tests nonce management, attestation processing, server registration, and management operations.
"""

import json
import pytest
import secrets
from datetime import datetime, timezone
from unittest.mock import AsyncMock, Mock, patch
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.exc import IntegrityError
from cryptography.fernet import Fernet

from api.server.service import (
    BootAttestationResult,
    create_nonce,
    validate_and_consume_nonce,
    verify_quote,
    process_boot_attestation,
    process_runtime_attestation,
    register_server,
    check_server_ownership,
    get_server_by_name,
    update_server_name,
    get_server_attestation_status,
    delete_server,
    process_luks_passphrase_request,
)
from api.server.util import (
    get_default_root_passphrase,
    get_root_passphrase_for_boot,
)
from api.server.schemas import (
    Server,
    ServerAttestation,
    BootAttestation,
    BootAttestationArgs,
    RuntimeAttestationArgs,
    ServerArgs,
    RootPassphraseDefault,
    VmCacheConfig,
    VmAuthKey,
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
from api.config import TeeMeasurementConfig
from api.constants import NoncePurpose
from api.node.schemas import NodeArgs
from tests.fixtures.gpus import TEST_GPU_NONCE

TEST_SERVER_IP = "127.0.0.1"
TEST_NONCE = TEST_GPU_NONCE


def _tee_measurements_for_service_tests():
    """TeeMeasurementConfig list matching sample_boot_quote and sample_runtime_quote."""
    return [
        TeeMeasurementConfig(
            version="1",
            mrtd="a" * 96,
            name="test",
            boot_rtmrs={
                "RTMR0": "b" * 96,
                "RTMR1": "c" * 96,
                "RTMR2": "d" * 96,
                "RTMR3": "e" * 96,
            },
            runtime_rtmrs={
                "RTMR0": "d" * 96,
                "RTMR1": "e" * 96,
                "RTMR2": "f" * 96,
                "RTMR3": "0" * 96,
            },
            expected_gpus=["h200"],
            gpu_count=None,  # allow any count in unit tests
        ),
    ]


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
    settings.tee_measurements = _tee_measurements_for_service_tests()
    settings.luks_passphrase = "test_luks_passphrase"
    # Provide a real Fernet cipher so encrypt_passphrase/decrypt_passphrase work in tests.
    settings.fernet_key = Fernet(Fernet.generate_key())

    with (
        patch("api.server.service.settings", settings),
        patch("api.server.util.settings", settings),
    ):
        yield settings


TEST_CERT_HASH = "test_cert_hash"


@pytest.fixture(autouse=True)
def mock_util_functions():
    """Mock utility functions that are consistently used."""
    with (
        patch("api.server.service.generate_nonce", return_value=TEST_GPU_NONCE) as mock_gen,
        patch("api.server.service.get_nonce_expiry_seconds", return_value=600) as mock_exp,
        patch(
            "api.server.util.extract_report_data",
            return_value=(TEST_GPU_NONCE, TEST_CERT_HASH),
        ) as mock_extract,
        patch("api.server.service.verify_gpu_evidence") as mock_verify_gpu,
    ):
        yield {
            "generate_nonce": mock_gen,
            "get_nonce_expiry_seconds": mock_exp,
            "extract_report_data": mock_extract,
            "mock_verify_gpu": mock_verify_gpu,
        }


@pytest.fixture(autouse=True)
def mock_sqlalchemy_func():
    """Mock SQLAlchemy func.now() - auto-applied to all tests."""
    with patch("api.server.service.func") as mock_func:
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
        report_data=None,
        user_data="746573745f6e6f6e63655f31323300000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000",  # TEST_NONCE
        platform_id="0" * 32,
        raw_quote_size=4096,
        parsed_at=datetime.now(timezone.utc).isoformat(),
        raw_bytes=b"dummy_boot_quote_bytes",
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
        report_data=None,
        user_data="72756e74696d655f6e6f6e63655f34353600000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000",  # runtime_nonce_456
        platform_id="0" * 32,
        raw_quote_size=4096,
        parsed_at=datetime.now(timezone.utc).isoformat(),
        raw_bytes=b"dummy_runtime_quote_bytes",
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
        status="UpToDate",
        advisory_ids=[],
        td_attributes="0000001000000000",
    )


@pytest.fixture
def boot_attestation_args(valid_quote_base64):
    """Sample BootAttestationArgs for testing."""
    return BootAttestationArgs(
        quote=valid_quote_base64,
        miner_hotkey="5FTestHotkey123",
        vm_name="test-vm",
    )


@pytest.fixture
def runtime_attestation_args(valid_quote_base64):
    """Sample RuntimeAttestationArgs for testing."""
    return RuntimeAttestationArgs(
        quote=valid_quote_base64  # base64 encoded "runtime_quote_data"
    )


def _sample_node_args():
    """Minimal NodeArgs for ServerArgs.gpus (matches tee_measurements expected_gpus h200)."""
    return NodeArgs(
        uuid="gpu-uuid-1",
        name="GPU 0",
        memory=80 * 1024,
        clock_rate=1.41,
        device_index=0,
        gpu_identifier="h200",
        verification_host=TEST_SERVER_IP,
        verification_port=443,
    )


@pytest.fixture
def server_args():
    """Sample ServerArgs for testing."""
    return ServerArgs(
        id="test-server-123",
        host=TEST_SERVER_IP,
        name="test-vm-name",
        gpus=[_sample_node_args()],
    )


@pytest.fixture
def sample_server():
    """Sample Server object for testing."""
    server = Server(
        server_id="test-server-123",
        ip=TEST_SERVER_IP,
        miner_hotkey="5FTestHotkey123",
        name="test-vm-name",
        created_at=datetime.now(timezone.utc),
        updated_at=None,
    )
    return server


@pytest.fixture
def sample_server_attestation():
    """Sample ServerAttestation object for testing."""
    return ServerAttestation(
        attestation_id="server-attest-123",
        server_id="test-server-123",
        quote_data="cnVudGltZV9xdW90ZV9kYXRh",
        verification_error=None,
        measurement_version="1",
        created_at=datetime.now(timezone.utc),
        verified_at=datetime.now(timezone.utc),
    )


# Mock verification functions as fixtures


@pytest.fixture
def mock_verify_quote_signature(sample_verification_result):
    """Mock verify_quote_signature function."""
    with patch(
        "api.server.util.verify_quote_signature",
        return_value=sample_verification_result,
    ) as mock:
        yield mock


@pytest.fixture
def mock_verify_measurements():
    """Mock verify_measurements function."""
    with patch("api.server.util.verify_measurements", return_value=True) as mock:
        yield mock


@pytest.fixture
def mock_validate_nonce():
    """Mock validate_and_consume_nonce function."""
    with patch("api.server.service.validate_and_consume_nonce") as mock:
        yield mock


@pytest.fixture
def mock_quote_parsing(sample_boot_quote, sample_runtime_quote):
    """Mock quote parsing functions."""
    with patch(
        "api.server.service.BootTdxQuote.from_base64", return_value=sample_boot_quote
    ) as mock_boot:
        with patch(
            "api.server.service.RuntimeTdxQuote.from_base64",
            return_value=sample_runtime_quote,
        ) as mock_runtime:
            yield {"boot": mock_boot, "runtime": mock_runtime}


# Nonce Management Tests


@pytest.mark.asyncio
async def test_create_nonce(mock_settings):
    """Test creating a boot nonce."""
    result = await create_nonce(TEST_SERVER_IP, NoncePurpose.BOOT)

    assert result["nonce"] == TEST_NONCE
    assert "expires_at" in result

    # Verify Redis operations (value is JSON: server_ip + purpose + miner_hotkey)
    expected_value = json.dumps(
        {
            "server_ip": TEST_SERVER_IP,
            "purpose": NoncePurpose.BOOT.value,
            "miner_hotkey": None,
        }
    )
    mock_settings.redis_client.setex.assert_called_once_with(
        f"nonce:{TEST_NONCE}", 600, expected_value
    )


@pytest.mark.asyncio
async def test_validate_and_consume_nonce_success(mock_settings):
    """Test successful nonce validation and consumption."""
    mock_settings.redis_client.getdel.return_value = json.dumps(
        {"server_ip": TEST_SERVER_IP, "purpose": NoncePurpose.BOOT.value}
    ).encode()
    stored_hotkey = await validate_and_consume_nonce(
        TEST_GPU_NONCE, TEST_SERVER_IP, NoncePurpose.BOOT
    )

    mock_settings.redis_client.getdel.assert_called_once_with(f"nonce:{TEST_NONCE}")
    assert stored_hotkey is None


@pytest.mark.asyncio
async def test_validate_and_consume_nonce_returns_hotkey(mock_settings):
    """Test that validate_and_consume_nonce returns a stored miner_hotkey."""
    test_hotkey = "5FTestHotkey123"
    mock_settings.redis_client.getdel.return_value = json.dumps(
        {
            "server_ip": TEST_SERVER_IP,
            "purpose": NoncePurpose.BOOT.value,
            "miner_hotkey": test_hotkey,
        }
    ).encode()
    stored_hotkey = await validate_and_consume_nonce(
        TEST_GPU_NONCE, TEST_SERVER_IP, NoncePurpose.BOOT
    )

    assert stored_hotkey == test_hotkey


@pytest.mark.asyncio
async def test_validate_and_consume_nonce_not_found(mock_settings):
    """Test nonce validation when nonce doesn't exist (or was already consumed)."""
    mock_settings.redis_client.getdel.return_value = None

    with pytest.raises(NonceError, match="Nonce not found or expired"):
        await validate_and_consume_nonce("invalid_nonce", TEST_SERVER_IP, NoncePurpose.BOOT)


@pytest.mark.asyncio
async def test_validate_and_consume_nonce_server_mismatch(mock_settings):
    """Test nonce validation with wrong server ID."""
    mock_settings.redis_client.getdel.return_value = json.dumps(
        {"server_ip": TEST_SERVER_IP, "purpose": NoncePurpose.BOOT.value}
    ).encode()

    with pytest.raises(NonceError, match="Nonce server mismatch"):
        await validate_and_consume_nonce(TEST_GPU_NONCE, "192.168.0.1", NoncePurpose.BOOT)


# Quote Verification Tests


@pytest.mark.asyncio
async def test_verify_quote_success(
    sample_boot_quote,
    mock_validate_nonce,
    mock_verify_quote_signature,
    mock_verify_measurements,
):
    """Test successful quote verification."""
    result = await verify_quote(sample_boot_quote, TEST_NONCE, TEST_CERT_HASH)

    assert isinstance(result, TdxVerificationResult)
    mock_verify_quote_signature.assert_called_once_with(sample_boot_quote)
    mock_verify_measurements.assert_called_once_with(sample_boot_quote)


@pytest.mark.asyncio
async def test_verify_quote_nonce_failure(sample_boot_quote, mock_validate_nonce):
    """Test quote verification with nonce failure."""
    mock_validate_nonce.side_effect = NonceError("Invalid nonce")

    with pytest.raises(NonceError):
        await verify_quote(sample_boot_quote, "INVALID_NONCE", TEST_CERT_HASH)


@pytest.mark.asyncio
async def test_verify_quote_signature_failure(
    sample_boot_quote, mock_validate_nonce, mock_verify_quote_signature
):
    """Test quote verification with signature failure."""
    mock_verify_quote_signature.side_effect = InvalidSignatureError("Invalid signature")

    with pytest.raises(InvalidSignatureError):
        await verify_quote(sample_boot_quote, TEST_NONCE, TEST_CERT_HASH)


@pytest.mark.asyncio
async def test_verify_quote_measurement_failure(
    sample_boot_quote,
    mock_validate_nonce,
    mock_verify_quote_signature,
    mock_verify_measurements,
):
    """Test quote verification with measurement failure."""
    mock_verify_measurements.side_effect = MeasurementMismatchError("MRTD mismatch")

    with pytest.raises(MeasurementMismatchError):
        await verify_quote(sample_boot_quote, TEST_NONCE, TEST_CERT_HASH)


# Boot Attestation Tests


@pytest.mark.asyncio
async def test_process_boot_attestation_success(
    mock_db_session,
    boot_attestation_args,
    mock_quote_parsing,
    mock_verify_quote_signature,
    mock_verify_measurements,
    mock_validate_nonce,
):
    """Test successful boot attestation processing."""
    # Setup mocks for verification success
    with patch("api.server.service.verify_quote") as mock_verify:
        mock_verify.return_value = TdxVerificationResult(
            mrtd="a" * 96,
            rtmr0="b" * 96,
            rtmr1="c" * 96,
            rtmr2="d" * 96,
            rtmr3="e" * 96,
            user_data="test",
            parsed_at=datetime.now(timezone.utc),
            status="UpToDate",
            advisory_ids=[],
            td_attributes="0000001000000000",
        )

        # Mock database refresh to set attestation_id
        def mock_refresh(obj):
            obj.attestation_id = "boot-attest-123"
            obj.verified_at = datetime.now(timezone.utc)

        mock_db_session.refresh.side_effect = mock_refresh

        mock_keypair = Mock()
        mock_keypair.ss58_address = "5EphemeralSS58TestAddress"

        with (
            patch(
                "api.server.service.generate_and_store_boot_token",
                return_value="test-boot-token",
            ),
            patch(
                "api.server.service._handle_boot_version_update",
                new_callable=AsyncMock,
            ),
            patch(
                "api.server.service.get_root_passphrase_for_boot",
                new_callable=AsyncMock,
                return_value=("test_root_key", None, None),
            ),
            patch(
                "api.server.service._generate_and_store_vm_auth_key",
                new_callable=AsyncMock,
                return_value=mock_keypair,
            ),
        ):
            result = await process_boot_attestation(
                mock_db_session,
                TEST_SERVER_IP,
                boot_attestation_args,
                TEST_NONCE,
                TEST_CERT_HASH,
            )

        assert isinstance(result, BootAttestationResult)
        assert result.boot_token == "test-boot-token"
        assert result.root_key == "test_root_key"
        assert result.root_next is None
        assert result.root_confirm_nonce is None
        assert result.vm_auth_ss58 == "5EphemeralSS58TestAddress"

        # Verify database operations
        mock_db_session.add.assert_called_once()
        mock_db_session.commit.assert_called_once()


@pytest.mark.asyncio
async def test_process_boot_attestation_quote_failure(mock_db_session, boot_attestation_args):
    """Test boot attestation with quote parsing failure."""
    with patch(
        "api.server.service.BootTdxQuote.from_base64",
        side_effect=InvalidQuoteError("Invalid quote"),
    ):
        with pytest.raises(InvalidQuoteError):
            await process_boot_attestation(
                mock_db_session,
                TEST_SERVER_IP,
                boot_attestation_args,
                TEST_NONCE,
                TEST_CERT_HASH,
            )


@pytest.mark.asyncio
async def test_process_boot_attestation_verification_failure(
    mock_db_session, boot_attestation_args, sample_boot_quote
):
    """Test boot attestation with verification failure."""
    with patch("api.server.service.BootTdxQuote.from_base64", return_value=sample_boot_quote):
        with patch(
            "api.server.service.verify_quote",
            side_effect=MeasurementMismatchError("Measurement failed"),
        ):
            with pytest.raises(MeasurementMismatchError):
                await process_boot_attestation(
                    mock_db_session,
                    TEST_SERVER_IP,
                    boot_attestation_args,
                    TEST_NONCE,
                    TEST_CERT_HASH,
                )

            # Should still create failed attestation record
            mock_db_session.add.assert_called_once()
            mock_db_session.commit.assert_called_once()


# Runtime Attestation Tests


@pytest.mark.asyncio
async def test_process_runtime_attestation_success(
    mock_db_session, runtime_attestation_args, sample_server, sample_runtime_quote
):
    """Test successful runtime attestation processing."""
    server_id = "test-server-123"
    miner_hotkey = "5FTestHotkey123"

    with patch("api.server.service.check_server_ownership", return_value=sample_server):
        with patch(
            "api.server.service.RuntimeTdxQuote.from_base64",
            return_value=sample_runtime_quote,
        ):
            with patch("api.server.service.verify_quote") as mock_verify:
                mock_verify.return_value = TdxVerificationResult(
                    mrtd="a" * 96,
                    rtmr0="d" * 96,
                    rtmr1="e" * 96,
                    rtmr2="f" * 96,
                    rtmr3="0" * 96,
                    user_data="test",
                    parsed_at=datetime.now(timezone.utc),
                    status="UpToDate",
                    advisory_ids=[],
                    td_attributes="0000001000000000",
                )

                def mock_refresh(obj):
                    obj.attestation_id = "runtime-attest-123"
                    obj.verified_at = datetime.now(timezone.utc)

                mock_db_session.refresh.side_effect = mock_refresh

                result = await process_runtime_attestation(
                    mock_db_session,
                    server_id,
                    TEST_SERVER_IP,
                    runtime_attestation_args,
                    miner_hotkey,
                    TEST_NONCE,
                    TEST_CERT_HASH,
                )

            assert result["attestation_id"] == "runtime-attest-123"
            assert result["status"] == "verified"
            assert "verified_at" in result

            mock_db_session.add.assert_called_once()
            mock_db_session.commit.assert_called_once()


@pytest.mark.asyncio
async def test_process_runtime_attestation_server_not_found(
    mock_db_session, runtime_attestation_args
):
    """Test runtime attestation when server is not found."""
    server_id = "nonexistent-server"
    miner_hotkey = "5FTestHotkey123"

    with patch(
        "api.server.service.check_server_ownership",
        side_effect=ServerNotFoundError(server_id),
    ):
        with pytest.raises(ServerNotFoundError):
            await process_runtime_attestation(
                mock_db_session,
                server_id,
                TEST_SERVER_IP,
                runtime_attestation_args,
                miner_hotkey,
                TEST_NONCE,
                TEST_CERT_HASH,
            )


# Server Registration Tests


@pytest.mark.asyncio
async def test_register_server_success(mock_db_session, server_args, sample_server):
    """Test successful server registration."""
    miner_hotkey = "5FTestHotkey123"

    with patch("api.server.service._track_server", return_value=sample_server):
        with patch("api.server.service._track_nodes", new_callable=AsyncMock):
            with patch(
                "api.server.service.verify_server",
                new_callable=AsyncMock,
                return_value="1.0.0",
            ):
                await register_server(mock_db_session, server_args, miner_hotkey)

    assert sample_server.version == "1.0.0"
    mock_db_session.commit.assert_called()


@pytest.mark.asyncio
async def test_register_server_integrity_error(mock_db_session, server_args, sample_server):
    """Test server registration handles IntegrityError from _track_nodes."""
    miner_hotkey = "5FTestHotkey123"

    with patch("api.server.service._track_server", return_value=sample_server):
        with patch(
            "api.server.service._track_nodes",
            new_callable=AsyncMock,
            side_effect=IntegrityError("Duplicate key", None, None),
        ):
            with patch(
                "api.server.service.verify_server",
                new_callable=AsyncMock,
                return_value="1.0.0",
            ):
                with pytest.raises(ServerRegistrationError):
                    await register_server(mock_db_session, server_args, miner_hotkey)

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
async def test_get_server_attestation_status_with_attestation(
    mock_db_session, sample_server, sample_server_attestation
):
    """Test getting server attestation status with existing attestation."""
    server_id = "test-server-123"
    miner_hotkey = "5FTestHotkey123"

    with patch("api.server.service.check_server_ownership", return_value=sample_server):
        mock_result = Mock()
        mock_result.scalar_one_or_none.return_value = sample_server_attestation
        mock_db_session.execute.return_value = mock_result

        result = await get_server_attestation_status(mock_db_session, server_id, miner_hotkey)

        assert result["server_id"] == server_id
        assert result["attestation_status"] == "verified"
        assert (
            result["last_attestation"]["attestation_id"] == sample_server_attestation.attestation_id
        )


@pytest.mark.asyncio
async def test_get_server_attestation_status_no_attestation(mock_db_session, sample_server):
    """Test getting server attestation status with no attestations."""
    server_id = "test-server-123"
    miner_hotkey = "5FTestHotkey123"

    with patch("api.server.service.check_server_ownership", return_value=sample_server):
        mock_result = Mock()
        mock_result.scalar_one_or_none.return_value = None
        mock_db_session.execute.return_value = mock_result

        result = await get_server_attestation_status(mock_db_session, server_id, miner_hotkey)

        assert result["server_id"] == server_id
        assert result["attestation_status"] == "never_attested"
        assert result["last_attestation"] is None


# Server Deletion Tests


@pytest.mark.asyncio
async def test_delete_server_success(mock_db_session, sample_server):
    """Test successful server deletion (preserves LUKS config for potential reboot)."""
    server_id = "test-server-123"
    miner_hotkey = "5FTestHotkey123"

    with patch("api.server.service.check_server_ownership", return_value=sample_server):
        result = await delete_server(mock_db_session, server_id, miner_hotkey)

        assert result is True
        mock_db_session.delete.assert_called_once()
        mock_db_session.commit.assert_called_once()


@pytest.mark.asyncio
async def test_delete_server_not_found(mock_db_session):
    """Test server deletion when server not found."""
    server_id = "nonexistent-server"
    miner_hotkey = "5FTestHotkey123"

    with patch(
        "api.server.service.check_server_ownership",
        side_effect=ServerNotFoundError(server_id),
    ):
        with pytest.raises(ServerNotFoundError):
            await delete_server(mock_db_session, server_id, miner_hotkey)


# update_server_vm_name (sync server names) tests


@pytest.mark.asyncio
async def test_get_server_by_name_success(mock_db_session, sample_server):
    """Test get_server_by_name returns server when found."""
    miner_hotkey = sample_server.miner_hotkey
    server_name = sample_server.name
    mock_result = Mock()
    mock_result.scalar_one_or_none.return_value = sample_server
    mock_db_session.execute.return_value = mock_result

    result = await get_server_by_name(mock_db_session, miner_hotkey, server_name)

    assert result == sample_server


@pytest.mark.asyncio
async def test_get_server_by_name_not_found(mock_db_session):
    """Test get_server_by_miner_and_vm raises when server not found."""
    mock_result = Mock()
    mock_result.scalar_one_or_none.return_value = None
    mock_db_session.execute.return_value = mock_result

    with pytest.raises(ServerNotFoundError) as exc_info:
        await get_server_by_name(mock_db_session, "5FTestHotkey123", "nonexistent-vm")
    assert "nonexistent-vm" in str(exc_info.value.detail)


@pytest.mark.asyncio
async def test_update_server_name_success(mock_db_session, sample_server):
    """Test update_server_name updates name and returns server."""
    server_id = sample_server.server_id
    miner_hotkey = sample_server.miner_hotkey
    new_name = "my-actual-vm-name"

    with patch("api.server.service.check_server_ownership", return_value=sample_server):
        result = await update_server_name(mock_db_session, miner_hotkey, server_id, new_name)

    assert result.name == new_name
    mock_db_session.commit.assert_called_once()
    mock_db_session.refresh.assert_called_once_with(sample_server)


@pytest.mark.asyncio
async def test_update_server_name_idempotent(mock_db_session, sample_server):
    """Test update_server_name is idempotent when name unchanged."""
    server_id = sample_server.server_id
    miner_hotkey = sample_server.miner_hotkey
    existing_name = sample_server.name

    with patch("api.server.service.check_server_ownership", return_value=sample_server):
        result = await update_server_name(mock_db_session, miner_hotkey, server_id, existing_name)

    assert result == sample_server
    mock_db_session.commit.assert_not_called()
    mock_db_session.refresh.assert_not_called()


@pytest.mark.asyncio
async def test_update_server_name_not_found(mock_db_session):
    """Test update_server_vm_name raises when server not found."""
    with patch(
        "api.server.service.check_server_ownership",
        side_effect=ServerNotFoundError("nonexistent-server"),
    ):
        with pytest.raises(ServerNotFoundError):
            await update_server_name(
                mock_db_session,
                "5FTestHotkey123",
                "nonexistent-server",
                "new-vm-name",
            )


@pytest.mark.asyncio
async def test_update_server_name_conflict(mock_db_session, sample_server):
    """Test update_server_vm_name raises 409 when vm_name already in use."""
    from fastapi import HTTPException

    server_id = sample_server.server_id
    miner_hotkey = sample_server.miner_hotkey
    new_vm_name = "taken-vm-name"

    with patch("api.server.service.check_server_ownership", return_value=sample_server):
        mock_db_session.commit.side_effect = IntegrityError("conflict", None, None)
        with pytest.raises(HTTPException) as exc_info:
            await update_server_name(mock_db_session, miner_hotkey, server_id, new_vm_name)
    assert exc_info.value.status_code == 409
    mock_db_session.rollback.assert_called_once()


# LUKS passphrase tests


@pytest.mark.asyncio
async def test_sync_luks_passphrase(mock_db_session, mock_redis_client):
    """Test POST LUKS sync: validates token, calls sync_server_luks_passphrases, consumes token."""
    boot_token = "test-boot-token"
    hotkey = "5FTestHotkey123"
    vm_name = "test-vm"
    volume_names = ["storage", "cache"]
    rekey = ["cache"]

    with (
        patch(
            "api.server.service._validate_boot_token_for_luks",
            new_callable=AsyncMock,
        ),
        patch(
            "api.server.service.sync_server_luks_passphrases",
            AsyncMock(return_value={"storage": "pass1", "cache": "pass2_new"}),
        ) as mock_sync,
        patch("api.server.service.settings") as mock_settings,
    ):
        mock_settings.redis_client.delete = AsyncMock(return_value=1)
        result = await process_luks_passphrase_request(
            mock_db_session,
            boot_token,
            hotkey,
            vm_name,
            volume_names,
            rekey_volume_names=rekey,
        )
        assert result == {"storage": "pass1", "cache": "pass2_new"}
        mock_sync.assert_called_once_with(
            mock_db_session, hotkey, vm_name, volume_names, rekey_volume_names=rekey
        )
        mock_settings.redis_client.delete.assert_called_once()


# Edge Cases and Error Handling Tests


@pytest.mark.asyncio
async def test_create_nonce_redis_failure(mock_settings):
    """Test nonce creation when Redis fails."""
    mock_settings.redis_client.setex.side_effect = Exception("Redis connection failed")

    with pytest.raises(Exception):
        await create_nonce(TEST_SERVER_IP, NoncePurpose.BOOT)


@pytest.mark.asyncio
async def test_validate_nonce_invalid_format(mock_settings):
    """Test nonce validation when Redis value can't be decoded as JSON."""
    mock_settings.redis_client.getdel.return_value = b"\xff\xfe\xfd"

    with pytest.raises(NonceError, match="Invalid nonce format"):
        await validate_and_consume_nonce(TEST_GPU_NONCE, TEST_SERVER_IP, NoncePurpose.BOOT)


@pytest.mark.asyncio
async def test_register_server_general_exception(mock_db_session, server_args, sample_server):
    """Test server registration handles unexpected exceptions."""
    miner_hotkey = "5FTestHotkey123"

    with patch("api.server.service._track_server", return_value=sample_server):
        with patch(
            "api.server.service._track_nodes",
            new_callable=AsyncMock,
            side_effect=Exception("Database error"),
        ):
            with patch(
                "api.server.service.verify_server",
                new_callable=AsyncMock,
                return_value="1.0.0",
            ):
                with pytest.raises(ServerRegistrationError):
                    await register_server(mock_db_session, server_args, miner_hotkey)

    mock_db_session.rollback.assert_called_once()


# Parameterized Tests
@pytest.mark.parametrize(
    "redis_value,expected_error",
    [
        (None, "Nonce not found or expired"),
        (TEST_SERVER_IP, "Invalid nonce format"),
        (json.dumps("192.168.0.1").encode(), "Nonce server mismatch"),
    ],
)
@pytest.mark.asyncio
async def test_nonce_validation_error_cases(mock_settings, redis_value, expected_error):
    """Test various nonce validation error scenarios."""
    mock_settings.redis_client.getdel.return_value = redis_value

    with pytest.raises(NonceError, match=expected_error):
        await validate_and_consume_nonce(TEST_GPU_NONCE, TEST_SERVER_IP, NoncePurpose.BOOT)


# Integration-style Tests (Testing Multiple Functions Together)


@pytest.mark.asyncio
async def test_full_boot_flow_end_to_end(mock_db_session, mock_settings, mock_verify_measurements):
    """Test complete boot attestation flow."""
    # Step 1: Create nonce
    mock_settings.redis_client.get.return_value = json.dumps(
        {"server_ip": TEST_SERVER_IP, "purpose": NoncePurpose.BOOT.value}
    ).encode()

    nonce_result = await create_nonce(TEST_SERVER_IP, NoncePurpose.BOOT)
    assert nonce_result["nonce"] == TEST_GPU_NONCE

    # Step 2: Create quote with nonce
    boot_quote = BootTdxQuote(
        version=4,
        att_key_type=2,
        tee_type=0x81,
        mrtd="a" * 96,
        rtmr0="b" * 96,
        rtmr1="c" * 96,
        rtmr2="d" * 96,
        rtmr3="e" * 96,
        report_data=None,
        user_data="626f6f745f6e6f6e63655f31323300000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000",  # boot_nonce_123
        platform_id="0" * 32,
        raw_quote_size=4096,
        parsed_at=datetime.now(timezone.utc).isoformat(),
        raw_bytes=b"boot_quote",
    )

    # Step 3: Process attestation
    args = BootAttestationArgs(
        quote="dGVzdF9xdW90ZV9kYXRh",
        miner_hotkey="5FTestHotkey123",
        vm_name="test-vm",
    )

    with patch("api.server.service.BootTdxQuote.from_base64", return_value=boot_quote):
        with patch("api.server.util.verify_quote_signature") as mock_verify:
            mock_verify.return_value = TdxVerificationResult(
                mrtd="a" * 96,
                rtmr0="b" * 96,
                rtmr1="c" * 96,
                rtmr2="d" * 96,
                rtmr3="e" * 96,
                user_data="test",
                parsed_at=datetime.now(timezone.utc),
                status="UpToDate",
                advisory_ids=[],
                td_attributes="0000001000000000",
            )

            def mock_refresh(obj):
                obj.attestation_id = "boot-attest-123"
                obj.verified_at = datetime.now(timezone.utc)

            mock_db_session.refresh.side_effect = mock_refresh

            mock_keypair = Mock()
            mock_keypair.ss58_address = "5EphemeralSS58TestAddress"

            with (
                patch(
                    "api.server.service.generate_and_store_boot_token",
                    return_value="test-boot-token",
                ),
                patch(
                    "api.server.service._handle_boot_version_update",
                    new_callable=AsyncMock,
                ),
                patch(
                    "api.server.service.get_root_passphrase_for_boot",
                    new_callable=AsyncMock,
                    return_value=("test_root_key", None, None),
                ),
                patch(
                    "api.server.service._generate_and_store_vm_auth_key",
                    new_callable=AsyncMock,
                    return_value=mock_keypair,
                ),
            ):
                result = await process_boot_attestation(
                    mock_db_session,
                    TEST_SERVER_IP,
                    args,
                    TEST_NONCE,
                    TEST_CERT_HASH,
                )

            assert isinstance(result, BootAttestationResult)
            assert result.boot_token == "test-boot-token"
            assert result.root_key == "test_root_key"
            assert result.vm_auth_ss58 == "5EphemeralSS58TestAddress"


@pytest.mark.asyncio
async def test_full_runtime_flow_end_to_end(
    mock_db_session, mock_settings, sample_server, mock_verify_measurements
):
    """Test complete runtime attestation flow."""
    server_id = "test-server-123"
    miner_hotkey = "5FTestHotkey123"

    # Step 1: Create runtime nonce
    mock_settings.redis_client.get.return_value = json.dumps(
        {"server_ip": TEST_SERVER_IP, "purpose": NoncePurpose.RUNTIME.value}
    ).encode()

    nonce_result = await create_nonce(TEST_SERVER_IP, NoncePurpose.RUNTIME)
    assert nonce_result["nonce"] == TEST_NONCE

    # Step 2: Process runtime attestation
    runtime_quote = RuntimeTdxQuote(
        version=4,
        att_key_type=2,
        tee_type=0x81,
        mrtd="a" * 96,
        rtmr0="d" * 96,
        rtmr1="e" * 96,
        rtmr2="f" * 96,
        rtmr3="0" * 96,
        report_data=None,
        user_data="72756e74696d655f6e6f6e63655f34353600000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000",  # runtime_nonce_456
        platform_id="0" * 32,
        raw_quote_size=4096,
        parsed_at=datetime.now(timezone.utc).isoformat(),
        raw_bytes=b"runtime_quote",
    )

    args = RuntimeAttestationArgs(quote="cnVudGltZV9xdW90ZV9kYXRh")

    with patch("api.server.service.check_server_ownership", return_value=sample_server):
        with patch("api.server.service.RuntimeTdxQuote.from_base64", return_value=runtime_quote):
            with patch("api.server.util.verify_quote_signature") as mock_verify:
                mock_verify.return_value = TdxVerificationResult(
                    mrtd="a" * 96,
                    rtmr0="d" * 96,
                    rtmr1="e" * 96,
                    rtmr2="f" * 96,
                    rtmr3="0" * 96,
                    user_data="test",
                    parsed_at=datetime.now(timezone.utc),
                    status="UpToDate",
                    advisory_ids=[],
                    td_attributes="0000001000000000",
                )

                def mock_refresh(obj):
                    obj.attestation_id = "runtime-attest-123"
                    obj.verified_at = datetime.now(timezone.utc)

                mock_db_session.refresh.side_effect = mock_refresh

                result = await process_runtime_attestation(
                    mock_db_session,
                    server_id,
                    TEST_SERVER_IP,
                    args,
                    miner_hotkey,
                    TEST_NONCE,
                    TEST_CERT_HASH,
                )

                assert result["status"] == "verified"
                assert result["attestation_id"] == "runtime-attest-123"


@pytest.mark.asyncio
async def test_server_lifecycle_flow(mock_db_session, sample_server, server_args):
    """Test complete server lifecycle: register -> check ownership -> delete."""
    miner_hotkey = "5FTestHotkey123"

    with patch("api.server.service._track_server", return_value=sample_server):
        with patch("api.server.service._track_nodes", new_callable=AsyncMock):
            with patch(
                "api.server.service.verify_server",
                new_callable=AsyncMock,
                return_value="1.0.0",
            ):
                await register_server(mock_db_session, server_args, miner_hotkey)
    mock_db_session.commit.assert_called()

    # Step 2: Check ownership
    mock_ownership_result = Mock()
    mock_ownership_result.scalar_one_or_none.return_value = sample_server
    mock_db_session.execute.return_value = mock_ownership_result

    owned_server = await check_server_ownership(mock_db_session, "test-server-123", miner_hotkey)
    assert owned_server == sample_server

    # Step 3: Delete server
    with patch("api.server.service.check_server_ownership", return_value=sample_server):
        deleted = await delete_server(mock_db_session, "test-server-123", miner_hotkey)
        assert deleted is True


# Error Recovery Tests


@pytest.mark.asyncio
async def test_boot_attestation_partial_failure_recovery(
    mock_db_session, boot_attestation_args, sample_boot_quote
):
    """Test boot attestation handles partial failures gracefully."""
    # Simulate verification failure but ensure failed record is still created
    with patch("api.server.service.BootTdxQuote.from_base64", return_value=sample_boot_quote):
        with patch(
            "api.server.service.verify_quote",
            side_effect=MeasurementMismatchError("MRTD mismatch"),
        ):
            with pytest.raises(MeasurementMismatchError):
                await process_boot_attestation(
                    mock_db_session,
                    TEST_SERVER_IP,
                    boot_attestation_args,
                    TEST_NONCE,
                    TEST_CERT_HASH,
                )

            # Should still create failed attestation record
            mock_db_session.add.assert_called_once()
            mock_db_session.commit.assert_called_once()

            # Verify the failed record has correct fields
            call_args = mock_db_session.add.call_args[0][0]
            assert isinstance(call_args, BootAttestation)
            assert call_args.verification_error == "MRTD mismatch"


@pytest.mark.asyncio
async def test_runtime_attestation_partial_failure_recovery(
    mock_db_session, runtime_attestation_args, sample_runtime_quote, sample_server
):
    """Test runtime attestation handles partial failures gracefully."""
    server_id = "test-server-123"
    miner_hotkey = "5FTestHotkey123"

    with patch("api.server.service.check_server_ownership", return_value=sample_server):
        with patch(
            "api.server.service.RuntimeTdxQuote.from_base64",
            return_value=sample_runtime_quote,
        ):
            with patch(
                "api.server.service.verify_quote",
                side_effect=InvalidQuoteError("Invalid quote"),
            ):
                with pytest.raises(InvalidQuoteError):
                    await process_runtime_attestation(
                        mock_db_session,
                        server_id,
                        TEST_SERVER_IP,
                        runtime_attestation_args,
                        miner_hotkey,
                        TEST_NONCE,
                        TEST_CERT_HASH,
                    )

                # Should still create failed attestation record
                mock_db_session.add.assert_called_once()
                mock_db_session.commit.assert_called_once()

                # Verify the failed record has correct fields
                call_args = mock_db_session.add.call_args[0][0]
                assert isinstance(call_args, ServerAttestation)
                assert call_args.verification_error == "Invalid quote"


# ---------------------------------------------------------------------------
# VM Auth Key Tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_boot_attestation_returns_vm_auth_ss58(
    mock_db_session,
    boot_attestation_args,
    mock_quote_parsing,
    mock_verify_quote_signature,
    mock_verify_measurements,
    mock_validate_nonce,
):
    """Successful boot attestation returns vm_auth_ss58 in the 3-tuple response."""
    mock_keypair = Mock()
    mock_keypair.ss58_address = "5EphemeralSS58AddressHere"

    with patch("api.server.service.verify_quote") as mock_verify:
        mock_verify.return_value = TdxVerificationResult(
            mrtd="a" * 96,
            rtmr0="b" * 96,
            rtmr1="c" * 96,
            rtmr2="d" * 96,
            rtmr3="e" * 96,
            user_data="test",
            parsed_at=datetime.now(timezone.utc),
            status="UpToDate",
            advisory_ids=[],
            td_attributes="0000001000000000",
        )

        def mock_refresh(obj):
            obj.attestation_id = "boot-attest-123"
            obj.verified_at = datetime.now(timezone.utc)

        mock_db_session.refresh.side_effect = mock_refresh

        with (
            patch(
                "api.server.service.generate_and_store_boot_token",
                return_value="token-abc",
            ),
            patch("api.server.service._handle_boot_version_update", new_callable=AsyncMock),
            patch(
                "api.server.service._generate_and_store_vm_auth_key",
                new_callable=AsyncMock,
                return_value=mock_keypair,
            ),
        ):
            result = await process_boot_attestation(
                mock_db_session,
                TEST_SERVER_IP,
                boot_attestation_args,
                TEST_NONCE,
                TEST_CERT_HASH,
            )

    boot_token, luks_quote_nonce, vm_auth_ss58 = result
    assert boot_token == "token-abc"
    assert luks_quote_nonce is None
    assert vm_auth_ss58 == "5EphemeralSS58AddressHere"


@pytest.mark.asyncio
async def test_boot_attestation_vm_auth_key_not_generated_on_failure(
    mock_db_session, boot_attestation_args, sample_boot_quote
):
    """vm_auth_key is NOT generated when boot attestation verification fails."""
    with (
        patch(
            "api.server.service.BootTdxQuote.from_base64",
            return_value=sample_boot_quote,
        ),
        patch(
            "api.server.service.verify_quote",
            side_effect=MeasurementMismatchError("MRTD mismatch"),
        ),
        patch(
            "api.server.service._generate_and_store_vm_auth_key",
            new_callable=AsyncMock,
        ) as mock_gen_key,
    ):
        with pytest.raises(MeasurementMismatchError):
            await process_boot_attestation(
                mock_db_session,
                TEST_SERVER_IP,
                boot_attestation_args,
                TEST_NONCE,
                TEST_CERT_HASH,
            )

    mock_gen_key.assert_not_called()


@pytest.mark.asyncio
async def test_tee_server_client_uses_per_vm_keypair(sample_server):
    """TeeServerClient.create() reconstructs per-VM keypair from DB on every call."""
    from api.server.client import TeeServerClient
    import secrets as _secrets
    from bittensor_wallet.keypair import Keypair

    seed_hex = "0x" + _secrets.token_hex(32)
    real_keypair = Keypair.create_from_seed(seed_hex)

    vm_auth_key = VmAuthKey(
        miner_hotkey=sample_server.miner_hotkey,
        vm_name=sample_server.name,
        auth_seed="encrypted_seed_placeholder",
    )
    mock_result = Mock()
    mock_result.scalar_one_or_none.return_value = vm_auth_key
    mock_db = AsyncMock(spec=AsyncSession)
    mock_db.execute.return_value = mock_result

    with patch("api.server.client.decrypt_passphrase", return_value=seed_hex):
        client = await TeeServerClient.create(mock_db, sample_server)

    assert client._keypair.ss58_address == real_keypair.ss58_address
    mock_db.execute.assert_called_once()

    # Signing uses the per-VM SS58 address as the hotkey header
    headers, _ = client._sign_request(purpose="attest")
    assert headers["X-Chutes-Hotkey"] == real_keypair.ss58_address


@pytest.mark.asyncio
async def test_tee_server_client_falls_back_to_validator_keypair(sample_server):
    """TeeServerClient.create() falls back to validator keypair when no per-VM key in DB."""
    from api.server.client import TeeServerClient

    mock_validator_keypair = Mock()
    mock_validator_keypair.ss58_address = "5ValidatorSS58"
    mock_validator_keypair.sign = Mock(return_value=b"\x00" * 64)

    mock_result = Mock()
    mock_result.scalar_one_or_none.return_value = None  # no vm_auth_key in DB
    mock_db = AsyncMock(spec=AsyncSession)
    mock_db.execute.return_value = mock_result

    with patch("api.server.client.settings") as mock_settings:
        mock_settings.validator_keypair = mock_validator_keypair
        client = await TeeServerClient.create(mock_db, sample_server)

    assert client._keypair is mock_validator_keypair
    mock_db.execute.assert_called_once()


@pytest.mark.asyncio
async def test_tee_server_client_always_reads_db(sample_server):
    """TeeServerClient.create() always reads from DB (no in-process cache)."""
    from api.server.client import TeeServerClient
    import secrets as _secrets

    seed_hex = "0x" + _secrets.token_hex(32)
    vm_auth_key = VmAuthKey(
        miner_hotkey=sample_server.miner_hotkey,
        vm_name=sample_server.name,
        auth_seed="encrypted_seed_placeholder",
    )
    mock_result = Mock()
    mock_result.scalar_one_or_none.return_value = vm_auth_key
    mock_db = AsyncMock(spec=AsyncSession)
    mock_db.execute.return_value = mock_result

    with patch("api.server.client.decrypt_passphrase", return_value=seed_hex):
        await TeeServerClient.create(mock_db, sample_server)
        await TeeServerClient.create(mock_db, sample_server)

    # DB is queried on every call -- no caching
    assert mock_db.execute.call_count == 2


# Performance and Concurrency Tests


@pytest.mark.asyncio
async def test_multiple_nonce_operations_concurrent(mock_settings):
    """Test concurrent nonce operations don't interfere."""
    # Override the generate_nonce mock to return unique values for each call
    with patch("api.server.service.generate_nonce", side_effect=lambda: secrets.token_hex(16)):
        # Create multiple nonces concurrently
        import asyncio

        tasks = [create_nonce(TEST_SERVER_IP, NoncePurpose.BOOT) for _ in range(5)]
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
async def test_verify_quote_boot_vs_runtime_different_settings(mock_settings):
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
        report_data=None,
        user_data="test",
        platform_id="0" * 32,
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
        rtmr2="h" * 96,
        rtmr3="i" * 96,
        report_data=None,
        user_data="test",
        platform_id="0" * 32,
        raw_quote_size=4096,
        parsed_at=datetime.now(timezone.utc).isoformat(),
        raw_bytes=b"runtime",
    )

    mock_settings.tee_measurements = [
        TeeMeasurementConfig(
            version="1",
            mrtd="a" * 96,
            name="test",
            boot_rtmrs={
                "RTMR0": "boot_specific_rtmr0",
                "RTMR1": "boot_specific_rtmr1",
                "RTMR2": "d" * 96,
                "RTMR3": "e" * 96,
            },
            runtime_rtmrs={
                "RTMR0": "runtime_specific_rtmr0",
                "RTMR1": "runtime_specific_rtmr1",
                "RTMR2": "h" * 96,
                "RTMR3": "i" * 96,
            },
            expected_gpus=[],
            gpu_count=None,
        ),
    ]

    # DCAP result must match each quote for verify_result(); return matching result per call
    boot_dcap_result = TdxVerificationResult(
        mrtd="a" * 96,
        rtmr0="boot_specific_rtmr0",
        rtmr1="boot_specific_rtmr1",
        rtmr2="d" * 96,
        rtmr3="e" * 96,
        user_data="test",
        parsed_at=datetime.now(timezone.utc),
        status="UpToDate",
        advisory_ids=[],
        td_attributes="0000001000000000",
    )
    runtime_dcap_result = TdxVerificationResult(
        mrtd="a" * 96,
        rtmr0="runtime_specific_rtmr0",
        rtmr1="runtime_specific_rtmr1",
        rtmr2="h" * 96,
        rtmr3="i" * 96,
        user_data="test",
        parsed_at=datetime.now(timezone.utc),
        status="UpToDate",
        advisory_ids=[],
        td_attributes="0000001000000000",
    )

    with patch("api.server.util.verify_quote_signature") as mock_sig:
        mock_sig.side_effect = [boot_dcap_result, runtime_dcap_result]
        await verify_quote(boot_quote, TEST_NONCE, TEST_CERT_HASH)
        await verify_quote(runtime_quote, TEST_NONCE, TEST_CERT_HASH)

    assert mock_sig.call_count == 2


# Special Edge Cases


@pytest.mark.asyncio
async def test_get_server_attestation_status_failed_attestation(mock_db_session, sample_server):
    """Test getting server attestation status with failed attestation."""
    server_id = "test-server-123"
    miner_hotkey = "5FTestHotkey123"

    # Create failed attestation (verified inferred from verification_error is None)
    failed_attestation = ServerAttestation(
        attestation_id="failed-attest-123",
        server_id=server_id,
        quote_data=None,
        verification_error="Measurement mismatch",
        measurement_version=None,
        created_at=datetime.now(timezone.utc),
        verified_at=None,
    )

    with patch("api.server.service.check_server_ownership", return_value=sample_server):
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
async def test_boot_attestation_database_rollback_on_error(
    mock_db_session, boot_attestation_args, sample_boot_quote
):
    """Test that database operations are rolled back on errors."""
    with patch("api.server.service.BootTdxQuote.from_base64", return_value=sample_boot_quote):
        with patch("api.server.service.verify_quote") as mock_verify:
            mock_verify.return_value = TdxVerificationResult(
                mrtd="a" * 96,
                rtmr0="b" * 96,
                rtmr1="c" * 96,
                rtmr2="d" * 96,
                rtmr3="e" * 96,
                user_data="test",
                parsed_at=datetime.now(timezone.utc),
                status="UpToDate",
                advisory_ids=[],
                td_attributes="0000001000000000",
            )

            # Mock commit to fail after add
            mock_db_session.commit.side_effect = Exception("Database connection lost")

            with pytest.raises(Exception, match="Database connection lost"):
                await process_boot_attestation(
                    mock_db_session,
                    TEST_SERVER_IP,
                    boot_attestation_args,
                    TEST_NONCE,
                    TEST_CERT_HASH,
                )

            # Verify add was called but rollback should not be called
            # (since we're not explicitly handling this exception)
            mock_db_session.add.assert_called_once()
            mock_db_session.commit.assert_called_once()


@pytest.mark.asyncio
async def test_runtime_attestation_database_rollback_on_error(
    mock_db_session, runtime_attestation_args, sample_runtime_quote, sample_server
):
    """Test that runtime attestation database operations handle errors."""
    server_id = "test-server-123"
    miner_hotkey = "5FTestHotkey123"

    with patch("api.server.service.check_server_ownership", return_value=sample_server):
        with patch(
            "api.server.service.RuntimeTdxQuote.from_base64",
            return_value=sample_runtime_quote,
        ):
            with patch("api.server.service.verify_quote") as mock_verify:
                mock_verify.return_value = TdxVerificationResult(
                    mrtd="a" * 96,
                    rtmr0="d" * 96,
                    rtmr1="e" * 96,
                    rtmr2="f" * 96,
                    rtmr3="0" * 96,
                    user_data="test",
                    parsed_at=datetime.now(timezone.utc),
                    status="UpToDate",
                    advisory_ids=[],
                    td_attributes="0000001000000000",
                )

                # Mock refresh to fail
                mock_db_session.refresh.side_effect = Exception("Database error during refresh")

                with pytest.raises(Exception, match="Database error during refresh"):
                    await process_runtime_attestation(
                        mock_db_session,
                        server_id,
                        TEST_SERVER_IP,
                        runtime_attestation_args,
                        miner_hotkey,
                        TEST_NONCE,
                        TEST_CERT_HASH,
                    )

                mock_db_session.add.assert_called_once()
                mock_db_session.commit.assert_called_once()


# Comprehensive Quote Validation Tests


@pytest.mark.asyncio
async def test_verify_quote_with_different_quote_types(mock_verify_measurements):
    """Test quote verification with different quote implementations."""
    boot_result = TdxVerificationResult(
        mrtd="a" * 96,
        rtmr0="b" * 96,
        rtmr1="c" * 96,
        rtmr2="d" * 96,
        rtmr3="e" * 96,
        user_data="test",
        parsed_at=datetime.now(timezone.utc),
        status="UpToDate",
        advisory_ids=[],
        td_attributes="0000001000000000",
    )
    runtime_result = TdxVerificationResult(
        mrtd="a" * 96,
        rtmr0="d" * 96,
        rtmr1="e" * 96,
        rtmr2="f" * 96,
        rtmr3="0" * 96,
        user_data="test",
        parsed_at=datetime.now(timezone.utc),
        status="UpToDate",
        advisory_ids=[],
        td_attributes="0000001000000000",
    )

    # Test with BootTdxQuote
    boot_quote = BootTdxQuote(
        version=4,
        att_key_type=2,
        tee_type=0x81,
        mrtd="a" * 96,
        rtmr0="b" * 96,
        rtmr1="c" * 96,
        rtmr2="d" * 96,
        rtmr3="e" * 96,
        report_data=None,
        user_data="test",
        platform_id="0" * 32,
        raw_quote_size=4096,
        parsed_at=datetime.now(timezone.utc).isoformat(),
        raw_bytes=b"boot",
    )

    # Test with RuntimeTdxQuote
    runtime_quote = RuntimeTdxQuote(
        version=4,
        att_key_type=2,
        tee_type=0x81,
        mrtd="a" * 96,
        rtmr0="d" * 96,
        rtmr1="e" * 96,
        rtmr2="f" * 96,
        rtmr3="0" * 96,
        report_data=None,
        user_data="test",
        platform_id="0" * 32,
        raw_quote_size=4096,
        parsed_at=datetime.now(timezone.utc).isoformat(),
        raw_bytes=b"runtime",
    )

    with patch("api.server.util.verify_quote_signature") as mock_sig:
        mock_sig.side_effect = [boot_result, runtime_result]
        boot_verify_result = await verify_quote(boot_quote, TEST_NONCE, TEST_CERT_HASH)
        runtime_verify_result = await verify_quote(runtime_quote, TEST_NONCE, TEST_CERT_HASH)

    assert isinstance(boot_verify_result, TdxVerificationResult)
    assert isinstance(runtime_verify_result, TdxVerificationResult)
    assert mock_sig.call_count == 2
    assert mock_verify_measurements.call_count == 2


# ---------------------------------------------------------------------------
# Root Passphrase Rotation Tests
# ---------------------------------------------------------------------------


def _make_vm_config(passphrases: dict) -> VmCacheConfig:
    """Build a minimal VmCacheConfig with the given volume_passphrases dict."""
    cfg = VmCacheConfig(
        miner_hotkey="5FTestHotkey123",
        vm_name="test-vm",
        volume_passphrases=passphrases,
    )
    return cfg


@pytest.mark.asyncio
async def test_get_default_root_passphrase_from_db(mock_db_session):
    """Returns decrypted passphrase when a root_passphrase_defaults row exists."""
    from api.server.util import encrypt_passphrase

    encrypted = encrypt_passphrase("version-specific-pass")
    row = RootPassphraseDefault(image_version="1.4.0", encrypted_passphrase=encrypted)
    mock_db_session.get = AsyncMock(return_value=row)

    result = await get_default_root_passphrase(mock_db_session, "1.4.0")
    assert result == "version-specific-pass"


@pytest.mark.asyncio
async def test_get_default_root_passphrase_fallback_to_settings(mock_db_session, mock_settings):
    """Falls back to settings.luks_passphrase when no DB row exists."""
    mock_db_session.get = AsyncMock(return_value=None)
    mock_settings.luks_passphrase = "global-default-pass"

    result = await get_default_root_passphrase(mock_db_session, "1.4.0")
    assert result == "global-default-pass"


@pytest.mark.asyncio
async def test_get_default_root_passphrase_no_version_fallback(mock_db_session, mock_settings):
    """Falls back to settings.luks_passphrase when image_version is None."""
    mock_settings.luks_passphrase = "global-default-pass"

    result = await get_default_root_passphrase(mock_db_session, None)
    assert result == "global-default-pass"
    mock_db_session.get.assert_not_called()


@pytest.mark.asyncio
async def test_get_root_passphrase_for_boot_first_boot_no_prior_state(
    mock_db_session, mock_settings
):
    """first_boot=True with no existing root key: returns default passphrase + rotation fields."""
    mock_settings.luks_passphrase = "build-time-default"
    vm_config = _make_vm_config({})  # no root key stored yet
    mock_db_session.get = AsyncMock(return_value=None)  # no DB default either

    with (
        patch("api.server.util._get_vm_cache_config", AsyncMock(return_value=vm_config)),
        patch(
            "api.server.util.generate_confirm_nonce",
            AsyncMock(return_value="root-nonce-123"),
        ),
    ):
        key, root_next, root_confirm_nonce = await get_root_passphrase_for_boot(
            mock_db_session,
            "5FTestHotkey123",
            "test-vm",
            first_boot=True,
            measurement_version="1.4.0",
        )

    assert key == "build-time-default"
    assert (
        root_next is not None and len(root_next) == 128
    )  # generate_cache_passphrase is 128 hex chars
    assert root_confirm_nonce == "root-nonce-123"
    # pending_root should be written to vm_config
    assert "pending_root" in vm_config.volume_passphrases


@pytest.mark.asyncio
async def test_get_root_passphrase_for_boot_first_boot_with_prior_root(
    mock_db_session, mock_settings
):
    """first_boot=True with an existing root key: clears stored state, returns default."""
    from api.server.util import encrypt_passphrase

    old_encrypted = encrypt_passphrase("old-rotated-pass")
    vm_config = _make_vm_config({"root": old_encrypted})
    mock_settings.luks_passphrase = "build-time-default"
    mock_db_session.get = AsyncMock(return_value=None)

    with (
        patch("api.server.util._get_vm_cache_config", AsyncMock(return_value=vm_config)),
        patch(
            "api.server.util.generate_confirm_nonce",
            AsyncMock(return_value="nonce-abc"),
        ),
    ):
        key, root_next, root_confirm_nonce = await get_root_passphrase_for_boot(
            mock_db_session,
            "5FTestHotkey123",
            "test-vm",
            first_boot=True,
            measurement_version="1.4.0",
        )

    assert "root" not in vm_config.volume_passphrases
    assert key == "build-time-default"
    assert root_next is not None
    assert root_confirm_nonce == "nonce-abc"


@pytest.mark.asyncio
async def test_get_root_passphrase_for_boot_normal_boot_stored_root(mock_db_session, mock_settings):
    """first_boot=False with a stored root key: returns stored key + rotation."""
    from api.server.util import encrypt_passphrase

    stored_pass = "current-rotated-pass"
    vm_config = _make_vm_config({"root": encrypt_passphrase(stored_pass)})
    mock_settings.luks_passphrase = "build-time-default"

    with (
        patch("api.server.util._get_vm_cache_config", AsyncMock(return_value=vm_config)),
        patch(
            "api.server.util.generate_confirm_nonce",
            AsyncMock(return_value="nonce-xyz"),
        ),
    ):
        key, root_next, root_confirm_nonce = await get_root_passphrase_for_boot(
            mock_db_session,
            "5FTestHotkey123",
            "test-vm",
            first_boot=False,
            measurement_version="1.4.0",
        )

    assert key == stored_pass
    assert root_next is not None
    assert root_confirm_nonce == "nonce-xyz"
    assert "pending_root" in vm_config.volume_passphrases


@pytest.mark.asyncio
async def test_get_root_passphrase_for_boot_normal_boot_no_stored_root(
    mock_db_session, mock_settings
):
    """first_boot=False with no stored root key: falls back to version default + rotation."""
    vm_config = _make_vm_config({})
    mock_settings.luks_passphrase = "build-time-default"
    mock_db_session.get = AsyncMock(return_value=None)

    with (
        patch("api.server.util._get_vm_cache_config", AsyncMock(return_value=vm_config)),
        patch(
            "api.server.util.generate_confirm_nonce",
            AsyncMock(return_value="nonce-new"),
        ),
    ):
        key, root_next, root_confirm_nonce = await get_root_passphrase_for_boot(
            mock_db_session,
            "5FTestHotkey123",
            "test-vm",
            first_boot=False,
            measurement_version="1.4.0",
        )

    assert key == "build-time-default"
    assert root_next is not None
    assert root_confirm_nonce == "nonce-new"


@pytest.mark.asyncio
async def test_get_root_passphrase_for_boot_pre_rotation_version(mock_db_session, mock_settings):
    """VMs below 1.4.0 get no root_next or root_confirm_nonce."""
    vm_config = _make_vm_config({})
    mock_settings.luks_passphrase = "build-time-default"
    mock_db_session.get = AsyncMock(return_value=None)

    with patch("api.server.util._get_vm_cache_config", AsyncMock(return_value=vm_config)):
        key, root_next, root_confirm_nonce = await get_root_passphrase_for_boot(
            mock_db_session,
            "5FTestHotkey123",
            "test-vm",
            first_boot=False,
            measurement_version="1.3.0",
        )

    assert key == "build-time-default"
    assert root_next is None
    assert root_confirm_nonce is None
    assert "pending_root" not in (vm_config.volume_passphrases or {})


@pytest.mark.asyncio
async def test_get_root_passphrase_for_boot_discards_stale_pending(mock_db_session, mock_settings):
    """Stale pending_root from a prior unconfirmed rotation is discarded and replaced."""
    from api.server.util import encrypt_passphrase

    stale_enc = encrypt_passphrase("stale-pending")
    current_enc = encrypt_passphrase("current-root")
    vm_config = _make_vm_config({"root": current_enc, "pending_root": stale_enc})
    mock_settings.luks_passphrase = "build-time-default"

    with (
        patch("api.server.util._get_vm_cache_config", AsyncMock(return_value=vm_config)),
        patch(
            "api.server.util.generate_confirm_nonce",
            AsyncMock(return_value="nonce-fresh"),
        ),
    ):
        key, root_next, root_confirm_nonce = await get_root_passphrase_for_boot(
            mock_db_session,
            "5FTestHotkey123",
            "test-vm",
            first_boot=False,
            measurement_version="1.4.0",
        )

    assert key == "current-root"
    # The new pending_root should differ from the stale one
    from api.server.util import decrypt_passphrase

    new_pending_enc = vm_config.volume_passphrases.get("pending_root")
    assert new_pending_enc is not None
    assert decrypt_passphrase(new_pending_enc) != "stale-pending"
    assert root_next is not None


@pytest.mark.asyncio
async def test_process_boot_attestation_returns_root_rotation_fields(
    mock_db_session,
    boot_attestation_args,
    mock_quote_parsing,
    mock_verify_quote_signature,
    mock_verify_measurements,
    mock_validate_nonce,
):
    """process_boot_attestation wires root rotation fields through to its return value."""
    with patch("api.server.service.verify_quote") as mock_verify:
        mock_verify.return_value = TdxVerificationResult(
            mrtd="a" * 96,
            rtmr0="b" * 96,
            rtmr1="c" * 96,
            rtmr2="d" * 96,
            rtmr3="e" * 96,
            user_data="test",
            parsed_at=datetime.now(timezone.utc),
            status="UpToDate",
            advisory_ids=[],
            td_attributes="0000001000000000",
        )

        def mock_refresh(obj):
            obj.attestation_id = "boot-attest-123"
            obj.verified_at = datetime.now(timezone.utc)

        mock_db_session.refresh.side_effect = mock_refresh

        with (
            patch("api.server.service.generate_and_store_boot_token", return_value="bt"),
            patch("api.server.service._handle_boot_version_update", new_callable=AsyncMock),
            patch(
                "api.server.service.get_root_passphrase_for_boot",
                new_callable=AsyncMock,
                return_value=("root-key", "next-pass", "confirm-nonce"),
            ) as mock_root,
        ):
            result = await process_boot_attestation(
                mock_db_session,
                TEST_SERVER_IP,
                boot_attestation_args,
                TEST_NONCE,
                TEST_CERT_HASH,
            )

    assert isinstance(result, BootAttestationResult)
    assert result.root_key == "root-key"
    assert result.root_next == "next-pass"
    assert result.root_confirm_nonce == "confirm-nonce"
    mock_root.assert_called_once()


@pytest.mark.asyncio
async def test_confirm_luks_rotation_promotes_pending_root(mock_db_session):
    """process_luks_confirm promotes pending_root to root when rotated=True."""
    from api.server.service import process_luks_confirm
    from api.server.schemas import LuksConfirmRequest, LuksVolumeConfirmStatus
    from api.server.util import encrypt_passphrase

    encrypted_pending = encrypt_passphrase("new-root-pass")
    vm_config = _make_vm_config({"pending_root": encrypted_pending})

    with patch("api.server.service._get_vm_cache_config", AsyncMock(return_value=vm_config)):
        body = LuksConfirmRequest(volumes={"root": LuksVolumeConfirmStatus(rotated=True)})
        result = await process_luks_confirm(mock_db_session, "5FTestHotkey123", "test-vm", body)

    assert result.volumes["root"]["result"] == "promoted"
    assert "root" in vm_config.volume_passphrases
    assert "pending_root" not in vm_config.volume_passphrases
    # The promoted value should be the new passphrase
    from api.server.util import decrypt_passphrase

    assert decrypt_passphrase(vm_config.volume_passphrases["root"]) == "new-root-pass"


@pytest.mark.asyncio
async def test_confirm_luks_rotation_discards_pending_root_on_failure(mock_db_session):
    """process_luks_confirm discards pending_root when rotated=False."""
    from api.server.service import process_luks_confirm
    from api.server.schemas import LuksConfirmRequest, LuksVolumeConfirmStatus
    from api.server.util import encrypt_passphrase

    old_encrypted = encrypt_passphrase("current-root-pass")
    pending_encrypted = encrypt_passphrase("new-root-pass")
    vm_config = _make_vm_config({"root": old_encrypted, "pending_root": pending_encrypted})

    with patch("api.server.service._get_vm_cache_config", AsyncMock(return_value=vm_config)):
        body = LuksConfirmRequest(volumes={"root": LuksVolumeConfirmStatus(rotated=False)})
        result = await process_luks_confirm(mock_db_session, "5FTestHotkey123", "test-vm", body)

    assert result.volumes["root"]["result"] == "discarded"
    assert "pending_root" not in vm_config.volume_passphrases
    # Current root survives
    assert "root" in vm_config.volume_passphrases
