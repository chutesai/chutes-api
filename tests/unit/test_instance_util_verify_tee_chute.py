"""
Unit tests for verify_tee_chute in api/instance/util.py.
Tests chute attestation flow with e2e_pubkey hash for chutes >= 0.6.0.
"""

import hashlib
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from fastapi import HTTPException

from api.instance.util import verify_tee_chute
from api.server.quote import BootTdxQuote
from tests.fixtures.gpus import TEST_GPU_NONCE

EXPECTED_NONCE = TEST_GPU_NONCE
E2E_PUBKEY = "dGVzdF9lMmVfcHVia2V5"  # base64-like test value
EXPECTED_CERT_HASH = "a" * 64


def _make_instance(chutes_version: str | None, extra: dict | None = None):
    """Create a mock Instance with host, chutes_version, extra."""
    instance = MagicMock()
    instance.host = "192.168.1.1"
    instance.chutes_version = chutes_version
    instance.extra = extra
    return instance


def _make_launch_config():
    """Create a mock LaunchConfig."""
    launch_config = MagicMock()
    launch_config.miner_hotkey = "miner_hotkey_123"
    return launch_config


def _make_server():
    """Create a mock Server."""
    server = MagicMock()
    server.ip = "192.168.1.1"
    server.miner_hotkey = "miner_hotkey_123"
    return server


@pytest.fixture
def mock_db():
    """Mock database session that returns a server for the query."""
    db = AsyncMock()
    result = MagicMock()
    result.scalar_one_or_none.return_value = _make_server()
    db.execute = AsyncMock(return_value=result)
    return db


@pytest.fixture
def sample_quote():
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
        report_data=EXPECTED_NONCE + "0" * 64,
        user_data="test",
        platform_id="0" * 32,
        raw_quote_size=4096,
        parsed_at="2024-01-01T00:00:00Z",
        raw_bytes=b"dummy",
    )


@pytest.fixture
def mock_cert():
    """Mock x509 certificate with get_public_key_hash returning expected hash."""
    cert = MagicMock()
    return cert


@pytest.mark.asyncio
async def test_verify_tee_chute_chutes_060_uses_e2e_pubkey_hash(mock_db, sample_quote, mock_cert):
    """For chutes >= 0.6.0 with e2e_pubkey, verify_quote receives sha256(nonce+e2e_pubkey)."""
    instance = _make_instance("0.6.0", {"e2e_pubkey": E2E_PUBKEY})
    launch_config = _make_launch_config()

    expected_report_data = (
        hashlib.sha256((EXPECTED_NONCE + E2E_PUBKEY).encode()).hexdigest().lower()
    )

    with (
        patch("api.instance.util.TeeServerClient") as mock_client_cls,
        patch("api.instance.util.verify_quote", new_callable=AsyncMock) as mock_verify_quote,
        patch("api.instance.util.verify_gpu_evidence", new_callable=AsyncMock) as mock_verify_gpu,
        patch("api.instance.util.get_public_key_hash", return_value=EXPECTED_CERT_HASH),
    ):
        mock_client = MagicMock()
        mock_client.get_chute_evidence = AsyncMock(return_value=(sample_quote, [], mock_cert))
        mock_client_cls.return_value = mock_client

        await verify_tee_chute(mock_db, instance, launch_config, "deploy-123", EXPECTED_NONCE)

        mock_verify_quote.assert_called_once_with(
            sample_quote, expected_report_data, EXPECTED_CERT_HASH
        )
        mock_verify_gpu.assert_called_once_with([], expected_report_data)


@pytest.mark.asyncio
async def test_verify_tee_chute_chutes_059_uses_raw_nonce(mock_db, sample_quote, mock_cert):
    """For chutes < 0.6.0, verify_quote receives expected_nonce directly (old behavior)."""
    instance = _make_instance("0.5.9", {"e2e_pubkey": E2E_PUBKEY})
    launch_config = _make_launch_config()

    with (
        patch("api.instance.util.TeeServerClient") as mock_client_cls,
        patch("api.instance.util.verify_quote", new_callable=AsyncMock) as mock_verify_quote,
        patch("api.instance.util.verify_gpu_evidence", new_callable=AsyncMock) as mock_verify_gpu,
        patch("api.instance.util.get_public_key_hash", return_value=EXPECTED_CERT_HASH),
    ):
        mock_client = MagicMock()
        mock_client.get_chute_evidence = AsyncMock(return_value=(sample_quote, [], mock_cert))
        mock_client_cls.return_value = mock_client

        await verify_tee_chute(mock_db, instance, launch_config, "deploy-123", EXPECTED_NONCE)

        mock_verify_quote.assert_called_once_with(sample_quote, EXPECTED_NONCE, EXPECTED_CERT_HASH)
        mock_verify_gpu.assert_called_once_with([], EXPECTED_NONCE)


@pytest.mark.asyncio
async def test_verify_tee_chute_chutes_060_missing_e2e_pubkey_raises_400(
    mock_db, sample_quote, mock_cert
):
    """For chutes >= 0.6.0 without e2e_pubkey, raise HTTP 400."""
    instance = _make_instance("0.6.0", {})  # no e2e_pubkey
    launch_config = _make_launch_config()

    with (
        patch("api.instance.util.TeeServerClient") as mock_client_cls,
        patch("api.instance.util.verify_quote", new_callable=AsyncMock),
        patch("api.instance.util.verify_gpu_evidence", new_callable=AsyncMock),
        patch("api.instance.util.get_public_key_hash", return_value=EXPECTED_CERT_HASH),
    ):
        mock_client = MagicMock()
        mock_client.get_chute_evidence = AsyncMock(return_value=(sample_quote, [], mock_cert))
        mock_client_cls.return_value = mock_client

        with pytest.raises(HTTPException) as exc_info:
            await verify_tee_chute(mock_db, instance, launch_config, "deploy-123", EXPECTED_NONCE)

        assert exc_info.value.status_code == 400
        assert "e2e_pubkey required" in exc_info.value.detail


@pytest.mark.asyncio
async def test_verify_tee_chute_chutes_060_extra_none_raises_400(mock_db, sample_quote, mock_cert):
    """For chutes >= 0.6.0 with instance.extra None, raise HTTP 400."""
    instance = _make_instance("0.6.0", None)
    launch_config = _make_launch_config()

    with (
        patch("api.instance.util.TeeServerClient") as mock_client_cls,
        patch("api.instance.util.verify_quote", new_callable=AsyncMock),
        patch("api.instance.util.verify_gpu_evidence", new_callable=AsyncMock),
        patch("api.instance.util.get_public_key_hash", return_value=EXPECTED_CERT_HASH),
    ):
        mock_client = MagicMock()
        mock_client.get_chute_evidence = AsyncMock(return_value=(sample_quote, [], mock_cert))
        mock_client_cls.return_value = mock_client

        with pytest.raises(HTTPException) as exc_info:
            await verify_tee_chute(mock_db, instance, launch_config, "deploy-123", EXPECTED_NONCE)

        assert exc_info.value.status_code == 400
