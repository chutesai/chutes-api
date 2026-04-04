"""
Unit tests for Phase 4: boot version update hook and registration version population.
"""

import pytest
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

from api.server.schemas import Server, TeeUpgradeWindow
from api.server.exceptions import ServerNotFoundError
from api.server.service import _handle_boot_version_update

TEST_SERVER_ID = "server-abc-123"
TEST_HOTKEY = "5GrwvaEF5zXb26Fz9rcQpDWS57CtERHpNehXCPcNoHGKutQY"
TEST_VM_NAME = "my-tee-vm"
TEST_WINDOW_ID = "window-abc-123"
TEST_VERSION_OLD = "0.2.0"
TEST_VERSION_TARGET = "0.3.1"
TEST_VERSION_ABOVE = "0.4.0"
TEST_WINDOW_START = datetime(2026, 4, 1, tzinfo=timezone.utc)
TEST_WINDOW_END = datetime(2026, 4, 7, tzinfo=timezone.utc)


def _make_server(**overrides):
    defaults = dict(
        server_id=TEST_SERVER_ID,
        ip="10.0.0.1",
        miner_hotkey=TEST_HOTKEY,
        name=TEST_VM_NAME,
        netuid=64,
        is_tee=True,
        version=None,
        maintenance_pending_window_id=None,
    )
    defaults.update(overrides)
    return Server(**defaults)


def _make_window(**overrides):
    defaults = dict(
        id=TEST_WINDOW_ID,
        upgrade_window_start=TEST_WINDOW_START,
        upgrade_window_end=TEST_WINDOW_END,
        target_measurement_version=TEST_VERSION_TARGET,
        max_concurrent_per_miner=1,
        created_at=datetime(2026, 4, 1, tzinfo=timezone.utc),
    )
    defaults.update(overrides)
    return TeeUpgradeWindow(**defaults)


# ---------------------------------------------------------------------------
# _handle_boot_version_update
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@patch("api.server.service.get_server_by_name", new_callable=AsyncMock)
async def test_boot_no_server_row_is_noop(mock_get):
    """First boot before registration: no server found, no error."""
    mock_get.side_effect = ServerNotFoundError("not found")
    db = AsyncMock()
    await _handle_boot_version_update(db, TEST_HOTKEY, TEST_VM_NAME, TEST_VERSION_TARGET)
    db.commit.assert_not_awaited()


@pytest.mark.asyncio
@patch("api.server.service.get_server_by_name", new_callable=AsyncMock)
async def test_boot_updates_version_no_maintenance(mock_get):
    """Server exists, no pending maintenance: version updated, commit called."""
    server = _make_server()
    mock_get.return_value = server
    db = AsyncMock()

    await _handle_boot_version_update(db, TEST_HOTKEY, TEST_VM_NAME, TEST_VERSION_TARGET)

    assert server.version == TEST_VERSION_TARGET
    assert server.maintenance_pending_window_id is None
    db.commit.assert_awaited_once()


@pytest.mark.asyncio
@patch("api.server.service.get_server_by_name", new_callable=AsyncMock)
async def test_boot_meets_target_clears_maintenance(mock_get):
    """Boot version >= target: version updated, maintenance_pending_window_id cleared."""
    server = _make_server(maintenance_pending_window_id=TEST_WINDOW_ID)
    mock_get.return_value = server
    window = _make_window()

    db = AsyncMock()
    db.get = AsyncMock(return_value=window)

    await _handle_boot_version_update(db, TEST_HOTKEY, TEST_VM_NAME, TEST_VERSION_TARGET)

    assert server.version == TEST_VERSION_TARGET
    assert server.maintenance_pending_window_id is None
    db.get.assert_awaited_once_with(TeeUpgradeWindow, TEST_WINDOW_ID)
    db.commit.assert_awaited_once()


@pytest.mark.asyncio
@patch("api.server.service.get_server_by_name", new_callable=AsyncMock)
async def test_boot_above_target_clears_maintenance(mock_get):
    """Boot version > target: still clears maintenance."""
    server = _make_server(maintenance_pending_window_id=TEST_WINDOW_ID)
    mock_get.return_value = server
    window = _make_window()

    db = AsyncMock()
    db.get = AsyncMock(return_value=window)

    await _handle_boot_version_update(db, TEST_HOTKEY, TEST_VM_NAME, TEST_VERSION_ABOVE)

    assert server.version == TEST_VERSION_ABOVE
    assert server.maintenance_pending_window_id is None


@pytest.mark.asyncio
@patch("api.server.service.get_server_by_name", new_callable=AsyncMock)
async def test_boot_below_target_keeps_maintenance(mock_get):
    """Boot version < target: version updated, but maintenance slot stays."""
    server = _make_server(maintenance_pending_window_id=TEST_WINDOW_ID)
    mock_get.return_value = server
    window = _make_window()

    db = AsyncMock()
    db.get = AsyncMock(return_value=window)

    await _handle_boot_version_update(db, TEST_HOTKEY, TEST_VM_NAME, TEST_VERSION_OLD)

    assert server.version == TEST_VERSION_OLD
    assert server.maintenance_pending_window_id == TEST_WINDOW_ID
    db.commit.assert_awaited_once()


@pytest.mark.asyncio
@patch("api.server.service.get_server_by_name", new_callable=AsyncMock)
async def test_boot_stale_window_cleared(mock_get):
    """maintenance_pending_window_id points to missing window: cleared."""
    server = _make_server(maintenance_pending_window_id="missing-window-id")
    mock_get.return_value = server

    db = AsyncMock()
    db.get = AsyncMock(return_value=None)

    await _handle_boot_version_update(db, TEST_HOTKEY, TEST_VM_NAME, TEST_VERSION_TARGET)

    assert server.version == TEST_VERSION_TARGET
    assert server.maintenance_pending_window_id is None
    db.commit.assert_awaited_once()


# ---------------------------------------------------------------------------
# register_server sets version from verify_server return
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@patch("api.server.service._track_nodes", new_callable=AsyncMock)
@patch("api.server.service.verify_server", new_callable=AsyncMock)
@patch("api.server.service._track_server", new_callable=AsyncMock)
@patch(
    "api.server.service.SUPPORTED_GPUS",
    {"gpu-a100": {"processors": 1, "max_threads_per_processor": 1}},
)
async def test_register_server_sets_version(mock_track_server, mock_verify, mock_track_nodes):
    from api.server.service import register_server
    from api.server.schemas import ServerArgs
    from api.node.schemas import NodeArgs

    server = _make_server()
    mock_track_server.return_value = server
    mock_verify.return_value = TEST_VERSION_TARGET

    db = AsyncMock()

    gpu = MagicMock(spec=NodeArgs)
    gpu.gpu_identifier = "gpu-a100"
    args = MagicMock(spec=ServerArgs)
    args.id = TEST_SERVER_ID
    args.name = TEST_VM_NAME
    args.host = "10.0.0.1"
    args.gpus = [gpu]

    await register_server(db, args, TEST_HOTKEY)

    assert server.version == TEST_VERSION_TARGET
    db.commit.assert_awaited()


@pytest.mark.asyncio
@patch("api.server.service._track_nodes", new_callable=AsyncMock)
@patch("api.server.service.verify_server", new_callable=AsyncMock)
@patch("api.server.service._track_server", new_callable=AsyncMock)
@patch(
    "api.server.service.SUPPORTED_GPUS",
    {"gpu-a100": {"processors": 1, "max_threads_per_processor": 1}},
)
async def test_register_server_version_none_when_verify_returns_none(
    mock_track_server, mock_verify, mock_track_nodes
):
    from api.server.service import register_server
    from api.server.schemas import ServerArgs
    from api.node.schemas import NodeArgs

    server = _make_server()
    mock_track_server.return_value = server
    mock_verify.return_value = None

    db = AsyncMock()

    gpu = MagicMock(spec=NodeArgs)
    gpu.gpu_identifier = "gpu-a100"
    args = MagicMock(spec=ServerArgs)
    args.id = TEST_SERVER_ID
    args.name = TEST_VM_NAME
    args.host = "10.0.0.1"
    args.gpus = [gpu]

    await register_server(db, args, TEST_HOTKEY)

    assert server.version is None
