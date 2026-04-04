"""
Unit tests for TEE maintenance route handler functions (Phase 3).

These tests call the route handler functions directly with mocked dependencies,
matching the test style used elsewhere in this project.
"""

import pytest
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

from fastapi import HTTPException

from api.server.schemas import (
    Server,
    TeeUpgradeWindow,
    MaintenanceReason,
    PreflightResult,
    ConfirmMaintenanceResult,
    UpgradeWindowInfo,
    MaintenancePolicyResponse,
)
from api.server.router import (
    get_maintenance_policy,
    get_maintenance_preflight,
    put_confirm_maintenance,
    get_server_details,
)

TEST_SERVER_NAME_OR_ID = "my-tee-vm"

TEST_SERVER_ID = "server-abc-123"
TEST_HOTKEY = "5GrwvaEF5zXb26Fz9rcQpDWS57CtERHpNehXCPcNoHGKutQY"
TEST_VM_NAME = "my-tee-vm"
TEST_WINDOW_ID = "window-abc-123"
TEST_VERSION_OLD = "0.2.0"
TEST_VERSION_TARGET = "0.3.1"
TEST_WINDOW_START = datetime(2026, 4, 1, tzinfo=timezone.utc)
TEST_WINDOW_END = datetime(2026, 4, 7, tzinfo=timezone.utc)
DEFAULT_CONCURRENCY_LIMIT = 1


def _make_window(**overrides):
    defaults = dict(
        id=TEST_WINDOW_ID,
        upgrade_window_start=TEST_WINDOW_START,
        upgrade_window_end=TEST_WINDOW_END,
        target_measurement_version=TEST_VERSION_TARGET,
        max_concurrent_per_miner=DEFAULT_CONCURRENCY_LIMIT,
        created_at=datetime(2026, 4, 1, tzinfo=timezone.utc),
    )
    defaults.update(overrides)
    return TeeUpgradeWindow(**defaults)


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


def _mock_scalars_result(rows):
    scalars_mock = MagicMock()
    scalars_mock.all.return_value = rows
    result_mock = MagicMock()
    result_mock.scalars.return_value = scalars_mock
    return result_mock


# ---------------------------------------------------------------------------
# GET /servers/maintenance/policy
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@patch("api.server.router._count_active_maintenance_slots", new_callable=AsyncMock, return_value=0)
@patch("api.server.router.get_active_upgrade_window", new_callable=AsyncMock)
async def test_policy_returns_active_window(mock_get_window, _mock_slots):
    window = _make_window()
    mock_get_window.return_value = window

    db = AsyncMock()
    db.execute.return_value = _mock_scalars_result([])

    result = await get_maintenance_policy(db=db, hotkey=TEST_HOTKEY, _=None)
    assert isinstance(result, MaintenancePolicyResponse)
    assert result.active_window is not None
    assert result.active_window.id == TEST_WINDOW_ID
    assert result.active_window.max_concurrent_per_miner == DEFAULT_CONCURRENCY_LIMIT


@pytest.mark.asyncio
@patch("api.server.router.get_active_upgrade_window", new_callable=AsyncMock, return_value=None)
async def test_policy_returns_null_when_no_window(mock_get_window):
    db = AsyncMock()

    result = await get_maintenance_policy(db=db, hotkey=TEST_HOTKEY, _=None)
    assert result.active_window is None
    assert result.current_slots == 0
    assert result.pending_servers == []


@pytest.mark.asyncio
async def test_policy_rejects_missing_hotkey():
    db = AsyncMock()
    with pytest.raises(HTTPException) as exc_info:
        await get_maintenance_policy(db=db, hotkey=None, _=None)
    assert exc_info.value.status_code == 403


@pytest.mark.asyncio
@patch("api.server.router._count_active_maintenance_slots", new_callable=AsyncMock, return_value=1)
@patch("api.server.router.get_active_upgrade_window", new_callable=AsyncMock)
async def test_policy_includes_pending_servers(mock_get_window, _mock_slots):
    window = _make_window()
    mock_get_window.return_value = window

    pending_server = _make_server(
        maintenance_pending_window_id=TEST_WINDOW_ID,
        version=TEST_VERSION_OLD,
    )
    db = AsyncMock()
    db.execute.return_value = _mock_scalars_result([pending_server])

    result = await get_maintenance_policy(db=db, hotkey=TEST_HOTKEY, _=None)
    assert result.current_slots == 1
    assert len(result.pending_servers) == 1
    assert result.pending_servers[0].server_id == TEST_SERVER_ID
    assert result.pending_servers[0].version == TEST_VERSION_OLD
    assert result.pending_servers[0].target_version == TEST_VERSION_TARGET


# ---------------------------------------------------------------------------
# GET /servers/{server_id}/maintenance/preflight
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@patch("api.server.router.preflight_maintenance", new_callable=AsyncMock)
@patch("api.server.router.get_server_by_name_or_id", new_callable=AsyncMock)
async def test_preflight_route_returns_result(mock_lookup, mock_preflight):
    server = _make_server()
    mock_lookup.return_value = server
    expected = PreflightResult(eligible=True, current_slots=0, limit=1)
    mock_preflight.return_value = expected
    db = AsyncMock()

    result = await get_maintenance_preflight(
        server_name_or_id=TEST_SERVER_NAME_OR_ID, db=db, hotkey=TEST_HOTKEY, _=None
    )
    assert result is expected
    mock_lookup.assert_awaited_once_with(db, TEST_HOTKEY, TEST_SERVER_NAME_OR_ID)
    mock_preflight.assert_awaited_once_with(db, server, TEST_HOTKEY)


@pytest.mark.asyncio
@patch("api.server.router.preflight_maintenance", new_callable=AsyncMock)
@patch("api.server.router.get_server_by_name_or_id", new_callable=AsyncMock)
async def test_preflight_route_returns_ineligible(mock_lookup, mock_preflight):
    server = _make_server()
    mock_lookup.return_value = server
    expected = PreflightResult(
        eligible=False,
        denial_reasons=[MaintenanceReason(reason="concurrency_cap", current_slots=1, limit=1)],
        current_slots=1,
        limit=1,
    )
    mock_preflight.return_value = expected
    db = AsyncMock()

    result = await get_maintenance_preflight(
        server_name_or_id=TEST_SERVER_NAME_OR_ID, db=db, hotkey=TEST_HOTKEY, _=None
    )
    assert result.eligible is False
    assert len(result.denial_reasons) == 1


# ---------------------------------------------------------------------------
# PUT /servers/{server_id}/maintenance
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@patch("api.server.router.confirm_maintenance", new_callable=AsyncMock)
@patch("api.server.router.get_server_by_name_or_id", new_callable=AsyncMock)
async def test_confirm_route_returns_result(mock_lookup, mock_confirm):
    server = _make_server()
    mock_lookup.return_value = server
    expected = ConfirmMaintenanceResult(
        server_id=TEST_SERVER_ID,
        purged_instance_ids=["inst-1"],
        window=UpgradeWindowInfo(
            id=TEST_WINDOW_ID,
            target_measurement_version=TEST_VERSION_TARGET,
            upgrade_window_start=str(TEST_WINDOW_START),
            upgrade_window_end=str(TEST_WINDOW_END),
        ),
    )
    mock_confirm.return_value = expected
    db = AsyncMock()

    result = await put_confirm_maintenance(
        server_name_or_id=TEST_SERVER_NAME_OR_ID, db=db, hotkey=TEST_HOTKEY, _=None
    )
    assert result is expected
    mock_lookup.assert_awaited_once_with(db, TEST_HOTKEY, TEST_SERVER_NAME_OR_ID)
    mock_confirm.assert_awaited_once_with(db, server, TEST_HOTKEY)


@pytest.mark.asyncio
@patch("api.server.router.confirm_maintenance", new_callable=AsyncMock)
@patch("api.server.router.get_server_by_name_or_id", new_callable=AsyncMock)
async def test_confirm_route_propagates_409(mock_lookup, mock_confirm):
    server = _make_server()
    mock_lookup.return_value = server
    mock_confirm.side_effect = HTTPException(status_code=409, detail="conflict")
    db = AsyncMock()

    with pytest.raises(HTTPException) as exc_info:
        await put_confirm_maintenance(
            server_name_or_id=TEST_SERVER_NAME_OR_ID, db=db, hotkey=TEST_HOTKEY, _=None
        )
    assert exc_info.value.status_code == 409


@pytest.mark.asyncio
@patch("api.server.router.confirm_maintenance", new_callable=AsyncMock)
@patch("api.server.router.get_server_by_name_or_id", new_callable=AsyncMock)
async def test_confirm_route_propagates_403(mock_lookup, mock_confirm):
    server = _make_server()
    mock_lookup.return_value = server
    mock_confirm.side_effect = HTTPException(status_code=403, detail="forbidden")
    db = AsyncMock()

    with pytest.raises(HTTPException) as exc_info:
        await put_confirm_maintenance(
            server_name_or_id=TEST_SERVER_NAME_OR_ID, db=db, hotkey=TEST_HOTKEY, _=None
        )
    assert exc_info.value.status_code == 403


# ---------------------------------------------------------------------------
# GET /servers/{server_id} — includes version + maintenance info
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@patch("api.server.router.check_server_ownership", new_callable=AsyncMock)
async def test_get_server_details_includes_version(mock_ownership):
    server = _make_server(version=TEST_VERSION_OLD)
    server.created_at = TEST_WINDOW_START
    server.updated_at = None
    mock_ownership.return_value = server
    db = AsyncMock()

    result = await get_server_details(server_id=TEST_SERVER_ID, db=db, hotkey=TEST_HOTKEY, _=None)
    assert result["version"] == TEST_VERSION_OLD
    assert result["maintenance_pending_window_id"] is None
    assert "target_version" not in result


@pytest.mark.asyncio
@patch("api.server.router.check_server_ownership", new_callable=AsyncMock)
async def test_get_server_details_includes_target_when_pending(mock_ownership):
    server = _make_server(
        version=TEST_VERSION_OLD,
        maintenance_pending_window_id=TEST_WINDOW_ID,
    )
    server.created_at = TEST_WINDOW_START
    server.updated_at = None
    mock_ownership.return_value = server

    window = _make_window()
    db = AsyncMock()
    db.get = AsyncMock(return_value=window)

    result = await get_server_details(server_id=TEST_SERVER_ID, db=db, hotkey=TEST_HOTKEY, _=None)
    assert result["version"] == TEST_VERSION_OLD
    assert result["maintenance_pending_window_id"] == TEST_WINDOW_ID
    assert result["target_version"] == TEST_VERSION_TARGET
    db.get.assert_awaited_once_with(TeeUpgradeWindow, TEST_WINDOW_ID)
