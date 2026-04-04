"""
Unit tests for TEE maintenance service functions (Phase 2).
"""

import pytest
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

from fastapi import HTTPException

from api.server.schemas import (
    Server,
    TeeUpgradeWindow,
    MaintenanceReason,
    SoleSurvivorBlock,
    PreflightResult,
)
from api.server.service import (
    get_active_upgrade_window,
    preflight_maintenance,
    confirm_maintenance,
    _get_instances_on_server,
    _find_sole_survivor_chutes,
    _count_active_maintenance_slots,
)

TEST_SERVER_ID = "server-abc-123"
TEST_HOTKEY = "5GrwvaEF5zXb26Fz9rcQpDWS57CtERHpNehXCPcNoHGKutQY"
TEST_VM_NAME = "my-tee-vm"
TEST_WINDOW_ID = "window-abc-123"
TEST_WINDOW_ID_2 = "window-def-456"
TEST_VERSION_OLD = "0.2.0"
TEST_VERSION_TARGET = "0.3.1"
TEST_VERSION_ABOVE = "0.4.0"
TEST_WINDOW_START = datetime(2026, 4, 1, tzinfo=timezone.utc)
TEST_WINDOW_END = datetime(2026, 4, 7, tzinfo=timezone.utc)
TEST_CHUTE_ID = "chute-abc-123"
TEST_INSTANCE_ID = "inst-abc-123"
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


def _make_instance(**overrides):
    from api.instance.schemas import Instance

    defaults = dict(
        instance_id=TEST_INSTANCE_ID,
        chute_id=TEST_CHUTE_ID,
        miner_hotkey=TEST_HOTKEY,
        active=True,
    )
    defaults.update(overrides)
    return Instance(**defaults)


def _make_preflight(eligible=True, denial_reasons=None, blocking=None, current_slots=0, limit=1):
    return PreflightResult(
        eligible=eligible,
        denial_reasons=denial_reasons or [],
        blocking_chute_ids=blocking or [],
        current_slots=current_slots,
        limit=limit,
    )


def _mock_scalars_result(rows):
    """Build a mock result whose .scalars().all() returns the given rows."""
    scalars_mock = MagicMock()
    scalars_mock.all.return_value = rows
    result_mock = MagicMock()
    result_mock.scalars.return_value = scalars_mock
    return result_mock


def _mock_scalar_result(value):
    """Build a mock result whose .scalar() returns the given value."""
    result_mock = MagicMock()
    result_mock.scalar.return_value = value
    return result_mock


def _mock_scalar_one_or_none_result(value):
    """Build a mock result whose .scalar_one_or_none() returns the given value."""
    result_mock = MagicMock()
    result_mock.scalar_one_or_none.return_value = value
    return result_mock


# ---------------------------------------------------------------------------
# get_active_upgrade_window
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_get_active_upgrade_window_returns_none_when_no_rows():
    db = AsyncMock()
    db.execute.return_value = _mock_scalars_result([])
    result = await get_active_upgrade_window(db)
    assert result is None


@pytest.mark.asyncio
async def test_get_active_upgrade_window_returns_single_active_row():
    window = _make_window()
    db = AsyncMock()
    db.execute.return_value = _mock_scalars_result([window])
    result = await get_active_upgrade_window(db)
    assert result is window


@pytest.mark.asyncio
async def test_get_active_upgrade_window_picks_most_recent_when_overlapping():
    newer = _make_window(id=TEST_WINDOW_ID, created_at=datetime(2026, 4, 2, tzinfo=timezone.utc))
    older = _make_window(id=TEST_WINDOW_ID_2, created_at=datetime(2026, 4, 1, tzinfo=timezone.utc))
    db = AsyncMock()
    db.execute.return_value = _mock_scalars_result([newer, older])
    result = await get_active_upgrade_window(db)
    assert result is newer


# ---------------------------------------------------------------------------
# _get_instances_on_server
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_get_instances_on_server_returns_instances():
    inst = _make_instance()
    db = AsyncMock()
    db.execute.return_value = _mock_scalars_result([inst])
    result = await _get_instances_on_server(db, TEST_SERVER_ID)
    assert result == [inst]


@pytest.mark.asyncio
async def test_get_instances_on_server_returns_empty_list():
    db = AsyncMock()
    db.execute.return_value = _mock_scalars_result([])
    result = await _get_instances_on_server(db, TEST_SERVER_ID)
    assert result == []


# ---------------------------------------------------------------------------
# _find_sole_survivor_chutes
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_find_sole_survivor_chutes_no_blocking():
    inst = _make_instance()
    db = AsyncMock()
    db.execute.return_value = _mock_scalar_result(1)
    result = await _find_sole_survivor_chutes(db, [inst])
    assert result == []


@pytest.mark.asyncio
async def test_find_sole_survivor_chutes_blocks_sole_instance():
    inst = _make_instance()
    db = AsyncMock()
    db.execute.return_value = _mock_scalar_result(0)
    result = await _find_sole_survivor_chutes(db, [inst])
    assert len(result) == 1
    assert isinstance(result[0], SoleSurvivorBlock)
    assert result[0].chute_id == TEST_CHUTE_ID
    assert result[0].instance_id == TEST_INSTANCE_ID


@pytest.mark.asyncio
async def test_find_sole_survivor_chutes_deduplicates_by_chute():
    inst_a = _make_instance(instance_id="inst-1", chute_id=TEST_CHUTE_ID)
    inst_b = _make_instance(instance_id="inst-2", chute_id=TEST_CHUTE_ID)
    db = AsyncMock()
    db.execute.return_value = _mock_scalar_result(0)
    result = await _find_sole_survivor_chutes(db, [inst_a, inst_b])
    assert len(result) == 1


# ---------------------------------------------------------------------------
# _count_active_maintenance_slots
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_count_active_maintenance_slots():
    window = _make_window()
    db = AsyncMock()
    db.execute.return_value = _mock_scalar_result(2)
    result = await _count_active_maintenance_slots(db, TEST_HOTKEY, window)
    assert result == 2


@pytest.mark.asyncio
async def test_count_active_maintenance_slots_returns_zero_when_null():
    window = _make_window()
    db = AsyncMock()
    db.execute.return_value = _mock_scalar_result(None)
    result = await _count_active_maintenance_slots(db, TEST_HOTKEY, window)
    assert result == 0


# ---------------------------------------------------------------------------
# preflight_maintenance
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@patch("api.server.service.get_active_upgrade_window", new_callable=AsyncMock)
async def test_preflight_not_tee(_mock_window):
    server = _make_server(is_tee=False)
    db = AsyncMock()
    result = await preflight_maintenance(db, server, TEST_HOTKEY)
    assert isinstance(result, PreflightResult)
    assert result.eligible is False
    assert any(r.reason == "not_tee" for r in result.denial_reasons)


@pytest.mark.asyncio
@patch("api.server.service.get_active_upgrade_window", new_callable=AsyncMock, return_value=None)
async def test_preflight_no_active_window(_mock_window):
    server = _make_server()
    db = AsyncMock()
    result = await preflight_maintenance(db, server, TEST_HOTKEY)
    assert result.eligible is False
    assert any(r.reason == "no_active_window" for r in result.denial_reasons)


@pytest.mark.asyncio
@patch("api.server.service._find_sole_survivor_chutes", new_callable=AsyncMock, return_value=[])
@patch("api.server.service._get_instances_on_server", new_callable=AsyncMock, return_value=[])
@patch("api.server.service._count_active_maintenance_slots", new_callable=AsyncMock, return_value=0)
@patch("api.server.service.get_active_upgrade_window", new_callable=AsyncMock)
async def test_preflight_already_at_target(
    mock_window, _mock_slots, _mock_instances, _mock_survivors
):
    window = _make_window()
    mock_window.return_value = window
    server = _make_server(version=TEST_VERSION_ABOVE)
    db = AsyncMock()
    result = await preflight_maintenance(db, server, TEST_HOTKEY)
    assert result.eligible is False
    assert any(r.reason == "already_at_target" for r in result.denial_reasons)


@pytest.mark.asyncio
@patch("api.server.service._find_sole_survivor_chutes", new_callable=AsyncMock, return_value=[])
@patch("api.server.service._get_instances_on_server", new_callable=AsyncMock, return_value=[])
@patch("api.server.service._count_active_maintenance_slots", new_callable=AsyncMock, return_value=0)
@patch("api.server.service.get_active_upgrade_window", new_callable=AsyncMock)
async def test_preflight_maintenance_pending(
    mock_window, _mock_slots, _mock_instances, _mock_survivors
):
    window = _make_window()
    mock_window.return_value = window
    server = _make_server(maintenance_pending_window_id=TEST_WINDOW_ID)
    db = AsyncMock()
    result = await preflight_maintenance(db, server, TEST_HOTKEY)
    assert result.eligible is False
    assert any(r.reason == "maintenance_pending" for r in result.denial_reasons)


@pytest.mark.asyncio
@patch("api.server.service._find_sole_survivor_chutes", new_callable=AsyncMock, return_value=[])
@patch("api.server.service._get_instances_on_server", new_callable=AsyncMock, return_value=[])
@patch("api.server.service._count_active_maintenance_slots", new_callable=AsyncMock, return_value=0)
@patch("api.server.service.get_active_upgrade_window", new_callable=AsyncMock)
async def test_preflight_stale_window_gets_cleared(
    mock_window, _mock_slots, _mock_instances, _mock_survivors
):
    window = _make_window(id="new-window-id")
    mock_window.return_value = window
    server = _make_server(maintenance_pending_window_id="old-stale-window-id")
    db = AsyncMock()
    result = await preflight_maintenance(db, server, TEST_HOTKEY)
    assert result.eligible is True
    assert server.maintenance_pending_window_id is None


@pytest.mark.asyncio
@patch("api.server.service._find_sole_survivor_chutes", new_callable=AsyncMock, return_value=[])
@patch("api.server.service._get_instances_on_server", new_callable=AsyncMock, return_value=[])
@patch("api.server.service._count_active_maintenance_slots", new_callable=AsyncMock)
@patch("api.server.service.get_active_upgrade_window", new_callable=AsyncMock)
async def test_preflight_concurrency_cap(mock_window, mock_slots, _mock_instances, _mock_survivors):
    window = _make_window()
    mock_window.return_value = window
    mock_slots.return_value = DEFAULT_CONCURRENCY_LIMIT
    server = _make_server()
    db = AsyncMock()
    result = await preflight_maintenance(db, server, TEST_HOTKEY)
    assert result.eligible is False
    assert any(r.reason == "concurrency_cap" for r in result.denial_reasons)


@pytest.mark.asyncio
@patch("api.server.service._find_sole_survivor_chutes", new_callable=AsyncMock)
@patch("api.server.service._get_instances_on_server", new_callable=AsyncMock)
@patch("api.server.service._count_active_maintenance_slots", new_callable=AsyncMock, return_value=0)
@patch("api.server.service.get_active_upgrade_window", new_callable=AsyncMock)
async def test_preflight_sole_survivor_blocks(
    mock_window, _mock_slots, mock_instances, mock_survivors
):
    window = _make_window()
    mock_window.return_value = window
    inst = _make_instance()
    mock_instances.return_value = [inst]
    blocking = [SoleSurvivorBlock(chute_id=TEST_CHUTE_ID, instance_id=TEST_INSTANCE_ID)]
    mock_survivors.return_value = blocking
    server = _make_server()
    db = AsyncMock()
    result = await preflight_maintenance(db, server, TEST_HOTKEY)
    assert result.eligible is False
    assert any(r.reason == "sole_survivor" for r in result.denial_reasons)
    assert result.blocking_chute_ids == blocking


@pytest.mark.asyncio
@patch("api.server.service._find_sole_survivor_chutes", new_callable=AsyncMock, return_value=[])
@patch("api.server.service._get_instances_on_server", new_callable=AsyncMock, return_value=[])
@patch("api.server.service._count_active_maintenance_slots", new_callable=AsyncMock, return_value=0)
@patch("api.server.service.get_active_upgrade_window", new_callable=AsyncMock)
async def test_preflight_eligible_version_none(
    mock_window, _mock_slots, _mock_instances, _mock_survivors
):
    """Server with version=None should not be denied as 'already at target'."""

    window = _make_window()
    mock_window.return_value = window
    server = _make_server(version=None)
    db = AsyncMock()
    result = await preflight_maintenance(db, server, TEST_HOTKEY)
    assert result.eligible is True
    assert result.denial_reasons == []


@pytest.mark.asyncio
@patch("api.server.service._find_sole_survivor_chutes", new_callable=AsyncMock, return_value=[])
@patch("api.server.service._get_instances_on_server", new_callable=AsyncMock, return_value=[])
@patch("api.server.service._count_active_maintenance_slots", new_callable=AsyncMock, return_value=0)
@patch("api.server.service.get_active_upgrade_window", new_callable=AsyncMock)
async def test_preflight_eligible_old_version(
    mock_window, _mock_slots, _mock_instances, _mock_survivors
):
    window = _make_window()
    mock_window.return_value = window
    server = _make_server(version=TEST_VERSION_OLD)
    db = AsyncMock()
    result = await preflight_maintenance(db, server, TEST_HOTKEY)
    assert result.eligible is True
    assert result.denial_reasons == []


# ---------------------------------------------------------------------------
# confirm_maintenance
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@patch("api.server.service.get_active_upgrade_window", new_callable=AsyncMock)
@patch("api.server.service.preflight_maintenance", new_callable=AsyncMock)
async def test_confirm_raises_on_ineligible(mock_preflight, mock_window):
    mock_preflight.return_value = _make_preflight(
        eligible=False,
        denial_reasons=[MaintenanceReason(reason="concurrency_cap", current_slots=1, limit=1)],
        current_slots=1,
    )
    server = _make_server()
    db = AsyncMock()
    with pytest.raises(HTTPException) as exc_info:
        await confirm_maintenance(db, server, TEST_HOTKEY)
    assert exc_info.value.status_code == 409


@pytest.mark.asyncio
@patch("api.server.service.get_active_upgrade_window", new_callable=AsyncMock)
@patch("api.server.service.preflight_maintenance", new_callable=AsyncMock)
async def test_confirm_raises_403_for_non_tee(mock_preflight, mock_window):
    mock_preflight.return_value = _make_preflight(
        eligible=False,
        denial_reasons=[MaintenanceReason(reason="not_tee")],
    )
    server = _make_server(is_tee=False)
    db = AsyncMock()
    with pytest.raises(HTTPException) as exc_info:
        await confirm_maintenance(db, server, TEST_HOTKEY)
    assert exc_info.value.status_code == 403


@pytest.mark.asyncio
@patch("api.server.service.purge_and_notify", new_callable=AsyncMock)
@patch("api.server.service._get_instances_on_server", new_callable=AsyncMock)
@patch("api.server.service.get_active_upgrade_window", new_callable=AsyncMock)
@patch("api.server.service.preflight_maintenance", new_callable=AsyncMock)
async def test_confirm_success(mock_preflight, mock_window, mock_instances, mock_purge):
    window = _make_window()
    mock_preflight.return_value = _make_preflight(eligible=True)
    mock_window.return_value = window
    inst = _make_instance()
    mock_instances.return_value = [inst]
    server = _make_server()
    db = AsyncMock()
    result = await confirm_maintenance(db, server, TEST_HOTKEY)

    assert server.maintenance_pending_window_id == TEST_WINDOW_ID
    assert result.server_id == TEST_SERVER_ID
    assert inst.instance_id in result.purged_instance_ids
    assert result.window.id == TEST_WINDOW_ID
    mock_purge.assert_awaited_once_with(
        inst,
        reason="tee maintenance",
        valid_termination=True,
    )
    db.commit.assert_awaited_once()


@pytest.mark.asyncio
@patch("api.server.service.purge_and_notify", new_callable=AsyncMock)
@patch("api.server.service._get_instances_on_server", new_callable=AsyncMock)
@patch("api.server.service.get_active_upgrade_window", new_callable=AsyncMock)
@patch("api.server.service.preflight_maintenance", new_callable=AsyncMock)
async def test_confirm_purge_failure_does_not_crash(
    mock_preflight, mock_window, mock_instances, mock_purge
):
    window = _make_window()
    mock_preflight.return_value = _make_preflight(eligible=True)
    mock_window.return_value = window
    mock_instances.return_value = [_make_instance()]
    mock_purge.side_effect = RuntimeError("purge failed")
    server = _make_server()
    db = AsyncMock()
    result = await confirm_maintenance(db, server, TEST_HOTKEY)
    assert result.purged_instance_ids == []
