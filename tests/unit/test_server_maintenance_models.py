"""
Unit tests for TEE maintenance ORM models and config settings (Phase 1).
"""

from datetime import datetime, timezone

from api.server.schemas import TeeUpgradeWindow, Server

TEST_SERVER_ID = "node-abc-123"
TEST_IP = "10.0.0.1"
TEST_HOTKEY = "5GrwvaEF5zXb26Fz9rcQpDWS57CtERHpNehXCPcNoHGKutQY"
TEST_VM_NAME = "my-tee-vm"
TEST_WINDOW_ID = "window-abc-123"
TEST_VERSION = "0.3.1"
TEST_WINDOW_START = datetime(2026, 4, 1, tzinfo=timezone.utc)
TEST_WINDOW_END = datetime(2026, 4, 7, tzinfo=timezone.utc)


def _make_server(**overrides):
    defaults = dict(
        server_id=TEST_SERVER_ID,
        ip=TEST_IP,
        miner_hotkey=TEST_HOTKEY,
        name=TEST_VM_NAME,
        netuid=64,
        is_tee=True,
    )
    defaults.update(overrides)
    return Server(**defaults)


def test_tee_upgrade_window_columns():
    window = TeeUpgradeWindow(
        id=TEST_WINDOW_ID,
        upgrade_window_start=TEST_WINDOW_START,
        upgrade_window_end=TEST_WINDOW_END,
        target_measurement_version=TEST_VERSION,
    )
    assert window.id == TEST_WINDOW_ID
    assert window.upgrade_window_start == TEST_WINDOW_START
    assert window.upgrade_window_end == TEST_WINDOW_END
    assert window.target_measurement_version == TEST_VERSION


def test_server_maintenance_pending_window_id_defaults_to_none():
    assert _make_server().maintenance_pending_window_id is None


def test_server_version_defaults_to_none():
    assert _make_server().version is None


def test_server_maintenance_pending_window_id_can_be_set():
    server = _make_server(maintenance_pending_window_id=TEST_WINDOW_ID)
    assert server.maintenance_pending_window_id == TEST_WINDOW_ID


def test_server_version_can_be_set():
    server = _make_server(version=TEST_VERSION)
    assert server.version == TEST_VERSION


def test_server_existing_columns_unaffected():
    server = _make_server()
    assert server.server_id == TEST_SERVER_ID
    assert server.ip == TEST_IP
    assert server.miner_hotkey == TEST_HOTKEY
    assert server.name == TEST_VM_NAME
    assert server.is_tee is True


def test_server_has_pending_upgrade_window_relationship():
    assert hasattr(Server, "pending_upgrade_window")


def test_window_has_pending_servers_relationship():
    assert hasattr(TeeUpgradeWindow, "pending_servers")


def test_window_max_concurrent_per_miner_has_column_default():
    col = TeeUpgradeWindow.__table__.c.max_concurrent_per_miner
    assert col.server_default.arg == "1"
    assert col.nullable is False


def test_window_max_concurrent_per_miner_can_be_set():
    w = TeeUpgradeWindow(
        upgrade_window_start=datetime(2026, 4, 1, tzinfo=timezone.utc),
        upgrade_window_end=datetime(2026, 4, 7, tzinfo=timezone.utc),
        target_measurement_version="0.3.0",
        max_concurrent_per_miner=3,
    )
    assert w.max_concurrent_per_miner == 3
