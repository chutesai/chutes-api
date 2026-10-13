"""Isolated tests for the Loki log read routes.

This file builds its own FastAPI app. It does not use tests/conftest.py, api.main,
or a database.
"""

from datetime import datetime, timezone

from fastapi import FastAPI
from fastapi.testclient import TestClient

from api.chute_logs.read_access import get_catalog, get_log_reader
from api.chute_logs.router import logs_user, router
from api.chute_logs.schemas import StoredLogLine
from api.permissions import Permissioning
from chute_log_read_doubles import (
    Catalog,
    ChuteRow,
    InstanceRow,
    LaunchConfigRow,
    LokiSpy,
    UserStub,
)

BOOT_LINE = StoredLogLine(ts="2026-07-27T00:00:00.000000000Z", stream="stdout", log="booting")
BOOT_JSON = {"ts": BOOT_LINE.ts, "stream": BOOT_LINE.stream, "log": BOOT_LINE.log}
DENY_DETAIL = "You may only view logs for your own (private) chutes."


def _catalog(owner_id="owner-1", public=False, name="private-model"):
    catalog = Catalog()
    catalog.instances["inst-1"] = InstanceRow("inst-1", "cfg-1", "chute-1")
    catalog.chutes["chute-1"] = ChuteRow("chute-1", owner_id, public=public, name=name)
    return catalog


def _client(user=None, catalog=None, loki=None):
    app = FastAPI()
    app.include_router(router, prefix="/logs")
    if user is not None:
        app.dependency_overrides[logs_user] = lambda: user
    if catalog is not None:
        app.dependency_overrides[get_catalog] = lambda: catalog
    if loki is not None:
        app.dependency_overrides[get_log_reader] = lambda: loki
    return TestClient(app)


def test_should_reject_user_log_read_when_unauthenticated():
    response = _client().get("/logs/instances/inst-1")
    assert response.status_code == 401


def test_should_return_logs_when_owner_of_private_chute():
    owner = UserStub("owner-1")
    loki = LokiSpy([BOOT_LINE])

    response = _client(user=owner, catalog=_catalog(), loki=loki).get("/logs/instances/inst-1")

    assert response.status_code == 200
    assert response.json() == [BOOT_JSON]
    assert loki.calls == [("cfg-1", {"user_id": "owner-1"})]


def test_should_forbid_user_log_read_when_caller_is_not_owner_or_sharee():
    loki = LokiSpy([BOOT_LINE])

    response = _client(user=UserStub("other-1"), catalog=_catalog(), loki=loki).get(
        "/logs/instances/inst-1"
    )

    assert response.status_code == 403
    assert response.json()["detail"] == DENY_DETAIL
    assert loki.calls == []


def test_should_return_logs_when_sharee_of_private_chute():
    catalog = _catalog()
    catalog.shares.add(("chute-1", "sharee-1"))
    loki = LokiSpy([BOOT_LINE])

    response = _client(user=UserStub("sharee-1"), catalog=catalog, loki=loki).get(
        "/logs/instances/inst-1"
    )

    assert response.status_code == 200
    assert response.json() == [BOOT_JSON]


def test_should_forbid_user_log_read_when_chute_is_public():
    loki = LokiSpy([BOOT_LINE])

    response = _client(
        user=UserStub("owner-1"),
        catalog=_catalog(public=True, name="public-model"),
        loki=loki,
    ).get("/logs/instances/inst-1")

    assert response.status_code == 403
    assert response.json()["detail"] == DENY_DETAIL
    assert loki.calls == []


def test_should_return_logs_when_support_reads_public_chute():
    support = UserStub("support-1", roles=[Permissioning.chutes_support])
    loki = LokiSpy([BOOT_LINE])

    response = _client(
        user=support,
        catalog=_catalog(public=True, name="public-model"),
        loki=loki,
    ).get("/logs/instances/inst-1")

    assert response.status_code == 200
    assert response.json() == [BOOT_JSON]
    assert loki.calls == [("cfg-1", {"user_id": "owner-1"})]


def test_should_return_logs_when_subnet_admin_overrides_public_denial():
    admin = UserStub("admin-1", netuids=[120], roles=[Permissioning.subnet_admin])
    loki = LokiSpy([BOOT_LINE])

    response = _client(
        user=admin,
        catalog=_catalog(public=True, name="affine-model"),
        loki=loki,
    ).get("/logs/instances/inst-1")

    assert response.status_code == 200
    assert response.json() == [BOOT_JSON]
    assert loki.calls == [("cfg-1", {"user_id": "owner-1"})]


def test_should_forbid_user_log_read_when_subnet_admin_name_does_not_match():
    admin = UserStub("admin-1", netuids=[120], roles=[Permissioning.subnet_admin])
    loki = LokiSpy([BOOT_LINE])

    response = _client(
        user=admin,
        catalog=_catalog(public=True, name="private-model"),
        loki=loki,
    ).get("/logs/instances/inst-1")

    assert response.status_code == 403
    assert response.json()["detail"] == DENY_DETAIL
    assert loki.calls == []


def test_should_forbid_subnet_invoke_without_subnet_admin():
    invoker = UserStub("invoke-1", netuids=[120], roles=[Permissioning.subnet_invoke])
    loki = LokiSpy([BOOT_LINE])

    response = _client(
        user=invoker,
        catalog=_catalog(public=True, name="affine-model"),
        loki=loki,
    ).get("/logs/instances/inst-1")

    assert response.status_code == 403
    assert response.json()["detail"] == DENY_DETAIL
    assert loki.calls == []


def test_should_return_not_found_when_instance_is_missing():
    response = _client(user=UserStub("owner-1"), catalog=Catalog(), loki=LokiSpy([BOOT_LINE])).get(
        "/logs/instances/missing"
    )

    assert response.status_code == 404


def test_should_query_loki_with_owner_user_id_when_sharee_reads():
    catalog = _catalog()
    catalog.shares.add(("chute-1", "sharee-1"))
    catalog.instances["inst-1"].activated_at = datetime(2026, 6, 1, tzinfo=timezone.utc)
    early = StoredLogLine(ts="2026-01-01T00:00:00.000000000Z", stream="stdout", log="early")
    late = StoredLogLine(ts="2026-12-01T00:00:00.000000000Z", stream="stdout", log="late")
    loki = LokiSpy([early, late])

    response = _client(user=UserStub("sharee-1"), catalog=catalog, loki=loki).get(
        '/logs/instances/inst-1?query={app="x"}&config_id=client-supplied-config'
    )

    assert response.status_code == 200
    assert response.json() == [
        {"ts": early.ts, "stream": "stdout", "log": "early"},
        {"ts": late.ts, "stream": "stdout", "log": "late"},
    ]
    assert loki.calls == [("cfg-1", {"user_id": "owner-1"})]


def test_should_reject_miner_log_read_when_jwt_is_missing():
    response = _client().get("/logs/miner")
    assert response.status_code == 401


def test_should_reject_miner_log_read_when_jwt_is_expired(monkeypatch):
    import jwt

    def _expired(_token):
        raise jwt.InvalidTokenError("expired")

    # Stands in for _decode_chutes_jwt. The route binds this name at import.
    monkeypatch.setattr("api.chute_logs.router.decode_launch_jwt", _expired)

    response = _client().get("/logs/miner", headers={"Authorization": "Bearer expired-token"})

    assert response.status_code == 401


def test_should_return_logs_when_jwt_is_valid_for_public_unactivated_config(monkeypatch):
    monkeypatch.setattr("api.chute_logs.router.decode_launch_jwt", lambda _token: {"sub": "cfg-1"})
    catalog = Catalog()
    catalog.launch_configs["cfg-1"] = LaunchConfigRow("cfg-1", "chute-1", "5MinerHotkey")
    catalog.chutes["chute-1"] = ChuteRow("chute-1", "owner-1", public=True, name="public-model")
    loki = LokiSpy([BOOT_LINE])

    response = _client(catalog=catalog, loki=loki).get(
        "/logs/miner", headers={"Authorization": "Bearer launch-token"}
    )

    assert response.status_code == 200
    assert response.json() == [BOOT_JSON]


def test_should_forbid_miner_log_read_when_chute_is_private(monkeypatch):
    monkeypatch.setattr("api.chute_logs.router.decode_launch_jwt", lambda _token: {"sub": "cfg-1"})
    catalog = Catalog()
    catalog.launch_configs["cfg-1"] = LaunchConfigRow("cfg-1", "chute-1", "5MinerHotkey")
    catalog.chutes["chute-1"] = ChuteRow("chute-1", "owner-1", public=False, name="private-model")
    loki = LokiSpy([BOOT_LINE])

    response = _client(catalog=catalog, loki=loki).get(
        "/logs/miner", headers={"Authorization": "Bearer launch-token"}
    )

    assert response.status_code == 403
    assert response.json()["detail"] == "Logs are only available for public chutes."
    assert loki.calls == []


def test_should_return_only_lines_before_activated_at(monkeypatch):
    monkeypatch.setattr("api.chute_logs.router.decode_launch_jwt", lambda _token: {"sub": "cfg-1"})
    activated_at = datetime(2026, 6, 1, tzinfo=timezone.utc)
    catalog = Catalog()
    catalog.launch_configs["cfg-1"] = LaunchConfigRow(
        "cfg-1",
        "chute-1",
        "5MinerHotkey",
        instance=InstanceRow("inst-1", "cfg-1", "chute-1", activated_at=activated_at),
    )
    catalog.chutes["chute-1"] = ChuteRow("chute-1", "owner-1", public=True, name="public-model")
    early = StoredLogLine(ts="2026-01-01T00:00:00.000000000Z", stream="stdout", log="early")
    at_activation = StoredLogLine(ts="2026-06-01T00:00:00.000000000Z", stream="stdout", log="at")
    late = StoredLogLine(ts="2026-12-01T00:00:00.000000000Z", stream="stdout", log="late")
    loki = LokiSpy([early, at_activation, late])

    response = _client(catalog=catalog, loki=loki).get(
        "/logs/miner", headers={"Authorization": "Bearer launch-token"}
    )

    assert response.status_code == 200
    assert response.json() == [{"ts": early.ts, "stream": "stdout", "log": "early"}]


def test_should_return_not_found_when_launch_config_is_unknown(monkeypatch):
    monkeypatch.setattr(
        "api.chute_logs.router.decode_launch_jwt", lambda _token: {"sub": "missing"}
    )

    response = _client(catalog=Catalog(), loki=LokiSpy()).get(
        "/logs/miner", headers={"Authorization": "Bearer launch-token"}
    )

    assert response.status_code == 404


def test_should_query_loki_with_launch_config_miner_hotkey(monkeypatch):
    monkeypatch.setattr("api.chute_logs.router.decode_launch_jwt", lambda _token: {"sub": "cfg-1"})
    catalog = Catalog()
    catalog.launch_configs["cfg-1"] = LaunchConfigRow("cfg-1", "chute-1", "5MinerHotkey")
    catalog.chutes["chute-1"] = ChuteRow("chute-1", "owner-1", public=True, name="public-model")
    loki = LokiSpy([BOOT_LINE])

    response = _client(catalog=catalog, loki=loki).get(
        "/logs/miner?config_id=client-supplied-config",
        headers={"Authorization": "Bearer launch-token"},
    )

    assert response.status_code == 200
    assert loki.calls == [("cfg-1", {"miner_hotkey": "5MinerHotkey"})]


def test_should_drop_post_activation_lines_when_debug_override_is_on(monkeypatch):
    monkeypatch.setattr("api.chute_logs.router.decode_launch_jwt", lambda _token: {"sub": "cfg-1"})

    async def _debug_on(_chute_id):
        return True

    set_debug_calls = []

    async def _set_debug(*args, **kwargs):
        set_debug_calls.append((args, kwargs))

    monkeypatch.setattr("api.chute_logs.service.is_debug_enabled", _debug_on)
    monkeypatch.setattr("api.chute_logs.service.set_debug", _set_debug)

    activated_at = datetime(2026, 6, 1, tzinfo=timezone.utc)
    catalog = Catalog()
    catalog.launch_configs["cfg-1"] = LaunchConfigRow(
        "cfg-1",
        "chute-1",
        "5MinerHotkey",
        instance=InstanceRow("inst-1", "cfg-1", "chute-1", activated_at=activated_at),
    )
    catalog.chutes["chute-1"] = ChuteRow("chute-1", "owner-1", public=True, name="public-model")
    early = StoredLogLine(ts="2026-01-01T00:00:00.000000000Z", stream="stdout", log="early")
    late = StoredLogLine(ts="2026-12-01T00:00:00.000000000Z", stream="stdout", log="late")
    loki = LokiSpy([early, late])

    response = _client(catalog=catalog, loki=loki).get(
        "/logs/miner", headers={"Authorization": "Bearer launch-token"}
    )

    assert response.status_code == 200
    assert response.json() == [{"ts": early.ts, "stream": "stdout", "log": "early"}]
    assert set_debug_calls == []


def test_should_never_call_read_config_logs_with_an_empty_matcher(monkeypatch):
    monkeypatch.setattr("api.chute_logs.router.decode_launch_jwt", lambda _token: {"sub": "cfg-1"})
    user_catalog = _catalog()
    user_loki = LokiSpy([BOOT_LINE])
    user_response = _client(user=UserStub("owner-1"), catalog=user_catalog, loki=user_loki).get(
        "/logs/instances/inst-1"
    )

    miner_catalog = Catalog()
    miner_catalog.launch_configs["cfg-1"] = LaunchConfigRow("cfg-1", "chute-1", "5MinerHotkey")
    miner_catalog.chutes["chute-1"] = ChuteRow(
        "chute-1", "owner-1", public=True, name="public-model"
    )
    miner_loki = LokiSpy([BOOT_LINE])
    miner_response = _client(catalog=miner_catalog, loki=miner_loki).get(
        "/logs/miner", headers={"Authorization": "Bearer launch-token"}
    )

    assert user_response.status_code == 200
    assert miner_response.status_code == 200
    calls = user_loki.calls + miner_loki.calls
    assert calls
    assert all(forced for _config_id, forced in calls)
