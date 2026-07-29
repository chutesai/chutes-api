"""Unit tests for the chute log shipper (validator side).

Covers timestamp parsing, mTLS shipment authentication (cross-miner rejection),
the cutoff decision, ingest dedupe/enrichment, and the server-forced read query.
"""

import datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from cryptography import x509
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.x509.oid import NameOID
from fastapi import HTTPException

from api.chute_logs import loki, service
from api.chute_logs.schemas import LogLine, LogShipmentArgs
from api.config import settings


# ---------------------------------------------------------------------------
# Cert helpers
# ---------------------------------------------------------------------------
def _key():
    return rsa.generate_private_key(public_exponent=65537, key_size=2048, backend=default_backend())


def _ca(key, cn="sek8s-vm-root-ca"):
    name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, cn)])
    return (
        x509.CertificateBuilder()
        .subject_name(name)
        .issuer_name(name)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(datetime.datetime.utcnow())
        .not_valid_after(datetime.datetime.utcnow() + datetime.timedelta(days=1))
        .add_extension(x509.BasicConstraints(ca=True, path_length=None), critical=True)
        .sign(key, hashes.SHA256())
    )


def _leaf(leaf_key, ca_key, ca_cert, cn="sek8s-vm-registry-client"):
    name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, cn)])
    return (
        x509.CertificateBuilder()
        .subject_name(name)
        .issuer_name(ca_cert.subject)
        .public_key(leaf_key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(datetime.datetime.utcnow())
        .not_valid_after(datetime.datetime.utcnow() + datetime.timedelta(days=1))
        .sign(ca_key, hashes.SHA256())
    )


def _server_with_ca(ca_cert):
    """A Server-like stub whose parsed CA is ``ca_cert`` (bypasses PEM round-trip)."""
    return SimpleNamespace(vm_root_ca_certificate=ca_cert, vm_root_ca_cert="pem", miner_hotkey="hk")


def _mock_db(launch_config, chute, servers):
    """AsyncMock db whose two scalar() calls return (launch_config, chute) and execute() → servers."""
    db = MagicMock()
    db.scalar = AsyncMock(side_effect=[launch_config, chute])
    exec_result = MagicMock()
    exec_result.scalars.return_value.all.return_value = servers
    db.execute = AsyncMock(return_value=exec_result)
    return db


# ---------------------------------------------------------------------------
# Timestamp parsing
# ---------------------------------------------------------------------------
def test_rfc3339nano_preserves_nanoseconds():
    ns = loki.rfc3339nano_to_unix_ns("2026-07-27T00:00:00.123456789Z")
    assert ns % 1_000_000_000 == 123456789


def test_rfc3339nano_two_lines_distinct():
    a = loki.rfc3339nano_to_unix_ns("2026-07-27T00:00:00.000000001Z")
    b = loki.rfc3339nano_to_unix_ns("2026-07-27T00:00:00.000000002Z")
    assert b - a == 1


def test_rfc3339nano_microseconds_and_offset():
    assert loki.rfc3339nano_to_unix_ns("2026-07-27T00:00:00.123456+00:00") is not None
    assert loki.rfc3339nano_to_unix_ns(
        "2026-07-27T01:00:00.5+01:00"
    ) == loki.rfc3339nano_to_unix_ns("2026-07-27T00:00:00.5Z")


def test_rfc3339nano_invalid_returns_none():
    assert loki.rfc3339nano_to_unix_ns("not-a-timestamp") is None
    assert loki.rfc3339nano_to_unix_ns("") is None


# ---------------------------------------------------------------------------
# Authentication
# ---------------------------------------------------------------------------
def _lc_row(config_id="c1", chute_id="ch1", miner_hotkey="hk"):
    return SimpleNamespace(
        config_id=config_id,
        chute_id=chute_id,
        miner_hotkey=miner_hotkey,
        created_at=datetime.datetime.now(datetime.timezone.utc),
    )


@pytest.mark.asyncio
async def test_authenticate_success_returns_context():
    ca_key = _key()
    ca_cert = _ca(ca_key)
    leaf = _leaf(_key(), ca_key, ca_cert)
    chute = SimpleNamespace(chute_id="ch1", user_id="u1", public=True)
    db = _mock_db(_lc_row(), chute, [_server_with_ca(ca_cert)])

    ctx = await service._authenticate(db, "c1", leaf)
    assert ctx.config_id == "c1"
    assert ctx.chute_id == "ch1"
    assert ctx.user_id == "u1"
    assert ctx.miner_hotkey == "hk"


@pytest.mark.asyncio
async def test_authenticate_unknown_config_404():
    db = _mock_db(None, None, [])
    with pytest.raises(HTTPException) as exc:
        await service._authenticate(db, "missing", _leaf(_key(), _key(), _ca(_key())))
    assert exc.value.status_code == 404


@pytest.mark.asyncio
async def test_authenticate_cross_miner_403():
    """A leaf signed by a DIFFERENT miner's CA must not authenticate this config."""
    good_ca = _ca(_key())
    other_ca_key = _key()
    other_ca = _ca(other_ca_key, cn="other-ca")
    # Leaf signed by the attacker's CA, but the config-owner only has the good CA on file.
    attacker_leaf = _leaf(_key(), other_ca_key, other_ca)
    chute = SimpleNamespace(chute_id="ch1", user_id="u1", public=True)
    db = _mock_db(_lc_row(), chute, [_server_with_ca(good_ca)])

    with pytest.raises(HTTPException) as exc:
        await service._authenticate(db, "c1", attacker_leaf)
    assert exc.value.status_code == 403


# ---------------------------------------------------------------------------
# Cutoff (_compute_stop is a pure function of state + debug flag)
# ---------------------------------------------------------------------------
def _now_iso(minutes_ago=1):
    return (
        datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(minutes=minutes_ago)
    ).isoformat()


def _lc(failed=False, activated=False):
    instance = SimpleNamespace(activated_at=datetime.datetime.utcnow()) if activated else None
    return SimpleNamespace(
        config_id="c1",
        failed_at=datetime.datetime.utcnow() if failed else None,
        instance=instance,
    )


def test_stop_running_continues():
    assert service._compute_stop(_lc(), _now_iso(), debug=False) is False


def test_stop_activated_stops():
    assert service._compute_stop(_lc(activated=True), _now_iso(), debug=False) is True


def test_stop_failed_stops():
    assert service._compute_stop(_lc(failed=True), _now_iso(), debug=False) is True


def test_debug_override_continues_past_activation():
    assert service._compute_stop(_lc(activated=True), _now_iso(), debug=True) is False


def test_max_capture_backstop_wins_over_debug(monkeypatch):
    monkeypatch.setattr(settings, "chute_logs_max_capture_seconds", 60)
    # created 10 minutes ago, backstop is 60s -> stop even with debug on.
    assert service._compute_stop(_lc(), _now_iso(minutes_ago=10), debug=True) is True


def test_outcome_derivation():
    assert service._outcome(_lc()) == "running"
    assert service._outcome(_lc(failed=True)) == "failed"
    assert service._outcome(_lc(activated=True)) == "activated"


# ---------------------------------------------------------------------------
# Ingest
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_ingest_dedupes_on_watermark(monkeypatch):
    pushed = AsyncMock()
    monkeypatch.setattr(loki.LokiClient, "_instance", SimpleNamespace(push=pushed))
    monkeypatch.setattr(settings, "loki_url", "http://loki")
    # Watermark already at line-1's ts; only the later line is fresh.
    ts1 = "2026-07-27T00:00:00.000000001Z"
    ts2 = "2026-07-27T00:00:00.000000002Z"
    wm = str(loki.rfc3339nano_to_unix_ns(ts1))
    redis = AsyncMock(get=AsyncMock(return_value=wm), set=AsyncMock())
    monkeypatch.setattr(settings, "_redis_client", redis)

    ctx = service.LogCaptureContext(
        config_id="c1", chute_id="ch1", user_id="u1", miner_hotkey="hk", created_at_iso=""
    )
    args = LogShipmentArgs(
        deployment_id="d1",
        logs=[
            LogLine(ts=ts1, stream="stdout", log="old"),
            LogLine(ts=ts2, stream="stdout", log="new"),
        ],
    )
    stored = await service.ingest(args, ctx, "running", "1.2.3.4")
    assert stored == 1
    pushed.assert_awaited_once()
    # The enriched line carries server-derived identity, never self-asserted by the guest.
    streams = pushed.call_args.args[0]
    import json as _json

    record = _json.loads(streams[0]["values"][0][1])
    assert record["config_id"] == "c1"
    assert record["chute_id"] == "ch1"
    assert record["user_id"] == "u1"
    assert record["server_ip"] == "1.2.3.4"
    assert record["log"] == "new"


@pytest.mark.asyncio
async def test_ingest_noop_without_loki(monkeypatch):
    pushed = AsyncMock()
    monkeypatch.setattr(loki.LokiClient, "_instance", SimpleNamespace(push=pushed))
    monkeypatch.setattr(settings, "loki_url", None)
    monkeypatch.setattr(
        settings, "_redis_client", AsyncMock(get=AsyncMock(return_value=None), set=AsyncMock())
    )
    ctx = service.LogCaptureContext(
        config_id="c1", chute_id="ch1", user_id="u1", miner_hotkey="hk", created_at_iso=""
    )
    args = LogShipmentArgs(logs=[LogLine(ts="2026-07-27T00:00:00.5Z", log="x")])
    stored = await service.ingest(args, ctx, "running", None)
    # Accepted (counted) but not pushed anywhere.
    assert stored == 1
    pushed.assert_not_awaited()


# ---------------------------------------------------------------------------
# Reads — forced matcher / injection safety
# ---------------------------------------------------------------------------
def test_build_query_forces_matchers_and_escapes():
    q = service._build_query('c"1', {"user_id": 'u\\"x'})
    assert 'config_id="c\\"1"' in q
    assert 'user_id="u\\\\\\"x"' in q
    assert q.startswith(f'{{app="{loki.APP_LABEL}"}}')


@pytest.mark.asyncio
async def test_read_config_logs_forces_user_id(monkeypatch):
    captured = {}

    async def fake_query_range(logql, start, end, limit=5000):
        captured["logql"] = logql
        return [("1", {"ts": "t", "stream": "stdout", "log": "hello"})]

    monkeypatch.setattr(settings, "loki_url", "http://loki")
    monkeypatch.setattr(loki.LokiClient, "_instance", SimpleNamespace(query_range=fake_query_range))
    lines = await service.read_config_logs("c1", {"user_id": "u1"})
    assert lines[0].log == "hello"
    assert 'config_id="c1"' in captured["logql"]
    assert 'user_id="u1"' in captured["logql"]


@pytest.mark.asyncio
async def test_read_config_logs_empty_without_loki(monkeypatch):
    monkeypatch.setattr(settings, "loki_url", None)
    assert await service.read_config_logs("c1", {}) == []
