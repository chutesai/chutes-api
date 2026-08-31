from types import SimpleNamespace
from datetime import UTC, datetime

import pytest

import api.user.schemas  # noqa: F401 - register ORM relationships for isolated run
from api.external_backend import artifact_limits


class _ScalarResult:
    def __init__(self, value=None):
        self.value = value

    def scalar_one(self):
        return self.value


class _Session:
    def __init__(self, operation):
        self.operation = operation
        self.statements = []
        self.flushes = 0

    async def execute(self, statement, _params=None):
        self.statements.append((str(statement), _params))
        if "clock_timestamp" in str(statement):
            return _ScalarResult(datetime(2026, 8, 30, tzinfo=UTC))
        if "SUM(artifact_relay_bytes)" in str(statement):
            return _ScalarResult(0)
        return _ScalarResult()

    async def get(self, *_args, **_kwargs):
        return self.operation

    async def flush(self):
        self.flushes += 1


def _policy():
    return SimpleNamespace(
        artifact_max_bytes_per_operation=1_000,
        artifact_max_daily_bytes_per_user=10_000,
    )


@pytest.mark.asyncio
async def test_reservation_records_recoverable_claim(monkeypatch):
    operation = SimpleNamespace(user_id="user-id", settlement_metadata={})
    session = _Session(operation)
    monkeypatch.setattr(artifact_limits, "governance_policy", lambda _: _policy())

    reservation = await artifact_limits.reserve_artifact_bytes(
        session,
        operation_id="operation-id",
        user_id="user-id",
        connection_config={},
        expected_bytes=100,
        reservation_ttl_seconds=60,
    )

    assert reservation.max_bytes == 100
    assert operation.settlement_metadata["artifact_relay_bytes_reserved"] == 100
    record = operation.settlement_metadata["artifact_relay_reservations"][
        reservation.token
    ]
    assert record["bytes"] == 100
    assert record["expires_at"] > 0
    assert session.flushes == 1
    assert any(
        "INSERT INTO external_governance_buckets" in statement
        for statement, _ in session.statements
    )


@pytest.mark.asyncio
async def test_finalize_charges_actual_bytes_and_refunds_unused_budget():
    operation = SimpleNamespace(
        user_id="user-id",
        settlement_metadata={
            "artifact_relay_bytes_reserved": 100,
            "artifact_relay_reservations": {
                "reservation-id": {
                    "bytes": 100,
                    "bucket": 500,
                    "expires_at": 999_999_999_999,
                }
            },
        },
    )
    reservation = artifact_limits.ArtifactByteReservation(
        operation_id="operation-id",
        user_id="user-id",
        token="reservation-id",
        max_bytes=100,
    )

    session = _Session(operation)
    await artifact_limits.finalize_artifact_bytes(
        session, reservation, transferred_bytes=25
    )

    metadata = operation.settlement_metadata
    assert metadata["artifact_relay_bytes_reserved"] == 25
    assert metadata["artifact_relay_bytes_transferred"] == 25
    assert metadata["artifact_relay_reservations"] == {}
    assert session.flushes == 1
    refund_statements = [
        params
        for statement, params in session.statements
        if "UPDATE external_governance_buckets" in statement
    ]
    assert refund_statements[-1]["refund"] == 75


def test_expired_reservation_releases_budget_before_next_attempt():
    metadata = {
        "artifact_relay_bytes_reserved": 150,
        "artifact_relay_reservations": {
            "expired": {"bytes": 100, "bucket": 500, "expires_at": 100},
            "active": {"bytes": 50, "bucket": 500, "expires_at": 200},
        },
    }

    active, expired = artifact_limits._prune_expired_reservations(
        metadata, now_epoch=101
    )

    assert active == {"active": {"bytes": 50, "bucket": 500, "expires_at": 200}}
    assert expired == ({"bytes": 100, "bucket": 500, "expires_at": 100},)
    assert metadata["artifact_relay_bytes_reserved"] == 50
