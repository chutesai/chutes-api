from contextlib import asynccontextmanager
from copy import deepcopy
from datetime import UTC, datetime
from decimal import Decimal
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from api.external_backend import billing_outbox
from api.external_backend.schemas import ExternalSettlementStatus


class _ScalarResult:
    def __init__(self, value):
        self.value = value

    def scalar_one_or_none(self):
        return self.value


def _event(**overrides):
    values = {
        "event_id": "external-settlement:operation-id",
        "operation_id": "operation-id",
        "user_id": "user-id",
        "chute_id": "chute-id",
        "app_id": "app-id",
        "amount": Decimal("1.25"),
        "paygo_amount": Decimal("1.50"),
        "input_tokens": Decimal("11"),
        "output_tokens": Decimal("7"),
        "cached_tokens": Decimal("3"),
        "compute_time": 5.5,
        "track_task_completion": True,
        "free_invocation": False,
        "increment_invocation_quota": False,
        "occurred_at": datetime(2026, 8, 29, 17, 34, 12, tzinfo=UTC),
        "attempts": 0,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


@pytest.mark.asyncio
async def test_apply_usage_updates_reporting_app_reporting_and_balance():
    statements = []

    class Session:
        async def execute(self, statement, params=None):
            sql = str(statement)
            statements.append((sql, params))
            if "FROM users" in sql:
                return _ScalarResult(1)
            return _ScalarResult(None)

    await billing_outbox._apply_usage(Session(), _event())

    sql = "\n".join(statement for statement, _ in statements)
    assert "INSERT INTO usage_data" in sql
    assert "INSERT INTO app_usage_data" in sql
    assert "UPDATE users SET balance = balance -" in sql
    usage_params = next(
        params
        for statement, params in statements
        if "INSERT INTO usage_data" in statement
    )
    assert usage_params["bucket"] == datetime(2026, 8, 29, 17, tzinfo=UTC)
    assert usage_params["amount"] == Decimal("1.25")


@pytest.mark.asyncio
async def test_apply_usage_preserves_accepted_free_balance_but_records_paygo():
    statements = []

    class Session:
        async def execute(self, statement, params=None):
            sql = str(statement)
            statements.append((sql, params))
            if "FROM users" in sql:
                return _ScalarResult(1)
            return _ScalarResult(None)

    await billing_outbox._apply_usage(
        Session(), _event(app_id=None, amount=Decimal("0"))
    )

    sql = "\n".join(statement for statement, _ in statements)
    assert "INSERT INTO usage_data" in sql
    assert "INSERT INTO app_usage_data" not in sql
    assert "UPDATE users SET balance = balance -" not in sql
    usage_params = next(
        params
        for statement, params in statements
        if "INSERT INTO usage_data" in statement
    )
    assert usage_params["paygo_amount"] == Decimal("1.50")


@pytest.mark.asyncio
async def test_apply_usage_records_complete_zero_charge_without_balance_update():
    statements = []

    class Session:
        async def execute(self, statement, params=None):
            sql = str(statement)
            statements.append((sql, params))
            if "FROM users" in sql:
                return _ScalarResult(1)
            return _ScalarResult(None)

    await billing_outbox._apply_usage(
        Session(),
        _event(
            app_id=None,
            amount=Decimal("0"),
            paygo_amount=Decimal("0"),
            input_tokens=Decimal("0"),
            output_tokens=Decimal("0"),
            cached_tokens=Decimal("0"),
            compute_time=0,
        ),
    )

    sql = "\n".join(statement for statement, _ in statements)
    assert "INSERT INTO usage_data" in sql
    assert "UPDATE users SET balance = balance -" not in sql
    usage_params = next(
        params
        for statement, params in statements
        if "INSERT INTO usage_data" in statement
    )
    assert usage_params["amount"] == 0.0
    assert usage_params["paygo_amount"] == 0.0


@pytest.mark.asyncio
async def test_apply_usage_charges_persisted_amount_after_current_role_change():
    statements = []

    class Session:
        async def execute(self, statement, params=None):
            sql = str(statement)
            statements.append((sql, params))
            if "FROM users" in sql:
                # Delivery deliberately reads only existence while locking the
                # row; current permissions cannot rewrite the accepted charge.
                return _ScalarResult(1)
            return _ScalarResult(None)

    await billing_outbox._apply_usage(Session(), _event(app_id=None))

    sql = "\n".join(statement for statement, _ in statements)
    assert "permissions_bitmask" not in sql
    assert "UPDATE users SET balance = balance -" in sql


@pytest.mark.asyncio
async def test_delivery_crash_rolls_back_charge_settlement_and_ack(monkeypatch):
    state = SimpleNamespace(
        operation=SimpleNamespace(
            operation_id="operation-id",
            settlement_status=ExternalSettlementStatus.PENDING.value,
            settlement_metadata={"settlement_delivery": "pending"},
            settled_at=None,
            next_poll_at=datetime.now(UTC),
        ),
        event=_event(),
        balance=Decimal("10"),
    )

    class Session:
        async def get(self, *_args, **_kwargs):
            return state.operation

        async def execute(self, _statement, _params=None):
            return _ScalarResult(state.event)

        async def delete(self, _event_value):
            state.event = None

    attempts = 0

    async def apply(_session, event):
        nonlocal attempts
        attempts += 1
        state.balance -= event.amount
        if attempts == 1:
            raise OSError("database connection lost before commit")

    @asynccontextmanager
    async def session_factory():
        snapshot = deepcopy(state)
        try:
            yield Session()
        except Exception:
            state.operation = snapshot.operation
            state.event = snapshot.event
            state.balance = snapshot.balance
            raise

    record_failure = AsyncMock()
    monkeypatch.setattr(billing_outbox, "get_session", session_factory)
    monkeypatch.setattr(billing_outbox, "_apply_usage", apply)
    monkeypatch.setattr(billing_outbox, "_record_delivery_failure", record_failure)

    with pytest.raises(OSError):
        await billing_outbox.deliver_external_usage_event("operation-id")

    assert state.balance == Decimal("10")
    assert state.event is not None
    assert state.operation.settlement_status == ExternalSettlementStatus.PENDING.value
    record_failure.assert_awaited_once()

    receipt = await billing_outbox.deliver_external_usage_event("operation-id")

    assert receipt is not None
    assert state.balance == Decimal("8.75")
    assert state.event is None
    assert state.operation.settlement_status == ExternalSettlementStatus.SETTLED.value

    duplicate = await billing_outbox.deliver_external_usage_event("operation-id")
    assert duplicate is None
    assert attempts == 2
    assert state.balance == Decimal("8.75")
