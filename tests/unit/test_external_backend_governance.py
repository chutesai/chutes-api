from datetime import datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, call

import pytest
from pydantic import ValidationError

import api.external_backend.artifact_limits as artifact_limits
import api.external_backend.circuit as circuit
import api.external_backend.governance as governance
from api.config import Settings
from api.external_backend.governance import ExternalAdmissionRejected
from api.external_backend.operation_lifecycle import UsageBudgetMonitor
from api.external_backend.schemas import ExternalOperationMode


class _Rows:
    def __init__(self, rows=(), *, one=None, scalar=None):
        self._rows = rows
        self._one = one
        self._scalar = scalar

    def all(self):
        return list(self._rows)

    def mappings(self):
        return self

    def one(self):
        return self._one

    def one_or_none(self):
        return self._one

    def scalar_one_or_none(self):
        return self._scalar


class _AdmissionSession:
    def __init__(self, totals, *, balance=Decimal("1")):
        self.totals = totals
        self.balance = balance

    async def execute(self, statement, parameters=None, *_args, **_kwargs):
        if isinstance(parameters, dict) and "since" in parameters:
            return _Rows(one=self.totals)
        if "FROM users" in str(statement):
            return _Rows(scalar=self.balance)
        if isinstance(self.totals, SimpleNamespace):
            return _Rows(one=self.totals)
        return _Rows()


def _totals(**overrides):
    values = {
        "account_tasks": 0,
        "user_tasks": 0,
        "account_sync_requests": 0,
        "user_sync_requests": 0,
        "account_realtime": 0,
        "user_realtime": 0,
        "account_streams": 0,
        "user_streams": 0,
        "account_operations": 0,
        "user_operations": 0,
        "account_spend": 0,
        "user_spend": 0,
        "account_outstanding": 0,
        "user_outstanding": 0,
    }
    values.update(overrides)
    return values


def test_session_budget_synthesizes_a_nonzero_floor_and_hard_exposure_bound():
    policy = governance.compile_session_budget(
        {"governance": {"max_estimated_operation_cost_usd": "2"}},
        {},
        max_session_seconds=20,
    )

    assert policy.minimum_cost_per_second_usd == Decimal("0.1")
    assert policy.admission_exposure == Decimal("0.5")
    assert policy.exposure(Decimal(0), 10) == Decimal("1.0")
    assert (
        governance.session_budget_from_metadata({"session_budget": policy.snapshot()})
        == policy
    )


def test_explicit_session_floor_can_only_make_the_synthesized_bound_stricter():
    policy = governance.compile_session_budget(
        {"governance": {"max_estimated_operation_cost_usd": "2"}},
        {
            "session_budget": {
                "minimum_cost_per_second_usd": "0.25",
                "max_exposure_usd": "1",
                "check_interval_seconds": 2,
            }
        },
        max_session_seconds=10,
    )

    assert policy.minimum_cost_per_second_usd == Decimal("0.25")
    assert policy.admission_exposure == Decimal("0.50")
    assert policy.exposure(Decimal(0), 5) > policy.max_exposure_usd


def test_session_budget_rejects_exposure_larger_than_cap_before_first_check():
    with pytest.raises(
        governance.ExternalGovernanceConfigurationError,
        match="first check interval exceeds max_exposure_usd",
    ):
        governance.compile_session_budget(
            {},
            {
                "session_budget": {
                    "minimum_cost_per_second_usd": "100",
                    "max_exposure_usd": "1",
                    "check_interval_seconds": 0.1,
                }
            },
            max_session_seconds=10,
        )


@pytest.mark.asyncio
async def test_runtime_budget_uses_elapsed_floor_when_usage_is_unobservable(
    monkeypatch,
):
    now = datetime.now(timezone.utc)
    policy = governance.compile_session_budget(
        {"governance": {"max_daily_paygo_usd_per_user": "1"}},
        {"session_budget": {"max_exposure_usd": "1"}},
        max_session_seconds=10,
    )
    metadata = {
        "pricing": {"free_invocation": True},
        "session_budget": policy.snapshot(),
        "admission_cost_estimate": "0.5",
    }
    row = SimpleNamespace(
        user_id="user-id",
        account_id="account-id",
        operation_mode="stream",
        status="running",
        settlement_status="pending",
        settlement_metadata=metadata,
        created_at=now - timedelta(seconds=6),
        started_at=now - timedelta(seconds=6),
        connection_config={"governance": {"max_daily_paygo_usd_per_user": "1"}},
    )

    class Session:
        async def execute(self, *_args, **_kwargs):
            return _Rows(one=row)

    monkeypatch.setattr(governance, "_lock_governance_scope", AsyncMock())
    monkeypatch.setattr(
        governance,
        "_governance_totals",
        AsyncMock(return_value=_totals(user_spend=Decimal("0.5"))),
    )

    available, reason = await governance.running_budget_available(
        Session(),
        operation_id="operation-id",
        estimated_paygo=Decimal(0),
        now=now,
    )

    assert available is False
    assert reason == "user_spend"


@pytest.mark.asyncio
async def test_runtime_budget_honors_persisted_session_cancellation():
    row = SimpleNamespace(
        user_id="user-id",
        account_id="account-id",
        settlement_metadata={"cancel_requested": True},
        connection_config={},
    )

    available, reason = await governance.running_budget_available(
        _AdmissionSession(row),
        operation_id="operation-id",
        estimated_paygo=Decimal(0),
    )

    assert available is False
    assert reason == "cancel_requested"


@pytest.mark.asyncio
async def test_runtime_budget_excludes_current_operation_from_both_rollups(
    monkeypatch,
):
    now = datetime.now(timezone.utc)
    session_budget = governance.compile_session_budget({}, {}, max_session_seconds=10)
    row = SimpleNamespace(
        user_id="user-id",
        account_id="account-id",
        operation_mode="stream",
        status="running",
        settlement_status="pending",
        settlement_metadata={
            "observed_cost_estimate": "0.75",
            "pricing": {"free_invocation": False, "balance_exempt": False},
            "session_budget": session_budget.snapshot(),
        },
        created_at=now,
        started_at=now,
        connection_config={},
    )
    captured = {}

    async def totals(_db, **kwargs):
        captured.update(kwargs)
        return _totals()

    monkeypatch.setattr(governance, "_lock_governance_scope", AsyncMock())
    monkeypatch.setattr(
        governance,
        "_authoritative_effective_balance",
        AsyncMock(return_value=Decimal("100")),
    )
    monkeypatch.setattr(governance, "_governance_totals", totals)

    available, reason = await governance.running_budget_available(
        _AdmissionSession(row),
        operation_id="operation-id",
        estimated_paygo=Decimal("0.80"),
        now=now,
    )

    assert available is True
    assert reason is None
    assert captured["exclude_paygo"] == Decimal("0.75")
    assert captured["exclude_charge"] == Decimal("0.75")


def test_governance_admission_reads_compact_rollups_not_operation_history():
    query = str(governance._GOVERNANCE_ROLLUP_TOTALS)

    assert "external_governance_state" in query
    assert "external_governance_buckets" in query
    assert "external_operations" not in query


def test_explicit_zero_settlement_amount_releases_synthetic_session_exposure():
    assert (
        governance._stored_operation_paygo(
            status="succeeded",
            settlement_status="settled",
            settlement_metadata={
                "result": {"paygo_amount": "0"},
                "observed_cost_estimate": "10",
            },
        )
        == 0
    )


def test_rollup_migration_tracks_quarantined_exposure_and_artifact_bytes():
    migration = (
        Path(__file__).parents[2]
        / "api/migrations/20260830121000_external_governance_rollups.sql"
    ).read_text()

    assert "'pending', 'failed', 'quarantined'" in migration
    assert "artifact_relay_bytes BIGINT NOT NULL DEFAULT 0" in migration
    assert "unresolved_paygo NUMERIC NOT NULL DEFAULT 0" in migration
    assert "trg_external_governance_operation" in migration
    assert "trg_external_governance_operation_update" in migration
    assert "AFTER UPDATE OF" in migration
    assert "paygo_amount') IS NOT NULL" in migration
    assert "Lease acquisition/renewal" in migration
    assert "new_unresolved_paygo - old_unresolved_paygo" in migration
    assert "balance_exempt" in migration
    assert "GREATEST(operation_count + operation_delta, 0)" in migration


def test_governance_pruning_uses_database_time_and_daily_spend_uses_buckets():
    prune = str(governance._PRUNE_GOVERNANCE_BUCKETS)
    totals = str(governance._GOVERNANCE_ROLLUP_TOTALS)

    assert "clock_timestamp() - INTERVAL '24 hours'" in prune
    assert "buckets.unresolved_paygo + buckets.settled_paygo AS spend" in totals
    assert "SUM(bucket.unresolved_paygo)" in totals
    assert "- :exclude_paygo" in totals


@pytest.mark.asyncio
async def test_admission_locks_balance_before_trigger_owned_governance_rows(
    monkeypatch,
):
    statements: list[str] = []

    class Session:
        async def execute(self, statement, parameters=None, *_args, **_kwargs):
            sql = str(statement)
            statements.append(sql)
            if "FROM users" in sql:
                return _Rows(scalar=Decimal("10"))
            if isinstance(parameters, dict) and "since" in parameters:
                return _Rows(one=_totals())
            return _Rows()

    monkeypatch.setattr(governance, "circuit_is_open", AsyncMock(return_value=False))

    await governance.enforce_external_admission(
        Session(),
        account_id="account-id",
        user_id="user-id",
        operation_mode=ExternalOperationMode.STREAM,
        connection_config={},
        estimated_paygo=Decimal("0.1"),
        free_invocation=False,
    )

    balance_index = next(
        index for index, sql in enumerate(statements) if "FROM users" in sql
    )
    state_index = next(
        index
        for index, sql in enumerate(statements)
        if "external_governance_state" in sql
    )
    assert balance_index < state_index


@pytest.mark.asyncio
async def test_realtime_account_concurrency_is_enforced_independently(monkeypatch):
    async def circuit_closed(_account_id):
        return False

    monkeypatch.setattr(governance, "circuit_is_open", circuit_closed)
    totals = _totals(account_realtime=1, account_operations=1)

    with pytest.raises(ExternalAdmissionRejected) as error:
        await governance.enforce_external_admission(
            _AdmissionSession(totals),
            account_id="account-id",
            user_id="user-id",
            operation_mode=ExternalOperationMode.REALTIME,
            connection_config={"governance": {"max_realtime_sessions_per_account": 1}},
            estimated_paygo=Decimal(0),
            free_invocation=False,
        )

    assert error.value.reason == "account_realtime_concurrency"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("mode", "totals", "configured", "reason"),
    [
        (
            ExternalOperationMode.TASK,
            _totals(user_tasks=1),
            {"max_active_tasks_per_user": 1},
            "user_task_concurrency",
        ),
        (
            ExternalOperationMode.STREAM,
            _totals(account_streams=1),
            {"max_streams_per_account": 1},
            "account_stream_concurrency",
        ),
        (
            ExternalOperationMode.SYNC,
            _totals(user_sync_requests=1),
            {"max_active_sync_requests_per_user": 1},
            "user_sync_concurrency",
        ),
    ],
)
async def test_active_work_is_bounded_by_user_and_account(
    monkeypatch, mode, totals, configured, reason
):
    monkeypatch.setattr(governance, "circuit_is_open", AsyncMock(return_value=False))

    with pytest.raises(ExternalAdmissionRejected) as error:
        await governance.enforce_external_admission(
            _AdmissionSession(totals),
            account_id="account-id",
            user_id="user-id",
            operation_mode=mode,
            connection_config={"governance": configured},
            estimated_paygo=Decimal(0),
            free_invocation=False,
        )

    assert error.value.reason == reason


@pytest.mark.asyncio
async def test_free_invocations_remain_subject_to_funded_account_spend_cap(
    monkeypatch,
):
    monkeypatch.setattr(governance, "circuit_is_open", AsyncMock(return_value=False))

    with pytest.raises(ExternalAdmissionRejected) as error:
        await governance.enforce_external_admission(
            _AdmissionSession(_totals(account_spend=Decimal("0.99"))),
            account_id="account-id",
            user_id="user-id",
            operation_mode=ExternalOperationMode.SYNC,
            connection_config={
                "governance": {"max_daily_paygo_usd_per_account": Decimal("1")}
            },
            estimated_paygo=Decimal("0.02"),
            free_invocation=True,
        )

    assert error.value.reason == "account_spend"


@pytest.mark.asyncio
async def test_known_synchronous_cost_cannot_exceed_paygo_balance(monkeypatch):
    monkeypatch.setattr(governance, "circuit_is_open", AsyncMock(return_value=False))

    with pytest.raises(ExternalAdmissionRejected) as error:
        await governance.enforce_external_admission(
            _AdmissionSession(_totals(), balance=Decimal("1")),
            account_id="account-id",
            user_id="user-id",
            operation_mode=ExternalOperationMode.SYNC,
            connection_config={},
            estimated_paygo=Decimal("2"),
            free_invocation=False,
        )

    assert error.value.reason == "operation_balance"


@pytest.mark.asyncio
async def test_active_paygo_work_is_reserved_against_balance(monkeypatch):
    monkeypatch.setattr(governance, "circuit_is_open", AsyncMock(return_value=False))

    with pytest.raises(ExternalAdmissionRejected) as error:
        await governance.enforce_external_admission(
            _AdmissionSession(
                _totals(user_outstanding=Decimal("0.75")),
                balance=Decimal("1"),
            ),
            account_id="account-id",
            user_id="user-id",
            operation_mode=ExternalOperationMode.SYNC,
            connection_config={},
            estimated_paygo=Decimal("0.50"),
            free_invocation=False,
        )

    assert error.value.reason == "operation_balance"


@pytest.mark.asyncio
async def test_invoice_billing_is_not_rejected_by_prepaid_balance(monkeypatch):
    monkeypatch.setattr(governance, "circuit_is_open", AsyncMock(return_value=False))

    policy = await governance.enforce_external_admission(
        _AdmissionSession(_totals()),
        account_id="account-id",
        user_id="user-id",
        operation_mode=ExternalOperationMode.SYNC,
        connection_config={},
        estimated_paygo=Decimal("2"),
        free_invocation=False,
        balance_exempt=True,
    )

    assert policy.max_active_sync_requests_per_user >= 1


@pytest.mark.asyncio
async def test_running_cost_cannot_cross_per_operation_ceiling():
    row = SimpleNamespace(
        user_id="user-id",
        account_id="account-id",
        settlement_metadata={"pricing": {}},
        connection_config={"governance": {"max_estimated_operation_cost_usd": 1}},
    )
    available, reason = await governance.running_budget_available(
        _AdmissionSession(row),
        operation_id="operation-id",
        estimated_paygo=Decimal("1.01"),
    )

    assert available is False
    assert reason == "operation_cost_limit"


@pytest.mark.asyncio
async def test_running_cost_uses_current_stored_balance_not_materialized_view():
    row = SimpleNamespace(
        user_id="user-id",
        account_id="account-id",
        settlement_metadata={"pricing": {}},
        connection_config={},
    )

    class Session:
        async def execute(self, statement, parameters=None, *_args, **_kwargs):
            sql = str(statement)
            if isinstance(parameters, dict) and "since" in parameters:
                return _Rows(one=_totals())
            if "FROM users" in sql:
                assert "user_current_balance" not in sql
                assert "FOR UPDATE OF users" in sql
                return _Rows(scalar=Decimal("0.25"))
            return _Rows(one=row)

    available, reason = await governance.running_budget_available(
        Session(),
        operation_id="operation-id",
        estimated_paygo=Decimal("0.50"),
    )

    assert available is False
    assert reason == "balance"


@pytest.mark.asyncio
async def test_circuit_backend_failure_is_fail_open(monkeypatch):
    raw_redis = SimpleNamespace(
        get=AsyncMock(side_effect=OSError("unavailable")),
        delete=AsyncMock(side_effect=OSError("unavailable")),
        eval=AsyncMock(side_effect=OSError("unavailable")),
    )
    redis = SimpleNamespace(
        client=raw_redis,
        get=AsyncMock(return_value=None),
        delete=AsyncMock(return_value=None),
        eval=AsyncMock(return_value=None),
    )
    metric = MagicMock()
    metric_child = metric.labels.return_value
    monkeypatch.setattr(
        circuit,
        "settings",
        SimpleNamespace(
            redis_client=redis,
            external_circuit_auth_failure_threshold=3,
            external_circuit_service_failure_threshold=10,
            external_circuit_cooldown_seconds=300,
        ),
    )
    monkeypatch.setattr(circuit, "circuit_events", metric)

    assert await circuit.circuit_is_open("account-id") is False
    await circuit.record_upstream_result("account-id", status_code=200)
    await circuit.record_upstream_result("account-id", status_code=401)

    raw_redis.get.assert_awaited_once()
    raw_redis.delete.assert_awaited_once()
    raw_redis.eval.assert_awaited_once()
    redis.get.assert_not_awaited()
    redis.delete.assert_not_awaited()
    redis.eval.assert_not_awaited()
    assert metric.labels.call_args_list == [
        call(reason="backend", action="error"),
        call(reason="backend", action="error"),
        call(reason="backend", action="error"),
    ]
    assert metric_child.inc.call_count == 3


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("external_circuit_auth_failure_threshold", 0),
        ("external_circuit_service_failure_threshold", 1001),
        ("external_circuit_cooldown_seconds", 9),
    ],
)
def test_circuit_configuration_is_validated_at_process_startup(field, value):
    with pytest.raises(ValidationError):
        Settings(**{field: value})


@pytest.mark.asyncio
async def test_circuit_response_recording_uses_startup_validated_config(monkeypatch):
    raw_redis = SimpleNamespace(delete=AsyncMock())
    monkeypatch.setattr(
        circuit,
        "settings",
        SimpleNamespace(
            redis_client=SimpleNamespace(client=raw_redis),
            external_circuit_auth_failure_threshold=0,
            external_circuit_service_failure_threshold=0,
            external_circuit_cooldown_seconds=0,
        ),
    )

    await circuit.record_upstream_result("account-id", status_code=200)

    raw_redis.delete.assert_awaited_once()


@pytest.mark.asyncio
async def test_repeated_budget_backend_failures_close_long_session():
    closed = AsyncMock()
    check = AsyncMock(side_effect=OSError("database unavailable"))
    monitor = UsageBudgetMonitor(
        operation_id="operation-id",
        read_usage=lambda: {"requests": "1"},
        check_usage=check,
        on_exceeded=closed,
        interval_seconds=0.001,
        max_check_failures=3,
    )
    task = monitor.start()

    await task

    assert check.await_count == 3
    assert monitor.exceeded is True
    assert monitor.reason == "budget_unavailable"
    closed.assert_awaited_once_with("budget_unavailable")


@pytest.mark.asyncio
async def test_artifact_concurrency_lease_covers_configured_transport_timeout(
    monkeypatch,
):
    evaluate = AsyncMock(return_value=b"ok")
    redis = SimpleNamespace(client=SimpleNamespace(eval=evaluate), zrem=AsyncMock())
    monkeypatch.setattr(
        artifact_limits,
        "settings",
        SimpleNamespace(
            redis_client=redis,
            external_max_active_tasks_per_user=4,
            external_max_active_tasks_per_account=256,
            external_max_realtime_sessions_per_user=2,
            external_max_realtime_sessions_per_account=64,
            external_max_streams_per_user=4,
            external_max_streams_per_account=128,
            external_max_daily_operations_per_user=1000,
            external_max_daily_operations_per_account=100000,
            external_max_daily_paygo_usd_per_user=25,
            external_max_daily_paygo_usd_per_account=1000,
            external_max_estimated_operation_cost_usd=50,
            external_artifact_requests_per_minute=60,
            external_artifact_max_concurrent_per_user=3,
            external_artifact_max_bytes_per_operation=10 * 1024**3,
            external_artifact_max_daily_bytes_per_user=50 * 1024**3,
        ),
    )

    lease = await artifact_limits.acquire_artifact_relay(
        "user-id", {}, lease_seconds=3600
    )

    args = evaluate.await_args.args
    assert args[-1] == 3600
    assert args[5] - args[4] == pytest.approx(3600)
    assert "active_ttl < tonumber(ARGV[6])" in args[0]
    await lease.release()
