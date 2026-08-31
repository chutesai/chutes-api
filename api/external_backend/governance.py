"""Admission and funded-account guardrails for external execution."""

from __future__ import annotations

from dataclasses import dataclass, fields
from datetime import datetime, timedelta, timezone
from decimal import Decimal, InvalidOperation
import math
from collections.abc import Iterable
from typing import Any, Mapping

from sqlalchemy import select, text
from sqlalchemy.ext.asyncio import AsyncSession

from api.config import settings

from .circuit import circuit_is_open
from .metrics import admission_rejections, operation_admissions
from .schemas import (
    ExternalBackendAccount,
    ExternalOperation,
    ExternalOperationMode,
)


class ExternalGovernanceConfigurationError(ValueError):
    """Raised when account governance configuration is malformed."""


class ExternalAdmissionRejected(RuntimeError):
    """Raised before dispatch when a funded-account guardrail is exhausted."""

    def __init__(self, reason: str, detail: str, *, status_code: int = 429) -> None:
        super().__init__(detail)
        self.reason = reason
        self.detail = detail
        self.status_code = status_code


@dataclass(frozen=True, slots=True)
class GovernancePolicy:
    max_active_tasks_per_user: int
    max_active_tasks_per_account: int
    max_active_sync_requests_per_user: int
    max_active_sync_requests_per_account: int
    max_realtime_sessions_per_user: int
    max_realtime_sessions_per_account: int
    max_streams_per_user: int
    max_streams_per_account: int
    max_daily_operations_per_user: int
    max_daily_operations_per_account: int
    max_daily_paygo_usd_per_user: Decimal
    max_daily_paygo_usd_per_account: Decimal
    max_estimated_operation_cost_usd: Decimal
    artifact_requests_per_minute: int
    artifact_max_concurrent_per_user: int
    artifact_max_bytes_per_operation: int
    artifact_max_daily_bytes_per_user: int


@dataclass(frozen=True, slots=True)
class SessionBudgetPolicy:
    """A snapshotted, provider-neutral exposure bound for a live session."""

    minimum_cost_per_second_usd: Decimal
    max_exposure_usd: Decimal
    max_session_seconds: float
    check_interval_seconds: float

    @property
    def admission_exposure(self) -> Decimal:
        return min(
            self.max_exposure_usd,
            self.minimum_cost_per_second_usd
            * Decimal(str(self.check_interval_seconds)),
        )

    def exposure(self, observed_paygo: Decimal, elapsed_seconds: float) -> Decimal:
        if not math.isfinite(elapsed_seconds):
            return self.max_exposure_usd + self.minimum_cost_per_second_usd
        elapsed = max(float(elapsed_seconds), self.check_interval_seconds)
        return max(
            _money(observed_paygo),
            self.minimum_cost_per_second_usd * Decimal(str(elapsed)),
        )

    def snapshot(self) -> dict[str, Any]:
        return {
            "minimum_cost_per_second_usd": str(self.minimum_cost_per_second_usd),
            "max_exposure_usd": str(self.max_exposure_usd),
            "max_session_seconds": self.max_session_seconds,
            "check_interval_seconds": self.check_interval_seconds,
        }


def _positive_int(value: Any, name: str, *, maximum: int) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or not 1 <= value <= maximum
    ):
        raise ExternalGovernanceConfigurationError(
            f"connection_config.governance.{name} must be an integer between 1 and {maximum}"
        )
    return value


def _positive_decimal(value: Any, name: str, *, maximum: str) -> Decimal:
    if isinstance(value, bool):
        raise ExternalGovernanceConfigurationError(
            f"connection_config.governance.{name} must be a positive number"
        )
    try:
        result = Decimal(str(value))
        ceiling = Decimal(maximum)
    except (InvalidOperation, ValueError, TypeError) as exc:
        raise ExternalGovernanceConfigurationError(
            f"connection_config.governance.{name} must be a positive number"
        ) from exc
    if not result.is_finite() or result <= 0 or result > ceiling:
        raise ExternalGovernanceConfigurationError(
            f"connection_config.governance.{name} must be greater than zero and at most {maximum}"
        )
    return result


_INT_LIMITS = {
    "max_active_tasks_per_user": 10000,
    "max_active_tasks_per_account": 1000000,
    "max_active_sync_requests_per_user": 10000,
    "max_active_sync_requests_per_account": 1000000,
    "max_realtime_sessions_per_user": 10000,
    "max_realtime_sessions_per_account": 1000000,
    "max_streams_per_user": 10000,
    "max_streams_per_account": 1000000,
    "max_daily_operations_per_user": 10000000,
    "max_daily_operations_per_account": 100000000,
    "artifact_requests_per_minute": 100000,
    "artifact_max_concurrent_per_user": 1000,
    "artifact_max_bytes_per_operation": 1024**5,
    "artifact_max_daily_bytes_per_user": 1024**5,
}
_DECIMAL_LIMITS = {
    "max_daily_paygo_usd_per_user": "1000000000",
    "max_daily_paygo_usd_per_account": "1000000000",
    "max_estimated_operation_cost_usd": "1000000000",
}
_POLICY_FIELDS = frozenset({*_INT_LIMITS, *_DECIMAL_LIMITS})


def validate_governance_config(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ExternalGovernanceConfigurationError(
            "connection_config.governance must be an object"
        )
    unknown = sorted(str(key) for key in value if key not in _POLICY_FIELDS)
    if unknown:
        raise ExternalGovernanceConfigurationError(
            "connection_config.governance contains unsupported fields: "
            + ", ".join(unknown)
        )
    result: dict[str, Any] = {}
    for name, maximum in _INT_LIMITS.items():
        if name in value:
            result[name] = _positive_int(value[name], name, maximum=maximum)
    for name, maximum in _DECIMAL_LIMITS.items():
        if name in value:
            result[name] = _positive_decimal(value[name], name, maximum=maximum)
    return result


def _global_policy() -> GovernancePolicy:
    raw = {
        "max_active_tasks_per_user": settings.external_max_active_tasks_per_user,
        "max_active_tasks_per_account": settings.external_max_active_tasks_per_account,
        "max_active_sync_requests_per_user": settings.external_max_active_sync_requests_per_user,
        "max_active_sync_requests_per_account": settings.external_max_active_sync_requests_per_account,
        "max_realtime_sessions_per_user": settings.external_max_realtime_sessions_per_user,
        "max_realtime_sessions_per_account": settings.external_max_realtime_sessions_per_account,
        "max_streams_per_user": settings.external_max_streams_per_user,
        "max_streams_per_account": settings.external_max_streams_per_account,
        "max_daily_operations_per_user": settings.external_max_daily_operations_per_user,
        "max_daily_operations_per_account": settings.external_max_daily_operations_per_account,
        "max_daily_paygo_usd_per_user": settings.external_max_daily_paygo_usd_per_user,
        "max_daily_paygo_usd_per_account": settings.external_max_daily_paygo_usd_per_account,
        "max_estimated_operation_cost_usd": settings.external_max_estimated_operation_cost_usd,
        "artifact_requests_per_minute": settings.external_artifact_requests_per_minute,
        "artifact_max_concurrent_per_user": settings.external_artifact_max_concurrent_per_user,
        "artifact_max_bytes_per_operation": settings.external_artifact_max_bytes_per_operation,
        "artifact_max_daily_bytes_per_user": settings.external_artifact_max_daily_bytes_per_user,
    }
    validated = validate_governance_config(raw)
    return GovernancePolicy(**validated)


def governance_policy(connection_config: Mapping[str, Any] | None) -> GovernancePolicy:
    """Compile account limits, never allowing them to exceed process ceilings."""

    global_policy = _global_policy()
    connection = connection_config or {}
    if not isinstance(connection, Mapping):
        raise ExternalGovernanceConfigurationError(
            "connection_config must be an object"
        )
    configured = validate_governance_config(connection.get("governance"))
    values: dict[str, Any] = {}
    for item in fields(GovernancePolicy):
        ceiling = getattr(global_policy, item.name)
        override = configured.get(item.name)
        values[item.name] = ceiling if override is None else min(ceiling, override)
    return GovernancePolicy(**values)


def _money(value: Any) -> Decimal:
    try:
        parsed = Decimal(str(value))
    except (InvalidOperation, ValueError, TypeError):
        return Decimal(0)
    return parsed if parsed.is_finite() and parsed > 0 else Decimal(0)


def _optional_amount(value: Any) -> Decimal | None:
    try:
        parsed = Decimal(str(value))
    except (InvalidOperation, ValueError, TypeError):
        return None
    return parsed if parsed.is_finite() and parsed >= 0 else None


_SESSION_BUDGET_FIELDS = frozenset(
    {
        "minimum_cost_per_second_usd",
        "max_exposure_usd",
        "check_interval_seconds",
    }
)
_MAX_SESSION_BUDGET_CHECK_INTERVAL_SECONDS = 5.0
_MIN_SESSION_BUDGET_CHECK_INTERVAL_SECONDS = 0.1


def compile_session_budget(
    connection_config: Mapping[str, Any] | None,
    operation_config: Mapping[str, Any] | None,
    *,
    max_session_seconds: float,
) -> SessionBudgetPolicy:
    """Compile a non-zero exposure schedule for stream/realtime execution.

    Route authors can supply a more conservative provider-specific floor. When
    they do not, the floor is synthesized from the route's hard duration and
    funded-account per-operation ceiling, so an unmetered connection can never
    run indefinitely at a zero risk estimate.
    """

    if (
        isinstance(max_session_seconds, bool)
        or not isinstance(max_session_seconds, (int, float))
        or not math.isfinite(float(max_session_seconds))
        or max_session_seconds <= 0
        or max_session_seconds > 86400
    ):
        raise ExternalGovernanceConfigurationError(
            "session maximum duration must be between 0 and 86400 seconds"
        )
    operation = operation_config or {}
    if not isinstance(operation, Mapping):
        raise ExternalGovernanceConfigurationError("operation_config must be an object")
    configured = operation.get("session_budget")
    if configured is None:
        configured = {}
    if not isinstance(configured, Mapping):
        raise ExternalGovernanceConfigurationError(
            "operation_config.session_budget must be an object"
        )
    unknown = sorted(
        str(key) for key in configured if key not in _SESSION_BUDGET_FIELDS
    )
    if unknown:
        raise ExternalGovernanceConfigurationError(
            "operation_config.session_budget contains unsupported fields: "
            + ", ".join(unknown)
        )

    policy = governance_policy(connection_config)
    max_exposure = policy.max_estimated_operation_cost_usd
    if "max_exposure_usd" in configured:
        max_exposure = _positive_decimal(
            configured["max_exposure_usd"],
            "session_budget.max_exposure_usd",
            maximum="1000000000",
        )
        if max_exposure > policy.max_estimated_operation_cost_usd:
            raise ExternalGovernanceConfigurationError(
                "session_budget.max_exposure_usd exceeds the account operation ceiling"
            )

    duration = float(max_session_seconds)
    synthesized_floor = max_exposure / Decimal(str(duration))
    configured_floor = Decimal(0)
    if "minimum_cost_per_second_usd" in configured:
        configured_floor = _positive_decimal(
            configured["minimum_cost_per_second_usd"],
            "session_budget.minimum_cost_per_second_usd",
            maximum="1000000000",
        )
    minimum_cost = max(synthesized_floor, configured_floor)

    check_interval = min(
        _MAX_SESSION_BUDGET_CHECK_INTERVAL_SECONDS,
        duration,
    )
    if "check_interval_seconds" in configured:
        raw_interval = configured["check_interval_seconds"]
        if (
            isinstance(raw_interval, bool)
            or not isinstance(raw_interval, (int, float))
            or not math.isfinite(float(raw_interval))
            or raw_interval < _MIN_SESSION_BUDGET_CHECK_INTERVAL_SECONDS
            or raw_interval > _MAX_SESSION_BUDGET_CHECK_INTERVAL_SECONDS
        ):
            raise ExternalGovernanceConfigurationError(
                "session_budget.check_interval_seconds must be between 0.1 and 5"
            )
        check_interval = min(float(raw_interval), duration)

    if minimum_cost * Decimal(str(check_interval)) > max_exposure:
        raise ExternalGovernanceConfigurationError(
            "session_budget cost floor over its first check interval exceeds "
            "max_exposure_usd"
        )

    return SessionBudgetPolicy(
        minimum_cost_per_second_usd=minimum_cost,
        max_exposure_usd=max_exposure,
        max_session_seconds=duration,
        check_interval_seconds=check_interval,
    )


def session_budget_from_metadata(
    settlement_metadata: Mapping[str, Any] | None,
) -> SessionBudgetPolicy | None:
    """Parse the immutable session budget attached at admission, failing closed."""

    if not isinstance(settlement_metadata, Mapping):
        return None
    raw = settlement_metadata.get("session_budget")
    if not isinstance(raw, Mapping):
        return None
    try:
        floor = Decimal(str(raw["minimum_cost_per_second_usd"]))
        maximum = Decimal(str(raw["max_exposure_usd"]))
        duration = float(raw["max_session_seconds"])
        interval = float(raw["check_interval_seconds"])
    except (InvalidOperation, KeyError, TypeError, ValueError):
        return None
    if (
        not floor.is_finite()
        or floor <= 0
        or not maximum.is_finite()
        or maximum <= 0
        or not math.isfinite(duration)
        or not 0 < duration <= 86400
        or not math.isfinite(interval)
        or not _MIN_SESSION_BUDGET_CHECK_INTERVAL_SECONDS
        <= interval
        <= min(_MAX_SESSION_BUDGET_CHECK_INTERVAL_SECONDS, duration)
        or floor * Decimal(str(interval)) > maximum
        or floor * Decimal(str(duration)) < maximum
    ):
        return None
    return SessionBudgetPolicy(
        minimum_cost_per_second_usd=floor,
        max_exposure_usd=maximum,
        max_session_seconds=duration,
        check_interval_seconds=interval,
    )


def session_exposure_estimate(
    settlement_metadata: Mapping[str, Any] | None,
    observed_paygo: Decimal,
    *,
    elapsed_seconds: float,
) -> Decimal:
    policy = session_budget_from_metadata(settlement_metadata)
    if policy is None:
        return _money(observed_paygo)
    return policy.exposure(observed_paygo, elapsed_seconds)


_ENSURE_GOVERNANCE_SCOPE = text(
    r"""
    INSERT INTO external_governance_state (scope_type, scope_id)
    VALUES (:scope_type, :scope_id)
    ON CONFLICT (scope_type, scope_id) DO NOTHING
    """
)

_LOCK_GOVERNANCE_SCOPE = text(
    r"""
    SELECT scope_id
    FROM external_governance_state
    WHERE scope_type = :scope_type AND scope_id = :scope_id
    FOR UPDATE
    """
)

_PRUNE_GOVERNANCE_BUCKETS = text(
    r"""
    DELETE FROM external_governance_buckets
    WHERE bucket_start < date_trunc(
        'minute', clock_timestamp() - INTERVAL '24 hours'
    )
      AND (
          (scope_type = 'user' AND scope_id = :user_id)
          OR (scope_type = 'account' AND scope_id = :account_id)
      )
    """
)

_GOVERNANCE_ROLLUP_TOTALS = text(
    r"""
    WITH requested(scope_type, scope_id) AS (
        VALUES ('user'::varchar, :user_id), ('account'::varchar, :account_id)
    ), states AS (
        SELECT
            requested.scope_type,
            COALESCE(state.active_tasks, 0) AS active_tasks,
            COALESCE(state.active_sync_requests, 0) AS active_sync_requests,
            COALESCE(state.active_realtime, 0) AS active_realtime,
            COALESCE(state.active_streams, 0) AS active_streams,
            GREATEST(
                COALESCE(state.unresolved_charge, 0) - :exclude_charge,
                0::numeric
            ) AS unresolved_charge
        FROM requested
        LEFT JOIN external_governance_state AS state
          ON state.scope_type = requested.scope_type
         AND state.scope_id = requested.scope_id
    ), buckets AS (
        SELECT
            requested.scope_type,
            COALESCE(SUM(bucket.operation_count), 0) AS operation_count,
            GREATEST(
                COALESCE(SUM(bucket.unresolved_paygo), 0::numeric)
                    - :exclude_paygo,
                0::numeric
            ) AS unresolved_paygo,
            COALESCE(SUM(bucket.settled_paygo), 0::numeric) AS settled_paygo
        FROM requested
        LEFT JOIN external_governance_buckets AS bucket
          ON bucket.scope_type = requested.scope_type
         AND bucket.scope_id = requested.scope_id
         AND bucket.bucket_start >= date_trunc('minute', :since)
        GROUP BY requested.scope_type
    ), totals AS (
        SELECT
            states.scope_type,
            states.active_tasks,
            states.active_sync_requests,
            states.active_realtime,
            states.active_streams,
            buckets.operation_count,
            buckets.unresolved_paygo + buckets.settled_paygo AS spend,
            states.unresolved_charge AS outstanding
        FROM states
        JOIN buckets USING (scope_type)
    )
    SELECT
        COALESCE(MAX(active_tasks) FILTER (WHERE scope_type = 'account'), 0)
            AS account_tasks,
        COALESCE(MAX(active_tasks) FILTER (WHERE scope_type = 'user'), 0)
            AS user_tasks,
        COALESCE(MAX(active_sync_requests) FILTER (WHERE scope_type = 'account'), 0)
            AS account_sync_requests,
        COALESCE(MAX(active_sync_requests) FILTER (WHERE scope_type = 'user'), 0)
            AS user_sync_requests,
        COALESCE(MAX(active_realtime) FILTER (WHERE scope_type = 'account'), 0)
            AS account_realtime,
        COALESCE(MAX(active_realtime) FILTER (WHERE scope_type = 'user'), 0)
            AS user_realtime,
        COALESCE(MAX(active_streams) FILTER (WHERE scope_type = 'account'), 0)
            AS account_streams,
        COALESCE(MAX(active_streams) FILTER (WHERE scope_type = 'user'), 0)
            AS user_streams,
        COALESCE(MAX(operation_count) FILTER (WHERE scope_type = 'account'), 0)
            AS account_operations,
        COALESCE(MAX(operation_count) FILTER (WHERE scope_type = 'user'), 0)
            AS user_operations,
        COALESCE(MAX(spend) FILTER (WHERE scope_type = 'account'), 0::numeric)
            AS account_spend,
        COALESCE(MAX(spend) FILTER (WHERE scope_type = 'user'), 0::numeric)
            AS user_spend,
        COALESCE(MAX(outstanding) FILTER (WHERE scope_type = 'account'), 0::numeric)
            AS account_outstanding,
        COALESCE(MAX(outstanding) FILTER (WHERE scope_type = 'user'), 0::numeric)
            AS user_outstanding
    FROM totals
    """
)


_AUTHORITATIVE_EFFECTIVE_BALANCE = text(
    r"""
    SELECT
        COALESCE(users.balance, 0)::numeric
        - COALESCE((
            SELECT SUM(
                GREATEST(
                    EXTRACT(EPOCH FROM (
                        LEAST(instances.stop_billing_at, CURRENT_TIMESTAMP)
                        - instances.activated_at
                    )),
                    0
                ) / 3600.0 * instances.hourly_rate
            )
            FROM instances
            WHERE instances.billed_to = users.user_id
              AND instances.activated_at IS NOT NULL
              AND instances.stop_billing_at > CURRENT_TIMESTAMP
        ), 0)::numeric AS effective_balance
    FROM users
    WHERE users.user_id = :user_id
    FOR UPDATE OF users
    """
)


async def _authoritative_effective_balance(
    db: AsyncSession, user_id: str
) -> Decimal | None:
    """Lock and read the current stored balance less live private-instance cost.

    ``user_current_balance`` is a minute-refreshed materialized view.  External
    settlement debits ``users.balance`` immediately, so using that view here
    would reopen already-spent credit until its next refresh.
    """

    value = (
        await db.execute(
            _AUTHORITATIVE_EFFECTIVE_BALANCE,
            {"user_id": user_id},
        )
    ).scalar_one_or_none()
    if value is None:
        return None
    try:
        parsed = Decimal(str(value))
    except (InvalidOperation, ValueError, TypeError):
        return None
    return parsed if parsed.is_finite() else None


async def _lock_governance_scope(
    db: AsyncSession, *, account_id: str, user_id: str
) -> None:
    """Lock compact counters in the same user-then-account order as the trigger."""

    await lock_governance_state_rows(db, user_ids=(user_id,), account_ids=(account_id,))


async def lock_governance_state_rows(
    db: AsyncSession,
    *,
    user_ids: Iterable[str | None],
    account_ids: Iterable[str | None],
) -> None:
    """Acquire distinct governance rows in one process-wide deterministic order."""

    scopes = [
        *(
            ("user", scope_id)
            for scope_id in sorted({item for item in user_ids if item})
        ),
        *(
            ("account", scope_id)
            for scope_id in sorted({item for item in account_ids if item})
        ),
    ]
    for scope_type, scope_id in scopes:
        parameters = {"scope_type": scope_type, "scope_id": scope_id}
        await db.execute(_ENSURE_GOVERNANCE_SCOPE, parameters)
        result = await db.execute(_LOCK_GOVERNANCE_SCOPE, parameters)
        result.all()


def _stored_operation_paygo(
    *,
    status: str,
    settlement_status: str,
    settlement_metadata: Mapping[str, Any] | None,
) -> Decimal:
    metadata = settlement_metadata if isinstance(settlement_metadata, Mapping) else {}
    if settlement_status == "not_billable":
        return Decimal(0)
    result = metadata.get("result")
    if isinstance(result, Mapping):
        for name in ("paygo_amount", "amount"):
            amount = _optional_amount(result.get(name))
            if amount is not None:
                return amount
    observed = _optional_amount(metadata.get("observed_cost_estimate"))
    if observed is not None:
        return observed
    if status in {"pending", "submitted", "running"} or settlement_status in {
        "pending",
        "failed",
        "quarantined",
    }:
        return _money(metadata.get("admission_cost_estimate"))
    return Decimal(0)


async def _governance_totals(
    db: AsyncSession,
    *,
    account_id: str,
    user_id: str,
    since: datetime,
    exclude_paygo: Decimal = Decimal(0),
    exclude_charge: Decimal = Decimal(0),
) -> Mapping[str, Any]:
    result = await db.execute(
        _GOVERNANCE_ROLLUP_TOTALS,
        {
            "account_id": account_id,
            "user_id": user_id,
            "since": since,
            "exclude_paygo": _money(exclude_paygo),
            "exclude_charge": _money(exclude_charge),
        },
    )
    return result.mappings().one()


def _reject(reason: str, detail: str, *, status_code: int = 429) -> None:
    admission_rejections.labels(reason=reason).inc()
    raise ExternalAdmissionRejected(reason, detail, status_code=status_code)


async def enforce_external_admission(
    db: AsyncSession,
    *,
    account_id: str,
    user_id: str,
    operation_mode: ExternalOperationMode,
    connection_config: Mapping[str, Any] | None,
    estimated_paygo: Decimal,
    free_invocation: bool,
    balance_exempt: bool = False,
    now: datetime | None = None,
) -> GovernancePolicy:
    """Serialize and enforce account/user task, session, request, and spend caps."""

    policy = governance_policy(connection_config)
    if await circuit_is_open(account_id):
        _reject(
            "account_circuit_open",
            "External execution is temporarily unavailable.",
            status_code=503,
        )
    current = now or datetime.now(timezone.utc)
    if current.tzinfo is None:
        current = current.replace(tzinfo=timezone.utc)
    estimate = _money(estimated_paygo)
    balance_exempt = bool(balance_exempt or free_invocation)
    if estimate > policy.max_estimated_operation_cost_usd:
        _reject(
            "operation_cost_limit",
            "The requested external operation exceeds its configured cost limit.",
            status_code=402,
        )
    balance: Decimal | None = None
    if not balance_exempt:
        balance = await _authoritative_effective_balance(db, user_id)
        if balance is None or estimate > _money(balance):
            _reject(
                "operation_balance",
                "The available balance is insufficient for the known operation cost.",
                status_code=402,
            )

    # Trigger-maintained rows replace daily-volume scans and make admission's
    # check-and-insert decision atomic across API replicas. The user balance is
    # locked first to match settlement; counters then lock user before account,
    # which matches the operation trigger and prevents lock-order inversions.
    await _lock_governance_scope(db, account_id=account_id, user_id=user_id)
    since = current - timedelta(hours=24)
    await db.execute(
        _PRUNE_GOVERNANCE_BUCKETS,
        {"account_id": account_id, "user_id": user_id},
    )
    totals = await _governance_totals(
        db, account_id=account_id, user_id=user_id, since=since
    )
    account_tasks = int(totals["account_tasks"] or 0)
    user_tasks = int(totals["user_tasks"] or 0)
    account_sync_requests = int(totals["account_sync_requests"] or 0)
    user_sync_requests = int(totals["user_sync_requests"] or 0)
    account_realtime = int(totals["account_realtime"] or 0)
    user_realtime = int(totals["user_realtime"] or 0)
    account_streams = int(totals["account_streams"] or 0)
    user_streams = int(totals["user_streams"] or 0)
    account_operations = int(totals["account_operations"] or 0)
    user_operations = int(totals["user_operations"] or 0)
    account_spend = _money(totals["account_spend"])
    user_spend = _money(totals["user_spend"])
    user_outstanding = _money(totals["user_outstanding"])

    if operation_mode is ExternalOperationMode.TASK:
        if user_tasks >= policy.max_active_tasks_per_user:
            _reject(
                "user_task_concurrency", "Too many external tasks are already active."
            )
        if account_tasks >= policy.max_active_tasks_per_account:
            _reject(
                "account_task_concurrency",
                "External task capacity is temporarily full.",
            )
    if operation_mode is ExternalOperationMode.SYNC:
        if user_sync_requests >= policy.max_active_sync_requests_per_user:
            _reject(
                "user_sync_concurrency",
                "Too many external synchronous requests are already active.",
            )
        if account_sync_requests >= policy.max_active_sync_requests_per_account:
            _reject(
                "account_sync_concurrency",
                "External synchronous request capacity is temporarily full.",
            )
    if operation_mode is ExternalOperationMode.REALTIME:
        if user_realtime >= policy.max_realtime_sessions_per_user:
            _reject(
                "user_realtime_concurrency",
                "Too many external realtime sessions are already active.",
            )
        if account_realtime >= policy.max_realtime_sessions_per_account:
            _reject(
                "account_realtime_concurrency",
                "External realtime capacity is temporarily full.",
            )
    if operation_mode is ExternalOperationMode.STREAM:
        if user_streams >= policy.max_streams_per_user:
            _reject(
                "user_stream_concurrency",
                "Too many external response streams are already active.",
            )
        if account_streams >= policy.max_streams_per_account:
            _reject(
                "account_stream_concurrency",
                "External streaming capacity is temporarily full.",
            )
    if user_operations >= policy.max_daily_operations_per_user:
        _reject(
            "user_daily_operations", "The daily external operation limit was reached."
        )
    if account_operations >= policy.max_daily_operations_per_account:
        _reject(
            "account_daily_operations", "External account capacity is temporarily full."
        )
    if not balance_exempt and user_outstanding + estimate > _money(balance):
        _reject(
            "operation_balance",
            "The available balance is insufficient for active external work.",
            status_code=402,
        )
    if user_spend + estimate > policy.max_daily_paygo_usd_per_user:
        _reject(
            "user_spend", "The daily external spend limit was reached.", status_code=402
        )
    if account_spend + estimate > policy.max_daily_paygo_usd_per_account:
        _reject(
            "account_spend",
            "The external account spend limit was reached.",
            status_code=503,
        )
    operation_admissions.labels(mode=operation_mode.value, outcome="accepted").inc()
    return policy


async def running_budget_available(
    db: AsyncSession,
    *,
    operation_id: str,
    estimated_paygo: Decimal,
    now: datetime | None = None,
) -> tuple[bool, str | None]:
    """Check observed session cost against balance and rolling funded-account caps."""

    estimate = _money(estimated_paygo)
    current = now or datetime.now(timezone.utc)
    if current.tzinfo is None:
        current = current.replace(tzinfo=timezone.utc)
    row = (
        await db.execute(
            select(
                ExternalOperation.user_id,
                ExternalOperation.account_id,
                ExternalOperation.operation_mode,
                ExternalOperation.status,
                ExternalOperation.settlement_status,
                ExternalOperation.settlement_metadata,
                ExternalOperation.created_at,
                ExternalOperation.started_at,
                ExternalBackendAccount.connection_config,
            )
            .join(
                ExternalBackendAccount,
                ExternalBackendAccount.account_id == ExternalOperation.account_id,
            )
            .where(ExternalOperation.operation_id == operation_id)
        )
    ).one_or_none()
    if (
        row is None
        or not isinstance(row.user_id, str)
        or not isinstance(row.account_id, str)
    ):
        return False, "operation_unavailable"
    metadata = (
        row.settlement_metadata if isinstance(row.settlement_metadata, Mapping) else {}
    )
    if metadata.get("cancel_requested") is True:
        return False, "cancel_requested"
    policy = governance_policy(row.connection_config)
    session_budget = session_budget_from_metadata(metadata)
    operation_mode = getattr(row, "operation_mode", None)
    if operation_mode in {
        ExternalOperationMode.STREAM.value,
        ExternalOperationMode.REALTIME.value,
    }:
        if session_budget is None:
            admission_rejections.labels(reason="runtime_budget_unconfigured").inc()
            return False, "budget_unavailable"
        started_at = getattr(row, "started_at", None) or getattr(
            row, "created_at", current
        )
        if not isinstance(started_at, datetime):
            admission_rejections.labels(reason="runtime_budget_unconfigured").inc()
            return False, "budget_unavailable"
        if started_at.tzinfo is None:
            started_at = started_at.replace(tzinfo=timezone.utc)
        elapsed = max(0.0, (current - started_at).total_seconds())
        estimate = session_budget.exposure(estimate, elapsed)
        if estimate > session_budget.max_exposure_usd:
            admission_rejections.labels(reason="runtime_session_exposure_limit").inc()
            return False, "session_exposure_limit"
    if estimate > policy.max_estimated_operation_cost_usd:
        admission_rejections.labels(reason="runtime_operation_cost_limit").inc()
        return False, "operation_cost_limit"

    pricing = metadata.get("pricing")
    balance_exempt = isinstance(pricing, Mapping) and bool(
        pricing.get("balance_exempt") or pricing.get("free_invocation")
    )
    effective_balance: Decimal | None = None
    if not balance_exempt:
        effective_balance = await _authoritative_effective_balance(db, row.user_id)
        if effective_balance is None:
            admission_rejections.labels(reason="runtime_balance").inc()
            return False, "balance"
    await _lock_governance_scope(db, account_id=row.account_id, user_id=row.user_id)

    stored_paygo = _stored_operation_paygo(
        status=str(getattr(row, "status", "running")),
        settlement_status=str(getattr(row, "settlement_status", "pending")),
        settlement_metadata=metadata,
    )
    stored_charge = (
        Decimal(0)
        if isinstance(pricing, Mapping)
        and (
            pricing.get("free_invocation") is True
            or pricing.get("balance_exempt") is True
        )
        else stored_paygo
    )
    since = current - timedelta(hours=24)
    totals = await _governance_totals(
        db,
        account_id=row.account_id,
        user_id=row.user_id,
        since=since,
        exclude_paygo=stored_paygo,
        exclude_charge=stored_charge,
    )
    account_spend = _money(totals["account_spend"])
    user_spend = _money(totals["user_spend"])
    user_outstanding = _money(totals["user_outstanding"])
    if user_spend + estimate > policy.max_daily_paygo_usd_per_user:
        admission_rejections.labels(reason="runtime_user_spend").inc()
        return False, "user_spend"
    if account_spend + estimate > policy.max_daily_paygo_usd_per_account:
        admission_rejections.labels(reason="runtime_account_spend").inc()
        return False, "account_spend"
    if not balance_exempt:
        if user_outstanding + estimate > _money(effective_balance):
            admission_rejections.labels(reason="runtime_balance").inc()
            return False, "balance"
    return True, None


__all__ = [
    "ExternalAdmissionRejected",
    "ExternalGovernanceConfigurationError",
    "GovernancePolicy",
    "SessionBudgetPolicy",
    "compile_session_budget",
    "enforce_external_admission",
    "governance_policy",
    "running_budget_available",
    "session_budget_from_metadata",
    "session_exposure_estimate",
    "validate_governance_config",
]
