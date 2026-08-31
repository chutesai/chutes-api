"""Leased background processing for externally executed asynchronous operations."""

from __future__ import annotations

import asyncio
import copy
import inspect
import random
import re
import time
import uuid
from collections.abc import Awaitable, Callable, Mapping, Sequence
from contextlib import suppress
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from email.utils import parsedate_to_datetime
from typing import Any, Protocol

from loguru import logger
from sqlalchemy import Select, and_, delete, exists, func, or_, select, text, update
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import joinedload, noload

from api.database import SessionLocal, get_session
from api.external_transport import (
    BodyMode,
    BufferedResponse,
    ExternalExecutor,
    ExternalTransportError,
    JsonBody,
    OutboundRequest,
    ResponseMode,
)
from api.external_transport.header_policy import requires_secret_backing
from api.payment.pricing import NormalizedUsage, UsageValidationError

from .config import ExternalConfigurationError, build_endpoint_profile
from .circuit import record_upstream_result
from .governance import lock_governance_state_rows
from .artifact_policy import normalize_artifact_expiration
from .mapping import (
    ExtractedTask,
    MappingConfigurationError,
    MappingExtractionError,
    PayloadTransform,
    StreamUsageMode,
    TaskMapping,
    UsageMapping,
    ValueRule,
    is_missing_value,
    merge_stream_usage,
    scrub_public_response,
)
from .metrics import (
    governance_bucket_deletions,
    oldest_poll_lag,
    operation_queue_depth,
    retention_deletions,
    settlement_backlog,
    status_class,
    upstream_latency,
    upstream_requests,
)
from .public_urls import artifact_url
from .schemas import (
    ExternalGovernanceBucket,
    ExternalGovernanceState,
    ExternalOperation,
    ExternalOperationMode,
    ExternalOperationStatus,
    ExternalRouteConfig,
    ExternalUsageOutbox,
)
from .task_results import bounded_inline_result, inline_result_limit


_ACTIVE_STATUSES = frozenset(
    {
        ExternalOperationStatus.SUBMITTED.value,
        ExternalOperationStatus.RUNNING.value,
    }
)
_RECOVERABLE_SESSION_MODES = frozenset(
    {
        ExternalOperationMode.STREAM.value,
        ExternalOperationMode.REALTIME.value,
    }
)
_LEGACY_PENDING_RECOVERY_SECONDS = 8 * 86400 + 8 * 300 + 300
_TERMINAL_STATUSES = frozenset(
    {
        ExternalOperationStatus.SUCCEEDED.value,
        ExternalOperationStatus.FAILED.value,
        ExternalOperationStatus.CANCELLED.value,
        ExternalOperationStatus.EXPIRED.value,
    }
)
_SUCCESS_STATUSES = frozenset(range(200, 300))
_DEFAULT_RETRY_STATUSES = frozenset({408, 425, 429, 500, 502, 503, 504})
_DEFAULT_BILLABLE_TERMINAL_STATUSES = frozenset(
    {
        ExternalOperationStatus.FAILED.value,
        ExternalOperationStatus.CANCELLED.value,
    }
)
_MISSING = object()
_HEADER_NAME = re.compile(r"^[!#$%&'*+.^_`|~0-9A-Za-z-]+$")


def _artifact_expiration(value: Any) -> datetime | None:
    if isinstance(value, datetime):
        parsed = value
    elif isinstance(value, str):
        try:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            return None
    else:
        return None
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        return None
    return parsed.astimezone(timezone.utc)


def _retained_artifacts_expired(
    result_descriptor: Mapping[str, Any] | None, *, now: datetime
) -> bool:
    if not isinstance(result_descriptor, Mapping):
        return True
    artifacts = result_descriptor.get("artifacts", [])
    if not isinstance(artifacts, Sequence) or isinstance(artifacts, (str, bytes)):
        return False
    for artifact in artifacts:
        if not isinstance(artifact, Mapping):
            return False
        expiration = _artifact_expiration(artifact.get("expires_at"))
        if expiration is None or expiration > now:
            return False
    return True


class PollingConfigurationError(ValueError):
    """Raised when an asynchronous operation profile cannot be executed safely."""


class OperationLeaseLost(RuntimeError):
    """Raised when another worker owns an operation before an update is persisted."""


class OperationNotCancellable(RuntimeError):
    """Raised when an operation has no configured cancellation endpoint."""


class SessionFactory(Protocol):
    def __call__(self) -> Any: ...


TerminalHook = Callable[["TerminalOperationEvent"], None | Awaitable[None]]
ExecutorFactory = Callable[["AccountSnapshot"], ExternalExecutor]
ArtifactURLFactory = Callable[[str, int], str]
Clock = Callable[[], datetime]


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _object(value: object, name: str) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise PollingConfigurationError(f"{name} must be an object")
    return dict(value)


def _strict_keys(value: Mapping[str, Any], allowed: frozenset[str], name: str) -> None:
    unknown = sorted(str(key) for key in value if key not in allowed)
    if unknown:
        raise PollingConfigurationError(
            f"{name} contains unsupported fields: {', '.join(unknown)}"
        )


def _bounded_float(
    value: Any,
    *,
    name: str,
    default: float,
    minimum: float,
    maximum: float,
) -> float:
    if value is None:
        return default
    if isinstance(value, bool):
        raise PollingConfigurationError(f"{name} must be a number")
    try:
        parsed = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise PollingConfigurationError(f"{name} must be a number") from exc
    if not minimum <= parsed <= maximum:
        raise PollingConfigurationError(
            f"{name} must be between {minimum:g} and {maximum:g}"
        )
    return parsed


def _bounded_int(
    value: Any,
    *,
    name: str,
    default: int,
    minimum: int,
    maximum: int,
) -> int:
    parsed = _bounded_float(
        value,
        name=name,
        default=float(default),
        minimum=float(minimum),
        maximum=float(maximum),
    )
    if not parsed.is_integer():
        raise PollingConfigurationError(f"{name} must be an integer")
    return int(parsed)


def _status_codes(
    value: Any,
    *,
    name: str,
    default: frozenset[int],
) -> frozenset[int]:
    if value is None:
        return default
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise PollingConfigurationError(f"{name} must be an array of status codes")
    result: set[int] = set()
    for item in value:
        if (
            isinstance(item, bool)
            or not isinstance(item, int)
            or not 100 <= item <= 599
        ):
            raise PollingConfigurationError(f"{name} must be an array of status codes")
        result.add(item)
    if not result:
        raise PollingConfigurationError(f"{name} cannot be empty")
    return frozenset(result)


def _header_names(
    value: Any, *, name: str, default: Sequence[str] = ()
) -> tuple[str, ...]:
    if value is None:
        value = default
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise PollingConfigurationError(f"{name} must be an array of header names")
    result: list[str] = []
    for item in value:
        if not isinstance(item, str) or not _HEADER_NAME.fullmatch(item):
            raise PollingConfigurationError(f"{name} contains an invalid header name")
        normalized = item.lower()
        if normalized not in result:
            result.append(normalized)
    if len(result) > 32:
        raise PollingConfigurationError(f"{name} contains too many header names")
    return tuple(result)


def _operation_statuses(
    value: Any, *, name: str, default: frozenset[str] = frozenset()
) -> frozenset[str]:
    if value is None:
        return default
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise PollingConfigurationError(f"{name} must be an array")
    result = frozenset(str(item).lower() for item in value)
    if not result.issubset(_TERMINAL_STATUSES):
        raise PollingConfigurationError(f"{name} contains an unsupported status")
    return result


@dataclass(frozen=True, slots=True)
class AccountSnapshot:
    """The non-secret account material needed to build a request profile."""

    account_id: str
    user_id: str
    base_url: str
    credential_references: Mapping[str, str]
    auth_header_templates: tuple[Mapping[str, Any], ...]
    connection_config: Mapping[str, Any]


@dataclass(frozen=True, slots=True)
class WorkerSettings:
    batch_size: int = 16
    concurrency: int = 8
    lease_seconds: float = 60.0
    idle_seconds: float = 1.0
    shutdown_timeout_seconds: float = 30.0
    settlement_reconcile_interval_seconds: float = 5.0
    maintenance_interval_seconds: float = 60.0
    retention_days: int = 90
    retention_batch_size: int = 1000
    settlement_batch_size: int = 64

    def __post_init__(self) -> None:
        if not 1 <= self.batch_size <= 1000:
            raise PollingConfigurationError("batch_size must be between 1 and 1000")
        if not 1 <= self.concurrency <= 256:
            raise PollingConfigurationError("concurrency must be between 1 and 256")
        if not 3 <= self.lease_seconds <= 3600:
            raise PollingConfigurationError("lease_seconds must be between 3 and 3600")
        if not 0.05 <= self.idle_seconds <= 60:
            raise PollingConfigurationError("idle_seconds must be between 0.05 and 60")
        if not 0.1 <= self.shutdown_timeout_seconds <= 300:
            raise PollingConfigurationError(
                "shutdown_timeout_seconds must be between 0.1 and 300"
            )
        if not 0.5 <= self.settlement_reconcile_interval_seconds <= 3600:
            raise PollingConfigurationError(
                "settlement_reconcile_interval_seconds must be between 0.5 and 3600"
            )
        if not 5 <= self.maintenance_interval_seconds <= 86400:
            raise PollingConfigurationError(
                "maintenance_interval_seconds must be between 5 and 86400"
            )
        if not 1 <= self.retention_days <= 3650:
            raise PollingConfigurationError("retention_days must be between 1 and 3650")
        if not 1 <= self.retention_batch_size <= 10000:
            raise PollingConfigurationError(
                "retention_batch_size must be between 1 and 10000"
            )
        if not 1 <= self.settlement_batch_size <= 1000:
            raise PollingConfigurationError(
                "settlement_batch_size must be between 1 and 1000"
            )


@dataclass(frozen=True, slots=True)
class BackoffPolicy:
    interval_seconds: float = 2.0
    multiplier: float = 1.0
    maximum_seconds: float = 30.0
    jitter_fraction: float = 0.1

    def delay(self, attempt: int, random_value: float) -> float:
        base = min(
            self.maximum_seconds,
            self.interval_seconds * (self.multiplier ** max(0, attempt - 1)),
        )
        spread = base * self.jitter_fraction
        return max(0.0, base - spread + (2 * spread * random_value))


@dataclass(frozen=True, slots=True)
class PollRetryPolicy:
    statuses: frozenset[int] = _DEFAULT_RETRY_STATUSES
    max_attempts: int = 0
    transport_errors: bool = True
    retry_after_headers: tuple[str, ...] = ("retry-after",)

    def delay(
        self,
        response: BufferedResponse,
        backoff: BackoffPolicy,
        attempt: int,
        now: datetime,
        random_value: float,
    ) -> float:
        headers = response.private_headers
        for name in self.retry_after_headers:
            value = headers.get(name)
            if not value:
                continue
            try:
                return min(backoff.maximum_seconds, max(0.0, float(value)))
            except (TypeError, ValueError, OverflowError):
                try:
                    parsed = parsedate_to_datetime(value)
                    if parsed.tzinfo is None:
                        parsed = parsed.replace(tzinfo=timezone.utc)
                    return min(
                        backoff.maximum_seconds,
                        max(0.0, (parsed - now).total_seconds()),
                    )
                except (TypeError, ValueError, OverflowError):
                    continue
        return backoff.delay(attempt, random_value)


@dataclass(frozen=True, slots=True)
class MappedFields:
    fields: tuple[tuple[str, ValueRule], ...] = ()

    @classmethod
    def from_config(
        cls,
        value: Any,
        *,
        name: str,
        reject_sensitive_names: bool = False,
    ) -> MappedFields:
        if value is None:
            return cls()
        if not isinstance(value, Mapping):
            raise PollingConfigurationError(f"{name} must be an object")
        fields: list[tuple[str, ValueRule]] = []
        for raw_name, config in value.items():
            field_name = str(raw_name)
            if not field_name or any(
                character in field_name for character in "\r\n\x00"
            ):
                raise PollingConfigurationError(
                    f"{name} contains an invalid field name"
                )
            if reject_sensitive_names and requires_secret_backing(field_name):
                raise PollingConfigurationError(
                    f"{name} cannot contain credential-like names"
                )
            try:
                rule = ValueRule.from_config(config, default_source="context")
            except MappingConfigurationError as exc:
                raise PollingConfigurationError(
                    f"{name} contains an invalid mapping"
                ) from exc
            fields.append((field_name, rule))
        return cls(tuple(fields))

    def evaluate(self, sources: Mapping[str, Any]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for name, rule in self.fields:
            value = rule.evaluate(sources)
            if not is_missing_value(value):
                result[name] = value
        return result


_REQUEST_KEYS = frozenset({"path_parameters", "query", "body", "body_transform"})


@dataclass(frozen=True, slots=True)
class EndpointRequest:
    path_parameters: MappedFields = field(default_factory=MappedFields)
    query: MappedFields = field(default_factory=MappedFields)
    body: Any = _MISSING
    body_transform: PayloadTransform = field(default_factory=PayloadTransform)

    @classmethod
    def from_config(cls, value: Any, *, name: str) -> EndpointRequest:
        config = _object(value, name)
        _strict_keys(config, _REQUEST_KEYS, name)
        try:
            transform = PayloadTransform.from_config(config.get("body_transform"))
        except MappingConfigurationError as exc:
            raise PollingConfigurationError(
                f"{name}.body_transform is invalid"
            ) from exc
        return cls(
            path_parameters=MappedFields.from_config(
                config.get("path_parameters"), name=f"{name}.path_parameters"
            ),
            query=MappedFields.from_config(
                config.get("query"),
                name=f"{name}.query",
                reject_sensitive_names=True,
            ),
            body=copy.deepcopy(config["body"]) if "body" in config else _MISSING,
            body_transform=transform,
        )

    def build(self, profile: Any, sources: Mapping[str, Any]) -> OutboundRequest:
        path_parameters = self.path_parameters.evaluate(sources)
        if "{task_id}" in profile.path_template and "task_id" not in path_parameters:
            task_id = sources["context"].get("task_id")
            if not isinstance(task_id, (str, int)) or isinstance(task_id, bool):
                raise MappingExtractionError("task endpoint requires a task id")
            path_parameters["task_id"] = task_id
        for value in path_parameters.values():
            if isinstance(value, bool) or not isinstance(value, (str, int)):
                raise MappingExtractionError("mapped path parameter is invalid")
        query = self.query.evaluate(sources)
        for value in query.values():
            if value is None or isinstance(value, (str, int, float, bool)):
                continue
            if (
                isinstance(value, Sequence)
                and not isinstance(value, (str, bytes))
                and all(
                    item is None or isinstance(item, (str, int, float, bool))
                    for item in value
                )
            ):
                continue
            raise MappingExtractionError("mapped query parameter is invalid")

        if self.body is _MISSING:
            body_value: Any = {}
        else:
            body_value = copy.deepcopy(self.body)
        if self.body_transform != PayloadTransform():
            body_value = self.body_transform.apply(
                body_value,
                request=sources.get("request"),
                response=sources.get("response"),
                context=sources.get("context"),
            )
        if profile.body_mode is BodyMode.NONE:
            if self.body is not _MISSING or self.body_transform != PayloadTransform():
                raise PollingConfigurationError(
                    "body configuration is not allowed for a bodyless endpoint"
                )
            body = None
        elif profile.body_mode is BodyMode.JSON:
            body = JsonBody(body_value)
        else:
            raise PollingConfigurationError(
                "task lifecycle endpoints support only none and json body modes"
            )
        return OutboundRequest(
            path_parameters=path_parameters,
            query=query,
            headers={},
            body=body,
        )


_CALL_KEYS = frozenset(
    {
        "endpoint",
        "request",
        "response",
        "task",
        "task_mapping",
        "usage",
        "usage_mode",
        "success_statuses",
        "interval_seconds",
        "backoff",
        "retry",
        "billable_statuses",
        "public",
    }
)
_ENDPOINT_KEYS = frozenset(
    {
        "base_url",
        "path_template",
        "method",
        "body_mode",
        "response_mode",
        "static_headers",
        "allowed_request_headers",
        "allowed_response_headers",
        "private_response_headers",
        "timeouts",
        "allowed_hosts",
        "redirects",
        "max_response_bytes",
        "max_sse_event_bytes",
        "stream_chunk_bytes",
    }
)
_RESPONSE_KEYS = frozenset({"task", "usage"})
_BACKOFF_KEYS = frozenset({"multiplier", "maximum_seconds", "jitter_fraction"})
_RETRY_KEYS = frozenset(
    {"statuses", "max_attempts", "transport_errors", "retry_after_headers"}
)


@dataclass(frozen=True, slots=True)
class EndpointCall:
    endpoint: Mapping[str, Any]
    request: EndpointRequest
    task: TaskMapping | None
    usage: UsageMapping | None
    usage_mode: StreamUsageMode
    success_statuses: frozenset[int]
    backoff: BackoffPolicy
    retry: PollRetryPolicy
    billable_statuses: frozenset[str]
    public: Mapping[str, Any]

    @classmethod
    def from_config(
        cls,
        value: Any,
        *,
        name: str,
        fallback_task: Any = None,
        fallback_usage: Any = None,
        fallback_public: Any = None,
        fallback_billable_statuses: Any = None,
        require_task: bool,
    ) -> EndpointCall:
        config = _object(value, name)
        _strict_keys(config, _CALL_KEYS, name)
        endpoint = _object(config.get("endpoint"), f"{name}.endpoint")
        if not endpoint:
            raise PollingConfigurationError(f"{name}.endpoint is required")
        _strict_keys(endpoint, _ENDPOINT_KEYS, f"{name}.endpoint")
        endpoint_method = str(endpoint.get("method") or "GET").upper()
        endpoint.setdefault(
            "body_mode", "none" if endpoint_method in {"GET", "HEAD"} else "json"
        )
        endpoint.setdefault("response_mode", "buffered")
        response = _object(config.get("response"), f"{name}.response")
        _strict_keys(response, _RESPONSE_KEYS, f"{name}.response")
        task_config = (
            config.get("task")
            or config.get("task_mapping")
            or response.get("task")
            or fallback_task
        )
        if require_task and not isinstance(task_config, Mapping):
            raise PollingConfigurationError(f"{name}.task mapping is required")
        try:
            task = (
                TaskMapping.from_config(task_config)
                if task_config is not None
                else None
            )
            usage_config = (
                config.get("usage") or response.get("usage") or fallback_usage
            )
            usage = (
                UsageMapping.from_config(usage_config)
                if usage_config is not None
                else None
            )
        except MappingConfigurationError as exc:
            raise PollingConfigurationError(
                f"{name} response mapping is invalid"
            ) from exc
        if require_task and (task is None or task.status is None):
            raise PollingConfigurationError(f"{name}.task must map a status")
        try:
            usage_mode = StreamUsageMode(config.get("usage_mode", "cumulative"))
        except ValueError as exc:
            raise PollingConfigurationError(f"{name}.usage_mode is invalid") from exc
        if (
            usage is not None
            and usage_mode is StreamUsageMode.DELTA
            and usage.default_requests != 0
        ):
            raise PollingConfigurationError(
                f"{name}.usage must set default_requests to zero for delta observations"
            )

        interval = _bounded_float(
            config.get("interval_seconds"),
            name=f"{name}.interval_seconds",
            default=2.0,
            minimum=0.05,
            maximum=3600,
        )
        backoff_config = _object(config.get("backoff"), f"{name}.backoff")
        _strict_keys(backoff_config, _BACKOFF_KEYS, f"{name}.backoff")
        backoff = BackoffPolicy(
            interval_seconds=interval,
            multiplier=_bounded_float(
                backoff_config.get("multiplier"),
                name=f"{name}.backoff.multiplier",
                default=1.0,
                minimum=1.0,
                maximum=10.0,
            ),
            maximum_seconds=_bounded_float(
                backoff_config.get("maximum_seconds"),
                name=f"{name}.backoff.maximum_seconds",
                default=max(30.0, interval),
                minimum=interval,
                maximum=86400,
            ),
            jitter_fraction=_bounded_float(
                backoff_config.get("jitter_fraction"),
                name=f"{name}.backoff.jitter_fraction",
                default=0.1,
                minimum=0.0,
                maximum=1.0,
            ),
        )
        retry_config = _object(config.get("retry"), f"{name}.retry")
        _strict_keys(retry_config, _RETRY_KEYS, f"{name}.retry")
        transport_errors = retry_config.get("transport_errors", True)
        if not isinstance(transport_errors, bool):
            raise PollingConfigurationError(
                f"{name}.retry.transport_errors must be a boolean"
            )
        retry = PollRetryPolicy(
            statuses=_status_codes(
                retry_config.get("statuses"),
                name=f"{name}.retry.statuses",
                default=_DEFAULT_RETRY_STATUSES,
            ),
            max_attempts=_bounded_int(
                retry_config.get("max_attempts"),
                name=f"{name}.retry.max_attempts",
                default=0,
                minimum=0,
                maximum=1_000_000,
            ),
            transport_errors=transport_errors,
            retry_after_headers=_header_names(
                retry_config.get("retry_after_headers"),
                name=f"{name}.retry.retry_after_headers",
                default=("retry-after",),
            ),
        )
        private_headers = _header_names(
            endpoint.get("private_response_headers"),
            name=f"{name}.endpoint.private_response_headers",
        )
        endpoint["private_response_headers"] = sorted(
            {*private_headers, *retry.retry_after_headers}
        )
        public = _object(config.get("public", fallback_public), f"{name}.public")
        try:
            scrub_public_response({}, public)
        except MappingConfigurationError as exc:
            raise PollingConfigurationError(f"{name}.public is invalid") from exc
        return cls(
            endpoint=copy.deepcopy(endpoint),
            request=EndpointRequest.from_config(
                config.get("request"), name=f"{name}.request"
            ),
            task=task,
            usage=usage,
            usage_mode=usage_mode,
            success_statuses=_status_codes(
                config.get("success_statuses"),
                name=f"{name}.success_statuses",
                default=_SUCCESS_STATUSES,
            ),
            backoff=backoff,
            retry=retry,
            billable_statuses=_operation_statuses(
                (
                    config["billable_statuses"]
                    if "billable_statuses" in config
                    else fallback_billable_statuses
                ),
                name=f"{name}.billable_statuses",
                default=_DEFAULT_BILLABLE_TERMINAL_STATUSES,
            ),
            public=public,
        )


@dataclass(frozen=True, slots=True)
class TaskLifecyclePolicy:
    poll: EndpointCall
    cancel: EndpointCall | None = None

    @classmethod
    def from_route(cls, route: ExternalRouteConfig) -> TaskLifecyclePolicy:
        operation = _object(route.operation_config, "operation_config")
        task_container = _object(operation.get("task"), "operation_config.task")
        if "poll" in task_container or "cancel" in task_container:
            _strict_keys(
                task_container,
                frozenset(
                    {
                        "poll",
                        "cancel",
                        "poll_mapping",
                        "usage",
                        "public",
                        "billable_statuses",
                        "timeout_seconds",
                    }
                ),
                "operation_config.task",
            )
        poll_config = operation.get("poll", task_container.get("poll"))
        if poll_config is None:
            raise PollingConfigurationError("operation_config.poll is required")
        cancel_config = operation.get("cancel", task_container.get("cancel"))
        response = _object(route.response_config, "response_config")
        fallback_task = (
            operation.get("poll_mapping")
            or task_container.get("poll_mapping")
            or operation.get("task_mapping")
            or response.get("task")
        )
        fallback_usage = (
            task_container.get("usage")
            or operation.get("usage")
            or response.get("usage")
        )
        fallback_public = task_container.get("public") or response.get("public")
        if "billable_statuses" in task_container:
            fallback_billable = task_container["billable_statuses"]
        elif "billable_terminal_statuses" in operation:
            fallback_billable = operation["billable_terminal_statuses"]
        else:
            fallback_billable = operation.get("bill_statuses")
        poll = EndpointCall.from_config(
            poll_config,
            name="operation_config.poll",
            fallback_task=fallback_task,
            fallback_usage=fallback_usage,
            fallback_public=fallback_public,
            fallback_billable_statuses=fallback_billable,
            require_task=True,
        )
        cancel = (
            EndpointCall.from_config(
                cancel_config,
                name="operation_config.cancel",
                fallback_task=None,
                fallback_usage=fallback_usage,
                fallback_public=fallback_public,
                fallback_billable_statuses=fallback_billable,
                require_task=False,
            )
            if cancel_config is not None
            else None
        )
        return cls(poll=poll, cancel=cancel)


@dataclass(frozen=True, slots=True)
class LeasedOperation:
    operation_id: str
    lease_token: str
    status: str
    poll_attempts: int
    task_id: str
    route: ExternalRouteConfig
    account: AccountSnapshot
    request_metadata: Mapping[str, Any]
    upstream_metadata: Mapping[str, Any]
    usage: NormalizedUsage | None
    result_descriptor: Mapping[str, Any] | None
    expires_at: datetime | None
    cancel_requested: bool = False
    cancel_requested_at: datetime | str | None = None
    cancel_dispatched: bool = False


@dataclass(frozen=True, slots=True)
class PollOutcome:
    status: str
    upstream_status: str | None
    usage: NormalizedUsage | None
    result_descriptor: Mapping[str, Any] | None
    error: Mapping[str, Any] | None
    next_poll_at: datetime | None
    billable: bool = False
    cancel_dispatched: bool = False

    @property
    def terminal(self) -> bool:
        return self.status in _TERMINAL_STATUSES


@dataclass(frozen=True, slots=True)
class TerminalOperationEvent:
    operation_id: str
    status: str
    usage: NormalizedUsage
    billable: bool
    result_descriptor: Mapping[str, Any] | None
    error: Mapping[str, Any] | None


def build_claim_statement(now: datetime, batch_size: int) -> Select[Any]:
    """Build the lock-skipping query used for one short claim transaction."""

    return (
        select(ExternalOperation)
        .where(
            or_(
                and_(
                    ExternalOperation.operation_mode
                    == ExternalOperationMode.TASK.value,
                    ExternalOperation.status.in_(_ACTIVE_STATUSES),
                    or_(
                        ExternalOperation.next_poll_at.is_(None),
                        ExternalOperation.next_poll_at <= now,
                    ),
                ),
                and_(
                    ExternalOperation.status == ExternalOperationStatus.PENDING.value,
                    or_(
                        ExternalOperation.next_poll_at <= now,
                        and_(
                            ExternalOperation.next_poll_at.is_(None),
                            ExternalOperation.created_at
                            <= now
                            - timedelta(seconds=_LEGACY_PENDING_RECOVERY_SECONDS),
                        ),
                    ),
                ),
                and_(
                    ExternalOperation.operation_mode.in_(_RECOVERABLE_SESSION_MODES),
                    ExternalOperation.status == ExternalOperationStatus.RUNNING.value,
                    ExternalOperation.expires_at.is_not(None),
                    ExternalOperation.expires_at <= now,
                ),
            ),
            or_(
                ExternalOperation.lease_expires_at.is_(None),
                ExternalOperation.lease_expires_at <= now,
            ),
        )
        .order_by(
            ExternalOperation.next_poll_at.asc().nullsfirst(),
            ExternalOperation.created_at,
        )
        .limit(batch_size)
        .with_for_update(skip_locked=True, of=ExternalOperation)
    )


def build_settlement_reconcile_statement(now: datetime, batch_size: int) -> Select[Any]:
    """Lock due pre-price terminal settlements for one retry worker."""

    return (
        select(ExternalOperation)
        .where(
            ExternalOperation.status.in_(_TERMINAL_STATUSES),
            ExternalOperation.settlement_status.in_({"pending", "failed"}),
            ~exists(
                select(ExternalUsageOutbox.event_id).where(
                    ExternalUsageOutbox.operation_id == ExternalOperation.operation_id
                )
            ),
            or_(
                ExternalOperation.next_poll_at.is_(None),
                ExternalOperation.next_poll_at <= now,
            ),
        )
        .order_by(
            ExternalOperation.next_poll_at.asc().nullsfirst(),
            ExternalOperation.finished_at.asc().nullsfirst(),
        )
        .limit(batch_size)
        .with_for_update(skip_locked=True, of=ExternalOperation)
    )


def build_usage_outbox_reconcile_statement(
    now: datetime, batch_size: int
) -> Select[Any]:
    """Lock immutable due charges independently from presentation state."""

    return (
        select(ExternalUsageOutbox)
        .where(
            or_(
                ExternalUsageOutbox.next_attempt_at.is_(None),
                ExternalUsageOutbox.next_attempt_at <= now,
            )
        )
        .order_by(
            ExternalUsageOutbox.next_attempt_at.asc().nullsfirst(),
            ExternalUsageOutbox.created_at,
        )
        .limit(batch_size)
        .with_for_update(skip_locked=True, of=ExternalUsageOutbox)
    )


def build_governance_bucket_prune_statement(
    batch_size: int,
):
    """Delete a batch outside the DB clock's governance window and grace."""

    cutoff = func.date_trunc(
        "minute", func.clock_timestamp() - text("INTERVAL '24 hours 5 minutes'")
    )

    candidates = (
        select(
            ExternalGovernanceBucket.scope_type,
            ExternalGovernanceBucket.scope_id,
            ExternalGovernanceBucket.bucket_start,
        )
        .where(ExternalGovernanceBucket.bucket_start < cutoff)
        .order_by(
            ExternalGovernanceBucket.bucket_start,
            ExternalGovernanceBucket.scope_type,
            ExternalGovernanceBucket.scope_id,
        )
        .limit(batch_size)
        .with_for_update(skip_locked=True, of=ExternalGovernanceBucket)
        .cte("expired_external_governance_buckets")
    )
    return delete(ExternalGovernanceBucket).where(
        exists(
            select(1)
            .select_from(candidates)
            .where(
                candidates.c.scope_type == ExternalGovernanceBucket.scope_type,
                candidates.c.scope_id == ExternalGovernanceBucket.scope_id,
                candidates.c.bucket_start == ExternalGovernanceBucket.bucket_start,
            )
        )
    )


_RECONCILE_GOVERNANCE_SCOPE = text(
    r"""
    WITH recalculated AS (
        SELECT
            COUNT(*) FILTER (
                WHERE operation_mode = 'task'
                  AND status IN ('pending', 'submitted', 'running')
            )::bigint AS active_tasks,
            COUNT(*) FILTER (
                WHERE operation_mode = 'sync'
                  AND status IN ('pending', 'submitted', 'running')
            )::bigint AS active_sync_requests,
            COUNT(*) FILTER (
                WHERE operation_mode = 'realtime'
                  AND status IN ('pending', 'submitted', 'running')
            )::bigint AS active_realtime,
            COUNT(*) FILTER (
                WHERE operation_mode = 'stream'
                  AND status IN ('pending', 'submitted', 'running')
            )::bigint AS active_streams,
            COALESCE(SUM(
                CASE
                    WHEN status IN ('pending', 'submitted', 'running')
                         OR settlement_status IN (
                            'pending', 'failed', 'quarantined'
                         )
                        THEN external_governance_paygo(
                            status, settlement_status, settlement_metadata
                        )
                    ELSE 0::numeric
                END
            ), 0::numeric) AS unresolved_paygo,
            COALESCE(SUM(
                CASE
                    WHEN (
                        status IN ('pending', 'submitted', 'running')
                        OR settlement_status IN (
                            'pending', 'failed', 'quarantined'
                        )
                    )
                    AND settlement_metadata->'pricing'->>'free_invocation'
                        IS DISTINCT FROM 'true'
                    AND settlement_metadata->'pricing'->>'balance_exempt'
                        IS DISTINCT FROM 'true'
                        THEN external_governance_paygo(
                            status, settlement_status, settlement_metadata
                        )
                    ELSE 0::numeric
                END
            ), 0::numeric) AS unresolved_charge
        FROM external_operations
        WHERE (
            (:scope_type = 'user' AND user_id = :scope_id)
            OR (:scope_type = 'account' AND account_id = :scope_id)
        )
          AND (
            status IN ('pending', 'submitted', 'running')
            OR settlement_status IN ('pending', 'failed', 'quarantined')
          )
    )
    UPDATE external_governance_state
    SET active_tasks = recalculated.active_tasks,
        active_sync_requests = recalculated.active_sync_requests,
        active_realtime = recalculated.active_realtime,
        active_streams = recalculated.active_streams,
        unresolved_paygo = recalculated.unresolved_paygo,
        unresolved_charge = recalculated.unresolved_charge,
        updated_at = clock_timestamp()
    FROM recalculated
    WHERE scope_type = :scope_type AND scope_id = :scope_id
    """
)


def build_missing_governance_scope_statement(
    scope_type: str,
    batch_size: int,
) -> Select[Any]:
    """Find active/unresolved scopes whose compact state row was lost."""

    if scope_type == "user":
        scope_id = ExternalOperation.user_id
    elif scope_type == "account":
        scope_id = ExternalOperation.account_id
    else:
        raise ValueError("scope_type must be user or account")
    unresolved = or_(
        ExternalOperation.status.in_({"pending", "submitted", "running"}),
        ExternalOperation.settlement_status.in_({"pending", "failed", "quarantined"}),
    )
    state_exists = exists(
        select(1).where(
            ExternalGovernanceState.scope_type == scope_type,
            ExternalGovernanceState.scope_id == scope_id,
        )
    )
    return (
        select(scope_id.label("scope_id"))
        .where(scope_id.is_not(None), unresolved, ~state_exists)
        .distinct()
        .order_by(scope_id)
        .limit(batch_size)
    )


def _expired_session_billable(operation: ExternalOperation) -> bool:
    """Apply the persisted partial-work policy to a lost accepted session."""

    if operation.operation_mode == ExternalOperationMode.REALTIME.value:
        # RUNNING realtime rows have completed the upstream handshake, so the
        # accepted-work policy bills the usage retained before process loss.
        return True
    try:
        route = ExternalRouteConfig.model_validate(operation.route_snapshot)
        response_config = route.response_config or {}
        if not isinstance(response_config, Mapping):
            return True
        configured = response_config.get("bill_partial_streams", True)
        return configured if isinstance(configured, bool) else True
    except Exception:
        # A corrupt snapshot cannot safely turn known accepted work into a free
        # request. Valid configurations can explicitly opt out above.
        return True


def _expired_session_due(operation: ExternalOperation, now: datetime) -> bool:
    """Return whether an accepted local session needs crash recovery."""

    return (
        operation.operation_mode in _RECOVERABLE_SESSION_MODES
        and operation.status == ExternalOperationStatus.RUNNING.value
        and operation.expires_at is not None
        and operation.expires_at <= now
    )


def _terminalize_expired_session(
    operation: ExternalOperation,
    now: datetime,
) -> TerminalOperationEvent | None:
    """Finalize a non-task RUNNING row only after its persisted hard deadline."""

    if not _expired_session_due(operation, now):
        return None
    try:
        usage = _usage_from_json(operation.usage) or NormalizedUsage(requests=1)
    except MappingExtractionError:
        usage = NormalizedUsage(requests=1)
    billable = _expired_session_billable(operation)
    operation.status = ExternalOperationStatus.FAILED.value
    operation.error = _error("session_recovery_timeout", retryable=True)
    operation.finished_at = now
    operation.last_polled_at = now
    operation.next_poll_at = None
    operation.lease_owner = None
    operation.lease_expires_at = None
    operation.poll_attempts += 1
    settlement = dict(getattr(operation, "settlement_metadata", None) or {})
    settlement["billable"] = billable
    operation.settlement_metadata = settlement
    return TerminalOperationEvent(
        operation_id=operation.operation_id,
        status=ExternalOperationStatus.FAILED.value,
        usage=usage,
        billable=billable,
        result_descriptor=operation.result_descriptor,
        error=operation.error,
    )


def _usage_from_json(value: Mapping[str, Any] | None) -> NormalizedUsage | None:
    if not value:
        return None
    try:
        return NormalizedUsage.from_mapping(value)
    except UsageValidationError as exc:
        raise MappingExtractionError("stored operation usage is invalid") from exc


def _account_snapshot(account: Any) -> AccountSnapshot:
    return AccountSnapshot(
        account_id=account.account_id,
        user_id=account.user_id,
        base_url=account.base_url,
        credential_references=copy.deepcopy(dict(account.credential_references or {})),
        auth_header_templates=tuple(copy.deepcopy(account.auth_header_templates or ())),
        connection_config=copy.deepcopy(dict(account.connection_config or {})),
    )


def _lease_snapshot(operation: ExternalOperation, token: str) -> LeasedOperation:
    if not operation.upstream_operation_id:
        raise MappingExtractionError("active task is missing its remote identity")
    try:
        route = ExternalRouteConfig.model_validate(operation.route_snapshot)
    except Exception as exc:
        raise PollingConfigurationError("stored route snapshot is invalid") from exc
    settlement = dict(operation.settlement_metadata or {})
    cancel_requested = settlement.get("cancel_requested", False)
    if not isinstance(cancel_requested, bool):
        raise MappingExtractionError("stored cancellation state is invalid")
    cancel_requested_at = settlement.get("cancel_requested_at")
    if cancel_requested_at is not None and not isinstance(
        cancel_requested_at, (str, datetime)
    ):
        raise MappingExtractionError("stored cancellation timestamp is invalid")
    cancel_dispatched = settlement.get("cancel_dispatched", False)
    if not isinstance(cancel_dispatched, bool):
        raise MappingExtractionError("stored cancellation dispatch state is invalid")
    return LeasedOperation(
        operation_id=operation.operation_id,
        lease_token=token,
        status=operation.status,
        poll_attempts=operation.poll_attempts,
        task_id=operation.upstream_operation_id,
        route=route,
        account=_account_snapshot(operation.account),
        request_metadata=copy.deepcopy(dict(operation.request_metadata or {})),
        upstream_metadata=copy.deepcopy(dict(operation.upstream_metadata or {})),
        usage=_usage_from_json(operation.usage),
        result_descriptor=copy.deepcopy(operation.result_descriptor),
        expires_at=operation.expires_at,
        cancel_requested=cancel_requested,
        cancel_requested_at=cancel_requested_at,
        cancel_dispatched=cancel_dispatched,
    )


def _pending_recovery_billable(operation: ExternalOperation) -> bool:
    """Apply the snapshotted ambiguous-dispatch policy to a crash orphan."""

    try:
        route = ExternalRouteConfig.model_validate(operation.route_snapshot)
        config = route.operation_config or {}
        if not isinstance(config, Mapping):
            return True
        configured = config.get("bill_ambiguous_transport_errors", False)
        return configured if isinstance(configured, bool) else True
    except Exception:
        return True


def _local_artifact_url(operation_id: str, index: int) -> str:
    return artifact_url(operation_id, index)


def _context(lease: LeasedOperation) -> dict[str, Any]:
    return {
        "task_id": lease.task_id,
        "operation_id": lease.operation_id,
        "resource": lease.route.upstream_resource_id,
        "request_metadata": dict(lease.request_metadata),
        "upstream_metadata": dict(lease.upstream_metadata),
    }


def _sources(lease: LeasedOperation, response: Any = None) -> dict[str, Any]:
    context = _context(lease)
    return {
        "context": context,
        "request": dict(lease.request_metadata),
        "response": response,
        "payload": response,
        "item": None,
    }


def _error(code: str, *, retryable: bool) -> dict[str, Any]:
    return {
        "message": "The asynchronous operation could not be completed.",
        "code": code,
        "retryable": retryable,
        "details": {},
    }


def _contains_value(value: Any, expected: str) -> bool:
    if isinstance(value, Mapping):
        return any(_contains_value(child, expected) for child in value.values())
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return any(_contains_value(child, expected) for child in value)
    return isinstance(value, str) and value == expected


def _replace_scalar(value: Any, old: str, new: str) -> Any:
    if isinstance(value, Mapping):
        return {key: _replace_scalar(child, old, new) for key, child in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return [_replace_scalar(child, old, new) for child in value]
    return new if isinstance(value, str) and value == old else copy.deepcopy(value)


def _result_descriptor(
    lease: LeasedOperation,
    extracted: ExtractedTask,
    *,
    terminal: bool,
    public_rules: Mapping[str, Any],
    artifact_url_factory: ArtifactURLFactory,
    now: datetime,
) -> Mapping[str, Any] | None:
    existing = dict(lease.result_descriptor or {})
    artifacts: list[dict[str, Any]] = []
    for raw_item in existing.get("artifacts", []):
        if not isinstance(raw_item, Mapping) or not raw_item.get("reference"):
            continue
        item = copy.deepcopy(dict(raw_item))
        item["expires_at"] = normalize_artifact_expiration(
            item.get("expires_at"), lease.route.response_config, now=now
        ).isoformat()
        attributes = dict(item.get("attributes") or {})
        if item.get("local_path"):
            attributes.setdefault("local_path", item.pop("local_path"))
        item["attributes"] = attributes
        artifacts.append(item)
    by_reference = {item["reference"]: index for index, item in enumerate(artifacts)}
    for item in extracted.artifacts:
        if item.source_url in by_reference:
            index = by_reference[item.source_url]
        else:
            index = len(artifacts)
            by_reference[item.source_url] = index
            artifacts.append({})
        expires_at = normalize_artifact_expiration(
            item.expires_at, lease.route.response_config, now=now
        ).isoformat()
        attributes = dict(item.metadata)
        attributes["local_path"] = artifact_url_factory(lease.operation_id, index)
        artifacts[index] = {
            "kind": item.kind,
            "reference": item.source_url,
            "content_type": item.content_type,
            "size_bytes": item.size_bytes,
            "expires_at": expires_at,
            "attributes": attributes,
        }

    metadata = copy.deepcopy(dict(existing.get("metadata") or {}))
    persisted_result_limit = inline_result_limit(lease.route.operation_config)
    if extracted.result is not None and persisted_result_limit is not None:
        replacements = {
            source_url: artifact_url_factory(lease.operation_id, index)
            for source_url, index in by_reference.items()
            if _contains_value(extracted.result, source_url)
        }
        metadata["inline_result"] = bounded_inline_result(
            scrub_public_response(
                _replace_scalar(
                    extracted.result,
                    lease.task_id,
                    lease.operation_id,
                ),
                public_rules,
                artifact_urls=replacements,
            ),
            persisted_result_limit,
        )
    if not artifacts and (not metadata or not terminal):
        return None
    return {
        "status": "complete" if terminal else "partial",
        "artifacts": artifacts,
        "metadata": metadata,
    }


def _merge_usage(
    lease: LeasedOperation,
    observed: NormalizedUsage | None,
    mode: StreamUsageMode,
) -> NormalizedUsage | None:
    if observed is None:
        return lease.usage
    previous = lease.usage or NormalizedUsage(requests=1)
    return merge_stream_usage(previous, observed, mode)


def _lifecycle_usage_observation(
    mapping: UsageMapping,
    *,
    response: Any,
    context: Mapping[str, Any],
) -> NormalizedUsage:
    """Extract only values that can change in a polling response.

    Request- and context-derived usage is captured once when the task is
    submitted.  Re-evaluating those fields against the reduced persisted
    request metadata either loses required request values or repeatedly adds
    fixed quantities when a route uses delta observations.
    """

    observation = UsageMapping(
        fields=tuple(
            field
            for field in mapping.fields
            if field.rule.source in {"response", "payload"}
        ),
        default_requests=0,
    )
    return observation.extract(
        request={},
        response=response,
        context=context,
        payload=response,
    )


class ExternalOperationPoller:
    """Claim, poll, and finalize task operations without long database transactions."""

    def __init__(
        self,
        *,
        session_factory: SessionFactory = SessionLocal,
        executor_factory: ExecutorFactory | None = None,
        settlement_hook: TerminalHook | None = None,
        terminal_hooks: Sequence[TerminalHook] = (),
        settings: WorkerSettings | None = None,
        worker_id: str | None = None,
        clock: Clock = _now,
        artifact_url_factory: ArtifactURLFactory = _local_artifact_url,
        random_value: Callable[[], float] = random.random,
    ) -> None:
        self._session_factory = session_factory
        self._executor_factory = executor_factory or _default_executor_factory
        settlement = _settlement_hook if settlement_hook is None else settlement_hook
        self._terminal_hooks = (settlement, *terminal_hooks)
        self._settings = settings or WorkerSettings()
        self._worker_id = worker_id or f"external-poller-{uuid.uuid4()}"
        self._clock = clock
        self._artifact_url_factory = artifact_url_factory
        self._random_value = random_value
        self._stop = asyncio.Event()
        self._runner: asyncio.Task[None] | None = None
        self._last_settlement_reconcile = 0.0
        self._last_maintenance = 0.0

    @property
    def running(self) -> bool:
        return self._runner is not None and not self._runner.done()

    def start(self) -> asyncio.Task[None]:
        """Start the polling loop once in the current event loop."""

        if self.running:
            assert self._runner is not None
            return self._runner
        self._stop = asyncio.Event()
        self._runner = asyncio.create_task(
            self.run(), name=f"external-operation-poller:{self._worker_id}"
        )
        return self._runner

    async def stop(self) -> None:
        """Stop accepting work and allow in-flight requests a bounded grace period."""

        runner = self._runner
        if runner is None:
            return
        self._stop.set()
        try:
            await asyncio.wait_for(
                asyncio.shield(runner),
                timeout=self._settings.shutdown_timeout_seconds,
            )
        except asyncio.TimeoutError:
            runner.cancel()
            with suppress(asyncio.CancelledError):
                await runner
        finally:
            self._runner = None

    async def run(self) -> None:
        """Run until stopped, recovering from isolated claim and operation failures."""

        while not self._stop.is_set():
            try:
                await self._run_periodic_work()
                count = await self.poll_once()
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("External operation polling batch failed")
                count = 0
            if count == 0:
                try:
                    await asyncio.wait_for(
                        self._stop.wait(), timeout=self._settings.idle_seconds
                    )
                except asyncio.TimeoutError:
                    pass

    async def _run_periodic_work(self) -> None:
        monotonic_now = asyncio.get_running_loop().time()
        if (
            monotonic_now - self._last_settlement_reconcile
            >= self._settings.settlement_reconcile_interval_seconds
        ):
            self._last_settlement_reconcile = monotonic_now
            try:
                await self.reconcile_settlements_once()
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("External settlement reconciliation failed")
        if (
            monotonic_now - self._last_maintenance
            >= self._settings.maintenance_interval_seconds
        ):
            self._last_maintenance = monotonic_now
            try:
                await self.refresh_metrics()
                await self.collect_retained_operations()
                await self.collect_expired_governance_buckets()
                await self.reconcile_governance_state_once()
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("External operation maintenance failed")

    async def reconcile_settlements_once(self) -> int:
        """Re-drive terminal pending/failed settlements through an idempotent queue."""

        from .service import settle_operation

        now = self._clock()
        recovery_at = now + timedelta(seconds=self._settings.lease_seconds)
        async with self._session_factory() as session:
            async with session.begin():
                outbox_events = (
                    (
                        await session.execute(
                            build_usage_outbox_reconcile_statement(
                                now, self._settings.settlement_batch_size
                            )
                        )
                    )
                    .scalars()
                    .all()
                )
                outbox_operation_ids = []
                for event in outbox_events:
                    # The outbox's own due timestamp is the authoritative delivery
                    # queue and remains recoverable if this worker dies after claim.
                    event.next_attempt_at = recovery_at
                    outbox_operation_ids.append(event.operation_id)

                remaining = max(
                    0,
                    self._settings.settlement_batch_size - len(outbox_operation_ids),
                )
                operations = (
                    (
                        await session.execute(
                            build_settlement_reconcile_statement(now, remaining)
                        )
                    )
                    .unique()
                    .scalars()
                    .all()
                )
                operation_ids = []
                for operation in operations:
                    # Claim through the existing retry timestamp. A crashed worker
                    # makes the row eligible again at this bounded recovery deadline;
                    # a normal settlement replaces it with success or true backoff.
                    operation.next_poll_at = recovery_at
                    operation_ids.append(operation.operation_id)
        semaphore = asyncio.Semaphore(self._settings.concurrency)

        async def reconcile(operation_id: str) -> None:
            async with semaphore:
                await settle_operation(operation_id)

        await asyncio.gather(
            *(
                reconcile(operation_id)
                for operation_id in (*outbox_operation_ids, *operation_ids)
            )
        )
        return len(outbox_operation_ids) + len(operation_ids)

    async def refresh_metrics(self) -> None:
        """Publish queue depth, lag, and unresolved-settlement gauges."""

        now = self._clock()
        async with self._session_factory() as session:
            queue_rows = (
                await session.execute(
                    select(ExternalOperation.status, func.count())
                    .where(
                        ExternalOperation.operation_mode
                        == ExternalOperationMode.TASK.value,
                        ExternalOperation.status.in_(_ACTIVE_STATUSES),
                    )
                    .group_by(ExternalOperation.status)
                )
            ).all()
            settlement_rows = (
                await session.execute(
                    select(ExternalOperation.settlement_status, func.count())
                    .where(
                        ExternalOperation.status.in_(_TERMINAL_STATUSES),
                        ExternalOperation.settlement_status.in_(
                            {"pending", "failed", "quarantined"}
                        ),
                    )
                    .group_by(ExternalOperation.settlement_status)
                )
            ).all()
            oldest_due = (
                await session.execute(
                    select(func.min(ExternalOperation.next_poll_at)).where(
                        ExternalOperation.status.in_(_ACTIVE_STATUSES),
                        ExternalOperation.next_poll_at.is_not(None),
                        ExternalOperation.next_poll_at <= now,
                    )
                )
            ).scalar_one_or_none()
        queue_counts = dict(queue_rows)
        for state in ("pending", "submitted", "running"):
            operation_queue_depth.labels(status=state).set(queue_counts.get(state, 0))
        settlement_counts = dict(settlement_rows)
        for state in ("pending", "failed", "quarantined"):
            settlement_backlog.labels(status=state).set(settlement_counts.get(state, 0))
        lag = max(0.0, (now - oldest_due).total_seconds()) if oldest_due else 0.0
        oldest_poll_lag.set(lag)

    async def collect_retained_operations(self) -> int:
        """Remove only resolved terminal history after artifacts and retries are stale."""

        now = self._clock()
        cutoff = now - timedelta(days=self._settings.retention_days)
        async with self._session_factory() as session:
            async with session.begin():
                candidates = list(
                    (
                        await session.execute(
                            select(ExternalOperation)
                            .where(
                                ExternalOperation.status.in_(_TERMINAL_STATUSES),
                                ExternalOperation.settlement_status.in_(
                                    {"settled", "not_billable"}
                                ),
                                ExternalOperation.finished_at.is_not(None),
                                ExternalOperation.finished_at < cutoff,
                                or_(
                                    ExternalOperation.expires_at.is_(None),
                                    ExternalOperation.expires_at <= now,
                                ),
                            )
                            .order_by(ExternalOperation.finished_at)
                            .limit(self._settings.retention_batch_size * 4)
                            .with_for_update(skip_locked=True, of=ExternalOperation)
                        )
                    ).scalars()
                )
                candidate_ids = [
                    operation.operation_id
                    for operation in candidates
                    if _retained_artifacts_expired(operation.result_descriptor, now=now)
                ][: self._settings.retention_batch_size]
                candidate_id_set = set(candidate_ids)
                retained = [
                    operation
                    for operation in candidates
                    if operation.operation_id in candidate_id_set
                ]
                await lock_governance_state_rows(
                    session,
                    user_ids=(
                        getattr(operation, "user_id", None) for operation in retained
                    ),
                    account_ids=(
                        getattr(operation, "account_id", None) for operation in retained
                    ),
                )
                count = 0
                for operation_id in candidate_ids:
                    try:
                        # Isolate each delete in a savepoint. An unexpected retained
                        # reference can poison one historical row without rolling
                        # back every otherwise collectable operation in the batch.
                        async with session.begin_nested():
                            result = await session.execute(
                                delete(ExternalOperation).where(
                                    ExternalOperation.operation_id == operation_id,
                                    ExternalOperation.settlement_status.in_(
                                        {"settled", "not_billable"}
                                    ),
                                )
                            )
                            count += max(0, result.rowcount or 0)
                    except IntegrityError:
                        logger.warning(
                            "Skipping externally referenced retained operation {}",
                            operation_id,
                        )
        if count:
            retention_deletions.inc(count)
        return count

    async def collect_expired_governance_buckets(self) -> int:
        """Prune a bounded, lock-skipping batch outside the governance window."""

        async with self._session_factory() as session:
            async with session.begin():
                result = await session.execute(
                    build_governance_bucket_prune_statement(
                        self._settings.retention_batch_size,
                    )
                )
                count = max(0, result.rowcount or 0)
        if count:
            governance_bucket_deletions.inc(count)
        return count

    async def reconcile_governance_state_once(self) -> int:
        """Re-derive a bounded set of hot counters to self-heal rollup drift."""

        count = 0
        async with self._session_factory() as session:
            # One scope per transaction cannot invert two user-state locks with
            # admission, settlement, retention, or emergency bulk mutation.
            # updated_at is advanced by the resync, providing a durable fair cursor.
            for scope_type in ("user", "account"):
                async with session.begin():
                    missing_scope_ids = list(
                        (
                            await session.execute(
                                build_missing_governance_scope_statement(
                                    scope_type,
                                    self._settings.settlement_batch_size,
                                )
                            )
                        )
                        .scalars()
                        .all()
                    )
                processed = set(missing_scope_ids)
                for scope_id in missing_scope_ids:
                    async with session.begin():
                        await lock_governance_state_rows(
                            session,
                            user_ids=(scope_id,) if scope_type == "user" else (),
                            account_ids=(
                                (scope_id,) if scope_type == "account" else ()
                            ),
                        )
                        await session.execute(
                            _RECONCILE_GOVERNANCE_SCOPE,
                            {"scope_type": scope_type, "scope_id": scope_id},
                        )
                        count += 1

                remaining = max(
                    0,
                    self._settings.settlement_batch_size - len(missing_scope_ids),
                )
                for _ in range(remaining):
                    async with session.begin():
                        statement = select(ExternalGovernanceState).where(
                            ExternalGovernanceState.scope_type == scope_type
                        )
                        if processed:
                            statement = statement.where(
                                ExternalGovernanceState.scope_id.not_in(processed)
                            )
                        state = (
                            await session.execute(
                                statement.order_by(
                                    ExternalGovernanceState.updated_at,
                                    ExternalGovernanceState.scope_id,
                                )
                                .limit(1)
                                .with_for_update(
                                    skip_locked=True, of=ExternalGovernanceState
                                )
                            )
                        ).scalar_one_or_none()
                        if state is None:
                            break
                        await session.execute(
                            _RECONCILE_GOVERNANCE_SCOPE,
                            {
                                "scope_type": state.scope_type,
                                "scope_id": state.scope_id,
                            },
                        )
                        processed.add(state.scope_id)
                        count += 1
        return count

    async def poll_once(self) -> int:
        """Claim and process one bounded batch, returning the claim count."""

        leases = await self.claim_due()
        semaphore = asyncio.Semaphore(self._settings.concurrency)

        async def process(lease: LeasedOperation) -> None:
            async with semaphore:
                await self._process_lease(lease)

        await asyncio.gather(*(process(lease) for lease in leases))
        return len(leases)

    async def claim_due(self) -> tuple[LeasedOperation, ...]:
        now = self._clock()
        expires_at = now + timedelta(seconds=self._settings.lease_seconds)
        leases: list[LeasedOperation] = []
        terminal_events: list[TerminalOperationEvent] = []
        async with self._session_factory() as session:
            async with session.begin():
                result = await session.execute(
                    build_claim_statement(now, self._settings.batch_size).options(
                        joinedload(ExternalOperation.account),
                        noload(ExternalOperation.user),
                        noload(ExternalOperation.binding),
                    )
                )
                operations = result.unique().scalars().all()

                # Prepare every lease before changing ORM state. Only expired or
                # malformed rows change governance-driving columns in this batch;
                # pre-lock their distinct users and accounts in the platform-wide
                # order so two pollers cannot deadlock through opposite user order
                # on one shared account.
                expired_operation_ids = {
                    operation.operation_id
                    for operation in operations
                    if _expired_session_due(operation, now)
                }
                prepared_leases: dict[str, tuple[str, LeasedOperation]] = {}
                invalid_operation_ids: set[str] = set()
                for operation in operations:
                    if operation.operation_id in expired_operation_ids:
                        continue
                    token = f"{self._worker_id}:{uuid.uuid4()}"
                    try:
                        prepared_leases[operation.operation_id] = (
                            token,
                            _lease_snapshot(operation, token),
                        )
                    except Exception:
                        invalid_operation_ids.add(operation.operation_id)

                governance_mutations = [
                    operation
                    for operation in operations
                    if operation.operation_id
                    in expired_operation_ids | invalid_operation_ids
                ]
                await lock_governance_state_rows(
                    session,
                    user_ids=(
                        getattr(operation, "user_id", None)
                        for operation in governance_mutations
                    ),
                    account_ids=(
                        getattr(operation, "account_id", None)
                        for operation in governance_mutations
                    ),
                )
                for operation in operations:
                    expired_session = _terminalize_expired_session(operation, now)
                    if expired_session is not None:
                        terminal_events.append(expired_session)
                        logger.warning(
                            "Recovered external session {} after its hard deadline",
                            operation.operation_id,
                        )
                        continue
                    prepared = prepared_leases.get(operation.operation_id)
                    if prepared is None:
                        token = f"{self._worker_id}:{uuid.uuid4()}"
                    else:
                        token, prepared_lease = prepared
                    operation.lease_owner = token
                    operation.lease_expires_at = expires_at
                    try:
                        leases.append(
                            prepared_lease
                            if prepared is not None
                            else _lease_snapshot(operation, token)
                        )
                    except Exception:
                        original_status = operation.status
                        pending_recovery_billable = _pending_recovery_billable(
                            operation
                        )
                        operation.status = ExternalOperationStatus.FAILED.value
                        operation.error = _error(
                            "invalid_operation_state", retryable=False
                        )
                        operation.finished_at = now
                        operation.next_poll_at = None
                        operation.lease_owner = None
                        operation.lease_expires_at = None
                        settlement = dict(operation.settlement_metadata or {})
                        billable = (
                            True
                            if original_status != ExternalOperationStatus.PENDING.value
                            else pending_recovery_billable
                        )
                        settlement["billable"] = billable
                        operation.settlement_metadata = settlement
                        try:
                            usage = _usage_from_json(
                                operation.usage
                            ) or NormalizedUsage(requests=1)
                        except MappingExtractionError:
                            usage = NormalizedUsage(requests=1)
                        terminal_events.append(
                            TerminalOperationEvent(
                                operation_id=operation.operation_id,
                                status=ExternalOperationStatus.FAILED.value,
                                usage=usage,
                                # Submitted/running work has a durable upstream identity.
                                # A stale pre-dispatch row instead follows the route's
                                # snapshotted ambiguous-transport policy.
                                billable=billable,
                                result_descriptor=operation.result_descriptor,
                                error=operation.error,
                            )
                        )
                        logger.exception(
                            "External operation {} has an invalid persisted snapshot",
                            operation.operation_id,
                        )
        for event in terminal_events:
            await self._run_terminal_hooks(event)
        return tuple(leases)

    async def _process_lease(self, lease: LeasedOperation) -> None:
        lost = asyncio.Event()
        heartbeat = asyncio.create_task(self._renew_lease(lease, lost))
        try:
            outcome = (
                await self._cancel_outcome(lease)
                if lease.cancel_requested and not lease.cancel_dispatched
                else await self._poll(lease)
            )
        except asyncio.CancelledError:
            raise
        except (
            PollingConfigurationError,
            ExternalConfigurationError,
            MappingConfigurationError,
        ):
            logger.exception(
                "External operation {} has an invalid polling profile",
                lease.operation_id,
            )
            outcome = self._terminal_failure(
                lease,
                "configuration_error",
                billable=self._accepted_failure_billable(lease),
            )
        except MappingExtractionError:
            logger.exception(
                "External operation {} returned an invalid polling response",
                lease.operation_id,
            )
            outcome = self._terminal_failure(
                lease,
                "invalid_response",
                billable=self._accepted_failure_billable(
                    lease,
                    cancelling=lease.cancel_requested and not lease.cancel_dispatched,
                ),
            )
        except OperationNotCancellable:
            outcome = self._terminal_failure(
                lease, "cancellation_unavailable", billable=True
            )
        except ExternalTransportError:
            outcome = await self._transport_failure(
                lease,
                cancelling=lease.cancel_requested and not lease.cancel_dispatched,
            )
        except Exception:
            logger.exception("External operation {} poll failed", lease.operation_id)
            outcome = await self._transport_failure(
                lease,
                cancelling=lease.cancel_requested and not lease.cancel_dispatched,
            )
        finally:
            await self._stop_heartbeat(heartbeat, lost, lease.operation_id)
        if lost.is_set():
            logger.warning(
                "Lease lost while polling external operation {}", lease.operation_id
            )
            return
        try:
            event = await self._finalize(lease, outcome)
        except OperationLeaseLost:
            logger.warning(
                "Lease changed before external operation {} could be finalized",
                lease.operation_id,
            )
            return
        if event is not None:
            await self._run_terminal_hooks(event)

    async def _poll(self, lease: LeasedOperation) -> PollOutcome:
        now = self._clock()
        policy = TaskLifecyclePolicy.from_route(lease.route)
        if lease.expires_at is not None and lease.expires_at <= now:
            return self._expired_outcome(lease, policy)
        response = await self._execute_call(lease, policy.poll, "poll")
        attempt = lease.poll_attempts + 1
        if response.status_code in policy.poll.retry.statuses:
            return self._retry_or_fail(lease, policy.poll, attempt, response=response)
        if response.status_code not in policy.poll.success_statuses:
            return self._terminal_failure(
                lease,
                "rejected_response",
                billable=ExternalOperationStatus.FAILED.value
                in policy.poll.billable_statuses,
            )
        try:
            response_value = response.json()
        except Exception as exc:
            raise MappingExtractionError("poll response is not valid JSON") from exc
        assert policy.poll.task is not None
        extracted = policy.poll.task.extract(
            request=lease.request_metadata,
            response=response_value,
            context=_context(lease),
        )
        if extracted.status is None:
            raise MappingExtractionError("poll response did not contain a task status")
        upstream_status = extracted.status
        operation_status = (
            ExternalOperationStatus.SUBMITTED.value
            if extracted.status == ExternalOperationStatus.PENDING.value
            else extracted.status
        )
        observed = (
            _lifecycle_usage_observation(
                policy.poll.usage,
                response=response_value,
                context=_context(lease),
            )
            if policy.poll.usage is not None
            else None
        )
        usage = _merge_usage(lease, observed, policy.poll.usage_mode)
        terminal = operation_status in _TERMINAL_STATUSES
        descriptor = _result_descriptor(
            lease,
            extracted,
            terminal=terminal,
            public_rules=policy.poll.public,
            artifact_url_factory=self._artifact_url_factory,
            now=now,
        )
        if not terminal and self._attempts_exhausted(policy.poll, attempt):
            return self._terminal_failure(
                lease,
                "poll_limit_exceeded",
                billable=ExternalOperationStatus.FAILED.value
                in policy.poll.billable_statuses,
                usage=usage,
                result_descriptor=descriptor,
            )
        next_poll = None
        if not terminal:
            next_poll = now + timedelta(
                seconds=policy.poll.backoff.delay(attempt, self._random_value())
            )
        return PollOutcome(
            status=operation_status,
            upstream_status=upstream_status,
            usage=usage,
            result_descriptor=descriptor,
            error=(
                None
                if operation_status == ExternalOperationStatus.SUCCEEDED.value
                else (
                    _error("operation_not_completed", retryable=False)
                    if terminal
                    else None
                )
            ),
            next_poll_at=next_poll,
            billable=(
                operation_status == ExternalOperationStatus.SUCCEEDED.value
                or operation_status in policy.poll.billable_statuses
            ),
        )

    async def _execute_call(
        self,
        lease: LeasedOperation,
        call: EndpointCall,
        name_suffix: str,
    ) -> BufferedResponse:
        profile = build_endpoint_profile(
            lease.account,  # type: ignore[arg-type]
            lease.route,
            endpoint=call.endpoint,
            name_suffix=name_suffix,
        )
        if profile.response_mode is not ResponseMode.BUFFERED:
            raise PollingConfigurationError("task lifecycle endpoint must be buffered")
        outbound = call.request.build(profile, _sources(lease))
        started = time.monotonic()
        try:
            response = await self._executor_factory(lease.account).execute(
                profile, outbound
            )
        except ExternalTransportError:
            upstream_latency.labels(phase=name_suffix).observe(
                time.monotonic() - started
            )
            upstream_requests.labels(phase=name_suffix, outcome="transport_error").inc()
            await record_upstream_result(lease.account.account_id, transport_error=True)
            raise
        upstream_latency.labels(phase=name_suffix).observe(time.monotonic() - started)
        upstream_requests.labels(
            phase=name_suffix, outcome=status_class(response.status_code)
        ).inc()
        await record_upstream_result(
            lease.account.account_id, status_code=response.status_code
        )
        if not isinstance(response, BufferedResponse):
            await response.aclose()
            raise PollingConfigurationError("task lifecycle endpoint must be buffered")
        return response

    def _retry_or_fail(
        self,
        lease: LeasedOperation,
        call: EndpointCall,
        attempt: int,
        *,
        response: BufferedResponse | None = None,
    ) -> PollOutcome:
        if self._attempts_exhausted(call, attempt):
            return self._terminal_failure(
                lease,
                "poll_limit_exceeded",
                billable=ExternalOperationStatus.FAILED.value in call.billable_statuses,
            )
        delay = (
            call.retry.delay(
                response,
                call.backoff,
                attempt,
                self._clock(),
                self._random_value(),
            )
            if response is not None
            else call.backoff.delay(attempt, self._random_value())
        )
        return PollOutcome(
            status=lease.status,
            upstream_status=lease.status,
            usage=lease.usage,
            result_descriptor=lease.result_descriptor,
            error=None,
            next_poll_at=self._clock() + timedelta(seconds=delay),
        )

    async def _transport_failure(
        self, lease: LeasedOperation, *, cancelling: bool = False
    ) -> PollOutcome:
        try:
            policy = TaskLifecyclePolicy.from_route(lease.route)
            call = policy.cancel if cancelling else policy.poll
        except Exception:
            return self._terminal_failure(lease, "configuration_error", billable=True)
        if call is None:
            return self._terminal_failure(
                lease, "cancellation_unavailable", billable=True
            )
        attempt = lease.poll_attempts + 1
        if not call.retry.transport_errors:
            return self._terminal_failure(
                lease,
                "transport_error",
                billable=ExternalOperationStatus.FAILED.value in call.billable_statuses,
            )
        return self._retry_or_fail(lease, call, attempt)

    @staticmethod
    def _expired_outcome(
        lease: LeasedOperation, policy: TaskLifecyclePolicy
    ) -> PollOutcome:
        return PollOutcome(
            status=ExternalOperationStatus.EXPIRED.value,
            upstream_status=lease.status,
            usage=lease.usage,
            result_descriptor=lease.result_descriptor,
            error=_error("operation_expired", retryable=False),
            next_poll_at=None,
            billable=(
                ExternalOperationStatus.FAILED.value in policy.poll.billable_statuses
                or ExternalOperationStatus.EXPIRED.value
                in policy.poll.billable_statuses
            ),
        )

    @staticmethod
    def _accepted_failure_billable(
        lease: LeasedOperation, *, cancelling: bool = False
    ) -> bool:
        """Apply a valid route's opt-out policy to local post-accept failures."""

        try:
            policy = TaskLifecyclePolicy.from_route(lease.route)
            call = policy.cancel if cancelling else policy.poll
        except Exception:
            # The route is unusable, so no trustworthy opt-out can be read. The
            # durable task identity proves the upstream already accepted work.
            return True
        if call is None:
            return True
        return ExternalOperationStatus.FAILED.value in call.billable_statuses

    @staticmethod
    def _attempts_exhausted(call: EndpointCall, attempt: int) -> bool:
        return call.retry.max_attempts > 0 and attempt >= call.retry.max_attempts

    @staticmethod
    def _terminal_failure(
        lease: LeasedOperation,
        code: str,
        *,
        billable: bool,
        usage: NormalizedUsage | None = None,
        result_descriptor: Mapping[str, Any] | None = None,
        cancel_dispatched: bool = False,
    ) -> PollOutcome:
        return PollOutcome(
            status=ExternalOperationStatus.FAILED.value,
            upstream_status=lease.status,
            usage=usage or lease.usage,
            result_descriptor=result_descriptor or lease.result_descriptor,
            error=_error(code, retryable=False),
            next_poll_at=None,
            billable=billable,
            cancel_dispatched=cancel_dispatched,
        )

    async def _renew_lease(self, lease: LeasedOperation, lost: asyncio.Event) -> None:
        interval = max(1.0, min(30.0, self._settings.lease_seconds / 3))
        while True:
            await asyncio.sleep(interval)
            expires_at = self._clock() + timedelta(seconds=self._settings.lease_seconds)
            async with self._session_factory() as session:
                async with session.begin():
                    result = await session.execute(
                        update(ExternalOperation)
                        .where(
                            ExternalOperation.operation_id == lease.operation_id,
                            ExternalOperation.lease_owner == lease.lease_token,
                            ExternalOperation.status.in_(_ACTIVE_STATUSES),
                        )
                        .values(lease_expires_at=expires_at)
                    )
                    if result.rowcount != 1:
                        lost.set()
                        return

    @staticmethod
    async def _stop_heartbeat(
        heartbeat: asyncio.Task[None], lost: asyncio.Event, operation_id: str
    ) -> None:
        heartbeat.cancel()
        try:
            await heartbeat
        except asyncio.CancelledError:
            pass
        except Exception:
            lost.set()
            logger.exception(
                "Lease renewal failed for external operation {}", operation_id
            )

    async def _finalize(
        self, lease: LeasedOperation, outcome: PollOutcome
    ) -> TerminalOperationEvent | None:
        now = self._clock()
        async with self._session_factory() as session:
            async with session.begin():
                operation = await session.get(
                    ExternalOperation, lease.operation_id, with_for_update=True
                )
                if operation is None or operation.lease_owner != lease.lease_token:
                    raise OperationLeaseLost(lease.operation_id)
                if operation.status not in _ACTIVE_STATUSES:
                    operation.lease_owner = None
                    operation.lease_expires_at = None
                    return None
                operation.poll_attempts += 1
                operation.last_polled_at = now
                operation.status = outcome.status
                operation.upstream_status = outcome.upstream_status
                operation.usage = outcome.usage.to_dict() if outcome.usage else None
                operation.result_descriptor = (
                    copy.deepcopy(dict(outcome.result_descriptor))
                    if outcome.result_descriptor is not None
                    else None
                )
                operation.error = copy.deepcopy(outcome.error)
                settlement = dict(operation.settlement_metadata or {})
                cancel_arrived_during_poll = bool(
                    settlement.get("cancel_requested")
                    and not lease.cancel_requested
                    and not outcome.cancel_dispatched
                    and not outcome.terminal
                )
                operation.next_poll_at = (
                    now if cancel_arrived_during_poll else outcome.next_poll_at
                )
                operation.lease_owner = None
                operation.lease_expires_at = None
                if outcome.cancel_dispatched:
                    settlement["cancel_dispatched"] = True
                    settlement["cancel_dispatched_at"] = now.isoformat()
                    operation.settlement_metadata = settlement
                if (
                    outcome.status == ExternalOperationStatus.RUNNING.value
                    and operation.started_at is None
                ):
                    operation.started_at = now
                if outcome.terminal:
                    operation.finished_at = now
                    settlement["billable"] = outcome.billable
                    operation.settlement_metadata = settlement
                    usage = outcome.usage or NormalizedUsage(requests=1)
                    return TerminalOperationEvent(
                        operation_id=lease.operation_id,
                        status=outcome.status,
                        usage=usage,
                        billable=outcome.billable,
                        result_descriptor=outcome.result_descriptor,
                        error=outcome.error,
                    )
        return None

    async def _run_terminal_hooks(self, event: TerminalOperationEvent) -> None:
        for hook in self._terminal_hooks:
            try:
                result = hook(event)
                if inspect.isawaitable(result):
                    await result
            except Exception:
                logger.exception(
                    "External operation terminal hook failed for {}", event.operation_id
                )

    async def _cancel_outcome(self, lease: LeasedOperation) -> PollOutcome:
        policy = TaskLifecyclePolicy.from_route(lease.route)
        if lease.expires_at is not None and lease.expires_at <= self._clock():
            return self._expired_outcome(lease, policy)
        if policy.cancel is None:
            raise OperationNotCancellable(lease.operation_id)
        response = await self._execute_call(lease, policy.cancel, "cancel")
        if response.status_code in policy.cancel.retry.statuses:
            return self._retry_or_fail(
                lease, policy.cancel, lease.poll_attempts + 1, response=response
            )
        if response.status_code not in policy.cancel.success_statuses:
            return self._terminal_failure(
                lease,
                "cancellation_rejected",
                billable=ExternalOperationStatus.FAILED.value
                in policy.cancel.billable_statuses,
                cancel_dispatched=True,
            )
        extracted = ExtractedTask(status=ExternalOperationStatus.CANCELLED.value)
        response_value: Any = None
        if response.body:
            try:
                response_value = response.json()
            except Exception as exc:
                if policy.cancel.task is not None:
                    raise MappingExtractionError(
                        "cancellation response is not valid JSON"
                    ) from exc
        if policy.cancel.task is not None and response_value is not None:
            extracted = policy.cancel.task.extract(
                request=lease.request_metadata,
                response=response_value,
                context=_context(lease),
            )
        status = extracted.status or ExternalOperationStatus.CANCELLED.value
        if status == ExternalOperationStatus.PENDING.value:
            status = ExternalOperationStatus.SUBMITTED.value
        terminal = status in _TERMINAL_STATUSES
        observed = (
            _lifecycle_usage_observation(
                policy.cancel.usage,
                response=response_value,
                context=_context(lease),
            )
            if policy.cancel.usage is not None and response_value is not None
            else None
        )
        usage = _merge_usage(lease, observed, policy.cancel.usage_mode)
        descriptor = _result_descriptor(
            lease,
            extracted,
            terminal=terminal,
            public_rules=policy.cancel.public,
            artifact_url_factory=self._artifact_url_factory,
            now=self._clock(),
        )
        return PollOutcome(
            status=status,
            upstream_status=status,
            usage=usage,
            result_descriptor=descriptor,
            error=(
                _error("operation_cancelled", retryable=False)
                if status == ExternalOperationStatus.CANCELLED.value
                else None
            ),
            next_poll_at=(
                None
                if terminal
                else self._clock()
                + timedelta(
                    seconds=policy.cancel.backoff.delay(
                        lease.poll_attempts + 1, self._random_value()
                    )
                )
            ),
            billable=(
                status == ExternalOperationStatus.SUCCEEDED.value
                or status in policy.cancel.billable_statuses
            ),
            cancel_dispatched=True,
        )

    async def cancel_operation(self, operation_id: str) -> str:
        """Execute a configured cancellation endpoint and return the resulting status."""

        lease = await self._claim_one(operation_id)
        if lease is None:
            async with self._session_factory() as session:
                operation = await session.get(ExternalOperation, operation_id)
                if operation is None:
                    raise LookupError(operation_id)
                return operation.status
        if lease.cancel_dispatched:
            await self._release_lease(lease)
            return lease.status
        lost = asyncio.Event()
        heartbeat = asyncio.create_task(self._renew_lease(lease, lost))
        try:
            outcome = await self._cancel_outcome(lease)
        except Exception:
            await self._stop_heartbeat(heartbeat, lost, lease.operation_id)
            await self._release_lease(lease)
            raise
        finally:
            if not heartbeat.done():
                await self._stop_heartbeat(heartbeat, lost, lease.operation_id)
        if lost.is_set():
            raise OperationLeaseLost(operation_id)
        event = await self._finalize(lease, outcome)
        if event is not None:
            await self._run_terminal_hooks(event)
        return outcome.status

    async def _claim_one(self, operation_id: str) -> LeasedOperation | None:
        now = self._clock()
        async with self._session_factory() as session:
            async with session.begin():
                operation = (
                    (
                        await session.execute(
                            select(ExternalOperation)
                            .options(
                                joinedload(ExternalOperation.account),
                                noload(ExternalOperation.user),
                                noload(ExternalOperation.binding),
                            )
                            .where(ExternalOperation.operation_id == operation_id)
                            .with_for_update(of=ExternalOperation)
                        )
                    )
                    .unique()
                    .scalar_one_or_none()
                )
                if operation is None or operation.status not in _ACTIVE_STATUSES:
                    return None
                if (
                    operation.lease_expires_at is not None
                    and operation.lease_expires_at > now
                ):
                    raise OperationLeaseLost(operation_id)
                token = f"{self._worker_id}:{uuid.uuid4()}"
                operation.lease_owner = token
                operation.lease_expires_at = now + timedelta(
                    seconds=self._settings.lease_seconds
                )
                return _lease_snapshot(operation, token)

    async def _release_lease(self, lease: LeasedOperation) -> None:
        async with self._session_factory() as session:
            async with session.begin():
                await session.execute(
                    update(ExternalOperation)
                    .where(
                        ExternalOperation.operation_id == lease.operation_id,
                        ExternalOperation.lease_owner == lease.lease_token,
                    )
                    .values(lease_owner=None, lease_expires_at=None)
                )


async def _settlement_hook(event: TerminalOperationEvent) -> None:
    from .service import settle_operation

    await settle_operation(event.operation_id, event.usage, billable=event.billable)


def _default_executor_factory(account: AccountSnapshot) -> ExternalExecutor:
    references = set(account.credential_references.values())
    cache: dict[str, str] = {}

    async def resolve(reference: str) -> str:
        if reference not in references or not reference.startswith("secret://"):
            raise ExternalConfigurationError("credential reference is unavailable")
        if reference in cache:
            return cache[reference]
        from api.payment.util import decrypt_secret
        from api.secret.schemas import Secret

        secret_id = reference.removeprefix("secret://")
        async with get_session(readonly=True) as session:
            secret = (
                await session.execute(
                    select(Secret).where(
                        Secret.secret_id == secret_id,
                        Secret.user_id == account.user_id,
                        Secret.kind == "external_backend",
                    )
                )
            ).scalar_one_or_none()
        if secret is None:
            raise ExternalConfigurationError("credential is unavailable")
        cache[reference] = await decrypt_secret(secret.value)
        return cache[reference]

    return ExternalExecutor(secret_resolver=resolve)


_DEFAULT_POLLER: ExternalOperationPoller | None = None


def start_external_operation_poller(**kwargs: Any) -> ExternalOperationPoller:
    """Create and start the process-local polling service once."""

    global _DEFAULT_POLLER
    from api.config import settings as application_settings

    if "settings" not in kwargs:
        kwargs["settings"] = WorkerSettings(
            batch_size=application_settings.external_poller_batch_size,
            concurrency=application_settings.external_poller_concurrency,
            lease_seconds=application_settings.external_poller_lease_seconds,
            idle_seconds=application_settings.external_poller_idle_seconds,
            shutdown_timeout_seconds=(
                application_settings.external_poller_shutdown_timeout_seconds
            ),
            settlement_reconcile_interval_seconds=(
                application_settings.external_settlement_reconcile_interval_seconds
            ),
            maintenance_interval_seconds=(
                application_settings.external_operation_maintenance_interval_seconds
            ),
            retention_days=application_settings.external_operation_retention_days,
            retention_batch_size=(
                application_settings.external_operation_retention_batch_size
            ),
            settlement_batch_size=(application_settings.external_settlement_batch_size),
        )
    if _DEFAULT_POLLER is None:
        _DEFAULT_POLLER = ExternalOperationPoller(**kwargs)
    if application_settings.external_poller_enabled:
        _DEFAULT_POLLER.start()
    else:
        logger.info("External operation poller is disabled in this process")
    return _DEFAULT_POLLER


async def stop_external_operation_poller() -> None:
    """Stop and forget the process-local polling service."""

    global _DEFAULT_POLLER
    poller = _DEFAULT_POLLER
    _DEFAULT_POLLER = None
    if poller is not None:
        await poller.stop()


__all__ = [
    "AccountSnapshot",
    "BackoffPolicy",
    "EndpointCall",
    "EndpointRequest",
    "ExternalOperationPoller",
    "LeasedOperation",
    "OperationLeaseLost",
    "OperationNotCancellable",
    "PollOutcome",
    "PollRetryPolicy",
    "PollingConfigurationError",
    "TaskLifecyclePolicy",
    "TerminalOperationEvent",
    "WorkerSettings",
    "build_claim_statement",
    "build_settlement_reconcile_statement",
    "start_external_operation_poller",
    "stop_external_operation_poller",
]
