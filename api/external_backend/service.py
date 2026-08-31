"""Invocation and settlement services for externally executed Chutes."""

from __future__ import annotations

import asyncio
import copy
import hashlib
import math
import random
import re
import time
import uuid
from contextlib import suppress
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from email.utils import parsedate_to_datetime
from types import SimpleNamespace
from typing import Any, AsyncIterator, Awaitable, Mapping

import jsonschema
import orjson
from fastapi import HTTPException, Request, Response, status
from loguru import logger
from sqlalchemy import select, text
from sqlalchemy.orm import joinedload
from starlette.datastructures import UploadFile
from starlette.responses import StreamingResponse as StarletteStreamingResponse

from api.chute.schemas import Chute
from api.database import get_session
from api.external_transport import (
    BodyMode,
    BufferedResponse,
    ExternalExecutor,
    ExternalTransportError,
    JsonBody,
    MultipartBody,
    MultipartPart,
    OutboundRequest,
    ProfileError,
    RawBody,
    RequestRejectedError,
    ResponseMode,
    SSEEvent,
    StreamingResponse,
)
from api.invocation.util import build_response_headers
from api.metrics.capacity import track_request_completed, track_request_rate_limited
from api.payment.pricing import (
    NormalizedUsage,
    PricingConfigurationError,
    PricingContext,
    PricingRule,
    UsageValidationError,
    parse_pricing_rules,
    price_override,
    price_usage,
    validate_conditional_pricing_coverage,
    validate_legacy_pricing_rates,
)
from api.permissions import Permissioning
from api.payment.util import decrypt_secret
from api.secret.schemas import Secret
from api.user.schemas import PriceOverride, User

from .config import (
    ExternalConfigurationError,
    RetryPolicy,
    build_endpoint_profile,
    retry_policy,
    select_route,
)
from .circuit import record_upstream_result
from .governance import (
    ExternalAdmissionRejected,
    compile_session_budget,
    enforce_external_admission,
    running_budget_available,
    session_budget_from_metadata,
    session_exposure_estimate,
)
from .artifact_policy import normalize_artifact_expiration
from .billing_outbox import (
    ExternalUsageEvent,
    deliver_external_usage_event,
    enqueue_external_usage_event,
    external_usage_event_exists,
)
from .mapping import (
    MappingConfigurationError,
    MappingExtractionError,
    StreamUsageMode,
    UsageMapping,
    extract_task,
    extract_usage,
    extract_value,
    merge_stream_usage,
    scrub_public_response,
    transform_payload,
)
from .metrics import (
    settlement_attempts,
    status_class,
    upstream_latency,
    upstream_requests,
)
from .public_urls import artifact_url, operation_url
from .operation_lifecycle import (
    USAGE_CHECKPOINT_INTERVAL_SECONDS,
    UsageBudgetMonitor,
    UsageCheckpointLoop,
    session_recovery_deadline,
)
from .request_mapping import (
    ExternalRequestMappingError,
    map_upstream_query_parameters as _upstream_query_parameters,
)
from .schema_validation import (
    RemoteSchemaReferenceError,
    UnsafeSchemaError,
    local_json_schema_validator,
)
from .task_results import bounded_inline_result, inline_result_limit
from .schemas import (
    ExternalBackendAccount,
    ExternalChuteBinding,
    ExternalOperation,
    ExternalOperationMode,
    ExternalOperationStatus,
    ExternalResultStatus,
    ExternalRouteConfig,
    ExternalSettlementStatus,
)


_BACKGROUND_TASKS: set[asyncio.Task[Any]] = set()
_SENSITIVE_QUERY_NAMES = frozenset(
    {
        "access_key",
        "access_token",
        "api_key",
        "apikey",
        "authorization",
        "credential",
        "key",
        "password",
        "secret",
        "signature",
        "token",
    }
)
_PUBLIC_RESPONSE_HEADERS = frozenset({"content-type"})


class _PersistedUsageEventReady(Exception):
    """Internal control flow for delivering an already snapshotted charge."""


_TERMINAL_STATUSES = frozenset(
    {
        ExternalOperationStatus.SUCCEEDED.value,
        ExternalOperationStatus.FAILED.value,
        ExternalOperationStatus.CANCELLED.value,
        ExternalOperationStatus.EXPIRED.value,
    }
)
_RESOLVED_SETTLEMENT_STATUSES = frozenset(
    {
        ExternalSettlementStatus.SETTLED.value,
        ExternalSettlementStatus.NOT_BILLABLE.value,
    }
)


class ExternalInvocationError(RuntimeError):
    """An external invocation failed without exposing its provider details."""


def _is_sensitive_query_name(value: str) -> bool:
    normalized = re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")
    compact = normalized.replace("_", "")
    if normalized in _SENSITIVE_QUERY_NAMES or any(
        marker in compact
        for marker in (
            "accesskey",
            "accesstoken",
            "apikey",
            "authtoken",
            "bearertoken",
            "clientsecret",
            "privatekey",
            "secretkey",
            "sessiontoken",
            "signature",
            "subscriptionkey",
        )
    ):
        return True
    return bool(
        set(normalized.split("_"))
        & {
            "auth",
            "authorization",
            "credential",
            "credentials",
            "password",
            "secret",
            "token",
        }
    )


def _finish_background_task(task: asyncio.Task[Any]) -> None:
    _BACKGROUND_TASKS.discard(task)
    if task.cancelled():
        logger.warning("External invocation accounting task was cancelled")
        return
    exception = task.exception()
    if exception is not None and not isinstance(exception, HTTPException):
        logger.error(
            "External invocation background task failed ({})",
            type(exception).__name__,
        )


def _spawn(coroutine: Any) -> asyncio.Task[Any]:
    task = asyncio.create_task(coroutine)
    _BACKGROUND_TASKS.add(task)
    task.add_done_callback(_finish_background_task)
    return task


def _object(value: object, name: str) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ExternalConfigurationError(f"{name} must be an object")
    return dict(value)


def _safe_public_headers(headers: Mapping[str, str]) -> dict[str, str]:
    return {
        name: value
        for name, value in headers.items()
        if name.lower() in _PUBLIC_RESPONSE_HEADERS
    }


def _content_type(headers: Mapping[str, str], default: str) -> str:
    value = next(
        (value for name, value in headers.items() if name.lower() == "content-type"),
        default,
    )
    return value.split(";", 1)[0].strip() or default


def _validated_output_content_type(
    headers: Mapping[str, str], selected_cord: Mapping[str, Any]
) -> str | None:
    configured = selected_cord.get("output_content_type")
    if configured is None:
        return None
    if (
        not isinstance(configured, str)
        or not configured.strip()
        or "/" not in configured
        or any(character in configured for character in "\r\n\x00")
    ):
        raise ExternalConfigurationError("Cord output_content_type is invalid")
    expected = configured.split(";", 1)[0].strip().lower()
    actual = _content_type(headers, "").lower()
    if not actual or actual != expected:
        raise ExternalInvocationError(
            "upstream response content type did not match the Cord"
        )
    return expected


def _json_or_none(body: bytes) -> Any | None:
    if not body:
        return None
    try:
        return orjson.loads(body)
    except orjson.JSONDecodeError:
        return None


def _json_response(body: bytes) -> tuple[bool, Any | None]:
    """Distinguish a valid JSON null from bytes that are not JSON."""

    if not body:
        return False, None
    try:
        return True, orjson.loads(body)
    except orjson.JSONDecodeError:
        return False, None


def _is_json_content_type(headers: Mapping[str, str]) -> bool:
    media_type = _content_type(headers, "").lower()
    return media_type == "application/json" or media_type.endswith("+json")


def _public_mapping(
    route: ExternalRouteConfig,
    *,
    chute_name: str,
    invocation_id: str,
    operation_id: str | None = None,
) -> dict[str, Any]:
    response_config = _object(route.response_config, "response_config")
    configured = _object(response_config.get("public"), "response_config.public")
    result = copy.deepcopy(configured)
    rewrites = _object(result.get("rewrite_keys"), "public.rewrite_keys")
    rewrites["model"] = chute_name
    rewrites["model_id"] = chute_name
    rewrites["model_name"] = chute_name
    rewrites["request_id"] = invocation_id
    if operation_id:
        for key in ("task_id", "job_id", "operation_id"):
            rewrites[key] = operation_id
    result["rewrite_keys"] = rewrites
    return result


def _replace_scalar(value: Any, old: str | None, new: str) -> Any:
    if old is None:
        return value
    if isinstance(value, dict):
        return {key: _replace_scalar(item, old, new) for key, item in value.items()}
    if isinstance(value, list):
        return [_replace_scalar(item, old, new) for item in value]
    return new if isinstance(value, str) and value == old else value


def _canonical_status(value: str | None, default: str) -> str:
    if value == ExternalOperationStatus.PENDING.value:
        # PENDING is reserved for the local pre-dispatch window. Once an upstream
        # task id exists, the poller must be able to claim the operation.
        return ExternalOperationStatus.SUBMITTED.value
    if value in {
        "submitted",
        "running",
        "succeeded",
        "failed",
        "cancelled",
        "expired",
    }:
        return value
    return default


async def _load_binding(
    chute_id: str,
) -> tuple[ExternalChuteBinding, ExternalBackendAccount]:
    async with get_session(readonly=True) as db:
        binding = (
            (
                await db.execute(
                    select(ExternalChuteBinding)
                    .options(joinedload(ExternalChuteBinding.account))
                    .where(ExternalChuteBinding.chute_id == chute_id)
                )
            )
            .unique()
            .scalar_one_or_none()
        )
        if not binding or not binding.enabled or not binding.account.enabled:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="External execution is currently unavailable.",
            )
        return binding, binding.account


def _transport_snapshot_matches(
    row: Mapping[str, Any] | None,
    binding: ExternalChuteBinding,
    account: ExternalBackendAccount,
) -> bool:
    """Compare the locked transport rows with the configuration used to prepare work."""

    if row is None:
        return False
    expected = {
        "account_id": account.account_id,
        "account_user_id": account.user_id,
        "account_base_url": account.base_url,
        "credential_references": dict(account.credential_references or {}),
        "auth_header_templates": list(account.auth_header_templates or []),
        "connection_config": dict(account.connection_config or {}),
        "account_enabled": bool(account.enabled),
        "account_updated_at": account.updated_at,
        "binding_id": binding.binding_id,
        "binding_chute_id": binding.chute_id,
        "binding_account_id": binding.account_id,
        "binding_routes": list(binding.routes or []),
        "binding_enabled": bool(binding.enabled),
        "binding_updated_at": binding.updated_at,
    }
    return all(row.get(key) == value for key, value in expected.items())


async def _lock_transport_snapshot(
    db: Any,
    binding: ExternalChuteBinding,
    account: ExternalBackendAccount,
) -> None:
    """Serialize acceptance with mutations and reject a stale configuration snapshot."""

    result = await db.execute(
        text(
            """
            SELECT
                account.account_id AS account_id,
                account.user_id AS account_user_id,
                account.base_url AS account_base_url,
                account.credential_references AS credential_references,
                account.auth_header_templates AS auth_header_templates,
                account.connection_config AS connection_config,
                account.enabled AS account_enabled,
                account.updated_at AS account_updated_at,
                binding.binding_id AS binding_id,
                binding.chute_id AS binding_chute_id,
                binding.account_id AS binding_account_id,
                binding.routes AS binding_routes,
                binding.enabled AS binding_enabled,
                binding.updated_at AS binding_updated_at
            FROM external_chute_bindings AS binding
            JOIN external_backend_accounts AS account
              ON account.account_id = binding.account_id
            WHERE binding.binding_id = :binding_id
              AND binding.chute_id = :chute_id
              AND account.account_id = :account_id
            FOR SHARE OF account, binding
            """
        ),
        {
            "binding_id": binding.binding_id,
            "chute_id": binding.chute_id,
            "account_id": account.account_id,
        },
    )
    row = result.mappings().one_or_none()
    # Binding-before-Chute matches the external deletion guard and prevents a
    # cascade delete from deadlocking against operation acceptance.
    locked_chute = (
        await db.execute(
            text(
                """
                SELECT chute_id
                FROM chutes
                WHERE chute_id = :chute_id
                  AND execution_backend = 'external'
                  AND disabled IS FALSE
                FOR SHARE
                """
            ),
            {"chute_id": binding.chute_id},
        )
    ).scalar_one_or_none()
    if locked_chute is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="External execution configuration changed; retry the request.",
        )
    if not _transport_snapshot_matches(row, binding, account):
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="External execution configuration changed; retry the request.",
        )


def build_secret_resolver(account: ExternalBackendAccount):
    allowed = set((account.credential_references or {}).values())
    cache: dict[str, str] = {}

    async def resolve(reference: str) -> str:
        if reference not in allowed or not reference.startswith("secret://"):
            raise ExternalConfigurationError("credential reference is not configured")
        if reference in cache:
            return cache[reference]
        secret_id = reference.removeprefix("secret://")
        async with get_session(readonly=True) as db:
            secret = (
                await db.execute(
                    select(Secret).where(
                        Secret.secret_id == secret_id,
                        Secret.user_id == account.user_id,
                        Secret.kind == "external_backend",
                    )
                )
            ).scalar_one_or_none()
        if not secret:
            raise ExternalConfigurationError("credential is unavailable")
        value = await decrypt_secret(secret.value)
        cache[reference] = value
        return value

    return resolve


def _schema_validation_enabled(config: Mapping[str, Any], key: str, label: str) -> bool:
    enabled = config.get(key, False)
    if not isinstance(enabled, bool):
        raise ExternalConfigurationError(f"{label}.{key} must be a boolean")
    return enabled


def _schema_for_request(
    route: ExternalRouteConfig, cord: Mapping[str, Any]
) -> dict[str, Any] | None:
    request_config = _object(route.request_config, "request_config")
    if not _schema_validation_enabled(
        request_config, "validate_input_schema", "request_config"
    ):
        return None
    schema = cord.get("input_schema") or cord.get("minimal_input_schema")
    if not isinstance(schema, Mapping) or not schema:
        raise ExternalConfigurationError(
            "request_config.validate_input_schema=true requires a non-empty "
            "Cord input schema"
        )
    return dict(schema)


async def _validate_schema(payload: Any, schema: Mapping[str, Any], label: str) -> None:
    def validate() -> None:
        try:
            validator = local_json_schema_validator(schema)
        except (RemoteSchemaReferenceError, UnsafeSchemaError) as exc:
            raise ExternalConfigurationError(str(exc)) from exc
        if first := next(validator.iter_errors(payload), None):
            path = ".".join(str(item) for item in first.absolute_path)
            location = f" at {path}" if path else ""
            raise ValueError(f"{label}{location}: {first.message}")

    try:
        await asyncio.wait_for(asyncio.to_thread(validate), timeout=2.0)
    except asyncio.TimeoutError as exc:
        raise ExternalConfigurationError(f"{label} validation timed out") from exc


async def _request_body(
    request: Request, route: ExternalRouteConfig, body_mode: BodyMode
) -> tuple[Any, JsonBody | RawBody | MultipartBody | None]:
    request_config = _object(route.request_config, "request_config")
    configured_limit = request_config.get("max_request_bytes", 64 * 1024 * 1024)
    if isinstance(configured_limit, bool) or not isinstance(configured_limit, int):
        raise ExternalConfigurationError("max_request_bytes must be an integer")
    if configured_limit < 1 or configured_limit > 1024 * 1024 * 1024:
        raise ExternalConfigurationError(
            "max_request_bytes is outside the supported range"
        )

    if body_mode is BodyMode.NONE:
        return {}, None
    if body_mode is BodyMode.JSON:
        raw = await request.body()
        if len(raw) > configured_limit:
            raise HTTPException(status_code=413, detail="Request body is too large.")
        try:
            value = orjson.loads(raw) if raw else {}
        except orjson.JSONDecodeError as exc:
            raise HTTPException(
                status_code=400, detail="Invalid JSON request body."
            ) from exc
        return value, JsonBody(value)
    if body_mode is BodyMode.RAW:
        raw = await request.body()
        if len(raw) > configured_limit:
            raise HTTPException(status_code=413, detail="Request body is too large.")
        media_type = request.headers.get("content-type", "application/octet-stream")
        return raw, RawBody(raw, media_type)

    form = await request.form()
    parts: list[MultipartPart] = []
    schema_value: dict[str, Any] = {}
    total = 0
    for name, item in form.multi_items():
        if isinstance(item, UploadFile):
            data = await item.read()
            total += len(data)
            parts.append(
                MultipartPart(
                    name=name,
                    value=data,
                    filename=item.filename,
                    content_type=item.content_type,
                )
            )
            public_value: Any = {
                "filename": item.filename,
                "content_type": item.content_type,
                "size_bytes": len(data),
            }
        else:
            encoded = str(item).encode()
            total += len(encoded)
            parts.append(MultipartPart(name=name, value=str(item)))
            public_value = str(item)
        if name in schema_value:
            existing = schema_value[name]
            schema_value[name] = (
                existing + [public_value]
                if isinstance(existing, list)
                else [existing, public_value]
            )
        else:
            schema_value[name] = public_value
        if total > configured_limit:
            raise HTTPException(status_code=413, detail="Request body is too large.")
    return schema_value, MultipartBody(tuple(parts))


def _allowed_query(request: Request, route: ExternalRouteConfig) -> dict[str, Any]:
    request_config = _object(route.request_config, "request_config")
    configured = request_config.get("allowed_query_parameters", [])
    if not isinstance(configured, list) or any(
        not isinstance(item, str) for item in configured
    ):
        raise ExternalConfigurationError("allowed_query_parameters must be an array")
    allowed = set(configured)
    result: dict[str, Any] = {}
    for key, value in request.query_params.multi_items():
        if _is_sensitive_query_name(key) or key not in allowed:
            continue
        if key in result:
            current = result[key]
            result[key] = (
                current + [value] if isinstance(current, list) else [current, value]
            )
        else:
            result[key] = value
    return result


def _schema_request_value(
    value: Any,
    body: Any,
    *,
    body_mode: BodyMode | None = None,
    query: Mapping[str, Any] | None = None,
) -> Any:
    """Expose multipart files to OpenAPI/JSON Schema as binary strings."""

    if body_mode is BodyMode.NONE:
        return dict(query or {})
    if not isinstance(body, MultipartBody):
        return value

    def convert(item: Any) -> Any:
        if isinstance(item, Mapping) and set(item) == {
            "filename",
            "content_type",
            "size_bytes",
        }:
            filename = item.get("filename")
            return filename if isinstance(filename, str) else ""
        if isinstance(item, Mapping):
            return {str(key): convert(nested) for key, nested in item.items()}
        if isinstance(item, list):
            return [convert(nested) for nested in item]
        return item

    return convert(value)


def _validate_body_transform_mode(
    route: ExternalRouteConfig, body_mode: BodyMode
) -> None:
    request_config = _object(route.request_config, "request_config")
    if body_mode not in {BodyMode.RAW, BodyMode.NONE}:
        return
    if (
        request_config.get("transform") is not None
        or request_config.get("resource_path") is not None
    ):
        raise ExternalConfigurationError(
            f"request transforms are not supported for {body_mode.value} bodies"
        )


def _rebuild_multipart_body(original: MultipartBody, transformed: Any) -> MultipartBody:
    """Apply top-level metadata transforms to the actual multipart form parts."""

    if not isinstance(transformed, Mapping):
        raise ExternalConfigurationError(
            "multipart request transforms must produce an object"
        )
    file_parts: list[tuple[dict[str, Any], MultipartPart]] = []
    for part in original.parts:
        if isinstance(part.value, bytes):
            file_parts.append(
                (
                    {
                        "filename": part.filename,
                        "content_type": part.content_type,
                        "size_bytes": len(part.value),
                    },
                    part,
                )
            )
    used: set[int] = set()
    rebuilt: list[MultipartPart] = []

    def append(name: str, value: Any) -> None:
        if value is None:
            return
        if isinstance(value, Mapping):
            candidates = [
                index
                for index, (descriptor, _) in enumerate(file_parts)
                if dict(value) == descriptor
            ]
            if not candidates:
                raise ExternalConfigurationError(
                    "multipart transforms cannot synthesize or mutate file values"
                )
            selected = next(
                (index for index in candidates if index not in used), candidates[0]
            )
            used.add(selected)
            part = file_parts[selected][1]
            rebuilt.append(
                MultipartPart(
                    name=name,
                    value=part.value,
                    filename=part.filename,
                    content_type=part.content_type,
                )
            )
            return
        if isinstance(value, (list, tuple, bytes, bytearray)):
            raise ExternalConfigurationError(
                "multipart transformed fields must be scalar values or files"
            )
        if not isinstance(value, (str, int, float, bool)):
            raise ExternalConfigurationError("multipart transformed field is invalid")
        rebuilt.append(MultipartPart(name=name, value=str(value)))

    for raw_name, value in transformed.items():
        name = str(raw_name)
        if not name:
            raise ExternalConfigurationError("multipart field names cannot be empty")
        if isinstance(value, list):
            for item in value:
                append(name, item)
        else:
            append(name, value)
    return MultipartBody(tuple(rebuilt))


def _path_parameters(
    route: ExternalRouteConfig, request_body: Any, query: Mapping[str, Any]
) -> dict[str, str | int]:
    request_config = _object(route.request_config, "request_config")
    configured = _object(request_config.get("path_parameters"), "path_parameters")
    context = {
        "resource": route.upstream_resource_id,
        "model": route.upstream_resource_id,
        "upstream_resource_id": route.upstream_resource_id,
    }
    source = {"body": request_body, "query": query, "context": context}
    # Common resource placeholders are always server-owned.  Supplying them even
    # when a particular template does not use them is harmless, while allowing a
    # client-derived value to replace one could route spend to a different model.
    result: dict[str, str | int] = dict(context)
    for name, rule in configured.items():
        if isinstance(rule, str):
            value = extract_value(source, rule, required=True)
        elif isinstance(rule, Mapping):
            if "value" in rule:
                value = rule["value"]
            else:
                path = rule.get("path")
                if not isinstance(path, str):
                    raise ExternalConfigurationError(
                        "path parameter rule requires a path or value"
                    )
                value = extract_value(
                    source, path, required=bool(rule.get("required", True))
                )
        else:
            value = rule
        if isinstance(value, bool) or not isinstance(value, (str, int)):
            raise HTTPException(status_code=400, detail="Invalid path parameter.")
        normalized_name = str(name)
        if normalized_name in context and value != context[normalized_name]:
            raise RequestRejectedError(
                "resource path parameters must use the configured resource"
            )
        result[normalized_name] = value
    return result


def _request_transform(
    route: ExternalRouteConfig,
    payload: Any,
    *,
    invocation_id: str,
    chute_name: str,
) -> Any:
    request_config = _object(route.request_config, "request_config")
    transform = request_config.get("transform")
    context = {
        "resource": route.upstream_resource_id,
        "model": route.upstream_resource_id,
        "invocation_id": invocation_id,
        "chute_name": chute_name,
    }
    result = transform_payload(payload, transform, request=payload, context=context)
    if isinstance(result, dict):
        for name in ("model", "resource", "upstream_resource_id"):
            if name in result:
                result[name] = route.upstream_resource_id
    resource_path = request_config.get("resource_path")
    if resource_path is not None:
        if not isinstance(resource_path, str) or not resource_path.strip():
            raise ExternalConfigurationError("resource_path must be a non-empty path")
        result = transform_payload(
            result,
            {
                "rewrite": [
                    {
                        "target": resource_path,
                        "source": "context",
                        "path": "resource",
                    }
                ]
            },
            request=payload,
            context=context,
        )
        pinned = extract_value(result, resource_path, required=True)
        if pinned != route.upstream_resource_id:
            raise ExternalConfigurationError("resource_path could not be pinned")
    return result


def _pricing_context(
    selected_cord: Mapping[str, Any], request: Request, dimensions: Mapping[str, Any]
) -> PricingContext:
    public_path = (
        getattr(request.state, "invocation_public_path", None)
        or selected_cord.get("public_api_path")
        or request.url.path
    )
    public_method = selected_cord.get("public_api_method") or request.method
    return PricingContext(
        cord=str(selected_cord.get("function") or selected_cord.get("path")),
        path=str(public_path),
        method=str(public_method),
        dimensions=dimensions,
    )


def _rule_available_at_acceptance(rule: PricingRule, context: PricingContext) -> bool:
    """Select immutable rate candidates without requiring output dimensions yet."""

    if not rule.scope.matches(context):
        return False
    if rule.effective_from is not None and context.at < rule.effective_from:
        return False
    return rule.effective_to is None or context.at < rule.effective_to


def _rule_snapshot(rule: PricingRule, index: int) -> dict[str, Any]:
    return {
        "id": rule.rule_id or f"accepted-rule-{index + 1}",
        "metric": rule.metric.value,
        "unit_price": str(rule.unit_price),
        "bucket": rule.bucket,
        "unit_size": str(rule.unit_size),
        "conditions": copy.deepcopy(dict(rule.conditions)),
        "scope": {
            key: value
            for key, value in {
                "cord": rule.scope.cord,
                "path": rule.scope.path,
                "method": rule.scope.method,
            }.items()
            if value is not None
        },
        "effective_from": (
            rule.effective_from.isoformat() if rule.effective_from is not None else None
        ),
        "effective_to": (
            rule.effective_to.isoformat() if rule.effective_to is not None else None
        ),
        "rounding": rule.rounding,
        "minimum_units": str(rule.minimum_units),
        "maximum_units": (
            str(rule.maximum_units) if rule.maximum_units is not None else None
        ),
        "match_group": rule.match_group,
        "priority": rule.priority,
        "fallback": rule.fallback,
    }


def _condition_dimension(dimensions: Mapping[str, Any], name: str) -> tuple[bool, Any]:
    if name in dimensions:
        return True, dimensions[name]
    value: Any = dimensions
    for part in name.split("."):
        if not isinstance(value, Mapping) or part not in value:
            return False, None
        value = value[part]
    return True, value


def _accepted_pricing_dimensions(
    rules: list[tuple[int, PricingRule]], dimensions: Mapping[str, Any]
) -> dict[str, Any]:
    """Retain only request values referenced by a snapshotted price condition."""

    retained: dict[str, Any] = {}
    for _, rule in rules:
        for name in rule.conditions:
            found, value = _condition_dimension(dimensions, name)
            if not found:
                continue
            try:
                retained[name] = orjson.loads(orjson.dumps(value))
            except (TypeError, orjson.JSONEncodeError) as exc:
                raise ExternalConfigurationError(
                    "pricing condition dimensions must be JSON-compatible"
                ) from exc
    return retained


def _pricing_context_snapshot(
    context: PricingContext,
    rules: list[tuple[int, PricingRule]],
) -> dict[str, Any]:
    return {
        "cord": context.cord,
        "path": context.path,
        "method": context.method,
        "dimensions": _accepted_pricing_dimensions(rules, context.dimensions),
        "at": context.at.isoformat(),
    }


async def _pricing_snapshot(
    current_user: User,
    chute: Chute,
    selected_cord: Mapping[str, Any],
    request: Request,
    request_dimensions: Mapping[str, Any],
) -> dict[str, Any]:
    override = await PriceOverride.get(current_user.user_id, chute.chute_id)
    if override is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Pricing is not configured for this external Chute.",
        )
    context = _pricing_context(selected_cord, request, request_dimensions)
    raw_rules = getattr(override, "pricing_rules", None)
    if raw_rules:
        try:
            parsed_rules = validate_conditional_pricing_coverage(raw_rules)
        except PricingConfigurationError as exc:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Pricing is not safely configured for this endpoint.",
            ) from exc
        candidates = [
            (index, rule)
            for index, rule in enumerate(parsed_rules)
            if _rule_available_at_acceptance(rule, context)
        ]
        if not candidates:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Pricing is not configured for this endpoint.",
            )
        free_account = bool(
            int(getattr(current_user, "permissions_bitmask", 0) or 0)
            & Permissioning.free_account.bitmask
        )
        invoice_billing = bool(
            getattr(request.state, "invoice_billing", False)
            or (
                int(getattr(current_user, "permissions_bitmask", 0) or 0)
                & Permissioning.invoice_billing.bitmask
            )
        )
        quota_free_invocation = bool(getattr(request.state, "free_invocation", False))
        # Invoice billing takes precedence over the broad free-account role,
        # matching the hosted usage tracker.  An explicitly free invocation
        # (for example, a fully discounted Chute) remains free for either role.
        free_invocation = quota_free_invocation or (
            free_account and not invoice_billing
        )
        return {
            "source": "rules",
            "rules": [_rule_snapshot(rule, index) for index, rule in candidates],
            "context": _pricing_context_snapshot(context, candidates),
            "accepted_at": context.at.isoformat(),
            "billing_chute_id": chute.chute_id,
            "free_invocation": free_invocation,
            "balance_exempt": free_invocation or invoice_billing,
            "invoice_billing": invoice_billing,
            "increment_invocation_quota": bool(
                quota_free_invocation
                and not free_account
                and not invoice_billing
                and float(getattr(chute, "discount", 0.0) or 0.0) < 1.0
            ),
        }

    legacy = {
        name: getattr(override, name, None)
        for name in (
            "per_million_in",
            "per_million_out",
            "per_step",
            "per_request",
            "cache_discount",
        )
    }
    if not any(value is not None for value in legacy.values()):
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Pricing is not configured for this endpoint.",
        )
    try:
        validate_legacy_pricing_rates(legacy)
    except PricingConfigurationError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Pricing is not safely configured for this endpoint.",
        ) from exc
    free_account = bool(
        int(getattr(current_user, "permissions_bitmask", 0) or 0)
        & Permissioning.free_account.bitmask
    )
    invoice_billing = bool(
        getattr(request.state, "invoice_billing", False)
        or (
            int(getattr(current_user, "permissions_bitmask", 0) or 0)
            & Permissioning.invoice_billing.bitmask
        )
    )
    quota_free_invocation = bool(getattr(request.state, "free_invocation", False))
    free_invocation = quota_free_invocation or (free_account and not invoice_billing)
    return {
        "source": "legacy",
        "legacy": legacy,
        "context": _pricing_context_snapshot(context, []),
        "accepted_at": context.at.isoformat(),
        "billing_chute_id": chute.chute_id,
        "free_invocation": free_invocation,
        "balance_exempt": free_invocation or invoice_billing,
        "invoice_billing": invoice_billing,
        "increment_invocation_quota": bool(
            quota_free_invocation
            and not free_account
            and not invoice_billing
            and float(getattr(chute, "discount", 0.0) or 0.0) < 1.0
        ),
    }


def _pricing_result(snapshot: Mapping[str, Any], usage: NormalizedUsage):
    if snapshot.get("source") == "rules":
        raw_context = _object(snapshot.get("context"), "pricing context")
        dimensions = _object(raw_context.get("dimensions"), "pricing dimensions")
        return price_usage(
            usage,
            snapshot.get("rules") or (),
            PricingContext(
                cord=raw_context.get("cord"),
                path=raw_context.get("path"),
                method=raw_context.get("method"),
                dimensions=dimensions,
                at=raw_context.get("at") or snapshot.get("accepted_at"),
            ),
        )
    return price_override(
        SimpleNamespace(**dict(snapshot.get("legacy") or {})),
        usage,
        combine_legacy_components=True,
    )


def _admission_cost_estimate(
    snapshot: Mapping[str, Any], usage: NormalizedUsage | None
) -> Decimal:
    """Return only the price implied by metrics known at admission time.

    Token and other completion-dependent quantities are intentionally not guessed;
    concurrency and daily-operation caps bound that residual exposure.
    """

    if usage is None:
        return Decimal(0)
    try:
        result = _pricing_result(snapshot, usage)
    except Exception:
        return Decimal(0)
    return result.amount if result.applied and result.amount > 0 else Decimal(0)


async def _running_budget_check(
    operation_id: str,
    pricing_snapshot: Mapping[str, Any],
    usage_value: Mapping[str, Any],
) -> tuple[bool, str | None]:
    try:
        usage = NormalizedUsage.from_mapping(usage_value)
    except UsageValidationError:
        return True, None
    estimate = _admission_cost_estimate(pricing_snapshot, usage)
    # The runtime budget check locks and reads the authoritative users.balance
    # row so a just-settled charge cannot be hidden by the materialized balance
    # view's refresh interval.
    async with get_session() as db:
        return await running_budget_available(
            db, operation_id=operation_id, estimated_paygo=estimate
        )


def _observed_cost_metadata(
    operation: ExternalOperation,
    pricing_snapshot: Mapping[str, Any],
    usage: NormalizedUsage,
    *,
    elapsed_seconds: float = 0.0,
) -> dict[str, Any]:
    metadata = dict(getattr(operation, "settlement_metadata", None) or {})
    metadata["observed_cost_estimate"] = str(
        session_exposure_estimate(
            metadata,
            _admission_cost_estimate(pricing_snapshot, usage),
            elapsed_seconds=elapsed_seconds,
        )
    )
    return metadata


async def _checkpoint_running_usage(
    operation_id: str,
    pricing_snapshot: Mapping[str, Any],
    value: Mapping[str, Any],
    *,
    settlement_metadata: Mapping[str, Any] | None = None,
    elapsed_seconds: float = 0.0,
    update_operation: Any = None,
) -> None:
    usage = NormalizedUsage.from_mapping(value)
    estimate = session_exposure_estimate(
        settlement_metadata,
        _admission_cost_estimate(pricing_snapshot, usage),
        elapsed_seconds=elapsed_seconds,
    )
    updater = update_operation or _update_operation
    await updater(
        operation_id,
        usage=usage.to_dict(),
        _settlement_metadata_patch={"observed_cost_estimate": str(estimate)},
    )


def _usage_config(
    route: ExternalRouteConfig, *, task: bool = False
) -> Mapping[str, Any] | None:
    response_config = _object(route.response_config, "response_config")
    operation_config = _object(route.operation_config, "operation_config")
    if task:
        task_config = _object(operation_config.get("task"), "operation_config.task")
        poll = _object(
            operation_config.get("poll", task_config.get("poll")),
            "operation_config.poll",
        )
        return (
            poll.get("usage")
            or task_config.get("usage")
            or operation_config.get("usage")
            or response_config.get("usage")
        )
    if route.operation_mode is ExternalOperationMode.REALTIME:
        realtime_value = operation_config.get("realtime")
        websocket_value = operation_config.get("websocket")
        if realtime_value is not None and websocket_value is not None:
            raise ExternalConfigurationError(
                "operation_config cannot define both realtime and websocket"
            )
        realtime = _object(
            realtime_value if realtime_value is not None else websocket_value,
            "operation_config.realtime",
        )
        return (
            realtime.get("usage")
            or operation_config.get("usage")
            or response_config.get("usage")
        )
    return response_config.get("usage") or operation_config.get("usage")


def _validate_metering_config(
    route: ExternalRouteConfig, pricing_snapshot: Mapping[str, Any]
) -> None:
    """Fail before spending upstream money when a configured unit cannot be observed."""

    usage_config = _usage_config(
        route, task=route.operation_mode is ExternalOperationMode.TASK
    )
    usage_mapping = UsageMapping.from_config(usage_config) if usage_config else None
    targets = (
        {field.target for field in usage_mapping.fields} if usage_mapping else set()
    )

    metric_groups = {
        "token": "tokens",
        "image": "images",
        "input_media_second": "input_media_seconds",
        "output_media_second": "output_media_seconds",
        "character": "characters",
        "count": "counts",
        "tool": "tools",
    }

    def require_target(metric: str, bucket: str | None) -> None:
        group = metric_groups.get(metric)
        if group is None:
            raise ExternalConfigurationError("pricing metric is not supported")
        expected = f"{group}.{bucket}" if bucket is not None else None
        covered = (
            expected in targets
            if expected
            else any(target.startswith(f"{group}.") for target in targets)
        )
        if not covered:
            suffix = f" bucket {bucket!r}" if bucket is not None else ""
            raise ExternalConfigurationError(
                f"pricing metric {metric!r}{suffix} requires a matching usage field"
            )

    if pricing_snapshot.get("source") == "rules":
        rules = parse_pricing_rules(pricing_snapshot.get("rules") or ())
        pricing_context = _object(pricing_snapshot.get("context"), "pricing context")
        accepted_dimensions = _object(
            pricing_context.get("dimensions"), "pricing dimensions"
        )
        for rule in rules:
            if rule.metric.value != "request":
                require_target(rule.metric.value, rule.bucket)
            for name, expected in rule.conditions.items():
                found, _ = _condition_dimension(accepted_dimensions, name)
                normalized_expected = (
                    {
                        str(operator).removeprefix("$"): operand
                        for operator, operand in expected.items()
                    }
                    if isinstance(expected, Mapping)
                    else {}
                )
                explicitly_absent = normalized_expected.get("exists") is False
                if (
                    not found
                    and not explicitly_absent
                    and f"dimensions.{name}" not in targets
                ):
                    raise ExternalConfigurationError(
                        f"pricing condition {name!r} requires a matching usage dimension"
                    )
        return
    legacy = _object(pricing_snapshot.get("legacy"), "legacy pricing")
    if legacy.get("per_million_in") is not None:
        require_target("token", "input")
    if legacy.get("per_million_out") is not None:
        require_target("token", "output")
    if (
        legacy.get("per_million_in") is not None
        and legacy.get("cache_discount") is not None
    ):
        require_target("token", "cached_input")
    if legacy.get("per_step") is not None:
        require_target("count", "steps")


def _extract_initial_task_usage(
    route: ExternalRouteConfig, *, request_body: Any
) -> NormalizedUsage:
    """Persist request-side task usage before the original request can disappear."""

    config = _usage_config(route, task=True)
    if not config:
        return NormalizedUsage(requests=1)
    compiled = UsageMapping.from_config(config)
    request_fields = tuple(
        field
        for field in compiled.fields
        if field.rule.source in {"request", "context"}
    )
    request_mapping = UsageMapping(fields=request_fields)
    return request_mapping.extract(
        request=request_body,
        response={},
        payload=None,
        context={"resource": route.upstream_resource_id},
    )


def _usage_with_request_count(usage: NormalizedUsage, requests: int) -> NormalizedUsage:
    values = usage.to_dict()
    values["requests"] = requests
    return NormalizedUsage.from_mapping(values)


def _extract_initial_stream_usage(
    route: ExternalRouteConfig, *, request_body: Any
) -> NormalizedUsage:
    """Capture request-derived quantities once before consuming stream events."""

    config = _usage_config(route)
    if not config:
        return NormalizedUsage(requests=1)
    compiled = UsageMapping.from_config(config)
    request_fields = tuple(
        field
        for field in compiled.fields
        if field.rule.source in {"request", "context"}
    )
    request_mapping = UsageMapping(fields=request_fields)
    observed = request_mapping.extract(
        request=request_body,
        response={},
        payload=None,
        context={"resource": route.upstream_resource_id},
    )
    return _usage_with_request_count(observed, 1)


def _extract_stream_observation_usage(
    route: ExternalRouteConfig,
    *,
    request_body: Any,
    response_body: Any,
    payload: Any | None,
) -> NormalizedUsage:
    """Extract one stream observation without replaying request-side quantities."""

    config = _usage_config(route)
    if not config:
        return NormalizedUsage(requests=0)
    compiled = UsageMapping.from_config(config)
    observation_fields = tuple(
        field
        for field in compiled.fields
        if field.rule.source in {"response", "payload"}
    )
    observation_mapping = UsageMapping(fields=observation_fields)
    observed = observation_mapping.extract(
        request=request_body,
        response=response_body,
        payload=payload,
        context={"resource": route.upstream_resource_id},
    )
    return _usage_with_request_count(observed, 0)


def _extract_usage(
    route: ExternalRouteConfig,
    *,
    request_body: Any,
    response_body: Any,
    payload: Any | None = None,
    task: bool = False,
) -> NormalizedUsage:
    config = _usage_config(route, task=task)
    if not config:
        return NormalizedUsage(requests=1)
    return extract_usage(
        config,
        request=request_body,
        response=response_body,
        payload=payload,
        context={"resource": route.upstream_resource_id},
    )


def _metrics_from_usage(
    usage: NormalizedUsage, pricing: Mapping[str, Any]
) -> dict[str, Any]:
    return {
        "it": float(usage.tokens.get("input", 0)),
        "ot": float(usage.tokens.get("output", 0)),
        "ct": float(usage.tokens.get("cached_input", 0)),
        "external_usage": usage.to_dict(),
        "pricing": dict(pricing),
    }


async def _record_free_invocation_usage(
    user_id: str,
    chute_id: str,
    paygo_amount: float,
    *,
    increment_invocation_quota: bool,
) -> None:
    """Advance the same quota and subscription-cap caches as hosted execution."""

    from api.config import get_subscription_tier, settings
    from api.invocation.util import (
        SUBSCRIPTION_CACHE_PREFIX,
        build_subscription_periods,
    )
    from api.user.schemas import InvocationQuota

    if increment_invocation_quota:
        quota_key = await InvocationQuota.quota_key(user_id, chute_id)
        await settings.redis_client.incrbyfloat(quota_key, 1.0)
    if paygo_amount <= 0:
        return
    (
        sub_quota,
        subscription_anchor,
        _,
        _,
    ) = await InvocationQuota.get_subscription_record(user_id)
    if get_subscription_tier(sub_quota) is None:
        return
    periods = build_subscription_periods(subscription_anchor)
    month_key = f"{SUBSCRIPTION_CACHE_PREFIX}_{periods['monthly_period']}:{user_id}"
    four_hour_key = (
        f"{SUBSCRIPTION_CACHE_PREFIX}_{periods['four_hour_period']}:{user_id}"
    )
    await settings.redis_client.incrbyfloat(month_key, paygo_amount)
    await settings.redis_client.expire(month_key, 35 * 86400)
    await settings.redis_client.incrbyfloat(four_hour_key, paygo_amount)
    await settings.redis_client.expire(four_hour_key, 5 * 3600)


def _settlement_retry_delay(attempt: int) -> float:
    return min(3600.0, 5.0 * (2 ** min(max(0, attempt - 1), 10)))


def _settlement_quarantine_attempt_limit() -> int:
    from api.config import settings

    return settings.external_settlement_quarantine_attempts


def _settlement_failure_code(error: Exception) -> str:
    if isinstance(error, UsageValidationError):
        return "invalid_usage"
    if isinstance(error, PricingConfigurationError):
        return "unpriceable_usage"
    return "settlement_error"


async def _record_settlement_failure(operation_id: str, error: Exception) -> None:
    now = datetime.now(timezone.utc)
    async with get_session() as db:
        await db.execute(
            text("SELECT pg_advisory_xact_lock(hashtextextended(:key, 0))").bindparams(
                key=f"external-settlement:{operation_id}"
            )
        )
        operation = await db.get(ExternalOperation, operation_id, with_for_update=True)
        if (
            not operation
            or operation.settlement_status in _RESOLVED_SETTLEMENT_STATUSES
            or operation.settlement_status == ExternalSettlementStatus.QUARANTINED.value
        ):
            return
        has_immutable_event = await external_usage_event_exists(db, operation_id)
        metadata = dict(operation.settlement_metadata or {})
        attempts = metadata.get("settlement_attempts", 0)
        attempts = attempts if isinstance(attempts, int) and attempts >= 0 else 0
        attempts += 1
        failure_code = _settlement_failure_code(error)
        metadata.update(
            {
                "settlement_attempts": attempts,
                "settlement_failure_code": failure_code,
                "settlement_last_attempt_at": now.isoformat(),
            }
        )
        # An immutable outbox event is already a final price decision. It must be
        # retried until delivery succeeds and can never be quarantined or written
        # off. Only pre-price failures are bounded into operator review.
        if (
            not has_immutable_event
            and attempts >= _settlement_quarantine_attempt_limit()
        ):
            metadata["settlement_quarantined_at"] = now.isoformat()
            metadata.pop("settlement_next_attempt_at", None)
            operation.settlement_metadata = metadata
            operation.settlement_status = ExternalSettlementStatus.QUARANTINED.value
            operation.next_poll_at = None
            operation.settled_at = None
            settlement_attempts.labels(outcome="quarantined").inc()
            return

        retry_at = now + timedelta(seconds=_settlement_retry_delay(attempts))
        metadata["settlement_next_attempt_at"] = retry_at.isoformat()
        operation.settlement_metadata = metadata
        operation.settlement_status = ExternalSettlementStatus.FAILED.value
        operation.next_poll_at = retry_at
        operation.settled_at = None


async def settle_operation(
    operation_id: str,
    usage: NormalizedUsage | None = None,
    *,
    billable: bool | None = None,
) -> None:
    """Durably record and atomically apply one terminal operation's usage."""
    event_enqueued = False
    partial_pricing_enqueued = False
    settlement_usage = usage
    resolved_billable: bool | None = billable
    try:
        try:
            async with get_session() as db:
                # The transaction-scoped lock makes the first accepted usage payload
                # immutable even when terminal hooks and reconciliation race.
                await db.execute(
                    text(
                        "SELECT pg_advisory_xact_lock(hashtextextended(:key, 0))"
                    ).bindparams(key=f"external-settlement:{operation_id}")
                )
                operation = await db.get(
                    ExternalOperation, operation_id, with_for_update=True
                )
                if not operation:
                    return
                if await external_usage_event_exists(db, operation_id):
                    # The first accepted settlement decision is immutable. In
                    # particular, a later hook cannot turn a durable charge into
                    # NOT_BILLABLE after an application failure.
                    raise _PersistedUsageEventReady
                if operation.settlement_status in {
                    ExternalSettlementStatus.SETTLED.value,
                    ExternalSettlementStatus.NOT_BILLABLE.value,
                    ExternalSettlementStatus.QUARANTINED.value,
                }:
                    return
                settlement = dict(operation.settlement_metadata or {})
                snapshot = dict(settlement.get("pricing") or {})
                user_id = operation.user_id
                chute_id = operation.chute_id or snapshot.get("billing_chute_id")
                app_id = (operation.request_metadata or {}).get("app_id")
                # Once an operator has corrected a quarantined settlement, the
                # persisted usage is authoritative. A pathologically delayed
                # completion hook must not overwrite that reviewed input with a
                # stale in-memory usage object.
                operator_reviewed_retry = (
                    settlement.get("settlement_operator_retry_at") is not None
                )
                if operator_reviewed_retry:
                    settlement_usage = NormalizedUsage.from_mapping(
                        operation.usage or {"requests": 1}
                    )
                elif settlement_usage is None:
                    settlement_usage = NormalizedUsage.from_mapping(
                        operation.usage or {"requests": 1}
                    )
                if operator_reviewed_retry:
                    persisted = settlement.get("billable")
                    resolved_billable = (
                        persisted if isinstance(persisted, bool) else True
                    )
                elif resolved_billable is None:
                    persisted = settlement.get("billable")
                    # Deliberately free paths persist false or transition directly to
                    # NOT_BILLABLE. Missing metadata after accepted work fails toward
                    # charging, matching the configured funded-account risk policy.
                    resolved_billable = (
                        persisted if isinstance(persisted, bool) else True
                    )
                if not isinstance(settlement_usage, NormalizedUsage):
                    raise UsageValidationError("settlement usage is unavailable")
                now = datetime.now(timezone.utc)
                if not resolved_billable:
                    settlement.update(
                        {
                            "billable": False,
                            "settlement_last_attempt_at": now.isoformat(),
                        }
                    )
                    settlement.pop("settlement_next_attempt_at", None)
                    operation.settlement_metadata = settlement
                    operation.usage = settlement_usage.to_dict()
                    operation.settlement_status = (
                        ExternalSettlementStatus.NOT_BILLABLE.value
                    )
                    operation.settled_at = now
                    operation.next_poll_at = None
                    settlement_attempts.labels(outcome="not_billable").inc()
                    return
                if (
                    not isinstance(chute_id, str)
                    or not chute_id
                    or not isinstance(user_id, str)
                    or not user_id
                ):
                    raise PricingConfigurationError(
                        "settlement billing identifiers are unavailable"
                    )
                result = _pricing_result(snapshot, settlement_usage)
                if (
                    not result.applied
                    or result.amount < 0
                    or (not result.complete and result.amount == 0)
                ):
                    raise PricingConfigurationError(
                        "accepted pricing snapshot has no safe applied amount"
                    )
                corrected_maximum = settlement.get(
                    "settlement_pricing_correction_max_amount"
                )
                if corrected_maximum is not None:
                    try:
                        corrected_maximum_value = Decimal(str(corrected_maximum))
                    except (ArithmeticError, TypeError, ValueError) as exc:
                        raise PricingConfigurationError(
                            "settlement pricing correction limit is invalid"
                        ) from exc
                    if (
                        not corrected_maximum_value.is_finite()
                        or corrected_maximum_value < 0
                        or result.amount > corrected_maximum_value
                    ):
                        raise PricingConfigurationError(
                            "settlement price exceeds its operator-reviewed limit"
                        )
                paygo_amount = result.amount
                free_invocation = bool(snapshot.get("free_invocation"))
                charged_amount = Decimal(0) if free_invocation else paygo_amount
                pricing_data = result.to_dict(decimal_as_string=True)
                pricing_data["charged_amount"] = str(charged_amount)
                pricing_data["paygo_amount"] = str(paygo_amount)
                metrics = _metrics_from_usage(settlement_usage, pricing_data)
                media_seconds = sum(
                    settlement_usage.input_media_seconds.values()
                ) + sum(settlement_usage.output_media_seconds.values())
                event_enqueued = await enqueue_external_usage_event(
                    db,
                    ExternalUsageEvent(
                        event_id=f"external-settlement:{operation_id}",
                        operation_id=operation_id,
                        user_id=user_id,
                        chute_id=chute_id,
                        app_id=app_id if isinstance(app_id, str) else None,
                        amount=charged_amount,
                        paygo_amount=paygo_amount,
                        input_tokens=Decimal(str(metrics.get("it", 0))),
                        output_tokens=Decimal(str(metrics.get("ot", 0))),
                        cached_tokens=Decimal(str(metrics.get("ct", 0))),
                        compute_time=round(float(media_seconds), 4),
                        track_task_completion=bool(
                            getattr(operation, "operation_mode", None)
                            == ExternalOperationMode.TASK.value
                            and getattr(operation, "status", None)
                            == ExternalOperationStatus.SUCCEEDED.value
                        ),
                        free_invocation=free_invocation,
                        increment_invocation_quota=bool(
                            snapshot.get("increment_invocation_quota", False)
                        ),
                        occurred_at=now,
                    ),
                )
                if event_enqueued:
                    partial_pricing_enqueued = not result.complete
                    metadata = dict(operation.settlement_metadata or {})
                    metadata.update(
                        {
                            "billable": True,
                            "result": pricing_data,
                            "pricing_complete": result.complete,
                            "pricing_missing_rule_count": result.missing_rule_count,
                            "settlement_last_attempt_at": now.isoformat(),
                            "settlement_delivery": "pending",
                        }
                    )
                    metadata.pop("settlement_next_attempt_at", None)
                    operation.settlement_metadata = metadata
                    operation.usage = settlement_usage.to_dict()
                # SETTLED is reserved for the transaction which applies usage_data,
                # deducts balance, and deletes the outbox row. If this process dies
                # after commit, the ordinary settlement reconciler sees this row due.
                operation.settlement_status = ExternalSettlementStatus.PENDING.value
                operation.settled_at = None
                operation.next_poll_at = now
        except _PersistedUsageEventReady:
            pass

        if partial_pricing_enqueued:
            settlement_attempts.labels(outcome="partial_priced").inc()
        receipt = await deliver_external_usage_event(operation_id)
        settlement_attempts.labels(
            outcome="applied" if receipt is not None else "already_applied"
        ).inc()
    except Exception as exc:
        logger.exception("Failed to settle external operation {}", operation_id)
        settlement_attempts.labels(outcome="failed").inc()
        await _record_settlement_failure(operation_id, exc)
        return

    if receipt is None:
        return

    # Process-local counters and quota caches advance only after the durable
    # database charge commits. They remain best-effort accelerators; usage_data
    # and the balance transaction are authoritative.
    try:
        from api.metrics.invocation import track_invocation_usage

        track_invocation_usage(
            receipt.chute_id,
            float(receipt.amount),
            receipt.compute_time,
            float(receipt.paygo_amount),
        )
    except Exception:
        logger.exception(
            "Failed to publish usage metrics for external operation {}", operation_id
        )
    if receipt.track_task_completion:
        try:
            track_request_completed(receipt.chute_id)
        except Exception:
            logger.exception(
                "Failed to track completion for external task {}", operation_id
            )
    if receipt.free_invocation:
        try:
            await _record_free_invocation_usage(
                receipt.user_id,
                receipt.chute_id,
                float(receipt.paygo_amount),
                increment_invocation_quota=receipt.increment_invocation_quota,
            )
        except Exception:
            # The durable database usage transaction is authoritative for balance and
            # reporting; admission caches are best-effort accelerators.
            logger.exception(
                "Failed to update free-invocation caches for external operation {}",
                operation_id,
            )


async def _create_operation(
    *,
    binding: ExternalChuteBinding,
    account: ExternalBackendAccount,
    chute: Chute,
    current_user: User,
    route: ExternalRouteConfig,
    selected_cord: Mapping[str, Any],
    pricing_snapshot: Mapping[str, Any],
    request: Request,
    invocation_id: str,
    idempotency_key: str | None,
    idempotency_fingerprint: str,
    initial_usage: NormalizedUsage | None = None,
    dispatch_recovery_seconds: float = 900.0,
    session_timeout_seconds: float | None = None,
) -> tuple[ExternalOperation, bool]:
    body_sha256 = getattr(request.state, "body_sha256", None)
    accepted_usage = initial_usage or NormalizedUsage(requests=1)
    admission_cost_estimate = _admission_cost_estimate(pricing_snapshot, accepted_usage)
    session_budget = None
    if route.operation_mode in {
        ExternalOperationMode.STREAM,
        ExternalOperationMode.REALTIME,
    }:
        if session_timeout_seconds is None:
            raise ExternalConfigurationError(
                "live external operations require a hard session timeout"
            )
        session_budget = compile_session_budget(
            account.connection_config,
            route.operation_config,
            max_session_seconds=session_timeout_seconds,
        )
        if admission_cost_estimate > session_budget.max_exposure_usd:
            raise HTTPException(
                status_code=status.HTTP_402_PAYMENT_REQUIRED,
                detail="The requested external operation exceeds its configured cost limit.",
            )
        admission_cost_estimate = max(
            admission_cost_estimate, session_budget.admission_exposure
        )
    if not math.isfinite(dispatch_recovery_seconds) or dispatch_recovery_seconds <= 0:
        raise ExternalConfigurationError("dispatch recovery deadline is invalid")
    operation = ExternalOperation(
        operation_id=str(uuid.uuid4()),
        user_id=current_user.user_id,
        account_id=account.account_id,
        binding_id=binding.binding_id,
        chute_id=chute.chute_id,
        cord_path=str(selected_cord.get("path")),
        operation_mode=route.operation_mode.value,
        protocol=route.protocol,
        status=ExternalOperationStatus.PENDING.value,
        settlement_status=ExternalSettlementStatus.PENDING.value,
        idempotency_key=idempotency_key,
        route_snapshot=route.model_dump(mode="json"),
        usage=accepted_usage.to_dict(),
        next_poll_at=datetime.now(timezone.utc)
        + timedelta(seconds=dispatch_recovery_seconds),
        request_metadata={
            key: value
            for key, value in {
                "invocation_id": invocation_id,
                "method": request.method,
                "path": request.url.path,
                "body_sha256": body_sha256,
                "idempotency_fingerprint": idempotency_fingerprint,
                "app_id": getattr(request.state, "oauth_app_id", None),
            }.items()
            if value is not None
        },
        settlement_metadata={
            **(
                {"session_budget": session_budget.snapshot()}
                if session_budget is not None
                else {}
            ),
            "pricing": dict(pricing_snapshot),
            "admission_cost_estimate": str(admission_cost_estimate),
        },
    )
    async with get_session() as db:
        # The share locks close the stale-read window with management updates. Once
        # this transaction inserts the active operation, the mutation guard keeps
        # credentials and destination configuration stable until work terminates.
        await _lock_transport_snapshot(db, binding, account)
        if idempotency_key:
            lock_key = f"external-operation:{binding.binding_id}:{current_user.user_id}:{idempotency_key}"
            await db.execute(
                text("SELECT pg_advisory_xact_lock(hashtextextended(:key, 0))"),
                {"key": lock_key},
            )
            existing = (
                await db.execute(
                    select(ExternalOperation).where(
                        ExternalOperation.binding_id == binding.binding_id,
                        ExternalOperation.user_id == current_user.user_id,
                        ExternalOperation.idempotency_key == idempotency_key,
                    )
                )
            ).scalar_one_or_none()
            if existing:
                metadata = existing.request_metadata or {}
                stored_fingerprint = metadata.get("idempotency_fingerprint")
                if (
                    stored_fingerprint != idempotency_fingerprint
                    if stored_fingerprint is not None
                    else metadata.get("body_sha256") != body_sha256
                ):
                    raise HTTPException(
                        status_code=status.HTTP_409_CONFLICT,
                        detail="The idempotency key was already used for a different request.",
                    )
                return existing, True
        try:
            await enforce_external_admission(
                db,
                account_id=account.account_id,
                user_id=current_user.user_id,
                operation_mode=route.operation_mode,
                connection_config=account.connection_config,
                estimated_paygo=admission_cost_estimate,
                free_invocation=bool(pricing_snapshot.get("free_invocation")),
                balance_exempt=bool(pricing_snapshot.get("balance_exempt")),
            )
        except ExternalAdmissionRejected as exc:
            raise HTTPException(status_code=exc.status_code, detail=exc.detail) from exc
        db.add(operation)
    return operation, False


async def _existing_idempotent_operation(
    *,
    binding_id: str,
    user_id: str,
    idempotency_key: str | None,
    body_sha256: str | None,
    idempotency_fingerprint: str,
) -> ExternalOperation | None:
    if not idempotency_key:
        return None
    async with get_session(readonly=True) as db:
        operation = (
            await db.execute(
                select(ExternalOperation).where(
                    ExternalOperation.binding_id == binding_id,
                    ExternalOperation.user_id == user_id,
                    ExternalOperation.idempotency_key == idempotency_key,
                )
            )
        ).scalar_one_or_none()
    if operation:
        metadata = operation.request_metadata or {}
        stored_fingerprint = metadata.get("idempotency_fingerprint")
        differs = (
            stored_fingerprint != idempotency_fingerprint
            if stored_fingerprint is not None
            else metadata.get("body_sha256") != body_sha256
        )
        if differs:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="The idempotency key was already used for a different request.",
            )
    return operation


def _idempotency_key(request: Request, route: ExternalRouteConfig) -> str | None:
    if route.operation_mode is not ExternalOperationMode.TASK:
        return None
    value = request.headers.get("idempotency-key")
    if value is None:
        return None
    if (
        not value
        or len(value) > 255
        or any(character in value for character in "\r\n\x00")
    ):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Idempotency-Key must contain 1-255 safe characters.",
        )
    return value


def _idempotency_fingerprint(request: Request, selected_cord: Mapping[str, Any]) -> str:
    material = {
        "method": request.method.upper(),
        "path": request.url.path,
        "query": request.url.query,
        "cord_path": selected_cord.get("path"),
        "cord_function": selected_cord.get("function"),
        "body_sha256": getattr(request.state, "body_sha256", None),
    }
    return hashlib.sha256(
        orjson.dumps(material, option=orjson.OPT_SORT_KEYS)
    ).hexdigest()


def _idempotent_task_response(
    operation: ExternalOperation, request: Request
) -> Response:
    invocation_id = (operation.request_metadata or {}).get("invocation_id") or str(
        uuid.uuid4()
    )
    payload = {
        "id": operation.operation_id,
        "status": operation.status,
        "status_url": operation_url(operation.operation_id),
    }
    return Response(
        orjson.dumps(payload),
        status_code=(
            200 if operation.status in _TERMINAL_STATUSES else status.HTTP_202_ACCEPTED
        ),
        media_type="application/json",
        headers=build_response_headers(
            request,
            {
                "X-Chutes-InvocationID": invocation_id,
                "X-Chutes-OperationID": operation.operation_id,
                "Location": operation_url(operation.operation_id),
                "Cache-Control": "private, no-store",
                "X-Content-Type-Options": "nosniff",
            },
        ),
    )


async def _update_operation(operation_id: str, **values: Any) -> None:
    metadata_patch = values.pop("_settlement_metadata_patch", None)
    persisted_metadata = values.pop("settlement_metadata", None)
    if metadata_patch is not None and persisted_metadata is not None:
        raise TypeError("settlement metadata updates are ambiguous")
    next_settlement_status = values.get("settlement_status")
    async with get_session() as db:
        if next_settlement_status is not None:
            await db.execute(
                text(
                    "SELECT pg_advisory_xact_lock(hashtextextended(:key, 0))"
                ).bindparams(key=f"external-settlement:{operation_id}")
            )
        operation = await db.get(ExternalOperation, operation_id, with_for_update=True)
        if not operation:
            return
        if getattr(
            operation,
            "settlement_status",
            ExternalSettlementStatus.PENDING.value,
        ) in {
            *_RESOLVED_SETTLEMENT_STATUSES,
            ExternalSettlementStatus.QUARANTINED.value,
        }:
            # Presentation/error writers can arrive after the accounting path.
            # Once billing is resolved or held for an operator they may not alter
            # either outcome or its audit metadata.
            return
        if next_settlement_status is not None and await external_usage_event_exists(
            db, operation_id
        ):
            # The immutable event is the winning billing decision. In particular,
            # a late pre-acceptance failure cannot mark its operation free.
            return
        next_status = values.get("status")
        if (
            operation.status in _TERMINAL_STATUSES
            and next_status is not None
            and next_status != operation.status
        ):
            # A deadline reaper and the live producer can finish at nearly the same
            # instant. Once either side commits a terminal state, a delayed writer
            # must not resurrect the operation into a different outcome.
            return
        effective_metadata_patch = (
            metadata_patch if metadata_patch is not None else persisted_metadata
        )
        if effective_metadata_patch is not None:
            if not isinstance(effective_metadata_patch, Mapping):
                raise TypeError("settlement metadata patch must be an object")
            metadata = dict(operation.settlement_metadata or {})
            metadata.update(effective_metadata_patch)
            operation.settlement_metadata = metadata
        for name, value in values.items():
            setattr(operation, name, value)


def _settlement_metadata_for(
    operation: ExternalOperation, *, billable: bool
) -> dict[str, Any]:
    metadata = dict(getattr(operation, "settlement_metadata", None) or {})
    metadata["billable"] = billable
    return metadata


def _retry_delay(
    response: BufferedResponse, policy: RetryPolicy, attempt: int
) -> float:
    observed_headers = getattr(response, "private_headers", response.headers)
    for name in policy.retry_after_headers:
        value = observed_headers.get(name.lower())
        if not value:
            continue
        try:
            return min(policy.maximum_delay_seconds, max(0.0, float(value)))
        except ValueError:
            with suppress(ValueError, TypeError, OverflowError):
                parsed = parsedate_to_datetime(value)
                if parsed.tzinfo is None:
                    parsed = parsed.replace(tzinfo=timezone.utc)
                return min(
                    policy.maximum_delay_seconds,
                    max(0.0, (parsed - datetime.now(timezone.utc)).total_seconds()),
                )
    ceiling = min(
        policy.maximum_delay_seconds,
        policy.base_delay_seconds * (2 ** max(0, attempt - 1)),
    )
    return random.uniform(0, ceiling) if ceiling else 0.0


def _body_retryable(response: BufferedResponse, route: ExternalRouteConfig) -> bool:
    operation_config = _object(route.operation_config, "operation_config")
    retry = _object(operation_config.get("retry"), "operation_config.retry")
    path = retry.get("body_status_path")
    values = retry.get("body_statuses", [])
    if not path:
        return False
    if not isinstance(path, str) or not isinstance(values, list):
        raise ExternalConfigurationError("body retry configuration is invalid")
    body = _json_or_none(response.body)
    return body is not None and extract_value(body, path) in values


async def _execute_with_retry(
    executor: ExternalExecutor,
    profile: Any,
    outbound: OutboundRequest,
    route: ExternalRouteConfig,
) -> BufferedResponse | StreamingResponse:
    policy = retry_policy(route)
    method_idempotent = profile.method in {"GET", "HEAD", "PUT", "DELETE"}
    attempts = (
        policy.max_attempts if (method_idempotent or policy.retry_non_idempotent) else 1
    )
    for attempt in range(1, attempts + 1):
        response = await executor.execute(profile, outbound)
        if isinstance(response, StreamingResponse):
            return response
        retryable = response.status_code in policy.retry_statuses or _body_retryable(
            response, route
        )
        if not retryable or attempt == attempts:
            return response
        await asyncio.sleep(_retry_delay(response, policy, attempt))
    raise AssertionError("retry loop exhausted")


def _dispatch_recovery_seconds(
    profile: Any, attempts: int, maximum_delay: float
) -> float:
    """Bound crash recovery after every possible outbound attempt and backoff."""

    timeout = getattr(getattr(profile, "timeout", None), "total", 300.0)
    try:
        timeout = float(timeout)
    except (TypeError, ValueError):
        timeout = 300.0
    if not math.isfinite(timeout) or timeout <= 0:
        timeout = 300.0
    attempts = max(1, int(attempts))
    return timeout * attempts + max(0.0, maximum_delay) * (attempts - 1) + 60.0


def _result_descriptor(
    extracted: Any,
    operation_id: str,
    *,
    partial: bool = False,
    response_config: Mapping[str, Any] | None = None,
    now: datetime,
) -> dict[str, Any] | None:
    artifacts = []
    for index, item in enumerate(extracted.artifacts):
        expires_at = normalize_artifact_expiration(
            item.expires_at, response_config, now=now
        ).isoformat()
        attributes = dict(item.metadata)
        attributes["local_path"] = artifact_url(operation_id, index)
        artifacts.append(
            {
                "kind": item.kind,
                "reference": item.source_url,
                "content_type": item.content_type,
                "size_bytes": item.size_bytes,
                "expires_at": expires_at,
                "attributes": attributes,
            }
        )
    if not artifacts and extracted.result is None:
        return None
    return {
        "status": ExternalResultStatus.PARTIAL.value
        if partial
        else ExternalResultStatus.COMPLETE.value,
        "artifacts": artifacts,
        "metadata": {"has_inline_result": extracted.result is not None},
    }


def _artifact_replacements(descriptor: Mapping[str, Any] | None) -> dict[str, str]:
    if not descriptor:
        return {}
    replacements: dict[str, str] = {}
    for artifact in descriptor.get("artifacts", []):
        if not isinstance(artifact, Mapping):
            continue
        reference = artifact.get("reference")
        attributes = artifact.get("attributes")
        local_path = (
            attributes.get("local_path") if isinstance(attributes, Mapping) else None
        )
        if isinstance(reference, str) and isinstance(local_path, str):
            replacements[reference] = local_path
    return replacements


def _task_mapping(route: ExternalRouteConfig) -> Mapping[str, Any]:
    operation_config = _object(route.operation_config, "operation_config")
    response_config = _object(route.response_config, "response_config")
    mapping = (
        operation_config.get("submit_mapping")
        or operation_config.get("task_mapping")
        or response_config.get("task")
    )
    if not isinstance(mapping, Mapping):
        raise ExternalConfigurationError("task response mapping is not configured")
    return mapping


def _task_terminal_billable(route: ExternalRouteConfig, operation_status: str) -> bool:
    if operation_status == ExternalOperationStatus.SUCCEEDED.value:
        return True
    operation = _object(route.operation_config, "operation_config")
    task = _object(operation.get("task"), "operation_config.task")
    configured = task.get(
        "billable_statuses", operation.get("billable_terminal_statuses")
    )
    if configured is None:
        poll = _object(operation.get("poll", task.get("poll")), "operation_config.poll")
        configured = poll.get("billable_statuses")
    if configured is None:
        # Defensive compatibility for a legacy persisted snapshot. Route-save
        # validation requires an explicit policy for every new task route.
        configured = {
            ExternalOperationStatus.FAILED.value,
            ExternalOperationStatus.CANCELLED.value,
        }
    if not isinstance(configured, (list, tuple, set, frozenset)) or any(
        not isinstance(item, str) for item in configured
    ):
        raise ExternalConfigurationError("task billable statuses must be an array")
    return operation_status in configured


def _bill_ambiguous_transport_failure(route: ExternalRouteConfig) -> bool:
    operation = _object(route.operation_config, "operation_config")
    configured = operation.get("bill_ambiguous_transport_errors", False)
    if not isinstance(configured, bool):
        raise ExternalConfigurationError(
            "bill_ambiguous_transport_errors must be a boolean"
        )
    return configured


def _billable_http_statuses(route: ExternalRouteConfig) -> frozenset[int]:
    operation = _object(route.operation_config, "operation_config")
    configured = operation.get("billable_http_statuses")
    if configured is None:
        legacy = operation.get("bill_statuses", [])
        configured = (
            legacy
            if isinstance(legacy, list)
            and all(
                isinstance(item, int) and not isinstance(item, bool) for item in legacy
            )
            else []
        )
    if not isinstance(configured, list) or any(
        isinstance(item, bool) or not isinstance(item, int) or item < 400 or item > 599
        for item in configured
    ):
        raise ExternalConfigurationError(
            "billable_http_statuses must contain HTTP error status codes"
        )
    return frozenset(configured)


def _validate_retry_billing_policy(
    route: ExternalRouteConfig, profile: Any, billable_statuses: frozenset[int]
) -> None:
    operation = _object(route.operation_config, "operation_config")
    retry = _object(operation.get("retry"), "operation_config.retry")
    raw_non_idempotent = retry.get("retry_non_idempotent", False)
    if not isinstance(raw_non_idempotent, bool):
        raise ExternalConfigurationError("retry_non_idempotent must be a boolean")
    policy = retry_policy(route)
    attempts = (
        policy.max_attempts
        if profile.method in {"GET", "HEAD", "PUT", "DELETE"}
        or policy.retry_non_idempotent
        else 1
    )
    if attempts > 1 and billable_statuses & policy.retry_statuses:
        raise ExternalConfigurationError(
            "billable HTTP statuses cannot also be retried"
        )


def _validate_raw_stream_usage(route: ExternalRouteConfig, profile: Any) -> None:
    config = _usage_config(route)
    if not config:
        return
    compiled = UsageMapping.from_config(config)
    observable_headers = {
        *profile.allowed_response_headers,
        *profile.private_response_headers,
    }
    for field in compiled.fields:
        rule = field.rule
        if rule.source in {"request", "context"} or not rule.paths:
            continue
        if rule.source != "response":
            raise ExternalConfigurationError(
                "raw stream usage can only map request, context, or response headers"
            )
        for path in rule.paths:
            if (
                len(path.parts) < 2
                or path.parts[0] != "headers"
                or not isinstance(path.parts[1], str)
                or path.parts[1] != path.parts[1].lower()
                or path.parts[1] not in observable_headers
            ):
                raise ExternalConfigurationError(
                    "raw stream response usage must map an allowed lowercase header"
                )


def _bill_partial_stream(route: ExternalRouteConfig) -> bool:
    response = _object(route.response_config, "response_config")
    configured = response.get("bill_partial_streams", True)
    if not isinstance(configured, bool):
        raise ExternalConfigurationError("bill_partial_streams must be a boolean")
    return configured


def _task_timeout_seconds(route: ExternalRouteConfig) -> float:
    operation = _object(route.operation_config, "operation_config")
    task = _object(operation.get("task"), "operation_config.task")
    configured = task.get(
        "timeout_seconds", operation.get("task_timeout_seconds", 24 * 3600)
    )
    if isinstance(configured, bool) or not isinstance(configured, (int, float)):
        raise ExternalConfigurationError("task_timeout_seconds must be a number")
    value = float(configured)
    if not math.isfinite(value) or value < 60 or value > 30 * 86400:
        raise ExternalConfigurationError(
            "task_timeout_seconds must be between 60 and 2592000"
        )
    return value


async def _record_accepted_failure(
    *,
    operation: ExternalOperation,
    route: ExternalRouteConfig,
    request_body: Any,
    response_body: Any,
    code: str = "invalid_response",
) -> None:
    """Persist and bill work after a successful upstream acceptance response."""

    await _update_operation(
        operation.operation_id,
        status=ExternalOperationStatus.FAILED.value,
        settlement_metadata=_settlement_metadata_for(operation, billable=True),
        error={
            "message": "The external service returned an invalid response.",
            "code": code,
            "retryable": True,
            "details": {},
        },
        finished_at=datetime.now(timezone.utc),
    )
    try:
        usage = _extract_usage(
            route,
            request_body=request_body,
            response_body=response_body,
            task=route.operation_mode is ExternalOperationMode.TASK,
        )
    except Exception:
        # A broken response/usage mapping must not turn accepted paid work into a
        # free invocation.  Request pricing can still be applied from this minimum.
        logger.warning(
            "Could not extract usage for accepted external operation {}",
            operation.operation_id,
        )
        usage = NormalizedUsage(requests=1)
    await settle_operation(operation.operation_id, usage, billable=True)


async def _finalize_interrupted_invocation(
    *,
    request: Request,
    operation: ExternalOperation,
    route: ExternalRouteConfig,
    request_body: Any,
    response_body: Any,
    upstream_accepted: bool,
    accepted_billable: bool,
    ambiguous_billable: bool,
) -> None:
    """Leave a terminal, settled operation when shutdown interrupts an invocation."""

    if route.operation_mode is ExternalOperationMode.TASK:
        async with get_session(readonly=True) as db:
            durable = await db.get(ExternalOperation, operation.operation_id)
        if (
            durable is not None
            and durable.upstream_operation_id
            and durable.status != ExternalOperationStatus.PENDING.value
        ):
            # Submission completed before cancellation reached the request task.
            # The durable remote identity belongs to the poller now; failing it
            # here would discard a recoverable result while the provider works.
            request.state.external_attempt_billable = True
            return

    billable = accepted_billable if upstream_accepted else ambiguous_billable
    if billable:
        request.state.external_attempt_billable = True
    try:
        usage = _extract_usage(
            route,
            request_body=request_body,
            response_body=response_body,
            task=route.operation_mode is ExternalOperationMode.TASK,
        )
    except Exception:
        logger.warning(
            "Could not extract usage for interrupted external operation {}",
            operation.operation_id,
        )
        usage = NormalizedUsage(requests=1)
    try:
        await _update_operation(
            operation.operation_id,
            status=ExternalOperationStatus.FAILED.value,
            settlement_metadata=_settlement_metadata_for(operation, billable=billable),
            error={
                "message": "External execution was interrupted before completion.",
                "code": "execution_interrupted",
                "retryable": True,
                "details": {},
            },
            finished_at=datetime.now(timezone.utc),
        )
    except asyncio.CancelledError:
        raise
    except Exception:
        logger.exception(
            "Failed to record interrupted external invocation {}",
            operation.operation_id,
        )
    try:
        await settle_operation(operation.operation_id, usage, billable=billable)
    except asyncio.CancelledError:
        raise
    except Exception:
        logger.exception(
            "Failed to settle interrupted external invocation {}",
            operation.operation_id,
        )


def _schema_for_output(
    route: ExternalRouteConfig, selected_cord: Mapping[str, Any]
) -> dict[str, Any] | None:
    response_config = _object(route.response_config, "response_config")
    if not _schema_validation_enabled(
        response_config, "validate_output_schema", "response_config"
    ):
        return None
    schema = selected_cord.get("output_schema")
    if not isinstance(schema, Mapping) or not schema:
        raise ExternalConfigurationError(
            "response_config.validate_output_schema=true requires a non-empty "
            "Cord output schema"
        )
    return dict(schema)


async def _validate_task_submission_contract(
    *,
    route: ExternalRouteConfig,
    selected_cord: Mapping[str, Any],
    response: BufferedResponse,
    response_json: Any,
) -> str:
    """Apply an explicitly enabled submit contract, separate from task results."""

    operation = _object(route.operation_config, "operation_config")
    contract = _object(
        operation.get("submission_contract"),
        "operation_config.submission_contract",
    )
    unknown = set(contract) - {"enabled", "output_schema", "output_content_type"}
    if unknown:
        raise ExternalConfigurationError(
            "submission_contract contains unsupported fields"
        )
    enabled = contract.get("enabled", False)
    if not isinstance(enabled, bool):
        raise ExternalConfigurationError(
            "submission_contract.enabled must be a boolean"
        )
    if not enabled:
        return "application/json"
    if route.operation_mode is not ExternalOperationMode.TASK:
        raise ExternalConfigurationError(
            "submission_contract.enabled is only supported for task routes"
        )
    schema = (
        contract["output_schema"]
        if "output_schema" in contract
        else selected_cord.get("output_schema")
    )
    if schema is not None and not isinstance(schema, Mapping):
        raise ExternalConfigurationError(
            "submission_contract.output_schema must be an object"
        )
    content_type = (
        contract["output_content_type"]
        if "output_content_type" in contract
        else selected_cord.get("output_content_type")
    )
    public_content_type = "application/json"
    if content_type is not None:
        public_content_type = (
            _validated_output_content_type(
                response.headers, {"output_content_type": content_type}
            )
            or public_content_type
        )
    if not (isinstance(schema, Mapping) and schema) and content_type is None:
        raise ExternalConfigurationError(
            "submission_contract.enabled=true requires a non-empty output schema "
            "or output content type"
        )
    if schema:
        await _validate_schema(response_json, schema, "task submission response")
    return public_content_type


async def _handle_buffered(
    *,
    response: BufferedResponse,
    route: ExternalRouteConfig,
    chute: Chute,
    selected_cord: Mapping[str, Any],
    operation: ExternalOperation,
    request: Request,
    request_body: Any,
    invocation_id: str,
    task_timeout_seconds: float | None = None,
) -> Response:
    successful = 200 <= response.status_code < 300
    is_json, response_json = _json_response(response.body)
    if not successful:
        billable = response.status_code in _billable_http_statuses(route)
        if billable:
            request.state.external_attempt_billable = True
        try:
            usage = _extract_usage(
                route,
                request_body=request_body,
                response_body=response_json or {},
            )
        except Exception:
            if not billable:
                raise
            logger.warning(
                "Could not extract usage from billable error response for operation {}",
                operation.operation_id,
            )
            usage = NormalizedUsage(requests=1)
        await _update_operation(
            operation.operation_id,
            status=ExternalOperationStatus.FAILED.value,
            settlement_metadata=_settlement_metadata_for(operation, billable=billable),
            error={
                "message": "The external service could not complete the request.",
                "code": "upstream_rejected",
                "retryable": response.status_code in retry_policy(route).retry_statuses,
                "details": {},
            },
            finished_at=datetime.now(timezone.utc),
        )
        await settle_operation(operation.operation_id, usage, billable=billable)
        public_status = (
            429
            if response.status_code == 429
            else (400 if response.status_code in {400, 404, 409, 422} else 502)
        )
        if public_status == 429:
            track_request_rate_limited(chute.chute_id)
        raise HTTPException(
            status_code=public_status,
            detail=(
                "External capacity is temporarily unavailable."
                if public_status == 429
                else "The external service could not complete the request."
            ),
            headers=(
                {
                    "Retry-After": str(
                        max(
                            1, math.ceil(_retry_delay(response, retry_policy(route), 1))
                        )
                    )
                }
                if public_status == 429
                else None
            ),
        )

    now = datetime.now(timezone.utc)
    if route.operation_mode is ExternalOperationMode.TASK:
        if not is_json:
            raise ExternalInvocationError(
                "task submission returned an invalid response"
            )
        submission_content_type = await _validate_task_submission_contract(
            route=route,
            selected_cord=selected_cord,
            response=response,
            response_json=response_json,
        )
        extracted = extract_task(
            _task_mapping(route),
            request=request_body,
            response=response_json,
            context={"resource": route.upstream_resource_id},
        )
        if not extracted.task_id:
            raise ExternalInvocationError(
                "task submission did not return an operation id"
            )
        operation_status = _canonical_status(
            extracted.status, ExternalOperationStatus.SUBMITTED.value
        )
        descriptor = _result_descriptor(
            extracted,
            operation.operation_id,
            partial=operation_status not in _TERMINAL_STATUSES,
            response_config=route.response_config,
            now=now,
        )
        public_rules = _public_mapping(
            route,
            chute_name=chute.name,
            invocation_id=invocation_id,
            operation_id=operation.operation_id,
        )
        persisted_result_limit = inline_result_limit(route.operation_config)
        if (
            descriptor is not None
            and extracted.result is not None
            and persisted_result_limit is not None
        ):
            metadata = dict(descriptor.get("metadata") or {})
            metadata["inline_result"] = bounded_inline_result(
                scrub_public_response(
                    _replace_scalar(
                        extracted.result, extracted.task_id, operation.operation_id
                    ),
                    public_rules,
                    artifact_urls=_artifact_replacements(descriptor),
                ),
                persisted_result_limit,
            )
            descriptor["metadata"] = metadata
        usage = _extract_initial_task_usage(route, request_body=request_body)
        operation_values: dict[str, Any] = {
            "upstream_operation_id": extracted.task_id,
            "upstream_status": None,
            "status": operation_status,
            "submitted_at": now,
            "started_at": (
                now
                if operation_status == ExternalOperationStatus.RUNNING.value
                else None
            ),
            "finished_at": now if operation_status in _TERMINAL_STATUSES else None,
            "next_poll_at": (
                now if operation_status not in _TERMINAL_STATUSES else None
            ),
            "usage": usage.to_dict(),
            "result_descriptor": descriptor,
            "expires_at": now
            + timedelta(
                seconds=(
                    task_timeout_seconds
                    if task_timeout_seconds is not None
                    else _task_timeout_seconds(route)
                )
            ),
        }
        terminal_billable = None
        if operation_status in _TERMINAL_STATUSES:
            terminal_billable = _task_terminal_billable(route, operation_status)
            operation_values["settlement_metadata"] = _settlement_metadata_for(
                operation, billable=terminal_billable
            )
        await _update_operation(
            operation.operation_id,
            **operation_values,
        )
        if operation_status in _TERMINAL_STATUSES:
            usage = _extract_usage(
                route,
                request_body=request_body,
                response_body=response_json,
                task=True,
            )
            await _update_operation(operation.operation_id, usage=usage.to_dict())
            await settle_operation(
                operation.operation_id,
                usage,
                billable=bool(terminal_billable),
            )
        public_value = _replace_scalar(
            response_json, extracted.task_id, operation.operation_id
        )
        public_value = scrub_public_response(
            public_value,
            public_rules,
            artifact_urls=_artifact_replacements(descriptor),
        )
        if isinstance(public_value, dict):
            public_value.setdefault("id", operation.operation_id)
            public_value.setdefault("status", operation_status)
            public_value["status_url"] = operation_url(operation.operation_id)
        safe_headers = _safe_public_headers(response.headers)
        for name in tuple(safe_headers):
            if name.lower() == "content-type":
                safe_headers.pop(name)
        safe_headers["Content-Type"] = submission_content_type
        safe_headers["Cache-Control"] = "private, no-store"
        safe_headers["X-Content-Type-Options"] = "nosniff"
        headers = build_response_headers(
            request,
            {
                **safe_headers,
                "X-Chutes-InvocationID": invocation_id,
                "X-Chutes-OperationID": operation.operation_id,
                "Location": operation_url(operation.operation_id),
            },
        )
        return Response(
            orjson.dumps(public_value),
            status_code=202 if operation_status not in _TERMINAL_STATUSES else 200,
            media_type=submission_content_type,
            headers=headers,
        )

    configured_content_type = _validated_output_content_type(
        response.headers, selected_cord
    )
    output_schema = _schema_for_output(route, selected_cord)
    if (_is_json_content_type(response.headers) or output_schema) and not is_json:
        raise ExternalInvocationError("successful response was not valid JSON")
    descriptor = None
    if response_json is not None:
        response_config = _object(route.response_config, "response_config")
        artifact_config = response_config.get("artifacts")
        if artifact_config:
            extracted = extract_task(
                {"artifacts": artifact_config},
                request=request_body,
                response=response_json,
                context={"resource": route.upstream_resource_id},
            )
            descriptor = _result_descriptor(
                extracted,
                operation.operation_id,
                response_config=route.response_config,
                now=now,
            )
    if not is_json:
        if configured_content_type is None:
            raise ExternalConfigurationError(
                "non-JSON responses require a Cord output_content_type"
            )
        public_body = response.body
        public_media_type = configured_content_type
    else:
        if output_schema:
            await _validate_schema(response_json, output_schema, "response")
        public_value = scrub_public_response(
            response_json,
            _public_mapping(
                route,
                chute_name=chute.name,
                invocation_id=invocation_id,
                operation_id=operation.operation_id,
            ),
            artifact_urls=_artifact_replacements(descriptor),
        )
        public_body = orjson.dumps(public_value)
        public_media_type = configured_content_type or "application/json"

    usage = _extract_usage(
        route,
        request_body=request_body,
        response_body=response_json if response_json is not None else {},
    )
    safe_headers = _safe_public_headers(response.headers)
    for name in tuple(safe_headers):
        if name.lower() == "content-type":
            safe_headers.pop(name)
    safe_headers["Content-Type"] = public_media_type
    safe_headers["Cache-Control"] = "private, no-store"
    safe_headers["X-Content-Type-Options"] = "nosniff"
    headers = build_response_headers(
        request,
        {
            **safe_headers,
            "X-Chutes-InvocationID": invocation_id,
            "X-Chutes-OperationID": operation.operation_id,
        },
    )
    await _update_operation(
        operation.operation_id,
        status=ExternalOperationStatus.SUCCEEDED.value,
        result_descriptor=descriptor,
        started_at=now,
        finished_at=now,
    )
    await settle_operation(operation.operation_id, usage, billable=True)
    track_request_completed(chute.chute_id)
    return Response(
        public_body,
        status_code=response.status_code,
        media_type=public_media_type,
        headers=headers,
    )


def _encode_sse(event: SSEEvent, public_data: str) -> bytes:
    lines: list[str] = []
    if event.event:
        lines.append(f"event: {event.event}")
    for line in public_data.splitlines() or [""]:
        lines.append(f"data: {line}")
    return ("\n".join(lines) + "\n\n").encode()


async def _terminate_stream_consumer(
    queue: asyncio.Queue[bytes | None], consumer_alive: asyncio.Event
) -> None:
    """Wake an attached consumer even when its bounded delivery queue is full."""

    if not consumer_alive.is_set():
        return
    while consumer_alive.is_set():
        try:
            queue.put_nowait(None)
            consumer_alive.clear()
            return
        except asyncio.QueueFull:
            # Once execution is being terminated, queued provider bytes are no
            # longer useful. Drop one so the sentinel cannot be starved behind a
            # slow or paused downstream client.
            with suppress(asyncio.QueueEmpty):
                queue.get_nowait()
            await asyncio.sleep(0)


async def _finalize_interrupted_stream(
    *,
    operation: ExternalOperation,
    usage: NormalizedUsage,
    billable: bool,
    cancelled: bool = False,
) -> None:
    """Persist the terminal state and settle the usage observed before interruption."""

    try:
        await _update_operation(
            operation.operation_id,
            status=(
                ExternalOperationStatus.CANCELLED.value
                if cancelled
                else ExternalOperationStatus.FAILED.value
            ),
            settlement_metadata=_settlement_metadata_for(operation, billable=billable),
            error=(
                None
                if cancelled
                else {
                    "message": "The response stream ended before completion.",
                    "code": "stream_interrupted",
                    "retryable": True,
                    "details": {},
                }
            ),
            finished_at=datetime.now(timezone.utc),
            next_poll_at=None,
        )
    except asyncio.CancelledError:
        raise
    except Exception:
        # Settlement is still attempted when persisting the presentation state
        # fails; otherwise accepted work could become free solely because the
        # status update had a transient database failure.
        logger.exception(
            "Failed to record interrupted external stream {}", operation.operation_id
        )
    try:
        await settle_operation(
            operation.operation_id,
            usage,
            billable=billable,
        )
    except asyncio.CancelledError:
        raise
    except Exception:
        logger.exception(
            "Failed to settle interrupted external stream {}", operation.operation_id
        )


async def _sse_stream_producer(
    *,
    upstream: StreamingResponse,
    queue: asyncio.Queue[bytes | None],
    consumer_alive: asyncio.Event,
    route: ExternalRouteConfig,
    chute: Chute,
    operation: ExternalOperation,
    request_body: Any,
    invocation_id: str,
    partial_billable: bool,
    session_timeout_seconds: float = 300.0,
    usage_checkpoint_interval_seconds: float = USAGE_CHECKPOINT_INTERVAL_SECONDS,
) -> None:
    usage = NormalizedUsage(requests=1)
    usage_checkpoints: UsageCheckpointLoop | None = None
    budget_monitor: UsageBudgetMonitor | None = None

    async def stop_usage_checkpoints() -> None:
        nonlocal usage_checkpoints
        if usage_checkpoints is None:
            return
        checkpoints = usage_checkpoints
        usage_checkpoints = None
        await checkpoints.stop()

    async def stop_budget_monitor() -> None:
        nonlocal budget_monitor
        if budget_monitor is None:
            return
        monitor = budget_monitor
        budget_monitor = None
        await monitor.stop()

    async def terminate_stream(_reason: str) -> None:
        await _terminate_stream_consumer(queue, consumer_alive)
        await upstream.aclose()

    try:
        response_config = _object(route.response_config, "response_config")
        usage_mode = response_config.get("usage_mode", StreamUsageMode.CUMULATIVE.value)
        allow_non_json = response_config.get("allow_non_json_sse_data", False)
        if not isinstance(allow_non_json, bool):
            raise ExternalConfigurationError(
                "response_config.allow_non_json_sse_data must be a boolean"
            )
        if allow_non_json:
            raise ExternalConfigurationError(
                "non-JSON SSE data cannot cross the provider-obscuring boundary"
            )
        event_map = _object(
            response_config.get("sse_event_map"), "response_config.sse_event_map"
        )
        usage = _extract_initial_stream_usage(route, request_body=request_body)
        started_at = datetime.now(timezone.utc)
        started_monotonic = time.monotonic()
        recovery_deadline = session_recovery_deadline(
            started_at, session_timeout_seconds
        )
        operation_metadata = dict(getattr(operation, "settlement_metadata", None) or {})
        pricing_snapshot = dict(operation_metadata.get("pricing") or {})
        session_budget = session_budget_from_metadata(operation_metadata)
        if session_budget is None:
            session_budget = compile_session_budget(
                {},
                route.operation_config,
                max_session_seconds=session_timeout_seconds,
            )
            operation_metadata["session_budget"] = session_budget.snapshot()
            operation.settlement_metadata = operation_metadata
        await _update_operation(
            operation.operation_id,
            status=ExternalOperationStatus.RUNNING.value,
            started_at=started_at,
            expires_at=recovery_deadline,
            next_poll_at=recovery_deadline,
            usage=usage.to_dict(),
            settlement_metadata=_observed_cost_metadata(
                operation, pricing_snapshot, usage
            ),
        )
        usage_checkpoints = UsageCheckpointLoop(
            operation_id=operation.operation_id,
            read_usage=lambda: usage.to_dict(),
            persist_usage=lambda value: _checkpoint_running_usage(
                operation.operation_id,
                pricing_snapshot,
                value,
                settlement_metadata=operation_metadata,
                elapsed_seconds=max(0.0, time.monotonic() - started_monotonic),
            ),
            initial_usage=usage.to_dict(),
            interval_seconds=min(
                usage_checkpoint_interval_seconds,
                session_budget.check_interval_seconds,
            ),
            always_persist=True,
        )
        usage_checkpoints.start()
        budget_monitor = UsageBudgetMonitor(
            operation_id=operation.operation_id,
            read_usage=lambda: usage.to_dict(),
            check_usage=lambda value: _running_budget_check(
                operation.operation_id, pricing_snapshot, value
            ),
            on_exceeded=terminate_stream,
            interval_seconds=min(
                usage_checkpoint_interval_seconds,
                session_budget.check_interval_seconds,
            ),
        )
        budget_monitor.start()
        async for event in upstream.iter_sse():
            if event.data == "[DONE]":
                public_data = event.data
            else:
                is_json, value = _json_response(event.data.encode())
                if not is_json:
                    if not allow_non_json:
                        raise ExternalInvocationError(
                            "non-JSON SSE data was not permitted by the route"
                        )
                    public_data = event.data
                else:
                    config = _usage_config(route)
                    if config:
                        observed = _extract_stream_observation_usage(
                            route,
                            request_body=request_body,
                            response_body=value,
                            payload=value,
                        )
                        usage = merge_stream_usage(usage, observed, usage_mode)
                    public_value = scrub_public_response(
                        value,
                        _public_mapping(
                            route,
                            chute_name=chute.name,
                            invocation_id=invocation_id,
                            operation_id=operation.operation_id,
                        ),
                    )
                    public_data = orjson.dumps(public_value).decode()
            if consumer_alive.is_set():
                public_event_name = (
                    event_map.get(event.event) if event.event is not None else None
                )
                encoded = _encode_sse(
                    SSEEvent(data=public_data, event=public_event_name), public_data
                )
                while consumer_alive.is_set():
                    try:
                        await asyncio.wait_for(queue.put(encoded), timeout=0.25)
                        break
                    except asyncio.TimeoutError:
                        continue
        if budget_monitor is not None and budget_monitor.exceeded:
            if budget_monitor.reason == "cancel_requested":
                await stop_usage_checkpoints()
                await stop_budget_monitor()
                await _finalize_interrupted_stream(
                    operation=operation,
                    usage=usage,
                    billable=partial_billable,
                    cancelled=True,
                )
                return
            raise ExternalInvocationError("external stream spend limit was reached")
        await stop_usage_checkpoints()
        await stop_budget_monitor()
        await _update_operation(
            operation.operation_id,
            status=ExternalOperationStatus.SUCCEEDED.value,
            finished_at=datetime.now(timezone.utc),
            next_poll_at=None,
        )
        await settle_operation(operation.operation_id, usage, billable=True)
        track_request_completed(chute.chute_id)
    except asyncio.CancelledError:
        logger.warning(
            "External response stream was interrupted during shutdown for operation {}",
            operation.operation_id,
        )
        # Do not leave an active downstream consumer waiting on a producer that
        # shutdown is deliberately terminating.
        await _terminate_stream_consumer(queue, consumer_alive)
        with suppress(Exception, asyncio.CancelledError):
            await upstream.aclose()
        with suppress(Exception, asyncio.CancelledError):
            await stop_usage_checkpoints()
        with suppress(Exception, asyncio.CancelledError):
            await stop_budget_monitor()
        await _finalize_interrupted_stream(
            operation=operation,
            usage=usage,
            billable=partial_billable,
        )
        raise
    except Exception:
        logger.exception(
            "External response stream failed for operation {}", operation.operation_id
        )
        with suppress(Exception, asyncio.CancelledError):
            await stop_usage_checkpoints()
        with suppress(Exception, asyncio.CancelledError):
            await stop_budget_monitor()
        await _finalize_interrupted_stream(
            operation=operation,
            usage=usage,
            billable=partial_billable,
        )
        if consumer_alive.is_set():
            error = orjson.dumps(
                {"error": {"message": "The response stream ended unexpectedly."}}
            ).decode()
            with suppress(asyncio.QueueFull):
                queue.put_nowait(
                    _encode_sse(SSEEvent(data=error, event="error"), error)
                )
    finally:
        with suppress(Exception, asyncio.CancelledError):
            await stop_usage_checkpoints()
        with suppress(Exception, asyncio.CancelledError):
            await stop_budget_monitor()
        with suppress(Exception, asyncio.CancelledError):
            await upstream.aclose()
        await _terminate_stream_consumer(queue, consumer_alive)


async def _handle_sse_stream(
    *,
    response: StreamingResponse,
    route: ExternalRouteConfig,
    chute: Chute,
    selected_cord: Mapping[str, Any],
    operation: ExternalOperation,
    request: Request,
    request_body: Any,
    invocation_id: str,
    partial_billable: bool,
    session_timeout_seconds: float,
) -> StarletteStreamingResponse:
    _validated_output_content_type(response.headers, selected_cord)
    queue: asyncio.Queue[bytes | None] = asyncio.Queue(maxsize=32)
    consumer_alive = asyncio.Event()
    consumer_alive.set()

    async def consume() -> AsyncIterator[bytes]:
        try:
            while True:
                item = await queue.get()
                if item is None:
                    return
                yield item
        finally:
            # The producer deliberately continues draining and settling after a
            # downstream disconnect; it simply stops retaining delivery bytes.
            consumer_alive.clear()

    headers = build_response_headers(
        request,
        {
            "Content-Type": "text/event-stream",
            "Cache-Control": "private, no-store, no-transform",
            "X-Content-Type-Options": "nosniff",
            "X-Accel-Buffering": "no",
            "X-Chutes-InvocationID": invocation_id,
            "X-Chutes-OperationID": operation.operation_id,
        },
    )
    result = StarletteStreamingResponse(
        consume(),
        status_code=response.status_code,
        media_type="text/event-stream",
        headers=headers,
    )
    setattr(result, "_external_consumer_alive", consumer_alive)
    _spawn(
        _sse_stream_producer(
            upstream=response,
            queue=queue,
            consumer_alive=consumer_alive,
            route=route,
            chute=chute,
            operation=operation,
            request_body=request_body,
            invocation_id=invocation_id,
            partial_billable=partial_billable,
            session_timeout_seconds=session_timeout_seconds,
        )
    )
    return result


async def _raw_stream_producer(
    *,
    upstream: StreamingResponse,
    queue: asyncio.Queue[bytes | None],
    consumer_alive: asyncio.Event,
    route: ExternalRouteConfig,
    chute: Chute,
    operation: ExternalOperation,
    request_body: Any,
    partial_billable: bool,
    session_timeout_seconds: float = 300.0,
) -> None:
    usage = NormalizedUsage(requests=1)
    usage_checkpoints: UsageCheckpointLoop | None = None
    budget_monitor: UsageBudgetMonitor | None = None

    async def terminate_stream(_reason: str) -> None:
        await _terminate_stream_consumer(queue, consumer_alive)
        await upstream.aclose()

    try:
        usage = _extract_initial_stream_usage(route, request_body=request_body)
        header_observation = _extract_stream_observation_usage(
            route,
            request_body=request_body,
            response_body={"headers": dict(upstream.private_headers)},
            payload=None,
        )
        usage = merge_stream_usage(usage, header_observation, StreamUsageMode.DELTA)
        started_at = datetime.now(timezone.utc)
        started_monotonic = time.monotonic()
        recovery_deadline = session_recovery_deadline(
            started_at, session_timeout_seconds
        )
        operation_metadata = dict(getattr(operation, "settlement_metadata", None) or {})
        pricing_snapshot = dict(operation_metadata.get("pricing") or {})
        session_budget = session_budget_from_metadata(operation_metadata)
        if session_budget is None:
            session_budget = compile_session_budget(
                {},
                route.operation_config,
                max_session_seconds=session_timeout_seconds,
            )
            operation_metadata["session_budget"] = session_budget.snapshot()
            operation.settlement_metadata = operation_metadata
        await _update_operation(
            operation.operation_id,
            status=ExternalOperationStatus.RUNNING.value,
            started_at=started_at,
            expires_at=recovery_deadline,
            next_poll_at=recovery_deadline,
            usage=usage.to_dict(),
            settlement_metadata=_observed_cost_metadata(
                operation, pricing_snapshot, usage
            ),
        )
        usage_checkpoints = UsageCheckpointLoop(
            operation_id=operation.operation_id,
            read_usage=lambda: usage.to_dict(),
            persist_usage=lambda value: _checkpoint_running_usage(
                operation.operation_id,
                pricing_snapshot,
                value,
                settlement_metadata=operation_metadata,
                elapsed_seconds=max(0.0, time.monotonic() - started_monotonic),
            ),
            initial_usage=usage.to_dict(),
            interval_seconds=session_budget.check_interval_seconds,
            always_persist=True,
        )
        usage_checkpoints.start()
        budget_monitor = UsageBudgetMonitor(
            operation_id=operation.operation_id,
            read_usage=lambda: usage.to_dict(),
            check_usage=lambda value: _running_budget_check(
                operation.operation_id, pricing_snapshot, value
            ),
            on_exceeded=terminate_stream,
            interval_seconds=session_budget.check_interval_seconds,
        )
        budget_monitor.start()
        async for chunk in upstream.iter_bytes():
            if consumer_alive.is_set():
                while consumer_alive.is_set():
                    try:
                        await asyncio.wait_for(queue.put(chunk), timeout=0.25)
                        break
                    except asyncio.TimeoutError:
                        continue
        if budget_monitor is not None and budget_monitor.exceeded:
            if budget_monitor.reason == "cancel_requested":
                await usage_checkpoints.stop()
                usage_checkpoints = None
                await budget_monitor.stop()
                budget_monitor = None
                await _finalize_interrupted_stream(
                    operation=operation,
                    usage=usage,
                    billable=partial_billable,
                    cancelled=True,
                )
                return
            raise ExternalInvocationError("external stream spend limit was reached")
        if usage_checkpoints is not None:
            await usage_checkpoints.stop()
            usage_checkpoints = None
        if budget_monitor is not None:
            await budget_monitor.stop()
            budget_monitor = None
        await _update_operation(
            operation.operation_id,
            status=ExternalOperationStatus.SUCCEEDED.value,
            finished_at=datetime.now(timezone.utc),
            next_poll_at=None,
        )
        await settle_operation(operation.operation_id, usage, billable=True)
        track_request_completed(chute.chute_id)
    except asyncio.CancelledError:
        logger.warning(
            "External response stream was interrupted during shutdown for operation {}",
            operation.operation_id,
        )
        await _terminate_stream_consumer(queue, consumer_alive)
        with suppress(Exception, asyncio.CancelledError):
            await upstream.aclose()
        await _finalize_interrupted_stream(
            operation=operation,
            usage=usage,
            billable=partial_billable,
        )
        raise
    except Exception:
        logger.exception(
            "External response stream failed for operation {}", operation.operation_id
        )
        await _finalize_interrupted_stream(
            operation=operation,
            usage=usage,
            billable=partial_billable,
        )
    finally:
        if usage_checkpoints is not None:
            with suppress(Exception, asyncio.CancelledError):
                await usage_checkpoints.stop()
        if budget_monitor is not None:
            with suppress(Exception, asyncio.CancelledError):
                await budget_monitor.stop()
        with suppress(Exception, asyncio.CancelledError):
            await upstream.aclose()
        await _terminate_stream_consumer(queue, consumer_alive)


async def _handle_raw_stream(
    *,
    response: StreamingResponse,
    route: ExternalRouteConfig,
    chute: Chute,
    selected_cord: Mapping[str, Any],
    operation: ExternalOperation,
    request: Request,
    request_body: Any,
    invocation_id: str,
    partial_billable: bool,
    session_timeout_seconds: float,
) -> StarletteStreamingResponse:
    configured_content_type = _validated_output_content_type(
        response.headers, selected_cord
    )
    if configured_content_type is None:
        raise ExternalConfigurationError(
            "raw response streams require a Cord output_content_type"
        )
    public_content_type = configured_content_type
    queue: asyncio.Queue[bytes | None] = asyncio.Queue(maxsize=32)
    consumer_alive = asyncio.Event()
    consumer_alive.set()

    async def consume() -> AsyncIterator[bytes]:
        try:
            while True:
                item = await queue.get()
                if item is None:
                    return
                yield item
        finally:
            # Continue draining accepted work so billing does not depend on the
            # downstream connection remaining open.
            consumer_alive.clear()

    headers = build_response_headers(
        request,
        {
            "Content-Type": public_content_type,
            "Cache-Control": "private, no-store",
            "X-Content-Type-Options": "nosniff",
            "X-Accel-Buffering": "no",
            "X-Chutes-InvocationID": invocation_id,
            "X-Chutes-OperationID": operation.operation_id,
        },
    )
    result = StarletteStreamingResponse(
        consume(),
        status_code=response.status_code,
        media_type=public_content_type,
        headers=headers,
    )
    setattr(result, "_external_consumer_alive", consumer_alive)
    _spawn(
        _raw_stream_producer(
            upstream=response,
            queue=queue,
            consumer_alive=consumer_alive,
            route=route,
            chute=chute,
            operation=operation,
            request_body=request_body,
            partial_billable=partial_billable,
            session_timeout_seconds=session_timeout_seconds,
        )
    )
    return result


async def _handle_stream(
    *,
    response: StreamingResponse,
    route: ExternalRouteConfig,
    chute: Chute,
    selected_cord: Mapping[str, Any],
    operation: ExternalOperation,
    request: Request,
    request_body: Any,
    invocation_id: str,
    partial_billable: bool,
    session_timeout_seconds: float,
) -> StarletteStreamingResponse:
    if response.response_mode is ResponseMode.SSE:
        return await _handle_sse_stream(
            response=response,
            route=route,
            chute=chute,
            selected_cord=selected_cord,
            operation=operation,
            request=request,
            request_body=request_body,
            invocation_id=invocation_id,
            partial_billable=partial_billable,
            session_timeout_seconds=session_timeout_seconds,
        )
    if response.response_mode is ResponseMode.STREAM:
        return await _handle_raw_stream(
            response=response,
            route=route,
            chute=chute,
            selected_cord=selected_cord,
            operation=operation,
            request=request,
            request_body=request_body,
            invocation_id=invocation_id,
            partial_billable=partial_billable,
            session_timeout_seconds=session_timeout_seconds,
        )
    raise ExternalConfigurationError("streaming response mode is invalid")


async def _finish_accepted_task_handoff(
    completion: Awaitable[Response],
) -> Response:
    """Delay cancellation until an accepted task identity is durably attached."""

    task = asyncio.create_task(completion)
    try:
        return await asyncio.shield(task)
    except asyncio.CancelledError:
        # This short response-parse/DB handoff is what lets a different worker
        # find, cancel, or complete the already-funded remote task.
        with suppress(Exception, asyncio.CancelledError):
            await asyncio.shield(task)
        raise


async def invoke_external(
    *,
    request: Request,
    current_user: User,
    chute: Chute,
    selected_cord: Mapping[str, Any],
) -> Response:
    """Validate, execute, sanitize, account for, and return one external invocation."""

    invocation_id = str(uuid.uuid4())
    operation: ExternalOperation | None = None
    route: ExternalRouteConfig | None = None
    accepted_request_body: Any = {}
    accepted_response_body: Any = {}
    upstream_accepted = False
    bill_ambiguous_transport_errors = False
    billable_http_statuses: frozenset[int] = frozenset()
    partial_stream_billable = True
    task_timeout_seconds: float | None = None
    idempotency_key: str | None = None
    idempotency_fingerprint = ""
    upstream: BufferedResponse | StreamingResponse | None = None
    stream_handed_off = False
    try:
        binding, account = await _load_binding(chute.chute_id)
        route = select_route(binding.routes, selected_cord)
        if route.operation_mode is ExternalOperationMode.REALTIME:
            raise HTTPException(
                status_code=status.HTTP_426_UPGRADE_REQUIRED,
                detail="This Cord requires a realtime connection.",
            )
        idempotency_key = _idempotency_key(request, route)
        idempotency_fingerprint = _idempotency_fingerprint(request, selected_cord)
        if route.operation_mode is ExternalOperationMode.TASK:
            existing = await _existing_idempotent_operation(
                binding_id=binding.binding_id,
                user_id=current_user.user_id,
                idempotency_key=idempotency_key,
                body_sha256=getattr(request.state, "body_sha256", None),
                idempotency_fingerprint=idempotency_fingerprint,
            )
            if existing:
                return _idempotent_task_response(existing, request)
        bill_ambiguous_transport_errors = _bill_ambiguous_transport_failure(route)
        billable_http_statuses = _billable_http_statuses(route)
        if route.operation_mode is ExternalOperationMode.TASK:
            task_timeout_seconds = _task_timeout_seconds(route)
        profile = build_endpoint_profile(account, route)
        _validate_retry_billing_policy(route, profile, billable_http_statuses)
        _validate_body_transform_mode(route, profile.body_mode)
        if profile.response_mode in {ResponseMode.SSE, ResponseMode.STREAM}:
            partial_stream_billable = _bill_partial_stream(route)
        if (
            route.operation_mode is ExternalOperationMode.TASK
            and profile.response_mode is not ResponseMode.BUFFERED
        ):
            raise ExternalConfigurationError(
                "task submission responses must be buffered"
            )
        if profile.response_mode is ResponseMode.STREAM and _schema_for_output(
            route, selected_cord
        ):
            raise ExternalConfigurationError(
                "Cord output_schema cannot be enforced for a raw response stream"
            )
        if profile.response_mode is ResponseMode.STREAM:
            if not selected_cord.get("output_content_type"):
                raise ExternalConfigurationError(
                    "raw response streams require a Cord output_content_type"
                )
            _validate_raw_stream_usage(route, profile)
        request_body, outbound_body = await _request_body(
            request, route, profile.body_mode
        )
        client_query = _allowed_query(request, route)
        schema = _schema_for_request(route, selected_cord)
        if schema:
            await _validate_schema(
                _schema_request_value(
                    request_body,
                    outbound_body,
                    body_mode=profile.body_mode,
                    query=client_query,
                ),
                schema,
                "request",
            )
        transformed = _request_transform(
            route,
            request_body,
            invocation_id=invocation_id,
            chute_name=chute.name,
        )
        accepted_request_body = transformed
        if profile.body_mode is BodyMode.JSON:
            outbound_body = JsonBody(transformed)
        elif profile.body_mode is BodyMode.MULTIPART:
            if not isinstance(outbound_body, MultipartBody):
                raise ExternalConfigurationError(
                    "multipart request body is unavailable"
                )
            outbound_body = _rebuild_multipart_body(outbound_body, transformed)
        query = _upstream_query_parameters(route, transformed, client_query)
        pricing_snapshot = await _pricing_snapshot(
            current_user,
            chute,
            selected_cord,
            request,
            transformed if isinstance(transformed, Mapping) else {},
        )
        _validate_metering_config(route, pricing_snapshot)
        initial_usage = (
            _extract_initial_task_usage(route, request_body=transformed)
            if route.operation_mode is ExternalOperationMode.TASK
            else _extract_initial_stream_usage(route, request_body=transformed)
        )
        retry = retry_policy(route)
        retry_attempts = (
            retry.max_attempts
            if profile.method in {"GET", "HEAD", "PUT", "DELETE"}
            or retry.retry_non_idempotent
            else 1
        )
        dispatch_recovery_seconds = _dispatch_recovery_seconds(
            profile,
            retry_attempts,
            retry.maximum_delay_seconds,
        )
        operation, reused = await _create_operation(
            binding=binding,
            account=account,
            chute=chute,
            current_user=current_user,
            route=route,
            selected_cord=selected_cord,
            pricing_snapshot=pricing_snapshot,
            request=request,
            invocation_id=invocation_id,
            idempotency_key=idempotency_key,
            idempotency_fingerprint=idempotency_fingerprint,
            initial_usage=initial_usage,
            dispatch_recovery_seconds=dispatch_recovery_seconds,
            session_timeout_seconds=(
                profile.timeout.total
                if route.operation_mode is ExternalOperationMode.STREAM
                else None
            ),
        )
        if reused:
            return _idempotent_task_response(operation, request)
        executor = ExternalExecutor(secret_resolver=build_secret_resolver(account))
        outbound = OutboundRequest(
            path_parameters=_path_parameters(route, transformed, query),
            query=query,
            headers=dict(request.headers),
            body=outbound_body,
        )
        started = time.monotonic()
        try:
            upstream = await _execute_with_retry(executor, profile, outbound, route)
        except ExternalTransportError:
            upstream_latency.labels(phase="invoke").observe(time.monotonic() - started)
            upstream_requests.labels(phase="invoke", outcome="transport_error").inc()
            await record_upstream_result(account.account_id, transport_error=True)
            raise
        upstream_latency.labels(phase="invoke").observe(time.monotonic() - started)
        upstream_requests.labels(
            phase="invoke", outcome=status_class(upstream.status_code)
        ).inc()
        await record_upstream_result(
            account.account_id, status_code=upstream.status_code
        )
        upstream_accepted = (
            200 <= upstream.status_code < 300
            or upstream.status_code in billable_http_statuses
        )
        if isinstance(upstream, BufferedResponse):
            parsed_response = _json_or_none(upstream.body)
            accepted_response_body = (
                parsed_response if parsed_response is not None else {}
            )
        metadata = dict(operation.request_metadata or {})
        metadata["upstream_latency_ms"] = round((time.monotonic() - started) * 1000)
        await _update_operation(operation.operation_id, request_metadata=metadata)
        if isinstance(upstream, StreamingResponse):
            result = await _handle_stream(
                response=upstream,
                route=route,
                chute=chute,
                selected_cord=selected_cord,
                operation=operation,
                request=request,
                request_body=transformed,
                invocation_id=invocation_id,
                partial_billable=partial_stream_billable,
                session_timeout_seconds=profile.timeout.total,
            )
            stream_handed_off = True
            return result
        buffered_call = _handle_buffered(
            response=upstream,
            route=route,
            chute=chute,
            selected_cord=selected_cord,
            operation=operation,
            request=request,
            request_body=transformed,
            invocation_id=invocation_id,
            task_timeout_seconds=task_timeout_seconds,
        )
        if route.operation_mode is ExternalOperationMode.TASK and upstream_accepted:
            # Once a task submission has been accepted, response parsing and the
            # durable upstream-id attachment must outlive request/shutdown
            # cancellation.  The poller can recover it after this short handoff.
            return await _finish_accepted_task_handoff(buffered_call)
        return await buffered_call
    except asyncio.CancelledError:
        logger.warning(
            "External invocation was interrupted for chute {}", chute.chute_id
        )
        if isinstance(upstream, StreamingResponse) and not stream_handed_off:
            with suppress(Exception, asyncio.CancelledError):
                await upstream.aclose()
        if operation is not None and route is not None:
            await _finalize_interrupted_invocation(
                request=request,
                operation=operation,
                route=route,
                request_body=accepted_request_body,
                response_body=accepted_response_body,
                upstream_accepted=upstream_accepted,
                accepted_billable=(
                    partial_stream_billable
                    if isinstance(upstream, StreamingResponse)
                    else True
                ),
                ambiguous_billable=bill_ambiguous_transport_errors,
            )
        raise
    except HTTPException:
        raise
    except (
        ExternalConfigurationError,
        ExternalRequestMappingError,
        MappingConfigurationError,
        PricingConfigurationError,
        ProfileError,
        jsonschema.SchemaError,
    ) as exc:
        logger.exception(
            "External Chute configuration is invalid for {}", chute.chute_id
        )
        if operation and route and upstream_accepted:
            await _record_accepted_failure(
                operation=operation,
                route=route,
                request_body=accepted_request_body,
                response_body=accepted_response_body,
                code="response_configuration_error",
            )
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail="The external service returned an invalid response.",
            ) from exc
        if operation:
            await _update_operation(
                operation.operation_id,
                status=ExternalOperationStatus.FAILED.value,
                settlement_status=ExternalSettlementStatus.NOT_BILLABLE.value,
                error={
                    "message": "External execution is currently unavailable.",
                    "code": "configuration_error",
                    "retryable": False,
                    "details": {},
                },
                finished_at=datetime.now(timezone.utc),
            )
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="External execution is currently unavailable.",
        )
    except (RequestRejectedError, MappingExtractionError, ValueError) as exc:
        if operation and route and upstream_accepted:
            await _record_accepted_failure(
                operation=operation,
                route=route,
                request_body=accepted_request_body,
                response_body=accepted_response_body,
            )
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail="The external service returned an invalid response.",
            ) from exc
        if operation:
            await _update_operation(
                operation.operation_id,
                status=ExternalOperationStatus.FAILED.value,
                settlement_status=ExternalSettlementStatus.NOT_BILLABLE.value,
                error={
                    "message": "The request could not be processed.",
                    "code": "invalid_request",
                    "retryable": False,
                    "details": {},
                },
                finished_at=datetime.now(timezone.utc),
            )
        raise HTTPException(
            status_code=400, detail="The request could not be processed."
        ) from exc
    except ExternalTransportError as exc:
        logger.warning(
            "External transport failure for chute {}: {}",
            chute.chute_id,
            type(exc).__name__,
        )
        if operation:
            await _update_operation(
                operation.operation_id,
                status=ExternalOperationStatus.FAILED.value,
                **(
                    {}
                    if bill_ambiguous_transport_errors
                    else {
                        "settlement_status": ExternalSettlementStatus.NOT_BILLABLE.value
                    }
                ),
                error={
                    "message": "External capacity is temporarily unavailable.",
                    "code": "transport_error",
                    "retryable": True,
                    "details": {},
                },
                finished_at=datetime.now(timezone.utc),
            )
            if bill_ambiguous_transport_errors and route:
                request.state.external_attempt_billable = True
                try:
                    usage = _extract_usage(
                        route,
                        request_body=accepted_request_body,
                        response_body={},
                        task=route.operation_mode is ExternalOperationMode.TASK,
                    )
                except Exception:
                    usage = NormalizedUsage(requests=1)
                await settle_operation(operation.operation_id, usage, billable=True)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="External capacity is temporarily unavailable.",
        ) from exc
    except ExternalInvocationError as exc:
        logger.warning("Invalid external response for chute {}", chute.chute_id)
        if operation and route and upstream_accepted:
            await _record_accepted_failure(
                operation=operation,
                route=route,
                request_body=accepted_request_body,
                response_body=accepted_response_body,
            )
        elif operation:
            await _update_operation(
                operation.operation_id,
                status=ExternalOperationStatus.FAILED.value,
                settlement_status=ExternalSettlementStatus.NOT_BILLABLE.value,
                error={
                    "message": "The external service returned an invalid response.",
                    "code": "invalid_response",
                    "retryable": True,
                    "details": {},
                },
                finished_at=datetime.now(timezone.utc),
            )
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="The external service returned an invalid response.",
        ) from exc
    except Exception as exc:
        logger.exception("Unhandled external invocation failure for {}", chute.chute_id)
        if operation and route and upstream_accepted:
            await _record_accepted_failure(
                operation=operation,
                route=route,
                request_body=accepted_request_body,
                response_body=accepted_response_body,
                code="execution_error",
            )
        elif operation:
            await _update_operation(
                operation.operation_id,
                status=ExternalOperationStatus.FAILED.value,
                settlement_status=ExternalSettlementStatus.NOT_BILLABLE.value,
                error={
                    "message": "External execution failed.",
                    "code": "execution_error",
                    "retryable": True,
                    "details": {},
                },
                finished_at=datetime.now(timezone.utc),
            )
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="External execution failed.",
        ) from exc
    finally:
        if isinstance(upstream, StreamingResponse) and not stream_handed_off:
            with suppress(Exception, asyncio.CancelledError):
                await upstream.aclose()


async def invoke_external_resilient(**kwargs: Any) -> Response:
    """Keep accepted upstream work and accounting alive after a client disconnect."""

    task = _spawn(invoke_external(**kwargs))
    try:
        return await asyncio.shield(task)
    except asyncio.CancelledError:
        _spawn(_disconnect_orphaned_stream(task))
        raise


async def _disconnect_orphaned_stream(task: asyncio.Task[Response]) -> None:
    """Stop retaining bytes when cancellation races with response construction."""

    try:
        # This observer does not own the accounting task.  In particular,
        # cancelling it during shutdown must not deliver a second cancellation
        # into an invocation that is already finalizing its operation.
        response = await asyncio.shield(task)
    except (Exception, asyncio.CancelledError):
        return
    consumer_alive = getattr(response, "_external_consumer_alive", None)
    if isinstance(consumer_alive, asyncio.Event):
        consumer_alive.clear()


async def shutdown_external_invocations(
    timeout_seconds: float = 30.0,
    cancellation_timeout_seconds: float = 5.0,
) -> int:
    """Drain work, then cancel and briefly await any remaining accounting cleanup."""

    current = asyncio.current_task()

    def live_tasks() -> tuple[asyncio.Task[Any], ...]:
        return tuple(
            task
            for task in _BACKGROUND_TASKS
            if task is not current and not task.done()
        )

    loop = asyncio.get_running_loop()
    grace_deadline = loop.time() + max(0.0, timeout_seconds)
    while tasks := live_tasks():
        remaining = grace_deadline - loop.time()
        if remaining <= 0:
            break
        await asyncio.wait(tasks, timeout=remaining)

    tasks = live_tasks()
    if not tasks:
        return 0

    logger.warning(
        "Cancelling {} external invocation accounting tasks after shutdown grace",
        len(tasks),
    )
    cancellation_requested: set[asyncio.Task[Any]] = set()
    cleanup_deadline = loop.time() + max(0.0, cancellation_timeout_seconds)
    while tasks:
        for task in tasks:
            if task not in cancellation_requested:
                cancellation_requested.add(task)
                task.cancel()
        remaining = cleanup_deadline - loop.time()
        if remaining <= 0:
            # Give cancellation handlers one scheduling opportunity even when a
            # caller explicitly requests no cleanup grace.
            await asyncio.sleep(0)
            break
        await asyncio.wait(tasks, timeout=remaining)
        tasks = live_tasks()

    pending = live_tasks()
    if pending:
        logger.warning(
            "External invocation shutdown still has {} accounting tasks", len(pending)
        )
    return len(pending)


__all__ = [
    "build_secret_resolver",
    "invoke_external",
    "invoke_external_resilient",
    "settle_operation",
    "shutdown_external_invocations",
]
