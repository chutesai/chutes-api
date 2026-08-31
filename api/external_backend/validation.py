"""Fail-fast compilation for externally executed route configuration."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import math
import re
from urllib.parse import urlsplit, urlunsplit

import jsonschema

from api.config import settings
from api.external_transport import BodyMode, ResponseMode
from api.external_transport.errors import ProfileError
from api.external_transport.header_policy import requires_secret_backing
from api.external_transport.security import validate_profile_headers
from api.payment.pricing import (
    UsageMetric,
    validate_conditional_pricing_coverage,
)

from .config import (
    ExternalConfigurationError,
    build_artifact_profile,
    build_endpoint_profile,
    retry_policy,
)
from .governance import compile_session_budget
from .artifact_policy import artifact_relay_ttl_seconds
from .mapping import (
    ArtifactMapping,
    DataPath,
    MappingConfigurationError,
    PayloadTransform,
    PublicResponseRules,
    StreamUsageMode,
    TaskMapping,
    UsageMapping,
)
from .schemas import (
    ExternalBackendAccount,
    ExternalOperationMode,
    ExternalRouteConfig,
)
from .schema_validation import (
    RemoteSchemaReferenceError,
    UnsafeSchemaError,
    local_json_schema_validator,
)
from .task_results import InlineResultPolicyError, inline_result_limit


class RouteConfigurationError(ValueError):
    """A route cannot be safely executed as configured."""


_REQUEST_KEYS = frozenset(
    {
        "allowed_query_parameters",
        "allowed_request_headers",
        "body_mode",
        "max_request_bytes",
        "path_parameters",
        "query_parameters",
        "resource_path",
        "resource_query_parameter",
        "static_headers",
        "transform",
        "validate_input_schema",
    }
)
_RESPONSE_KEYS = frozenset(
    {
        "allowed_headers",
        "allow_non_json_sse_data",
        "artifact_relay_ttl_seconds",
        "artifacts",
        "bill_partial_streams",
        "max_response_bytes",
        "max_sse_event_bytes",
        "mode",
        "public",
        "redirects",
        "sse_event_map",
        "stream_chunk_bytes",
        "task",
        "usage",
        "usage_mode",
        "validate_output_schema",
    }
)
_OPERATION_KEYS = frozenset(
    {
        "artifact",
        "bill_ambiguous_transport_errors",
        "bill_statuses",
        "billable_http_statuses",
        "billable_terminal_statuses",
        "cancel",
        "poll",
        "poll_mapping",
        "persist_inline_result",
        "max_inline_result_bytes",
        "realtime",
        "retry",
        "session_budget",
        "submission_contract",
        "submit_mapping",
        "task",
        "task_mapping",
        "task_timeout_seconds",
        "usage",
        "websocket",
    }
)
_RETRY_KEYS = frozenset(
    {
        "base_delay_seconds",
        "body_status_path",
        "body_statuses",
        "max_attempts",
        "maximum_delay_seconds",
        "retry_after_headers",
        "retry_non_idempotent",
        "statuses",
    }
)
_REALTIME_KEYS = frozenset(
    {
        "allow_client_binary",
        "allow_client_non_json_text",
        "allow_upstream_binary",
        "allow_upstream_non_json_text",
        "allowed_hosts",
        "allowed_query_parameters",
        "allowed_request_headers",
        "allowed_subprotocols",
        "base_url",
        "handshake_timeout_seconds",
        "heartbeat_seconds",
        "idle_timeout_seconds",
        "max_message_bytes",
        "max_session_seconds",
        "message_schema",
        "path_parameters",
        "path_template",
        "pricing_dimensions",
        "require_subprotocol",
        "static_headers",
        "usage",
        "usage_mode",
        "validate_client_messages",
    }
)
_REDIRECT_KEYS = frozenset({"allow_cross_origin", "max_redirects"})
_ARTIFACT_PROFILE_KEYS = frozenset(
    {
        "allowed_hosts",
        "allowed_ports",
        "authenticated",
        "auth_allowed_origins",
        "max_bytes",
        "redirects",
        "static_headers",
        "stream_chunk_bytes",
        "timeouts",
    }
)
_SUBMISSION_CONTRACT_KEYS = frozenset(
    {"enabled", "output_content_type", "output_schema"}
)


def _object(value: object, label: str) -> Mapping[str, object]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise RouteConfigurationError(f"{label} must be an object")
    return value


def _strict_keys(
    value: Mapping[str, object], allowed: frozenset[str], label: str
) -> None:
    unknown = set(value) - allowed
    if unknown:
        names = ", ".join(sorted(map(str, unknown)))
        raise RouteConfigurationError(f"{label} contains unsupported fields: {names}")


def _compile_redirects(value: object, label: str) -> None:
    redirects = _object(value, label)
    _strict_keys(redirects, _REDIRECT_KEYS, label)
    if "allow_cross_origin" in redirects and not isinstance(
        redirects["allow_cross_origin"], bool
    ):
        raise RouteConfigurationError(f"{label}.allow_cross_origin must be a boolean")
    maximum = redirects.get("max_redirects")
    if maximum is not None and (
        isinstance(maximum, bool)
        or not isinstance(maximum, int)
        or not 0 <= maximum <= 10
    ):
        raise RouteConfigurationError(f"{label}.max_redirects must be between 0 and 10")


def _compile_json_schema(value: object, label: str) -> None:
    if value in (None, {}):
        return
    if not isinstance(value, Mapping):
        raise RouteConfigurationError(f"{label} must be an object")
    try:
        local_json_schema_validator(value)
    except (
        jsonschema.exceptions.SchemaError,
        RemoteSchemaReferenceError,
        UnsafeSchemaError,
    ) as exc:
        message = getattr(exc, "message", str(exc))
        raise RouteConfigurationError(f"{label} is invalid: {message}") from exc


def _boolean_option(
    config: Mapping[str, object], key: str, label: str, *, default: bool = False
) -> bool:
    value = config.get(key, default)
    if not isinstance(value, bool):
        raise RouteConfigurationError(f"{label} must be a boolean")
    return value


def _cord_value(cord: object | None, field: str) -> object:
    if cord is None:
        return None
    if isinstance(cord, Mapping):
        return cord.get(field)
    return getattr(cord, field, None)


def _selected_cord_input_schema(cord: object | None) -> tuple[object, str]:
    input_schema = _cord_value(cord, "input_schema")
    if input_schema:
        return input_schema, "cord.input_schema"
    return _cord_value(cord, "minimal_input_schema"), "cord.minimal_input_schema"


def _compile_required_schema(value: object, label: str, option: str) -> None:
    if not isinstance(value, Mapping) or not value:
        raise RouteConfigurationError(f"{option}=true requires a non-empty {label}")
    _compile_json_schema(value, label)


def _compile_path_parameters(value: object, label: str) -> None:
    configured = _object(value, label)
    for name, rule in configured.items():
        if not str(name):
            raise RouteConfigurationError(f"{label} names cannot be empty")
        if isinstance(rule, str):
            DataPath.parse(rule)
        elif isinstance(rule, Mapping):
            _strict_keys(rule, frozenset({"path", "required", "value"}), label)
            if "value" not in rule:
                path = rule.get("path")
                if not isinstance(path, str):
                    raise RouteConfigurationError(
                        f"{label} rule requires a path or value"
                    )
                DataPath.parse(path)


def _validate_query_parameter_name(value: object, label: str) -> str:
    if not isinstance(value, str) or not value or len(value) > 128:
        raise RouteConfigurationError(f"{label} is invalid")
    if re.search(r"[\x00-\x20&=#?]", value):
        raise RouteConfigurationError(f"{label} is invalid")
    return value


def _query_scalar(value: object, label: str) -> None:
    # ``None`` means omit a mapped value (and can deliberately remove an
    # allowlisted client query value); it is never serialized upstream.
    if value is None:
        return
    if not isinstance(value, (str, int, float, bool)):
        raise RouteConfigurationError(f"{label} must be a scalar")
    if isinstance(value, float) and not math.isfinite(value):
        raise RouteConfigurationError(f"{label} must be a finite scalar")


def _query_rule_literal(rule: object, label: str) -> object | None:
    if isinstance(rule, str):
        path = DataPath.parse(rule)
        if not path.parts or path.parts[0] not in {"body", "query", "context"}:
            raise RouteConfigurationError(
                f"{label} path must select body, query, or context"
            )
        return None
    if isinstance(rule, Mapping):
        _strict_keys(rule, frozenset({"path", "required", "value"}), label)
        if "value" in rule:
            if "path" in rule or "required" in rule:
                raise RouteConfigurationError(
                    f"{label} literal cannot define path or required"
                )
            _query_scalar(rule["value"], label)
            return rule["value"]
        path_value = rule.get("path")
        if not isinstance(path_value, str):
            raise RouteConfigurationError(f"{label} requires a path or value")
        required = rule.get("required", True)
        if not isinstance(required, bool):
            raise RouteConfigurationError(f"{label}.required must be a boolean")
        path = DataPath.parse(path_value)
        if not path.parts or path.parts[0] not in {"body", "query", "context"}:
            raise RouteConfigurationError(
                f"{label} path must select body, query, or context"
            )
        return None
    _query_scalar(rule, label)
    return rule


def _compile_query_parameters(
    request: Mapping[str, object], route: ExternalRouteConfig
) -> None:
    configured = _object(
        request.get("query_parameters"), "request_config.query_parameters"
    )
    if len(configured) > 100:
        raise RouteConfigurationError(
            "request_config.query_parameters contains too many entries"
        )
    resource_names = {"resource", "model", "upstream_resource_id"}
    for raw_name, rule in configured.items():
        name = _validate_query_parameter_name(
            raw_name, "request_config.query_parameters name"
        )
        if requires_secret_backing(name):
            raise RouteConfigurationError(
                "credential-like query parameters are not supported"
            )
        literal = _query_rule_literal(rule, f"request_config.query_parameters.{name}")
        if (
            name in resource_names
            and literal is not None
            and literal != route.upstream_resource_id
        ):
            raise RouteConfigurationError(
                f"query parameter {name!r} must use the configured resource"
            )
    resource_name = request.get("resource_query_parameter")
    if resource_name is not None:
        resource_name = _validate_query_parameter_name(
            resource_name, "request_config.resource_query_parameter"
        )
        if requires_secret_backing(resource_name):
            raise RouteConfigurationError(
                "credential-like query parameters are not supported"
            )


def _compile_artifact_mappings(value: object) -> None:
    if value is None:
        return
    mappings: Sequence[object]
    if isinstance(value, Mapping):
        mappings = [value]
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        mappings = value
    else:
        raise RouteConfigurationError(
            "response_config.artifacts must be an object or list"
        )
    for mapping in mappings:
        ArtifactMapping.from_config(mapping)


def _selected_usage_config(route: ExternalRouteConfig) -> object:
    response = _object(route.response_config, "response_config")
    operation = _object(route.operation_config, "operation_config")
    if route.operation_mode is ExternalOperationMode.TASK:
        task = _object(operation.get("task"), "operation_config.task")
        poll = _object(operation.get("poll", task.get("poll")), "operation_config.poll")
        return (
            poll.get("usage")
            or task.get("usage")
            or operation.get("usage")
            or response.get("usage")
        )
    if route.operation_mode is ExternalOperationMode.REALTIME:
        realtime = operation.get("realtime")
        websocket = operation.get("websocket")
        if realtime is not None and websocket is not None:
            raise RouteConfigurationError(
                "operation_config cannot define both realtime and websocket"
            )
        endpoint = _object(
            realtime if realtime is not None else websocket,
            "operation_config.realtime",
        )
        return endpoint.get("usage") or operation.get("usage") or response.get("usage")
    return response.get("usage") or operation.get("usage")


def _schema_paths(value: object, prefix: str = "") -> set[str]:
    """Collect dimension paths that a declared request schema can supply."""

    if not isinstance(value, Mapping):
        return set()
    paths: set[str] = set()
    properties = value.get("properties")
    if isinstance(properties, Mapping):
        for raw_name, child in properties.items():
            name = str(raw_name)
            path = f"{prefix}.{name}" if prefix else name
            paths.add(path)
            paths.update(_schema_paths(child, path))
    for keyword in ("allOf", "anyOf", "oneOf"):
        variants = value.get(keyword)
        if isinstance(variants, Sequence) and not isinstance(variants, (str, bytes)):
            for variant in variants:
                paths.update(_schema_paths(variant, prefix))
    return paths


def _flatten_dimension_paths(value: object, prefix: str = "") -> set[str]:
    if not isinstance(value, Mapping):
        return {prefix} if prefix else set()
    paths: set[str] = set()
    for raw_name, child in value.items():
        name = str(raw_name)
        path = f"{prefix}.{name}" if prefix else name
        paths.add(path)
        paths.update(_flatten_dimension_paths(child, path))
    return paths


def _acceptance_dimension_paths(
    route: ExternalRouteConfig,
    cord: object | None,
    *,
    include_cord_schema: bool = False,
) -> set[str]:
    """Return dimensions fixed or observable when the upstream call is accepted."""

    available: set[str] = set()
    if include_cord_schema and cord is not None:
        for field in ("input_schema", "minimal_input_schema"):
            schema = (
                cord.get(field)
                if isinstance(cord, Mapping)
                else getattr(cord, field, None)
            )
            available.update(_schema_paths(schema))

    request = _object(route.request_config, "request_config")
    transform = _object(request.get("transform"), "request_config.transform")
    removed = transform.get("remove", [])
    if isinstance(removed, Sequence) and not isinstance(removed, (str, bytes)):
        for raw_path in removed:
            path = DataPath.parse(raw_path).raw
            available = {
                candidate
                for candidate in available
                if candidate != path and not candidate.startswith(f"{path}.")
            }
    for section in ("inject", "rewrite"):
        mutations = transform.get(section, [])
        if isinstance(mutations, Mapping):
            mutations = [mutations]
        if isinstance(mutations, Sequence) and not isinstance(mutations, (str, bytes)):
            for mutation in mutations:
                if isinstance(mutation, Mapping) and isinstance(
                    mutation.get("target"), str
                ):
                    available.add(DataPath.parse(mutation["target"]).raw)
    resource_path = request.get("resource_path")
    if isinstance(resource_path, str) and resource_path.strip():
        available.add(DataPath.parse(resource_path).raw)

    if route.operation_mode is ExternalOperationMode.REALTIME:
        operation = _object(route.operation_config, "operation_config")
        endpoint = _object(
            operation.get("realtime", operation.get("websocket")),
            "operation_config.realtime",
        )
        available.update(_flatten_dimension_paths(endpoint.get("pricing_dimensions")))
    return available


def _validate_pricing_metering(
    pricing_rules: Sequence[Mapping[str, object]],
    usage: UsageMapping | None,
    acceptance_dimensions: set[str],
    cord: object | None,
) -> None:
    targets = {field.target for field in usage.fields} if usage else set()
    groups = {
        UsageMetric.TOKEN: "tokens",
        UsageMetric.IMAGE: "images",
        UsageMetric.INPUT_MEDIA_SECOND: "input_media_seconds",
        UsageMetric.OUTPUT_MEDIA_SECOND: "output_media_seconds",
        UsageMetric.CHARACTER: "characters",
        UsageMetric.COUNT: "counts",
        UsageMetric.TOOL: "tools",
    }

    def cord_value(name: str, default: object = None) -> object:
        if cord is None:
            return default
        if isinstance(cord, Mapping):
            return cord.get(name, default)
        return getattr(cord, name, default)

    cord_name = cord_value("function") or cord_value("path")
    public_path = cord_value("public_api_path") or cord_value("path")
    public_method = cord_value("public_api_method") or cord_value("method")
    if isinstance(public_method, str):
        public_method = public_method.upper()
    parsed_rules = validate_conditional_pricing_coverage(pricing_rules)
    rules = [
        rule
        for rule in parsed_rules
        if (rule.scope.cord is None or rule.scope.cord == cord_name)
        and (rule.scope.path is None or rule.scope.path == public_path)
        and (rule.scope.method is None or rule.scope.method == public_method)
    ]
    if pricing_rules and not rules:
        raise RouteConfigurationError(f"pricing rules do not cover Cord {cord_name!r}")
    for rule in rules:
        if rule.metric is UsageMetric.REQUEST:
            pass
        else:
            group = groups[rule.metric]
            expected = f"{group}.{rule.bucket}" if rule.bucket is not None else None
            covered = (
                expected in targets
                if expected is not None
                else any(target.startswith(f"{group}.") for target in targets)
            )
            if not covered:
                bucket = f" bucket {rule.bucket!r}" if rule.bucket is not None else ""
                raise RouteConfigurationError(
                    f"pricing metric {rule.metric.value!r}{bucket} requires a matching usage field"
                )
        for name, condition in rule.conditions.items():
            normalized = (
                {
                    str(operator).removeprefix("$"): operand
                    for operator, operand in condition.items()
                }
                if isinstance(condition, Mapping)
                else {}
            )
            if (
                normalized.get("exists") is not False
                and name not in acceptance_dimensions
                and f"dimensions.{name}" not in targets
            ):
                raise RouteConfigurationError(
                    f"pricing condition {name!r} requires a matching usage dimension"
                )


def _compile_realtime(
    account: ExternalBackendAccount,
    route: ExternalRouteConfig,
    cord: object | None,
) -> object:
    from .realtime import build_websocket_profile

    operation = _object(route.operation_config, "operation_config")
    realtime = operation.get("realtime")
    websocket = operation.get("websocket")
    if realtime is not None and websocket is not None:
        raise RouteConfigurationError(
            "operation_config cannot define both realtime and websocket"
        )
    endpoint = _object(
        realtime if realtime is not None else websocket,
        "operation_config.realtime",
    )
    _strict_keys(endpoint, _REALTIME_KEYS, "operation_config.realtime")
    profile = build_websocket_profile(account, route)
    validate_profile_headers(profile.static_headers)
    _compile_path_parameters(
        endpoint.get("path_parameters"), "operation_config.realtime.path_parameters"
    )
    for field in (
        "allow_client_binary",
        "allow_client_non_json_text",
        "allow_upstream_binary",
        "allow_upstream_non_json_text",
        "require_subprotocol",
        "validate_client_messages",
    ):
        if field in endpoint and not isinstance(endpoint[field], bool):
            raise RouteConfigurationError(f"realtime.{field} must be a boolean")
    if endpoint.get("validate_client_messages", False):
        schema = endpoint.get("message_schema")
        label = "operation_config.realtime.message_schema"
        if not schema:
            schema, label = _selected_cord_input_schema(cord)
        _compile_required_schema(
            schema,
            label,
            "operation_config.realtime.validate_client_messages",
        )
    if endpoint.get("pricing_dimensions") is not None:
        _object(endpoint.get("pricing_dimensions"), "realtime.pricing_dimensions")
    return profile


def _compile_task(account: ExternalBackendAccount, route: ExternalRouteConfig) -> None:
    from .polling import TaskLifecyclePolicy

    operation = _object(route.operation_config, "operation_config")
    response = _object(route.response_config, "response_config")
    task_config = (
        operation.get("submit_mapping")
        or operation.get("task_mapping")
        or response.get("task")
    )
    if not isinstance(task_config, Mapping):
        raise RouteConfigurationError("task response mapping is not configured")
    submit = TaskMapping.from_config(task_config)
    if submit.task_id is None:
        raise RouteConfigurationError("task response mapping must map task_id")

    operation_timeout = operation.get("task_timeout_seconds")
    task = _object(operation.get("task"), "operation_config.task")
    fallback_billability_explicit = (
        "billable_statuses" in task or "billable_terminal_statuses" in operation
    )
    raw_poll = operation.get("poll", task.get("poll"))
    raw_cancel = operation.get("cancel", task.get("cancel"))
    if not fallback_billability_explicit:
        poll_billability_explicit = isinstance(raw_poll, Mapping) and (
            "billable_statuses" in raw_poll
        )
        cancel_billability_explicit = raw_cancel is None or (
            isinstance(raw_cancel, Mapping) and "billable_statuses" in raw_cancel
        )
        if not poll_billability_explicit or not cancel_billability_explicit:
            raise RouteConfigurationError(
                "task routes must explicitly configure billable_statuses for "
                "polling and cancellation terminal outcomes"
            )
    configured_timeout = task.get("timeout_seconds", operation_timeout)
    if configured_timeout is not None:
        if (
            isinstance(configured_timeout, bool)
            or not isinstance(configured_timeout, (int, float))
            or not math.isfinite(configured_timeout)
            or not 60 <= configured_timeout <= 30 * 86400
        ):
            raise RouteConfigurationError(
                "task timeout must be between 60 and 2592000 seconds"
            )

    lifecycle = TaskLifecyclePolicy.from_route(route)
    for name, call in (("poll", lifecycle.poll), ("cancel", lifecycle.cancel)):
        if call is None:
            continue
        profile = build_endpoint_profile(
            account,
            route,
            endpoint=call.endpoint,
            name_suffix=name,
        )
        validate_profile_headers(profile.static_headers)
        if profile.response_mode is not ResponseMode.BUFFERED:
            raise RouteConfigurationError(
                f"task {name} endpoint requires response mode: buffered"
            )
        if profile.body_mode not in {BodyMode.NONE, BodyMode.JSON}:
            raise RouteConfigurationError(
                f"task {name} endpoint supports only none and json body modes"
            )
        raw_call = operation.get(name, task.get(name))
        call_config = _object(raw_call, f"operation_config.{name}")
        call_request = _object(
            call_config.get("request"), f"operation_config.{name}.request"
        )
        if profile.body_mode is BodyMode.NONE and (
            "body" in call_request or "body_transform" in call_request
        ):
            raise RouteConfigurationError(
                f"task {name} endpoint cannot configure a body in none mode"
            )


def _compile_artifact_profile(
    account: ExternalBackendAccount, route: ExternalRouteConfig
) -> None:
    operation = _object(route.operation_config, "operation_config")
    if operation.get("artifact") is None:
        return
    artifact = _object(operation.get("artifact"), "operation_config.artifact")
    _strict_keys(artifact, _ARTIFACT_PROFILE_KEYS, "operation_config.artifact")
    _compile_redirects(artifact.get("redirects"), "artifact.redirects")
    if "authenticated" in artifact and not isinstance(artifact["authenticated"], bool):
        raise RouteConfigurationError("artifact.authenticated must be a boolean")
    base = urlsplit(str(route.base_url or account.base_url))
    reference = urlunsplit((base.scheme, base.netloc, "/__validation__", "", ""))
    profile = build_artifact_profile(account, route, reference)
    validate_profile_headers(profile.static_headers)


def _compile_submission_contract(
    operation: Mapping[str, object],
    cord: object | None,
    operation_mode: ExternalOperationMode,
) -> None:
    contract = _object(
        operation.get("submission_contract"), "operation_config.submission_contract"
    )
    _strict_keys(
        contract,
        _SUBMISSION_CONTRACT_KEYS,
        "operation_config.submission_contract",
    )
    enabled = contract.get("enabled", False)
    if not isinstance(enabled, bool):
        raise RouteConfigurationError("submission_contract.enabled must be a boolean")
    if not enabled:
        return
    if operation_mode is not ExternalOperationMode.TASK:
        raise RouteConfigurationError(
            "submission_contract.enabled is only supported for task routes"
        )
    if "output_schema" in contract:
        schema = contract.get("output_schema")
        schema_label = "submission_contract.output_schema"
    else:
        schema = _cord_value(cord, "output_schema")
        schema_label = "cord.output_schema"
    _compile_json_schema(schema, schema_label)
    content_type = (
        contract.get("output_content_type")
        if "output_content_type" in contract
        else _cord_value(cord, "output_content_type")
    )
    if content_type is not None:
        _compile_content_type(content_type, "submission_contract.output_content_type")
    if not (isinstance(schema, Mapping) and schema) and content_type is None:
        raise RouteConfigurationError(
            "submission_contract.enabled=true requires a non-empty output schema "
            "or output content type"
        )


def _compile_content_type(value: object, label: str) -> None:
    if (
        not isinstance(value, str)
        or not value.strip()
        or "/" not in value
        or any(character in value for character in "\r\n\x00")
    ):
        raise RouteConfigurationError(f"{label} is invalid")


def _compile_billing_policy(
    route: ExternalRouteConfig,
    profile: object,
    operation: Mapping[str, object],
    response: Mapping[str, object],
) -> None:
    ambiguous = operation.get("bill_ambiguous_transport_errors", False)
    if not isinstance(ambiguous, bool):
        raise RouteConfigurationError(
            "bill_ambiguous_transport_errors must be a boolean"
        )
    partial = response.get("bill_partial_streams", True)
    if not isinstance(partial, bool):
        raise RouteConfigurationError("bill_partial_streams must be a boolean")

    configured = operation.get("billable_http_statuses")
    legacy = operation.get("bill_statuses", [])
    if configured is None:
        configured = (
            legacy
            if isinstance(legacy, list)
            and all(
                isinstance(item, int) and not isinstance(item, bool) for item in legacy
            )
            else []
        )
    if not isinstance(configured, list) or any(
        isinstance(item, bool) or not isinstance(item, int) or not 400 <= item <= 599
        for item in configured
    ):
        raise RouteConfigurationError(
            "billable_http_statuses must contain HTTP error status codes"
        )
    retry = _object(operation.get("retry"), "operation_config.retry")
    non_idempotent = retry.get("retry_non_idempotent", False)
    if not isinstance(non_idempotent, bool):
        raise RouteConfigurationError("retry_non_idempotent must be a boolean")
    policy = retry_policy(route)
    method = getattr(profile, "method", route.method)
    attempts = (
        policy.max_attempts
        if method in {"GET", "HEAD", "PUT", "DELETE"} or policy.retry_non_idempotent
        else 1
    )
    if attempts > 1 and set(configured) & policy.retry_statuses:
        raise RouteConfigurationError("billable HTTP statuses cannot also be retried")


def validate_route_configuration(
    account: ExternalBackendAccount,
    route: ExternalRouteConfig,
    *,
    cord: object | None = None,
    pricing_rules: Sequence[Mapping[str, object]] = (),
) -> None:
    """Compile all runtime route artifacts before configuration is persisted."""

    try:
        request = _object(route.request_config, "request_config")
        response = _object(route.response_config, "response_config")
        operation = _object(route.operation_config, "operation_config")
        _strict_keys(request, _REQUEST_KEYS, "request_config")
        _strict_keys(response, _RESPONSE_KEYS, "response_config")
        _strict_keys(operation, _OPERATION_KEYS, "operation_config")

        validate_input_schema = _boolean_option(
            request,
            "validate_input_schema",
            "request_config.validate_input_schema",
        )
        validate_output_schema = _boolean_option(
            response,
            "validate_output_schema",
            "response_config.validate_output_schema",
        )
        if (
            validate_input_schema
            and route.operation_mode is ExternalOperationMode.REALTIME
        ):
            raise RouteConfigurationError(
                "request_config.validate_input_schema is not supported for realtime "
                "routes; use operation_config.realtime.validate_client_messages"
            )
        if (
            validate_output_schema
            and route.operation_mode is not ExternalOperationMode.SYNC
        ):
            raise RouteConfigurationError(
                "response_config.validate_output_schema is only supported for "
                "buffered sync routes"
            )

        retry = _object(operation.get("retry"), "operation_config.retry")
        _strict_keys(retry, _RETRY_KEYS, "operation_config.retry")
        retry_policy(route)
        if retry.get("body_status_path") is not None:
            DataPath.parse(retry["body_status_path"])
        body_statuses = retry.get("body_statuses", [])
        if not isinstance(body_statuses, list):
            raise RouteConfigurationError("retry.body_statuses must be an array")
        _compile_submission_contract(operation, cord, route.operation_mode)

        PayloadTransform.from_config(request.get("transform"))
        resource_path = request.get("resource_path")
        if resource_path is not None:
            if not isinstance(resource_path, str) or not resource_path.strip():
                raise RouteConfigurationError("resource_path must be a non-empty path")
            DataPath.parse(resource_path)
        _compile_path_parameters(request.get("path_parameters"), "path_parameters")
        allowed_query = request.get("allowed_query_parameters", [])
        if not isinstance(allowed_query, list) or any(
            not isinstance(item, str) for item in allowed_query
        ):
            raise RouteConfigurationError(
                "allowed_query_parameters must be an array of strings"
            )
        _compile_query_parameters(request, route)
        max_request_bytes = request.get("max_request_bytes", 64 * 1024 * 1024)
        if (
            isinstance(max_request_bytes, bool)
            or not isinstance(max_request_bytes, int)
            or not 1 <= max_request_bytes <= settings.max_request_body_bytes
        ):
            raise RouteConfigurationError(
                "max_request_bytes exceeds the process request-body ceiling"
            )

        PublicResponseRules.from_config(response.get("public"))
        artifact_relay_ttl_seconds(response)
        allow_non_json_sse_data = response.get("allow_non_json_sse_data", False)
        if not isinstance(allow_non_json_sse_data, bool):
            raise RouteConfigurationError(
                "response_config.allow_non_json_sse_data must be a boolean"
            )
        if allow_non_json_sse_data:
            raise RouteConfigurationError(
                "non-JSON SSE data cannot cross the provider-obscuring boundary"
            )
        sse_event_map = response.get("sse_event_map", {})
        if not isinstance(sse_event_map, Mapping):
            raise RouteConfigurationError(
                "response_config.sse_event_map must be an object"
            )
        for source_name, public_name in sse_event_map.items():
            if (
                not isinstance(source_name, str)
                or not source_name
                or len(source_name) > 128
                or any(character in source_name for character in "\r\n\x00")
            ):
                raise RouteConfigurationError(
                    "response_config.sse_event_map contains an invalid source name"
                )
            if public_name is not None and (
                not isinstance(public_name, str)
                or not re.fullmatch(r"[A-Za-z0-9_.-]{1,64}", public_name)
            ):
                raise RouteConfigurationError(
                    "response_config.sse_event_map contains an invalid public name"
                )
        _compile_artifact_mappings(response.get("artifacts"))
        _compile_redirects(response.get("redirects"), "response_config.redirects")
        try:
            StreamUsageMode(response.get("usage_mode", "cumulative"))
        except ValueError as exc:
            raise RouteConfigurationError("response usage_mode is invalid") from exc

        usage_config = _selected_usage_config(route)
        usage = UsageMapping.from_config(usage_config) if usage_config else None
        _validate_pricing_metering(
            pricing_rules,
            usage,
            _acceptance_dimension_paths(
                route,
                cord,
                include_cord_schema=validate_input_schema,
            ),
            cord,
        )

        content_type = None
        if cord is not None:
            if validate_input_schema:
                schema, label = _selected_cord_input_schema(cord)
                _compile_required_schema(
                    schema,
                    label,
                    "request_config.validate_input_schema",
                )
            if validate_output_schema:
                _compile_required_schema(
                    _cord_value(cord, "output_schema"),
                    "cord.output_schema",
                    "response_config.validate_output_schema",
                )
            content_type = _cord_value(cord, "output_content_type")
            if content_type is not None:
                _compile_content_type(content_type, "cord.output_content_type")
        elif validate_input_schema or validate_output_schema:
            option = (
                "request_config.validate_input_schema"
                if validate_input_schema
                else "response_config.validate_output_schema"
            )
            raise RouteConfigurationError(f"{option}=true requires a matching Cord")

        if route.operation_mode is ExternalOperationMode.REALTIME:
            profile = _compile_realtime(account, route, cord)
            compile_session_budget(
                account.connection_config,
                operation,
                max_session_seconds=float(getattr(profile, "max_session_seconds")),
            )
        else:
            profile = build_endpoint_profile(account, route)
            validate_profile_headers(profile.static_headers)
            if validate_input_schema and profile.body_mode is BodyMode.RAW:
                raise RouteConfigurationError(
                    "request_config.validate_input_schema is not supported for raw "
                    "request bodies"
                )
            if profile.body_mode in {BodyMode.NONE, BodyMode.RAW} and (
                request.get("transform") is not None
                or request.get("resource_path") is not None
            ):
                raise RouteConfigurationError(
                    f"request transforms are not supported for {profile.body_mode.value} bodies"
                )
            _compile_billing_policy(route, profile, operation, response)
            expected_modes = {
                ExternalOperationMode.SYNC: {ResponseMode.BUFFERED},
                ExternalOperationMode.STREAM: {
                    ResponseMode.SSE,
                    ResponseMode.STREAM,
                },
                ExternalOperationMode.TASK: {ResponseMode.BUFFERED},
            }[route.operation_mode]
            if profile.response_mode not in expected_modes:
                allowed = ", ".join(sorted(mode.value for mode in expected_modes))
                raise RouteConfigurationError(
                    f"{route.operation_mode.value} routes require response mode: {allowed}"
                )
            if (
                validate_output_schema
                and profile.response_mode is not ResponseMode.BUFFERED
            ):
                raise RouteConfigurationError(
                    "response_config.validate_output_schema is only supported for "
                    "buffered sync routes"
                )
            if profile.response_mode is ResponseMode.STREAM and not content_type:
                raise RouteConfigurationError(
                    "raw response streams require a Cord output_content_type"
                )
            if route.operation_mode is ExternalOperationMode.STREAM:
                compile_session_budget(
                    account.connection_config,
                    operation,
                    max_session_seconds=float(profile.timeout.total),
                )
            elif operation.get("session_budget") is not None:
                raise RouteConfigurationError(
                    "operation_config.session_budget is only valid for stream and realtime routes"
                )
        if route.operation_mode is ExternalOperationMode.TASK:
            try:
                inline_result_limit(operation)
            except InlineResultPolicyError as exc:
                raise RouteConfigurationError(str(exc)) from exc
            _compile_task(account, route)
        _compile_artifact_profile(account, route)
    except RouteConfigurationError:
        raise
    except (
        ExternalConfigurationError,
        MappingConfigurationError,
        ProfileError,
        jsonschema.exceptions.SchemaError,
        TypeError,
        ValueError,
    ) as exc:
        raise RouteConfigurationError(str(exc)) from exc


__all__ = ["RouteConfigurationError", "validate_route_configuration"]
