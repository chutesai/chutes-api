"""Authenticated, metered realtime relay for externally executed Chutes."""

from __future__ import annotations

import asyncio
import copy
import re
import time
import uuid
from contextlib import asynccontextmanager, suppress
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Mapping
from urllib.parse import urlsplit, urlunsplit

import orjson
from fastapi import APIRouter, HTTPException, Request, WebSocket, WebSocketDisconnect
from loguru import logger
from sqlalchemy import and_, exists, select
from sqlalchemy.orm import joinedload

from api.api_key.util import get_and_check_api_key
from api.chute.schemas import Chute, ChuteShare
from api.database import get_session
from api.external_transport import (
    MessageDirection,
    NetworkPolicy,
    RequestRejectedError,
    SecretHeaderTemplate,
    UsageObservation,
    WebSocketFrame,
    WebSocketFrameType,
    WebSocketProfile,
    WebSocketRelay,
    WebSocketRelayResult,
    WebSocketRequest,
)
from api.external_transport.errors import ExternalTransportError, ProfileError
from api.external_transport.security import sanitize_client_headers
from api.invocation.util import (
    build_response_headers,
    check_quota_and_balance,
    resolve_rate_limit_headers,
)
from api.metrics.capacity import track_request_completed
from api.payment.pricing import NormalizedUsage, PricingConfigurationError
from api.user.schemas import User
from api.user.service import subnet_role_accessible
from api.user.tokens import get_user_from_token

from .config import ExternalConfigurationError, allow_insecure_http
from .circuit import record_upstream_result
from .mapping import (
    MappingConfigurationError,
    MappingExtractionError,
    StreamUsageMode,
    UsageMapping,
    extract_value,
    merge_stream_usage,
    scrub_public_response,
    transform_payload,
)
from .request_mapping import (
    ExternalRequestMappingError,
    map_upstream_query_parameters,
)
from .operation_lifecycle import (
    USAGE_CHECKPOINT_INTERVAL_SECONDS,
    UsageBudgetMonitor,
    UsageCheckpointLoop,
    session_recovery_deadline,
)
from .metrics import upstream_latency, upstream_requests
from .schema_validation import (
    RemoteSchemaReferenceError,
    UnsafeSchemaError,
    local_json_schema_validator,
)
from .schemas import (
    ExternalBackendAccount,
    ExternalChuteBinding,
    ExternalOperation,
    ExternalOperationMode,
    ExternalOperationStatus,
    ExternalRouteConfig,
)

router = APIRouter()

_FINALIZERS: set[asyncio.Task[Any]] = set()
_HOST_RE = re.compile(r"^[a-z0-9][a-z0-9.-]{0,252}$")
_PUBLIC_CLOSE_CODES = frozenset(
    {1000, 1001, 1002, 1003, 1007, 1008, 1009, 1011, 1012, 1013, 1014}
)
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


@dataclass(frozen=True, slots=True)
class RealtimeContext:
    chute: Chute
    binding: ExternalChuteBinding
    account: ExternalBackendAccount
    route: ExternalRouteConfig
    cord: dict[str, Any]


# These narrow adapters keep importing this optional protocol module inexpensive.
# They also leave the HTTP service as the single implementation of operation and
# settlement semantics.
async def _pricing_snapshot(*args: Any, **kwargs: Any):
    from .service import _pricing_snapshot as implementation

    return await implementation(*args, **kwargs)


def _validate_metering_config(*args: Any, **kwargs: Any) -> None:
    from .service import _validate_metering_config as implementation

    implementation(*args, **kwargs)


def _upstream_query_parameters(
    route: ExternalRouteConfig, query: Mapping[str, Any]
) -> dict[str, Any]:
    return map_upstream_query_parameters(route, {}, query)


async def _create_operation(*args: Any, **kwargs: Any):
    from .service import _create_operation as implementation

    return await implementation(*args, **kwargs)


async def _update_operation(*args: Any, **kwargs: Any) -> None:
    from .service import _update_operation as implementation

    await implementation(*args, **kwargs)


def _secret_resolver(*args: Any, **kwargs: Any):
    from .service import build_secret_resolver as implementation

    return implementation(*args, **kwargs)


async def settle_operation(*args: Any, **kwargs: Any) -> None:
    from .service import settle_operation as implementation

    await implementation(*args, **kwargs)


async def _running_budget_check(*args: Any, **kwargs: Any):
    from .service import _running_budget_check as implementation

    return await implementation(*args, **kwargs)


async def _checkpoint_running_usage(*args: Any, **kwargs: Any) -> None:
    from .service import _checkpoint_running_usage as implementation

    kwargs.setdefault("update_operation", _update_operation)
    await implementation(*args, **kwargs)


def _observed_cost_metadata(*args: Any, **kwargs: Any):
    from .service import _observed_cost_metadata as implementation

    return implementation(*args, **kwargs)


def _session_budget_from_metadata(*args: Any, **kwargs: Any):
    from .governance import session_budget_from_metadata as implementation

    return implementation(*args, **kwargs)


def _compile_session_budget(*args: Any, **kwargs: Any):
    from .governance import compile_session_budget as implementation

    return implementation(*args, **kwargs)


async def chute_id_by_slug(slug: str) -> str | None:
    async with get_session(readonly=True) as db:
        return (
            await db.execute(select(Chute.chute_id).where(Chute.slug == slug))
        ).scalar_one_or_none()


async def is_shared(chute_id: str, user_id: str) -> bool:
    async with get_session(readonly=True) as db:
        return bool(
            (
                await db.execute(
                    select(
                        exists().where(
                            and_(
                                ChuteShare.chute_id == chute_id,
                                ChuteShare.shared_to == user_id,
                            )
                        )
                    )
                )
            ).scalar()
        )


def _object(value: object, label: str) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ExternalConfigurationError(f"{label} must be an object")
    return dict(value)


def _string_list(value: object, label: str, *, maximum: int = 128) -> tuple[str, ...]:
    if value is None:
        return ()
    if not isinstance(value, list) or len(value) > maximum:
        raise ExternalConfigurationError(f"{label} must be an array of strings")
    result: list[str] = []
    for item in value:
        if not isinstance(item, str) or not item.strip():
            raise ExternalConfigurationError(f"{label} must be an array of strings")
        result.append(item.strip())
    return tuple(result)


def _number(
    value: object,
    default: float,
    label: str,
    *,
    minimum: float,
    maximum: float,
) -> float:
    if value is None:
        return default
    if isinstance(value, bool):
        raise ExternalConfigurationError(f"{label} must be a number")
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ExternalConfigurationError(f"{label} must be a number") from exc
    if not minimum <= parsed <= maximum:
        raise ExternalConfigurationError(
            f"{label} must be between {minimum:g} and {maximum:g}"
        )
    return parsed


def _integer(
    value: object,
    default: int,
    label: str,
    *,
    minimum: int,
    maximum: int,
) -> int:
    parsed = _number(
        value, default, label, minimum=float(minimum), maximum=float(maximum)
    )
    if not parsed.is_integer():
        raise ExternalConfigurationError(f"{label} must be an integer")
    return int(parsed)


def _realtime_config(route: ExternalRouteConfig) -> dict[str, Any]:
    operation = _object(route.operation_config, "operation_config")
    realtime = operation.get("realtime")
    websocket = operation.get("websocket")
    if realtime is not None and websocket is not None:
        raise ExternalConfigurationError(
            "operation_config must not define both realtime and websocket"
        )
    return _object(
        realtime if realtime is not None else websocket,
        "operation_config.realtime",
    )


def _websocket_base_url(
    account: ExternalBackendAccount,
    route: ExternalRouteConfig,
    endpoint: Mapping[str, Any],
) -> str:
    raw = str(endpoint.get("base_url") or route.base_url or account.base_url).rstrip(
        "/"
    )
    parsed = urlsplit(raw)
    scheme = parsed.scheme.lower()
    connection = _object(account.connection_config, "connection_config")
    if scheme in {"https", "wss"}:
        scheme = "wss"
    elif scheme in {"http", "ws"} and allow_insecure_http(connection):
        scheme = "ws"
    else:
        raise ExternalConfigurationError(
            "realtime endpoints must use secure WebSockets"
        )
    if (
        not parsed.hostname
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
    ):
        raise ExternalConfigurationError("realtime base URL is invalid")
    return urlunsplit((scheme, parsed.netloc, parsed.path.rstrip("/"), "", ""))


def _secret_headers(
    account: ExternalBackendAccount,
) -> tuple[SecretHeaderTemplate, ...]:
    references = dict(account.credential_references or {})
    result: list[SecretHeaderTemplate] = []
    for raw in account.auth_header_templates or []:
        try:
            resolved = {
                field: references[credential_name]
                for field, credential_name in raw["references"].items()
            }
            result.append(
                SecretHeaderTemplate(
                    name=raw["name"],
                    template=raw["template"],
                    references=resolved,
                )
            )
        except (KeyError, TypeError) as exc:
            raise ExternalConfigurationError(
                "realtime credential template is invalid"
            ) from exc
    return tuple(result)


def build_websocket_profile(
    account: ExternalBackendAccount, route: ExternalRouteConfig
) -> WebSocketProfile:
    """Compile one persisted realtime route into a bounded transport profile."""

    if route.operation_mode is not ExternalOperationMode.REALTIME:
        raise ExternalConfigurationError("route is not configured for realtime use")
    endpoint = _realtime_config(route)
    request_config = _object(route.request_config, "request_config")
    connection = _object(account.connection_config, "connection_config")
    network_config = _object(connection.get("network"), "connection_config.network")
    base_url = _websocket_base_url(account, route, endpoint)
    parsed = urlsplit(base_url)
    assert parsed.hostname is not None

    allowed_hosts = {parsed.hostname.lower().rstrip(".")}
    allowed_hosts.update(
        item.lower().rstrip(".")
        for item in _string_list(
            network_config.get("allowed_hosts"),
            "connection_config.network.allowed_hosts",
        )
    )
    allowed_hosts.update(
        item.lower().rstrip(".")
        for item in _string_list(
            endpoint.get("allowed_hosts"), "realtime.allowed_hosts"
        )
    )
    ports = {parsed.port or (443 if parsed.scheme == "wss" else 80)}
    for item in network_config.get("allowed_ports", []):
        if isinstance(item, bool) or not isinstance(item, int):
            raise ExternalConfigurationError("allowed ports must be integers")
        ports.add(item)

    static_headers = dict(
        _object(request_config.get("static_headers"), "request_config.static_headers")
    )
    static_headers.update(
        _object(endpoint.get("static_headers"), "realtime.static_headers")
    )
    allowed_request_headers = _string_list(
        endpoint.get(
            "allowed_request_headers", request_config.get("allowed_request_headers")
        ),
        "realtime.allowed_request_headers",
    )

    heartbeat = endpoint.get("heartbeat_seconds", 30.0)
    if heartbeat is not None:
        heartbeat = _number(
            heartbeat,
            30.0,
            "realtime.heartbeat_seconds",
            minimum=0.1,
            maximum=3600.0,
        )
    return WebSocketProfile(
        name=f"{route.protocol}.realtime",
        base_url=base_url,
        path_template=str(endpoint.get("path_template") or route.path_template),
        allowed_request_headers=frozenset(allowed_request_headers),
        static_headers=static_headers,
        secret_headers=_secret_headers(account),
        allowed_subprotocols=_string_list(
            endpoint.get("allowed_subprotocols"),
            "realtime.allowed_subprotocols",
            maximum=32,
        ),
        require_subprotocol=bool(endpoint.get("require_subprotocol", False)),
        network=NetworkPolicy(
            allowed_schemes=frozenset({parsed.scheme}),
            allowed_hosts=tuple(sorted(allowed_hosts)),
            allowed_ports=frozenset(ports),
            allow_private_networks=False,
        ),
        handshake_timeout_seconds=_number(
            endpoint.get("handshake_timeout_seconds"),
            15.0,
            "realtime.handshake_timeout_seconds",
            minimum=0.1,
            maximum=120.0,
        ),
        idle_timeout_seconds=_number(
            endpoint.get("idle_timeout_seconds"),
            60.0,
            "realtime.idle_timeout_seconds",
            minimum=0.1,
            maximum=86400.0,
        ),
        max_session_seconds=_number(
            endpoint.get("max_session_seconds"),
            3600.0,
            "realtime.max_session_seconds",
            minimum=0.1,
            maximum=86400.0,
        ),
        max_message_bytes=_integer(
            endpoint.get("max_message_bytes"),
            4 * 1024 * 1024,
            "realtime.max_message_bytes",
            minimum=1,
            maximum=64 * 1024 * 1024,
        ),
        heartbeat_seconds=heartbeat,
    )


def _host_slug(host: str) -> str | None:
    try:
        hostname = urlsplit(f"//{host}").hostname
    except ValueError:
        return None
    if not hostname:
        return None
    hostname = hostname.lower().rstrip(".")
    if not _HOST_RE.fullmatch(hostname) or "." not in hostname:
        return None
    slug = hostname.split(".", 1)[0]
    return slug if slug != "api" else None


def _cord_dict(value: Any) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return dict(value)
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    raise ExternalConfigurationError("external Cord configuration is invalid")


def _select_realtime_route(
    binding: ExternalChuteBinding, chute: Chute, public_path: str
) -> tuple[ExternalRouteConfig, dict[str, Any]]:
    routes = [ExternalRouteConfig.model_validate(item) for item in binding.routes]
    matches: list[tuple[ExternalRouteConfig, dict[str, Any]]] = []
    for raw_cord in chute.cords or []:
        cord = _cord_dict(raw_cord)
        cord_path = cord.get("path")
        exposed_path = cord.get("public_api_path") or cord_path
        if exposed_path != public_path:
            continue
        for route in routes:
            if (
                route.cord_path == cord_path
                and route.operation_mode is ExternalOperationMode.REALTIME
            ):
                matches.append((route, cord))
    if len(matches) != 1:
        raise ExternalConfigurationError("realtime Cord route is not configured")
    return matches[0]


async def _realtime_target(websocket: WebSocket) -> tuple[str | None, str]:
    """Resolve hostname and canonical API-domain sockets to one public Cord path."""

    slug = _host_slug(websocket.headers.get("host", ""))
    if slug:
        return await chute_id_by_slug(slug), websocket.url.path
    canonical = re.fullmatch(r"/chutes/([^/]+)/(.+)", websocket.url.path)
    if canonical:
        return canonical.group(1), f"/{canonical.group(2)}"
    return None, websocket.url.path


async def _resolve_context(websocket: WebSocket) -> RealtimeContext:
    chute_id, public_path = await _realtime_target(websocket)
    if not chute_id:
        raise HTTPException(status_code=404, detail="No matching chute found")
    async with get_session(readonly=True) as db:
        chute = (
            (
                await db.execute(
                    select(Chute)
                    .options(
                        joinedload(Chute.user).joinedload(User.current_balance),
                        joinedload(Chute.external_binding).joinedload(
                            ExternalChuteBinding.account
                        ),
                    )
                    .where(Chute.chute_id == chute_id)
                )
            )
            .unique()
            .scalar_one_or_none()
        )
    if (
        not chute
        or chute.execution_backend != "external"
        or chute.disabled
        or not chute.external_binding
        or not chute.external_binding.enabled
        or not chute.external_binding.account
        or not chute.external_binding.account.enabled
    ):
        raise HTTPException(status_code=404, detail="No matching chute found")
    route, cord = _select_realtime_route(chute.external_binding, chute, public_path)
    return RealtimeContext(
        chute=chute,
        binding=chute.external_binding,
        account=chute.external_binding.account,
        route=route,
        cord=cord,
    )


def _request_adapter(websocket: WebSocket, chute_id: str) -> Request:
    source = websocket.scope
    scope = {
        "type": "http",
        "asgi": source.get("asgi", {"version": "3.0"}),
        "http_version": source.get("http_version", "1.1"),
        "method": "GET",
        "scheme": "https" if source.get("scheme") == "wss" else "http",
        "path": source.get("path", "/"),
        "raw_path": source.get("raw_path", source.get("path", "/").encode()),
        "query_string": source.get("query_string", b""),
        "root_path": source.get("root_path", ""),
        "headers": source.get("headers", []),
        "client": source.get("client"),
        "server": source.get("server"),
        "state": {},
    }
    request = Request(scope)
    request.state.auth_method = "invoke"
    request.state.auth_object_type = "chutes"
    request.state.auth_object_id = chute_id
    request.state.chute_id = chute_id
    request.state.body_sha256 = None
    request.state.free_invocation = False
    request.state.started_at = time.time()
    request.state.client_ip = websocket.client.host if websocket.client else ""
    request.state.has_resolved_ip = False
    return request


async def _authenticate(websocket: WebSocket, chute_id: str) -> tuple[User, Request]:
    request = _request_adapter(websocket, chute_id)
    authorization = websocket.headers.get("authorization", "").strip()
    if not authorization or len(authorization) > 8192:
        raise HTTPException(status_code=401, detail="Authentication required")
    pieces = authorization.split(None, 1)
    if len(pieces) == 2:
        if pieces[0].lower() != "bearer":
            raise HTTPException(status_code=401, detail="Authentication required")
        token = pieces[1].strip()
        bearer = True
    else:
        token = pieces[0]
        bearer = False
    if not token:
        raise HTTPException(status_code=401, detail="Authentication required")

    user = None
    if bearer and not token.startswith(("cpk_", "cak_")):
        user = await get_user_from_token(token, request)
    if user is None:
        credential = await get_and_check_api_key(token, request)
        if credential:
            request.state.api_key = credential
            user = credential.user
    if user is None:
        raise HTTPException(status_code=401, detail="Authentication required")
    return user, request


def _allowed_query(websocket: WebSocket, route: ExternalRouteConfig) -> dict[str, Any]:
    request_config = _object(route.request_config, "request_config")
    endpoint = _realtime_config(route)
    allowed = set(
        _string_list(
            endpoint.get(
                "allowed_query_parameters",
                request_config.get("allowed_query_parameters", []),
            ),
            "realtime.allowed_query_parameters",
        )
    )
    result: dict[str, Any] = {}
    for key, value in websocket.query_params.multi_items():
        if key not in allowed or _is_sensitive_query_name(key):
            continue
        if key in result:
            current = result[key]
            result[key] = (
                current + [value] if isinstance(current, list) else [current, value]
            )
        else:
            result[key] = value
    return result


def _path_parameters(
    route: ExternalRouteConfig, query: Mapping[str, Any]
) -> dict[str, str | int]:
    request_config = _object(route.request_config, "request_config")
    endpoint = _realtime_config(route)
    configured = _object(
        endpoint.get("path_parameters", request_config.get("path_parameters")),
        "realtime.path_parameters",
    )
    sources = {
        "query": query,
        "context": {
            "resource": route.upstream_resource_id,
            "model": route.upstream_resource_id,
            "upstream_resource_id": route.upstream_resource_id,
        },
    }
    result: dict[str, str | int] = {
        "resource": route.upstream_resource_id,
        "model": route.upstream_resource_id,
        "upstream_resource_id": route.upstream_resource_id,
    }
    for name, rule in configured.items():
        if isinstance(rule, Mapping):
            if "value" in rule:
                value = rule["value"]
            else:
                path = rule.get("path")
                if not isinstance(path, str):
                    raise ExternalConfigurationError(
                        "realtime path parameter requires a path or value"
                    )
                value = extract_value(
                    sources, path, required=bool(rule.get("required", True))
                )
        elif isinstance(rule, str) and rule.startswith(("query.", "context.")):
            value = extract_value(sources, rule, required=True)
        else:
            value = rule
        if isinstance(value, bool) or not isinstance(value, (str, int)):
            raise RequestRejectedError("invalid realtime path parameter")
        normalized_name = str(name)
        if (
            normalized_name in {"resource", "model", "upstream_resource_id"}
            and value != route.upstream_resource_id
        ):
            raise RequestRejectedError(
                "realtime resource path parameters must use the configured resource"
            )
        result[normalized_name] = value
    return result


def _client_subprotocols(websocket: WebSocket) -> tuple[str, ...]:
    value = websocket.headers.get("sec-websocket-protocol", "")
    return tuple(item.strip() for item in value.split(",") if item.strip())


def _public_mapping(
    route: ExternalRouteConfig,
    *,
    chute_name: str,
    invocation_id: str,
    operation_id: str,
) -> dict[str, Any]:
    response = _object(route.response_config, "response_config")
    result = copy.deepcopy(_object(response.get("public"), "response_config.public"))
    rewrites = _object(result.get("rewrite_keys"), "public.rewrite_keys")
    for key in ("model", "model_id", "model_name"):
        rewrites[key] = chute_name
    rewrites["request_id"] = invocation_id
    for key in ("task_id", "job_id", "operation_id"):
        rewrites[key] = operation_id
    result["rewrite_keys"] = rewrites
    return result


def _request_message_transform(
    route: ExternalRouteConfig,
    payload: Any,
    *,
    invocation_id: str,
    chute_name: str,
) -> Any:
    request_config = _object(route.request_config, "request_config")
    context = {
        "resource": route.upstream_resource_id,
        "model": route.upstream_resource_id,
        "invocation_id": invocation_id,
        "chute_name": chute_name,
    }
    result = transform_payload(
        payload,
        request_config.get("transform"),
        request=payload,
        context=context,
    )
    if isinstance(result, dict):
        for key in ("model", "resource", "upstream_resource_id"):
            if key in result:
                result[key] = route.upstream_resource_id
    resource_path = request_config.get("resource_path")
    if resource_path:
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
        if (
            extract_value(result, resource_path, required=True)
            != route.upstream_resource_id
        ):
            raise ExternalConfigurationError(
                "realtime resource_path could not be pinned"
            )
    return result


async def _validate_message_schema(payload: Any, schema: Mapping[str, Any]) -> None:
    def validate() -> None:
        try:
            validator = local_json_schema_validator(schema)
        except (RemoteSchemaReferenceError, UnsafeSchemaError) as exc:
            raise ExternalConfigurationError(str(exc)) from exc
        if error := next(validator.iter_errors(payload), None):
            raise ValueError(error.message)

    try:
        await asyncio.wait_for(asyncio.to_thread(validate), timeout=2.0)
    except asyncio.TimeoutError as exc:
        raise ExternalConfigurationError(
            "realtime message validation timed out"
        ) from exc


class RealtimeUsageMeter:
    """Accumulate authoritative upstream observations for one socket session."""

    def __init__(self, route: ExternalRouteConfig) -> None:
        response = _object(route.response_config, "response_config")
        operation = _object(route.operation_config, "operation_config")
        realtime = _realtime_config(route)
        self._config = (
            realtime.get("usage") or operation.get("usage") or response.get("usage")
        )
        compiled = UsageMapping.from_config(self._config) if self._config else None
        self._request_usage = (
            UsageMapping(
                fields=tuple(
                    field for field in compiled.fields if field.rule.source == "request"
                ),
                default_requests=0,
            )
            if compiled is not None
            else None
        )
        self._response_usage = (
            UsageMapping(
                fields=tuple(
                    field for field in compiled.fields if field.rule.source != "request"
                ),
                default_requests=0,
            )
            if compiled is not None
            else None
        )
        try:
            self._mode = StreamUsageMode(
                realtime.get(
                    "usage_mode",
                    operation.get(
                        "usage_mode", response.get("usage_mode", "cumulative")
                    ),
                )
            )
        except ValueError as exc:
            raise ExternalConfigurationError(
                "realtime usage mode is not supported"
            ) from exc
        self._route = route
        self._usage = NormalizedUsage(requests=1)
        self._last_request: Any = None
        self._lock = asyncio.Lock()

    @property
    def usage(self) -> NormalizedUsage:
        return self._usage

    async def observe(self, observation: UsageObservation) -> None:
        if observation.json_value is None:
            return
        async with self._lock:
            if observation.direction is MessageDirection.OUTBOUND:
                self._last_request = copy.deepcopy(observation.json_value)
                if self._request_usage is not None and self._request_usage.fields:
                    measured = self._request_usage.extract(
                        request=observation.json_value,
                        response={},
                        payload=observation.json_value,
                        context={"resource": self._route.upstream_resource_id},
                    )
                    self._usage = merge_stream_usage(self._usage, measured, self._mode)
                return
            if (
                observation.direction is not MessageDirection.INBOUND
                or self._response_usage is None
                or not self._response_usage.fields
            ):
                return
            measured = self._response_usage.extract(
                request=self._last_request,
                response=observation.json_value,
                payload=observation.json_value,
                context={"resource": self._route.upstream_resource_id},
            )
            self._usage = merge_stream_usage(self._usage, measured, self._mode)


class StarletteWebSocketPeer:
    """Adapt Starlette frames while transforming requests and scrubbing responses."""

    def __init__(
        self,
        websocket: WebSocket,
        *,
        route: ExternalRouteConfig,
        cord: Mapping[str, Any],
        chute_name: str,
        invocation_id: str,
        operation_id: str,
    ) -> None:
        self._websocket = websocket
        self._route = route
        self._cord = dict(cord)
        self._chute_name = chute_name
        self._invocation_id = invocation_id
        self._public = _public_mapping(
            route,
            chute_name=chute_name,
            invocation_id=invocation_id,
            operation_id=operation_id,
        )
        endpoint = _realtime_config(route)
        validate_messages = endpoint.get("validate_client_messages", False)
        if not isinstance(validate_messages, bool):
            raise ExternalConfigurationError(
                "realtime.validate_client_messages must be a boolean"
            )
        self._validate_messages = validate_messages
        self._schema: dict[str, Any] | None = None
        if self._validate_messages:
            schema = endpoint.get("message_schema")
            if not schema:
                schema = self._cord.get("input_schema") or self._cord.get(
                    "minimal_input_schema"
                )
            if not isinstance(schema, Mapping) or not schema:
                raise ExternalConfigurationError(
                    "realtime client validation requires a non-empty message schema"
                )
            self._schema = dict(schema)
        self._allow_client_non_json_text = endpoint.get(
            "allow_client_non_json_text", False
        )
        self._allow_client_binary = endpoint.get("allow_client_binary", False)
        self._allow_upstream_non_json_text = endpoint.get(
            "allow_upstream_non_json_text", False
        )
        self._allow_upstream_binary = endpoint.get("allow_upstream_binary", False)
        for name, value in (
            ("allow_client_non_json_text", self._allow_client_non_json_text),
            ("allow_client_binary", self._allow_client_binary),
            ("allow_upstream_non_json_text", self._allow_upstream_non_json_text),
            ("allow_upstream_binary", self._allow_upstream_binary),
        ):
            if not isinstance(value, bool):
                raise ExternalConfigurationError(f"realtime.{name} must be a boolean")
        self.client_disconnected = False
        self.client_rejected = False
        self._closed = False

    async def receive(self) -> WebSocketFrame:
        try:
            message = await self._websocket.receive()
        except WebSocketDisconnect as exc:
            self.client_disconnected = True
            return WebSocketFrame(WebSocketFrameType.CLOSE, close_code=exc.code or 1001)
        message_type = message.get("type")
        if message_type == "websocket.disconnect":
            self.client_disconnected = True
            return WebSocketFrame(
                WebSocketFrameType.CLOSE, close_code=message.get("code") or 1001
            )
        if message_type != "websocket.receive":
            self.client_rejected = True
            return WebSocketFrame(WebSocketFrameType.CLOSE, close_code=1002)
        if message.get("text") is not None:
            text = message["text"]
            try:
                payload = orjson.loads(text)
            except orjson.JSONDecodeError:
                if not self._allow_client_non_json_text:
                    self.client_rejected = True
                    return WebSocketFrame(WebSocketFrameType.CLOSE, close_code=1008)
                return WebSocketFrame(WebSocketFrameType.TEXT, text)
            try:
                if self._validate_messages and self._schema is not None:
                    await _validate_message_schema(payload, self._schema)
                payload = _request_message_transform(
                    self._route,
                    payload,
                    invocation_id=self._invocation_id,
                    chute_name=self._chute_name,
                )
            except Exception:
                self.client_rejected = True
                return WebSocketFrame(WebSocketFrameType.CLOSE, close_code=1008)
            return WebSocketFrame(
                WebSocketFrameType.TEXT, orjson.dumps(payload).decode("utf-8")
            )
        if message.get("bytes") is not None:
            if not self._allow_client_binary:
                self.client_rejected = True
                return WebSocketFrame(WebSocketFrameType.CLOSE, close_code=1008)
            return WebSocketFrame(WebSocketFrameType.BINARY, message["bytes"])
        self.client_rejected = True
        return WebSocketFrame(WebSocketFrameType.CLOSE, close_code=1002)

    async def send(self, frame: WebSocketFrame) -> None:
        if frame.kind is WebSocketFrameType.TEXT:
            assert isinstance(frame.data, str)
            try:
                payload = orjson.loads(frame.data)
            except orjson.JSONDecodeError:
                if not self._allow_upstream_non_json_text:
                    raise MappingExtractionError(
                        "opaque upstream text frames are not permitted"
                    )
                await self._websocket.send_text(frame.data)
                return
            public = scrub_public_response(payload, self._public)
            await self._websocket.send_text(orjson.dumps(public).decode("utf-8"))
        elif frame.kind is WebSocketFrameType.BINARY:
            assert isinstance(frame.data, bytes)
            if not self._allow_upstream_binary:
                raise MappingExtractionError(
                    "opaque upstream binary frames are not permitted"
                )
            await self._websocket.send_bytes(frame.data)
        else:
            await self.close(frame.close_code or 1000, frame.close_reason)

    async def close(self, code: int = 1000, reason: str = "") -> None:
        if self._closed:
            return
        self._closed = True
        safe_code = code if code in _PUBLIC_CLOSE_CODES else 1011
        safe_reason = {
            1000: "normal closure",
            1001: "endpoint going away",
            1002: "protocol error",
            1003: "unsupported data",
            1007: "invalid data",
            1008: "policy violation",
            1009: "message too large",
            1011: "realtime execution failed",
            1012: "service restart",
            1013: "retry later",
            1014: "gateway error",
        }[safe_code]
        with suppress(Exception):
            await self._websocket.close(code=safe_code, reason=safe_reason)


def _accept_headers(request: Request, invocation_id: str, operation_id: str):
    values = build_response_headers(
        request,
        {
            "X-Chutes-InvocationID": invocation_id,
            "X-Chutes-OperationID": operation_id,
        },
    )
    return [
        (name.lower().encode("ascii"), str(value).encode("latin-1"))
        for name, value in values.items()
    ]


async def _close_local(websocket: WebSocket, code: int, reason: str) -> None:
    with suppress(Exception):
        await websocket.close(code=code, reason=reason)


def _finish_finalizer(task: asyncio.Task[Any]) -> None:
    _FINALIZERS.discard(task)
    if task.cancelled():
        logger.error("Realtime accounting task was cancelled")
        return
    if task.exception() is not None:
        logger.error("Realtime accounting task failed")


def _schedule_finalizer(coroutine: Any) -> asyncio.Task[Any]:
    task = asyncio.create_task(coroutine)
    _FINALIZERS.add(task)
    task.add_done_callback(_finish_finalizer)
    return task


async def _finalize_operation(
    operation_id: str,
    *,
    chute_id: str,
    status_value: str,
    usage: NormalizedUsage,
    billable: bool,
    error: dict[str, Any] | None,
) -> None:
    try:
        await _update_operation(
            operation_id,
            status=status_value,
            error=error,
            finished_at=datetime.now(timezone.utc),
            next_poll_at=None,
            _settlement_metadata_patch={"billable": billable},
        )
    finally:
        await settle_operation(operation_id, usage, billable=billable)
    if billable:
        with suppress(Exception):
            track_request_completed(chute_id)


async def shutdown_realtime(timeout_seconds: float = 30.0) -> int:
    """Wait for disconnect-triggered accounting tasks during application shutdown."""

    tasks = tuple(_FINALIZERS)
    if not tasks:
        return 0
    done, pending = await asyncio.wait(tasks, timeout=max(0.0, timeout_seconds))
    for task in done:
        with suppress(Exception, asyncio.CancelledError):
            task.result()
    if pending:
        logger.warning("Realtime shutdown still has {} accounting tasks", len(pending))
    return len(pending)


@asynccontextmanager
async def realtime_lifespan(_app: Any = None):
    """Optional lifespan hook for applications that mount :data:`router`."""

    yield
    await shutdown_realtime()


async def handle_external_realtime(websocket: WebSocket) -> None:
    """Resolve, authenticate, relay, sanitize, and settle one realtime session."""

    try:
        context = await _resolve_context(websocket)
    except HTTPException:
        await _close_local(websocket, 4404, "route unavailable")
        return
    except Exception:
        await _close_local(websocket, 1013, "temporarily unavailable")
        return

    try:
        current_user, request = await _authenticate(websocket, context.chute.chute_id)
    except HTTPException as exc:
        code = 4403 if exc.status_code == 403 else 4401
        await _close_local(websocket, code, "authentication failed")
        return
    except Exception:
        await _close_local(websocket, 4401, "authentication failed")
        return

    try:
        accessible = (
            context.chute.public
            or context.chute.user_id == current_user.user_id
            or await is_shared(context.chute.chute_id, current_user.user_id)
            or subnet_role_accessible(context.chute, current_user)
        )
    except Exception:
        await _close_local(websocket, 1013, "temporarily unavailable")
        return
    if not accessible:
        await _close_local(websocket, 4404, "route unavailable")
        return
    if (
        websocket.headers.get("x-tee-only", "").lower() == "true"
        and not context.chute.tee
    ):
        await _close_local(websocket, 4426, "requested security mode unavailable")
        return

    invocation_id = str(uuid.uuid4())
    operation: ExternalOperation | None = None
    meter: RealtimeUsageMeter | None = None
    connection: Any = None
    usage_checkpoints: UsageCheckpointLoop | None = None
    budget_monitor: UsageBudgetMonitor | None = None
    pricing: Mapping[str, Any] = {}
    connected = False
    final_status = ExternalOperationStatus.FAILED.value
    final_error: dict[str, Any] | None = {
        "message": "Realtime execution could not be completed.",
        "code": "realtime_error",
        "retryable": True,
        "details": {},
    }
    peer: StarletteWebSocketPeer | None = None
    try:
        request.scope["method"] = str(
            context.cord.get("public_api_method") or request.method
        ).upper()
        resolve_rate_limit_headers(request, current_user, context.chute)
        await check_quota_and_balance(request, current_user, context.chute)
        endpoint = _realtime_config(context.route)
        dimensions = _object(endpoint.get("pricing_dimensions"), "pricing_dimensions")
        pricing = await _pricing_snapshot(
            current_user,
            context.chute,
            context.cord,
            request,
            dimensions,
        )
        _validate_metering_config(context.route, pricing)
        profile = build_websocket_profile(context.account, context.route)
        created_operation, reused = await _create_operation(
            binding=context.binding,
            account=context.account,
            chute=context.chute,
            current_user=current_user,
            route=context.route,
            selected_cord=context.cord,
            pricing_snapshot=pricing,
            request=request,
            invocation_id=invocation_id,
            idempotency_key=None,
            idempotency_fingerprint="",
            dispatch_recovery_seconds=profile.handshake_timeout_seconds + 60.0,
            session_timeout_seconds=profile.max_session_seconds,
        )
        if reused:
            # Realtime sessions do not accept idempotency keys and must always have
            # independent lifecycle and usage records.
            raise ExternalConfigurationError(
                "realtime operation was unexpectedly reused"
            )
        operation = created_operation
        query = _upstream_query_parameters(
            context.route, _allowed_query(websocket, context.route)
        )
        safe_headers = sanitize_client_headers(
            dict(websocket.headers), profile.allowed_request_headers
        )
        outbound = WebSocketRequest(
            path_parameters=_path_parameters(context.route, query),
            query=query,
            headers=safe_headers,
            subprotocols=_client_subprotocols(websocket),
        )
        meter = RealtimeUsageMeter(context.route)
        connect_started = time.monotonic()
        try:
            connection = await WebSocketRelay(
                secret_resolver=_secret_resolver(context.account),
                usage_hooks=(meter.observe,),
            ).connect(profile, outbound)
        except ExternalTransportError:
            upstream_latency.labels(phase="realtime_connect").observe(
                time.monotonic() - connect_started
            )
            upstream_requests.labels(
                phase="realtime_connect", outcome="transport_error"
            ).inc()
            await record_upstream_result(
                context.account.account_id, transport_error=True
            )
            raise
        upstream_latency.labels(phase="realtime_connect").observe(
            time.monotonic() - connect_started
        )
        upstream_requests.labels(phase="realtime_connect", outcome="connected").inc()
        await record_upstream_result(context.account.account_id, status_code=101)
        connected = True
        started_at = datetime.now(timezone.utc)
        started_monotonic = time.monotonic()
        recovery_deadline = session_recovery_deadline(
            started_at, profile.max_session_seconds
        )
        operation_metadata = dict(getattr(operation, "settlement_metadata", None) or {})
        session_budget = _session_budget_from_metadata(operation_metadata)
        if session_budget is None:
            session_budget = _compile_session_budget(
                context.account.connection_config,
                context.route.operation_config,
                max_session_seconds=profile.max_session_seconds,
            )
            operation_metadata["session_budget"] = session_budget.snapshot()
            operation.settlement_metadata = operation_metadata
        await _update_operation(
            operation.operation_id,
            status=ExternalOperationStatus.RUNNING.value,
            started_at=started_at,
            expires_at=recovery_deadline,
            next_poll_at=recovery_deadline,
            usage=meter.usage.to_dict(),
            settlement_metadata=_observed_cost_metadata(
                operation, pricing, meter.usage
            ),
        )
        usage_checkpoints = UsageCheckpointLoop(
            operation_id=operation.operation_id,
            read_usage=lambda: meter.usage.to_dict(),
            persist_usage=lambda value: _checkpoint_running_usage(
                operation.operation_id,
                pricing,
                value,
                settlement_metadata=operation_metadata,
                elapsed_seconds=max(0.0, time.monotonic() - started_monotonic),
            ),
            initial_usage=meter.usage.to_dict(),
            interval_seconds=min(
                USAGE_CHECKPOINT_INTERVAL_SECONDS,
                session_budget.check_interval_seconds,
            ),
            always_persist=True,
        )
        usage_checkpoints.start()
        budget_monitor = UsageBudgetMonitor(
            operation_id=operation.operation_id,
            read_usage=lambda: meter.usage.to_dict(),
            check_usage=lambda value: _running_budget_check(
                operation.operation_id, pricing, value
            ),
            on_exceeded=lambda _reason: connection.close(1008),
            interval_seconds=min(
                USAGE_CHECKPOINT_INTERVAL_SECONDS,
                session_budget.check_interval_seconds,
            ),
        )
        budget_monitor.start()
        await websocket.accept(
            subprotocol=connection.subprotocol,
            headers=_accept_headers(request, invocation_id, operation.operation_id),
        )
        peer = StarletteWebSocketPeer(
            websocket,
            route=context.route,
            cord=context.cord,
            chute_name=context.chute.name,
            invocation_id=invocation_id,
            operation_id=operation.operation_id,
        )
        relay_result: WebSocketRelayResult = await connection.relay(peer)
        if budget_monitor.exceeded:
            if budget_monitor.reason == "cancel_requested":
                final_status = ExternalOperationStatus.CANCELLED.value
                final_error = None
            else:
                final_status = ExternalOperationStatus.FAILED.value
                final_error = {
                    "message": "The realtime session reached its spending limit.",
                    "code": "spend_limit",
                    "retryable": False,
                    "details": {},
                }
        elif peer.client_rejected:
            final_status = ExternalOperationStatus.FAILED.value
            final_error = {
                "message": "A realtime message was rejected.",
                "code": "invalid_message",
                "retryable": False,
                "details": {},
            }
        elif peer.client_disconnected:
            final_status = ExternalOperationStatus.CANCELLED.value
            final_error = None
        elif relay_result.close_code in {1000, 1001}:
            final_status = ExternalOperationStatus.SUCCEEDED.value
            final_error = None
        else:
            final_status = ExternalOperationStatus.FAILED.value
    except HTTPException as exc:
        code = (
            4429
            if exc.status_code == 429
            else 4402
            if exc.status_code == 402
            else 1013
            if exc.status_code >= 500
            else 4403
        )
        await _close_local(websocket, code, "request rejected")
        final_error = {
            "message": "The realtime request was rejected.",
            "code": "request_rejected",
            "retryable": exc.status_code in {429, 503},
            "details": {},
        }
    except asyncio.CancelledError:
        final_status = ExternalOperationStatus.CANCELLED.value
        final_error = None
        raise
    except (
        ExternalConfigurationError,
        ExternalRequestMappingError,
        MappingConfigurationError,
        PricingConfigurationError,
        ProfileError,
        RequestRejectedError,
    ):
        logger.warning(
            "Realtime configuration or request failure for chute {}",
            context.chute.chute_id,
        )
        await _close_local(websocket, 1013, "temporarily unavailable")
        final_error = {
            "message": "Realtime execution is currently unavailable.",
            "code": "configuration_error",
            "retryable": False,
            "details": {},
        }
    except (ExternalTransportError, MappingExtractionError):
        logger.warning(
            "Realtime transport failure for chute {}", context.chute.chute_id
        )
        await _close_local(websocket, 1013, "temporarily unavailable")
    except Exception as exc:
        logger.warning(
            "Realtime execution failure for chute {} ({})",
            context.chute.chute_id,
            type(exc).__name__,
        )
        await _close_local(websocket, 1011, "realtime execution failed")
    finally:
        if budget_monitor is not None:
            with suppress(Exception, asyncio.CancelledError):
                await budget_monitor.stop()
        if usage_checkpoints is not None:
            with suppress(Exception, asyncio.CancelledError):
                await usage_checkpoints.stop()
        if connection is not None:
            with suppress(Exception):
                await connection.close(
                    1001
                    if final_status == ExternalOperationStatus.CANCELLED.value
                    else 1000
                )
        if operation is not None:
            usage = meter.usage if meter is not None else NormalizedUsage(requests=1)
            task = _schedule_finalizer(
                _finalize_operation(
                    operation.operation_id,
                    chute_id=context.chute.chute_id,
                    status_value=final_status,
                    usage=usage,
                    billable=connected,
                    error=final_error,
                )
            )
            with suppress(asyncio.CancelledError):
                await asyncio.shield(task)


@router.websocket("/{path:path}", name="external-realtime")
async def external_realtime_socket(websocket: WebSocket, path: str) -> None:
    del path
    await handle_external_realtime(websocket)


__all__ = [
    "RealtimeContext",
    "RealtimeUsageMeter",
    "StarletteWebSocketPeer",
    "build_websocket_profile",
    "handle_external_realtime",
    "realtime_lifespan",
    "router",
    "shutdown_realtime",
]
