"""Compile persisted external-backend configuration into transport profiles."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping
from urllib.parse import urlsplit

from api.config import settings
from api.external_transport import (
    ArtifactProfile,
    BodyMode,
    EndpointProfile,
    NetworkPolicy,
    RedirectProfile,
    ResponseMode,
    SecretHeaderTemplate,
    TimeoutProfile,
)

from .schemas import ExternalBackendAccount, ExternalRouteConfig


class ExternalConfigurationError(ValueError):
    """Raised when a persisted route cannot be executed safely."""


def _mapping(value: object, name: str) -> Mapping[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ExternalConfigurationError(f"{name} must be an object")
    return value


def allow_insecure_http(connection: Mapping[str, Any]) -> bool:
    value = connection.get("allow_insecure_http", False)
    if not isinstance(value, bool):
        raise ExternalConfigurationError(
            "connection_config.allow_insecure_http must be a boolean"
        )
    return value and settings.external_allow_insecure_upstreams


def _string_list(value: object, name: str, *, maximum: int = 128) -> tuple[str, ...]:
    if value is None:
        return ()
    if not isinstance(value, list) or len(value) > maximum:
        raise ExternalConfigurationError(f"{name} must be an array of strings")
    result: list[str] = []
    for item in value:
        if not isinstance(item, str) or not item.strip():
            raise ExternalConfigurationError(f"{name} must be an array of strings")
        result.append(item.strip())
    return tuple(result)


def _bounded_number(
    value: object,
    default: float,
    name: str,
    *,
    minimum: float,
    maximum: float,
) -> float:
    if value is None:
        return default
    if isinstance(value, bool):
        raise ExternalConfigurationError(f"{name} must be a number")
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ExternalConfigurationError(f"{name} must be a number") from exc
    if parsed < minimum or parsed > maximum:
        raise ExternalConfigurationError(
            f"{name} must be between {minimum:g} and {maximum:g}"
        )
    return parsed


def _bounded_int(
    value: object,
    default: int,
    name: str,
    *,
    minimum: int,
    maximum: int,
) -> int:
    parsed = _bounded_number(value, default, name, minimum=minimum, maximum=maximum)
    if not parsed.is_integer():
        raise ExternalConfigurationError(f"{name} must be an integer")
    return int(parsed)


def _port_list(value: object, name: str) -> tuple[int, ...]:
    if value is None:
        return ()
    if not isinstance(value, list) or len(value) > 64:
        raise ExternalConfigurationError(f"{name} must be an array of ports")
    result: list[int] = []
    for candidate in value:
        if (
            isinstance(candidate, bool)
            or not isinstance(candidate, int)
            or candidate < 1
            or candidate > 65535
        ):
            raise ExternalConfigurationError(
                f"{name} entries must be integers from 1 through 65535"
            )
        result.append(candidate)
    return tuple(result)


def _http_origin(
    value: str,
    name: str,
    *,
    require_origin_only: bool,
) -> tuple[str, str, int]:
    parsed = urlsplit(value)
    scheme = parsed.scheme.lower()
    if scheme not in {"http", "https"} or not parsed.hostname:
        raise ExternalConfigurationError(f"{name} must be an absolute HTTP origin")
    if parsed.username is not None or parsed.password is not None:
        raise ExternalConfigurationError(f"{name} must not contain credentials")
    if require_origin_only and (
        parsed.path not in {"", "/"} or parsed.query or parsed.fragment
    ):
        raise ExternalConfigurationError(
            f"{name} must contain only a scheme, hostname, and optional port"
        )
    try:
        port = parsed.port or (443 if scheme == "https" else 80)
    except ValueError as exc:
        raise ExternalConfigurationError(f"{name} has an invalid port") from exc
    host = parsed.hostname.lower().rstrip(".")
    if not host or "*" in host or ":" in host or "/" in host:
        raise ExternalConfigurationError(f"{name} has an invalid hostname")
    return scheme, host, port


def _host_is_allowed(host: str, allowed_hosts: set[str]) -> bool:
    return any(
        host == allowed
        or (
            allowed.startswith("*.")
            and host.endswith(allowed[1:])
            and host != allowed[2:]
        )
        for allowed in allowed_hosts
    )


def select_route(
    routes: list[dict[str, Any]], selected_cord: Mapping[str, Any]
) -> ExternalRouteConfig:
    """Resolve one route by the Cord's stable internal path."""

    cord_path = selected_cord.get("path")
    matches = [
        ExternalRouteConfig.model_validate(item)
        for item in routes
        if item.get("cord_path") == cord_path
    ]
    if len(matches) != 1:
        raise ExternalConfigurationError("external Cord route is not configured")
    return matches[0]


def _base_url(account: ExternalBackendAccount, route: ExternalRouteConfig) -> str:
    return str(route.base_url or account.base_url).rstrip("/")


def _network_policy(
    account: ExternalBackendAccount,
    route: ExternalRouteConfig,
    base_url: str,
    endpoint: Mapping[str, Any],
) -> NetworkPolicy:
    connection = _mapping(account.connection_config, "connection_config")
    network = _mapping(connection.get("network"), "connection_config.network")
    parsed = urlsplit(base_url)
    if not parsed.hostname:
        raise ExternalConfigurationError("external base URL has no hostname")

    allowed_hosts = {parsed.hostname.lower().rstrip(".")}
    for candidate in _string_list(
        network.get("allowed_hosts"), "connection_config.network.allowed_hosts"
    ):
        allowed_hosts.add(candidate.lower().rstrip("."))
    endpoint_hosts = _string_list(
        endpoint.get("allowed_hosts"), "endpoint.allowed_hosts"
    )
    allowed_hosts.update(item.lower().rstrip(".") for item in endpoint_hosts)

    scheme = parsed.scheme.lower()
    if scheme != "https" and not allow_insecure_http(connection):
        raise ExternalConfigurationError("external endpoints must use HTTPS")
    schemes = frozenset({scheme})
    port = parsed.port or (443 if scheme == "https" else 80)
    additional_ports = network.get("allowed_ports", [])
    if not isinstance(additional_ports, list):
        raise ExternalConfigurationError(
            "connection_config.network.allowed_ports must be an array"
        )
    ports = {port}
    for candidate in additional_ports:
        if isinstance(candidate, bool) or not isinstance(candidate, int):
            raise ExternalConfigurationError("allowed ports must be integers")
        ports.add(candidate)
    return NetworkPolicy(
        allowed_schemes=schemes,
        allowed_hosts=tuple(sorted(allowed_hosts)),
        allowed_ports=frozenset(ports),
        # This remains false even for privileged configuration: DNS is resolved and
        # pinned by the transport so provider routes cannot become an SSRF primitive.
        allow_private_networks=False,
    )


def _timeouts(
    account: ExternalBackendAccount, endpoint: Mapping[str, Any]
) -> TimeoutProfile:
    connection = _mapping(account.connection_config, "connection_config")
    defaults = _mapping(connection.get("timeouts"), "connection_config.timeouts")
    overrides = _mapping(endpoint.get("timeouts"), "endpoint.timeouts")

    def value(name: str, default: float, maximum: float) -> float:
        configured = overrides.get(name, defaults.get(name))
        return _bounded_number(
            configured,
            default,
            f"timeout.{name}",
            minimum=0.1,
            maximum=maximum,
        )

    return TimeoutProfile(
        total=value("total", 300.0, 86400.0),
        connect=value("connect", 10.0, 120.0),
        socket_connect=value("socket_connect", 10.0, 120.0),
        socket_read=value("socket_read", 300.0, 86400.0),
    )


def _secret_headers(
    account: ExternalBackendAccount,
) -> tuple[SecretHeaderTemplate, ...]:
    credential_references = dict(account.credential_references or {})
    result: list[SecretHeaderTemplate] = []
    for raw_template in account.auth_header_templates or []:
        references = {
            field: credential_references[credential_name]
            for field, credential_name in raw_template["references"].items()
        }
        result.append(
            SecretHeaderTemplate(
                name=raw_template["name"],
                template=raw_template["template"],
                references=references,
            )
        )
    return tuple(result)


def build_endpoint_profile(
    account: ExternalBackendAccount,
    route: ExternalRouteConfig,
    *,
    endpoint: Mapping[str, Any] | None = None,
    name_suffix: str = "invoke",
) -> EndpointProfile:
    """Compile a route or one of its task sub-endpoints into an HTTP profile."""

    endpoint = _mapping(endpoint, "endpoint")
    request_config = _mapping(route.request_config, "request_config")
    response_config = _mapping(route.response_config, "response_config")
    base_url = str(endpoint.get("base_url") or _base_url(account, route)).rstrip("/")
    body_mode_value = endpoint.get("body_mode", request_config.get("body_mode"))
    if body_mode_value is None:
        body_mode_value = "none" if route.method in {"GET", "HEAD"} else "json"
    response_mode_value = endpoint.get(
        "response_mode",
        response_config.get(
            "mode", "sse" if route.operation_mode.value == "stream" else "buffered"
        ),
    )
    try:
        body_mode = BodyMode(str(body_mode_value).lower())
        response_mode = ResponseMode(str(response_mode_value).lower())
    except ValueError as exc:
        raise ExternalConfigurationError("unsupported body or response mode") from exc

    static_headers = dict(
        _mapping(request_config.get("static_headers"), "static_headers")
    )
    static_headers.update(
        _mapping(endpoint.get("static_headers"), "endpoint.static_headers")
    )
    allowed_request_headers = _string_list(
        endpoint.get(
            "allowed_request_headers", request_config.get("allowed_request_headers")
        ),
        "allowed_request_headers",
    )
    allowed_response_headers = _string_list(
        endpoint.get(
            "allowed_response_headers", response_config.get("allowed_headers")
        ),
        "allowed_response_headers",
    )
    operation_config = _mapping(route.operation_config, "operation_config")
    retry_config = _mapping(operation_config.get("retry"), "operation_config.retry")
    control_headers = _string_list(
        endpoint.get(
            "private_response_headers",
            retry_config.get("retry_after_headers", ["retry-after"]),
        ),
        "private_response_headers",
        maximum=16,
    )

    redirect_config = _mapping(
        endpoint.get("redirects", response_config.get("redirects")), "redirects"
    )
    redirects = RedirectProfile(
        max_redirects=_bounded_int(
            redirect_config.get("max_redirects"),
            0,
            "redirects.max_redirects",
            minimum=0,
            maximum=10,
        ),
        allow_cross_origin=bool(redirect_config.get("allow_cross_origin", False)),
    )

    method = str(endpoint.get("method") or route.method).upper()
    path_template = str(endpoint.get("path_template") or route.path_template)
    return EndpointProfile(
        name=f"{route.protocol}.{name_suffix}",
        base_url=base_url,
        path_template=path_template,
        method=method,
        body_mode=body_mode,
        response_mode=response_mode,
        allowed_request_headers=frozenset(allowed_request_headers),
        static_headers=static_headers,
        secret_headers=_secret_headers(account),
        allowed_response_headers=frozenset(allowed_response_headers),
        private_response_headers=frozenset(control_headers),
        timeout=_timeouts(account, endpoint),
        network=_network_policy(account, route, base_url, endpoint),
        redirects=redirects,
        max_response_bytes=_bounded_int(
            endpoint.get(
                "max_response_bytes", response_config.get("max_response_bytes")
            ),
            8 * 1024 * 1024,
            "max_response_bytes",
            minimum=1,
            maximum=128 * 1024 * 1024,
        ),
        max_sse_event_bytes=_bounded_int(
            endpoint.get(
                "max_sse_event_bytes", response_config.get("max_sse_event_bytes")
            ),
            1024 * 1024,
            "max_sse_event_bytes",
            minimum=1,
            maximum=16 * 1024 * 1024,
        ),
        stream_chunk_bytes=_bounded_int(
            endpoint.get(
                "stream_chunk_bytes", response_config.get("stream_chunk_bytes")
            ),
            64 * 1024,
            "stream_chunk_bytes",
            minimum=1024,
            maximum=1024 * 1024,
        ),
    )


@dataclass(frozen=True, slots=True)
class RetryPolicy:
    """Bounded retry policy for responses known not to have been accepted."""

    max_attempts: int = 1
    base_delay_seconds: float = 0.5
    maximum_delay_seconds: float = 30.0
    retry_statuses: frozenset[int] = frozenset({429, 502, 503, 504})
    retry_after_headers: tuple[str, ...] = ("retry-after",)
    retry_non_idempotent: bool = False


def retry_policy(route: ExternalRouteConfig) -> RetryPolicy:
    config = _mapping(route.operation_config, "operation_config")
    raw = _mapping(config.get("retry"), "operation_config.retry")
    statuses = raw.get("statuses", [429, 502, 503, 504])
    if not isinstance(statuses, list) or any(
        isinstance(item, bool) or not isinstance(item, int) or item < 100 or item > 599
        for item in statuses
    ):
        raise ExternalConfigurationError("retry statuses must be HTTP status codes")
    return RetryPolicy(
        max_attempts=_bounded_int(
            raw.get("max_attempts"), 1, "retry.max_attempts", minimum=1, maximum=8
        ),
        base_delay_seconds=_bounded_number(
            raw.get("base_delay_seconds"),
            0.5,
            "retry.base_delay_seconds",
            minimum=0.0,
            maximum=60.0,
        ),
        maximum_delay_seconds=_bounded_number(
            raw.get("maximum_delay_seconds"),
            30.0,
            "retry.maximum_delay_seconds",
            minimum=0.0,
            maximum=300.0,
        ),
        retry_statuses=frozenset(statuses),
        retry_after_headers=_string_list(
            raw.get("retry_after_headers", ["retry-after"]),
            "retry.retry_after_headers",
            maximum=16,
        ),
        retry_non_idempotent=bool(raw.get("retry_non_idempotent", False)),
    )


def build_artifact_profile(
    account: ExternalBackendAccount,
    route: ExternalRouteConfig,
    reference: str,
) -> ArtifactProfile:
    """Compile the allowlist used to relay one remote result without storing it."""

    operation = _mapping(route.operation_config, "operation_config")
    artifact = _mapping(operation.get("artifact"), "operation_config.artifact")
    connection = _mapping(account.connection_config, "connection_config")
    base_origin = _http_origin(
        _base_url(account, route),
        "artifact account origin",
        require_origin_only=False,
    )
    reference_origin = _http_origin(
        reference,
        "artifact reference",
        require_origin_only=False,
    )
    _, base_host, base_port = base_origin
    scheme, reference_host, reference_port = reference_origin
    allowed_hosts = {base_host}
    allowed_hosts.update(
        host.lower().rstrip(".")
        for host in _string_list(
            artifact.get("allowed_hosts"), "operation_config.artifact.allowed_hosts"
        )
    )
    if not _host_is_allowed(reference_host, allowed_hosts):
        raise ExternalConfigurationError("artifact hostname is not allowlisted")
    if scheme != "https" and not allow_insecure_http(connection):
        raise ExternalConfigurationError("artifact references must use HTTPS")
    allowed_ports = {
        base_port,
        *_port_list(
            artifact.get("allowed_ports"),
            "operation_config.artifact.allowed_ports",
        ),
    }
    if reference_port not in allowed_ports:
        raise ExternalConfigurationError("artifact port is not allowlisted")
    redirect_config = _mapping(artifact.get("redirects"), "artifact.redirects")
    authenticated = bool(artifact.get("authenticated", False))
    auth_origins = {base_origin}
    for index, configured_origin in enumerate(
        _string_list(
            artifact.get("auth_allowed_origins"),
            "operation_config.artifact.auth_allowed_origins",
            maximum=64,
        )
    ):
        auth_origin = _http_origin(
            configured_origin,
            f"operation_config.artifact.auth_allowed_origins[{index}]",
            require_origin_only=True,
        )
        auth_scheme, auth_host, auth_port = auth_origin
        if auth_scheme != "https" and not allow_insecure_http(connection):
            raise ExternalConfigurationError(
                "artifact authentication origins must use HTTPS"
            )
        if not _host_is_allowed(auth_host, allowed_hosts):
            raise ExternalConfigurationError(
                "artifact authentication origin hostname is not allowlisted"
            )
        if auth_port not in allowed_ports:
            raise ExternalConfigurationError(
                "artifact authentication origin port is not allowlisted"
            )
        auth_origins.add(auth_origin)
    if authenticated:
        if reference_origin not in auth_origins:
            raise ExternalConfigurationError(
                "artifact authentication is not permitted for this origin"
            )
    return ArtifactProfile(
        network=NetworkPolicy(
            allowed_schemes=frozenset({scheme}),
            allowed_hosts=tuple(sorted(allowed_hosts)),
            allowed_ports=frozenset(allowed_ports),
            allow_private_networks=False,
        ),
        static_headers=_mapping(
            artifact.get("static_headers"), "artifact.static_headers"
        ),
        secret_headers=_secret_headers(account) if authenticated else (),
        timeout=_timeouts(account, artifact),
        redirects=RedirectProfile(
            max_redirects=_bounded_int(
                redirect_config.get("max_redirects"),
                3,
                "artifact.redirects.max_redirects",
                minimum=0,
                maximum=10,
            ),
            allow_cross_origin=bool(redirect_config.get("allow_cross_origin", False)),
        ),
        max_bytes=_bounded_int(
            artifact.get("max_bytes"),
            512 * 1024 * 1024,
            "artifact.max_bytes",
            minimum=1024,
            maximum=10 * 1024 * 1024 * 1024,
        ),
        stream_chunk_bytes=_bounded_int(
            artifact.get("stream_chunk_bytes"),
            128 * 1024,
            "artifact.stream_chunk_bytes",
            minimum=1024,
            maximum=1024 * 1024,
        ),
    )
