"""Data-only profiles and values used by the external HTTP transport."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType
from typing import Any, Awaitable, Callable, Mapping, Sequence, TypeAlias

from .errors import ProfileError


class BodyMode(str, Enum):
    """Supported outbound request body encodings."""

    NONE = "none"
    JSON = "json"
    RAW = "raw"
    MULTIPART = "multipart"


class ResponseMode(str, Enum):
    """Supported upstream response handling modes."""

    BUFFERED = "buffered"
    SSE = "sse"
    STREAM = "stream"


class MessageDirection(str, Enum):
    """Direction of an observed duplex message."""

    INBOUND = "inbound"
    OUTBOUND = "outbound"


class WebSocketFrameType(str, Enum):
    """Frame types exposed by the transport-neutral WebSocket interface."""

    TEXT = "text"
    BINARY = "binary"
    CLOSE = "close"


def _frozen_mapping(value: Mapping[str, Any] | None) -> Mapping[str, Any]:
    return MappingProxyType(dict(value or {}))


def _positive(value: float, name: str) -> None:
    if value <= 0:
        raise ProfileError(f"{name} must be greater than zero")


@dataclass(frozen=True, slots=True)
class TimeoutProfile:
    """Connection and response deadlines, in seconds."""

    total: float = 300.0
    connect: float = 10.0
    socket_connect: float = 10.0
    socket_read: float = 300.0

    def __post_init__(self) -> None:
        for name in ("total", "connect", "socket_connect", "socket_read"):
            _positive(getattr(self, name), name)


@dataclass(frozen=True, slots=True)
class NetworkPolicy:
    """Network destinations permitted for a request."""

    allowed_schemes: frozenset[str] = frozenset({"https"})
    allowed_hosts: tuple[str, ...] = ()
    allowed_ports: frozenset[int] = frozenset()
    allow_private_networks: bool = False

    def __post_init__(self) -> None:
        schemes = frozenset(item.lower() for item in self.allowed_schemes)
        if not schemes or not schemes.issubset({"http", "https", "ws", "wss"}):
            raise ProfileError("allowed_schemes contains an unsupported URL scheme")
        hosts = tuple(item.lower().rstrip(".") for item in self.allowed_hosts)
        if any(not item or ":" in item or "/" in item for item in hosts):
            raise ProfileError(
                "allowed_hosts entries must be hostnames without ports or paths"
            )
        ports = frozenset(self.allowed_ports)
        if any(not isinstance(port, int) or port < 1 or port > 65535 for port in ports):
            raise ProfileError(
                "allowed_ports entries must be integers from 1 through 65535"
            )
        object.__setattr__(self, "allowed_schemes", schemes)
        object.__setattr__(self, "allowed_hosts", hosts)
        object.__setattr__(self, "allowed_ports", ports)


@dataclass(frozen=True, slots=True)
class RedirectProfile:
    """Redirect behavior for an endpoint or artifact request."""

    max_redirects: int = 0
    allow_cross_origin: bool = False

    def __post_init__(self) -> None:
        if self.max_redirects < 0 or self.max_redirects > 10:
            raise ProfileError("max_redirects must be between 0 and 10")


@dataclass(frozen=True, slots=True)
class SecretHeaderTemplate:
    """A header assembled from values obtained through opaque secret references."""

    name: str
    template: str
    references: Mapping[str, str]

    def __post_init__(self) -> None:
        object.__setattr__(self, "references", _frozen_mapping(self.references))


@dataclass(frozen=True, slots=True)
class EndpointProfile:
    """Declarative description of one outbound API endpoint."""

    name: str
    base_url: str
    path_template: str
    method: str = "POST"
    body_mode: BodyMode = BodyMode.JSON
    response_mode: ResponseMode = ResponseMode.BUFFERED
    allowed_request_headers: frozenset[str] = frozenset()
    static_headers: Mapping[str, str] = field(default_factory=dict)
    secret_headers: tuple[SecretHeaderTemplate, ...] = ()
    allowed_response_headers: frozenset[str] = frozenset()
    private_response_headers: frozenset[str] = frozenset()
    timeout: TimeoutProfile = field(default_factory=TimeoutProfile)
    network: NetworkPolicy = field(default_factory=NetworkPolicy)
    redirects: RedirectProfile = field(default_factory=RedirectProfile)
    max_response_bytes: int = 8 * 1024 * 1024
    max_sse_event_bytes: int = 1024 * 1024
    stream_chunk_bytes: int = 64 * 1024

    def __post_init__(self) -> None:
        method = self.method.upper()
        if not self.name:
            raise ProfileError("endpoint profile name is required")
        if method not in {"GET", "POST", "PUT", "PATCH", "DELETE", "HEAD"}:
            raise ProfileError(f"unsupported endpoint method: {method}")
        if not self.path_template.startswith("/"):
            raise ProfileError("path_template must start with '/'")
        if any(token in self.path_template for token in ("://", "?", "#", "\\")):
            raise ProfileError(
                "path_template must be a path without a query or fragment"
            )
        if any(part == ".." for part in self.path_template.split("/")):
            raise ProfileError(
                "path_template must not contain parent traversal segments"
            )
        if not self.network.allowed_schemes.issubset({"http", "https"}):
            raise ProfileError(
                "HTTP endpoint profiles support only http and https URLs"
            )
        if self.max_response_bytes <= 0:
            raise ProfileError("max_response_bytes must be greater than zero")
        if self.max_sse_event_bytes <= 0:
            raise ProfileError("max_sse_event_bytes must be greater than zero")
        if self.stream_chunk_bytes <= 0:
            raise ProfileError("stream_chunk_bytes must be greater than zero")
        object.__setattr__(self, "method", method)
        object.__setattr__(
            self,
            "allowed_request_headers",
            frozenset(item.lower() for item in self.allowed_request_headers),
        )
        object.__setattr__(
            self,
            "allowed_response_headers",
            frozenset(item.lower() for item in self.allowed_response_headers),
        )
        object.__setattr__(
            self,
            "private_response_headers",
            frozenset(item.lower() for item in self.private_response_headers),
        )
        object.__setattr__(self, "static_headers", _frozen_mapping(self.static_headers))
        object.__setattr__(self, "secret_headers", tuple(self.secret_headers))


@dataclass(frozen=True, slots=True)
class JsonBody:
    value: Any


@dataclass(frozen=True, slots=True)
class RawBody:
    value: bytes | str
    content_type: str | None = None


@dataclass(frozen=True, slots=True)
class MultipartPart:
    name: str
    value: bytes | str
    filename: str | None = None
    content_type: str | None = None

    def __post_init__(self) -> None:
        if not self.name:
            raise ProfileError("multipart field name is required")


@dataclass(frozen=True, slots=True)
class MultipartBody:
    parts: tuple[MultipartPart, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "parts", tuple(self.parts))


RequestBody: TypeAlias = JsonBody | RawBody | MultipartBody | None
QueryValue: TypeAlias = (
    str | int | float | bool | None | Sequence[str | int | float | bool]
)


@dataclass(frozen=True, slots=True)
class OutboundRequest:
    """Values supplied for one execution of an endpoint profile."""

    path_parameters: Mapping[str, str | int] = field(default_factory=dict)
    query: Mapping[str, QueryValue] = field(default_factory=dict)
    headers: Mapping[str, str] = field(default_factory=dict)
    body: RequestBody = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "path_parameters", _frozen_mapping(self.path_parameters)
        )
        object.__setattr__(self, "query", _frozen_mapping(self.query))
        object.__setattr__(self, "headers", _frozen_mapping(self.headers))


@dataclass(frozen=True, slots=True)
class SSEEvent:
    """One decoded server-sent event."""

    data: str
    event: str | None = None
    event_id: str | None = None
    retry: int | None = None
    comments: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class UsageObservation:
    """Structured response material made available to accounting hooks."""

    profile_name: str
    status_code: int
    response_headers: Mapping[str, str]
    json_value: Any | None = None
    sse_event: SSEEvent | None = None
    direction: MessageDirection | None = None


SecretResolver: TypeAlias = Callable[[str], str | Awaitable[str]]
UsageHook: TypeAlias = Callable[[UsageObservation], None | Awaitable[None]]


@dataclass(frozen=True, slots=True)
class ArtifactProfile:
    """Security and timeout policy for streaming remote artifacts."""

    network: NetworkPolicy
    static_headers: Mapping[str, str] = field(default_factory=dict)
    secret_headers: tuple[SecretHeaderTemplate, ...] = ()
    timeout: TimeoutProfile = field(default_factory=TimeoutProfile)
    redirects: RedirectProfile = field(
        default_factory=lambda: RedirectProfile(
            max_redirects=3, allow_cross_origin=False
        )
    )
    max_bytes: int = 512 * 1024 * 1024
    stream_chunk_bytes: int = 128 * 1024

    def __post_init__(self) -> None:
        if not self.network.allowed_schemes.issubset({"http", "https"}):
            raise ProfileError("artifact profiles support only http and https URLs")
        if not self.network.allowed_hosts:
            raise ProfileError(
                "artifact network policy requires an explicit host allowlist"
            )
        if self.max_bytes <= 0:
            raise ProfileError("max_bytes must be greater than zero")
        if self.stream_chunk_bytes <= 0:
            raise ProfileError("stream_chunk_bytes must be greater than zero")
        object.__setattr__(self, "static_headers", _frozen_mapping(self.static_headers))
        object.__setattr__(self, "secret_headers", tuple(self.secret_headers))


@dataclass(frozen=True, slots=True)
class WebSocketProfile:
    """Declarative handshake and resource limits for a WebSocket endpoint."""

    name: str
    base_url: str
    path_template: str
    allowed_request_headers: frozenset[str] = frozenset()
    static_headers: Mapping[str, str] = field(default_factory=dict)
    secret_headers: tuple[SecretHeaderTemplate, ...] = ()
    allowed_subprotocols: tuple[str, ...] = ()
    require_subprotocol: bool = False
    network: NetworkPolicy = field(
        default_factory=lambda: NetworkPolicy(allowed_schemes=frozenset({"wss"}))
    )
    handshake_timeout_seconds: float = 15.0
    idle_timeout_seconds: float = 60.0
    max_session_seconds: float = 3600.0
    max_message_bytes: int = 4 * 1024 * 1024
    heartbeat_seconds: float | None = 30.0

    def __post_init__(self) -> None:
        if not self.name:
            raise ProfileError("WebSocket profile name is required")
        if not self.path_template.startswith("/"):
            raise ProfileError("path_template must start with '/'")
        if any(token in self.path_template for token in ("://", "?", "#", "\\")):
            raise ProfileError(
                "path_template must be a path without a query or fragment"
            )
        if any(part == ".." for part in self.path_template.split("/")):
            raise ProfileError(
                "path_template must not contain parent traversal segments"
            )
        if not self.network.allowed_schemes.issubset({"ws", "wss"}):
            raise ProfileError("WebSocket profiles support only ws and wss URLs")
        for name in (
            "handshake_timeout_seconds",
            "idle_timeout_seconds",
            "max_session_seconds",
        ):
            _positive(getattr(self, name), name)
        if self.max_message_bytes <= 0:
            raise ProfileError("max_message_bytes must be greater than zero")
        if self.heartbeat_seconds is not None:
            _positive(self.heartbeat_seconds, "heartbeat_seconds")
        if self.require_subprotocol and not self.allowed_subprotocols:
            raise ProfileError("required subprotocols need a non-empty allowlist")
        object.__setattr__(
            self,
            "allowed_request_headers",
            frozenset(item.lower() for item in self.allowed_request_headers),
        )
        object.__setattr__(self, "static_headers", _frozen_mapping(self.static_headers))
        object.__setattr__(self, "secret_headers", tuple(self.secret_headers))
        object.__setattr__(
            self, "allowed_subprotocols", tuple(self.allowed_subprotocols)
        )


@dataclass(frozen=True, slots=True)
class WebSocketRequest:
    """Client-controlled values accepted for one server-side handshake."""

    path_parameters: Mapping[str, str | int] = field(default_factory=dict)
    query: Mapping[str, QueryValue] = field(default_factory=dict)
    headers: Mapping[str, str] = field(default_factory=dict)
    subprotocols: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "path_parameters", _frozen_mapping(self.path_parameters)
        )
        object.__setattr__(self, "query", _frozen_mapping(self.query))
        object.__setattr__(self, "headers", _frozen_mapping(self.headers))
        object.__setattr__(self, "subprotocols", tuple(self.subprotocols))


@dataclass(frozen=True, slots=True)
class WebSocketFrame:
    """Transport-neutral WebSocket frame."""

    kind: WebSocketFrameType
    data: str | bytes = b""
    close_code: int | None = None
    close_reason: str = ""

    def __post_init__(self) -> None:
        if self.kind is WebSocketFrameType.TEXT and not isinstance(self.data, str):
            raise ProfileError("text frames require string data")
        if self.kind is WebSocketFrameType.BINARY and not isinstance(self.data, bytes):
            raise ProfileError("binary frames require byte data")
