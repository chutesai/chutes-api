"""Header and network boundary enforcement for outbound requests."""

from __future__ import annotations

import inspect
import ipaddress
import posixpath
import re
import string
from typing import Any, Mapping
from urllib.parse import quote, unquote, urlencode, urljoin, urlsplit, urlunsplit

import aiohttp
from aiohttp.abc import AbstractResolver

from .errors import ProfileError, RedirectRejectedError, RequestRejectedError
from .header_policy import requires_secret_backing
from .models import NetworkPolicy, RedirectProfile, SecretHeaderTemplate, SecretResolver


_HEADER_NAME_RE = re.compile(r"^[!#$%&'*+.^_`|~0-9A-Za-z-]+$")
_TEMPLATE_FIELD_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")

HOP_BY_HOP_HEADERS = frozenset(
    {
        "connection",
        "keep-alive",
        "proxy-authenticate",
        "proxy-authorization",
        "te",
        "trailer",
        "transfer-encoding",
        "upgrade",
    }
)

WEBSOCKET_NEGOTIATION_HEADERS = frozenset(
    {
        "sec-websocket-accept",
        "sec-websocket-extensions",
        "sec-websocket-key",
        "sec-websocket-protocol",
        "sec-websocket-version",
    }
)

CLIENT_SENSITIVE_HEADERS = (
    frozenset(
        {
            "authorization",
            "cookie",
            "origin",
            "referer",
            "host",
            "proxy-connection",
            "set-cookie",
            "forwarded",
            "user-agent",
            "via",
            "x-real-ip",
            "x-client-ip",
            "x-cluster-client-ip",
            "x-forwarded-for",
            "x-forwarded-host",
            "x-forwarded-port",
            "x-forwarded-proto",
            "x-forwarded-server",
            "cf-connecting-ip",
            "true-client-ip",
            "request-id",
            "traceparent",
            "tracestate",
            "x-correlation-id",
            "x-request-id",
            "x-trace-id",
        }
    )
    | WEBSOCKET_NEGOTIATION_HEADERS
)

PROFILE_FORBIDDEN_HEADERS = (
    HOP_BY_HOP_HEADERS
    | WEBSOCKET_NEGOTIATION_HEADERS
    | frozenset(
        {
            "host",
            "content-length",
            "proxy-connection",
            "forwarded",
            "set-cookie",
        }
    )
)

RESPONSE_FORBIDDEN_HEADERS = HOP_BY_HOP_HEADERS | frozenset(
    {
        "age",
        "content-disposition",
        "content-length",
        "date",
        "etag",
        "expires",
        "last-modified",
        "set-cookie",
        "proxy-connection",
        "www-authenticate",
        "proxy-authenticate",
        "location",
        "request-id",
        "retry-after",
        "server",
        "server-timing",
        "traceparent",
        "tracestate",
        "via",
        "x-correlation-id",
        "x-powered-by",
        "x-request-id",
        "x-trace-id",
    }
)

DEFAULT_RESPONSE_HEADERS = frozenset(
    {
        "cache-control",
        "content-language",
        "content-type",
    }
)

ARTIFACT_RESPONSE_HEADERS = frozenset(
    {
        "accept-ranges",
        "content-encoding",
        "content-length",
        "content-range",
        "content-type",
    }
)

CROSS_ORIGIN_REDIRECT_HEADERS = frozenset(
    {
        "accept",
        "accept-encoding",
        "if-match",
        "if-modified-since",
        "if-none-match",
        "if-range",
        "if-unmodified-since",
        "range",
    }
)

ARTIFACT_REQUEST_HEADERS = frozenset(
    {
        "if-match",
        "if-modified-since",
        "if-none-match",
        "if-range",
        "if-unmodified-since",
        "range",
    }
)

_BYTE_RANGE_RE = re.compile(r"^bytes=\d*-\d*$")
_CONTENT_TYPE_RE = re.compile(r"^[A-Za-z0-9!#$&^_.+-]+/[A-Za-z0-9!#$&^_.+-]+$")
_CONTENT_RANGE_RE = re.compile(
    r"^bytes (?:(\d+)-(\d+)|(\*))/(\d+|\*)$",
    re.I,
)
_ARTIFACT_CONTENT_ENCODINGS = frozenset({"br", "deflate", "gzip", "identity", "zstd"})


def _is_client_identity_header(name: str) -> bool:
    if name in CLIENT_SENSITIVE_HEADERS or requires_secret_backing(name):
        return True
    return _is_forwarding_header(name)


def _is_forwarding_header(name: str) -> bool:
    return name.startswith(
        (
            "forwarded-",
            "x-forwarded-",
            "x-original-forwarded-",
        )
    )


def _is_obscured_response_header(name: str) -> bool:
    """Block transport and upstream-identifying metadata, even if allowlisted."""

    if name in RESPONSE_FORBIDDEN_HEADERS:
        return True
    compact = name.lower().replace("_", "-")
    identifier = compact.replace("-", "")
    if any(
        marker in identifier for marker in ("correlationid", "requestid", "traceid")
    ):
        return True
    if "ratelimit" in identifier or "ratelimited" in identifier:
        return True
    if compact.startswith(
        (
            "ratelimit-",
            "rate-limit-",
            "x-ratelimit-",
            "x-rate-limit-",
            "x-quota-",
            "x-b3-",
        )
    ):
        return True
    return compact in {"b3", "ratelimit", "rate-limit", "x-ratelimit"}


def _validate_header_name(name: str) -> str:
    normalized = name.lower()
    if not _HEADER_NAME_RE.fullmatch(name):
        raise ProfileError(f"invalid header name: {name!r}")
    return normalized


def _validate_header_value(value: str) -> str:
    if "\r" in value or "\n" in value or "\x00" in value:
        raise RequestRejectedError("header values must not contain control delimiters")
    return value


def sanitize_client_headers(
    headers: Mapping[str, str], allowed_headers: frozenset[str]
) -> dict[str, str]:
    """Copy only explicitly permitted, non-sensitive client headers."""

    allowed = frozenset(item.lower() for item in allowed_headers)
    result: dict[str, str] = {}
    for raw_name, raw_value in headers.items():
        name = _validate_header_name(str(raw_name))
        if (
            name not in allowed
            or name in PROFILE_FORBIDDEN_HEADERS
            or _is_client_identity_header(name)
        ):
            continue
        result[name] = _validate_header_value(str(raw_value))
    return result


def sanitize_artifact_request_headers(headers: Mapping[str, str]) -> dict[str, str]:
    """Allow representation and range validators, never client identity headers."""

    result = sanitize_client_headers(headers, ARTIFACT_REQUEST_HEADERS)
    range_value = result.get("range")
    if range_value is not None:
        compact = range_value.replace(" ", "")
        if (
            len(compact) > 256
            or not _BYTE_RANGE_RE.fullmatch(compact)
            or any(item == "-" for item in compact[6:].split(","))
        ):
            raise RequestRejectedError("invalid byte range")
        result["range"] = compact
    return result


def public_artifact_content_type(value: object) -> str:
    """Normalize a mapped public media type without trusting upstream headers."""

    if isinstance(value, str):
        normalized = value.split(";", 1)[0].strip()
        if _CONTENT_TYPE_RE.fullmatch(normalized):
            return normalized.lower()
    return "application/octet-stream"


def public_artifact_disposition(content_type: str, artifact_index: int) -> str:
    """Keep active formats from executing in the authenticated API origin."""

    passive = content_type.startswith(("audio/", "video/")) or content_type in {
        "image/avif",
        "image/bmp",
        "image/gif",
        "image/jpeg",
        "image/png",
        "image/tiff",
        "image/webp",
    }
    disposition = "inline" if passive else "attachment"
    return f'{disposition}; filename="artifact-{max(0, artifact_index)}"'


def sanitize_response_headers(
    headers: Mapping[str, str],
    allowed_headers: frozenset[str],
    *,
    artifact: bool = False,
) -> dict[str, str]:
    """Return a small safe subset of upstream response headers."""

    baseline = ARTIFACT_RESPONSE_HEADERS if artifact else DEFAULT_RESPONSE_HEADERS
    # Artifact metadata is intentionally fixed: callers may add locally generated
    # metadata after the relay, but upstream values outside this set never cross it.
    allowed = (
        baseline
        if artifact
        else baseline | frozenset(item.lower() for item in allowed_headers)
    )
    result: dict[str, str] = {}
    for raw_name, raw_value in headers.items():
        try:
            name = _validate_header_name(str(raw_name))
            value = _validate_header_value(str(raw_value))
        except (ProfileError, RequestRejectedError):
            continue
        artifact_framing_header = artifact and name in {
            "content-length",
        }
        if name in allowed and (
            artifact_framing_header or not _is_obscured_response_header(name)
        ):
            if artifact:
                value = _sanitize_artifact_response_value(name, value)
                if value is None:
                    continue
            result[name] = value
    return result


def _sanitize_artifact_response_value(name: str, value: str) -> str | None:
    if name == "content-length":
        if not re.fullmatch(r"0|[1-9]\d{0,19}", value):
            return None
        return str(int(value))
    if name == "accept-ranges":
        normalized = value.strip().lower()
        return normalized if normalized in {"bytes", "none"} else None
    if name == "content-encoding":
        encodings = [item.strip().lower() for item in value.split(",")]
        if not encodings or any(
            item not in _ARTIFACT_CONTENT_ENCODINGS for item in encodings
        ):
            return None
        return ", ".join(encodings)
    if name == "content-range":
        match = _CONTENT_RANGE_RE.fullmatch(value.strip())
        if not match:
            return None
        start, end, unsatisfied, total = match.groups()
        if unsatisfied:
            if total == "*":
                return None
            return f"bytes */{int(total)}"
        assert start is not None and end is not None
        start_value = int(start)
        end_value = int(end)
        if start_value > end_value:
            return None
        if total != "*" and end_value >= int(total):
            return None
        return (
            f"bytes {start_value}-{end_value}/{total if total == '*' else int(total)}"
        )
    return value


def observe_response_headers(
    headers: Mapping[str, str], allowed_headers: frozenset[str]
) -> dict[str, str]:
    """Capture explicitly configured headers for internal control flow only.

    These values are kept separate from the public-safe response headers so rate-limit
    and retry policies can be data driven without revealing upstream identity metadata.
    """

    allowed = frozenset(item.lower() for item in allowed_headers)
    result: dict[str, str] = {}
    for raw_name, raw_value in headers.items():
        try:
            name = _validate_header_name(str(raw_name))
            value = _validate_header_value(str(raw_value))
        except (ProfileError, RequestRejectedError):
            continue
        if (
            name in allowed
            and name not in HOP_BY_HOP_HEADERS
            and name
            not in {
                "set-cookie",
                "www-authenticate",
                "proxy-authenticate",
                "location",
            }
        ):
            result[name] = value
    return result


def validate_profile_headers(headers: Mapping[str, str]) -> dict[str, str]:
    """Validate administrator-controlled static headers."""

    result: dict[str, str] = {}
    for raw_name, raw_value in headers.items():
        name = _validate_header_name(str(raw_name))
        if (
            name in PROFILE_FORBIDDEN_HEADERS
            or _is_forwarding_header(name)
            or requires_secret_backing(name)
        ):
            raise ProfileError(
                f"header must be secret-backed or cannot be configured: {name}"
            )
        result[name] = _validate_header_value(str(raw_value))
    return result


async def render_secret_headers(
    templates: tuple[SecretHeaderTemplate, ...], resolver: SecretResolver | None
) -> tuple[dict[str, str], frozenset[str]]:
    """Resolve and render secret-backed headers without storing secret values in profiles."""

    if templates and resolver is None:
        raise ProfileError("a secret resolver is required by this endpoint profile")
    result: dict[str, str] = {}
    secret_names: set[str] = set()
    formatter = string.Formatter()
    for header in templates:
        name = _validate_header_name(header.name)
        if name in PROFILE_FORBIDDEN_HEADERS or _is_forwarding_header(name):
            raise ProfileError(f"secret header is not permitted: {name}")
        fields: set[str] = set()
        for _literal, field_name, format_spec, conversion in formatter.parse(
            header.template
        ):
            if field_name is None:
                continue
            if (
                not _TEMPLATE_FIELD_RE.fullmatch(field_name)
                or format_spec
                or conversion
            ):
                raise ProfileError("secret templates support simple named fields only")
            fields.add(field_name)
        if fields != set(header.references):
            raise ProfileError(
                "secret template fields must exactly match its references"
            )
        values: dict[str, str] = {}
        for field_name in fields:
            resolved = resolver(header.references[field_name])  # type: ignore[misc]
            if inspect.isawaitable(resolved):
                resolved = await resolved
            if not isinstance(resolved, str) or not resolved:
                raise ProfileError(
                    "secret references must resolve to non-empty strings"
                )
            values[field_name] = resolved
        result[name] = _validate_header_value(header.template.format_map(values))
        secret_names.add(name)
    return result, frozenset(secret_names)


def normalize_hostname(hostname: str) -> str:
    try:
        return hostname.rstrip(".").encode("idna").decode("ascii").lower()
    except UnicodeError as exc:
        raise RequestRejectedError("invalid destination hostname") from exc


def host_matches(hostname: str, patterns: tuple[str, ...]) -> bool:
    host = normalize_hostname(hostname)
    for pattern in patterns:
        normalized = normalize_hostname(pattern)
        if normalized.startswith("*."):
            suffix = normalized[1:]
            if host.endswith(suffix) and host != normalized[2:]:
                return True
        elif host == normalized:
            return True
    return False


_IPV4_COMPATIBLE_NETWORK = ipaddress.IPv6Network("::/96")
_NAT64_WELL_KNOWN_NETWORK = ipaddress.IPv6Network("64:ff9b::/96")


def _embedded_ipv4_is_blocked(address: ipaddress.IPv6Address) -> bool:
    """Reject transition addresses that can route to a non-public IPv4 target."""

    embedded: list[ipaddress.IPv4Address] = []
    if address.ipv4_mapped is not None:
        embedded.append(address.ipv4_mapped)
    if address.sixtofour is not None:
        embedded.append(address.sixtofour)
    if address.teredo is not None:
        server, client = address.teredo
        embedded.extend((server, client))
    if address in _IPV4_COMPATIBLE_NETWORK:
        # The deprecated IPv4-compatible form (for example ::127.0.0.1) is
        # still interpreted by some network stacks and translators.
        embedded.append(ipaddress.IPv4Address(int(address) & 0xFFFFFFFF))
    if address in _NAT64_WELL_KNOWN_NETWORK:
        embedded.append(ipaddress.IPv4Address(int(address) & 0xFFFFFFFF))
    if address.packed[8:12] in {b"\x00\x00\x5e\xfe", b"\x02\x00\x5e\xfe"}:
        # ISATAP embeds its IPv4 destination in the low 32 bits.
        embedded.append(ipaddress.IPv4Address(address.packed[-4:]))
    return any(not candidate.is_global for candidate in embedded)


def _is_blocked_address(value: str) -> bool:
    try:
        address = ipaddress.ip_address(value.split("%", 1)[0])
    except ValueError as exc:
        raise RequestRejectedError(
            "destination did not resolve to an IP address"
        ) from exc
    if not address.is_global:
        return True
    return isinstance(address, ipaddress.IPv6Address) and _embedded_ipv4_is_blocked(
        address
    )


def _contains_dot_segment(value: str) -> bool:
    """Catch literal and repeatedly encoded path traversal segments."""

    decoded = value.replace("\\", "/")
    for _ in range(8):
        if any(segment in {".", ".."} for segment in decoded.split("/")):
            return True
        next_value = unquote(decoded).replace("\\", "/")
        if next_value == decoded:
            return False
        decoded = next_value
    # Excessive nested encoding is ambiguous across proxy/server decode layers.
    return True


def _path_within_prefix(path: str, prefix: str) -> bool:
    normalized_path = posixpath.normpath(path)
    normalized_prefix = posixpath.normpath(prefix or "/")
    if normalized_prefix == "/":
        return normalized_path.startswith("/")
    return normalized_path == normalized_prefix or normalized_path.startswith(
        f"{normalized_prefix.rstrip('/')}/"
    )


def validate_url(url: str, policy: NetworkPolicy) -> str:
    """Validate URL syntax, host allowlist, scheme, credentials, and IP literals."""

    parsed = urlsplit(url)
    if parsed.scheme.lower() not in policy.allowed_schemes:
        raise RequestRejectedError("destination URL scheme is not permitted")
    if not parsed.hostname:
        raise RequestRejectedError("destination URL must include a hostname")
    if parsed.username is not None or parsed.password is not None:
        raise RequestRejectedError("destination URL must not include credentials")
    if parsed.fragment:
        raise RequestRejectedError("destination URL must not include a fragment")
    hostname = normalize_hostname(parsed.hostname)
    if policy.allowed_hosts and not host_matches(hostname, policy.allowed_hosts):
        raise RequestRejectedError("destination hostname is not allowlisted")
    try:
        port = parsed.port or (443 if parsed.scheme.lower() in {"https", "wss"} else 80)
    except ValueError as exc:
        raise RequestRejectedError("destination URL has an invalid port") from exc
    default_ports = {
        443 if scheme in {"https", "wss"} else 80 for scheme in policy.allowed_schemes
    }
    allowed_ports = policy.allowed_ports or frozenset(default_ports)
    if port not in allowed_ports:
        raise RequestRejectedError("destination port is not permitted")
    try:
        ipaddress.ip_address(hostname.split("%", 1)[0])
    except ValueError:
        pass
    else:
        if not policy.allow_private_networks and _is_blocked_address(hostname):
            raise RequestRejectedError("destination IP address is not permitted")
    return urlunsplit(
        (parsed.scheme.lower(), parsed.netloc, parsed.path or "/", parsed.query, "")
    )


def origin(url: str) -> tuple[str, str, int]:
    parsed = urlsplit(url)
    if not parsed.hostname:
        raise RequestRejectedError("destination URL must include a hostname")
    default_port = 443 if parsed.scheme.lower() in {"https", "wss"} else 80
    try:
        port = parsed.port or default_port
    except ValueError as exc:
        raise RequestRejectedError("destination URL has an invalid port") from exc
    return parsed.scheme.lower(), normalize_hostname(parsed.hostname), port


def endpoint_network_policy(base_url: str, policy: NetworkPolicy) -> NetworkPolicy:
    """Restrict an endpoint to its base host unless redirect hosts were explicitly supplied."""

    parsed = urlsplit(base_url)
    if not parsed.hostname:
        raise ProfileError("base_url must include a hostname")
    hosts = policy.allowed_hosts or (normalize_hostname(parsed.hostname),)
    try:
        base_port = parsed.port or (
            443 if parsed.scheme.lower() in {"https", "wss"} else 80
        )
    except ValueError as exc:
        raise ProfileError("base_url has an invalid port") from exc
    ports = policy.allowed_ports or frozenset({base_port})
    return NetworkPolicy(
        allowed_schemes=policy.allowed_schemes,
        allowed_hosts=hosts,
        allowed_ports=ports,
        allow_private_networks=policy.allow_private_networks,
    )


def build_endpoint_url(
    base_url: str,
    path_template: str,
    path_parameters: Mapping[str, str | int],
    query: Mapping[str, Any],
    policy: NetworkPolicy,
) -> str:
    """Expand a path safely while preserving the configured endpoint origin."""

    parsed_base = urlsplit(base_url)
    if (
        parsed_base.query
        or parsed_base.fragment
        or parsed_base.username
        or parsed_base.password
    ):
        raise ProfileError("base_url must not contain query, fragment, or credentials")
    if _contains_dot_segment(parsed_base.path):
        raise ProfileError("base_url must not contain path traversal segments")
    for value in path_parameters.values():
        if _contains_dot_segment(str(value)):
            raise RequestRejectedError(
                "endpoint path parameters must not contain traversal segments"
            )
    encoded = {
        name: quote(str(value), safe="") for name, value in path_parameters.items()
    }
    try:
        expanded_path = path_template.format_map(encoded)
    except (KeyError, ValueError) as exc:
        raise RequestRejectedError(
            "missing or invalid endpoint path parameter"
        ) from exc
    if any(token in expanded_path for token in ("://", "?", "#", "\\")):
        raise RequestRejectedError("expanded endpoint path is invalid")
    if _contains_dot_segment(expanded_path):
        raise RequestRejectedError("expanded endpoint path contains traversal segments")
    base_path = parsed_base.path.rstrip("/")
    path = f"{base_path}{expanded_path}"
    if not _path_within_prefix(path, base_path or "/"):
        raise RequestRejectedError(
            "expanded endpoint path escaped its configured prefix"
        )
    pairs: list[tuple[str, Any]] = []
    for key, value in query.items():
        if value is None:
            continue
        if isinstance(value, (list, tuple)):
            pairs.extend((str(key), item) for item in value)
        else:
            pairs.append((str(key), value))
    target = urlunsplit(
        (
            parsed_base.scheme,
            parsed_base.netloc,
            path,
            urlencode(pairs, doseq=True),
            "",
        )
    )
    validate_url(target, policy)
    if origin(target) != origin(base_url):
        raise RequestRejectedError("expanded endpoint URL changed origin")
    return target


def validate_redirect(
    current_url: str,
    location: str,
    network: NetworkPolicy,
    redirects: RedirectProfile,
) -> tuple[str, bool]:
    """Validate one redirect target and return it with its cross-origin flag."""

    if not location or "\r" in location or "\n" in location:
        raise RedirectRejectedError("upstream redirect has an invalid location")
    target = validate_url(urljoin(current_url, location), network)
    cross_origin = origin(target) != origin(current_url)
    if cross_origin and not redirects.allow_cross_origin:
        raise RedirectRejectedError("cross-origin upstream redirect is not permitted")
    return target, cross_origin


def strip_headers_for_cross_origin_redirect(
    headers: Mapping[str, str],
) -> dict[str, str]:
    """Keep only representation-neutral headers when changing origin."""

    return {
        name: value
        for name, value in headers.items()
        if name in CROSS_ORIGIN_REDIRECT_HEADERS
    }


class GuardedResolver(AbstractResolver):
    """Filter the exact addresses returned to the HTTP connector."""

    def __init__(self, *, allow_private_networks: bool = False) -> None:
        self._delegate = aiohttp.resolver.DefaultResolver()
        self._allow_private_networks = allow_private_networks

    async def resolve(
        self, host: str, port: int = 0, family: int = 0
    ) -> list[dict[str, Any]]:
        records = await self._delegate.resolve(host, port, family)
        if not records:
            raise OSError("destination hostname did not resolve")
        if not self._allow_private_networks:
            for record in records:
                if _is_blocked_address(str(record["host"])):
                    raise RequestRejectedError(
                        "destination resolved to a blocked address"
                    )
        return records

    async def close(self) -> None:
        await self._delegate.close()


def create_connector(policy: NetworkPolicy) -> aiohttp.TCPConnector:
    return aiohttp.TCPConnector(
        resolver=GuardedResolver(allow_private_networks=policy.allow_private_networks),
        use_dns_cache=True,
        ttl_dns_cache=60,
    )
