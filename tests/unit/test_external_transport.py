"""Focused tests for the isolated external transport boundary."""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from urllib.parse import urlsplit

import pytest
from aiohttp import WSMsgType, web
from aiohttp.test_utils import TestServer

from api.external_transport import (
    ArtifactProfile,
    ArtifactRelay,
    BodyMode,
    BufferedResponse,
    EndpointProfile,
    ExternalExecutor,
    JsonBody,
    MessageDirection,
    MultipartBody,
    MultipartPart,
    NetworkPolicy,
    OutboundRequest,
    ProfileError,
    RawBody,
    RedirectProfile,
    RedirectRejectedError,
    RequestRejectedError,
    ResponseMode,
    ResponseTooLargeError,
    SecretHeaderTemplate,
    StreamingResponse,
    TimeoutProfile,
    UpstreamTimeoutError,
    WebSocketFrame,
    WebSocketFrameType,
    WebSocketLimitError,
    WebSocketProfile,
    WebSocketRelay,
    WebSocketRequest,
)
from api.external_transport.security import (
    build_endpoint_url,
    observe_response_headers,
    public_artifact_content_type,
    public_artifact_disposition,
    sanitize_client_headers,
    sanitize_response_headers,
    validate_profile_headers,
    validate_redirect,
    validate_url,
)


@pytest.fixture
def anyio_backend():
    return "asyncio"


@asynccontextmanager
async def running_app(app: web.Application):
    server = TestServer(app)
    await server.start_server()
    try:
        yield str(server.make_url("/")).rstrip("/")
    finally:
        await server.close()


def local_network(
    *,
    websocket: bool = False,
    hosts: tuple[str, ...] = (),
    ports: frozenset[int] = frozenset(),
) -> NetworkPolicy:
    return NetworkPolicy(
        allowed_schemes=frozenset({"ws" if websocket else "http"}),
        allowed_hosts=hosts,
        allowed_ports=ports,
        allow_private_networks=True,
    )


def test_client_identity_headers_are_removed_even_if_allowlisted():
    headers = sanitize_client_headers(
        {
            "Authorization": "client-secret",
            "Cookie": "session=value",
            "Forwarded": "for=127.0.0.1",
            "User-Agent": "client-identity/1.0",
            "X-API-Key": "client-key",
            "X-Forwarded-For": "127.0.0.1",
            "Sec-WebSocket-Key": "client-key",
            "Content-Length": "999",
            "X-Trace": "trace-value",
            "X-Dropped": "drop-value",
        },
        frozenset(
            {
                "authorization",
                "cookie",
                "forwarded",
                "user-agent",
                "x-api-key",
                "x-forwarded-for",
                "sec-websocket-key",
                "content-length",
                "x-trace",
            }
        ),
    )
    assert headers == {"x-trace": "trace-value"}


def test_public_artifact_content_type_never_uses_unmapped_upstream_metadata():
    assert public_artifact_content_type("video/MP4; upstream=private") == "video/mp4"
    assert public_artifact_content_type(None) == "application/octet-stream"
    assert (
        public_artifact_content_type("text/plain\r\nX-Private: leaked")
        == "application/octet-stream"
    )


def test_active_artifact_types_are_forced_to_download():
    assert public_artifact_disposition("text/html", 3) == (
        'attachment; filename="artifact-3"'
    )
    assert public_artifact_disposition("image/svg+xml", 0).startswith("attachment;")
    assert public_artifact_disposition("video/mp4", 2).startswith("inline;")


def test_artifact_response_protocol_headers_are_value_sanitized():
    headers = sanitize_response_headers(
        {
            "Accept-Ranges": "PrivateService",
            "Content-Encoding": "private-codec",
            "Content-Length": "not-a-number",
            "Content-Range": "private range metadata",
            "Content-Type": "application/vnd.private",
        },
        frozenset(),
        artifact=True,
    )
    assert headers == {"content-type": "application/vnd.private"}

    assert sanitize_response_headers(
        {
            "Accept-Ranges": "BYTES",
            "Content-Encoding": "GZIP",
            "Content-Length": "0010",
            "Content-Range": "bytes 2-5/10",
        },
        frozenset(),
        artifact=True,
    ) == {
        "accept-ranges": "bytes",
        "content-encoding": "gzip",
        "content-range": "bytes 2-5/10",
    }


def test_sensitive_static_headers_require_secret_backing_and_metadata_is_obscured():
    for name in (
        "Authorization",
        "Cookie",
        "ApiKey",
        "X-API-Key",
        "X-Auth-Token",
    ):
        with pytest.raises(ProfileError):
            validate_profile_headers({name: "static-value"})

    assert validate_profile_headers({"User-Agent": "relay/1"}) == {
        "user-agent": "relay/1"
    }
    headers = sanitize_response_headers(
        {
            "Content-Type": "application/json",
            "Date": "Sat, 01 Jan 2000 00:00:00 GMT",
            "Server": "hidden",
            "Via": "hidden",
            "X-Powered-By": "hidden",
            "Set-Cookie": "hidden=value",
            "WWW-Authenticate": "hidden",
            "Location": "https://hidden.example",
            "X-Request-ID": "hidden",
            "X-Service-RequestId": "hidden",
            "X-RateLimit-Remaining": "4",
            "X-Public-Result": "visible",
        },
        frozenset(
            {
                "date",
                "server",
                "via",
                "x-powered-by",
                "set-cookie",
                "www-authenticate",
                "location",
                "x-request-id",
                "x-service-requestid",
                "x-ratelimit-remaining",
                "x-public-result",
            }
        ),
    )
    assert headers == {
        "content-type": "application/json",
        "x-public-result": "visible",
    }
    assert observe_response_headers(
        {
            "Retry-After": "12",
            "X-Capacity-Reset": "34",
            "Set-Cookie": "never=observed",
        },
        frozenset({"retry-after", "x-capacity-reset", "set-cookie"}),
    ) == {"retry-after": "12", "x-capacity-reset": "34"}


def test_url_builder_encodes_path_values_and_rejects_private_destinations():
    policy = NetworkPolicy(
        allowed_schemes=frozenset({"https"}),
        allowed_hosts=("service.example",),
    )
    url = build_endpoint_url(
        "https://service.example/api",
        "/tasks/{task_id}",
        {"task_id": "part/with space"},
        {"tag": ["a", "b"]},
        policy,
    )
    assert url == "https://service.example/api/tasks/part%2Fwith%20space?tag=a&tag=b"
    with pytest.raises(RequestRejectedError):
        validate_url(
            "http://127.0.0.1/resource",
            NetworkPolicy(allowed_schemes=frozenset({"http"})),
        )
    with pytest.raises(RequestRejectedError):
        validate_url("https://service.example:8443/resource", policy)
    redirect_network = NetworkPolicy(
        allowed_schemes=frozenset({"https"}),
        allowed_hosts=("service.example", "other.example"),
    )
    with pytest.raises(RedirectRejectedError):
        validate_redirect(
            "https://service.example/start",
            "https://other.example/result",
            redirect_network,
            RedirectProfile(max_redirects=1, allow_cross_origin=False),
        )


@pytest.mark.parametrize(
    "path_value",
    [
        ".",
        "..",
        "../admin",
        r"..\admin",
        "%2e%2e",
        "%252e%252e",
        "%2e%2e%2fadmin",
    ],
)
def test_url_builder_rejects_encoded_path_parameter_traversal(path_value):
    policy = NetworkPolicy(
        allowed_schemes=frozenset({"https"}),
        allowed_hosts=("service.example",),
    )

    with pytest.raises(RequestRejectedError, match="traversal"):
        build_endpoint_url(
            "https://service.example/api",
            "/tasks/{task_id}/result",
            {"task_id": path_value},
            {},
            policy,
        )


def test_url_builder_rejects_traversal_in_configured_base_or_expanded_path():
    policy = NetworkPolicy(
        allowed_schemes=frozenset({"https"}),
        allowed_hosts=("service.example",),
    )

    with pytest.raises(ProfileError, match="traversal"):
        build_endpoint_url(
            "https://service.example/api/../private",
            "/tasks",
            {},
            {},
            policy,
        )
    with pytest.raises(RequestRejectedError, match="traversal"):
        build_endpoint_url(
            "https://service.example/api",
            "/../private",
            {},
            {},
            policy,
        )


@pytest.mark.parametrize(
    "address",
    [
        "::127.0.0.1",
        "64:ff9b::7f00:1",
        "2002:7f00:1::",
        "::ffff:127.0.0.1",
    ],
)
def test_url_validation_rejects_ipv6_addresses_embedding_private_ipv4(address):
    with pytest.raises(RequestRejectedError, match="not permitted"):
        validate_url(
            f"https://[{address}]/resource",
            NetworkPolicy(allowed_schemes=frozenset({"https"})),
        )


def test_url_validation_allows_nat64_address_embedding_public_ipv4():
    assert (
        validate_url(
            "https://[64:ff9b::808:808]/resource",
            NetworkPolicy(allowed_schemes=frozenset({"https"})),
        )
        == "https://[64:ff9b::808:808]/resource"
    )


@pytest.mark.anyio
async def test_buffered_json_secret_headers_sanitization_and_usage_hook():
    seen = {}

    async def handler(request: web.Request) -> web.Response:
        seen["headers"] = dict(request.headers)
        seen["json"] = await request.json()
        seen["item"] = request.match_info["item"]
        return web.json_response(
            {"result": "ok", "usage": {"units": 2}},
            headers={
                "Set-Cookie": "upstream=value",
                "X-Request-ID": "request-1",
                "X-Public-Result": "visible",
                "X-Control-Wait": "7",
                "X-Unsafe": "not-forwarded",
            },
        )

    app = web.Application()
    app.router.add_post("/items/{item}", handler)
    observations = []

    async def resolve_secret(reference: str) -> str:
        assert reference == "credential-ref"
        return "server-secret"

    async with running_app(app) as base_url:
        profile = EndpointProfile(
            name="create-item",
            base_url=base_url,
            path_template="/items/{item}",
            body_mode=BodyMode.JSON,
            allowed_request_headers=frozenset(
                {
                    "authorization",
                    "cookie",
                    "forwarded",
                    "user-agent",
                    "x-forwarded-for",
                    "x-trace",
                }
            ),
            allowed_response_headers=frozenset(
                {"x-public-result", "x-request-id", "date"}
            ),
            private_response_headers=frozenset({"x-control-wait"}),
            static_headers={"X-Static": "static-value"},
            secret_headers=(
                SecretHeaderTemplate(
                    name="Authorization",
                    template="Bearer {credential}",
                    references={"credential": "credential-ref"},
                ),
            ),
            network=local_network(),
        )
        result = await ExternalExecutor(
            secret_resolver=resolve_secret,
            usage_hooks=(observations.append,),
        ).execute(
            profile,
            OutboundRequest(
                path_parameters={"item": "item-1"},
                headers={
                    "Authorization": "client-secret",
                    "Cookie": "client=value",
                    "Forwarded": "for=127.0.0.1",
                    "User-Agent": "client-identity/1.0",
                    "X-Forwarded-For": "127.0.0.1",
                    "X-Trace": "trace-value",
                },
                body=JsonBody({"name": "sample"}),
            ),
        )

    assert result.status_code == 200
    assert result.json()["result"] == "ok"
    assert result.headers["content-type"] == "application/json; charset=utf-8"
    assert result.headers["x-public-result"] == "visible"
    assert "x-request-id" not in result.headers
    assert "date" not in result.headers
    assert "set-cookie" not in result.headers
    assert "x-unsafe" not in result.headers
    assert "x-control-wait" not in result.headers
    assert result.private_headers["x-control-wait"] == "7"
    lowered = {name.lower(): value for name, value in seen["headers"].items()}
    assert lowered["authorization"] == "Bearer server-secret"
    assert lowered["x-static"] == "static-value"
    assert lowered["x-trace"] == "trace-value"
    assert "cookie" not in lowered
    assert "forwarded" not in lowered
    assert "x-forwarded-for" not in lowered
    assert "user-agent" not in lowered
    assert seen["json"] == {"name": "sample"}
    assert seen["item"] == "item-1"
    assert observations[0].json_value["usage"] == {"units": 2}
    assert observations[0].response_headers["x-control-wait"] == "7"


@pytest.mark.anyio
async def test_raw_and_multipart_request_modes():
    seen = {}

    async def raw_handler(request: web.Request) -> web.Response:
        seen["raw"] = await request.read()
        seen["raw_type"] = request.headers["Content-Type"]
        return web.Response(body=b"raw-ok")

    async def multipart_handler(request: web.Request) -> web.Response:
        reader = await request.multipart()
        field = await reader.next()
        seen["field_name"] = field.name
        seen["filename"] = field.filename
        seen["field_value"] = await field.read()
        return web.Response(body=b"multipart-ok")

    app = web.Application()
    app.router.add_put("/raw", raw_handler)
    app.router.add_post("/multipart", multipart_handler)

    async with running_app(app) as base_url:
        executor = ExternalExecutor()
        raw = await executor.execute(
            EndpointProfile(
                name="raw-upload",
                base_url=base_url,
                path_template="/raw",
                method="PUT",
                body_mode=BodyMode.RAW,
                network=local_network(),
            ),
            OutboundRequest(body=RawBody(b"raw-value", "application/custom")),
        )
        multipart = await executor.execute(
            EndpointProfile(
                name="multipart-upload",
                base_url=base_url,
                path_template="/multipart",
                body_mode=BodyMode.MULTIPART,
                network=local_network(),
            ),
            OutboundRequest(
                body=MultipartBody(
                    (
                        MultipartPart(
                            name="asset",
                            value=b"asset-value",
                            filename="asset.bin",
                            content_type="application/octet-stream",
                        ),
                    )
                )
            ),
        )

    assert raw.body == b"raw-ok"
    assert multipart.body == b"multipart-ok"
    assert seen == {
        "raw": b"raw-value",
        "raw_type": "application/custom",
        "field_name": "asset",
        "filename": "asset.bin",
        "field_value": b"asset-value",
    }


@pytest.mark.anyio
async def test_sse_stream_is_relayed_and_observed_incrementally():
    async def handler(request: web.Request) -> web.StreamResponse:
        response = web.StreamResponse(
            headers={
                "Content-Type": "text/event-stream",
                "Set-Cookie": "upstream=value",
                "X-Request-ID": "stream-1",
            }
        )
        await response.prepare(request)
        await response.write(b'data: {"usage":')
        await response.write(b' {"units": 3}}\n\n')
        await response.write(b"data: [DONE]\n\n")
        await response.write_eof()
        return response

    app = web.Application()
    app.router.add_get("/events", handler)
    observations = []
    async with running_app(app) as base_url:
        result = await ExternalExecutor(usage_hooks=(observations.append,)).execute(
            EndpointProfile(
                name="event-stream",
                base_url=base_url,
                path_template="/events",
                method="GET",
                body_mode=BodyMode.NONE,
                response_mode=ResponseMode.SSE,
                network=local_network(),
                stream_chunk_bytes=7,
            ),
            OutboundRequest(),
        )
        assert isinstance(result, StreamingResponse)
        content = b"".join([chunk async for chunk in result.iter_bytes()])

    assert content == b'data: {"usage": {"units": 3}}\n\ndata: [DONE]\n\n'
    assert "x-request-id" not in result.headers
    assert "set-cookie" not in result.headers
    assert observations[0].json_value == {"usage": {"units": 3}}
    assert observations[0].sse_event.data == '{"usage": {"units": 3}}'


@pytest.mark.anyio
async def test_raw_stream_relay_is_sanitized_chunked_and_observed_once():
    async def handler(request: web.Request) -> web.StreamResponse:
        response = web.StreamResponse(
            status=201,
            headers={
                "Content-Type": "audio/mpeg",
                "Set-Cookie": "upstream=value",
                "Server": "upstream-identity",
                "X-Request-ID": "stream-1",
                "X-Public-Result": "visible",
                "X-Control-Wait": "7",
            },
        )
        await response.prepare(request)
        await response.write(b"abcde")
        await response.write(b"fghij")
        await response.write_eof()
        return response

    app = web.Application()
    app.router.add_get("/raw-stream", handler)
    observations = []
    async with running_app(app) as base_url:
        result = await ExternalExecutor(usage_hooks=(observations.append,)).execute(
            EndpointProfile(
                name="raw-stream",
                base_url=base_url,
                path_template="/raw-stream",
                method="GET",
                body_mode=BodyMode.NONE,
                response_mode=ResponseMode.STREAM,
                allowed_response_headers=frozenset(
                    {
                        "server",
                        "set-cookie",
                        "x-request-id",
                        "x-public-result",
                    }
                ),
                private_response_headers=frozenset({"x-control-wait"}),
                network=local_network(),
                max_response_bytes=10,
                stream_chunk_bytes=3,
            ),
            OutboundRequest(),
        )
        assert isinstance(result, StreamingResponse)
        assert result.response_mode is ResponseMode.STREAM
        raw_sse_iterator = result.iter_sse()
        with pytest.raises(RuntimeError, match="only available for SSE"):
            await anext(raw_sse_iterator)
        chunks = [chunk async for chunk in result.iter_bytes()]

    assert b"".join(chunks) == b"abcdefghij"
    assert all(0 < len(chunk) <= 3 for chunk in chunks)
    assert result.headers == {
        "content-type": "audio/mpeg",
        "x-public-result": "visible",
    }
    assert len(observations) == 1
    assert observations[0].status_code == 201
    assert observations[0].response_headers["x-control-wait"] == "7"
    assert observations[0].json_value is None
    assert observations[0].sse_event is None
    with pytest.raises(RuntimeError, match="only be consumed once"):
        await anext(result.iter_bytes())


@pytest.mark.anyio
async def test_raw_stream_errors_stay_buffered_and_success_body_is_bounded():
    async def handler(request: web.Request) -> web.Response:
        if request.match_info["kind"] == "error":
            return web.Response(
                status=429,
                body=b"retry later",
                content_type="text/plain",
                headers={"X-Request-ID": "error-1", "Set-Cookie": "hidden=value"},
            )
        if request.match_info["kind"] == "empty":
            return web.Response(status=204)
        return web.Response(body=b"12345", content_type="application/octet-stream")

    app = web.Application()
    app.router.add_get("/{kind}", handler)
    async with running_app(app) as base_url:
        executor = ExternalExecutor()
        error = await executor.execute(
            EndpointProfile(
                name="raw-stream-error",
                base_url=base_url,
                path_template="/error",
                method="GET",
                body_mode=BodyMode.NONE,
                response_mode=ResponseMode.STREAM,
                allowed_response_headers=frozenset({"x-request-id", "set-cookie"}),
                network=local_network(),
            ),
            OutboundRequest(),
        )
        empty = await executor.execute(
            EndpointProfile(
                name="empty-raw-stream",
                base_url=base_url,
                path_template="/empty",
                method="GET",
                body_mode=BodyMode.NONE,
                response_mode=ResponseMode.STREAM,
                network=local_network(),
            ),
            OutboundRequest(),
        )
        limited = await executor.execute(
            EndpointProfile(
                name="bounded-raw-stream",
                base_url=base_url,
                path_template="/success",
                method="GET",
                body_mode=BodyMode.NONE,
                response_mode=ResponseMode.STREAM,
                network=local_network(),
                max_response_bytes=4,
                stream_chunk_bytes=2,
            ),
            OutboundRequest(),
        )
        assert isinstance(limited, StreamingResponse)
        with pytest.raises(ResponseTooLargeError):
            _ = [chunk async for chunk in limited.iter_bytes()]
        assert limited._response.closed
        assert limited._session.closed

    assert isinstance(error, BufferedResponse)
    assert error.status_code == 429
    assert error.body == b"retry later"
    assert error.headers == {"content-type": "text/plain"}
    assert isinstance(empty, BufferedResponse)
    assert empty.status_code == 204
    assert empty.body == b""


@pytest.mark.anyio
async def test_raw_stream_cancellation_closes_upstream_resources():
    class BlockingContent:
        async def iter_chunked(self, _chunk_bytes: int):
            yield b"first"
            await asyncio.Event().wait()

    class StubResponse:
        status = 200

        def __init__(self):
            self.content = BlockingContent()
            self.closed = False

        def close(self):
            self.closed = True

    class StubSession:
        def __init__(self):
            self.closed = False

        async def close(self):
            self.closed = True

    upstream = StubResponse()
    session = StubSession()
    response = StreamingResponse(
        profile=EndpointProfile(
            name="cancelled-raw-stream",
            base_url="https://service.example",
            path_template="/stream",
            method="GET",
            body_mode=BodyMode.NONE,
            response_mode=ResponseMode.STREAM,
        ),
        response=upstream,
        session=session,
        headers={"content-type": "application/octet-stream"},
        private_headers={},
        hooks=(),
    )
    iterator = response.iter_bytes()
    assert await anext(iterator) == b"first"
    pending = asyncio.create_task(anext(iterator))
    await asyncio.sleep(0)
    pending.cancel()
    with pytest.raises(asyncio.CancelledError):
        await pending

    assert upstream.closed
    assert session.closed


@pytest.mark.anyio
async def test_response_size_and_timeout_limits():
    async def large_handler(_request: web.Request) -> web.Response:
        return web.Response(body=b"12345")

    async def slow_handler(_request: web.Request) -> web.Response:
        await asyncio.sleep(0.15)
        return web.Response(body=b"late")

    app = web.Application()
    app.router.add_get("/large", large_handler)
    app.router.add_get("/slow", slow_handler)
    async with running_app(app) as base_url:
        executor = ExternalExecutor()
        with pytest.raises(ResponseTooLargeError):
            await executor.execute(
                EndpointProfile(
                    name="bounded",
                    base_url=base_url,
                    path_template="/large",
                    method="GET",
                    body_mode=BodyMode.NONE,
                    network=local_network(),
                    max_response_bytes=4,
                ),
                OutboundRequest(),
            )
        with pytest.raises(UpstreamTimeoutError):
            await executor.execute(
                EndpointProfile(
                    name="timed",
                    base_url=base_url,
                    path_template="/slow",
                    method="GET",
                    body_mode=BodyMode.NONE,
                    network=local_network(),
                    timeout=TimeoutProfile(
                        total=0.05,
                        connect=0.05,
                        socket_connect=0.05,
                        socket_read=0.05,
                    ),
                ),
                OutboundRequest(),
            )


@pytest.mark.anyio
async def test_artifact_range_and_head_are_streamed_without_identity_headers():
    blob = b"0123456789"
    seen = []

    async def handler(request: web.Request) -> web.Response:
        lowered = {name.lower(): value for name, value in request.headers.items()}
        seen.append((request.method, lowered))
        headers = {
            "Accept-Ranges": "bytes",
            "Cache-Control": "public, s-maxage=86400",
            "Content-Disposition": 'attachment; filename="upstream.bin"',
            "ETag": '"artifact-1"',
            "X-Request-ID": "artifact-request-1",
            "Set-Cookie": "upstream=value",
        }
        if request.method == "HEAD":
            headers["Content-Length"] = str(len(blob))
            return web.Response(status=200, headers=headers)
        range_value = request.headers.get("Range")
        if range_value == "bytes=2-5":
            headers["Content-Range"] = f"bytes 2-5/{len(blob)}"
            return web.Response(status=206, body=blob[2:6], headers=headers)
        return web.Response(body=blob, headers=headers)

    app = web.Application()
    app.router.add_route("*", "/artifact", handler)
    async with running_app(app) as base_url:
        parsed_url = urlsplit(base_url)
        assert parsed_url.hostname and parsed_url.port
        relay = ArtifactRelay(
            ArtifactProfile(
                network=local_network(
                    hosts=(parsed_url.hostname,),
                    ports=frozenset({parsed_url.port}),
                ),
                stream_chunk_bytes=2,
            )
        )
        ranged = await relay.open(
            f"{base_url}/artifact",
            request_headers={
                "Range": "bytes=2-5",
                "Authorization": "client-secret",
                "Cookie": "client=value",
            },
        )
        ranged_body = b"".join([chunk async for chunk in ranged.iter_bytes()])
        head = await relay.open(f"{base_url}/artifact", method="HEAD")
        head_body = b"".join([chunk async for chunk in head.iter_bytes()])

    assert ranged.status_code == 206
    assert ranged_body == b"2345"
    assert ranged.headers["content-range"] == "bytes 2-5/10"
    assert "content-disposition" not in ranged.headers
    assert "cache-control" not in ranged.headers
    assert "etag" not in ranged.headers
    assert "x-request-id" not in ranged.headers
    assert "set-cookie" not in ranged.headers
    assert head.status_code == 200
    assert head.headers["content-length"] == "10"
    assert head_body == b""
    assert seen[0][1]["range"] == "bytes=2-5"
    assert "authorization" not in seen[0][1]
    assert "cookie" not in seen[0][1]
    assert "user-agent" not in seen[0][1]
    with pytest.raises(RequestRejectedError):
        await relay.open(f"{base_url}/artifact", request_headers={"Range": "items=1-2"})
    with pytest.raises(RequestRejectedError):
        await relay.open(
            f"{base_url}/artifact", request_headers={"Range": "bytes=0-1,4-5"}
        )


@pytest.mark.anyio
async def test_artifact_relay_rejects_declared_and_streamed_oversize_bodies():
    async def declared(_request: web.Request) -> web.Response:
        return web.Response(body=b"12345")

    async def chunked(request: web.Request) -> web.StreamResponse:
        response = web.StreamResponse()
        response.enable_chunked_encoding()
        await response.prepare(request)
        await response.write(b"123")
        await response.write(b"456")
        await response.write_eof()
        return response

    app = web.Application()
    app.router.add_get("/declared", declared)
    app.router.add_get("/chunked", chunked)
    async with running_app(app) as base_url:
        parsed_url = urlsplit(base_url)
        assert parsed_url.hostname and parsed_url.port
        relay = ArtifactRelay(
            ArtifactProfile(
                network=local_network(
                    hosts=(parsed_url.hostname,),
                    ports=frozenset({parsed_url.port}),
                ),
                max_bytes=4,
                stream_chunk_bytes=3,
            )
        )
        with pytest.raises(ResponseTooLargeError):
            await relay.open(f"{base_url}/declared")

        streamed = await relay.open(f"{base_url}/chunked")
        with pytest.raises(ResponseTooLargeError):
            _ = [chunk async for chunk in streamed.iter_bytes()]

        larger_profile = ArtifactRelay(
            ArtifactProfile(
                network=local_network(
                    hosts=(parsed_url.hostname,),
                    ports=frozenset({parsed_url.port}),
                ),
                max_bytes=10,
            )
        )
        reserved = await larger_profile.open(f"{base_url}/declared")
        with pytest.raises(ResponseTooLargeError):
            _ = [chunk async for chunk in reserved.iter_bytes(max_bytes=4)]


def test_artifact_profile_requires_a_positive_byte_limit():
    with pytest.raises(ProfileError, match="max_bytes"):
        ArtifactProfile(
            network=NetworkPolicy(allowed_hosts=("assets.example.test",)),
            max_bytes=0,
        )


class _RelayPeer:
    def __init__(self) -> None:
        self.sent = []
        self.closed = None
        self._echo_received = asyncio.Event()
        self._receive_count = 0

    async def receive(self) -> WebSocketFrame:
        self._receive_count += 1
        if self._receive_count == 1:
            return WebSocketFrame(WebSocketFrameType.TEXT, '{"request": 1}')
        await self._echo_received.wait()
        return WebSocketFrame(WebSocketFrameType.CLOSE, close_code=1000)

    async def send(self, frame: WebSocketFrame) -> None:
        self.sent.append(frame)
        self._echo_received.set()

    async def close(self, code: int = 1000, reason: str = "") -> None:
        self.closed = (code, reason)
        self._echo_received.set()


@pytest.mark.anyio
async def test_websocket_authenticated_relay_subprotocols_frames_and_usage_hooks():
    seen = {}

    async def handler(request: web.Request) -> web.WebSocketResponse:
        seen["headers"] = {
            name.lower(): value for name, value in request.headers.items()
        }
        websocket = web.WebSocketResponse(protocols=("events.v1",))
        await websocket.prepare(request)
        async for message in websocket:
            if message.type is WSMsgType.TEXT:
                await websocket.send_str('{"usage": {"units": 4}}')
            elif message.type is WSMsgType.BINARY:
                await websocket.send_bytes(message.data)
        return websocket

    app = web.Application()
    app.router.add_get("/socket", handler)
    observations = []

    async def resolve_secret(reference: str) -> str:
        assert reference == "socket-credential"
        return "server-secret"

    async with running_app(app) as base_url:
        socket_url = base_url.replace("http://", "ws://")
        connection = await WebSocketRelay(
            secret_resolver=resolve_secret,
            usage_hooks=(observations.append,),
        ).connect(
            WebSocketProfile(
                name="duplex-events",
                base_url=socket_url,
                path_template="/socket",
                allowed_request_headers=frozenset(
                    {"authorization", "cookie", "x-forwarded-for", "x-trace"}
                ),
                secret_headers=(
                    SecretHeaderTemplate(
                        name="Authorization",
                        template="Bearer {credential}",
                        references={"credential": "socket-credential"},
                    ),
                ),
                allowed_subprotocols=("events.v1",),
                require_subprotocol=True,
                network=local_network(websocket=True),
                idle_timeout_seconds=2,
                max_session_seconds=10,
                heartbeat_seconds=None,
            ),
            WebSocketRequest(
                headers={
                    "Authorization": "client-secret",
                    "Cookie": "client=value",
                    "X-Forwarded-For": "127.0.0.1",
                    "X-Trace": "trace-value",
                },
                subprotocols=("unlisted", "events.v1"),
            ),
        )
        peer = _RelayPeer()
        relay_result = await connection.relay(peer)

    assert connection.subprotocol == "events.v1"
    assert seen["headers"]["authorization"] == "Bearer server-secret"
    assert seen["headers"]["x-trace"] == "trace-value"
    assert "cookie" not in seen["headers"]
    assert "x-forwarded-for" not in seen["headers"]
    assert peer.sent == [
        WebSocketFrame(WebSocketFrameType.TEXT, '{"usage": {"units": 4}}')
    ]
    assert relay_result.close_code == 1000
    assert relay_result.outbound_messages == 1
    assert relay_result.inbound_messages == 1
    assert [item.direction for item in observations] == [
        MessageDirection.OUTBOUND,
        MessageDirection.INBOUND,
    ]
    assert observations[1].json_value == {"usage": {"units": 4}}


@pytest.mark.anyio
async def test_websocket_handshake_cancellation_closes_the_upstream_session(
    monkeypatch,
):
    class StubSession:
        def __init__(self):
            self.started = asyncio.Event()
            self.closed = False

        async def ws_connect(self, *_args, **_kwargs):
            self.started.set()
            await asyncio.Event().wait()

        async def close(self):
            self.closed = True

    session = StubSession()
    monkeypatch.setattr(
        "api.external_transport.websocket.aiohttp.ClientSession",
        lambda **_kwargs: session,
    )
    monkeypatch.setattr(
        "api.external_transport.websocket.create_connector",
        lambda _network: object(),
    )
    task = asyncio.create_task(
        WebSocketRelay().connect(
            WebSocketProfile(
                name="cancelled-handshake",
                base_url="wss://service.example.test",
                path_template="/socket",
            ),
            WebSocketRequest(),
        )
    )
    await session.started.wait()
    task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await task

    assert session.closed is True


@pytest.mark.anyio
async def test_websocket_message_limit_and_close_reason_are_local():
    async def handler(request: web.Request) -> web.WebSocketResponse:
        websocket = web.WebSocketResponse()
        await websocket.prepare(request)
        message = await websocket.receive()
        if message.type is WSMsgType.TEXT:
            await websocket.close(code=1008, message=b"sensitive upstream detail")
        return websocket

    app = web.Application()
    app.router.add_get("/socket", handler)
    async with running_app(app) as base_url:
        socket_url = base_url.replace("http://", "ws://")
        relay = WebSocketRelay()
        limited = await relay.connect(
            WebSocketProfile(
                name="small-messages",
                base_url=socket_url,
                path_template="/socket",
                network=local_network(websocket=True),
                max_message_bytes=4,
                heartbeat_seconds=None,
            ),
            WebSocketRequest(),
        )
        with pytest.raises(WebSocketLimitError):
            await limited.send(WebSocketFrame(WebSocketFrameType.TEXT, "12345"))
        await limited.close()

        sanitizing = await relay.connect(
            WebSocketProfile(
                name="safe-close",
                base_url=socket_url,
                path_template="/socket",
                network=local_network(websocket=True),
                heartbeat_seconds=None,
            ),
            WebSocketRequest(),
        )
        await sanitizing.send(WebSocketFrame(WebSocketFrameType.TEXT, "ok"))
        close_frame = await sanitizing.receive()
        await sanitizing.close()

    assert close_frame.kind is WebSocketFrameType.CLOSE
    assert close_frame.close_code == 1008
    assert close_frame.close_reason == "policy violation"
    assert "sensitive" not in close_frame.close_reason
