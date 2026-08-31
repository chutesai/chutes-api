"""Profile-driven execution for buffered and streaming HTTP APIs."""

from __future__ import annotations

import asyncio
import inspect
import re
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Mapping, Sequence

import aiohttp
import orjson

from .errors import (
    RedirectRejectedError,
    RequestRejectedError,
    ResponseTooLargeError,
    StreamProtocolError,
    UpstreamConnectionError,
    UpstreamTimeoutError,
)
from .models import (
    BodyMode,
    EndpointProfile,
    JsonBody,
    MultipartBody,
    OutboundRequest,
    RawBody,
    ResponseMode,
    SSEEvent,
    SecretResolver,
    UsageHook,
    UsageObservation,
)
from .security import (
    build_endpoint_url,
    create_connector,
    endpoint_network_policy,
    observe_response_headers,
    render_secret_headers,
    sanitize_client_headers,
    sanitize_response_headers,
    strip_headers_for_cross_origin_redirect,
    validate_profile_headers,
    validate_redirect,
)


_REDIRECT_STATUSES = frozenset({301, 302, 303, 307, 308})
_SSE_BOUNDARY_RE = re.compile(rb"\r?\n\r?\n")


@dataclass(frozen=True, slots=True)
class BufferedResponse:
    """A bounded upstream response held in memory."""

    status_code: int
    headers: Mapping[str, str]
    body: bytes
    private_headers: Mapping[str, str] = field(default_factory=dict, repr=False)

    def json(self) -> Any:
        return orjson.loads(self.body)


async def _invoke_hooks(
    hooks: Sequence[UsageHook], observation: UsageObservation
) -> None:
    for hook in hooks:
        result = hook(observation)
        if inspect.isawaitable(result):
            await result


def _json_value(body: bytes) -> Any | None:
    if not body:
        return None
    try:
        return orjson.loads(body)
    except orjson.JSONDecodeError:
        return None


class _SSEDecoder:
    def __init__(self, max_event_bytes: int) -> None:
        self._buffer = bytearray()
        self._max_event_bytes = max_event_bytes

    def feed(self, chunk: bytes) -> list[SSEEvent]:
        self._buffer.extend(chunk)
        events: list[SSEEvent] = []
        while match := _SSE_BOUNDARY_RE.search(self._buffer):
            raw = bytes(self._buffer[: match.start()])
            del self._buffer[: match.end()]
            if len(raw) > self._max_event_bytes:
                raise StreamProtocolError(
                    "server-sent event exceeded its configured byte limit"
                )
            event = self._decode(raw)
            if event is not None:
                events.append(event)
        if len(self._buffer) > self._max_event_bytes:
            raise StreamProtocolError(
                "server-sent event exceeded its configured byte limit"
            )
        return events

    def finish(self) -> list[SSEEvent]:
        if not self._buffer:
            return []
        raw = bytes(self._buffer)
        self._buffer.clear()
        if len(raw) > self._max_event_bytes:
            raise StreamProtocolError(
                "server-sent event exceeded its configured byte limit"
            )
        event = self._decode(raw)
        return [event] if event is not None else []

    @staticmethod
    def _decode(raw: bytes) -> SSEEvent | None:
        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise StreamProtocolError("server-sent events must be UTF-8") from exc
        data: list[str] = []
        event_name: str | None = None
        event_id: str | None = None
        retry: int | None = None
        comments: list[str] = []
        for line in text.replace("\r\n", "\n").replace("\r", "\n").split("\n"):
            if line.startswith(":"):
                comments.append(line[1:].lstrip(" "))
                continue
            field, separator, value = line.partition(":")
            if separator and value.startswith(" "):
                value = value[1:]
            if field == "data":
                data.append(value)
            elif field == "event":
                event_name = value
            elif field == "id" and "\x00" not in value:
                event_id = value
            elif field == "retry" and value.isdigit():
                retry = int(value)
        if (
            not data
            and event_name is None
            and event_id is None
            and retry is None
            and not comments
        ):
            return None
        return SSEEvent(
            data="\n".join(data),
            event=event_name,
            event_id=event_id,
            retry=retry,
            comments=tuple(comments),
        )


class StreamingResponse:
    """An open, bounded response stream with one-shot consumption."""

    def __init__(
        self,
        *,
        profile: EndpointProfile,
        response: aiohttp.ClientResponse,
        session: aiohttp.ClientSession,
        headers: Mapping[str, str],
        private_headers: Mapping[str, str],
        hooks: Sequence[UsageHook],
    ) -> None:
        self.status_code = response.status
        self.headers = headers
        self.private_headers = private_headers
        self.response_mode = profile.response_mode
        self._profile = profile
        self._response = response
        self._session = session
        self._hooks = tuple(hooks)
        self._started = False
        self._closed = False

    def _start(self) -> None:
        if self._started:
            raise RuntimeError("streaming response can only be consumed once")
        if self._closed:
            raise RuntimeError("streaming response is closed")
        self._started = True

    def _count_bytes(self, total: int, chunk: bytes) -> int:
        total += len(chunk)
        if total > self._profile.max_response_bytes:
            raise ResponseTooLargeError(
                "upstream response exceeds configured byte limit"
            )
        return total

    async def _observe(self, event: SSEEvent) -> None:
        await _invoke_hooks(
            self._hooks,
            UsageObservation(
                profile_name=self._profile.name,
                status_code=self.status_code,
                response_headers=self.private_headers,
                json_value=_json_value(event.data.encode("utf-8")),
                sse_event=event,
            ),
        )

    async def iter_bytes(self) -> AsyncIterator[bytes]:
        """Relay bounded raw bytes with normal consumer backpressure."""

        self._start()
        decoder = (
            _SSEDecoder(self._profile.max_sse_event_bytes)
            if self.response_mode is ResponseMode.SSE
            else None
        )
        total = 0
        try:
            async for chunk in self._response.content.iter_chunked(
                self._profile.stream_chunk_bytes
            ):
                total = self._count_bytes(total, chunk)
                if decoder is not None:
                    for event in decoder.feed(chunk):
                        await self._observe(event)
                yield chunk
            if decoder is not None:
                for event in decoder.finish():
                    await self._observe(event)
        except asyncio.TimeoutError as exc:
            raise UpstreamTimeoutError("upstream stream timed out") from exc
        except aiohttp.ClientError as exc:
            raise UpstreamConnectionError("upstream stream ended unexpectedly") from exc
        finally:
            await self.aclose()

    async def iter_sse(self) -> AsyncIterator[SSEEvent]:
        """Decode and yield server-sent events."""

        if self.response_mode is not ResponseMode.SSE:
            raise RuntimeError("iter_sse is only available for SSE responses")
        self._start()
        decoder = _SSEDecoder(self._profile.max_sse_event_bytes)
        total = 0
        try:
            async for chunk in self._response.content.iter_chunked(
                self._profile.stream_chunk_bytes
            ):
                total = self._count_bytes(total, chunk)
                for event in decoder.feed(chunk):
                    await self._observe(event)
                    yield event
            for event in decoder.finish():
                await self._observe(event)
                yield event
        except asyncio.TimeoutError as exc:
            raise UpstreamTimeoutError("upstream stream timed out") from exc
        except aiohttp.ClientError as exc:
            raise UpstreamConnectionError("upstream stream ended unexpectedly") from exc
        finally:
            await self.aclose()

    async def aclose(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._response.close()
        await self._session.close()

    async def __aenter__(self) -> StreamingResponse:
        return self

    async def __aexit__(self, *_args: object) -> None:
        await self.aclose()


class ExternalExecutor:
    """Execute declarative endpoint profiles across a strict network boundary."""

    def __init__(
        self,
        *,
        secret_resolver: SecretResolver | None = None,
        usage_hooks: Sequence[UsageHook] = (),
    ) -> None:
        self._secret_resolver = secret_resolver
        self._usage_hooks = tuple(usage_hooks)

    async def execute(
        self,
        profile: EndpointProfile,
        request: OutboundRequest,
        *,
        usage_hooks: Sequence[UsageHook] = (),
    ) -> BufferedResponse | StreamingResponse:
        """Execute one request and return a buffered or open streaming response."""

        network = endpoint_network_policy(profile.base_url, profile.network)
        target_url = build_endpoint_url(
            profile.base_url,
            profile.path_template,
            request.path_parameters,
            request.query,
            network,
        )
        headers = sanitize_client_headers(
            request.headers, profile.allowed_request_headers
        )
        headers.update(validate_profile_headers(profile.static_headers))
        secret_headers, _secret_names = await render_secret_headers(
            profile.secret_headers, self._secret_resolver
        )
        headers.update(secret_headers)
        headers.setdefault("accept-encoding", "identity")
        if profile.response_mode is ResponseMode.SSE:
            headers.setdefault("accept", "text/event-stream")
        self._validate_body(profile.body_mode, request.body)

        timeout = aiohttp.ClientTimeout(
            total=profile.timeout.total,
            connect=profile.timeout.connect,
            sock_connect=profile.timeout.socket_connect,
            sock_read=profile.timeout.socket_read,
        )
        session = aiohttp.ClientSession(
            connector=create_connector(network),
            timeout=timeout,
            auto_decompress=True,
            cookie_jar=aiohttp.DummyCookieJar(),
            skip_auto_headers={"User-Agent"},
            trust_env=False,
        )
        response: aiohttp.ClientResponse | None = None
        handed_off = False
        method = profile.method
        current_url = target_url
        current_headers = headers
        current_body = request.body
        try:
            redirect_count = 0
            while True:
                request_headers = dict(current_headers)
                request_kwargs = self._body_kwargs(
                    profile.body_mode, current_body, request_headers
                )
                response = await session.request(
                    method,
                    current_url,
                    headers=request_headers,
                    allow_redirects=False,
                    **request_kwargs,
                )
                if response.status not in _REDIRECT_STATUSES:
                    break
                if redirect_count >= profile.redirects.max_redirects:
                    raise RedirectRejectedError("upstream redirect limit exceeded")
                if method not in {"GET", "HEAD"} and response.status not in {307, 308}:
                    raise RedirectRejectedError(
                        "redirect would change the method or discard the request body"
                    )
                next_url, cross_origin = validate_redirect(
                    current_url,
                    response.headers.get("Location", ""),
                    network,
                    profile.redirects,
                )
                response.release()
                response = None
                if cross_origin:
                    current_headers = strip_headers_for_cross_origin_redirect(
                        current_headers
                    )
                current_url = next_url
                redirect_count += 1

            safe_headers = sanitize_response_headers(
                response.headers,
                profile.allowed_response_headers,
            )
            private_headers = observe_response_headers(
                response.headers,
                profile.allowed_response_headers | profile.private_response_headers,
            )
            hooks = self._usage_hooks + tuple(usage_hooks)
            if (
                profile.response_mode in {ResponseMode.SSE, ResponseMode.STREAM}
                and 200 <= response.status < 300
                and response.status != 204
            ):
                if profile.response_mode is ResponseMode.SSE:
                    media_type = response.headers.get("Content-Type", "").lower()
                    if not media_type.startswith("text/event-stream"):
                        raise StreamProtocolError(
                            "upstream response is not a server-sent-event stream"
                        )
                elif hooks:
                    await _invoke_hooks(
                        hooks,
                        UsageObservation(
                            profile_name=profile.name,
                            status_code=response.status,
                            response_headers=private_headers,
                        ),
                    )
                result = StreamingResponse(
                    profile=profile,
                    response=response,
                    session=session,
                    headers=safe_headers,
                    private_headers=private_headers,
                    hooks=hooks,
                )
                response = None
                handed_off = True
                return result

            body = await self._read_bounded(response, profile.max_response_bytes)
            result = BufferedResponse(
                status_code=response.status,
                headers=safe_headers,
                body=body,
                private_headers=private_headers,
            )
            await _invoke_hooks(
                hooks,
                UsageObservation(
                    profile_name=profile.name,
                    status_code=result.status_code,
                    response_headers=result.private_headers,
                    json_value=_json_value(body),
                ),
            )
            response.release()
            response = None
            await session.close()
            return result
        except asyncio.TimeoutError as exc:
            raise UpstreamTimeoutError("upstream request timed out") from exc
        except aiohttp.ClientError as exc:
            raise UpstreamConnectionError("upstream request failed") from exc
        finally:
            if response is not None and not response.closed:
                response.close()
            if not handed_off and not session.closed:
                await session.close()

    @staticmethod
    def _validate_body(mode: BodyMode, body: object) -> None:
        expected: dict[BodyMode, type | tuple[type, ...]] = {
            BodyMode.NONE: type(None),
            BodyMode.JSON: JsonBody,
            BodyMode.RAW: RawBody,
            BodyMode.MULTIPART: MultipartBody,
        }
        if not isinstance(body, expected[mode]):
            raise RequestRejectedError(
                f"request body does not match {mode.value} body mode"
            )

    @staticmethod
    def _body_kwargs(
        mode: BodyMode,
        body: JsonBody | RawBody | MultipartBody | None,
        headers: dict[str, str],
    ) -> dict[str, Any]:
        if mode is BodyMode.NONE:
            return {}
        if mode is BodyMode.JSON:
            assert isinstance(body, JsonBody)
            try:
                payload = orjson.dumps(body.value)
            except (TypeError, ValueError) as exc:
                raise RequestRejectedError(
                    "JSON request body is not serializable"
                ) from exc
            headers["content-type"] = "application/json"
            return {"data": payload}
        if mode is BodyMode.RAW:
            assert isinstance(body, RawBody)
            payload = (
                body.value.encode("utf-8")
                if isinstance(body.value, str)
                else body.value
            )
            if body.content_type:
                if any(char in body.content_type for char in "\r\n\x00"):
                    raise RequestRejectedError("raw body content type is invalid")
                headers["content-type"] = body.content_type
            else:
                headers.setdefault("content-type", "application/octet-stream")
            return {"data": payload}
        assert isinstance(body, MultipartBody)
        headers.pop("content-type", None)
        form = aiohttp.FormData(quote_fields=True)
        for part in body.parts:
            for value in (part.name, part.filename, part.content_type):
                if value is not None and any(char in value for char in "\r\n\x00"):
                    raise RequestRejectedError("multipart metadata is invalid")
            form.add_field(
                part.name,
                part.value,
                filename=part.filename,
                content_type=part.content_type,
            )
        return {"data": form}

    @staticmethod
    async def _read_bounded(response: aiohttp.ClientResponse, limit: int) -> bytes:
        declared = response.headers.get("Content-Length")
        if declared:
            try:
                if int(declared) > limit:
                    raise ResponseTooLargeError(
                        "upstream response exceeds configured byte limit"
                    )
            except ValueError:
                pass
        chunks: list[bytes] = []
        size = 0
        async for chunk in response.content.iter_chunked(min(64 * 1024, limit)):
            size += len(chunk)
            if size > limit:
                raise ResponseTooLargeError(
                    "upstream response exceeds configured byte limit"
                )
            chunks.append(chunk)
        return b"".join(chunks)
