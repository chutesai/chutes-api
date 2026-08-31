"""Bounded, transport-neutral WebSocket connections and relays."""

from __future__ import annotations

import asyncio
import contextlib
import inspect
import re
from dataclasses import dataclass
from typing import Protocol, Sequence

import aiohttp
import orjson

from .errors import (
    RedirectRejectedError,
    RequestRejectedError,
    UpstreamConnectionError,
    UpstreamTimeoutError,
    WebSocketLimitError,
)
from .models import (
    MessageDirection,
    SecretResolver,
    UsageHook,
    UsageObservation,
    WebSocketFrame,
    WebSocketFrameType,
    WebSocketProfile,
    WebSocketRequest,
)
from .security import (
    build_endpoint_url,
    create_connector,
    endpoint_network_policy,
    render_secret_headers,
    sanitize_client_headers,
    validate_profile_headers,
)


_SUBPROTOCOL_RE = re.compile(r"^[!#$%&'*+.^_`|~0-9A-Za-z-]+$")
_VALID_STANDARD_CLOSE_CODES = frozenset(
    {1000, 1001, 1002, 1003, 1007, 1008, 1009, 1010, 1011, 1012, 1013, 1014}
)


class WebSocketPeer(Protocol):
    """Minimal downstream interface required by the relay."""

    async def receive(self) -> WebSocketFrame: ...

    async def send(self, frame: WebSocketFrame) -> None: ...

    async def close(self, code: int = 1000, reason: str = "") -> None: ...


@dataclass(frozen=True, slots=True)
class WebSocketRelayResult:
    close_code: int
    close_reason: str
    outbound_messages: int
    inbound_messages: int


def _safe_close_code(code: int | None, *, abnormal: bool = False) -> int:
    if code in _VALID_STANDARD_CLOSE_CODES or (
        code is not None and 3000 <= code <= 4999
    ):
        return code
    return 1011 if abnormal else 1000


def _safe_close_reason(code: int) -> str:
    return {
        1000: "normal closure",
        1001: "endpoint going away",
        1002: "protocol error",
        1003: "unsupported data",
        1007: "invalid data",
        1008: "policy violation",
        1009: "message too large",
        1010: "extension required",
        1011: "upstream error",
        1012: "service restart",
        1013: "retry later",
        1014: "gateway error",
    }.get(code, "connection closed")


async def _invoke_hooks(
    hooks: Sequence[UsageHook], observation: UsageObservation
) -> None:
    for hook in hooks:
        result = hook(observation)
        if inspect.isawaitable(result):
            await result


class WebSocketConnection:
    """An authenticated upstream WebSocket with bounded message and lifetime operations."""

    def __init__(
        self,
        *,
        profile: WebSocketProfile,
        websocket: aiohttp.ClientWebSocketResponse,
        session: aiohttp.ClientSession,
        hooks: Sequence[UsageHook],
    ) -> None:
        self.subprotocol = websocket.protocol
        self._profile = profile
        self._websocket = websocket
        self._session = session
        self._hooks = tuple(hooks)
        loop = asyncio.get_running_loop()
        self._started_at = loop.time()
        self._last_activity_at = self._started_at
        self._closed = False

    def _message_size(self, frame: WebSocketFrame) -> int:
        if frame.kind is WebSocketFrameType.TEXT:
            assert isinstance(frame.data, str)
            return len(frame.data.encode("utf-8"))
        if frame.kind is WebSocketFrameType.BINARY:
            assert isinstance(frame.data, bytes)
            return len(frame.data)
        return 0

    def _remaining_limit(self) -> tuple[float, str]:
        now = asyncio.get_running_loop().time()
        session_left = self._profile.max_session_seconds - (now - self._started_at)
        idle_left = self._profile.idle_timeout_seconds - (now - self._last_activity_at)
        if session_left <= idle_left:
            return session_left, "session limit reached"
        return idle_left, "idle timeout"

    def _touch(self) -> None:
        self._last_activity_at = asyncio.get_running_loop().time()

    async def _observe_text(self, text: str, direction: MessageDirection) -> None:
        try:
            value = orjson.loads(text)
        except orjson.JSONDecodeError:
            return
        await _invoke_hooks(
            self._hooks,
            UsageObservation(
                profile_name=self._profile.name,
                status_code=101,
                response_headers={},
                json_value=value,
                direction=direction,
            ),
        )

    def _check_message(self, frame: WebSocketFrame) -> None:
        if frame.kind not in {
            WebSocketFrameType.TEXT,
            WebSocketFrameType.BINARY,
            WebSocketFrameType.CLOSE,
        }:
            raise RequestRejectedError("unsupported WebSocket frame type")
        if self._message_size(frame) > self._profile.max_message_bytes:
            raise WebSocketLimitError(
                "WebSocket message exceeded its configured byte limit"
            )

    async def send(self, frame: WebSocketFrame) -> None:
        """Send one text, binary, or close frame upstream."""

        if self._closed:
            raise UpstreamConnectionError("WebSocket connection is closed")
        remaining, reason = self._remaining_limit()
        if remaining <= 0:
            await self.close(1001)
            raise WebSocketLimitError(reason)
        try:
            self._check_message(frame)
            if frame.kind is WebSocketFrameType.TEXT:
                assert isinstance(frame.data, str)
                await self._websocket.send_str(frame.data)
                await self._observe_text(frame.data, MessageDirection.OUTBOUND)
                self._touch()
            elif frame.kind is WebSocketFrameType.BINARY:
                assert isinstance(frame.data, bytes)
                await self._websocket.send_bytes(frame.data)
                self._touch()
            else:
                await self.close(frame.close_code)
        except WebSocketLimitError:
            await self.close(1009)
            raise
        except aiohttp.ClientError as exc:
            raise UpstreamConnectionError("failed to send WebSocket frame") from exc

    async def _receive_frame(self) -> WebSocketFrame:
        while True:
            message = await self._websocket.receive()
            if message.type is aiohttp.WSMsgType.TEXT:
                frame = WebSocketFrame(WebSocketFrameType.TEXT, message.data)
                self._check_message(frame)
                await self._observe_text(message.data, MessageDirection.INBOUND)
                self._touch()
                return frame
            if message.type is aiohttp.WSMsgType.BINARY:
                frame = WebSocketFrame(WebSocketFrameType.BINARY, bytes(message.data))
                self._check_message(frame)
                self._touch()
                return frame
            if message.type is aiohttp.WSMsgType.CLOSE:
                code = _safe_close_code(message.data)
                return WebSocketFrame(
                    WebSocketFrameType.CLOSE,
                    close_code=code,
                    close_reason=_safe_close_reason(code),
                )
            if message.type is aiohttp.WSMsgType.CLOSED:
                code = _safe_close_code(self._websocket.close_code)
                return WebSocketFrame(
                    WebSocketFrameType.CLOSE,
                    close_code=code,
                    close_reason=_safe_close_reason(code),
                )
            if message.type is aiohttp.WSMsgType.ERROR:
                raise UpstreamConnectionError("upstream WebSocket reported an error")

    async def receive(self) -> WebSocketFrame:
        """Receive one upstream frame while enforcing idle and session deadlines."""

        if self._closed:
            return WebSocketFrame(
                WebSocketFrameType.CLOSE,
                close_code=1000,
                close_reason=_safe_close_reason(1000),
            )
        while True:
            remaining, reason = self._remaining_limit()
            if remaining <= 0:
                await self.close(1001)
                raise WebSocketLimitError(reason)
            try:
                return await asyncio.wait_for(self._receive_frame(), timeout=remaining)
            except asyncio.TimeoutError as exc:
                remaining, reason = self._remaining_limit()
                if remaining > 0:
                    continue
                await self.close(1001)
                raise WebSocketLimitError(reason) from exc
            except aiohttp.ClientError as exc:
                await self.close(1011)
                raise UpstreamConnectionError(
                    "failed to receive WebSocket frame"
                ) from exc

    async def close(self, code: int | None = 1000) -> None:
        if self._closed:
            return
        self._closed = True
        safe_code = _safe_close_code(code)
        reason = _safe_close_reason(safe_code).encode("utf-8")
        with contextlib.suppress(Exception):
            await self._websocket.close(code=safe_code, message=reason)
        await self._session.close()

    async def relay(self, peer: WebSocketPeer) -> WebSocketRelayResult:
        """Relay text and binary frames in both directions until close or a limit."""

        outbound_messages = 0
        inbound_messages = 0
        terminal_code = 1000
        terminal_reason = _safe_close_reason(1000)

        async def outbound() -> tuple[int, str, int]:
            nonlocal outbound_messages
            while True:
                frame = await peer.receive()
                self._check_message(frame)
                if frame.kind is WebSocketFrameType.CLOSE:
                    code = _safe_close_code(frame.close_code)
                    await self.close(code)
                    return code, _safe_close_reason(code), outbound_messages
                await self.send(frame)
                outbound_messages += 1

        async def inbound() -> tuple[int, str, int]:
            nonlocal inbound_messages
            while True:
                frame = await self._receive_frame()
                if frame.kind is WebSocketFrameType.CLOSE:
                    code = _safe_close_code(frame.close_code)
                    await peer.close(code, _safe_close_reason(code))
                    return code, _safe_close_reason(code), inbound_messages
                await peer.send(frame)
                inbound_messages += 1

        tasks = {asyncio.create_task(outbound()), asyncio.create_task(inbound())}
        try:
            while tasks:
                remaining, limit_reason = self._remaining_limit()
                if remaining <= 0:
                    terminal_code = 1001
                    terminal_reason = limit_reason
                    raise WebSocketLimitError(limit_reason)
                done, pending = await asyncio.wait(
                    tasks,
                    timeout=remaining,
                    return_when=asyncio.FIRST_COMPLETED,
                )
                if not done:
                    continue
                completed = done.pop()
                terminal_code, terminal_reason, _count = await completed
                for task in pending | done:
                    task.cancel()
                await asyncio.gather(*pending, *done, return_exceptions=True)
                tasks.clear()
        except WebSocketLimitError as exc:
            if "message" in str(exc).lower():
                terminal_code = 1009
                terminal_reason = _safe_close_reason(1009)
            else:
                terminal_code = 1001
                terminal_reason = str(exc)
            await self.close(terminal_code)
            with contextlib.suppress(Exception):
                await peer.close(terminal_code, terminal_reason)
        except Exception:
            terminal_code = 1011
            terminal_reason = _safe_close_reason(terminal_code)
            await self.close(terminal_code)
            with contextlib.suppress(Exception):
                await peer.close(terminal_code, terminal_reason)
            raise
        finally:
            for task in tasks:
                task.cancel()
            if tasks:
                await asyncio.gather(*tasks, return_exceptions=True)
            await self.close(terminal_code)
        return WebSocketRelayResult(
            close_code=terminal_code,
            close_reason=terminal_reason,
            outbound_messages=outbound_messages,
            inbound_messages=inbound_messages,
        )

    async def __aenter__(self) -> WebSocketConnection:
        return self

    async def __aexit__(self, *_args: object) -> None:
        await self.close()


class WebSocketRelay:
    """Create authenticated upstream WebSockets from declarative profiles."""

    def __init__(
        self,
        *,
        secret_resolver: SecretResolver | None = None,
        usage_hooks: Sequence[UsageHook] = (),
    ) -> None:
        self._secret_resolver = secret_resolver
        self._usage_hooks = tuple(usage_hooks)

    async def connect(
        self,
        profile: WebSocketProfile,
        request: WebSocketRequest,
        *,
        usage_hooks: Sequence[UsageHook] = (),
    ) -> WebSocketConnection:
        network = endpoint_network_policy(profile.base_url, profile.network)
        url = build_endpoint_url(
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

        allowed = set(profile.allowed_subprotocols)
        protocols: list[str] = []
        for protocol in request.subprotocols:
            if len(protocol) > 128 or not _SUBPROTOCOL_RE.fullmatch(protocol):
                raise RequestRejectedError("invalid WebSocket subprotocol")
            if protocol in allowed and protocol not in protocols:
                protocols.append(protocol)
        if profile.require_subprotocol and not protocols:
            raise RequestRejectedError(
                "no requested WebSocket subprotocol is permitted"
            )

        async def reject_redirect(
            _session: aiohttp.ClientSession,
            _context: object,
            _params: aiohttp.TraceRequestRedirectParams,
        ) -> None:
            raise RedirectRejectedError(
                "WebSocket handshake redirects are not permitted"
            )

        trace = aiohttp.TraceConfig()
        trace.on_request_redirect.append(reject_redirect)
        timeout = aiohttp.ClientTimeout(total=profile.handshake_timeout_seconds)
        session = aiohttp.ClientSession(
            connector=create_connector(network),
            timeout=timeout,
            auto_decompress=False,
            cookie_jar=aiohttp.DummyCookieJar(),
            skip_auto_headers={"User-Agent"},
            trust_env=False,
            trace_configs=[trace],
        )
        try:
            websocket = await session.ws_connect(
                url,
                headers=headers,
                protocols=protocols,
                timeout=aiohttp.ClientWSTimeout(ws_close=5.0, ws_receive=None),
                autoclose=True,
                autoping=True,
                heartbeat=profile.heartbeat_seconds,
                compress=0,
                max_msg_size=profile.max_message_bytes,
            )
            if websocket.protocol is not None and websocket.protocol not in protocols:
                await websocket.close(code=1002)
                raise UpstreamConnectionError(
                    "upstream selected an unoffered subprotocol"
                )
            return WebSocketConnection(
                profile=profile,
                websocket=websocket,
                session=session,
                hooks=self._usage_hooks + tuple(usage_hooks),
            )
        except asyncio.CancelledError:
            await session.close()
            raise
        except asyncio.TimeoutError as exc:
            await session.close()
            raise UpstreamTimeoutError("WebSocket handshake timed out") from exc
        except aiohttp.ClientError as exc:
            await session.close()
            raise UpstreamConnectionError("WebSocket handshake failed") from exc
        except Exception:
            await session.close()
            raise
