from __future__ import annotations

import asyncio
import sys
from types import ModuleType
from collections import deque
from types import SimpleNamespace
from unittest.mock import AsyncMock

import orjson
import pytest
from starlette.datastructures import Address, Headers, QueryParams, URL

# The development environment currently contains both implementations of the same
# substrate namespace. Realtime tests do not exercise that optional client.
_substrate_package = ModuleType("async_substrate_interface")
_substrate_module = ModuleType("async_substrate_interface.async_substrate")
_substrate_sync_module = ModuleType("async_substrate_interface.sync_substrate")
_substrate_module.AsyncSubstrateInterface = object
_substrate_sync_module.SubstrateInterface = object
_substrate_package.AsyncSubstrateInterface = object
sys.modules.setdefault("async_substrate_interface", _substrate_package)
sys.modules.setdefault("async_substrate_interface.async_substrate", _substrate_module)
sys.modules.setdefault(
    "async_substrate_interface.sync_substrate", _substrate_sync_module
)
_ss58_module = ModuleType("scalecodec.utils.ss58")
_ss58_module.is_valid_ss58_address = lambda _value: False
_ss58_module.ss58_decode = lambda _value: ""
sys.modules.setdefault("scalecodec.utils.ss58", _ss58_module)

from api.external_backend import realtime
from api.external_backend.schemas import ExternalOperationMode, ExternalRouteConfig
from api.external_backend.validation import (
    RouteConfigurationError,
    validate_route_configuration,
)
from api.external_transport import (
    MessageDirection,
    UsageObservation,
    WebSocketFrame,
    WebSocketFrameType,
    WebSocketRelayResult,
)


def _route(
    *,
    usage: dict | None = None,
    usage_mode: str = "cumulative",
    input_schema: dict | None = None,
) -> ExternalRouteConfig:
    return ExternalRouteConfig.model_validate(
        {
            "cord_path": "/session",
            "upstream_resource_id": "resource-v1",
            "operation_mode": ExternalOperationMode.REALTIME.value,
            "protocol": "duplex.events",
            "path_template": "/sessions/{model}",
            "method": "GET",
            "request_config": {
                "allowed_request_headers": [
                    "Authorization",
                    "Cookie",
                    "X-Forwarded-For",
                    "X-Feature",
                ],
                "allowed_query_parameters": ["language", "token"],
                "resource_path": "model",
            },
            "response_config": {"usage": usage} if usage else {},
            "operation_config": {
                "realtime": {
                    "allowed_subprotocols": ["events.v1"],
                    "usage_mode": usage_mode,
                    **({"message_schema": input_schema} if input_schema else {}),
                }
            },
            "capabilities": {},
        }
    )


def _account(**overrides):
    values = {
        "account_id": "account-1",
        "user_id": "owner-1",
        "base_url": "https://gateway.example/v1",
        "credential_references": {"primary": "secret://secret-1"},
        "auth_header_templates": [
            {
                "name": "Authorization",
                "template": "Bearer {token}",
                "references": {"token": "primary"},
            }
        ],
        "connection_config": {
            "network": {
                "allowed_hosts": ["artifacts.example"],
                "allowed_ports": [8443],
            }
        },
    }
    values.update(overrides)
    return SimpleNamespace(**values)


class FakeWebSocket:
    def __init__(
        self,
        *,
        headers: dict[str, str] | None = None,
        query: str = "",
        path: str = "/v1/session",
        incoming: list[dict] | None = None,
    ) -> None:
        raw_headers = [
            (name.lower().encode(), value.encode())
            for name, value in (headers or {}).items()
        ]
        self.scope = {
            "type": "websocket",
            "asgi": {"version": "3.0"},
            "http_version": "1.1",
            "scheme": "wss",
            "path": path,
            "raw_path": path.encode(),
            "query_string": query.encode(),
            "headers": raw_headers,
            "client": ("203.0.113.8", 45000),
            "server": ("socket.example", 443),
            "state": {},
        }
        self.headers = Headers(raw=raw_headers)
        self.query_params = QueryParams(query)
        self.url = URL(f"wss://socket.example{path}?{query}")
        self.client = Address("203.0.113.8", 45000)
        self.incoming = deque(incoming or [])
        self.sent_text: list[str] = []
        self.sent_bytes: list[bytes] = []
        self.accepted: dict | None = None
        self.closed: tuple[int, str] | None = None

    async def receive(self):
        if self.incoming:
            return self.incoming.popleft()
        return {"type": "websocket.disconnect", "code": 1001}

    async def send_text(self, value: str):
        self.sent_text.append(value)

    async def send_bytes(self, value: bytes):
        self.sent_bytes.append(value)

    async def accept(self, subprotocol=None, headers=None):
        self.accepted = {"subprotocol": subprotocol, "headers": headers or []}

    async def close(self, code=1000, reason=""):
        self.closed = (code, reason)


def test_build_websocket_profile_is_data_driven_and_secret_backed():
    profile = realtime.build_websocket_profile(_account(), _route())

    assert profile.base_url == "wss://gateway.example/v1"
    assert profile.path_template == "/sessions/{model}"
    assert profile.allowed_request_headers == {
        "authorization",
        "cookie",
        "x-forwarded-for",
        "x-feature",
    }
    assert profile.allowed_subprotocols == ("events.v1",)
    assert profile.network.allowed_hosts == (
        "artifacts.example",
        "gateway.example",
    )
    assert profile.network.allowed_ports == {443, 8443}
    assert profile.network.allow_private_networks is False
    assert profile.secret_headers[0].references == {"token": "secret://secret-1"}


def test_websocket_profile_rejects_truthy_non_boolean_insecure_transport_flag():
    account = _account(
        base_url="http://gateway.example/v1",
        connection_config={"allow_insecure_http": "false"},
    )

    with pytest.raises(
        realtime.ExternalConfigurationError,
        match="allow_insecure_http must be a boolean",
    ):
        realtime.build_websocket_profile(account, _route())


def test_admin_compiler_accepts_realtime_profile_and_rejects_config_typos():
    configured = _route()
    validate_route_configuration(
        _account(),
        configured,
        cord={
            "path": "/session",
            "function": "session",
            "input_schema": {},
            "minimal_input_schema": {},
            "output_schema": {},
        },
        pricing_rules=({"metric": "request", "unit_price": "0.1"},),
    )

    configured.operation_config["realtime"]["validate_client_messages"] = "yes"
    with pytest.raises(RouteConfigurationError, match="must be a boolean"):
        validate_route_configuration(_account(), configured)


@pytest.mark.parametrize(
    "field",
    [
        "allow_client_binary",
        "allow_client_non_json_text",
        "allow_upstream_binary",
        "allow_upstream_non_json_text",
    ],
)
def test_admin_compiler_requires_boolean_realtime_opaque_frame_opt_ins(field):
    configured = _route()
    configured.operation_config["realtime"][field] = "yes"

    with pytest.raises(
        RouteConfigurationError, match=rf"realtime\.{field} must be a boolean"
    ):
        validate_route_configuration(_account(), configured)


def _peer(websocket: FakeWebSocket, route: ExternalRouteConfig | None = None):
    return realtime.StarletteWebSocketPeer(
        websocket,
        route=route or _route(),
        cord={"input_schema": {}},
        chute_name="public-model",
        invocation_id="local-invocation",
        operation_id="local-operation",
    )


def _route_with_realtime_flags(**flags: bool) -> ExternalRouteConfig:
    route = _route()
    operation = dict(route.operation_config)
    endpoint = dict(operation["realtime"])
    endpoint.update(flags)
    operation["realtime"] = endpoint
    return route.model_copy(update={"operation_config": operation})


@pytest.mark.anyio
async def test_realtime_message_schema_is_metadata_only_without_explicit_opt_in(
    monkeypatch,
):
    route = _route(
        input_schema={
            "type": "object",
            "required": ["prompt"],
        }
    )
    validate = AsyncMock(side_effect=AssertionError("schema must remain unused"))
    monkeypatch.setattr(realtime, "_validate_message_schema", validate)
    peer = _peer(
        FakeWebSocket(
            incoming=[{"type": "websocket.receive", "text": "{}"}],
        ),
        route,
    )

    frame = await peer.receive()

    assert frame.kind is WebSocketFrameType.TEXT
    assert orjson.loads(frame.data) == {"model": "resource-v1"}
    assert peer.client_rejected is False
    validate.assert_not_awaited()


@pytest.mark.anyio
async def test_realtime_disabled_validation_does_not_inspect_schema_metadata():
    route = _route()
    route.operation_config["realtime"]["message_schema"] = "not-a-schema-object"
    route.operation_config["realtime"]["validate_client_messages"] = False
    peer = realtime.StarletteWebSocketPeer(
        FakeWebSocket(
            incoming=[{"type": "websocket.receive", "text": '{"prompt":"ok"}'}],
        ),
        route=route,
        cord={"input_schema": "also-not-a-schema-object"},
        chute_name="public-model",
        invocation_id="local-invocation",
        operation_id="local-operation",
    )

    frame = await peer.receive()

    assert frame.kind is WebSocketFrameType.TEXT
    assert orjson.loads(frame.data) == {
        "model": "resource-v1",
        "prompt": "ok",
    }
    assert peer.client_rejected is False


def test_realtime_explicit_validation_requires_a_nonempty_schema():
    route = _route_with_realtime_flags(validate_client_messages=True)

    with pytest.raises(
        realtime.ExternalConfigurationError,
        match="requires a non-empty message schema",
    ):
        _peer(FakeWebSocket(), route)


@pytest.mark.anyio
@pytest.mark.parametrize("schema_source", ["realtime", "cord"])
async def test_realtime_message_validation_requires_explicit_opt_in(schema_source):
    schema = {
        "type": "object",
        "properties": {"prompt": {"type": "string"}},
        "required": ["prompt"],
    }
    route = _route(input_schema=schema if schema_source == "realtime" else None)
    route.operation_config["realtime"]["validate_client_messages"] = True
    cord = {"input_schema": schema if schema_source == "cord" else {}}
    peer = realtime.StarletteWebSocketPeer(
        FakeWebSocket(
            incoming=[{"type": "websocket.receive", "text": "{}"}],
        ),
        route=route,
        cord=cord,
        chute_name="public-model",
        invocation_id="local-invocation",
        operation_id="local-operation",
    )

    frame = await peer.receive()

    assert frame.kind is WebSocketFrameType.CLOSE
    assert frame.close_code == 1008
    assert peer.client_rejected is True


@pytest.mark.anyio
async def test_realtime_empty_message_schema_falls_back_to_cord_schema():
    schema = {
        "type": "object",
        "properties": {"prompt": {"type": "string"}},
        "required": ["prompt"],
    }
    route = _route()
    route.operation_config["realtime"].update(
        {"validate_client_messages": True, "message_schema": {}}
    )
    cord = {"input_schema": schema}

    validate_route_configuration(_account(), route, cord=cord)
    peer = realtime.StarletteWebSocketPeer(
        FakeWebSocket(incoming=[{"type": "websocket.receive", "text": "{}"}]),
        route=route,
        cord=cord,
        chute_name="public-model",
        invocation_id="local-invocation",
        operation_id="local-operation",
    )

    frame = await peer.receive()

    assert frame.kind is WebSocketFrameType.CLOSE
    assert frame.close_code == 1008
    assert peer.client_rejected is True


@pytest.mark.anyio
@pytest.mark.parametrize(
    ("message", "flag"),
    [
        ({"type": "websocket.receive", "text": "opaque"}, "allow_client_non_json_text"),
        ({"type": "websocket.receive", "bytes": b"opaque"}, "allow_client_binary"),
    ],
)
async def test_peer_denies_opaque_client_frames_by_default_and_requires_opt_in(
    message, flag
):
    denied = _peer(FakeWebSocket(incoming=[message]))
    denied_frame = await denied.receive()

    assert denied_frame.kind is WebSocketFrameType.CLOSE
    assert denied_frame.close_code == 1008
    assert denied.client_rejected is True

    allowed = _peer(
        FakeWebSocket(incoming=[message]),
        _route_with_realtime_flags(**{flag: True}),
    )
    allowed_frame = await allowed.receive()

    expected_kind = (
        WebSocketFrameType.BINARY if "bytes" in message else WebSocketFrameType.TEXT
    )
    assert allowed_frame.kind is expected_kind
    assert allowed_frame.data == message.get("bytes", message.get("text"))
    assert allowed.client_rejected is False


@pytest.mark.anyio
@pytest.mark.parametrize(
    ("frame", "flag"),
    [
        (
            WebSocketFrame(WebSocketFrameType.TEXT, "opaque"),
            "allow_upstream_non_json_text",
        ),
        (
            WebSocketFrame(WebSocketFrameType.BINARY, b"opaque"),
            "allow_upstream_binary",
        ),
    ],
)
async def test_peer_denies_opaque_upstream_frames_by_default_and_requires_opt_in(
    frame, flag
):
    denied_socket = FakeWebSocket()
    denied = _peer(denied_socket)

    with pytest.raises(realtime.MappingExtractionError, match="opaque upstream"):
        await denied.send(frame)
    assert denied_socket.sent_text == []
    assert denied_socket.sent_bytes == []

    allowed_socket = FakeWebSocket()
    allowed = _peer(
        allowed_socket,
        _route_with_realtime_flags(**{flag: True}),
    )
    await allowed.send(frame)

    if frame.kind is WebSocketFrameType.TEXT:
        assert allowed_socket.sent_text == ["opaque"]
        assert allowed_socket.sent_bytes == []
    else:
        assert allowed_socket.sent_text == []
        assert allowed_socket.sent_bytes == [b"opaque"]


@pytest.mark.anyio
async def test_peer_pins_client_resource_and_scrubs_upstream_json():
    route = _route(
        input_schema={
            "type": "object",
            "properties": {"model": {"type": "string"}},
            "required": ["model"],
        }
    ).model_copy(
        update={
            "response_config": {
                "public": {
                    "rewrite_keys": {
                        "model": "configured-private-model",
                        "request_id": "configured-private-request",
                        "operation_id": "configured-private-operation",
                    }
                }
            }
        }
    )
    websocket = FakeWebSocket(
        incoming=[
            {
                "type": "websocket.receive",
                "text": '{"model":"client-selected","prompt":"hello"}',
            }
        ]
    )
    peer = realtime.StarletteWebSocketPeer(
        websocket,
        route=route,
        cord={"input_schema": {}},
        chute_name="public-model",
        invocation_id="local-invocation",
        operation_id="local-operation",
    )

    outbound = await peer.receive()
    assert outbound.kind is WebSocketFrameType.TEXT
    assert orjson.loads(outbound.data) == {
        "model": "resource-v1",
        "prompt": "hello",
    }

    await peer.send(
        WebSocketFrame(
            WebSocketFrameType.TEXT,
            orjson.dumps(
                {
                    "provider": "private-name",
                    "model": "resource-v1",
                    "request_id": "remote-request",
                    "operation_id": "remote-operation",
                    "output": "hello",
                }
            ).decode(),
        )
    )
    assert orjson.loads(websocket.sent_text[0]) == {
        "model": "public-model",
        "request_id": "local-invocation",
        "operation_id": "local-operation",
        "output": "hello",
    }


def test_realtime_common_resource_fields_are_server_owned_without_explicit_path():
    route = _route().model_copy(
        update={
            "request_config": {
                "transform": {"inject": {"model": "client-cannot-replace-this"}}
            }
        }
    )

    transformed = realtime._request_message_transform(
        route,
        {
            "model": "client-selected",
            "resource": "another-resource",
            "prompt": "hello",
        },
        invocation_id="local-invocation",
        chute_name="public-model",
    )

    assert transformed == {
        "model": "resource-v1",
        "resource": "resource-v1",
        "prompt": "hello",
    }


def test_realtime_query_mapping_strips_secrets_and_pins_resource():
    route = _route().model_copy(
        update={
            "request_config": {
                "allowed_query_parameters": [
                    "language",
                    "model",
                    "deployment",
                    "token",
                ],
                "query_parameters": {
                    "locale": "query.language",
                    "model": "query.model",
                    "fixed": {"value": "server-value"},
                },
                "resource_query_parameter": "deployment",
            }
        }
    )
    websocket = FakeWebSocket(
        query=(
            "language=en&model=client-model&deployment=client-deployment&token=secret"
        )
    )

    allowed = realtime._allowed_query(websocket, route)
    assert allowed == {
        "language": "en",
        "model": "client-model",
        "deployment": "client-deployment",
    }
    assert realtime._upstream_query_parameters(route, allowed) == {
        "language": "en",
        "locale": "en",
        "model": "resource-v1",
        "deployment": "resource-v1",
        "fixed": "server-value",
    }


@pytest.mark.anyio
async def test_peer_does_not_expose_custom_upstream_close_codes_or_reasons():
    websocket = FakeWebSocket()
    peer = realtime.StarletteWebSocketPeer(
        websocket,
        route=_route(),
        cord={"input_schema": {}},
        chute_name="public-model",
        invocation_id="local-invocation",
        operation_id="local-operation",
    )

    await peer.close(4321, "private upstream reason")

    assert websocket.closed == (1011, "realtime execution failed")


@pytest.mark.anyio
async def test_realtime_usage_meter_merges_deltas_but_counts_one_request():
    meter = realtime.RealtimeUsageMeter(
        _route(
            usage={
                "default_requests": 0,
                "fields": {
                    "tokens.input": {
                        "source": "request",
                        "path": "usage.input_tokens",
                    },
                    "tokens.output": {"path": "usage.output_tokens"},
                },
            },
            usage_mode="delta",
        )
    )
    await meter.observe(
        UsageObservation(
            profile_name="duplex",
            status_code=101,
            response_headers={},
            json_value={
                "prompt": "not persisted",
                "usage": {"input_tokens": 4},
            },
            direction=MessageDirection.OUTBOUND,
        )
    )
    for quantity in (2, 3):
        await meter.observe(
            UsageObservation(
                profile_name="duplex",
                status_code=101,
                response_headers={},
                json_value={"usage": {"output_tokens": quantity}},
                direction=MessageDirection.INBOUND,
            )
        )

    assert meter.usage.requests == 1
    assert meter.usage.tokens["input"] == 4
    assert meter.usage.tokens["output"] == 5


@pytest.mark.anyio
@pytest.mark.parametrize(
    ("authorization", "expects_jwt"),
    [("Bearer signed-jwt", True), ("Bearer cpk_local-token", False)],
)
async def test_auth_adapter_supports_jwt_and_api_keys(
    monkeypatch, authorization, expects_jwt
):
    user = SimpleNamespace(user_id="user-1")
    jwt_auth = AsyncMock(return_value=user if expects_jwt else None)
    key_auth = AsyncMock(
        return_value=None if expects_jwt else SimpleNamespace(user=user)
    )
    monkeypatch.setattr(realtime, "get_user_from_token", jwt_auth)
    monkeypatch.setattr(realtime, "get_and_check_api_key", key_auth)
    websocket = FakeWebSocket(headers={"Authorization": authorization})

    result, request = await realtime._authenticate(websocket, "chute-1")

    assert result is user
    assert request.state.auth_object_type == "chutes"
    assert request.state.auth_object_id == "chute-1"
    assert request.state.auth_method == "invoke"
    if expects_jwt:
        jwt_auth.assert_awaited_once()
        key_auth.assert_not_awaited()
    else:
        jwt_auth.assert_not_awaited()
        key_auth.assert_awaited_once()


@pytest.mark.anyio
async def test_disconnect_still_settles_and_never_forwards_identity_headers(
    monkeypatch,
):
    route = _route(
        usage={
            "default_requests": 0,
            "fields": {"tokens.output": {"path": "usage.output_tokens"}},
        }
    )
    account = _account()
    chute = SimpleNamespace(
        chute_id="chute-1",
        user_id="owner-1",
        name="public-model",
        public=True,
    )
    context = realtime.RealtimeContext(
        chute=chute,
        binding=SimpleNamespace(binding_id="binding-1"),
        account=account,
        route=route,
        cord={"path": "/session", "function": "session", "input_schema": {}},
    )
    websocket = FakeWebSocket(
        headers={
            "Authorization": "Bearer client-secret",
            "Cookie": "session=private",
            "X-Forwarded-For": "198.51.100.1",
            "X-Feature": "enabled",
            "Sec-WebSocket-Protocol": "events.v1",
        },
        query="language=en&token=private",
        incoming=[{"type": "websocket.disconnect", "code": 1001}],
    )
    user = SimpleNamespace(user_id="user-1")
    request = realtime._request_adapter(websocket, chute.chute_id)
    operation = SimpleNamespace(operation_id="local-operation")

    monkeypatch.setattr(realtime, "_resolve_context", AsyncMock(return_value=context))
    monkeypatch.setattr(
        realtime, "_authenticate", AsyncMock(return_value=(user, request))
    )
    monkeypatch.setattr(realtime, "resolve_rate_limit_headers", lambda *_: None)
    monkeypatch.setattr(realtime, "check_quota_and_balance", AsyncMock())
    monkeypatch.setattr(
        realtime,
        "_pricing_snapshot",
        AsyncMock(
            return_value={
                "source": "rules",
                "rules": [
                    {
                        "id": "output",
                        "metric": "token",
                        "bucket": "output",
                        "unit_price": "0.01",
                        "unit_size": "1",
                        "rounding": "none",
                        "minimum_units": "0",
                    }
                ],
            }
        ),
    )
    monkeypatch.setattr(realtime, "_validate_metering_config", lambda *_: None)
    create_operation = AsyncMock(return_value=(operation, False))
    monkeypatch.setattr(realtime, "_create_operation", create_operation)
    checkpointed = asyncio.Event()
    updates = AsyncMock()

    async def observe_update(_operation_id, **values):
        if values.get("usage", {}).get("tokens") == {"output": "7"}:
            checkpointed.set()

    updates.side_effect = observe_update
    settlement = AsyncMock()
    monkeypatch.setattr(realtime, "_update_operation", updates)
    monkeypatch.setattr(realtime, "settle_operation", settlement)
    monkeypatch.setattr(
        realtime,
        "_running_budget_check",
        AsyncMock(return_value=(True, None)),
    )
    monkeypatch.setattr(realtime, "_secret_resolver", lambda _account: lambda _ref: "x")
    monkeypatch.setattr(realtime, "track_request_completed", lambda _chute_id: None)
    monkeypatch.setattr(realtime, "USAGE_CHECKPOINT_INTERVAL_SECONDS", 0.01)

    seen = {}

    class Connection:
        subprotocol = "events.v1"

        def __init__(self, hook):
            self.hook = hook

        async def relay(self, peer):
            await self.hook(
                UsageObservation(
                    profile_name="duplex",
                    status_code=101,
                    response_headers={},
                    json_value={"usage": {"output_tokens": 7}},
                    direction=MessageDirection.INBOUND,
                )
            )
            await asyncio.wait_for(checkpointed.wait(), timeout=1)
            await peer.send(
                WebSocketFrame(
                    WebSocketFrameType.TEXT,
                    '{"provider":"private","model":"resource-v1","output":"ok"}',
                )
            )
            close = await peer.receive()
            assert close.kind is WebSocketFrameType.CLOSE
            return WebSocketRelayResult(1001, "endpoint going away", 0, 1)

    class Relay:
        def __init__(self, *, secret_resolver, usage_hooks):
            del secret_resolver
            self.hook = usage_hooks[0]

        async def connect(self, profile, outbound):
            seen["profile"] = profile
            seen["outbound"] = outbound
            return Connection(self.hook)

    monkeypatch.setattr(realtime, "WebSocketRelay", Relay)

    await realtime.handle_external_realtime(websocket)

    assert websocket.accepted is not None
    assert create_operation.await_args.kwargs["idempotency_key"] is None
    assert create_operation.await_args.kwargs["idempotency_fingerprint"] == ""
    assert seen["outbound"].headers == {"x-feature": "enabled"}
    assert seen["outbound"].query == {"language": "en"}
    assert "authorization" not in seen["outbound"].headers
    assert "cookie" not in seen["outbound"].headers
    assert "x-forwarded-for" not in seen["outbound"].headers
    assert orjson.loads(websocket.sent_text[0]) == {
        "model": "public-model",
        "output": "ok",
    }
    settlement.assert_awaited_once()
    assert settlement.await_args.args[0] == "local-operation"
    assert settlement.await_args.args[1].tokens["output"] == 7
    assert settlement.await_args.kwargs == {"billable": True}
    running = next(
        call
        for call in updates.await_args_list
        if call.kwargs.get("status") == "running"
    )
    assert running.kwargs["expires_at"] == running.kwargs["next_poll_at"]
    assert (
        running.kwargs["expires_at"] - running.kwargs["started_at"]
    ).total_seconds() == 3660
    assert running.kwargs["usage"]["requests"] == "1"
    usage_checkpoints = [
        call
        for call in updates.await_args_list
        if call.kwargs.get("usage", {}).get("tokens") == {"output": "7"}
        and "started_at" not in call.kwargs
    ]
    assert len(usage_checkpoints) >= 1
    assert usage_checkpoints[0].kwargs["usage"]["tokens"] == {"output": "7"}
    assert any(
        call.kwargs.get("status") == "cancelled" for call in updates.await_args_list
    )
    final_update = next(
        call
        for call in updates.await_args_list
        if call.kwargs.get("status") == "cancelled"
    )
    assert final_update.kwargs["_settlement_metadata_patch"] == {"billable": True}


def test_router_exposes_a_catch_all_websocket_route():
    paths = [route.path for route in realtime.router.routes]
    assert "/{path:path}" in paths


@pytest.mark.anyio
async def test_hostname_and_canonical_api_websockets_resolve_the_same_cord(monkeypatch):
    route = _route()
    account = _account(enabled=True)
    binding = SimpleNamespace(
        binding_id="binding-1",
        enabled=True,
        account=account,
        routes=[route.model_dump(mode="json")],
    )
    chute = SimpleNamespace(
        chute_id="chute-1",
        user_id="owner-1",
        name="public-model",
        execution_backend="external",
        disabled=False,
        external_binding=binding,
        cords=[
            {
                "path": "/session",
                "function": "session",
                "public_api_path": "/session",
                "public_api_method": "GET",
            }
        ],
    )

    class Result:
        def unique(self):
            return self

        def scalar_one_or_none(self):
            return chute

    class Session:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            return None

        async def execute(self, _statement):
            return Result()

    slug_lookup = AsyncMock(return_value="chute-1")
    monkeypatch.setattr(realtime, "chute_id_by_slug", slug_lookup)
    monkeypatch.setattr(realtime, "get_session", lambda **_kwargs: Session())

    hostname = await realtime._resolve_context(
        FakeWebSocket(headers={"Host": "public-model.example.test"}, path="/session")
    )
    canonical = await realtime._resolve_context(
        FakeWebSocket(
            headers={"Host": "api.example.test"},
            path="/chutes/chute-1/session",
        )
    )

    assert hostname.chute is chute
    assert canonical.chute is chute
    assert hostname.route == canonical.route == route
    assert hostname.cord == canonical.cord == chute.cords[0]
    slug_lookup.assert_awaited_once_with("public-model")
