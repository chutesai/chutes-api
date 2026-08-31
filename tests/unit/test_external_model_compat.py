from contextlib import asynccontextmanager
from datetime import UTC, datetime
import json
import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import HTTPException
from starlette.requests import Request


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

from api.api_key.util import OAuthTokenWrapper
from api.chute.standard_templates import standard_template_matches

with patch("ctypes.CDLL", return_value=MagicMock()):
    import api.database.orms  # noqa: F401
    from api import model_routing

from api.external_backend.config import select_route
from api.external_backend.model_compat import (
    credential_allows_chute,
    external_chute_version,
    external_llm_model_details,
    public_charge_line_items,
    public_pricing_rules,
    routed_fallback_allowed,
    routed_model_payload,
    select_external_cord,
)


def _cord(path: str, *, stream: bool) -> dict:
    return {
        "path": path,
        "public_api_path": "/v1/chat/completions",
        "public_api_method": "POST",
        "stream": stream,
    }


def _route(path: str, *, method: str = "GET") -> dict:
    return {
        "cord_path": path,
        "upstream_resource_id": "resource-id",
        "operation_mode": "sync",
        "protocol": "generic-json",
        "path_template": "/upstream/chat",
        "method": method,
    }


def test_stream_flag_selects_between_shared_public_routes():
    nonstream = _cord("/chat", stream=False)
    streaming = _cord("/chat_stream", stream=True)
    cords = [nonstream, streaming]

    assert (
        select_external_cord(
            cords,
            public_path="/v1/chat/completions",
            method="POST",
            stream=False,
        )
        is nonstream
    )
    assert (
        select_external_cord(
            cords,
            public_path="/v1/chat/completions",
            method="post",
            stream="true",
        )
        is streaming
    )


def test_stream_selection_fails_closed_when_cords_are_ambiguous():
    cords = [_cord("/chat-a", stream=True), _cord("/chat-b", stream=True)]

    assert (
        select_external_cord(
            cords,
            public_path="/v1/chat/completions",
            method="POST",
            stream=True,
        )
        is None
    )


def test_upstream_method_is_independent_of_public_cord_method():
    cord = _cord("/chat", stream=False)

    selected = select_route([_route("/chat", method="GET")], cord)

    assert cord["public_api_method"] == "POST"
    assert selected.method == "GET"


def test_external_model_catalog_entry_is_provider_neutral():
    created_at = datetime(2026, 1, 2, 3, 4, tzinfo=UTC)

    details = external_llm_model_details(
        chute_id="chute-id", name="public/model", created_at=created_at
    )

    assert details == {
        "id": "public/model",
        "object": "model",
        "created": int(created_at.timestamp()),
        "owned_by": "chutes",
        "root": "public/model",
        "parent": None,
        "chute_id": "chute-id",
        "confidential_compute": False,
    }


def test_routed_model_payload_is_pinned_without_mutating_fallback_source():
    original = {"model": "user-alias", "messages": [{"role": "user"}]}

    first = routed_model_payload(original, "model-a")
    first["messages"][0]["role"] = "mutated-by-first-attempt"
    fallback = routed_model_payload(original, "model-b")

    assert original["model"] == "user-alias"
    assert original["messages"][0]["role"] == "user"
    assert fallback == {"model": "model-b", "messages": [{"role": "user"}]}


def test_embedding_endpoint_preserves_hosted_template_match_and_external_tei_compatibility():
    assert standard_template_matches("embedding", "embedding")
    assert not standard_template_matches("tei", "embedding")
    assert standard_template_matches("tei", "embedding", execution_backend="external")
    assert not standard_template_matches("vllm", "embedding")


def test_external_chute_version_covers_standard_template():
    values = {
        "account_id": "account-id",
        "cords": [_cord("/chat", stream=False)],
        "routes": [_route("/chat")],
        "pricing_rules": [{"metric": "request", "unit_price": "0.1"}],
    }

    untyped = external_chute_version(standard_template=None, **values)
    llm = external_chute_version(standard_template="vllm", **values)

    assert untyped != llm
    assert llm == external_chute_version(standard_template="vllm", **values)


def test_resolved_chute_is_rechecked_against_mega_route_credential():
    class Credential:
        def has_access(self, object_type, object_id, action):
            return (object_type, object_id, action) == (
                "chutes",
                "allowed-chute",
                "invoke",
            )

    credential = Credential()

    assert credential_allows_chute(credential, "allowed-chute")
    assert not credential_allows_chute(credential, "different-chute")
    assert credential_allows_chute(None, "jwt-authenticated-chute")


def test_central_routing_does_not_retry_after_a_billable_upstream_attempt():
    assert routed_fallback_allowed(429, attempt_billable=False)
    assert routed_fallback_allowed(503, attempt_billable=False)
    assert not routed_fallback_allowed(429, attempt_billable=True)
    assert not routed_fallback_allowed(503, attempt_billable=True)
    assert not routed_fallback_allowed(502, attempt_billable=False)


def test_oauth_specific_chute_scope_is_provisionally_admitted_to_mega_route():
    credential = OAuthTokenWrapper(object(), ["chutes:external-chute-id:invoke"])

    assert credential.has_access("chutes", "__megallm__", "invoke")
    assert credential.has_access("chutes", "external-chute-id", "invoke")
    assert not credential.has_access("chutes", "other-chute", "invoke")


@pytest.mark.asyncio
async def test_exact_external_resolution_keeps_candidates_for_common_access_filter(
    monkeypatch,
):
    private_candidate = SimpleNamespace(standard_template="vllm")
    public_candidate = SimpleNamespace(standard_template="vllm")

    class Result:
        def unique(self):
            return self

        def scalars(self):
            return self

        def all(self):
            return [private_candidate, public_candidate]

    class Session:
        async def execute(self, statement):
            query = str(statement)
            assert "external_chute_bindings" in query
            assert "external_backend_accounts" in query
            assert "chutes.disabled IS false" in query
            assert "external_chute_bindings.enabled IS true" in query
            assert "external_backend_accounts.enabled IS true" in query
            return Result()

    @asynccontextmanager
    async def fake_session(*, readonly=False):
        assert readonly is True
        yield Session()

    monkeypatch.setattr(model_routing, "get_session", fake_session)

    assert await model_routing.resolve_exact_external_models(
        "exact-name", "requesting-user", "vllm"
    ) == [private_candidate, public_candidate]


@pytest.mark.asyncio
async def test_exact_external_resolution_drops_a_disabled_stale_candidate(monkeypatch):
    enabled = SimpleNamespace(standard_template="vllm", disabled=False)
    disabled = SimpleNamespace(standard_template="vllm", disabled=True)

    class Result:
        def unique(self):
            return self

        def scalars(self):
            return self

        def all(self):
            return [disabled, enabled]

    class Session:
        async def execute(self, _statement):
            return Result()

    @asynccontextmanager
    async def fake_session(*, readonly=False):
        assert readonly is True
        yield Session()

    monkeypatch.setattr(model_routing, "get_session", fake_session)

    assert await model_routing.resolve_exact_external_models(
        "exact-name", "requesting-user", "vllm"
    ) == [enabled]


@pytest.mark.asyncio
async def test_backend_constrained_resolution_can_fall_through_to_hosted(monkeypatch):
    hosted = SimpleNamespace(standard_template="vllm")
    calls = []

    async def get_candidate(name, *, user_id, execution_backend=None):
        calls.append((name, user_id, execution_backend))
        return hosted

    monkeypatch.setattr(model_routing, "_get_routing_chute", get_candidate)

    resolved, mode = await model_routing.resolve_model_parameter(
        "shared-name",
        "requesting-user",
        "vllm",
        execution_backend="hosted",
    )

    assert resolved == [hosted]
    assert mode is None
    assert calls == [("shared-name", "requesting-user", "hosted")]


@pytest.mark.asyncio
async def test_common_exact_access_filter_includes_subnet_visible_candidate(
    monkeypatch,
):
    from api.invocation import router as invocation_router

    candidate = SimpleNamespace(
        chute_id="subnet-visible",
        user_id="different-owner",
        public=False,
        standard_template="vllm",
    )
    user = SimpleNamespace(user_id="requesting-user")
    monkeypatch.setattr(invocation_router, "is_shared", lambda *_args: _false())
    monkeypatch.setattr(
        invocation_router,
        "subnet_role_accessible",
        lambda chute, _user: chute.chute_id == "subnet-visible",
    )

    assert await invocation_router._accessible_routed_chutes(
        [candidate],
        current_user=user,
        template="vllm",
        credential=None,
    ) == [candidate]


async def _false():
    return False


def test_external_mega_guardrails_strip_privileged_fields_before_forwarding():
    from api.invocation import router as invocation_router

    payload = {
        "model": "public-model",
        "logprobs": True,
        "top_logprobs": 5,
        "cache_salt": "private-cache-namespace",
    }
    user = SimpleNamespace(has_role=lambda _permission: False)
    chute = SimpleNamespace(name="public-model")

    invocation_router._apply_mega_request_guardrails(payload, user, chute)

    assert payload == {"model": "public-model"}


def test_external_mega_guardrails_reject_file_and_regex_content():
    from api.invocation import router as invocation_router

    user = SimpleNamespace(has_role=lambda _permission: False)
    chute = SimpleNamespace(name="public-model")
    with pytest.raises(HTTPException, match="File content"):
        invocation_router._apply_mega_request_guardrails(
            {
                "messages": [
                    {"role": "user", "content": [{"file_url": "https://files.test/a"}]}
                ]
            },
            user,
            chute,
        )
    with pytest.raises(HTTPException, match="Regex-based"):
        invocation_router._apply_mega_request_guardrails(
            {"guided_regex": "secret.*"},
            user,
            chute,
        )


@pytest.mark.asyncio
async def test_external_mega_guardrails_ignore_untrusted_content_type(monkeypatch):
    from api.invocation import router as invocation_router

    payload = {
        "model": "public-model",
        "messages": [
            {"role": "user", "content": [{"file_url": "https://files.test/a"}]}
        ],
    }
    raw_body = json.dumps(payload).encode("utf-8")
    messages = [{"type": "http.request", "body": raw_body, "more_body": False}]

    async def receive():
        return messages.pop(0) if messages else {"type": "http.disconnect"}

    request = Request(
        {
            "type": "http",
            "http_version": "1.1",
            "method": "POST",
            "scheme": "https",
            "path": "/v1/chat/completions",
            "raw_path": b"/v1/chat/completions",
            "query_string": b"",
            "headers": [
                (b"host", b"llm.chutes.ai"),
                (b"content-type", b"text/plain"),
            ],
            "client": ("127.0.0.1", 12345),
            "server": ("llm.chutes.ai", 443),
        },
        receive,
    )
    request.state.chute_id = "external-id"
    request.state.mega_request = True
    request.state.invocation_public_path = "/v1/chat/completions"
    chute = SimpleNamespace(
        chute_id="external-id",
        user_id="owner-id",
        public=True,
        disabled=False,
        tee=False,
        execution_backend="external",
        standard_template="vllm",
        name="public-model",
        cords=[_cord("/chat", stream=False)],
    )
    user = SimpleNamespace(user_id="requesting-user", has_role=lambda _role: False)
    dispatch = AsyncMock()
    monkeypatch.setattr(invocation_router, "get_one", AsyncMock(return_value=chute))
    monkeypatch.setattr(invocation_router, "resolve_rate_limit_headers", MagicMock())
    monkeypatch.setattr(invocation_router, "check_quota_and_balance", AsyncMock())
    monkeypatch.setattr(invocation_router, "invoke_external_resilient", dispatch)

    with pytest.raises(HTTPException, match="File content"):
        await invocation_router._invoke(request, user)

    dispatch.assert_not_awaited()


def _mega_request(
    payload: dict, *, content_type: bytes = b"application/json"
) -> Request:
    raw_body = json.dumps(payload).encode("utf-8")
    messages = [{"type": "http.request", "body": raw_body, "more_body": False}]

    async def receive():
        return messages.pop(0) if messages else {"type": "http.disconnect"}

    request = Request(
        {
            "type": "http",
            "http_version": "1.1",
            "method": "POST",
            "scheme": "https",
            "path": "/v1/chat/completions",
            "raw_path": b"/v1/chat/completions",
            "query_string": b"",
            "headers": [
                (b"host", b"llm.chutes.ai"),
                (b"content-type", content_type),
            ],
            "client": ("127.0.0.1", 12345),
            "server": ("llm.chutes.ai", 443),
        },
        receive,
    )
    request.state.chute_id = "__megallm__"
    request.state.invocation_public_path = "/v1/chat/completions"
    return request


@pytest.mark.asyncio
async def test_centrally_routed_hosted_mega_happy_path_rewrites_body_as_bytes(
    monkeypatch,
):
    from api.invocation import router as invocation_router

    request = _mega_request(
        {"model": "public-alias", "messages": [{"role": "user", "content": "hi"}]}
    )
    chute = SimpleNamespace(
        chute_id="hosted-id",
        user_id="owner-id",
        public=True,
        execution_backend="hosted",
        standard_template="vllm",
        name="canonical-hosted-model",
    )
    user = SimpleNamespace(user_id="requesting-user")
    dispatch = AsyncMock(return_value=object())
    monkeypatch.setattr(
        model_routing, "resolve_exact_external_models", AsyncMock(return_value=[])
    )
    monkeypatch.setattr(
        model_routing,
        "resolve_model_parameter",
        AsyncMock(return_value=([chute], None)),
    )
    monkeypatch.setattr(invocation_router, "_invoke", dispatch)

    response = await invocation_router.hostname_invocation(request, user)

    assert response is dispatch.return_value
    assert isinstance(request._body, bytes)
    assert json.loads(request._body) == {
        "model": "canonical-hosted-model",
        "messages": [{"role": "user", "content": "hi"}],
    }
    dispatch.assert_awaited_once_with(request, user)


@pytest.mark.asyncio
async def test_centrally_routed_external_mega_happy_path_persists_guarded_body_bytes(
    monkeypatch,
):
    from api.invocation import router as invocation_router

    request = _mega_request(
        {
            "model": "public-model",
            "messages": [{"role": "user", "content": "hi"}],
            "logprobs": True,
            "top_logprobs": 5,
            "cache_salt": "private-cache-namespace",
        },
        content_type=b"text/plain",
    )
    chute = SimpleNamespace(
        chute_id="external-id",
        user_id="owner-id",
        public=True,
        disabled=False,
        tee=False,
        execution_backend="external",
        standard_template="vllm",
        name="public-model",
        cords=[_cord("/chat", stream=False)],
    )
    user = SimpleNamespace(
        user_id="requesting-user", has_role=lambda _permission: False
    )
    dispatch = AsyncMock(return_value=object())
    monkeypatch.setattr(
        model_routing,
        "resolve_exact_external_models",
        AsyncMock(return_value=[chute]),
    )
    monkeypatch.setattr(invocation_router, "get_one", AsyncMock(return_value=chute))
    monkeypatch.setattr(invocation_router, "resolve_rate_limit_headers", MagicMock())
    monkeypatch.setattr(invocation_router, "check_quota_and_balance", AsyncMock())
    monkeypatch.setattr(invocation_router, "invoke_external_resilient", dispatch)

    response = await invocation_router.hostname_invocation(request, user)

    assert response is dispatch.return_value
    assert isinstance(request._body, bytes)
    assert json.loads(request._body) == {
        "model": "public-model",
        "messages": [{"role": "user", "content": "hi"}],
    }
    dispatch.assert_awaited_once()


def test_public_pricing_projection_replaces_operator_identifiers():
    rules = [
        {
            "id": "private-service-video-tier",
            "metric": "output_media_second",
            "unit_price": "0.25",
            "match_group": "private-service-resolution-group",
            "priority": 20,
        }
    ]
    line_items = [
        {
            "rule_id": "private-service-video-tier",
            "metric": "output_media_second",
            "bucket": "video",
            "quantity": "5",
            "unit_price": "0.25",
            "amount": "1.25",
            "internal": "not-public",
        }
    ]

    assert public_pricing_rules(rules) == [
        {
            "id": "price-rule-1",
            "metric": "output_media_second",
            "unit_price": "0.25",
            "match_group": "price-group-1",
            "priority": 20,
        }
    ]
    assert public_charge_line_items(line_items) == [
        {
            "rule_id": "charge-line-1",
            "metric": "output_media_second",
            "bucket": "video",
            "quantity": "5",
            "unit_price": "0.25",
            "amount": "1.25",
        }
    ]
