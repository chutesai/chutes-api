import asyncio
import ctypes
import sys
from contextlib import asynccontextmanager
from datetime import UTC, datetime, timedelta
from types import ModuleType
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
from fastapi import HTTPException, Response
from starlette.requests import Request

# These tests exercise the external invocation service, not the optional chain
# client. The development environment may contain incompatible implementations
# of that client's codec namespace, so isolate the unit under test from it.
_optional_modules = (
    "async_substrate_interface",
    "async_substrate_interface.async_substrate",
    "async_substrate_interface.sync_substrate",
    "scalecodec.utils.ss58",
)
_original_optional_modules = {name: sys.modules.get(name) for name in _optional_modules}
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


class _FakeNativeFunction:
    argtypes = None
    restype = None

    def __call__(self, *_args, **_kwargs):
        return 0


class _FakeNativeLibrary:
    def __getattr__(self, _name):
        return _FakeNativeFunction()


_original_cdll = ctypes.CDLL
ctypes.CDLL = lambda _path: _FakeNativeLibrary()  # type: ignore[assignment]

import api.database.orms  # noqa: E402,F401

from api.external_backend.schemas import ExternalRouteConfig
from api.external_backend.service import (
    ExternalInvocationError,
    _bill_ambiguous_transport_failure,
    _billable_http_statuses,
    _bill_partial_stream,
    _dispatch_recovery_seconds,
    _extract_initial_stream_usage,
    _extract_stream_observation_usage,
    _finish_accepted_task_handoff,
    _finalize_interrupted_invocation,
    _handle_buffered,
    _idempotency_fingerprint,
    _idempotency_key,
    _path_parameters,
    _public_mapping,
    _raw_stream_producer,
    _rebuild_multipart_body,
    _record_free_invocation_usage,
    _record_settlement_failure,
    _replace_scalar,
    _request_transform,
    _schema_for_output,
    _schema_for_request,
    _schema_request_value,
    _spawn,
    _sse_stream_producer,
    _task_timeout_seconds,
    _terminate_stream_consumer,
    _transport_snapshot_matches,
    _upstream_query_parameters,
    _update_operation,
    _validate_body_transform_mode,
    _validate_metering_config,
    _validate_raw_stream_usage,
    _validate_retry_billing_policy,
    _validate_task_submission_contract,
    _validated_output_content_type,
    invoke_external,
    settle_operation,
    shutdown_external_invocations,
)
from api.external_backend.mapping import merge_stream_usage
from api.external_transport import (
    BodyMode,
    BufferedResponse,
    JsonBody,
    MultipartBody,
    MultipartPart,
    RequestRejectedError,
    ResponseMode,
    SSEEvent,
)
from api.external_backend.config import ExternalConfigurationError
from api.external_backend.schemas import ExternalSettlementStatus
from api.external_backend.billing_outbox import ExternalUsageDeliveryReceipt
from api.payment.pricing import NormalizedUsage, PricingConfigurationError

ctypes.CDLL = _original_cdll
for _module_name, _original_module in _original_optional_modules.items():
    if _original_module is None:
        sys.modules.pop(_module_name, None)
    else:
        sys.modules[_module_name] = _original_module


def _route(**overrides) -> ExternalRouteConfig:
    values = {
        "cord_path": "/generate",
        "upstream_resource_id": "resource-v1",
        "operation_mode": "sync",
        "protocol": "generic-json",
        "path_template": "/generate/{resource}",
        "method": "POST",
        "request_config": {"body_mode": "json"},
        "response_config": {},
        "operation_config": {},
    }
    values.update(overrides)
    return ExternalRouteConfig.model_validate(values)


def test_transport_snapshot_requires_exact_locked_account_and_binding_version():
    now = datetime.now(UTC)
    account = SimpleNamespace(
        account_id="account-id",
        user_id="owner-id",
        base_url="https://gateway.example.test",
        credential_references={"access": "secret://one"},
        auth_header_templates=[{"name": "Authorization"}],
        connection_config={"region": "test"},
        enabled=True,
        updated_at=now,
    )
    binding = SimpleNamespace(
        binding_id="binding-id",
        chute_id="chute-id",
        account_id="account-id",
        routes=[{"cord_path": "/generate"}],
        enabled=True,
        updated_at=now,
    )
    row = {
        "account_id": "account-id",
        "account_user_id": "owner-id",
        "account_base_url": "https://gateway.example.test",
        "credential_references": {"access": "secret://one"},
        "auth_header_templates": [{"name": "Authorization"}],
        "connection_config": {"region": "test"},
        "account_enabled": True,
        "account_updated_at": now,
        "binding_id": "binding-id",
        "binding_chute_id": "chute-id",
        "binding_account_id": "account-id",
        "binding_routes": [{"cord_path": "/generate"}],
        "binding_enabled": True,
        "binding_updated_at": now,
    }

    assert _transport_snapshot_matches(row, binding, account)

    rotated = {**row, "account_updated_at": None}
    assert not _transport_snapshot_matches(rotated, binding, account)
    moved = {**row, "binding_account_id": "other-account"}
    assert not _transport_snapshot_matches(moved, binding, account)


def _request(path: str = "/generate", method: str = "POST") -> Request:
    return Request(
        {
            "type": "http",
            "http_version": "1.1",
            "method": method,
            "scheme": "https",
            "path": path,
            "raw_path": path.encode(),
            "query_string": b"",
            "headers": [],
            "client": ("203.0.113.10", 1234),
            "server": ("api.example.test", 443),
        }
    )


def test_resource_path_parameters_are_server_owned_and_have_context_defaults():
    route = _route(
        request_config={
            "body_mode": "json",
            "path_parameters": {
                "tenant": "query.tenant",
                "resource": "context.resource",
            },
        }
    )

    assert _path_parameters(route, {}, {"tenant": "tenant-a"}) == {
        "resource": "resource-v1",
        "model": "resource-v1",
        "upstream_resource_id": "resource-v1",
        "tenant": "tenant-a",
    }

    route.request_config["path_parameters"]["resource"] = "body.resource"
    with pytest.raises(RequestRejectedError, match="configured resource"):
        _path_parameters(route, {"resource": "client-selected"}, {"tenant": "tenant-a"})


def test_request_transform_pins_configured_resource_path():
    route = _route(request_config={"body_mode": "json", "resource_path": "input.model"})

    assert _request_transform(
        route,
        {"input": {"model": "client-selected", "prompt": "hello"}},
        invocation_id="invocation-id",
        chute_name="public-name",
    ) == {"input": {"model": "resource-v1", "prompt": "hello"}}

    route.request_config["resource_path"] = ""
    with pytest.raises(ExternalConfigurationError, match="resource_path"):
        _request_transform(
            route,
            {},
            invocation_id="invocation-id",
            chute_name="public-name",
        )


def test_request_transform_pins_present_common_resource_keys_without_injecting_them():
    route = _route()

    assert _request_transform(
        route,
        {"model": "client-selected", "prompt": "hello"},
        invocation_id="invocation-id",
        chute_name="public-name",
    ) == {"model": "resource-v1", "prompt": "hello"}
    assert _request_transform(
        route,
        {"prompt": "hello"},
        invocation_id="invocation-id",
        chute_name="public-name",
    ) == {"prompt": "hello"}


def test_query_mapping_is_scalar_only_and_resource_values_are_server_owned():
    route = _route(
        request_config={
            "body_mode": "json",
            "query_parameters": {
                "quality": "body.quality",
                "optional": {"path": "body.missing", "required": False},
                "tenant": {"value": "server-tenant"},
                "model": "query.model",
            },
            "resource_query_parameter": "deployment",
        }
    )
    assert _upstream_query_parameters(
        route,
        {"quality": "high"},
        {
            "client": "kept",
            "optional": "removed",
            "model": "client-model",
            "deployment": "client-deployment",
        },
    ) == {
        "client": "kept",
        "quality": "high",
        "tenant": "server-tenant",
        "model": "resource-v1",
        "deployment": "resource-v1",
    }

    route.request_config["query_parameters"]["quality"] = {"value": ["high"]}
    with pytest.raises(RequestRejectedError, match="mapped query"):
        _upstream_query_parameters(route, {}, {})

    route.request_config["query_parameters"] = {}
    with pytest.raises(RequestRejectedError, match="mapped query"):
        _upstream_query_parameters(route, {}, {"duplicate": ["one", "two"]})


def test_multipart_transform_rebuilds_real_parts_and_preserves_files():
    descriptor = {
        "filename": "input.png",
        "content_type": "image/png",
        "size_bytes": 4,
    }
    original = MultipartBody(
        (
            MultipartPart(name="model", value="client-model"),
            MultipartPart(
                name="file",
                value=b"data",
                filename="input.png",
                content_type="image/png",
            ),
            MultipartPart(name="tag", value="one"),
            MultipartPart(name="tag", value="two"),
        )
    )
    route = _route(
        request_config={
            "body_mode": "multipart",
            "resource_path": "model",
            "transform": {
                "inject": {"injected": {"value": 3}},
                "rewrite": [
                    {
                        "target": "image",
                        "source": "payload",
                        "path": "file",
                        "remove_source": True,
                    }
                ],
            },
        }
    )
    transformed = _request_transform(
        route,
        {
            "model": "client-model",
            "file": descriptor,
            "tag": ["one", "two"],
        },
        invocation_id="invocation-id",
        chute_name="public-name",
    )
    rebuilt = _rebuild_multipart_body(
        original,
        transformed,
    )

    assert [
        (part.name, part.value) for part in rebuilt.parts if part.name == "tag"
    ] == [
        ("tag", "one"),
        ("tag", "two"),
    ]
    assert next(part.value for part in rebuilt.parts if part.name == "model") == (
        "resource-v1"
    )
    assert next(part.value for part in rebuilt.parts if part.name == "injected") == "3"
    image = next(part for part in rebuilt.parts if part.name == "image")
    assert image.value == b"data"
    assert image.filename == "input.png"
    assert image.content_type == "image/png"
    assert _schema_request_value({"file": descriptor}, original) == {
        "file": "input.png"
    }


def test_raw_body_transform_is_rejected_instead_of_silently_ignored():
    route = _route(request_config={"body_mode": "raw", "resource_path": "model"})
    with pytest.raises(ExternalConfigurationError, match="not supported"):
        _validate_body_transform_mode(route, BodyMode.RAW)


def test_bodyless_cord_input_contract_uses_allowlisted_query_values():
    assert _schema_request_value(
        {},
        None,
        body_mode=BodyMode.NONE,
        query={"prompt": "hello", "count": "2"},
    ) == {"prompt": "hello", "count": "2"}


def test_idempotency_fingerprint_covers_cord_and_query_and_keys_are_bounded():
    first = _request("/generate")
    first.scope["query_string"] = b"quality=high"
    first.state.body_sha256 = "body-hash"
    second = _request("/generate")
    second.scope["query_string"] = b"quality=low"
    second.state.body_sha256 = "body-hash"

    cord = {"path": "/generate", "function": "generate"}
    assert _idempotency_fingerprint(first, cord) != _idempotency_fingerprint(
        second, cord
    )

    task_route = _route(operation_mode="task")
    keyed = _request()
    keyed.scope["headers"] = [(b"idempotency-key", b"stable-key")]
    assert _idempotency_key(keyed, task_route) == "stable-key"
    oversized = _request()
    oversized.scope["headers"] = [(b"idempotency-key", b"x" * 256)]
    with pytest.raises(HTTPException) as error:
        _idempotency_key(oversized, task_route)
    assert error.value.status_code == 400


def test_ambiguous_transport_billing_requires_an_explicit_boolean():
    assert _bill_ambiguous_transport_failure(_route()) is False
    assert (
        _bill_ambiguous_transport_failure(
            _route(operation_config={"bill_ambiguous_transport_errors": True})
        )
        is True
    )
    with pytest.raises(ExternalConfigurationError, match="must be a boolean"):
        _bill_ambiguous_transport_failure(
            _route(operation_config={"bill_ambiguous_transport_errors": "yes"})
        )


def test_dispatch_recovery_deadline_covers_all_attempts_and_backoff():
    profile = SimpleNamespace(timeout=SimpleNamespace(total=42))

    assert _dispatch_recovery_seconds(profile, 3, 5) == 196
    assert _dispatch_recovery_seconds(SimpleNamespace(), 1, 0) == 360


def test_billable_error_statuses_cannot_also_be_retried():
    route = _route(
        operation_config={
            "billable_http_statuses": [429],
            "retry": {"max_attempts": 2, "statuses": [429]},
        }
    )
    statuses = _billable_http_statuses(route)
    with pytest.raises(ExternalConfigurationError, match="cannot also be retried"):
        _validate_retry_billing_policy(route, SimpleNamespace(method="GET"), statuses)


def test_raw_stream_usage_is_limited_to_configured_response_headers():
    profile = SimpleNamespace(
        allowed_response_headers=frozenset(),
        private_response_headers=frozenset({"x-output-seconds"}),
    )
    route = _route(
        operation_mode="stream",
        response_config={
            "usage": {
                "fields": {
                    "output_media_seconds.generated": {
                        "source": "response",
                        "path": "headers.x-output-seconds",
                    }
                }
            }
        },
    )
    _validate_raw_stream_usage(route, profile)

    route.response_config["usage"]["fields"]["output_media_seconds.generated"] = {
        "source": "payload",
        "path": "seconds",
    }
    with pytest.raises(ExternalConfigurationError, match="response headers"):
        _validate_raw_stream_usage(route, profile)


def test_cord_output_content_type_is_enforced_and_normalized():
    assert (
        _validated_output_content_type(
            {"content-type": "Application/JSON; charset=utf-8"},
            {"output_content_type": "application/json"},
        )
        == "application/json"
    )
    with pytest.raises(ExternalInvocationError, match="content type"):
        _validated_output_content_type(
            {"content-type": "text/plain"},
            {"output_content_type": "application/json"},
        )


def test_task_id_replacement_does_not_rewrite_numeric_counters():
    assert _replace_scalar(
        {"task_id": "7", "count": 7, "nested": ["7", 7]}, "7", "local-id"
    ) == {
        "task_id": "local-id",
        "count": 7,
        "nested": ["local-id", 7],
    }


def test_public_mapping_cannot_override_server_owned_identity_rewrites():
    route = _route(
        response_config={
            "public": {
                "rewrite_keys": {
                    "model": "upstream-model",
                    "task_id": "upstream-task",
                    "custom": "preserved",
                }
            }
        }
    )
    rules = _public_mapping(
        route,
        chute_name="public-name",
        invocation_id="invocation-id",
        operation_id="operation-id",
    )
    assert rules["rewrite_keys"] == {
        "model": "public-name",
        "task_id": "operation-id",
        "custom": "preserved",
        "model_id": "public-name",
        "model_name": "public-name",
        "request_id": "invocation-id",
        "job_id": "operation-id",
        "operation_id": "operation-id",
    }


@pytest.mark.asyncio
async def test_task_submission_schema_validation_requires_explicit_contract(
    monkeypatch,
):
    validate = AsyncMock()
    monkeypatch.setattr("api.external_backend.service._validate_schema", validate)
    response = BufferedResponse(
        status_code=202,
        headers={"content-type": "application/json; charset=utf-8"},
        body=b'{"id": "remote-id"}',
    )
    route = _route(operation_mode="task")

    await _validate_task_submission_contract(
        route=route,
        selected_cord={"output_schema": {"type": "string"}},
        response=response,
        response_json={"id": "remote-id"},
    )
    validate.assert_not_awaited()

    route = _route(
        operation_mode="task",
        operation_config={"submission_contract": {"enabled": True}},
    )
    assert (
        await _validate_task_submission_contract(
            route=route,
            selected_cord={
                "output_schema": {"type": "object"},
                "output_content_type": "application/json",
            },
            response=response,
            response_json={"id": "remote-id"},
        )
        == "application/json"
    )
    validate.assert_awaited_once_with(
        {"id": "remote-id"}, {"type": "object"}, "task submission response"
    )

    validate.reset_mock()
    route = _route(
        operation_mode="task",
        operation_config={"submission_contract": {"enabled": False}},
    )
    await _validate_task_submission_contract(
        route=route,
        selected_cord={"output_schema": {"type": "string"}},
        response=response,
        response_json={"id": "remote-id"},
    )
    validate.assert_not_awaited()


@pytest.mark.asyncio
async def test_enabled_task_submission_contract_requires_an_effective_constraint():
    response = BufferedResponse(
        status_code=202,
        headers={"content-type": "application/json"},
        body=b'{"id": "remote-id"}',
    )
    enabled = {"submission_contract": {"enabled": True}}

    with pytest.raises(
        ExternalConfigurationError,
        match="requires a non-empty output schema or output content type",
    ):
        await _validate_task_submission_contract(
            route=_route(operation_mode="task", operation_config=enabled),
            selected_cord={},
            response=response,
            response_json={"id": "remote-id"},
        )

    with pytest.raises(
        ExternalConfigurationError,
        match="only supported for task routes",
    ):
        await _validate_task_submission_contract(
            route=_route(operation_config=enabled),
            selected_cord={"output_content_type": "application/json"},
            response=response,
            response_json={"id": "remote-id"},
        )


def test_http_schema_validation_is_default_off_and_explicitly_opted_in():
    cord = {
        "input_schema": {"type": "object"},
        "output_schema": {"type": "string"},
    }

    assert _schema_for_request(_route(), cord) is None
    assert _schema_for_output(_route(), cord) is None
    assert _schema_for_request(
        _route(
            request_config={
                "body_mode": "json",
                "validate_input_schema": True,
            }
        ),
        cord,
    ) == {"type": "object"}
    assert _schema_for_output(
        _route(response_config={"validate_output_schema": True}), cord
    ) == {"type": "string"}

    with pytest.raises(ExternalConfigurationError, match="must be a boolean"):
        _schema_for_request(
            _route(
                request_config={
                    "body_mode": "json",
                    "validate_input_schema": "true",
                }
            ),
            cord,
        )
    with pytest.raises(ExternalConfigurationError, match="must be a boolean"):
        _schema_for_output(
            _route(response_config={"validate_output_schema": "true"}), cord
        )


@pytest.mark.parametrize("missing_schema", [{}, None, "not-an-object"])
def test_http_schema_opt_in_fails_closed_for_missing_or_invalid_metadata(
    missing_schema,
):
    malformed_cord = {
        "input_schema": missing_schema,
        "minimal_input_schema": {},
        "output_schema": missing_schema,
    }

    # Schema metadata remains completely inert until an external route opts in.
    assert _schema_for_request(_route(), malformed_cord) is None
    assert _schema_for_output(_route(), malformed_cord) is None

    with pytest.raises(ExternalConfigurationError, match="requires a non-empty"):
        _schema_for_request(
            _route(
                request_config={
                    "body_mode": "json",
                    "validate_input_schema": True,
                }
            ),
            malformed_cord,
        )
    with pytest.raises(ExternalConfigurationError, match="requires a non-empty"):
        _schema_for_output(
            _route(response_config={"validate_output_schema": True}),
            malformed_cord,
        )


@pytest.mark.asyncio
@pytest.mark.parametrize("enabled", [False, True])
async def test_http_invocation_only_validates_input_schema_when_enabled(
    monkeypatch, enabled
):
    request_config = {"body_mode": "json"}
    if enabled:
        request_config["validate_input_schema"] = True
    route = _route(request_config=request_config)
    binding = SimpleNamespace(routes=[route.model_dump(mode="json")])
    profile = SimpleNamespace(
        body_mode=BodyMode.JSON,
        response_mode=ResponseMode.BUFFERED,
        method="POST",
    )
    validate = AsyncMock()
    monkeypatch.setattr(
        "api.external_backend.service._load_binding",
        AsyncMock(return_value=(binding, SimpleNamespace())),
    )
    monkeypatch.setattr(
        "api.external_backend.service.build_endpoint_profile", lambda *_args: profile
    )
    monkeypatch.setattr(
        "api.external_backend.service._validate_retry_billing_policy",
        lambda *_args: None,
    )
    monkeypatch.setattr(
        "api.external_backend.service._request_body",
        AsyncMock(return_value=({"prompt": 7}, JsonBody({"prompt": 7}))),
    )
    monkeypatch.setattr("api.external_backend.service._validate_schema", validate)
    monkeypatch.setattr(
        "api.external_backend.service._pricing_snapshot",
        AsyncMock(side_effect=RuntimeError("stop after request validation")),
    )

    with pytest.raises(HTTPException) as error:
        await invoke_external(
            request=_request(),
            current_user=SimpleNamespace(user_id="user-id"),
            chute=SimpleNamespace(chute_id="chute-id", name="public-name"),
            selected_cord={
                "path": "/generate",
                "public_api_method": "POST",
                "input_schema": {
                    "type": "object",
                    "properties": {"prompt": {"type": "string"}},
                },
            },
        )

    assert error.value.status_code == 502
    if enabled:
        validate.assert_awaited_once_with(
            {"prompt": 7},
            {
                "type": "object",
                "properties": {"prompt": {"type": "string"}},
            },
            "request",
        )
    else:
        validate.assert_not_awaited()


@pytest.mark.asyncio
async def test_task_submission_sets_a_bounded_operation_expiration(monkeypatch):
    update = AsyncMock()
    monkeypatch.setattr("api.external_backend.service._update_operation", update)
    before = datetime.now(UTC)

    response = await _handle_buffered(
        response=BufferedResponse(
            status_code=202,
            headers={"content-type": "application/json"},
            body=b'{"id": "remote-id"}',
        ),
        route=_route(
            operation_mode="task",
            operation_config={"submit_mapping": {"task_id": "id"}},
        ),
        chute=SimpleNamespace(chute_id="chute-id", name="public-name"),
        selected_cord={},
        operation=SimpleNamespace(operation_id="local-id"),
        request=_request(),
        request_body={"prompt": "hello"},
        invocation_id="invocation-id",
        task_timeout_seconds=3600,
    )

    assert response.status_code == 202
    expires_at = update.await_args.kwargs["expires_at"]
    assert (
        before.timestamp() + 3599
        <= expires_at.timestamp()
        <= datetime.now(UTC).timestamp() + 3601
    )


@pytest.mark.asyncio
async def test_deferred_pending_cancel_wakes_polling_after_task_identity_is_attached(
    monkeypatch,
):
    update = AsyncMock()
    monkeypatch.setattr("api.external_backend.service._update_operation", update)
    before = datetime.now(UTC)

    await _handle_buffered(
        response=BufferedResponse(
            status_code=202,
            headers={"content-type": "application/json"},
            body=b'{"id": "remote-id"}',
        ),
        route=_route(
            operation_mode="task",
            operation_config={"submit_mapping": {"task_id": "id"}},
        ),
        chute=SimpleNamespace(chute_id="chute-id", name="public-name"),
        selected_cord={},
        operation=SimpleNamespace(
            operation_id="local-id",
            settlement_metadata={"cancel_requested": True},
        ),
        request=_request(),
        request_body={"prompt": "hello"},
        invocation_id="invocation-id",
    )

    attached = update.await_args.kwargs
    assert attached["upstream_operation_id"] == "remote-id"
    assert before <= attached["next_poll_at"] <= datetime.now(UTC)


@pytest.mark.asyncio
async def test_billable_upstream_error_marks_attempt_to_disable_hidden_failover(
    monkeypatch,
):
    update = AsyncMock()
    settle = AsyncMock()
    monkeypatch.setattr("api.external_backend.service._update_operation", update)
    monkeypatch.setattr("api.external_backend.service.settle_operation", settle)
    monkeypatch.setattr(
        "api.external_backend.service.track_request_rate_limited",
        lambda _chute_id: None,
    )
    request = _request()

    with pytest.raises(HTTPException) as error:
        await _handle_buffered(
            response=BufferedResponse(
                status_code=429,
                headers={"content-type": "application/json"},
                body=b'{"error":"capacity"}',
            ),
            route=_route(operation_config={"billable_http_statuses": [429]}),
            chute=SimpleNamespace(chute_id="chute-id", name="public-name"),
            selected_cord={},
            operation=SimpleNamespace(operation_id="operation-id"),
            request=request,
            request_body={"prompt": "hello"},
            invocation_id="invocation-id",
        )

    assert error.value.status_code == 429
    assert request.state.external_attempt_billable is True
    settle.assert_awaited_once()
    assert settle.await_args.kwargs == {"billable": True}


def test_output_dependent_price_conditions_require_a_usage_dimension_mapping():
    snapshot = {
        "source": "rules",
        "rules": [
            {
                "metric": "request",
                "unit_price": "1",
                "conditions": {"resolution": {"eq": "high"}},
            }
        ],
        "context": {"dimensions": {}},
    }
    with pytest.raises(ExternalConfigurationError, match="usage dimension"):
        _validate_metering_config(_route(), snapshot)

    route = _route(
        response_config={
            "usage": {
                "fields": {
                    "dimensions.resolution": {
                        "source": "response",
                        "path": "usage.resolution",
                    }
                }
            }
        }
    )
    _validate_metering_config(route, snapshot)

    absent_snapshot = {
        **snapshot,
        "rules": [
            {
                "metric": "request",
                "unit_price": "1",
                "conditions": {"resolution": {"exists": False}},
            }
        ],
    }
    _validate_metering_config(_route(), absent_snapshot)


def test_stream_usage_keeps_one_request_across_delta_events():
    route = _route(
        operation_mode="stream",
        response_config={
            "usage": {
                "default_requests": 1,
                "fields": {
                    "tokens.input": {"source": "request", "path": "input_tokens"},
                    "tokens.output": {"source": "response", "path": "output_tokens"},
                    "counts.resource_chars": {
                        "source": "context",
                        "path": "resource",
                        "aggregate": "length",
                    },
                },
            }
        },
    )
    usage = _extract_initial_stream_usage(route, request_body={"input_tokens": 4})
    assert usage.requests == 1
    assert usage.tokens == {"input": 4}
    assert usage.counts == {"resource_chars": 11}

    for tokens in (2, 3):
        observation = _extract_stream_observation_usage(
            route,
            request_body={"input_tokens": 4},
            response_body={"output_tokens": tokens},
            payload={"output_tokens": tokens},
        )
        usage = merge_stream_usage(usage, observation, "delta")
    assert usage.requests == 1
    assert usage.tokens == {"input": 4, "output": 5}
    assert usage.counts == {"resource_chars": 11}


@pytest.mark.asyncio
async def test_raw_stream_drains_after_disconnect_closes_and_settles(monkeypatch):
    class Upstream:
        private_headers = {"duration": "4"}

        def __init__(self):
            self.chunks = 0
            self.closed = False

        async def iter_bytes(self):
            for chunk in (b"one", b"two"):
                self.chunks += 1
                yield chunk

        async def aclose(self):
            self.closed = True

    upstream = Upstream()
    update = AsyncMock()
    settle = AsyncMock()
    completed = Mock()
    monkeypatch.setattr("api.external_backend.service._update_operation", update)
    monkeypatch.setattr("api.external_backend.service.settle_operation", settle)
    monkeypatch.setattr(
        "api.external_backend.service.track_request_completed", completed
    )
    consumer_alive = asyncio.Event()

    await _raw_stream_producer(
        upstream=upstream,
        queue=asyncio.Queue(maxsize=1),
        consumer_alive=consumer_alive,
        route=_route(
            operation_mode="stream",
            response_config={
                "usage": {
                    "fields": {
                        "output_media_seconds.video": {
                            "source": "response",
                            "path": "headers.duration",
                        }
                    }
                }
            },
        ),
        chute=SimpleNamespace(chute_id="chute-id"),
        operation=SimpleNamespace(operation_id="operation-id"),
        request_body={"prompt": "hello"},
        partial_billable=True,
        session_timeout_seconds=120,
    )

    assert upstream.chunks == 2
    assert upstream.closed is True
    running = next(
        call
        for call in update.await_args_list
        if call.kwargs.get("status") == "running"
    )
    assert running.kwargs["expires_at"] == running.kwargs["next_poll_at"]
    assert (
        running.kwargs["expires_at"] - running.kwargs["started_at"]
    ).total_seconds() == 180
    assert running.kwargs["usage"]["requests"] == "1"
    assert running.kwargs["usage"]["output_media_seconds"] == {"video": "4"}
    settle.assert_awaited_once()
    assert settle.await_args.kwargs["billable"] is True
    completed.assert_called_once_with("chute-id")


@pytest.mark.asyncio
async def test_late_stream_writer_cannot_resurrect_reaped_terminal_operation(
    monkeypatch,
):
    operation = SimpleNamespace(status="failed", finished_at=datetime.now(UTC))

    class Session:
        async def get(self, *_args, **_kwargs):
            return operation

    @asynccontextmanager
    async def session_factory(*_args, **_kwargs):
        yield Session()

    monkeypatch.setattr("api.external_backend.service.get_session", session_factory)

    await _update_operation(
        "operation-id",
        status="succeeded",
        finished_at=datetime.now(UTC),
        next_poll_at=None,
    )

    assert operation.status == "failed"


@pytest.mark.asyncio
async def test_late_not_billable_writer_cannot_override_settled_operation(monkeypatch):
    operation = SimpleNamespace(
        status="failed",
        settlement_status=ExternalSettlementStatus.SETTLED.value,
        settlement_metadata={"operator_actions": [{"action": "retry"}]},
        error={"code": "original"},
    )

    class Session:
        async def execute(self, _statement):
            return SimpleNamespace()

        async def get(self, *_args, **_kwargs):
            return operation

    @asynccontextmanager
    async def session_factory(*_args, **_kwargs):
        yield Session()

    exists = AsyncMock(return_value=False)
    monkeypatch.setattr("api.external_backend.service.get_session", session_factory)
    monkeypatch.setattr(
        "api.external_backend.service.external_usage_event_exists", exists
    )

    await _update_operation(
        "operation-id",
        status="succeeded",
        settlement_status=ExternalSettlementStatus.NOT_BILLABLE.value,
        settlement_metadata={"billable": False},
        error={"code": "late"},
    )

    assert operation.status == "failed"
    assert operation.settlement_status == ExternalSettlementStatus.SETTLED.value
    assert operation.error == {"code": "original"}
    assert operation.settlement_metadata == {"operator_actions": [{"action": "retry"}]}
    exists.assert_not_awaited()


@pytest.mark.asyncio
async def test_late_not_billable_writer_cannot_override_immutable_outbox(monkeypatch):
    operation = SimpleNamespace(
        status="pending",
        settlement_status=ExternalSettlementStatus.PENDING.value,
        settlement_metadata={"pricing": {"source": "rules"}},
        error=None,
    )

    class Session:
        async def execute(self, _statement):
            return SimpleNamespace()

        async def get(self, *_args, **_kwargs):
            return operation

    @asynccontextmanager
    async def session_factory(*_args, **_kwargs):
        yield Session()

    monkeypatch.setattr("api.external_backend.service.get_session", session_factory)
    monkeypatch.setattr(
        "api.external_backend.service.external_usage_event_exists",
        AsyncMock(return_value=True),
    )

    await _update_operation(
        "operation-id",
        status="failed",
        settlement_status=ExternalSettlementStatus.NOT_BILLABLE.value,
        settlement_metadata={"billable": False},
        error={"code": "late"},
    )

    assert operation.status == "pending"
    assert operation.settlement_status == ExternalSettlementStatus.PENDING.value
    assert operation.error is None
    assert operation.settlement_metadata == {"pricing": {"source": "rules"}}


@pytest.mark.asyncio
async def test_operation_metadata_updates_merge_under_lock(monkeypatch):
    operation = SimpleNamespace(
        status="running",
        settlement_status=ExternalSettlementStatus.PENDING.value,
        settlement_metadata={
            "pricing": {"source": "rules"},
            "operator_actions": [{"action": "retry"}],
        },
    )

    class Session:
        async def get(self, *_args, **_kwargs):
            return operation

    @asynccontextmanager
    async def session_factory(*_args, **_kwargs):
        yield Session()

    monkeypatch.setattr("api.external_backend.service.get_session", session_factory)

    await _update_operation(
        "operation-id",
        settlement_metadata={"billable": True},
    )

    assert operation.settlement_metadata == {
        "pricing": {"source": "rules"},
        "operator_actions": [{"action": "retry"}],
        "billable": True,
    }


@pytest.mark.asyncio
async def test_raw_stream_setup_failure_closes_and_applies_partial_policy(monkeypatch):
    class Upstream:
        private_headers = {}

        def __init__(self):
            self.closed = False
            self.iterated = False

        async def iter_bytes(self):
            self.iterated = True
            yield b"unused"

        async def aclose(self):
            self.closed = True

    route = _route(
        operation_mode="stream",
        response_config={
            "usage": {
                "fields": {
                    "tokens.input": {
                        "source": "request",
                        "path": "missing",
                        "required": True,
                    }
                }
            }
        },
    )
    upstream = Upstream()
    settle = AsyncMock()
    monkeypatch.setattr("api.external_backend.service._update_operation", AsyncMock())
    monkeypatch.setattr("api.external_backend.service.settle_operation", settle)

    await _raw_stream_producer(
        upstream=upstream,
        queue=asyncio.Queue(maxsize=1),
        consumer_alive=asyncio.Event(),
        route=route,
        chute=SimpleNamespace(chute_id="chute-id"),
        operation=SimpleNamespace(operation_id="operation-id"),
        request_body={},
        partial_billable=False,
    )

    assert upstream.iterated is False
    assert upstream.closed is True
    assert settle.await_args.kwargs["billable"] is False


@pytest.mark.asyncio
async def test_shutdown_cancels_raw_stream_records_usage_and_awaits_cleanup(
    monkeypatch,
):
    class Upstream:
        private_headers = {}

        def __init__(self):
            self.started = asyncio.Event()
            self.closed = False

        async def iter_bytes(self):
            self.started.set()
            await asyncio.Event().wait()
            yield b"unreachable"

        async def aclose(self):
            await asyncio.sleep(0)
            self.closed = True

    route = _route(
        operation_mode="stream",
        response_config={
            "usage": {
                "fields": {
                    "tokens.input": {
                        "source": "request",
                        "path": "input_tokens",
                    }
                }
            }
        },
    )
    upstream = Upstream()
    update = AsyncMock()
    settle = AsyncMock()
    monkeypatch.setattr("api.external_backend.service._update_operation", update)
    monkeypatch.setattr("api.external_backend.service.settle_operation", settle)
    consumer_alive = asyncio.Event()
    consumer_alive.set()
    task = _spawn(
        _raw_stream_producer(
            upstream=upstream,
            queue=asyncio.Queue(maxsize=1),
            consumer_alive=consumer_alive,
            route=route,
            chute=SimpleNamespace(chute_id="chute-id"),
            operation=SimpleNamespace(operation_id="operation-id"),
            request_body={"input_tokens": 7},
            partial_billable=False,
        )
    )
    await upstream.started.wait()

    remaining = await shutdown_external_invocations(
        timeout_seconds=0,
        cancellation_timeout_seconds=1,
    )

    assert remaining == 0
    assert task.cancelled()
    assert upstream.closed is True
    assert consumer_alive.is_set() is False
    running = next(
        call
        for call in update.await_args_list
        if call.kwargs.get("status") == "running"
    )
    assert running.kwargs["usage"]["tokens"] == {"input": "7"}
    assert update.await_args_list[-1].kwargs["status"] == "failed"
    assert update.await_args_list[-1].kwargs["error"]["code"] == "stream_interrupted"
    settle.assert_awaited_once()
    assert settle.await_args.args[1].tokens == {"input": 7}
    assert settle.await_args.kwargs == {"billable": False}


@pytest.mark.asyncio
async def test_shutdown_cancels_sse_stream_with_observed_partial_usage(monkeypatch):
    class Upstream:
        def __init__(self):
            self.waiting = asyncio.Event()
            self.closed = False

        async def iter_sse(self):
            yield SSEEvent(data='{"usage":{"output_tokens":3}}')
            self.waiting.set()
            await asyncio.Event().wait()
            yield SSEEvent(data="[DONE]")

        async def aclose(self):
            self.closed = True

    route = _route(
        operation_mode="stream",
        response_config={
            "usage_mode": "delta",
            "usage": {
                "fields": {
                    "tokens.output": {
                        "source": "response",
                        "path": "usage.output_tokens",
                    }
                }
            },
        },
    )
    upstream = Upstream()
    update = AsyncMock()
    settle = AsyncMock()
    monkeypatch.setattr("api.external_backend.service._update_operation", update)
    monkeypatch.setattr("api.external_backend.service.settle_operation", settle)
    task = _spawn(
        _sse_stream_producer(
            upstream=upstream,
            queue=asyncio.Queue(maxsize=1),
            consumer_alive=asyncio.Event(),
            route=route,
            chute=SimpleNamespace(chute_id="chute-id", name="public-name"),
            operation=SimpleNamespace(operation_id="operation-id"),
            request_body={"prompt": "hello"},
            invocation_id="invocation-id",
            partial_billable=True,
        )
    )
    await upstream.waiting.wait()

    remaining = await shutdown_external_invocations(
        timeout_seconds=0,
        cancellation_timeout_seconds=1,
    )

    assert remaining == 0
    assert task.cancelled()
    assert upstream.closed is True
    assert update.await_args_list[-1].kwargs["status"] == "failed"
    settle.assert_awaited_once()
    assert settle.await_args.args[1].requests == 1
    assert settle.await_args.args[1].tokens == {"output": 3}
    assert settle.await_args.kwargs == {"billable": True}


@pytest.mark.asyncio
async def test_sse_usage_is_checkpointed_once_for_many_fast_events(monkeypatch):
    checkpointed = asyncio.Event()

    class Upstream:
        async def iter_sse(self):
            for _ in range(50):
                yield SSEEvent(data='{"usage":{"output_tokens":1}}')
            await asyncio.wait_for(checkpointed.wait(), timeout=1)
            yield SSEEvent(data="[DONE]")

        async def aclose(self):
            return None

    route = _route(
        operation_mode="stream",
        response_config={
            "usage_mode": "delta",
            "usage": {
                "fields": {
                    "tokens.output": {
                        "source": "response",
                        "path": "usage.output_tokens",
                    }
                }
            },
        },
    )
    updates = []

    async def update(_operation_id, **values):
        updates.append(values)
        if values.get("usage", {}).get("tokens") == {"output": "50"}:
            checkpointed.set()

    settle = AsyncMock()
    monkeypatch.setattr("api.external_backend.service._update_operation", update)
    monkeypatch.setattr("api.external_backend.service.settle_operation", settle)
    monkeypatch.setattr(
        "api.external_backend.service.track_request_completed", lambda _chute_id: None
    )

    await _sse_stream_producer(
        upstream=Upstream(),
        queue=asyncio.Queue(maxsize=1),
        consumer_alive=asyncio.Event(),
        route=route,
        chute=SimpleNamespace(chute_id="chute-id", name="public-name"),
        operation=SimpleNamespace(operation_id="operation-id"),
        request_body={"prompt": "hello"},
        invocation_id="invocation-id",
        partial_billable=True,
        usage_checkpoint_interval_seconds=0.01,
    )

    running = next(value for value in updates if value.get("status") == "running")
    checkpoints = [
        value
        for value in updates
        if value.get("usage", {}).get("tokens") == {"output": "50"}
        and "started_at" not in value
    ]
    assert running["usage"]["requests"] == "1"
    assert len(checkpoints) == 1
    assert checkpoints[0]["usage"]["tokens"] == {"output": "50"}
    assert settle.await_args.args[1].tokens == {"output": 50}


@pytest.mark.asyncio
async def test_shutdown_cancellation_cleanup_is_bounded():
    started = asyncio.Event()
    release = asyncio.Event()

    async def stubborn_cleanup():
        started.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            await release.wait()

    task = _spawn(stubborn_cleanup())
    await started.wait()
    remaining = await shutdown_external_invocations(
        timeout_seconds=0,
        cancellation_timeout_seconds=0.01,
    )

    assert remaining == 1
    release.set()
    await task
    await asyncio.sleep(0)


@pytest.mark.asyncio
async def test_interrupted_submitted_task_is_left_for_poller(monkeypatch):
    operation = SimpleNamespace(
        operation_id="operation-id",
        settlement_metadata={},
    )
    durable = SimpleNamespace(
        upstream_operation_id="upstream-task-id",
        status="submitted",
    )

    class Session:
        async def get(self, *_args, **_kwargs):
            return durable

    @asynccontextmanager
    async def session_factory(*, readonly=False):
        assert readonly is True
        yield Session()

    update = AsyncMock()
    settle = AsyncMock()
    monkeypatch.setattr("api.external_backend.service.get_session", session_factory)
    monkeypatch.setattr("api.external_backend.service._update_operation", update)
    monkeypatch.setattr("api.external_backend.service.settle_operation", settle)
    request = _request()

    await _finalize_interrupted_invocation(
        request=request,
        operation=operation,
        route=_route(operation_mode="task"),
        request_body={},
        response_body={},
        upstream_accepted=True,
        accepted_billable=True,
        ambiguous_billable=False,
    )

    assert request.state.external_attempt_billable is True
    update.assert_not_awaited()
    settle.assert_not_awaited()


@pytest.mark.asyncio
async def test_accepted_task_handoff_outlives_caller_cancellation():
    started = asyncio.Event()
    release = asyncio.Event()
    completed = asyncio.Event()

    async def attach():
        started.set()
        await release.wait()
        completed.set()
        return Response(status_code=202)

    caller = asyncio.create_task(_finish_accepted_task_handoff(attach()))
    await started.wait()
    caller.cancel()
    await asyncio.sleep(0)
    assert caller.done() is False
    assert completed.is_set() is False

    release.set()
    with pytest.raises(asyncio.CancelledError):
        await caller
    assert completed.is_set() is True


@pytest.mark.asyncio
async def test_pending_operation_cancel_does_not_accelerate_recovery(monkeypatch):
    from api.external_backend import router as operation_router
    from fastapi import Response

    recovery_deadline = datetime.now(UTC)
    operation = SimpleNamespace(status="pending", next_poll_at=recovery_deadline)
    owned = AsyncMock(return_value=operation)
    monkeypatch.setattr(operation_router, "_owned_operation", owned)

    with pytest.raises(HTTPException) as error:
        await operation_router.cancel_operation(
            "operation-id",
            Response(),
            db=SimpleNamespace(),
            current_user=SimpleNamespace(user_id="user-id"),
        )

    assert error.value.status_code == 409
    assert operation.next_poll_at is recovery_deadline
    assert owned.await_args.kwargs["for_update"] is True


@pytest.mark.asyncio
async def test_running_stream_cancel_is_local_and_preserves_recovery_deadline(
    monkeypatch,
):
    from api.external_backend import router as operation_router

    recovery_deadline = datetime.now(UTC)
    operation = SimpleNamespace(
        operation_id="operation-id",
        status="running",
        operation_mode="stream",
        route_snapshot=_route(operation_mode="stream").model_dump(mode="json"),
        settlement_metadata={},
        next_poll_at=recovery_deadline,
    )
    owned = AsyncMock(return_value=operation)
    db = SimpleNamespace(flush=AsyncMock())
    monkeypatch.setattr(operation_router, "_owned_operation", owned)

    result = await operation_router.cancel_operation(
        "operation-id",
        Response(),
        db=db,
        current_user=SimpleNamespace(user_id="user-id"),
    )

    assert result["cancellation"] == "requested"
    assert operation.settlement_metadata["cancel_requested"] is True
    assert operation.next_poll_at is recovery_deadline
    db.flush.assert_awaited_once()


def test_force_rotation_cutoff_expires_only_preexisting_artifact_relays():
    from api.external_backend import router as operation_router

    cutoff = datetime.now(UTC)
    artifact = {"expires_at": (cutoff + timedelta(days=1)).isoformat()}
    operation = SimpleNamespace(
        created_at=cutoff - timedelta(seconds=1),
        expires_at=cutoff + timedelta(days=1),
        account=SimpleNamespace(artifact_relay_invalidated_at=cutoff),
    )

    assert operation_router._is_expired(operation, artifact, now=cutoff)
    operation.created_at = cutoff + timedelta(seconds=1)
    assert not operation_router._is_expired(operation, artifact, now=cutoff)


@pytest.mark.asyncio
async def test_stream_terminator_wakes_consumer_when_delivery_queue_is_full():
    queue: asyncio.Queue[bytes | None] = asyncio.Queue(maxsize=1)
    queue.put_nowait(b"stale-provider-bytes")
    consumer_alive = asyncio.Event()
    consumer_alive.set()

    await _terminate_stream_consumer(queue, consumer_alive)

    assert await asyncio.wait_for(queue.get(), timeout=0.1) is None


@pytest.mark.asyncio
async def test_persisted_cancel_terminates_live_raw_stream_and_wakes_consumer(
    monkeypatch,
):
    class Upstream:
        private_headers = {}

        def __init__(self):
            self.closed = asyncio.Event()

        async def iter_bytes(self):
            await self.closed.wait()
            if False:
                yield b"unreachable"

        async def aclose(self):
            self.closed.set()

    route = _route(
        operation_mode="stream",
        operation_config={"session_budget": {"check_interval_seconds": 0.1}},
    )
    operation = SimpleNamespace(operation_id="operation-id")
    queue: asyncio.Queue[bytes | None] = asyncio.Queue(maxsize=1)
    queue.put_nowait(b"queued-before-cancel")
    consumer_alive = asyncio.Event()
    consumer_alive.set()
    updates = AsyncMock()
    settlement = AsyncMock()
    monkeypatch.setattr(
        "api.external_backend.service._running_budget_check",
        AsyncMock(return_value=(False, "cancel_requested")),
    )
    monkeypatch.setattr("api.external_backend.service._update_operation", updates)
    monkeypatch.setattr("api.external_backend.service.settle_operation", settlement)

    await _raw_stream_producer(
        upstream=Upstream(),
        queue=queue,
        consumer_alive=consumer_alive,
        route=route,
        chute=SimpleNamespace(chute_id="chute-id"),
        operation=operation,
        request_body={},
        partial_billable=True,
        session_timeout_seconds=10,
    )

    assert await asyncio.wait_for(queue.get(), timeout=0.1) is None
    assert consumer_alive.is_set() is False
    assert any(
        item.kwargs.get("status") == "cancelled" for item in updates.await_args_list
    )
    settlement.assert_awaited_once()
    assert settlement.await_args.kwargs == {"billable": True}


def test_immediate_task_terminal_uses_explicit_poll_billability():
    from api.external_backend.service import _task_terminal_billable

    route = _route(
        operation_mode="task",
        operation_config={
            "poll": {
                "method": "GET",
                "path": "/tasks/{task_id}",
                "billable_statuses": [],
            }
        },
    )

    assert _task_terminal_billable(route, "failed") is False


@pytest.mark.asyncio
async def test_concurrent_settlement_uses_one_idempotent_usage_event(
    monkeypatch,
):
    operation = SimpleNamespace(
        operation_id="operation-id",
        settlement_status=ExternalSettlementStatus.PENDING.value,
        settlement_metadata={
            "pricing": {
                "source": "legacy",
                "legacy": {
                    "per_request": 1,
                    "per_million_in": None,
                    "per_million_out": None,
                    "per_step": None,
                    "cache_discount": None,
                },
                "billing_chute_id": "chute-id",
                "free_invocation": False,
            }
        },
        request_metadata={},
        user_id="user-id",
        chute_id="chute-id",
        usage=None,
        settled_at=None,
        operation_mode="task",
        status="succeeded",
    )
    transaction_lock = asyncio.Lock()

    class Result:
        def scalar_one_or_none(self):
            return operation

    class Session:
        async def execute(self, _query):
            await asyncio.sleep(0)
            return Result()

        async def get(self, *_args, **_kwargs):
            await asyncio.sleep(0)
            return operation

    @asynccontextmanager
    async def session_factory(readonly=False):
        if readonly:
            yield Session()
        else:
            async with transaction_lock:
                yield Session()

    enqueue_calls = 0
    queued_amounts = []

    async def enqueue_side_effect(*args, **_kwargs):
        nonlocal enqueue_calls
        await asyncio.sleep(0)
        enqueue_calls += 1
        queued_amounts.append(args[1].amount)
        return enqueue_calls == 1

    enqueue = AsyncMock(side_effect=enqueue_side_effect)
    delivery_lock = asyncio.Lock()

    async def deliver(_operation_id):
        async with delivery_lock:
            if operation.settlement_status == ExternalSettlementStatus.SETTLED.value:
                return None
            operation.settlement_status = ExternalSettlementStatus.SETTLED.value
            return ExternalUsageDeliveryReceipt(
                event_id="external-settlement:operation-id",
                operation_id="operation-id",
                user_id="user-id",
                chute_id="chute-id",
                amount=queued_amounts[0],
                paygo_amount=queued_amounts[0],
                compute_time=0,
                track_task_completion=True,
                free_invocation=False,
                increment_invocation_quota=False,
            )

    monkeypatch.setattr("api.external_backend.service.get_session", session_factory)
    monkeypatch.setattr(
        "api.external_backend.service.enqueue_external_usage_event", enqueue
    )
    monkeypatch.setattr(
        "api.external_backend.service.deliver_external_usage_event", deliver
    )
    monkeypatch.setattr(
        "api.external_backend.service.external_usage_event_exists",
        AsyncMock(side_effect=[False, True]),
    )
    completed = Mock()
    monkeypatch.setattr(
        "api.external_backend.service.track_request_completed", completed
    )

    await asyncio.gather(
        settle_operation("operation-id", NormalizedUsage(requests=1), billable=True),
        settle_operation("operation-id", NormalizedUsage(requests=2), billable=True),
    )

    assert enqueue.await_count == 1
    assert {call.args[1].event_id for call in enqueue.await_args_list} == {
        "external-settlement:operation-id"
    }
    completed.assert_called_once_with("chute-id")
    assert operation.settlement_status == ExternalSettlementStatus.SETTLED.value
    assert float(operation.settlement_metadata["result"]["amount"]) == queued_amounts[0]


@pytest.mark.asyncio
async def test_failed_settlement_handoff_is_redriven_without_marking_settled(
    monkeypatch,
):
    operation = SimpleNamespace(
        operation_id="operation-id",
        settlement_status=ExternalSettlementStatus.PENDING.value,
        settlement_metadata={
            "pricing": {
                "source": "legacy",
                "legacy": {
                    "per_request": 1,
                    "per_million_in": None,
                    "per_million_out": None,
                    "per_step": None,
                    "cache_discount": None,
                },
                "billing_chute_id": "chute-id",
                "free_invocation": False,
            },
            "billable": True,
        },
        request_metadata={},
        user_id="user-id",
        chute_id="chute-id",
        usage={"requests": "1"},
        settled_at=None,
        next_poll_at=None,
        operation_mode="task",
        status="succeeded",
    )

    class Session:
        async def execute(self, _query):
            return SimpleNamespace()

        async def get(self, *_args, **_kwargs):
            return operation

    @asynccontextmanager
    async def session_factory(readonly=False):
        del readonly
        yield Session()

    enqueue = AsyncMock(side_effect=[True, False])
    receipt = ExternalUsageDeliveryReceipt(
        event_id="external-settlement:operation-id",
        operation_id="operation-id",
        user_id="user-id",
        chute_id="chute-id",
        amount=1,
        paygo_amount=1,
        compute_time=0,
        track_task_completion=False,
        free_invocation=False,
        increment_invocation_quota=False,
    )
    delivery_calls = 0

    async def deliver_side_effect(_operation_id):
        nonlocal delivery_calls
        delivery_calls += 1
        if delivery_calls == 1:
            raise OSError("database unavailable")
        operation.settlement_status = ExternalSettlementStatus.SETTLED.value
        operation.settled_at = datetime.now(UTC)
        return receipt

    deliver = AsyncMock(side_effect=deliver_side_effect)
    monkeypatch.setattr("api.external_backend.service.get_session", session_factory)
    monkeypatch.setattr(
        "api.external_backend.service.enqueue_external_usage_event", enqueue
    )
    monkeypatch.setattr(
        "api.external_backend.service.deliver_external_usage_event", deliver
    )
    monkeypatch.setattr(
        "api.external_backend.service.external_usage_event_exists",
        AsyncMock(side_effect=[False, True, True]),
    )
    monkeypatch.setattr("api.external_backend.service.track_request_completed", Mock())

    await settle_operation("operation-id")

    assert operation.settlement_status == ExternalSettlementStatus.FAILED.value
    assert operation.settled_at is None
    assert operation.next_poll_at is not None

    # The durable first decision wins even if a later terminal hook disagrees.
    await settle_operation("operation-id", billable=False)

    assert enqueue.await_count == 1
    assert deliver.await_count == 2
    assert operation.settlement_status == ExternalSettlementStatus.SETTLED.value
    assert operation.settled_at is not None


@pytest.mark.asyncio
async def test_operator_retry_uses_persisted_usage_over_a_delayed_completion_hook(
    monkeypatch,
):
    operation = SimpleNamespace(
        operation_id="operation-id",
        settlement_status=ExternalSettlementStatus.FAILED.value,
        settlement_metadata={
            "pricing": {
                "source": "legacy",
                "legacy": {
                    "per_request": 1,
                    "per_million_in": None,
                    "per_million_out": None,
                    "per_step": None,
                    "cache_discount": None,
                },
                "billing_chute_id": "chute-id",
                "free_invocation": False,
            },
            "billable": True,
            "settlement_operator_retry_at": "2030-01-01T00:00:00+00:00",
            "settlement_pricing_correction_max_amount": "2",
        },
        request_metadata={},
        user_id="user-id",
        chute_id="chute-id",
        usage={"requests": "2"},
        settled_at=None,
        next_poll_at=None,
        operation_mode="sync",
        status="succeeded",
    )

    class Session:
        async def execute(self, _query):
            return SimpleNamespace()

        async def get(self, *_args, **_kwargs):
            return operation

    @asynccontextmanager
    async def session_factory(readonly=False):
        del readonly
        yield Session()

    enqueue = AsyncMock(return_value=True)
    receipt = ExternalUsageDeliveryReceipt(
        event_id="external-settlement:operation-id",
        operation_id="operation-id",
        user_id="user-id",
        chute_id="chute-id",
        amount=2,
        paygo_amount=2,
        compute_time=0,
        track_task_completion=False,
        free_invocation=False,
        increment_invocation_quota=False,
    )
    monkeypatch.setattr("api.external_backend.service.get_session", session_factory)
    monkeypatch.setattr(
        "api.external_backend.service.external_usage_event_exists",
        AsyncMock(return_value=False),
    )
    monkeypatch.setattr(
        "api.external_backend.service.enqueue_external_usage_event", enqueue
    )
    monkeypatch.setattr(
        "api.external_backend.service.deliver_external_usage_event",
        AsyncMock(return_value=receipt),
    )

    await settle_operation("operation-id", NormalizedUsage(requests=99), billable=False)

    event = enqueue.await_args.args[1]
    assert event.amount == 2
    assert operation.usage["requests"] == "2"
    assert operation.settlement_metadata["result"]["amount"] == "2"


@pytest.mark.asyncio
async def test_unpriceable_settlement_is_quarantined_after_bounded_attempts(
    monkeypatch,
):
    operation = SimpleNamespace(
        settlement_status=ExternalSettlementStatus.FAILED.value,
        settlement_metadata={"settlement_attempts": 7},
        next_poll_at=datetime.now(UTC),
        settled_at=None,
    )

    class Session:
        async def execute(self, _statement):
            return SimpleNamespace()

        async def get(self, *_args, **_kwargs):
            return operation

    @asynccontextmanager
    async def session_factory(*_args, **_kwargs):
        yield Session()

    monkeypatch.setattr("api.external_backend.service.get_session", session_factory)
    monkeypatch.setattr(
        "api.external_backend.service.external_usage_event_exists",
        AsyncMock(return_value=False),
    )

    await _record_settlement_failure(
        "operation-id", PricingConfigurationError("missing usage")
    )

    assert operation.settlement_status == ExternalSettlementStatus.QUARANTINED.value
    assert operation.next_poll_at is None
    assert operation.settlement_metadata["settlement_attempts"] == 8
    assert operation.settlement_metadata["settlement_failure_code"] == (
        "unpriceable_usage"
    )
    assert "settlement_quarantined_at" in operation.settlement_metadata
    assert "settlement_next_attempt_at" not in operation.settlement_metadata


@pytest.mark.asyncio
async def test_immutable_outbox_delivery_failure_is_never_quarantined(monkeypatch):
    operation = SimpleNamespace(
        settlement_status=ExternalSettlementStatus.FAILED.value,
        settlement_metadata={"settlement_attempts": 7},
        next_poll_at=None,
        settled_at=None,
    )

    class Session:
        async def execute(self, _statement):
            return SimpleNamespace()

        async def get(self, *_args, **_kwargs):
            return operation

    @asynccontextmanager
    async def session_factory(*_args, **_kwargs):
        yield Session()

    monkeypatch.setattr("api.external_backend.service.get_session", session_factory)
    monkeypatch.setattr(
        "api.external_backend.service.external_usage_event_exists",
        AsyncMock(return_value=True),
    )

    await _record_settlement_failure("operation-id", OSError("database unavailable"))

    assert operation.settlement_status == ExternalSettlementStatus.FAILED.value
    assert operation.next_poll_at is not None
    assert operation.settlement_metadata["settlement_attempts"] == 8
    assert "settlement_quarantined_at" not in operation.settlement_metadata


@pytest.mark.asyncio
async def test_free_invocation_updates_quota_and_subscription_cap_caches(monkeypatch):
    from api.config import settings
    from api.user.schemas import InvocationQuota

    quota_key = AsyncMock(return_value="quota-key")
    subscription = AsyncMock(return_value=(300, datetime.now(UTC), None, None))
    increment = AsyncMock()
    expire = AsyncMock()
    monkeypatch.setattr(InvocationQuota, "quota_key", quota_key)
    monkeypatch.setattr(InvocationQuota, "get_subscription_record", subscription)
    monkeypatch.setattr(settings.redis_client, "incrbyfloat", increment)
    monkeypatch.setattr(settings.redis_client, "expire", expire)

    await _record_free_invocation_usage(
        "user-id",
        "chute-id",
        2.5,
        increment_invocation_quota=True,
    )

    assert increment.await_count == 3
    assert increment.await_args_list[0].args == ("quota-key", 1.0)
    assert expire.await_count == 2

    quota_key.reset_mock()
    increment.reset_mock()
    await _record_free_invocation_usage(
        "user-id",
        "chute-id",
        2.5,
        increment_invocation_quota=False,
    )
    quota_key.assert_not_awaited()
    assert increment.await_count == 2


def test_stream_partial_billing_and_task_lifetime_are_strict_and_bounded():
    assert _bill_partial_stream(_route(operation_mode="stream")) is True
    with pytest.raises(ExternalConfigurationError, match="must be a boolean"):
        _bill_partial_stream(
            _route(
                operation_mode="stream",
                response_config={"bill_partial_streams": "false"},
            )
        )

    assert _task_timeout_seconds(_route(operation_mode="task")) == 24 * 3600
    assert (
        _task_timeout_seconds(
            _route(
                operation_mode="task",
                operation_config={"task_timeout_seconds": 3600},
            )
        )
        == 3600
    )
    assert (
        _task_timeout_seconds(
            _route(
                operation_mode="task",
                operation_config={"task": {"timeout_seconds": 7200}},
            )
        )
        == 7200
    )
    with pytest.raises(ExternalConfigurationError, match="between 60"):
        _task_timeout_seconds(
            _route(
                operation_mode="task",
                operation_config={"task_timeout_seconds": 30},
            )
        )


@pytest.mark.asyncio
async def test_sync_output_is_validated_before_success_or_settlement(monkeypatch):
    validate = AsyncMock(side_effect=ValueError("invalid upstream output"))
    update = AsyncMock()
    settle = AsyncMock()
    monkeypatch.setattr("api.external_backend.service._validate_schema", validate)
    monkeypatch.setattr("api.external_backend.service._update_operation", update)
    monkeypatch.setattr("api.external_backend.service.settle_operation", settle)

    with pytest.raises(ValueError, match="invalid upstream output"):
        await _handle_buffered(
            response=BufferedResponse(
                status_code=200,
                headers={"content-type": "application/json"},
                body=b'{"value": 1}',
            ),
            route=_route(response_config={"validate_output_schema": True}),
            chute=SimpleNamespace(chute_id="chute-id", name="public-name"),
            selected_cord={"output_schema": {"type": "string"}},
            operation=SimpleNamespace(operation_id="operation-id"),
            request=_request(),
            request_body={"prompt": "hello"},
            invocation_id="invocation-id",
        )

    update.assert_not_awaited()
    settle.assert_not_awaited()


@pytest.mark.asyncio
async def test_sync_output_schema_is_metadata_without_explicit_opt_in(monkeypatch):
    validate = AsyncMock(side_effect=AssertionError("schema validation was invoked"))
    update = AsyncMock()
    settle = AsyncMock()
    monkeypatch.setattr("api.external_backend.service._validate_schema", validate)
    monkeypatch.setattr("api.external_backend.service._update_operation", update)
    monkeypatch.setattr("api.external_backend.service.settle_operation", settle)
    monkeypatch.setattr(
        "api.external_backend.service.track_request_completed", lambda _chute_id: None
    )

    response = await _handle_buffered(
        response=BufferedResponse(
            status_code=200,
            headers={"content-type": "application/json"},
            body=b'{"value": 1}',
        ),
        route=_route(),
        chute=SimpleNamespace(chute_id="chute-id", name="public-name"),
        selected_cord={"output_schema": {"type": "string"}},
        operation=SimpleNamespace(operation_id="operation-id"),
        request=_request(),
        request_body={"prompt": "hello"},
        invocation_id="invocation-id",
    )

    assert response.status_code == 200
    validate.assert_not_awaited()
    update.assert_awaited_once()
    settle.assert_awaited_once()


@pytest.mark.asyncio
async def test_http_invocation_rejects_realtime_route_before_spending(monkeypatch):
    route = _route(operation_mode="realtime")
    binding = SimpleNamespace(routes=[route.model_dump(mode="json")])
    monkeypatch.setattr(
        "api.external_backend.service._load_binding",
        AsyncMock(return_value=(binding, SimpleNamespace())),
    )

    with pytest.raises(HTTPException) as error:
        await invoke_external(
            request=_request(),
            current_user=SimpleNamespace(user_id="user-id"),
            chute=SimpleNamespace(chute_id="chute-id", name="public-name"),
            selected_cord={
                "path": "/generate",
                "public_api_method": "POST",
            },
        )

    assert error.value.status_code == 426


@pytest.mark.asyncio
@pytest.mark.parametrize("ambiguous_billable", [False, True])
async def test_inflight_invocation_cancellation_reaches_terminal_settlement(
    monkeypatch,
    ambiguous_billable,
):
    route = _route(
        operation_config={
            "bill_ambiguous_transport_errors": ambiguous_billable,
        }
    )
    binding = SimpleNamespace(
        binding_id="binding-id",
        routes=[route.model_dump(mode="json")],
    )
    account = SimpleNamespace(account_id="account-id")
    operation = SimpleNamespace(
        operation_id="operation-id",
        request_metadata={},
    )
    profile = SimpleNamespace(
        body_mode=BodyMode.JSON,
        response_mode=ResponseMode.BUFFERED,
        method="POST",
    )
    execution_started = asyncio.Event()

    async def wait_for_upstream(*_args):
        execution_started.set()
        await asyncio.Event().wait()
        raise AssertionError("unreachable")

    update = AsyncMock()
    settle = AsyncMock()
    monkeypatch.setattr(
        "api.external_backend.service._load_binding",
        AsyncMock(return_value=(binding, account)),
    )
    monkeypatch.setattr(
        "api.external_backend.service.build_endpoint_profile",
        lambda *_args: profile,
    )
    monkeypatch.setattr(
        "api.external_backend.service._request_body",
        AsyncMock(return_value=({"prompt": "hello"}, JsonBody({"prompt": "hello"}))),
    )
    monkeypatch.setattr(
        "api.external_backend.service._pricing_snapshot",
        AsyncMock(return_value={}),
    )
    monkeypatch.setattr(
        "api.external_backend.service._validate_metering_config", lambda *_args: None
    )
    monkeypatch.setattr(
        "api.external_backend.service._validate_retry_billing_policy",
        lambda *_args: None,
    )
    monkeypatch.setattr(
        "api.external_backend.service._create_operation",
        AsyncMock(return_value=(operation, False)),
    )
    monkeypatch.setattr(
        "api.external_backend.service.build_secret_resolver", lambda *_args: None
    )
    monkeypatch.setattr(
        "api.external_backend.service.ExternalExecutor", lambda **_kwargs: object()
    )
    monkeypatch.setattr(
        "api.external_backend.service._execute_with_retry", wait_for_upstream
    )
    monkeypatch.setattr("api.external_backend.service._update_operation", update)
    monkeypatch.setattr("api.external_backend.service.settle_operation", settle)
    request = _request()
    task = asyncio.create_task(
        invoke_external(
            request=request,
            current_user=SimpleNamespace(user_id="user-id"),
            chute=SimpleNamespace(chute_id="chute-id", name="public-name"),
            selected_cord={
                "path": "/generate",
                "public_api_method": "POST",
            },
        )
    )
    await execution_started.wait()

    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert update.await_args.kwargs["status"] == "failed"
    assert update.await_args.kwargs["error"]["code"] == "execution_interrupted"
    settle.assert_awaited_once()
    assert settle.await_args.args[1].requests == 1
    assert settle.await_args.kwargs == {"billable": ambiguous_billable}
    assert (
        getattr(request.state, "external_attempt_billable", False) is ambiguous_billable
    )
