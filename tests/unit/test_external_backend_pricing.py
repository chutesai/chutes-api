import asyncio
import ctypes
import sys
from decimal import Decimal
from types import ModuleType, SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from starlette.requests import Request

# These tests exercise pricing snapshots, not the optional chain/native clients.
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


ctypes.CDLL = lambda _path: _FakeNativeLibrary()  # type: ignore[assignment]

import api.database.orms  # noqa: E402,F401
import api.external_backend.service as service
from api.external_backend.billing_outbox import ExternalUsageDeliveryReceipt
from api.external_backend.config import ExternalConfigurationError
from api.external_backend.schemas import ExternalRouteConfig
from api.external_backend.schemas import ExternalSettlementStatus
from api.payment.pricing import NormalizedUsage, PricingResult, parse_pricing_rules


def route(*, mode="sync", response_config=None, operation_config=None):
    return ExternalRouteConfig.model_validate(
        {
            "cord_path": "/generate",
            "upstream_resource_id": "resource-id",
            "operation_mode": mode,
            "protocol": "generic-json",
            "path_template": "/generate",
            "method": "POST",
            "request_config": {"body_mode": "json"},
            "response_config": response_config or {},
            "operation_config": operation_config or {},
        }
    )


def request(path="/invoke", method="POST"):
    value = Request(
        {
            "type": "http",
            "method": method,
            "path": path,
            "raw_path": path.encode(),
            "query_string": b"",
            "headers": [],
            "scheme": "https",
            "server": ("api.example.test", 443),
            "client": ("127.0.0.1", 1234),
        }
    )
    value.state.free_invocation = False
    return value


@pytest.mark.asyncio
async def test_canonical_http_pricing_uses_the_normalized_public_cord_route(
    monkeypatch,
):
    override = SimpleNamespace(
        pricing_rules=[
            {
                "metric": "request",
                "unit_price": "0.1",
                "scope": {"path": "/invoke", "method": "POST"},
            }
        ]
    )
    monkeypatch.setattr(
        service.PriceOverride,
        "get",
        AsyncMock(return_value=override),
    )
    canonical = request("/chutes/chute-id/invoke")
    canonical.state.invocation_public_path = "/invoke"

    snapshot = await service._pricing_snapshot(
        SimpleNamespace(user_id="user-id"),
        SimpleNamespace(chute_id="chute-id"),
        {
            "function": "generate",
            "path": "/generate",
            "public_api_path": "/invoke",
            "public_api_method": "POST",
        },
        canonical,
        {},
    )

    assert snapshot["context"]["path"] == "/invoke"
    assert snapshot["context"]["method"] == "POST"


@pytest.mark.asyncio
async def test_realtime_pricing_uses_public_method_not_upstream_method(monkeypatch):
    override = SimpleNamespace(
        pricing_rules=[
            {
                "metric": "request",
                "unit_price": "0.1",
                "scope": {"path": "/session", "method": "GET"},
            }
        ]
    )
    monkeypatch.setattr(
        service.PriceOverride,
        "get",
        AsyncMock(return_value=override),
    )
    upstream_shaped_request = request("/session", method="POST")

    snapshot = await service._pricing_snapshot(
        SimpleNamespace(user_id="user-id"),
        SimpleNamespace(chute_id="chute-id"),
        {
            "function": "session",
            "path": "/session",
            "public_api_path": "/session",
            "public_api_method": "GET",
        },
        upstream_shaped_request,
        {},
    )

    assert snapshot["context"]["path"] == "/session"
    assert snapshot["context"]["method"] == "GET"


def test_pricing_snapshot_freezes_output_conditional_candidates(monkeypatch):
    override = SimpleNamespace(
        pricing_rules=[
            {
                "id": "video-default",
                "metric": "output_media_second",
                "bucket": "video",
                "unit_price": "0",
                "scope": {
                    "cord": "generate",
                    "path": "/invoke",
                    "method": "POST",
                },
                "effective_from": "2020-01-01T00:00:00Z",
                "effective_to": "2100-01-01T00:00:00Z",
            },
            {
                "id": "video-high",
                "metric": "output_media_second",
                "bucket": "video",
                "unit_price": "0.25",
                "conditions": {
                    "parameters.resolution": "high",
                    "output.codec": "mp4",
                },
                "scope": {
                    "cord": "generate",
                    "path": "/invoke",
                    "method": "POST",
                },
                "effective_from": "2020-01-01T00:00:00Z",
                "effective_to": "2100-01-01T00:00:00Z",
            },
        ]
    )

    async def get_override(_user_id, _chute_id):
        return override

    monkeypatch.setattr(service.PriceOverride, "get", get_override)
    snapshot = asyncio.run(
        service._pricing_snapshot(
            SimpleNamespace(user_id="user-id"),
            SimpleNamespace(chute_id="chute-id"),
            {"function": "generate", "path": "/generate"},
            request(),
            {
                "parameters": {"resolution": "high"},
                "prompt": "must not be retained when it is not a price dimension",
            },
        )
    )

    conditional = next(item for item in snapshot["rules"] if item["id"] == "video-high")
    assert conditional["conditions"]["output.codec"] == "mp4"
    assert conditional["scope"]["cord"] == "generate"
    assert conditional["effective_to"].startswith("2100-01-01")
    assert snapshot["context"]["dimensions"] == {"parameters.resolution": "high"}

    result = service._pricing_result(
        snapshot,
        NormalizedUsage(
            output_media_seconds={"video": 8},
            dimensions={"output": {"codec": "mp4"}},
        ),
    )
    assert result.amount == Decimal("2.00")


@pytest.mark.asyncio
async def test_pricing_snapshot_rejects_conditional_only_ungrouped_rules(monkeypatch):
    monkeypatch.setattr(
        service.PriceOverride,
        "get",
        AsyncMock(
            return_value=SimpleNamespace(
                pricing_rules=[
                    {
                        "metric": "request",
                        "unit_price": "1",
                        "conditions": {"tier": "premium"},
                    }
                ]
            )
        ),
    )

    with pytest.raises(service.HTTPException) as error:
        await service._pricing_snapshot(
            SimpleNamespace(user_id="user-id"),
            SimpleNamespace(chute_id="chute-id"),
            {"function": "generate", "path": "/generate"},
            request(),
            {"tier": "premium"},
        )

    assert error.value.status_code == 503


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("per_request", -0.01),
        ("per_million_in", -0.01),
        ("per_million_out", -0.01),
        ("per_step", -0.01),
        ("cache_discount", -0.01),
        ("cache_discount", 1.01),
    ],
)
@pytest.mark.asyncio
async def test_pricing_snapshot_rejects_unsafe_legacy_rates(monkeypatch, field, value):
    legacy = {
        "per_request": None,
        "per_million_in": None,
        "per_million_out": None,
        "per_step": None,
        "cache_discount": None,
    }
    legacy[field] = value
    monkeypatch.setattr(
        service.PriceOverride,
        "get",
        AsyncMock(return_value=SimpleNamespace(pricing_rules=None, **legacy)),
    )

    with pytest.raises(service.HTTPException) as error:
        await service._pricing_snapshot(
            SimpleNamespace(user_id="user-id", permissions_bitmask=0),
            SimpleNamespace(chute_id="chute-id", discount=0),
            {"function": "generate", "path": "/generate"},
            request(),
            {},
        )

    assert error.value.status_code == 503
    assert error.value.detail == "Pricing is not safely configured for this endpoint."


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("permissions", "expected_free", "expected_invoice"),
    [
        (1 << 4, True, False),
        (1 << 8, False, True),
        ((1 << 4) | (1 << 8), False, True),
    ],
)
async def test_pricing_snapshot_preserves_free_and_invoice_billing_semantics(
    monkeypatch, permissions, expected_free, expected_invoice
):
    monkeypatch.setattr(
        service.PriceOverride,
        "get",
        AsyncMock(
            return_value=SimpleNamespace(
                pricing_rules=[{"metric": "request", "unit_price": "1"}]
            )
        ),
    )

    snapshot = await service._pricing_snapshot(
        SimpleNamespace(user_id="user-id", permissions_bitmask=permissions),
        SimpleNamespace(chute_id="chute-id", discount=0),
        {"function": "generate", "path": "/generate"},
        request(),
        {},
    )

    assert snapshot["free_invocation"] is expected_free
    assert snapshot["invoice_billing"] is expected_invoice
    assert snapshot["balance_exempt"] is True
    assert snapshot["increment_invocation_quota"] is False


def test_pricing_snapshot_retains_group_selection_policy():
    rules = parse_pricing_rules(
        [
            {
                "metric": "output_media_second",
                "bucket": "video",
                "unit_price": "0.2",
                "conditions": {"resolution": "high"},
                "match_group": "internal-video-tier",
                "priority": 20,
            },
            {
                "metric": "output_media_second",
                "bucket": "video",
                "unit_price": "0.3",
                "match_group": "internal-video-tier",
                "fallback": True,
            },
        ]
    )

    tier = service._rule_snapshot(rules[0], 0)
    fallback = service._rule_snapshot(rules[1], 1)

    assert tier["match_group"] == "internal-video-tier"
    assert tier["priority"] == 20
    assert tier["fallback"] is False
    assert fallback["fallback"] is True


@pytest.mark.parametrize(
    ("pricing_rule", "usage_field"),
    [
        (
            {"metric": "output_media_second", "bucket": "video", "unit_price": 1},
            "output_media_seconds.video",
        ),
        ({"metric": "image", "unit_price": 1}, "images.generated"),
        (
            {"metric": "character", "bucket": "input", "unit_price": 1},
            "characters.input",
        ),
        ({"metric": "tool", "bucket": "search", "unit_price": 1}, "tools.search"),
    ],
)
def test_metering_validation_requires_the_matching_metric_bucket(
    pricing_rule, usage_field
):
    configured = route(
        response_config={
            "usage": {"fields": {usage_field: {"source": "response", "path": "usage"}}}
        }
    )
    snapshot = {"source": "rules", "rules": [pricing_rule]}

    service._validate_metering_config(configured, snapshot)

    wrong_bucket = route(
        response_config={
            "usage": {
                "fields": {"tokens.output": {"source": "response", "path": "usage"}}
            }
        }
    )
    with pytest.raises(ExternalConfigurationError, match="matching usage field"):
        service._validate_metering_config(wrong_bucket, snapshot)


def test_legacy_metering_validation_checks_each_legacy_quantity():
    configured = route(
        response_config={
            "usage": {
                "fields": {
                    "tokens.input": "usage.input",
                    "tokens.output": "usage.output",
                    "tokens.cached_input": "usage.cached",
                    "counts.steps": "usage.steps",
                }
            }
        }
    )
    snapshot = {
        "source": "legacy",
        "legacy": {
            "per_million_in": 1,
            "per_million_out": 2,
            "cache_discount": 0.5,
            "per_step": 0.01,
        },
    }
    service._validate_metering_config(configured, snapshot)

    missing_cache = route(
        response_config={
            "usage": {
                "fields": {
                    "tokens.input": "usage.input",
                    "tokens.output": "usage.output",
                    "counts.steps": "usage.steps",
                }
            }
        }
    )
    with pytest.raises(ExternalConfigurationError, match="cached_input"):
        service._validate_metering_config(missing_cache, snapshot)


def test_initial_task_usage_retains_request_units_for_later_polling():
    configured = route(
        mode="task",
        operation_config={
            "poll": {
                "usage": {
                    "default_requests": 0,
                    "fields": {
                        "output_media_seconds.video": {
                            "source": "request",
                            "path": "parameters.duration",
                        },
                        "dimensions.resolution": {
                            "source": "request",
                            "path": "parameters.resolution",
                        },
                        "tokens.output": {
                            "source": "response",
                            "path": "usage.output_tokens",
                            "required": True,
                        },
                    },
                }
            }
        },
    )

    usage = service._extract_initial_task_usage(
        configured,
        request_body={"parameters": {"duration": 5, "resolution": "high"}},
    )

    assert usage.requests == Decimal("1")
    assert usage.output_media_seconds == {"video": Decimal("5")}
    assert usage.dimensions == {"resolution": "high"}
    assert usage.tokens == {}


def test_settlement_persists_actual_charge_and_paygo_equivalent(monkeypatch):
    operation = SimpleNamespace(
        operation_id="operation-id",
        user_id="user-id",
        chute_id="chute-id",
        request_metadata={"app_id": "app-id"},
        usage=None,
        settlement_status=ExternalSettlementStatus.PENDING.value,
        settlement_metadata={
            "pricing": {
                "source": "rules",
                "rules": [
                    {"metric": "request", "unit_price": "1.25"},
                    {
                        "metric": "token",
                        "bucket": "output",
                        "unit_price": "2",
                        "unit_size": "1000000",
                    },
                ],
                "context": {
                    "cord": "generate",
                    "path": "/invoke",
                    "method": "POST",
                    "dimensions": {},
                    "at": "2030-01-01T00:00:00Z",
                },
                "billing_chute_id": "chute-id",
                "free_invocation": True,
            }
        },
        settled_at=None,
    )

    class Result:
        def scalar_one_or_none(self):
            return operation

    class Session:
        async def execute(self, _statement):
            return Result()

        async def get(self, _model, _identifier, **_kwargs):
            return operation

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            return False

    def get_session(*, readonly=False):
        return Session()

    observed = {}

    async def enqueue_usage(_session, event):
        observed.update(
            user_id=event.user_id,
            chute_id=event.chute_id,
            amount=float(event.amount),
            compute_time=event.compute_time,
            paygo_amount=float(event.paygo_amount),
            app_id=event.app_id,
            event_id=event.event_id,
        )
        return True

    async def deliver_usage(_operation_id):
        operation.settlement_status = ExternalSettlementStatus.SETTLED.value
        return ExternalUsageDeliveryReceipt(
            event_id=observed["event_id"],
            operation_id=operation.operation_id,
            user_id=observed["user_id"],
            chute_id=observed["chute_id"],
            amount=Decimal(str(observed["amount"])),
            paygo_amount=Decimal(str(observed["paygo_amount"])),
            compute_time=observed["compute_time"],
            track_task_completion=False,
            free_invocation=True,
            increment_invocation_quota=False,
        )

    monkeypatch.setattr(service, "get_session", get_session)
    monkeypatch.setattr(
        service, "external_usage_event_exists", AsyncMock(return_value=False)
    )
    monkeypatch.setattr(service, "enqueue_external_usage_event", enqueue_usage)
    monkeypatch.setattr(service, "deliver_external_usage_event", deliver_usage)
    monkeypatch.setattr(service, "_record_free_invocation_usage", AsyncMock())

    asyncio.run(
        service.settle_operation(
            operation.operation_id, NormalizedUsage(requests=1), billable=True
        )
    )

    pricing = operation.settlement_metadata["result"]
    assert pricing["amount"] == "1.25"
    assert pricing["charged_amount"] == "0"
    assert pricing["paygo_amount"] == "1.25"
    assert pricing["complete"] is False
    assert pricing["missing_rule_count"] == 1
    assert operation.settlement_metadata["pricing_complete"] is False
    assert operation.settlement_metadata["pricing_missing_rule_count"] == 1
    assert observed["amount"] == 0.0
    assert observed["paygo_amount"] == 1.25
    assert observed["app_id"] == "app-id"
    assert operation.settlement_status == ExternalSettlementStatus.SETTLED.value


def test_complete_explicit_zero_usage_settles_through_zero_charge_outbox(monkeypatch):
    operation = SimpleNamespace(
        operation_id="zero-operation-id",
        user_id="user-id",
        chute_id="chute-id",
        request_metadata={},
        usage=None,
        settlement_status=ExternalSettlementStatus.PENDING.value,
        settlement_metadata={
            "pricing": {
                "source": "rules",
                "rules": [
                    {
                        "metric": "token",
                        "bucket": "input",
                        "unit_price": "2",
                        "unit_size": "1000000",
                    },
                    {
                        "metric": "token",
                        "bucket": "output",
                        "unit_price": "4",
                        "unit_size": "1000000",
                    },
                ],
                "context": {
                    "cord": "generate",
                    "path": "/invoke",
                    "method": "POST",
                    "dimensions": {},
                    "at": "2030-01-01T00:00:00Z",
                },
                "billing_chute_id": "chute-id",
                "free_invocation": False,
            }
        },
        settled_at=None,
        operation_mode="sync",
        status="succeeded",
    )

    class Result:
        def scalar_one_or_none(self):
            return operation

    class Session:
        async def execute(self, _statement):
            return Result()

        async def get(self, _model, _identifier, **_kwargs):
            return operation

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            return False

    def get_session(*, readonly=False):
        return Session()

    observed = {}

    async def enqueue_usage(_session, event):
        observed["event"] = event
        return True

    async def deliver_usage(_operation_id):
        event = observed["event"]
        operation.settlement_status = ExternalSettlementStatus.SETTLED.value
        return ExternalUsageDeliveryReceipt(
            event_id=event.event_id,
            operation_id=event.operation_id,
            user_id=event.user_id,
            chute_id=event.chute_id,
            amount=event.amount,
            paygo_amount=event.paygo_amount,
            compute_time=event.compute_time,
            track_task_completion=False,
            free_invocation=False,
            increment_invocation_quota=False,
        )

    settlement_failure = AsyncMock()
    monkeypatch.setattr(service, "get_session", get_session)
    monkeypatch.setattr(
        service, "external_usage_event_exists", AsyncMock(return_value=False)
    )
    monkeypatch.setattr(service, "enqueue_external_usage_event", enqueue_usage)
    monkeypatch.setattr(service, "deliver_external_usage_event", deliver_usage)
    monkeypatch.setattr(service, "_record_settlement_failure", settlement_failure)

    asyncio.run(
        service.settle_operation(
            operation.operation_id,
            NormalizedUsage(tokens={"input": 0, "output": 0}),
            billable=True,
        )
    )

    event = observed["event"]
    assert event.amount == Decimal("0")
    assert event.paygo_amount == Decimal("0")
    assert operation.settlement_metadata["pricing_complete"] is True
    assert operation.settlement_metadata["pricing_missing_rule_count"] == 0
    assert operation.settlement_metadata["result"]["amount"] == "0"
    assert operation.settlement_status == ExternalSettlementStatus.SETTLED.value
    settlement_failure.assert_not_awaited()


@pytest.mark.parametrize(
    "pricing_result",
    [
        PricingResult(
            amount=Decimal("0"),
            matched_rule_count=1,
            source="rules",
            missing_rule_count=1,
        ),
        PricingResult(
            amount=Decimal("0"),
            matched_rule_count=0,
            source="none",
            missing_rule_count=0,
        ),
    ],
    ids=("incomplete-zero", "not-applied"),
)
def test_zero_settlement_rejection_paths_record_failure(monkeypatch, pricing_result):
    operation = SimpleNamespace(
        operation_id="unpriceable-operation-id",
        user_id="user-id",
        chute_id="chute-id",
        request_metadata={},
        usage=None,
        settlement_status=ExternalSettlementStatus.PENDING.value,
        settlement_metadata={"pricing": {"source": "rules"}},
        settled_at=None,
        operation_mode="sync",
        status="succeeded",
    )

    class Session:
        async def execute(self, _statement):
            return SimpleNamespace()

        async def get(self, _model, _identifier, **_kwargs):
            return operation

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            return False

    settlement_failure = AsyncMock()
    enqueue = AsyncMock()
    deliver = AsyncMock()
    monkeypatch.setattr(service, "get_session", lambda **_kwargs: Session())
    monkeypatch.setattr(
        service, "external_usage_event_exists", AsyncMock(return_value=False)
    )
    monkeypatch.setattr(service, "_pricing_result", lambda *_args: pricing_result)
    monkeypatch.setattr(service, "enqueue_external_usage_event", enqueue)
    monkeypatch.setattr(service, "deliver_external_usage_event", deliver)
    monkeypatch.setattr(service, "_record_settlement_failure", settlement_failure)

    asyncio.run(
        service.settle_operation(
            operation.operation_id, NormalizedUsage(requests=1), billable=True
        )
    )

    settlement_failure.assert_awaited_once()
    assert isinstance(
        settlement_failure.await_args.args[1], service.PricingConfigurationError
    )
    enqueue.assert_not_awaited()
    deliver.assert_not_awaited()
