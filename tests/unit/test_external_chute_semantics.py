import sys
import fnmatch
from types import ModuleType, SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from fastapi import HTTPException
from starlette.routing import Match


# The development environment contains two implementations of this optional
# namespace. These model tests do not exercise the substrate client.
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

import api.database.orms  # noqa: E402, F401

with patch("ctypes.CDLL", return_value=MagicMock()):
    from api.chute.router import router as chute_router  # noqa: E402
from api.chute.response import ChuteResponse
from api.chute.schemas import Chute, ChuteHistory, Cord
from api.chute import util as chute_util
from api.user.schemas import PriceOverride


class _SlugCacheRedis:
    def __init__(self):
        self.values = {}

    async def get(self, key):
        return self.values.get(key)

    async def set(self, key, value, **_kwargs):
        self.values[key] = str(value).encode()

    async def delete(self, *keys):
        for key in keys:
            self.values.pop(key, None)


class _PriceCacheRedis(_SlugCacheRedis):
    def __init__(self):
        super().__init__()
        self.deleted = []

    async def scan(self, cursor, match, count):
        del cursor, count
        found = [key for key in self.values if fnmatch.fnmatch(str(key), match)]
        return 0, found

    async def delete(self, *keys):
        self.deleted.extend(keys)
        await super().delete(*keys)


def test_hosted_pricing_dimensions_ignore_unaddressable_client_keys():
    assert chute_util._hosted_pricing_dimensions(
        {
            "": 1,
            " valid ": {"": "ignored", "tier": "high"},
            "parameters": {"resolution": "1080p"},
        }
    ) == {
        "valid": {"tier": "high"},
        "parameters": {"resolution": "1080p"},
    }


def test_price_override_write_boundary_parses_rules_and_match_groups():
    override = PriceOverride(user_id="*", chute_id="hosted-id")

    with pytest.raises(ValueError, match="missing unit_price"):
        override.pricing_rules = [{"metric": "request"}]

    with pytest.raises(ValueError, match="tier and one fallback"):
        override.pricing_rules = [
            {
                "metric": "request",
                "unit_price": "1",
                "match_group": "request-tier",
                "priority": 10,
                "conditions": {"mode": "priority"},
            }
        ]

    rules = [
        {
            "metric": "request",
            "unit_price": "1",
            "match_group": "request-tier",
            "priority": 10,
            "conditions": {"mode": "priority"},
        },
        {
            "metric": "request",
            "unit_price": "0.5",
            "match_group": "request-tier",
            "fallback": True,
        },
    ]
    override.pricing_rules = rules

    assert override.pricing_rules == rules


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
def test_price_override_write_boundary_rejects_unsafe_legacy_rates(field, value):
    override = PriceOverride(user_id="*", chute_id="hosted-id")

    with pytest.raises(ValueError, match=field):
        setattr(override, field, value)

    setattr(override, field, None)
    assert getattr(override, field) is None


def test_hosted_rule_pricing_does_not_take_over_with_legacy_fallback_fields():
    override = SimpleNamespace(
        per_request=99,
        per_million_in=2,
        per_million_out=None,
        pricing_rules=[
            {
                "metric": "request",
                "unit_price": 1,
                "scope": {"cord": "different-cord"},
            }
        ],
    )

    result = chute_util._price_hosted_rule_override(
        override,
        {"it": 100, "ot": 200},
        {"json": {"model": "test"}},
        cord="chat",
        path="/v1/chat/completions",
        method="POST",
        chute_id="hosted-id",
    )

    assert result is not None
    assert not result.applied
    assert result.source == "none"


def test_hosted_rule_pricing_missing_metric_and_malformed_dimensions_are_safe():
    missing = chute_util._price_hosted_rule_override(
        SimpleNamespace(
            pricing_rules=[
                {
                    "metric": "token",
                    "bucket": "output",
                    "unit_price": 2,
                }
            ]
        ),
        None,
        {"json": {"": 1}},
        cord="chat",
        path="/v1/chat/completions",
        method="POST",
        chute_id="hosted-id",
    )
    malformed = chute_util._price_hosted_rule_override(
        SimpleNamespace(pricing_rules={"not": "an array"}),
        {"ot": 100},
        {"json": {"": 1}},
        cord="chat",
        path="/v1/chat/completions",
        method="POST",
        chute_id="hosted-id",
    )

    assert missing is not None
    assert not missing.applied
    assert not missing.complete
    assert malformed is None


def test_external_chute_has_no_fake_source_or_hosted_runtime_semantics():
    assert Chute.__table__.c.code.nullable is True
    assert Chute.__table__.c.filename.nullable is True
    assert Chute.__table__.c.ref_str.nullable is True
    assert ChuteHistory.__table__.c.code.nullable is True
    assert ChuteHistory.__table__.c.filename.nullable is True
    assert ChuteHistory.__table__.c.ref_str.nullable is True
    assert Chute.__table__.c.disabled.nullable is False

    chute = Chute(
        chute_id="external-id",
        user_id="owner-id",
        name="external-model",
        execution_backend="external",
        public=True,
        cords=[],
        jobs=[],
        node_selector={},
        code=None,
        filename=None,
        ref_str=None,
    )
    assert chute.code is None
    assert chute.filename is None
    assert chute.ref_str is None
    assert chute.preemptible is False

    disabled_response = ChuteResponse.model_construct(
        execution_backend="external", disabled=True, instances=[]
    )
    enabled_response = ChuteResponse.model_construct(
        execution_backend="external", disabled=False, instances=[]
    )
    assert disabled_response.hot is False
    assert enabled_response.hot is True


@pytest.mark.asyncio
async def test_slug_cache_invalidation_allows_delete_and_recreate(monkeypatch):
    redis = _SlugCacheRedis()
    monkeypatch.setattr(chute_util.settings, "_redis_client", redis)
    slug = "owner-public-model"
    cache_key = f"idbyslug:{slug}"
    chute_util.chute_id_by_slug.cache_clear()
    try:
        await redis.set(cache_key, "old-chute-id")
        assert await chute_util.chute_id_by_slug(slug) == "old-chute-id"

        await chute_util.invalidate_chute_cache(
            "old-chute-id",
            "public-model",
            slug,
        )
        await redis.set(cache_key, "new-chute-id")

        assert await chute_util.chute_id_by_slug(slug) == "new-chute-id"
    finally:
        chute_util.chute_id_by_slug.cache_clear()


@pytest.mark.asyncio
async def test_price_cache_invalidation_clears_all_user_and_global_variants(
    monkeypatch,
):
    redis = _PriceCacheRedis()
    monkeypatch.setattr(chute_util.settings, "_redis_client", redis)
    redis.values.update(
        {
            "priceoverride2:*:chute-id": b"global",
            "priceoverride2:user-id:chute-id": b"user",
            "mtokenprice3:user-id:chute-id": b"tokens",
            "priceoverride2:user-id:other-id": b"keep",
        }
    )

    await chute_util.invalidate_price_override_cache("chute-id")

    assert "priceoverride2:user-id:other-id" in redis.values
    assert not any(key.endswith(":chute-id") for key in redis.values)


@pytest.mark.parametrize("method", ["GET", "POST", "PUT", "PATCH", "DELETE", "HEAD"])
def test_chute_orm_preserves_external_cords_for_every_gateway_method(method):
    chute = Chute(
        chute_id=f"external-{method.lower()}",
        user_id="owner-id",
        name=f"external-{method.lower()}",
        execution_backend="external",
        cords=[
            Cord(
                method=method,
                path="/generate",
                function="generate",
                stream=False,
                public_api_path="/generate",
                public_api_method=method,
            )
        ],
        jobs=[],
        node_selector={},
        code=None,
        filename=None,
        ref_str=None,
    )

    assert chute.cords[0]["public_api_method"] == method


@pytest.mark.parametrize("backend", ["hosted", "external"])
@pytest.mark.parametrize(
    "reserved_path", ["/evidence", "/EVIDENCE", "/hf_info", "/HF_INFO"]
)
def test_chute_orm_rejects_reserved_cord_paths(backend, reserved_path):
    with pytest.raises(ValueError, match="Reserved canonical Chute path"):
        Chute(
            chute_id=f"{backend}-reserved-path",
            user_id="owner-id",
            name=f"{backend}-reserved-path",
            execution_backend=backend,
            cords=[
                Cord(
                    method="GET",
                    path="/generate",
                    function="generate",
                    stream=False,
                    public_api_path=reserved_path,
                    public_api_method="GET",
                )
            ],
            jobs=[],
            node_selector={},
            code=None,
            filename=None,
            ref_str=None,
        )


@pytest.mark.parametrize("method", ["GET", "POST", "PUT", "PATCH", "DELETE", "HEAD"])
def test_canonical_nested_chute_url_reaches_cord_dispatch_before_catchalls(method):
    from api.invocation.router import conditional_invocation_router

    scope = {
        "type": "http",
        "method": method,
        "path": "/chutes/external-id/generate/video",
        "root_path": "",
        "state": {"invocation_dispatch": True},
    }

    match, child_scope = conditional_invocation_router.routes[0].matches(scope)

    assert match == Match.FULL
    assert child_scope["endpoint"].__name__ == "hostname_invocation"


def test_nested_declared_cord_remains_available_outside_reserved_management_paths():
    from api.invocation.router import conditional_invocation_router

    scope = {
        "type": "http",
        "method": "GET",
        "path": "/chutes/external-id/nested/evidence",
        "root_path": "",
        "state": {"invocation_dispatch": True},
    }

    match, child_scope = conditional_invocation_router.routes[0].matches(scope)

    assert match == Match.FULL
    assert child_scope["endpoint"].__name__ == "hostname_invocation"


@pytest.mark.asyncio
async def test_unauthenticated_canonical_cord_does_not_disclose_declared_path():
    from api.invocation.router import hostname_invocation

    request = SimpleNamespace(
        state=SimpleNamespace(
            chute_id="private-chute-id",
            invocation_public_path="/generate",
        )
    )

    with pytest.raises(HTTPException) as exc_info:
        await hostname_invocation(request, current_user=None)

    assert exc_info.value.status_code == 404
    assert exc_info.value.detail == "Not Found"


@pytest.mark.parametrize(
    ("path", "expected_endpoint"),
    [
        ("/hosted-id/hf_info", "get_chute_hf_info"),
        ("/hosted-id/evidence", "get_tee_chute_evidence"),
    ],
)
def test_non_invocation_management_paths_fall_through_conditional_cord_route(
    path, expected_endpoint
):
    scope = {
        "type": "http",
        "method": "GET",
        "path": path,
        "root_path": "",
        "state": {"auth_method": "read", "chute_id": None},
    }

    for route in chute_router.routes:
        match, child_scope = route.matches(scope)
        if match == Match.FULL:
            assert child_scope["endpoint"].__name__ == expected_endpoint
            break
    else:
        pytest.fail("no management route matched")


@pytest.mark.parametrize("path", ["/ping", "/users/me", "/chutes/id/hf_info"])
def test_hostname_invocation_dispatch_precedes_colliding_api_paths(path):
    from api.invocation.router import conditional_invocation_router

    scope = {
        "type": "http",
        "method": "GET",
        "path": path,
        "root_path": "",
        "state": {"invocation_dispatch": True},
    }

    match, child_scope = conditional_invocation_router.routes[0].matches(scope)

    assert match == Match.FULL
    assert child_scope["endpoint"].__name__ == "hostname_invocation"


def test_hostname_dispatch_route_falls_through_without_middleware_classification():
    from api.invocation.router import conditional_invocation_router

    scope = {
        "type": "http",
        "method": "GET",
        "path": "/ping",
        "root_path": "",
        "state": {"invocation_dispatch": False},
    }

    match, _child_scope = conditional_invocation_router.routes[0].matches(scope)

    assert match == Match.NONE
