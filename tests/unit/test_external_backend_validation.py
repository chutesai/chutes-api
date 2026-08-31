from types import SimpleNamespace

import pytest

from api.external_backend.config import (
    ExternalConfigurationError,
    build_artifact_profile,
)
from api.external_backend.schemas import ExternalRouteConfig
from api.external_backend.schema_validation import (
    RemoteSchemaReferenceError,
    UnsafeSchemaError,
    local_json_schema_validator,
)
from api.external_backend.validation import (
    RouteConfigurationError,
    validate_route_configuration,
)


def _account():
    return SimpleNamespace(
        account_id="account-id",
        base_url="https://gateway.example.test/v1",
        credential_references={"primary": "secret://secret-id"},
        auth_header_templates=[
            {
                "name": "Authorization",
                "template": "Bearer {token}",
                "references": {"token": "primary"},
            }
        ],
        connection_config={},
    )


def _cord(**overrides):
    value = {
        "method": "POST",
        "path": "/generate",
        "function": "generate",
        "stream": False,
        "public_api_path": "/v1/generate",
        "public_api_method": "POST",
        "input_schema": {
            "type": "object",
            "properties": {"prompt": {"type": "string"}},
        },
        "minimal_input_schema": {},
        "output_schema": {"type": "object"},
    }
    value.update(overrides)
    return value


def _route(**overrides):
    value = {
        "cord_path": "/generate",
        "upstream_resource_id": "model-id",
        "operation_mode": "sync",
        "protocol": "generic-json",
        "path_template": "/models/{model}/generate",
        "method": "GET",
        "request_config": {
            "body_mode": "none",
            "path_parameters": {"model": {"value": "model-id"}},
        },
        "response_config": {
            "public": {"remove_keys": ["remote_vendor"]},
        },
        "operation_config": {"retry": {"statuses": [429, 503]}},
        "capabilities": {},
    }
    value.update(overrides)
    return ExternalRouteConfig.model_validate(value)


def test_compiler_accepts_public_post_translated_to_upstream_get():
    validate_route_configuration(
        _account(),
        _route(),
        cord=_cord(),
        pricing_rules=({"metric": "request", "unit_price": "0.1"},),
    )


def test_route_request_limit_cannot_exceed_process_ceiling(monkeypatch):
    from api.external_backend import validation

    monkeypatch.setattr(validation.settings, "max_request_body_bytes", 10)
    configured = _route(
        request_config={
            "body_mode": "none",
            "path_parameters": {"model": {"value": "model-id"}},
            "max_request_bytes": 11,
        }
    )

    with pytest.raises(RouteConfigurationError, match="process request-body ceiling"):
        validate_route_configuration(
            _account(),
            configured,
            cord=_cord(),
            pricing_rules=({"metric": "request", "unit_price": "0.1"},),
        )


def test_artifact_profile_uses_configured_ports_and_a_bounded_size():
    route = _route(
        operation_config={
            "artifact": {
                "allowed_hosts": ["assets.example.test"],
                "allowed_ports": [8443],
                "max_bytes": 123456,
            }
        }
    )

    profile = build_artifact_profile(
        _account(), route, "https://assets.example.test:8443/result"
    )

    assert profile.network.allowed_ports == {443, 8443}
    assert profile.max_bytes == 123456


def test_artifact_reference_cannot_authorize_its_own_port():
    route = _route(
        operation_config={"artifact": {"allowed_hosts": ["assets.example.test"]}}
    )

    with pytest.raises(ExternalConfigurationError, match="port is not allowlisted"):
        build_artifact_profile(
            _account(), route, "https://assets.example.test:8443/result"
        )


def test_authenticated_artifact_requires_an_exact_configured_origin():
    route = _route(
        operation_config={
            "artifact": {
                "authenticated": True,
                "allowed_ports": [8443],
            }
        }
    )

    with pytest.raises(
        ExternalConfigurationError,
        match="authentication is not permitted for this origin",
    ):
        build_artifact_profile(
            _account(), route, "https://gateway.example.test:8443/result"
        )

    route.operation_config["artifact"]["auth_allowed_origins"] = [
        "https://gateway.example.test:8443"
    ]
    profile = build_artifact_profile(
        _account(), route, "https://gateway.example.test:8443/result"
    )
    assert profile.secret_headers


@pytest.mark.parametrize(
    ("artifact", "message"),
    [
        ({"allowed_ports": "443"}, "array of ports"),
        ({"allowed_ports": [0]}, "integers from 1 through 65535"),
        ({"max_bytes": 100}, "between 1024"),
        ({"max_bytes": 10 * 1024 * 1024 * 1024 + 1}, "between 1024"),
        (
            {"auth_allowed_origins": ["https://gateway.example.test/path"]},
            "scheme, hostname, and optional port",
        ),
    ],
)
def test_compiler_rejects_invalid_artifact_relay_limits(artifact, message):
    with pytest.raises(RouteConfigurationError, match=message):
        validate_route_configuration(
            _account(), _route(operation_config={"artifact": artifact})
        )


def test_json_schema_validation_is_local_only_and_allows_local_fragments():
    schema = {
        "$defs": {"prompt": {"type": "string"}},
        "type": "object",
        "properties": {"prompt": {"$ref": "#/$defs/prompt"}},
    }
    validator = local_json_schema_validator(schema)

    assert list(validator.iter_errors({"prompt": "hello"})) == []
    assert list(validator.iter_errors({"prompt": 7}))

    for reference in (
        "https://schemas.example.test/request.json",
        "relative-schema.json",
        "data:application/schema+json,%7B%7D",
    ):
        with pytest.raises(RemoteSchemaReferenceError, match="document-local"):
            local_json_schema_validator({"$ref": reference})


def test_route_compiler_only_checks_remote_cord_schema_references_when_enabled():
    cord = _cord(input_schema={"$ref": "http://127.0.0.1/schema"})
    validate_route_configuration(
        _account(),
        _route(),
        cord=cord,
        pricing_rules=({"metric": "request", "unit_price": "0.1"},),
    )

    configured = _route()
    configured.request_config["validate_input_schema"] = True
    with pytest.raises(RouteConfigurationError, match="document-local"):
        validate_route_configuration(
            _account(),
            configured,
            cord=cord,
            pricing_rules=({"metric": "request", "unit_price": "0.1"},),
        )


@pytest.mark.parametrize(
    "schema",
    [
        {"type": "string", "pattern": "^(a+)+$"},
        {"type": "object", "patternProperties": {"^(a+)+$": {"type": "string"}}},
    ],
)
def test_json_schema_compiler_rejects_uncancellable_regex_keywords(schema):
    with pytest.raises(UnsafeSchemaError, match="regex keywords"):
        local_json_schema_validator(schema)
    configured = _route()
    configured.request_config["validate_input_schema"] = True
    with pytest.raises(RouteConfigurationError, match="regex keywords"):
        validate_route_configuration(
            _account(),
            configured,
            cord=_cord(input_schema=schema),
            pricing_rules=({"metric": "request", "unit_price": "0.1"},),
        )


def test_cord_schemas_are_unchecked_metadata_by_default(monkeypatch):
    from api.external_backend import validation

    def fail_if_schema_is_traversed(*_args, **_kwargs):
        pytest.fail("disabled Cord schema metadata was traversed")

    monkeypatch.setattr(validation, "_schema_paths", fail_if_schema_is_traversed)
    validate_route_configuration(
        _account(),
        _route(),
        cord=_cord(
            input_schema={"type": "not-a-json-schema-type"},
            minimal_input_schema={"$ref": "https://schemas.example/input.json"},
            output_schema={"pattern": "^(a+)+$"},
        ),
        pricing_rules=({"metric": "request", "unit_price": "0.1"},),
    )


@pytest.mark.parametrize(
    ("config_name", "flag"),
    [
        ("request_config", "validate_input_schema"),
        ("response_config", "validate_output_schema"),
    ],
)
def test_schema_validation_opt_ins_are_strict_booleans(config_name, flag):
    configured = _route()
    getattr(configured, config_name)[flag] = "true"

    with pytest.raises(RouteConfigurationError, match=rf"{flag} must be a boolean"):
        validate_route_configuration(_account(), configured, cord=_cord())


def test_input_schema_validation_requires_a_nonempty_cord_schema():
    configured = _route()
    configured.request_config["validate_input_schema"] = True
    cord = _cord(input_schema={}, minimal_input_schema={})

    with pytest.raises(RouteConfigurationError, match="requires a non-empty"):
        validate_route_configuration(_account(), configured, cord=cord)


def test_output_schema_validation_requires_and_compiles_the_cord_schema():
    configured = _route()
    configured.response_config["validate_output_schema"] = True

    with pytest.raises(RouteConfigurationError, match="requires a non-empty"):
        validate_route_configuration(
            _account(), configured, cord=_cord(output_schema={})
        )

    with pytest.raises(RouteConfigurationError, match="not valid"):
        validate_route_configuration(
            _account(),
            configured,
            cord=_cord(output_schema={"type": "not-a-json-schema-type"}),
        )


@pytest.mark.parametrize("operation_mode", ["stream", "task", "realtime"])
def test_output_schema_validation_is_rejected_outside_sync(operation_mode):
    configured = _route(
        operation_mode=operation_mode,
        response_config={"validate_output_schema": True},
    )

    with pytest.raises(RouteConfigurationError, match="buffered sync routes"):
        validate_route_configuration(_account(), configured, cord=_cord())


def test_input_schema_validation_is_rejected_for_raw_and_realtime_routes():
    raw = _route(method="POST")
    raw.request_config["body_mode"] = "raw"
    raw.request_config["validate_input_schema"] = True
    with pytest.raises(RouteConfigurationError, match="raw request bodies"):
        validate_route_configuration(_account(), raw, cord=_cord())

    realtime = _route(operation_mode="realtime")
    realtime.request_config["validate_input_schema"] = True
    with pytest.raises(RouteConfigurationError, match="validate_client_messages"):
        validate_route_configuration(_account(), realtime, cord=_cord())


def test_raw_response_schema_metadata_is_allowed_when_validation_is_disabled():
    configured = _route(
        operation_mode="stream",
        response_config={"mode": "stream"},
    )
    validate_route_configuration(
        _account(),
        configured,
        cord=_cord(
            output_schema={"$ref": "https://schemas.example/output.json"},
            output_content_type="application/octet-stream",
        ),
    )


@pytest.mark.parametrize("enabled", [None, False])
def test_disabled_task_submission_contract_does_not_compile_schema_metadata(enabled):
    contract = {
        "output_schema": {"$ref": "https://schemas.example/task.json"},
        "output_content_type": "not a content type",
    }
    if enabled is not None:
        contract["enabled"] = enabled
    validate_route_configuration(
        _account(),
        _route(
            operation_config={
                "submission_contract": contract,
            }
        ),
        cord=_cord(),
    )


@pytest.mark.parametrize("operation_mode", ["sync", "stream", "realtime"])
def test_enabled_submission_contract_is_rejected_outside_task_routes(operation_mode):
    configured = _route(
        operation_mode=operation_mode,
        operation_config={
            "submission_contract": {
                "enabled": True,
                "output_content_type": "application/json",
            }
        },
    )

    with pytest.raises(RouteConfigurationError, match="only supported for task"):
        validate_route_configuration(_account(), configured, cord=_cord())


def test_enabled_task_submission_contract_requires_an_effective_constraint():
    configured = _route(
        operation_mode="task",
        operation_config={"submission_contract": {"enabled": True}},
    )

    with pytest.raises(
        RouteConfigurationError,
        match="requires a non-empty output schema or output content type",
    ):
        validate_route_configuration(
            _account(),
            configured,
            cord=_cord(output_schema={}, output_content_type=None),
        )


def test_route_compiler_rejects_opaque_sse_passthrough():
    with pytest.raises(RouteConfigurationError, match="provider-obscuring"):
        validate_route_configuration(
            _account(),
            _route(response_config={"allow_non_json_sse_data": True}),
        )


def test_compiler_accepts_data_driven_query_mapping_and_custom_resource_pin():
    configured = _route(
        request_config={
            "body_mode": "none",
            "query_parameters": {
                "model": "context.resource",
                "quality": {"path": "body.parameters.quality", "required": False},
                "limit": {"value": 10},
                "preview": True,
            },
            "resource_query_parameter": "deployment_id",
        }
    )

    validate_route_configuration(_account(), configured)


@pytest.mark.parametrize(
    ("query_parameters", "message"),
    [
        ({"quality": ["high"]}, "must be a scalar"),
        ({"quality": "payload.quality"}, "body, query, or context"),
        ({"bad&name": {"value": "x"}}, "name is invalid"),
        ({"model": {"value": "different-model"}}, "configured resource"),
    ],
)
def test_compiler_rejects_unsafe_or_ambiguous_query_mapping(query_parameters, message):
    configured = _route(
        request_config={
            "body_mode": "none",
            "query_parameters": query_parameters,
        }
    )

    with pytest.raises(RouteConfigurationError, match=message):
        validate_route_configuration(_account(), configured)


def test_compiler_rejects_unknown_transform_field_and_bad_json_schema():
    with pytest.raises(RouteConfigurationError, match="unsupported fields: tranform"):
        validate_route_configuration(
            _account(),
            _route(request_config={"body_mode": "json", "tranform": {}}),
        )


def test_compiler_rejects_runtime_policy_gaps_before_invocation():
    with pytest.raises(RouteConfigurationError, match="raw bodies"):
        validate_route_configuration(
            _account(),
            _route(
                method="POST",
                request_config={
                    "body_mode": "raw",
                    "transform": {"remove": ["provider"]},
                },
            ),
        )

    with pytest.raises(RouteConfigurationError, match="cannot also be retried"):
        validate_route_configuration(
            _account(),
            _route(
                operation_config={
                    "billable_http_statuses": [429],
                    "retry": {"statuses": [429], "max_attempts": 2},
                }
            ),
        )

    with pytest.raises(RouteConfigurationError, match="must be a boolean"):
        validate_route_configuration(
            _account(),
            _route(response_config={"redirects": {"allow_cross_origin": "false"}}),
        )

    with pytest.raises(RouteConfigurationError, match="output_schema"):
        validate_route_configuration(
            _account(),
            _route(
                operation_mode="task",
                operation_config={
                    "submission_contract": {
                        "enabled": True,
                        "output_schema": {"type": "not-a-json-schema-type"},
                    }
                },
            ),
        )

    invalid_input = _route()
    invalid_input.request_config["validate_input_schema"] = True
    with pytest.raises(RouteConfigurationError, match="not valid"):
        validate_route_configuration(
            _account(),
            invalid_input,
            cord=_cord(input_schema={"type": "not-a-json-schema-type"}),
        )


def test_compiler_requires_task_polling_before_submission_can_be_configured():
    task_route = _route(
        method="POST",
        operation_mode="task",
        request_config={"body_mode": "json"},
        operation_config={
            "submit_mapping": {"task_id": "task_id"},
            "task": {"billable_statuses": ["failed", "cancelled"]},
        },
    )

    with pytest.raises(RouteConfigurationError, match="poll is required"):
        validate_route_configuration(_account(), task_route)


def test_compiler_accepts_complete_task_lifecycle_configuration():
    task_route = _route(
        method="POST",
        operation_mode="task",
        request_config={"body_mode": "json"},
        operation_config={
            "submit_mapping": {"task_id": "task_id"},
            "submission_contract": {
                "enabled": True,
                "output_content_type": "application/json",
            },
            "task": {"billable_statuses": ["failed", "cancelled"]},
            "task_timeout_seconds": 3600,
            "poll": {
                "endpoint": {
                    "path_template": "/tasks/{task_id}",
                    "method": "GET",
                },
                "task": {
                    "status": {
                        "path": "state",
                        "map": {
                            "waiting": "pending",
                            "working": "running",
                            "done": "succeeded",
                            "error": "failed",
                        },
                        "required": True,
                    },
                    "result": "output",
                },
            },
        },
    )

    validate_route_configuration(
        _account(),
        task_route,
        pricing_rules=({"metric": "request", "unit_price": "0.1"},),
    )


def test_compiler_rejects_streaming_task_lifecycle_endpoint():
    task_route = _route(
        method="POST",
        operation_mode="task",
        request_config={"body_mode": "json"},
        operation_config={
            "submit_mapping": {"task_id": "task_id"},
            "task": {"billable_statuses": ["failed", "cancelled"]},
            "poll": {
                "endpoint": {
                    "path_template": "/tasks/{task_id}",
                    "method": "GET",
                    "response_mode": "sse",
                },
                "task": {
                    "status": {
                        "path": "state",
                        "map": {"done": "succeeded"},
                    }
                },
            },
        },
    )

    with pytest.raises(
        RouteConfigurationError,
        match="task poll endpoint requires response mode: buffered",
    ):
        validate_route_configuration(_account(), task_route)


@pytest.mark.parametrize(
    ("endpoint", "lifecycle_request", "message"),
    (
        (
            {"path_template": "/tasks/{task_id}", "body_mode": "raw"},
            None,
            "supports only none and json body modes",
        ),
        (
            {"path_template": "/tasks/{task_id}", "body_mode": "none"},
            {"body": {"include": "result"}},
            "cannot configure a body in none mode",
        ),
    ),
)
def test_compiler_rejects_invalid_task_lifecycle_request_transport(
    endpoint, lifecycle_request, message
):
    poll = {
        "endpoint": endpoint,
        "task": {
            "status": {
                "path": "state",
                "map": {"done": "succeeded"},
            }
        },
    }
    if lifecycle_request is not None:
        poll["request"] = lifecycle_request
    task_route = _route(
        method="POST",
        operation_mode="task",
        request_config={"body_mode": "json"},
        operation_config={
            "submit_mapping": {"task_id": "task_id"},
            "task": {"billable_statuses": ["failed", "cancelled"]},
            "poll": poll,
        },
    )

    with pytest.raises(RouteConfigurationError, match=message):
        validate_route_configuration(_account(), task_route)


def test_compiler_requires_explicit_task_terminal_billability_policy():
    task_route = _route(
        method="POST",
        operation_mode="task",
        request_config={"body_mode": "json"},
        operation_config={
            "submit_mapping": {"task_id": "task_id"},
            "poll": {
                "endpoint": {
                    "path_template": "/tasks/{task_id}",
                    "method": "GET",
                },
                "task": {
                    "status": {
                        "path": "state",
                        "map": {"done": "succeeded", "error": "failed"},
                    }
                },
            },
        },
    )

    with pytest.raises(RouteConfigurationError, match="billable_statuses"):
        validate_route_configuration(_account(), task_route)


def test_compiler_requires_usage_mapping_for_non_request_pricing():
    with pytest.raises(
        RouteConfigurationError, match="requires a matching usage field"
    ):
        validate_route_configuration(
            _account(),
            _route(),
            pricing_rules=({"metric": "output_media_second", "unit_price": "0.1"},),
        )


def test_compiler_requires_output_pricing_conditions_to_have_dimension_source():
    configured = _route(
        response_config={
            "usage": {
                "fields": {
                    "output_media_seconds.video": "usage.duration",
                }
            }
        }
    )
    rule = {
        "metric": "output_media_second",
        "bucket": "video",
        "unit_price": "0.1",
        "conditions": {"output.codec": "mp4"},
    }
    baseline = {
        "metric": "output_media_second",
        "bucket": "video",
        "unit_price": "0.05",
    }

    with pytest.raises(RouteConfigurationError, match="output.codec.*usage dimension"):
        validate_route_configuration(
            _account(), configured, cord=_cord(), pricing_rules=(baseline, rule)
        )

    configured.response_config["usage"]["fields"]["dimensions.output.codec"] = (
        "output.codec"
    )
    validate_route_configuration(
        _account(), configured, cord=_cord(), pricing_rules=(baseline, rule)
    )


def test_compiler_only_uses_validated_request_schema_for_pricing_conditions():
    configured = _route(
        response_config={
            "usage": {
                "fields": {
                    "output_media_seconds.video": "usage.duration",
                }
            }
        }
    )
    cord = _cord(
        input_schema={
            "type": "object",
            "properties": {
                "parameters": {
                    "type": "object",
                    "properties": {"resolution": {"type": "string"}},
                }
            },
        }
    )

    pricing_rules = (
        {
            "metric": "output_media_second",
            "bucket": "video",
            "unit_price": "0.05",
        },
        {
            "metric": "output_media_second",
            "bucket": "video",
            "unit_price": "0.1",
            "conditions": {"parameters.resolution": "high"},
        },
    )

    with pytest.raises(
        RouteConfigurationError, match="parameters.resolution.*usage dimension"
    ):
        validate_route_configuration(
            _account(), configured, cord=cord, pricing_rules=pricing_rules
        )

    configured.request_config["validate_input_schema"] = True
    validate_route_configuration(
        _account(), configured, cord=cord, pricing_rules=pricing_rules
    )


def test_compiler_requires_grouped_price_tiers_to_have_an_unambiguous_fallback():
    configured = _route(
        response_config={
            "usage": {
                "fields": {
                    "output_media_seconds.video": "usage.duration",
                    "dimensions.height": "usage.height",
                }
            }
        }
    )
    tier = {
        "metric": "output_media_second",
        "bucket": "video",
        "unit_price": "0.2",
        "conditions": {"height": {"gte": 1080}},
        "match_group": "resolution-tier",
        "priority": 10,
    }

    with pytest.raises(RouteConfigurationError, match="fallback"):
        validate_route_configuration(
            _account(), configured, cord=_cord(), pricing_rules=(tier,)
        )

    validate_route_configuration(
        _account(),
        configured,
        cord=_cord(),
        pricing_rules=(
            tier,
            {
                "metric": "output_media_second",
                "bucket": "video",
                "unit_price": "0.3",
                "match_group": "resolution-tier",
                "fallback": True,
            },
        ),
    )


def test_compiler_only_requires_metering_for_rules_scoped_to_the_cord():
    validate_route_configuration(
        _account(),
        _route(),
        cord=_cord(function="generate"),
        pricing_rules=(
            {
                "metric": "request",
                "unit_price": "0.1",
                "scope": {"cord": "generate"},
            },
            {
                "metric": "output_media_second",
                "bucket": "video",
                "unit_price": "0.2",
                "scope": {"cord": "video"},
            },
        ),
    )

    with pytest.raises(RouteConfigurationError, match="do not cover Cord"):
        validate_route_configuration(
            _account(),
            _route(),
            cord=_cord(function="missing"),
            pricing_rules=(
                {
                    "metric": "request",
                    "unit_price": "0.1",
                    "scope": {"cord": "generate"},
                },
            ),
        )


@pytest.mark.parametrize(
    "scope",
    (
        {"path": "/v1/other"},
        {"method": "GET"},
        {"path": "/v1/generate", "method": "GET"},
    ),
)
def test_compiler_requires_pricing_scope_to_match_public_cord_route(scope):
    with pytest.raises(RouteConfigurationError, match="do not cover Cord"):
        validate_route_configuration(
            _account(),
            _route(),
            cord=_cord(),
            pricing_rules=(
                {
                    "metric": "request",
                    "unit_price": "0.1",
                    "scope": scope,
                },
            ),
        )


def test_compiler_accepts_pricing_scope_for_public_cord_route():
    validate_route_configuration(
        _account(),
        _route(),
        cord=_cord(),
        pricing_rules=(
            {
                "metric": "request",
                "unit_price": "0.1",
                "scope": {
                    "cord": "generate",
                    "path": "/v1/generate",
                    "method": "post",
                },
            },
        ),
    )


def test_compiler_allows_explicit_absence_pricing_condition_without_source():
    validate_route_configuration(
        _account(),
        _route(),
        pricing_rules=(
            {"metric": "request", "unit_price": "0.05"},
            {
                "metric": "request",
                "unit_price": "0.1",
                "conditions": {"optional.output": {"exists": False}},
            },
        ),
    )


def test_compiler_rejects_conditional_ungrouped_pricing_without_a_baseline():
    with pytest.raises(RouteConfigurationError, match="unconditional rule"):
        validate_route_configuration(
            _account(),
            _route(),
            pricing_rules=(
                {
                    "metric": "request",
                    "unit_price": "0.1",
                    "conditions": {"mode": "premium"},
                },
            ),
        )


@pytest.mark.parametrize(
    ("operation_mode", "response_mode", "message"),
    [
        ("sync", "sse", "sync routes require response mode: buffered"),
        ("stream", "buffered", "stream routes require response mode: sse, stream"),
        ("task", "stream", "task routes require response mode: buffered"),
    ],
)
def test_compiler_rejects_operation_mode_transport_mismatches(
    operation_mode, response_mode, message
):
    operation_config = {}
    if operation_mode == "task":
        operation_config = {"submit_mapping": {"task_id": "task_id"}}
    configured = _route(
        operation_mode=operation_mode,
        response_config={"mode": response_mode},
        operation_config=operation_config,
    )

    with pytest.raises(RouteConfigurationError, match=message):
        validate_route_configuration(_account(), configured)


def test_compiler_rejects_session_budget_on_non_session_route():
    configured = _route(operation_config={"session_budget": {"max_exposure_usd": "1"}})

    with pytest.raises(
        RouteConfigurationError,
        match="only valid for stream and realtime routes",
    ):
        validate_route_configuration(_account(), configured)


def test_compiler_bounds_live_session_budget_check_interval():
    configured = _route(
        operation_mode="stream",
        response_config={"mode": "sse"},
        operation_config={"session_budget": {"check_interval_seconds": 6}},
    )

    with pytest.raises(
        RouteConfigurationError,
        match="check_interval_seconds must be between 0.1 and 5",
    ):
        validate_route_configuration(_account(), configured)
