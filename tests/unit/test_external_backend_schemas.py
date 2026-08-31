from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from pydantic import ValidationError

from api.external_backend.schemas import (
    ExternalAccountBulkCancelRequest,
    ExternalBackendAccount,
    ExternalBackendAccountCreate,
    ExternalBackendAccountResponse,
    ExternalBackendAccountUpdate,
    ExternalChuteBinding,
    ExternalChuteBindingCreate,
    ExternalChuteBindingUpdate,
    ExternalChuteCreate,
    ExternalChuteUpdate,
    ExternalCredentialForceRotateRequest,
    ExternalCord,
    ExternalOperation,
    ExternalOperationMode,
    ExternalOperationStatus,
    ExternalOperationUpdate,
    ExternalResultStatus,
    ExternalSettlementStatus,
    ExternalSettlementRetryRequest,
    ExternalSettlementWriteOffRequest,
)


def _account_payload() -> dict:
    return {
        "name": "primary",
        "adapter": "generic-http",
        "base_url": "https://service.example.test/v1",
        "credentials": {
            "access": "write-only-access-value",
            "signature": "write-only-signature-value",
        },
        "auth_header_templates": [
            {
                "name": "Authorization",
                "template": "Bearer {token}",
                "references": {"token": "access"},
            },
            {
                "name": "X-Signature",
                "template": "{value}",
                "references": {"value": "signature"},
            },
        ],
        "connection_config": {"region": "test-region", "timeout_seconds": 300},
    }


def _route(cord_path: str, mode: str, *, base_url: str | None = None) -> dict:
    return {
        "cord_path": cord_path,
        "upstream_resource_id": f"resource-{mode}",
        "operation_mode": mode,
        "protocol": "generic-json",
        "base_url": base_url,
        "path_template": f"/operations/{mode}",
        "method": "post",
        "request_config": {"body_mode": "json"},
        "response_config": {"codec": "json"},
        "operation_config": {"timeout_seconds": 300},
        "capabilities": {"mode": mode},
    }


def test_account_supports_multiple_secret_references_and_write_only_response():
    request = ExternalBackendAccountCreate.model_validate(_account_payload())
    assert set(request.credentials) == {"access", "signature"}
    assert request.auth_header_templates[0].references == {"token": "access"}

    now = datetime.now(UTC)
    response = ExternalBackendAccountResponse.model_validate(
        {
            "account_id": "account-id",
            "user_id": "user-id",
            "created_at": now,
            "updated_at": now,
            **request.model_dump(mode="json", exclude={"credentials"}),
            "credential_references": {
                "access": "secret://encrypted/access",
                "signature": "secret://encrypted/signature",
            },
        }
    )
    dumped = response.model_dump(mode="json")

    assert "credential_references" not in dumped
    assert dumped["credential_configured"] is True
    assert dumped["credential_names"] == ["access", "signature"]
    assert dumped["auth_header_templates"][0]["references"] == {"token": "access"}


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("credentials", {"access": ""}),
        ("base_url", "https://username:password@service.example.test"),
        ("connection_config", {"nested": {"api-key": "plain-text-value"}}),
    ],
)
def test_account_rejects_inline_credentials(field, value):
    payload = _account_payload()
    payload[field] = value

    with pytest.raises(ValidationError):
        ExternalBackendAccountCreate.model_validate(payload)


def test_account_accepts_credentials_as_masked_write_only_values():
    request = ExternalBackendAccountCreate.model_validate(_account_payload())

    assert request.credentials["access"].get_secret_value() == "write-only-access-value"
    assert "write-only-access-value" not in repr(request)
    assert request.model_dump(mode="json")["credentials"]["access"] == "**********"


@pytest.mark.parametrize(
    "model",
    [ExternalBackendAccountCreate, ExternalBackendAccountUpdate],
)
def test_account_rejects_non_boolean_insecure_transport_flag(model):
    payload = (
        _account_payload()
        if model is ExternalBackendAccountCreate
        else {"connection_config": {"allow_insecure_http": "false"}}
    )
    if model is ExternalBackendAccountCreate:
        payload["connection_config"] = {"allow_insecure_http": "false"}

    with pytest.raises(ValidationError, match="allow_insecure_http must be a boolean"):
        model.model_validate(payload)


def test_account_rejects_unknown_or_duplicate_auth_references():
    payload = _account_payload()
    payload["auth_header_templates"][0]["references"] = {"token": "missing"}
    with pytest.raises(ValidationError, match="unknown credential"):
        ExternalBackendAccountCreate.model_validate(payload)

    payload = _account_payload()
    payload["auth_header_templates"].append(payload["auth_header_templates"][0])
    with pytest.raises(ValidationError, match="unique header names"):
        ExternalBackendAccountCreate.model_validate(payload)


def test_binding_supports_multiple_route_modes_and_endpoint_override():
    binding = ExternalChuteBindingCreate.model_validate(
        {
            "chute_id": "chute-id",
            "account_id": "account-id",
            "routes": [
                _route("/generate", ExternalOperationMode.SYNC.value),
                _route(
                    "/events",
                    ExternalOperationMode.STREAM.value,
                    base_url="https://stream.example.test/v2",
                ),
                _route("/jobs", ExternalOperationMode.TASK.value),
                _route("/session", ExternalOperationMode.REALTIME.value),
            ],
        }
    )

    assert {route.operation_mode for route in binding.routes} == set(
        ExternalOperationMode
    )
    assert str(binding.routes[1].base_url) == "https://stream.example.test/v2"
    assert all(route.method == "POST" for route in binding.routes)


def test_binding_rejects_duplicate_cord_paths_and_secret_config():
    duplicate = _route("/generate", "sync")
    with pytest.raises(ValidationError, match="unique cord_path"):
        ExternalChuteBindingCreate.model_validate(
            {
                "chute_id": "chute-id",
                "account_id": "account-id",
                "routes": [duplicate, duplicate],
            }
        )


def test_external_chute_requires_matching_routes_and_valid_pricing():
    payload = {
        "account_id": "account-id",
        "name": "external-test",
        "cords": [
            {
                "method": "POST",
                "path": "/generate",
                "function": "generate",
                "stream": False,
                "public_api_path": "/external-test/generate",
                "public_api_method": "POST",
            }
        ],
        "routes": [_route("/generate", "sync")],
        "pricing_rules": [{"metric": "request", "unit_price": "0.1"}],
    }
    created = ExternalChuteCreate.model_validate(payload)
    assert created.pricing_rules[0]["metric"] == "request"

    payload["routes"][0]["method"] = "GET"
    translated = ExternalChuteCreate.model_validate(payload)
    assert translated.cords[0].public_api_method == "POST"
    assert translated.routes[0].method == "GET"

    payload["pricing_rules"] = [{"metric": "unsupported", "unit_price": 1}]
    with pytest.raises(ValidationError, match="Unsupported usage metric"):
        ExternalChuteCreate.model_validate(payload)

    unsafe = _route("/generate", "sync")
    unsafe["request_config"] = {"headers": {"authorization": "plain-text-value"}}
    with pytest.raises(ValidationError, match="credential values"):
        ExternalChuteBindingCreate.model_validate(
            {
                "chute_id": "chute-id",
                "account_id": "account-id",
                "routes": [unsafe],
            }
        )


def _cord_payload(**overrides) -> dict:
    return {
        "method": "POST",
        "path": "/generate",
        "function": "generate",
        "stream": False,
        "public_api_path": "/generate",
        "public_api_method": "POST",
        **overrides,
    }


def _credential_named_json_schema() -> dict:
    return {
        "type": "object",
        "properties": {
            "password": {"type": "string"},
            "api_key": {"type": "string"},
        },
        "required": ["password", "api_key"],
    }


def test_external_cord_treats_schemas_as_bounded_metadata_not_credentials():
    schema = _credential_named_json_schema()

    cord = ExternalCord.model_validate(
        _cord_payload(
            input_schema=schema,
            minimal_input_schema=schema,
            output_schema=schema,
        )
    )

    assert set(cord.input_schema["properties"]) == {"password", "api_key"}
    assert set(cord.minimal_input_schema["properties"]) == {
        "password",
        "api_key",
    }
    assert set(cord.output_schema["properties"]) == {"password", "api_key"}


@pytest.mark.parametrize(
    "operation_config",
    [
        {
            "submission_contract": {
                "enabled": False,
                "output_schema": _credential_named_json_schema(),
            }
        },
        {
            "realtime": {
                "validate_client_messages": False,
                "message_schema": _credential_named_json_schema(),
            }
        },
        {
            "websocket": {
                "validate_client_messages": False,
                "message_schema": _credential_named_json_schema(),
            }
        },
    ],
)
def test_disabled_route_schemas_are_inert_metadata(operation_config):
    route = _route("/generate", "task")
    route["operation_config"] = operation_config

    created = ExternalChuteBindingCreate.model_validate(
        {
            "chute_id": "chute-id",
            "account_id": "account-id",
            "routes": [route],
        }
    )

    assert created.routes[0].operation_config == operation_config


def test_schema_metadata_exemption_does_not_cover_sibling_route_config():
    route = _route("/generate", "task")
    route["operation_config"] = {
        "submission_contract": {
            "enabled": False,
            "output_schema": _credential_named_json_schema(),
        },
        "password": "plain-text-value",
    }

    with pytest.raises(ValidationError, match="credential values"):
        ExternalChuteBindingCreate.model_validate(
            {
                "chute_id": "chute-id",
                "account_id": "account-id",
                "routes": [route],
            }
        )


def test_schema_metadata_still_requires_bounded_json_values():
    with pytest.raises(ValidationError, match="JSON-compatible"):
        ExternalCord.model_validate(_cord_payload(input_schema={"const": object()}))

    too_deep: dict = {}
    cursor = too_deep
    for _ in range(65):
        nested: dict = {}
        cursor["properties"] = nested
        cursor = nested
    with pytest.raises(ValidationError, match="JSON complexity limit"):
        ExternalCord.model_validate(_cord_payload(input_schema=too_deep))

    with pytest.raises(ValidationError, match="JSON complexity limit"):
        ExternalCord.model_validate(
            _cord_payload(input_schema={"enum": list(range(10_000))})
        )

    route = _route("/generate", "task")
    route["operation_config"] = {
        "submission_contract": {"enabled": False, "output_schema": []}
    }
    with pytest.raises(ValidationError, match="output_schema must be an object"):
        ExternalChuteBindingCreate.model_validate(
            {
                "chute_id": "chute-id",
                "account_id": "account-id",
                "routes": [route],
            }
        )


@pytest.mark.parametrize("method", ["GET", "POST", "PUT", "PATCH", "DELETE", "HEAD"])
def test_external_cord_supports_all_gateway_http_methods(method):
    payload = {
        "account_id": "account-id",
        "name": f"external-{method.lower()}",
        "cords": [
            {
                "method": method,
                "path": "/generate",
                "function": "generate",
                "stream": False,
                "public_api_path": "/generate",
                "public_api_method": method,
            }
        ],
        "routes": [{**_route("/generate", "sync"), "method": method}],
        "pricing_rules": [{"metric": "request", "unit_price": "0.1"}],
    }

    created = ExternalChuteCreate.model_validate(payload)

    assert created.cords[0].public_api_method == method
    assert created.routes[0].method == method


@pytest.mark.parametrize(
    "reserved_path", ["/evidence", "/EVIDENCE", "/hf_info", "/HF_INFO"]
)
@pytest.mark.parametrize("operation", ["create", "update"])
def test_external_chute_rejects_reserved_canonical_cord_paths(reserved_path, operation):
    cord = {
        "method": "POST",
        "path": "/generate",
        "function": "generate",
        "stream": False,
        "public_api_path": reserved_path,
        "public_api_method": "POST",
    }
    routes = [_route("/generate", "sync")]
    pricing_rules = [{"metric": "request", "unit_price": "0.1"}]
    if operation == "create":
        model = ExternalChuteCreate
        payload = {
            "account_id": "account-id",
            "name": "external-reserved-path",
            "cords": [cord],
            "routes": routes,
            "pricing_rules": pricing_rules,
        }
    else:
        model = ExternalChuteUpdate
        payload = {
            "cords": [cord],
            "routes": routes,
            "pricing_rules": pricing_rules,
        }

    with pytest.raises(
        ValidationError, match="reserved for a canonical Chute endpoint"
    ):
        model.model_validate(payload)


@pytest.mark.parametrize("public_path", ["/nested/evidence", "/nested/HF_INFO"])
def test_external_chute_allows_nested_management_path_names(public_path):
    payload = {
        "account_id": "account-id",
        "name": "external-nested-path",
        "cords": [
            {
                "method": "POST",
                "path": "/generate",
                "function": "generate",
                "stream": False,
                "public_api_path": public_path,
                "public_api_method": "POST",
            }
        ],
        "routes": [_route("/generate", "sync")],
        "pricing_rules": [{"metric": "request", "unit_price": "0.1"}],
    }

    created = ExternalChuteCreate.model_validate(payload)

    assert created.cords[0].public_api_path == public_path


@pytest.mark.parametrize("header_name", ["X-API-Key", "X-Subscription-Key"])
def test_route_rejects_plaintext_credential_headers_before_persistence(header_name):
    route = _route("/generate", "sync")
    route["request_config"] = {"static_headers": {header_name: "plain-text-credential"}}

    with pytest.raises(ValidationError, match="credential values"):
        ExternalChuteBindingCreate.model_validate(
            {
                "chute_id": "chute-id",
                "account_id": "account-id",
                "routes": [route],
            }
        )


@pytest.mark.parametrize("query_name", ["signature", "x-api-key", "access_token"])
def test_route_rejects_plaintext_credential_query_values_before_persistence(query_name):
    route = _route("/generate", "sync")
    route["request_config"] = {
        "body_mode": "json",
        "query_parameters": {query_name: {"value": "plain-text-value"}},
    }

    with pytest.raises(ValidationError, match="credential values"):
        ExternalChuteBindingCreate.model_validate(
            {
                "chute_id": "chute-id",
                "account_id": "account-id",
                "routes": [route],
            }
        )


def test_route_allows_non_header_configuration_with_key_like_names():
    route = _route("/generate", "sync")
    route["request_config"] = {
        "auth_allowed_hosts": ["service.example.test"],
        "authenticated": True,
        "idempotency_key": "request-id",
        "static_headers": {"X-Feature-Mode": "fast"},
    }

    created = ExternalChuteBindingCreate.model_validate(
        {
            "chute_id": "chute-id",
            "account_id": "account-id",
            "routes": [route],
        }
    )

    assert created.routes[0].request_config["authenticated"] is True


def test_external_chute_standard_template_is_optional_and_validated():
    payload = {
        "account_id": "account-id",
        "name": "external-llm",
        "standard_template": "vllm",
        "cords": [
            {
                "method": "POST",
                "path": "/chat",
                "function": "chat",
                "stream": False,
                "public_api_path": "/v1/chat/completions",
                "public_api_method": "POST",
            }
        ],
        "routes": [_route("/chat", "sync")],
        "pricing_rules": [{"metric": "request", "unit_price": "0.1"}],
    }

    assert ExternalChuteCreate.model_validate(payload).standard_template == "vllm"
    assert (
        ExternalChuteUpdate.model_validate(
            {"standard_template": None}
        ).standard_template
        is None
    )

    payload["standard_template"] = "provider-specific-template"
    with pytest.raises(ValidationError, match="Invalid standard template"):
        ExternalChuteCreate.model_validate(payload)


def test_external_chute_requires_unique_cord_paths():
    cord = {
        "method": "POST",
        "path": "/chat",
        "function": "chat",
        "stream": False,
        "public_api_path": "/v1/chat/completions",
        "public_api_method": "POST",
    }
    payload = {
        "account_id": "account-id",
        "name": "external-llm",
        "cords": [cord, {**cord, "function": "chat_stream", "stream": True}],
        "routes": [_route("/chat", "sync")],
        "pricing_rules": [{"metric": "request", "unit_price": "0.1"}],
    }

    with pytest.raises(ValidationError, match="cord paths must be unique"):
        ExternalChuteCreate.model_validate(payload)


def test_external_chute_rejects_ambiguous_public_cord_selectors():
    first = {
        "method": "POST",
        "path": "/chat",
        "function": "chat",
        "stream": False,
        "public_api_path": "/v1/chat/completions",
        "public_api_method": "POST",
    }
    second = {**first, "path": "/chat_again", "function": "chat_again"}
    payload = {
        "account_id": "account-id",
        "name": "external-llm",
        "cords": [first, second],
        "routes": [_route("/chat", "sync"), _route("/chat_again", "sync")],
        "pricing_rules": [{"metric": "request", "unit_price": "0.1"}],
    }

    with pytest.raises(ValidationError, match="selectors must be unique"):
        ExternalChuteCreate.model_validate(payload)

    payload["cords"][1]["stream"] = True
    assert len(ExternalChuteCreate.model_validate(payload).cords) == 2


def test_update_models_reject_null_for_non_nullable_configuration():
    with pytest.raises(ValidationError, match="base_url cannot be null"):
        ExternalBackendAccountUpdate.model_validate({"base_url": None})

    with pytest.raises(ValidationError, match="enabled cannot be null"):
        ExternalChuteBindingUpdate.model_validate({"enabled": None})


def test_operation_relationships_do_not_eager_load_sensitive_account_graphs():
    relationships = ExternalOperation.__mapper__._props
    assert relationships["user"].lazy == "select"
    assert relationships["account"].lazy == "select"
    assert relationships["binding"].lazy == "select"


def test_operation_result_supports_partial_multi_artifact_output():
    expires_at = datetime.now(UTC) + timedelta(hours=1)
    update = ExternalOperationUpdate.model_validate(
        {
            "status": "succeeded",
            "settlement_status": "pending",
            "usage": {
                "requests": 1,
                "tokens": {"input": 10},
                "output_media_seconds": {"video": 5.5},
                "dimensions": {"cache": "miss"},
            },
            "result_descriptor": {
                "status": "partial",
                "artifacts": [
                    {
                        "kind": "video",
                        "reference": "objects/result-1",
                        "content_type": "video/mp4",
                        "size_bytes": 1024,
                        "expires_at": expires_at,
                        "attributes": {
                            "local_path": "/external/operations/id/artifacts/0"
                        },
                    },
                    {
                        "kind": "preview",
                        "reference": "objects/result-1-preview",
                        "content_type": "image/webp",
                        "size_bytes": 128,
                    },
                ],
                "metadata": {"expected_artifacts": 3},
            },
        }
    )

    assert update.status is ExternalOperationStatus.SUCCEEDED
    assert update.settlement_status is ExternalSettlementStatus.PENDING
    assert update.result_descriptor.status is ExternalResultStatus.PARTIAL
    assert len(update.result_descriptor.artifacts) == 2
    assert update.usage.tokens == {"input": 10.0}


def test_quarantine_operator_requests_require_a_nonempty_audit_reason():
    assert ExternalSettlementStatus("quarantined") is (
        ExternalSettlementStatus.QUARANTINED
    )
    retry = ExternalSettlementRetryRequest.model_validate(
        {
            "reason": "  corrected from invoice  ",
            "usage": {"requests": 1, "tokens": {"output": 42}},
        }
    )
    assert retry.reason == "corrected from invoice"
    assert retry.usage.tokens == {"output": 42.0}

    with pytest.raises(ValidationError, match="reason cannot be empty"):
        ExternalSettlementWriteOffRequest(reason="   ")

    with pytest.raises(ValidationError, match="must be supplied together"):
        ExternalSettlementRetryRequest(
            reason="incomplete pricing correction",
            pricing_snapshot={"source": "legacy"},
        )


def test_account_emergency_requests_are_audited_and_credentials_are_write_only():
    cancel = ExternalAccountBulkCancelRequest(reason="  provider incident  ")
    rotation = ExternalCredentialForceRotateRequest(
        reason="  key exposure  ", credentials={"primary": "replacement-value"}
    )

    assert cancel.reason == "provider incident"
    assert rotation.reason == "key exposure"
    assert rotation.credentials["primary"].get_secret_value() == "replacement-value"
    assert "replacement-value" not in repr(rotation)
    with pytest.raises(ValidationError, match="reason cannot be empty"):
        ExternalAccountBulkCancelRequest(reason="   ")


def test_public_operation_schema_never_serializes_upstream_artifact_references():
    from api.external_backend.schemas import ExternalOperationResponse

    response = ExternalOperationResponse.model_validate(
        {
            "operation_id": "operation-id",
            "user_id": "user-id",
            "chute_id": "chute-id",
            "cord_path": "/generate",
            "operation_mode": "task",
            "protocol": "generic-json",
            "status": "succeeded",
            "settlement_status": "settled",
            "settlement_metadata": {
                "pricing": {"source": "legacy", "legacy": {"per_request": 1}}
            },
            "upstream_status": "done",
            "usage": None,
            "result_descriptor": {
                "status": "complete",
                "artifacts": [
                    {
                        "kind": "video",
                        "reference": "https://private-upstream.example/result.mp4",
                        "content_type": "video/mp4",
                        "attributes": {"provider_job": "private"},
                    }
                ],
                "metadata": {"provider": "private"},
            },
            "error": None,
            "created_at": datetime.now(UTC),
            "updated_at": datetime.now(UTC),
            "submitted_at": None,
            "started_at": None,
            "finished_at": datetime.now(UTC),
            "expires_at": None,
            "settled_at": datetime.now(UTC),
        }
    )

    serialized = response.model_dump(mode="json")
    result = serialized["result_descriptor"]
    assert "upstream_status" not in serialized
    assert "settlement_metadata" not in serialized
    assert response.pricing_snapshot_sha256 is not None
    assert serialized["pricing_snapshot_sha256"] == response.pricing_snapshot_sha256
    assert "metadata" not in result
    assert "reference" not in result["artifacts"][0]
    assert "attributes" not in result["artifacts"][0]


def test_operation_descriptors_reject_invalid_quantities_and_empty_partial_result():
    with pytest.raises(ValidationError, match="non-negative"):
        ExternalOperationUpdate.model_validate(
            {"usage": {"output_media_seconds": {"video": -1}}}
        )

    with pytest.raises(ValidationError, match="at least one artifact"):
        ExternalOperationUpdate.model_validate(
            {"result_descriptor": {"status": "partial", "artifacts": []}}
        )

    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        ExternalOperationUpdate.model_validate(
            {
                "result_descriptor": {
                    "artifacts": [
                        {
                            "kind": "video",
                            "reference": "objects/result",
                            "local_path": "/external/operations/id/artifacts/0",
                        }
                    ]
                }
            }
        )


def test_orm_shape_keeps_public_and_upstream_ids_separate():
    account_columns = set(ExternalBackendAccount.__table__.columns.keys())
    binding_columns = set(ExternalChuteBinding.__table__.columns.keys())
    operation_columns = set(ExternalOperation.__table__.columns.keys())

    assert "credential_references" in account_columns
    assert "management_metadata" in account_columns
    assert "artifact_relay_invalidated_at" in account_columns
    assert "credential_value" not in account_columns
    assert "api_key" not in account_columns
    assert binding_columns >= {"binding_id", "chute_id", "account_id", "routes"}
    assert operation_columns >= {
        "operation_id",
        "upstream_operation_id",
        "cord_path",
        "route_snapshot",
        "usage",
        "result_descriptor",
        "expires_at",
        "settlement_status",
    }

    binding_account_fk = next(
        iter(ExternalChuteBinding.__table__.c.account_id.foreign_keys)
    )
    account_user_fk = next(
        iter(ExternalBackendAccount.__table__.c.user_id.foreign_keys)
    )
    operation_user_fk = next(iter(ExternalOperation.__table__.c.user_id.foreign_keys))
    operation_account_fk = next(
        iter(ExternalOperation.__table__.c.account_id.foreign_keys)
    )
    operation_binding_fk = next(
        iter(ExternalOperation.__table__.c.binding_id.foreign_keys)
    )
    operation_chute_fk = next(iter(ExternalOperation.__table__.c.chute_id.foreign_keys))
    assert account_user_fk.ondelete == "RESTRICT"
    assert operation_user_fk.ondelete == "SET NULL"
    assert ExternalOperation.__table__.c.user_id.nullable is True
    assert binding_account_fk.ondelete == "RESTRICT"
    assert operation_account_fk.ondelete == "SET NULL"
    assert ExternalOperation.__table__.c.account_id.nullable is True
    assert operation_binding_fk.ondelete == "SET NULL"
    assert operation_chute_fk.ondelete == "SET NULL"

    idempotency_index = next(
        index
        for index in ExternalOperation.__table__.indexes
        if index.name == "uq_external_operations_idempotency_key"
    )
    assert [column.name for column in idempotency_index.columns] == [
        "binding_id",
        "user_id",
        "idempotency_key",
    ]


def test_emergency_control_migration_matches_account_orm():
    migration = (
        Path(__file__).parents[2]
        / "api/migrations/20260830120500_external_account_emergency_controls.sql"
    ).read_text()

    assert "management_metadata JSONB NOT NULL DEFAULT '{}'::jsonb" in migration
    assert "artifact_relay_invalidated_at TIMESTAMPTZ" in migration
    assert "ck_external_backend_accounts_management_metadata" in migration
