from decimal import Decimal

import pytest

from api.external_backend.mapping import (
    DataPath,
    MappingConfigurationError,
    MappingExtractionError,
    PayloadTransform,
    PublicResponseRules,
    TaskMapping,
    UsageMapping,
    extract_artifacts,
    extract_task,
    extract_usage,
    extract_value,
    extract_values,
    merge_stream_usage,
    scrub_public_response,
    transform_payload,
)
from api.payment.pricing import NormalizedUsage


def test_safe_paths_extract_arrays_wildcards_and_pointer_escapes():
    payload = {
        "items": [
            {"usage": {"value": 2}},
            {"usage": {"value": 3}},
        ],
        "key/with/slash": {"~key": "found"},
        "0": "mapping key",
    }

    assert extract_values(payload, "items[*].usage.value") == (2, 3)
    assert extract_values(payload, "/items/*/usage/value") == (2, 3)
    assert extract_value(payload, "/key~1with~1slash/~0key", required=True) == "found"
    assert extract_value(payload, "/0", required=True) == "mapping key"
    assert extract_value(payload, "missing", default={"safe": True}) == {"safe": True}


@pytest.mark.parametrize(
    "path",
    ["a..b", "a[-1]", "a[nope]", "/bad~2escape", "$broken", "a.constructor.x"],
)
def test_invalid_or_reserved_paths_fail_during_compilation(path):
    with pytest.raises(MappingConfigurationError):
        DataPath.parse(path)


def test_extract_value_rejects_ambiguous_wildcard():
    with pytest.raises(MappingExtractionError, match="more than one"):
        extract_value({"items": [1, 2]}, "items[*]")


def test_payload_transform_removes_injects_rewrites_and_never_mutates_input():
    original = {
        "auth": "remove-me",
        "old_name": "value",
        "items": [{"state": "READY", "private": 1}, {"state": "WAIT", "private": 2}],
    }
    config = {
        "remove": ["auth", "items[*].private"],
        "inject": {
            "resource": "internal-resource",
            "options.mode": {"value": "fast"},
        },
        "rewrite": [
            {
                "target": "name",
                "source": "payload",
                "path": "old_name",
                "remove_source": True,
            },
            {
                "target": "items[*].state",
                "source": "payload",
                "path": "items[0].state",
                "map": {"READY": "running"},
            },
            {
                "target": "from_request",
                "source": "request",
                "path": "client_value",
                "required": True,
            },
        ],
    }

    result = transform_payload(original, config, request={"client_value": 7})

    assert result == {
        "items": [{"state": "running"}, {"state": "running"}],
        "resource": "internal-resource",
        "options": {"mode": "fast"},
        "name": "value",
        "from_request": 7,
    }
    assert original["auth"] == "remove-me"
    assert original["items"][0]["private"] == 1


def test_injection_preserves_existing_value_and_can_fill_each_array_item():
    result = transform_payload(
        {"mode": "client", "items": [{}, {}]},
        {
            "inject": [
                {"target": "mode", "value": "configured"},
                {"target": "items[*].kind", "value": "result"},
            ]
        },
    )

    assert result == {
        "mode": "client",
        "items": [{"kind": "result"}, {"kind": "result"}],
    }


@pytest.mark.parametrize(
    "config",
    [
        {"unknown": []},
        {"remove": "secret"},
        {"rewrite": [{"target": "x", "path": "a", "value": 1}]},
        {"rewrite": [{"target": "$", "value": 1}]},
        {"rewrite": [{"target": "x", "path": "a", "divide": 0}]},
        {"rewrite": [{"target": "x", "path": "a", "aggregate": "unsafe"}]},
    ],
)
def test_payload_transform_configuration_fails_closed(config):
    with pytest.raises(MappingConfigurationError):
        PayloadTransform.from_config(config)


def test_usage_mapping_combines_request_response_wildcards_and_arithmetic():
    request = {"duration": "4", "quality": "high"}
    response = {
        "usage": {"input": 12, "output": "8", "cached": 2},
        "results": [{"seconds": 1.5}, {"seconds": "2.5"}],
    }
    config = {
        "default_requests": 1,
        "fields": {
            "tokens": {
                "input": "usage.input",
                "output": "usage.output",
                "cached_input": "usage.cached",
            },
            "output_media_seconds.video": {
                "path": "results[*].seconds",
                "aggregate": "sum",
            },
            "counts.frames": {
                "source": "request",
                "path": "duration",
                "cast": "number",
                "multiply": 24,
            },
            "images.generated": {
                "path": "results[*]",
                "aggregate": "count",
            },
            "dimensions.quality": {"source": "request", "path": "quality"},
        },
    }

    usage = extract_usage(config, request=request, response=response)

    assert usage.requests == Decimal("1")
    assert usage.tokens == {
        "input": Decimal("12"),
        "output": Decimal("8"),
        "cached_input": Decimal("2"),
    }
    assert usage.output_media_seconds == {"video": Decimal("4.0")}
    assert usage.counts == {"frames": Decimal("96")}
    assert usage.images == {"generated": Decimal("2")}
    assert usage.dimensions == {"quality": "high"}


def test_usage_mapping_supports_fallback_defaults_and_required_values():
    optional = UsageMapping.from_config(
        {
            "default_requests": 0,
            "fields": {
                "counts.steps": {"paths": ["missing", "also_missing"], "default": 4}
            },
        }
    )
    assert optional.extract(response={}).counts == {"steps": Decimal("4")}
    assert optional.extract(response={}).requests == 0

    required = UsageMapping.from_config(
        {"fields": {"tokens.output": {"path": "usage.output", "required": True}}}
    )
    with pytest.raises(MappingExtractionError, match="required"):
        required.extract(response={})


@pytest.mark.parametrize(
    "config",
    [
        {"fields": {"unknown.value": "usage.value"}},
        {"fields": {"tokens.input": {"value": -1}}},
        {"fields": {"tokens.input": {"path": "x", "aggregate": "list"}}},
        {"fields": {"requests": {"value": "NaN"}}},
        {"fields": {}, "unexpected": True},
    ],
)
def test_usage_mapping_configuration_fails_closed(config):
    with pytest.raises(MappingConfigurationError):
        UsageMapping.from_config(config)


def test_usage_extraction_rejects_non_numeric_runtime_values():
    mapping = UsageMapping.from_config({"fields": {"tokens.input": "usage.input"}})

    with pytest.raises(MappingExtractionError, match="invalid"):
        mapping.extract(response={"usage": {"input": "not-a-number"}})


def test_stream_usage_delta_adds_and_cumulative_keeps_monotonic_snapshots():
    previous = NormalizedUsage(
        requests=1,
        tokens={"input": 10, "output": 2},
        counts={"steps": 2},
        dimensions={"mode": "one", "nested": {"a": 1}},
    )
    observation = NormalizedUsage(
        requests=1,
        tokens={"input": 4, "output": 5},
        images={"generated": 1},
        dimensions={"mode": "two", "nested": {"b": 2}},
    )

    delta = merge_stream_usage(previous, observation, "delta")
    cumulative = merge_stream_usage(previous, observation, "cumulative")

    assert delta.requests == 2
    assert delta.tokens == {"input": 14, "output": 7}
    assert delta.counts == {"steps": 2}
    assert cumulative.requests == 1
    assert cumulative.tokens == {"input": 10, "output": 5}
    assert cumulative.images == {"generated": 1}
    assert cumulative.dimensions == {"mode": "two", "nested": {"a": 1, "b": 2}}
    assert merge_stream_usage(None, observation, "delta") is observation


def test_task_mapping_extracts_identity_status_result_and_artifacts():
    response = {
        "task": {"identifier": 123, "state": "COMPLETE"},
        "output": {
            "summary": {"count": 2},
            "files": [
                {
                    "location": "https://objects.invalid/a.mp4",
                    "mime": "video/mp4",
                    "bytes": "12",
                    "expires": "2030-01-01T00:00:00Z",
                    "label": "first",
                },
                {
                    "location": "https://objects.invalid/b.mp4",
                    "mime": "video/mp4",
                    "bytes": 15,
                    "expires": "2030-01-01T00:00:00Z",
                    "label": "second",
                },
            ],
        },
    }
    config = {
        "task_id": {"path": "task.identifier", "required": True},
        "status": {
            "path": "task.state",
            "required": True,
            "map": {"COMPLETE": "succeeded"},
        },
        "result": "output.summary",
        "artifacts": {
            "items": "output.files[*]",
            "url": {"path": "location", "required": True},
            "kind": {"value": "video"},
            "content_type": "mime",
            "size_bytes": "bytes",
            "expires_at": "expires",
            "metadata": {"label": "label"},
            "required": True,
        },
    }

    task = extract_task(config, response=response)

    assert task.task_id == "123"
    assert task.status == "succeeded"
    assert task.result == {"count": 2}
    assert [artifact.source_url for artifact in task.artifacts] == [
        "https://objects.invalid/a.mp4",
        "https://objects.invalid/b.mp4",
    ]
    assert task.artifacts[0].kind == "video"
    assert task.artifacts[0].content_type == "video/mp4"
    assert task.artifacts[0].size_bytes == 12
    assert task.artifacts[1].metadata == {"label": "second"}

    artifacts = extract_artifacts(config["artifacts"], response=response)
    assert artifacts == task.artifacts


def test_task_mapping_rejects_unknown_status_and_missing_required_artifact():
    mapping = TaskMapping.from_config(
        {
            "status": "state",
            "artifacts": {
                "items": "files[*]",
                "url": {"path": "url", "required": True},
                "required": True,
            },
        }
    )

    with pytest.raises(MappingExtractionError, match="status"):
        mapping.extract(response={"state": "UNKNOWN", "files": []})
    with pytest.raises(MappingExtractionError, match="artifact"):
        mapping.extract(response={"state": "running", "files": []})


@pytest.mark.parametrize(
    "config",
    [
        {"status": {"path": "state", "map": {"DONE": "unknown"}}},
        {"artifacts": {"items": "files[*]"}},
        {"artifacts": {"url": "url", "unexpected": True}},
        {"unexpected": "x"},
    ],
)
def test_task_mapping_configuration_fails_closed(config):
    with pytest.raises(MappingConfigurationError):
        TaskMapping.from_config(config)


def test_public_scrubbing_is_recursive_rewrites_keys_and_localizes_artifacts():
    original = {
        "request_id": "private-request",
        "model": "private-model",
        "data": [
            {
                "provider_name": "private-name",
                "url": "https://objects.invalid/one",
                "nested": {"trace-id": "private-trace", "keep": 1},
            },
            {
                "url": "https://objects.invalid/two",
                "secret_note": "remove",
            },
        ],
    }
    config = {
        "remove_keys": ["secret_note"],
        "rewrite_keys": {"model": "public-model"},
        "artifact_paths": ["data[*].url"],
    }

    result = scrub_public_response(
        original,
        config,
        artifact_urls=[
            "/operations/local/artifacts/0",
            "/operations/local/artifacts/1",
        ],
    )

    assert result == {
        "model": "public-model",
        "data": [
            {"url": "/operations/local/artifacts/0", "nested": {"keep": 1}},
            {"url": "/operations/local/artifacts/1"},
        ],
    }
    assert original["request_id"] == "private-request"
    assert original["data"][0]["url"] == "https://objects.invalid/one"


def test_public_scrubbing_can_replace_exact_urls_without_exposing_unmapped_values():
    response = {
        "result": {
            "download": "https://objects.invalid/file",
            "copies": ["https://objects.invalid/file"],
        }
    }
    result = scrub_public_response(
        response,
        artifact_urls={"https://objects.invalid/file": "/artifacts/local"},
    )
    assert result == {
        "result": {"download": "/artifacts/local", "copies": ["/artifacts/local"]}
    }

    with pytest.raises(MappingExtractionError, match="replacement"):
        scrub_public_response(
            response,
            {"artifact_paths": ["result.download"]},
            artifact_urls={"https://objects.invalid/other": "/artifacts/local"},
        )


def test_public_scrubbing_removes_credential_like_keys_recursively():
    response = {
        "authorization": "Bearer private",
        "api_key": "private",
        "password": "private",
        "nested": {
            "headers": {
                "X-Api-Key": "private",
                "Set-Cookie": "private",
                "Content-Type": "application/json",
            },
            "result": "visible",
        },
    }

    assert scrub_public_response(response) == {
        "nested": {
            "headers": {"Content-Type": "application/json"},
            "result": "visible",
        }
    }


def test_public_scrubbing_removes_structural_identity_and_unmapped_operation_links():
    response = {
        "vendor_request_id": "private-request",
        "remote_provider": "private-provider",
        "service_endpoint": "https://private.example.test",
        "nested": {
            "status_url": "https://private.example.test/tasks/one",
            "href": "https://private.example.test/tasks/one",
            "result_url": "https://private.example.test/files/one",
            "url": "https://private.example.test/files/two",
            "output_url": "https://private.example.test/files/three",
            "x-request-id": "private-request-two",
            "output": "visible",
        },
    }

    assert scrub_public_response(response) == {"nested": {"output": "visible"}}


def test_public_scrubbing_preserves_a_structural_link_only_after_local_artifact_rewrite():
    response = {"result_url": "https://private.example.test/files/one"}

    assert scrub_public_response(
        response,
        {"artifact_paths": ["result_url"]},
        artifact_urls=["https://api.example.test/external/operations/one/artifacts/0"],
    ) == {"result_url": "https://api.example.test/external/operations/one/artifacts/0"}


def test_public_scrubbing_drops_absolute_upstream_urls_under_generic_keys():
    response = {
        "output": "https://private.example.test/tasks/one",
        "download": "s3://private-bucket/result",
        "values": ["visible", "wss://private.example.test/session"],
        "message": "visible text remains intact",
    }

    assert scrub_public_response(response) == {
        "values": ["visible"],
        "message": "visible text remains intact",
    }


def test_public_scrubbing_applies_path_removals_and_rewrites_after_recursive_policy():
    rules = PublicResponseRules.from_config(
        {
            "remove_paths": ["meta.internal"],
            "rewrite": [
                {
                    "target": "state",
                    "path": "raw_state",
                    "map": {"DONE": "succeeded"},
                    "remove_source": True,
                }
            ],
        }
    )

    assert rules.scrub({"meta": {"internal": 1, "public": 2}, "raw_state": "DONE"}) == {
        "meta": {"public": 2},
        "state": "succeeded",
    }


def test_public_rewrite_cannot_reintroduce_private_sources_or_reserved_targets():
    with pytest.raises(MappingConfigurationError, match="scrubbed payload"):
        PublicResponseRules.from_config(
            {
                "rewrite": [
                    {
                        "target": "output",
                        "source": "response",
                        "path": "authorization",
                    }
                ]
            }
        )

    rules = PublicResponseRules.from_config(
        {
            "rewrite_keys": {"model": "public-model"},
            "rewrite": [
                {"target": "model", "path": "output"},
                {"target": "status_url", "path": "output"},
                {"target": "leaked_auth", "path": "output"},
            ],
        }
    )
    assert rules.scrub({"output": "visible"}) == {
        "model": "public-model",
        "output": "visible",
    }


def test_public_scrubbing_rejects_bad_configuration_and_structural_overflow():
    with pytest.raises(MappingConfigurationError):
        PublicResponseRules.from_config({"artifact_paths": "result.url"})
    with pytest.raises(MappingConfigurationError):
        PublicResponseRules.from_config({"remove_keys": [], "max_depth": 0})

    rules = PublicResponseRules.from_config({"max_depth": 2})
    with pytest.raises(MappingExtractionError, match="structural"):
        rules.scrub({"one": {"two": {"three": True}}})
