import pytest

from api.external_backend.public_urls import (
    artifact_path,
    artifact_url,
    external_api_origin,
    operation_path,
    operation_url,
)


def test_external_public_urls_are_canonical_and_escape_identifiers():
    assert external_api_origin(base_domain="Example.TEST") == "https://api.example.test"
    assert operation_path("operation/id") == "/external/operations/operation%2Fid"
    assert operation_url("operation/id", base_domain="example.test") == (
        "https://api.example.test/external/operations/operation%2Fid"
    )
    assert artifact_path("operation/id", 2) == (
        "/external/operations/operation%2Fid/artifacts/2"
    )
    assert artifact_url("operation/id", 2, base_domain="example.test") == (
        "https://api.example.test/external/operations/operation%2Fid/artifacts/2"
    )


@pytest.mark.parametrize("domain", ["", "example.test/path", "user@example.test"])
def test_external_api_origin_rejects_invalid_domains(domain):
    with pytest.raises(ValueError, match="base domain"):
        external_api_origin(base_domain=domain)


@pytest.mark.parametrize("index", [-1, True])
def test_artifact_urls_reject_invalid_indexes(index):
    with pytest.raises(ValueError, match="artifact index"):
        artifact_path("operation-id", index)
