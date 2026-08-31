from datetime import datetime, timedelta, timezone

import pytest

from api.external_backend.artifact_policy import (
    ArtifactPolicyError,
    normalize_artifact_expiration,
)


NOW = datetime(2030, 1, 1, tzinfo=timezone.utc)


def test_artifact_expiration_defaults_to_configured_relay_window():
    assert normalize_artifact_expiration(
        None, {"artifact_relay_ttl_seconds": 600}, now=NOW
    ) == NOW + timedelta(minutes=10)


def test_upstream_artifact_expiration_can_shorten_but_not_extend_window():
    config = {"artifact_relay_ttl_seconds": 600}

    assert normalize_artifact_expiration(
        NOW + timedelta(minutes=5), config, now=NOW
    ) == NOW + timedelta(minutes=5)
    assert normalize_artifact_expiration(
        (NOW + timedelta(days=365)).isoformat(), config, now=NOW
    ) == NOW + timedelta(minutes=10)


@pytest.mark.parametrize("value", ["not-a-date", "2030-01-01T00:00:00"])
def test_artifact_expiration_rejects_invalid_or_naive_upstream_values(value):
    with pytest.raises(ArtifactPolicyError):
        normalize_artifact_expiration(value, {}, now=NOW)
