"""Provider-neutral lifetime policy for remotely retained result artifacts."""

from __future__ import annotations

import math
from collections.abc import Mapping
from datetime import datetime, timedelta, timezone
from typing import Any


DEFAULT_ARTIFACT_RELAY_TTL_SECONDS = 24 * 3600
MAX_ARTIFACT_RELAY_TTL_SECONDS = 30 * 24 * 3600


class ArtifactPolicyError(ValueError):
    """Artifact relay lifetime configuration is invalid."""


def artifact_relay_ttl_seconds(response_config: Mapping[str, Any] | None) -> float:
    config = dict(response_config or {})
    configured = config.get(
        "artifact_relay_ttl_seconds", DEFAULT_ARTIFACT_RELAY_TTL_SECONDS
    )
    if isinstance(configured, bool) or not isinstance(configured, (int, float)):
        raise ArtifactPolicyError("artifact_relay_ttl_seconds must be a number")
    value = float(configured)
    if not math.isfinite(value) or value < 60 or value > MAX_ARTIFACT_RELAY_TTL_SECONDS:
        raise ArtifactPolicyError(
            "artifact_relay_ttl_seconds must be between 60 and 2592000"
        )
    return value


def default_artifact_expiration(
    response_config: Mapping[str, Any] | None, *, now: datetime
) -> datetime:
    return now + timedelta(seconds=artifact_relay_ttl_seconds(response_config))


def normalize_artifact_expiration(
    value: datetime | str | None,
    response_config: Mapping[str, Any] | None,
    *,
    now: datetime,
) -> datetime:
    """Return an aware, policy-bounded expiration for an upstream artifact.

    Upstream metadata is advisory.  It may shorten the relay window, but it may
    never extend the operator-configured lifetime or create an unbounded
    credential-bearing relay.
    """

    if now.tzinfo is None or now.utcoffset() is None:
        raise ArtifactPolicyError("artifact expiration reference time must be aware")
    maximum = default_artifact_expiration(response_config, now=now)
    if value is None:
        return maximum
    parsed = value
    if isinstance(parsed, str):
        try:
            parsed = datetime.fromisoformat(parsed.replace("Z", "+00:00"))
        except ValueError as exc:
            raise ArtifactPolicyError(
                "artifact expires_at must be an ISO-8601 timestamp"
            ) from exc
    if (
        not isinstance(parsed, datetime)
        or parsed.tzinfo is None
        or parsed.utcoffset() is None
    ):
        raise ArtifactPolicyError("artifact expires_at must include a timezone")
    return min(parsed.astimezone(timezone.utc), maximum.astimezone(timezone.utc))


__all__ = [
    "ArtifactPolicyError",
    "DEFAULT_ARTIFACT_RELAY_TTL_SECONDS",
    "MAX_ARTIFACT_RELAY_TTL_SECONDS",
    "artifact_relay_ttl_seconds",
    "default_artifact_expiration",
    "normalize_artifact_expiration",
]
