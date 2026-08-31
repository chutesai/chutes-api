"""Canonical public URLs for externally executed operations and artifacts."""

from __future__ import annotations

import re
from urllib.parse import quote

from api.config import settings


def _path_segment(value: str | int) -> str:
    return quote(str(value), safe="")


def external_api_origin(*, base_domain: str | None = None) -> str:
    configured = settings.base_domain if base_domain is None else base_domain
    domain = str(configured or "").strip().strip(".")
    if (
        not domain
        or len(domain) > 253
        or not re.fullmatch(r"[A-Za-z0-9](?:[A-Za-z0-9.-]*[A-Za-z0-9])?", domain)
        or any(not label or len(label) > 63 for label in domain.split("."))
    ):
        raise ValueError("base domain is invalid")
    return f"https://api.{domain.lower()}"


def operation_path(operation_id: str) -> str:
    return f"/external/operations/{_path_segment(operation_id)}"


def operation_url(operation_id: str, *, base_domain: str | None = None) -> str:
    return (
        f"{external_api_origin(base_domain=base_domain)}{operation_path(operation_id)}"
    )


def artifact_path(operation_id: str, artifact_index: int) -> str:
    if (
        isinstance(artifact_index, bool)
        or not isinstance(artifact_index, int)
        or artifact_index < 0
    ):
        raise ValueError("artifact index must be non-negative")
    return f"{operation_path(operation_id)}/artifacts/{_path_segment(artifact_index)}"


def artifact_url(
    operation_id: str,
    artifact_index: int,
    *,
    base_domain: str | None = None,
) -> str:
    return (
        f"{external_api_origin(base_domain=base_domain)}"
        f"{artifact_path(operation_id, artifact_index)}"
    )


__all__ = [
    "artifact_path",
    "artifact_url",
    "external_api_origin",
    "operation_path",
    "operation_url",
]
