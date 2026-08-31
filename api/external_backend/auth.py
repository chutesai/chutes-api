"""Authentication-scope routing for external management and operation APIs."""

from __future__ import annotations

from typing import NamedTuple


class ExternalAuthScope(NamedTuple):
    object_type: str
    object_id: str


def external_auth_scope(path: str) -> ExternalAuthScope | None:
    segments = [segment for segment in path.split("/") if segment]
    if len(segments) < 2 or segments[0].lower() != "external":
        return None

    collection = segments[1].lower()
    object_id = segments[2] if len(segments) >= 3 else "__list_or_invalid__"
    if collection == "operations":
        return ExternalAuthScope("invocations", object_id)
    if collection in {"accounts", "bindings"}:
        return ExternalAuthScope("account", object_id)
    if collection == "chutes":
        return ExternalAuthScope("chutes", object_id)
    return None


__all__ = ["ExternalAuthScope", "external_auth_scope"]
