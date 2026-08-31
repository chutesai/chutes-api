"""Backend-specific policy for the shared Chute metadata update endpoint."""

from collections.abc import Mapping, Set
from typing import Any


EXTERNAL_CATALOG_UPDATE_FIELDS = (
    "tagline",
    "readme",
    "tool_description",
    "logo_id",
)
EXTERNAL_HOSTED_ONLY_UPDATE_FIELDS = frozenset(
    {
        "max_instances",
        "scaling_threshold",
        "shutdown_after_seconds",
    }
)


def plan_external_chute_update(
    values: Mapping[str, Any],
    supplied_fields: Set[str],
) -> tuple[dict[str, Any], tuple[str, ...]]:
    """Return explicit external updates and explicitly supplied hosted-only fields."""

    unsupported = tuple(
        sorted(EXTERNAL_HOSTED_ONLY_UPDATE_FIELDS.intersection(supplied_fields))
    )
    updates = {
        field: values.get(field)
        for field in EXTERNAL_CATALOG_UPDATE_FIELDS
        if field in supplied_fields
    }

    # These catalog columns are rendered as strings by the common Chute response.
    # Treat an explicit null as clearing the value without persisting a null that
    # would make response validation fail.
    for field in ("tagline", "readme"):
        if field in updates and updates[field] is None:
            updates[field] = ""
    if "logo_id" in updates and not updates["logo_id"]:
        updates["logo_id"] = None

    disabled = values.get("disabled")
    if "disabled" in supplied_fields and disabled is not None:
        updates["disabled"] = disabled
    return updates, unsupported


__all__ = [
    "EXTERNAL_CATALOG_UPDATE_FIELDS",
    "EXTERNAL_HOSTED_ONLY_UPDATE_FIELDS",
    "plan_external_chute_update",
]
