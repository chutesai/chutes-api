"""Provider-neutral compatibility helpers for standard model endpoints."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from copy import deepcopy
from datetime import datetime
import hashlib
from typing import Any
import uuid

import orjson


def _cord_value(cord: object, name: str, default: Any = None) -> Any:
    if isinstance(cord, Mapping):
        return cord.get(name, default)
    return getattr(cord, name, default)


def _stream_requested(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return value == 1


def select_external_cord(
    cords: Sequence[object],
    *,
    public_path: str,
    method: str,
    stream: object = False,
) -> object | None:
    """Select a public Cord, using its stream flag to resolve shared routes."""

    normalized_method = method.upper()
    candidates = [
        cord
        for cord in cords
        if _cord_value(cord, "public_api_path") == public_path
        and str(_cord_value(cord, "public_api_method", "POST")).upper()
        == normalized_method
    ]
    if len(candidates) == 1:
        return candidates[0]
    if not candidates:
        return None

    requested = _stream_requested(stream)
    matches = [
        cord
        for cord in candidates
        if bool(_cord_value(cord, "stream", False)) is requested
    ]
    return matches[0] if len(matches) == 1 else None


def external_llm_model_details(
    *,
    chute_id: str,
    name: str,
    created_at: datetime | None,
) -> dict[str, Any]:
    """Build a provider-neutral OpenAI-compatible model catalog entry."""

    created = int(created_at.timestamp()) if created_at is not None else 0
    return {
        "id": name,
        "object": "model",
        "created": created,
        "owned_by": "chutes",
        "root": name,
        "parent": None,
        "chute_id": chute_id,
        "confidential_compute": False,
    }


def routed_model_payload(payload: Mapping[str, Any], model_name: str) -> dict[str, Any]:
    """Copy a central model request and pin it to one resolved public model."""

    result = deepcopy(dict(payload))
    result["model"] = model_name
    return result


def external_chute_version(
    *,
    account_id: str,
    standard_template: str | None,
    cords: Sequence[object],
    routes: Sequence[object],
    pricing_rules: Sequence[object],
) -> str:
    """Create a stable version that covers all execution-relevant metadata."""

    def serialize(items: Sequence[object]) -> list[object]:
        return [
            item.model_dump(mode="json") if hasattr(item, "model_dump") else item
            for item in items
        ]

    payload = orjson.dumps(
        {
            "account_id": account_id,
            "standard_template": standard_template,
            "cords": serialize(cords),
            "routes": serialize(routes),
            "pricing_rules": serialize(pricing_rules),
        },
        option=orjson.OPT_SORT_KEYS,
    )
    digest = hashlib.sha256(payload).hexdigest()
    return str(uuid.uuid5(uuid.NAMESPACE_OID, digest))


def credential_allows_chute(credential: object | None, chute_id: str) -> bool:
    """Recheck a mega-route credential after its concrete Chute resolves."""

    if credential is None:
        return True
    checker = getattr(credential, "has_access", None)
    return bool(checker and checker("chutes", chute_id, "invoke"))


def routed_fallback_allowed(status_code: int, *, attempt_billable: bool) -> bool:
    """Avoid a transparent second upstream charge after a billable attempt."""

    return status_code in {429, 503} and not attempt_billable


def public_pricing_rules(rules: Sequence[object]) -> list[dict[str, Any]]:
    """Project configured prices without publishing operator-defined identifiers."""

    result: list[dict[str, Any]] = []
    public_groups: dict[str, str] = {}
    for index, value in enumerate(rules):
        if not isinstance(value, Mapping):
            continue
        rule = deepcopy(dict(value))
        rule["id"] = f"price-rule-{index + 1}"
        internal_group = rule.get("match_group")
        if isinstance(internal_group, str) and internal_group:
            if internal_group not in public_groups:
                public_groups[internal_group] = f"price-group-{len(public_groups) + 1}"
            rule["match_group"] = public_groups[internal_group]
        result.append(rule)
    return result


def public_charge_line_items(items: Sequence[object]) -> list[dict[str, Any]]:
    """Return the client-relevant charge breakdown with neutral line identifiers."""

    allowed = {
        "metric",
        "bucket",
        "quantity",
        "billable_units",
        "unit_price",
        "amount",
    }
    result: list[dict[str, Any]] = []
    for index, value in enumerate(items):
        if not isinstance(value, Mapping):
            continue
        line = {key: deepcopy(value[key]) for key in allowed if key in value}
        line["rule_id"] = f"charge-line-{index + 1}"
        result.append(line)
    return result


__all__ = [
    "external_llm_model_details",
    "external_chute_version",
    "credential_allows_chute",
    "public_charge_line_items",
    "public_pricing_rules",
    "routed_fallback_allowed",
    "routed_model_payload",
    "select_external_cord",
]
