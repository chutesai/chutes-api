"""Bounded persistence policy for asynchronous inline result metadata."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import orjson


class InlineResultPolicyError(ValueError):
    """Inline task result persistence is invalid or exceeds its configured bound."""


def inline_result_limit(operation_config: Mapping[str, Any] | None) -> int | None:
    config = dict(operation_config or {})
    enabled = config.get("persist_inline_result", False)
    if not isinstance(enabled, bool):
        raise InlineResultPolicyError("persist_inline_result must be a boolean")
    configured = config.get("max_inline_result_bytes", 64 * 1024)
    if isinstance(configured, bool) or not isinstance(configured, int):
        raise InlineResultPolicyError("max_inline_result_bytes must be an integer")
    if not 1 <= configured <= 1024 * 1024:
        raise InlineResultPolicyError(
            "max_inline_result_bytes must be between 1 and 1048576"
        )
    return configured if enabled else None


def bounded_inline_result(value: Any, limit: int) -> Any:
    if len(orjson.dumps(value)) > limit:
        raise InlineResultPolicyError(
            "inline task result exceeds max_inline_result_bytes"
        )
    return value


__all__ = [
    "InlineResultPolicyError",
    "bounded_inline_result",
    "inline_result_limit",
]
