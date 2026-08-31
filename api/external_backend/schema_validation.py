"""JSON Schema validation that never retrieves remote resources."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import jsonschema
from referencing import Registry
from referencing.exceptions import NoSuchResource


class RemoteSchemaReferenceError(ValueError):
    """A schema tried to reference a resource outside its own document."""


class UnsafeSchemaError(ValueError):
    """A schema contains constructs that cannot be evaluated with a hard bound."""


def _reject_retrieval(uri: str):
    raise NoSuchResource(ref=uri)


LOCAL_ONLY_REGISTRY: Registry[Any] = Registry(retrieve=_reject_retrieval)


def reject_remote_schema_references(value: object) -> None:
    """Allow document-local fragments and reject every retrievable reference."""

    if isinstance(value, Mapping):
        for key, child in value.items():
            if key in {"$ref", "$dynamicRef", "$recursiveRef"}:
                if not isinstance(child, str) or not child.startswith("#"):
                    raise RemoteSchemaReferenceError(
                        "JSON Schema references must be document-local fragments"
                    )
            reject_remote_schema_references(child)
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        for child in value:
            reject_remote_schema_references(child)


def reject_unsafe_schema_keywords(value: object) -> None:
    """Reject Python-regex-backed keywords whose evaluation cannot be cancelled.

    ``asyncio.wait_for`` cannot stop a running validation thread, and CPython's
    regex engine can hold the GIL on catastrophic expressions.  External Cords
    therefore use structural JSON Schema constraints only.
    """

    pending: list[tuple[object, int]] = [(value, 0)]
    nodes = 0
    while pending:
        current, depth = pending.pop()
        nodes += 1
        if nodes > 10_000 or depth > 64:
            raise UnsafeSchemaError("JSON Schema exceeds the complexity limit")
        if isinstance(current, Mapping):
            forbidden = {"pattern", "patternProperties"}.intersection(current)
            if forbidden:
                names = ", ".join(sorted(forbidden))
                raise UnsafeSchemaError(
                    f"JSON Schema regex keywords are not supported: {names}"
                )
            pending.extend((child, depth + 1) for child in current.values())
        elif isinstance(current, Sequence) and not isinstance(current, (str, bytes)):
            pending.extend((child, depth + 1) for child in current)


def local_json_schema_validator(
    schema: Mapping[str, Any],
) -> jsonschema.protocols.Validator:
    """Compile a validator with a registry that cannot perform I/O."""

    reject_remote_schema_references(schema)
    reject_unsafe_schema_keywords(schema)
    validator_class = jsonschema.validators.validator_for(schema)
    validator_class.check_schema(schema)
    return validator_class(schema, registry=LOCAL_ONLY_REGISTRY)


__all__ = [
    "LOCAL_ONLY_REGISTRY",
    "RemoteSchemaReferenceError",
    "UnsafeSchemaError",
    "local_json_schema_validator",
    "reject_remote_schema_references",
    "reject_unsafe_schema_keywords",
]
