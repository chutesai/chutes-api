"""Validated, transport-independent mapping of JSON-compatible data."""

from __future__ import annotations

import copy
import re
from urllib.parse import urlsplit
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal, InvalidOperation
from enum import Enum
from typing import Any

from api.external_transport.header_policy import requires_secret_backing

from api.payment.pricing import NormalizedUsage, UsageValidationError


_MISSING = object()
_SCRUBBED = object()
_WILDCARD = object()
_MAX_PATH_LENGTH = 1024
_MAX_PATH_PARTS = 128
_MAX_CREATED_LIST_INDEX = 10_000
_UNSAFE_PATH_PARTS = frozenset({"__proto__", "prototype", "constructor"})
_SEQUENCE_TYPES = (list, tuple)


class MappingConfigurationError(ValueError):
    """Raised before use when a data mapping is unsafe or ambiguous."""


class MappingExtractionError(ValueError):
    """Raised when valid mapping rules cannot safely map a payload."""


class StreamUsageMode(str, Enum):
    """How usage observations emitted over time relate to one another."""

    CUMULATIVE = "cumulative"
    DELTA = "delta"


@dataclass(frozen=True)
class DataPath:
    """A compiled dotted or JSON Pointer path."""

    raw: str
    parts: tuple[str | int | object, ...]

    @classmethod
    def parse(cls, value: str | DataPath) -> DataPath:
        if isinstance(value, cls):
            return value
        if not isinstance(value, str):
            raise MappingConfigurationError("data paths must be strings")
        if len(value) > _MAX_PATH_LENGTH:
            raise MappingConfigurationError("data path is too long")

        if value in {"", "$"}:
            return cls(raw=value, parts=())
        if value.startswith("/"):
            parts = _parse_json_pointer(value)
        else:
            parts = _parse_dotted_path(value)
        if len(parts) > _MAX_PATH_PARTS:
            raise MappingConfigurationError("data path has too many parts")
        for part in parts:
            if isinstance(part, str) and part.lower() in _UNSAFE_PATH_PARTS:
                raise MappingConfigurationError("data path contains a reserved part")
        return cls(raw=value, parts=tuple(parts))

    @property
    def has_wildcard(self) -> bool:
        return any(part is _WILDCARD for part in self.parts)


def _parse_json_pointer(value: str) -> list[str | object]:
    parts: list[str | object] = []
    for encoded in value[1:].split("/"):
        if re.search(r"~(?![01])", encoded):
            raise MappingConfigurationError("JSON Pointer contains an invalid escape")
        part = encoded.replace("~1", "/").replace("~0", "~")
        if part == "*":
            parts.append(_WILDCARD)
        else:
            parts.append(part)
    return parts


def _parse_dotted_path(value: str) -> list[str | int | object]:
    path = value
    if path.startswith("$"):
        path = path[1:]
        if path.startswith("."):
            path = path[1:]
        elif path and not path.startswith("["):
            raise MappingConfigurationError(
                "root marker must be followed by a dot or bracket"
            )
    if not path:
        return []

    parts: list[str | int | object] = []
    position = 0
    expect_name = True
    while position < len(path):
        character = path[position]
        if character == ".":
            if expect_name:
                raise MappingConfigurationError("dotted path contains an empty part")
            expect_name = True
            position += 1
            continue
        if character == "[":
            closing = path.find("]", position + 1)
            if closing < 0:
                raise MappingConfigurationError("dotted path has an unclosed bracket")
            token = path[position + 1 : closing]
            if token == "*":
                parts.append(_WILDCARD)
            elif token.isdigit():
                parts.append(int(token))
            else:
                raise MappingConfigurationError(
                    "bracket path parts must be non-negative indexes or wildcards"
                )
            position = closing + 1
            expect_name = False
            if position < len(path) and path[position] not in ".[":
                raise MappingConfigurationError(
                    "invalid character after bracket path part"
                )
            continue

        ending = position
        while ending < len(path) and path[ending] not in ".[":
            ending += 1
        token = path[position:ending]
        if not token:
            raise MappingConfigurationError("dotted path contains an empty part")
        if token == "*":
            parts.append(_WILDCARD)
        elif token.isdigit():
            parts.append(token)
        else:
            parts.append(token)
        position = ending
        expect_name = False
    if expect_name:
        raise MappingConfigurationError("dotted path cannot end with a dot")
    return parts


def _is_sequence(value: Any) -> bool:
    return isinstance(value, _SEQUENCE_TYPES)


def _sequence_index(part: str | int | object) -> int | None:
    if isinstance(part, int):
        return part
    if isinstance(part, str) and part.isdigit():
        return int(part)
    return None


def _walk_values(value: Any, parts: tuple[str | int | object, ...]) -> list[Any]:
    if not parts:
        return [value]
    part, remaining = parts[0], parts[1:]
    if part is _WILDCARD:
        if isinstance(value, Mapping):
            children = value.values()
        elif _is_sequence(value):
            children = value
        else:
            return []
        found: list[Any] = []
        for child in children:
            found.extend(_walk_values(child, remaining))
        return found
    if isinstance(value, Mapping):
        if not isinstance(part, str) or part not in value:
            return []
        return _walk_values(value[part], remaining)
    if _is_sequence(value):
        index = _sequence_index(part)
        if index is None or index >= len(value):
            return []
        return _walk_values(value[index], remaining)
    return []


def extract_values(data: Any, path: str | DataPath) -> tuple[Any, ...]:
    """Extract all path matches without evaluating attributes or expressions."""

    compiled = DataPath.parse(path)
    return tuple(_walk_values(data, compiled.parts))


def extract_value(
    data: Any,
    path: str | DataPath,
    *,
    default: Any = _MISSING,
    required: bool = False,
) -> Any:
    """Extract exactly one value, rejecting ambiguous wildcard matches."""

    values = extract_values(data, path)
    if len(values) == 1:
        return values[0]
    if len(values) > 1:
        raise MappingExtractionError("data path matched more than one value")
    if default is not _MISSING:
        return copy.deepcopy(default)
    if required:
        raise MappingExtractionError("required data path did not match")
    return None


def is_missing_value(value: Any) -> bool:
    """Return whether ``ValueRule.evaluate`` omitted an optional value."""

    return value is _MISSING


def _path_exists(data: Any, path: DataPath) -> bool:
    return bool(_walk_values(data, path.parts))


def _new_container(next_part: str | int | object) -> dict[str, Any] | list[Any]:
    return [] if _sequence_index(next_part) is not None else {}


def _assign_path(
    current: Any,
    parts: tuple[str | int | object, ...],
    value: Any,
    *,
    create: bool,
) -> None:
    if not parts:
        raise MappingExtractionError("root replacement is not supported by field rules")
    part, remaining = parts[0], parts[1:]

    if part is _WILDCARD:
        if not remaining:
            if isinstance(current, dict):
                for key in list(current):
                    current[key] = copy.deepcopy(value)
                return
            if isinstance(current, list):
                for index in range(len(current)):
                    current[index] = copy.deepcopy(value)
                return
            raise MappingExtractionError(
                "wildcard target does not refer to a collection"
            )
        if isinstance(current, Mapping):
            children = list(current.values())
        elif isinstance(current, list):
            children = list(current)
        else:
            raise MappingExtractionError(
                "wildcard target does not refer to a collection"
            )
        for child in children:
            _assign_path(child, remaining, copy.deepcopy(value), create=create)
        return

    if not remaining:
        if isinstance(current, dict) and isinstance(part, str):
            current[part] = copy.deepcopy(value)
            return
        if isinstance(current, list):
            index = _sequence_index(part)
            if index is None:
                raise MappingExtractionError("array target requires a numeric index")
            if index > _MAX_CREATED_LIST_INDEX:
                raise MappingExtractionError("array target index is too large")
            if index >= len(current):
                if not create:
                    raise MappingExtractionError("array target does not exist")
                current.extend([None] * (index - len(current) + 1))
            current[index] = copy.deepcopy(value)
            return
        raise MappingExtractionError("field target parent is not a collection")

    if isinstance(current, dict) and isinstance(part, str):
        if part not in current:
            if not create:
                raise MappingExtractionError("field target parent does not exist")
            current[part] = _new_container(remaining[0])
        child = current[part]
        if child is None and create:
            child = _new_container(remaining[0])
            current[part] = child
        _assign_path(child, remaining, value, create=create)
        return
    if isinstance(current, list):
        index = _sequence_index(part)
        if index is None:
            raise MappingExtractionError("array target requires a numeric index")
        if index > _MAX_CREATED_LIST_INDEX:
            raise MappingExtractionError("array target index is too large")
        if index >= len(current):
            if not create:
                raise MappingExtractionError("array target does not exist")
            current.extend([None] * (index - len(current) + 1))
        if current[index] is None and create:
            current[index] = _new_container(remaining[0])
        _assign_path(current[index], remaining, value, create=create)
        return
    raise MappingExtractionError("field target parent is not a collection")


def _matching_parents(
    current: Any,
    parts: tuple[str | int | object, ...],
) -> list[tuple[Any, str | int]]:
    if not parts:
        return []
    part, remaining = parts[0], parts[1:]
    if not remaining:
        if part is _WILDCARD:
            if isinstance(current, dict):
                return [(current, key) for key in current]
            if isinstance(current, list):
                return [(current, index) for index in range(len(current))]
            return []
        if isinstance(current, Mapping) and isinstance(part, str) and part in current:
            return [(current, part)]
        if isinstance(current, list):
            index = _sequence_index(part)
            if index is not None and index < len(current):
                return [(current, index)]
        return []
    if part is _WILDCARD:
        if isinstance(current, Mapping):
            children = current.values()
        elif _is_sequence(current):
            children = current
        else:
            return []
        matches: list[tuple[Any, str | int]] = []
        for child in children:
            matches.extend(_matching_parents(child, remaining))
        return matches
    if isinstance(current, Mapping) and isinstance(part, str) and part in current:
        return _matching_parents(current[part], remaining)
    if _is_sequence(current):
        index = _sequence_index(part)
        if index is not None and index < len(current):
            return _matching_parents(current[index], remaining)
    return []


def _remove_path(data: Any, path: DataPath) -> None:
    parents = _matching_parents(data, path.parts)
    list_removals: dict[int, tuple[list[Any], set[int]]] = {}
    for parent, key in parents:
        if isinstance(parent, dict):
            parent.pop(key, None)
        elif isinstance(parent, list) and isinstance(key, int):
            identifier = id(parent)
            if identifier not in list_removals:
                list_removals[identifier] = (parent, set())
            list_removals[identifier][1].add(key)
    for parent, indexes in list_removals.values():
        for index in sorted(indexes, reverse=True):
            del parent[index]


_VALUE_RULE_KEYS = frozenset(
    {
        "source",
        "path",
        "paths",
        "value",
        "default",
        "required",
        "aggregate",
        "cast",
        "map",
        "multiply",
        "divide",
        "add",
        "separator",
    }
)
_SOURCES = frozenset({"request", "response", "context", "payload", "item"})
_AGGREGATES = frozenset(
    {"only", "first", "last", "list", "sum", "count", "min", "max", "length", "join"}
)
_CASTS = frozenset({"string", "integer", "number", "boolean"})


def _strict_keys(value: Mapping[str, Any], allowed: frozenset[str], label: str) -> None:
    unknown = sorted(str(key) for key in value if key not in allowed)
    if unknown:
        raise MappingConfigurationError(
            f"{label} contains unsupported fields: {', '.join(unknown)}"
        )


def _finite_decimal(value: Any, label: str, *, non_negative: bool = False) -> Decimal:
    if isinstance(value, bool) or value is None:
        raise MappingConfigurationError(f"{label} must be a finite number")
    try:
        result = value if isinstance(value, Decimal) else Decimal(str(value))
    except (InvalidOperation, TypeError, ValueError) as exc:
        raise MappingConfigurationError(f"{label} must be a finite number") from exc
    if not result.is_finite() or (non_negative and result < 0):
        raise MappingConfigurationError(f"{label} must be a finite non-negative number")
    return result


@dataclass(frozen=True)
class ValueRule:
    """A validated rule for selecting and normalizing one value."""

    source: str
    paths: tuple[DataPath, ...] = ()
    literal: Any = _MISSING
    default: Any = _MISSING
    required: bool = False
    aggregate: str = "only"
    cast: str | None = None
    value_map: tuple[tuple[Any, Any], ...] = ()
    multiply: Decimal = Decimal("1")
    divide: Decimal = Decimal("1")
    add: Decimal = Decimal("0")
    separator: str = ""

    @classmethod
    def from_config(
        cls,
        config: Any,
        *,
        default_source: str,
        strings_are_paths: bool = True,
    ) -> ValueRule:
        if isinstance(config, cls):
            return config
        if isinstance(config, str) and strings_are_paths:
            config = {"path": config}
        elif not isinstance(config, Mapping):
            config = {"value": config}
        config = dict(config)
        _strict_keys(config, _VALUE_RULE_KEYS, "value rule")

        source = config.get("source", default_source)
        if source not in _SOURCES:
            raise MappingConfigurationError("value rule source is not supported")
        has_literal = "value" in config
        has_path = "path" in config or "paths" in config
        if has_literal and has_path:
            raise MappingConfigurationError(
                "value rule cannot contain both value and path"
            )
        if "path" in config and "paths" in config:
            raise MappingConfigurationError(
                "value rule cannot contain both path and paths"
            )
        if has_literal and "source" in config:
            raise MappingConfigurationError("literal value rule cannot select a source")
        if config.get("required", False) and "default" in config:
            raise MappingConfigurationError("required value rule cannot have a default")
        required = config.get("required", False)
        if not isinstance(required, bool):
            raise MappingConfigurationError("value rule required must be a boolean")

        raw_paths: Sequence[Any]
        if "path" in config:
            raw_paths = [config["path"]]
        elif "paths" in config:
            raw_paths = config["paths"]
            if not isinstance(raw_paths, Sequence) or isinstance(
                raw_paths, (str, bytes)
            ):
                raise MappingConfigurationError("value rule paths must be a list")
            if not raw_paths:
                raise MappingConfigurationError("value rule paths cannot be empty")
        elif has_literal:
            raw_paths = []
        else:
            raw_paths = ["$"]
        paths = tuple(DataPath.parse(path) for path in raw_paths)

        aggregate = config.get("aggregate", "only")
        if aggregate not in _AGGREGATES:
            raise MappingConfigurationError("value rule aggregate is not supported")
        cast = config.get("cast")
        if cast is not None and cast not in _CASTS:
            raise MappingConfigurationError("value rule cast is not supported")
        raw_map = config.get("map", {})
        if not isinstance(raw_map, Mapping):
            raise MappingConfigurationError("value rule map must be an object")
        value_map = tuple(
            (copy.deepcopy(key), copy.deepcopy(value)) for key, value in raw_map.items()
        )
        multiply = _finite_decimal(config.get("multiply", 1), "value rule multiply")
        divide = _finite_decimal(config.get("divide", 1), "value rule divide")
        add = _finite_decimal(config.get("add", 0), "value rule add")
        if divide == 0:
            raise MappingConfigurationError("value rule divide cannot be zero")
        separator = config.get("separator", "")
        if not isinstance(separator, str):
            raise MappingConfigurationError("value rule separator must be a string")
        if aggregate == "join" and cast not in {None, "string"}:
            raise MappingConfigurationError("joined values can only be cast to strings")

        return cls(
            source=source,
            paths=paths,
            literal=(copy.deepcopy(config["value"]) if "value" in config else _MISSING),
            default=(
                copy.deepcopy(config["default"]) if "default" in config else _MISSING
            ),
            required=required,
            aggregate=aggregate,
            cast=cast,
            value_map=value_map,
            multiply=multiply,
            divide=divide,
            add=add,
            separator=separator,
        )

    def evaluate(self, sources: Mapping[str, Any]) -> Any:
        if self.literal is not _MISSING:
            value = copy.deepcopy(self.literal)
        else:
            if self.source not in sources:
                raise MappingExtractionError("configured value source is unavailable")
            selected: list[Any] = []
            for path in self.paths:
                selected.extend(_walk_values(sources[self.source], path.parts))
            if not selected:
                if self.default is not _MISSING:
                    value = copy.deepcopy(self.default)
                elif self.required:
                    raise MappingExtractionError("required mapped value is missing")
                else:
                    return _MISSING
            else:
                value = _aggregate_values(selected, self.aggregate, self.separator)

        for original, replacement in self.value_map:
            if type(value) is type(original) and value == original:  # noqa: E721
                value = copy.deepcopy(replacement)
                break
        value = _cast_value(value, self.cast)
        if self.multiply != 1 or self.divide != 1 or self.add != 0:
            number = _runtime_decimal(value)
            value = (number * self.multiply / self.divide) + self.add
        return value


def _runtime_decimal(value: Any) -> Decimal:
    if isinstance(value, bool) or value is None:
        raise MappingExtractionError("mapped value is not a finite number")
    try:
        result = value if isinstance(value, Decimal) else Decimal(str(value))
    except (InvalidOperation, TypeError, ValueError) as exc:
        raise MappingExtractionError("mapped value is not a finite number") from exc
    if not result.is_finite():
        raise MappingExtractionError("mapped value is not a finite number")
    return result


def _aggregate_values(values: list[Any], aggregate: str, separator: str) -> Any:
    if aggregate == "only":
        if len(values) != 1:
            raise MappingExtractionError("mapped value has more than one match")
        return values[0]
    if aggregate == "first":
        return values[0]
    if aggregate == "last":
        return values[-1]
    if aggregate == "list":
        return copy.deepcopy(values)
    if aggregate == "count":
        return len(values)
    if aggregate == "length":
        if len(values) != 1 or not isinstance(
            values[0], (str, bytes, Mapping, Sequence)
        ):
            raise MappingExtractionError("length aggregate requires one sized value")
        return len(values[0])
    if aggregate == "join":
        return separator.join(str(value) for value in values)
    numbers = [_runtime_decimal(value) for value in values]
    if aggregate == "sum":
        return sum(numbers, Decimal("0"))
    if aggregate == "min":
        return min(numbers)
    if aggregate == "max":
        return max(numbers)
    raise MappingExtractionError("mapped value aggregate is unsupported")


def _cast_value(value: Any, cast: str | None) -> Any:
    if cast is None:
        return value
    if cast == "string":
        if isinstance(value, (Mapping, Sequence)) and not isinstance(
            value, (str, bytes)
        ):
            raise MappingExtractionError("mapped collection cannot be cast to string")
        return str(value)
    if cast == "number":
        return _runtime_decimal(value)
    if cast == "integer":
        number = _runtime_decimal(value)
        if number != number.to_integral_value():
            raise MappingExtractionError("mapped value is not an integer")
        return int(number)
    if cast == "boolean":
        if isinstance(value, bool):
            return value
        if type(value) in {int, str} and value in (0, "0", "false", "False"):
            return False
        if type(value) in {int, str} and value in (1, "1", "true", "True"):
            return True
        raise MappingExtractionError("mapped value is not a boolean")
    raise MappingExtractionError("mapped value cast is unsupported")


_MUTATION_KEYS = _VALUE_RULE_KEYS | frozenset({"target", "remove_source"})


@dataclass(frozen=True)
class FieldMutation:
    target: DataPath
    value: ValueRule
    remove_source: bool = False

    @classmethod
    def from_config(
        cls,
        config: Mapping[str, Any],
        *,
        default_source: str,
    ) -> FieldMutation:
        if not isinstance(config, Mapping):
            raise MappingConfigurationError("field mutation must be an object")
        config = dict(config)
        _strict_keys(config, _MUTATION_KEYS, "field mutation")
        if "target" not in config:
            raise MappingConfigurationError("field mutation requires a target")
        target = DataPath.parse(config.pop("target"))
        if not target.parts:
            raise MappingConfigurationError("field mutation cannot target the root")
        remove_source = config.pop("remove_source", False)
        if not isinstance(remove_source, bool):
            raise MappingConfigurationError("remove_source must be a boolean")
        if not ({"value", "path", "paths"} & config.keys()):
            config["path"] = target.raw
            config.setdefault("source", "payload")
        return cls(
            target=target,
            value=ValueRule.from_config(config, default_source=default_source),
            remove_source=remove_source,
        )


@dataclass(frozen=True)
class PayloadTransform:
    """Compiled removal, injection, and rewrite rules for one payload."""

    remove: tuple[DataPath, ...] = ()
    inject: tuple[FieldMutation, ...] = ()
    rewrite: tuple[FieldMutation, ...] = ()

    @classmethod
    def from_config(
        cls, config: Mapping[str, Any] | PayloadTransform | None
    ) -> PayloadTransform:
        if isinstance(config, cls):
            return config
        if config is None:
            return cls()
        if not isinstance(config, Mapping):
            raise MappingConfigurationError("payload transform must be an object")
        _strict_keys(
            config, frozenset({"remove", "inject", "rewrite"}), "payload transform"
        )
        remove_config = config.get("remove", [])
        if not isinstance(remove_config, Sequence) or isinstance(
            remove_config, (str, bytes)
        ):
            raise MappingConfigurationError("payload transform remove must be a list")
        remove = tuple(DataPath.parse(path) for path in remove_config)
        if any(not path.parts for path in remove):
            raise MappingConfigurationError("payload transform cannot remove the root")
        return cls(
            remove=remove,
            inject=_compile_mutations(
                config.get("inject", []), default_source="context"
            ),
            rewrite=_compile_mutations(
                config.get("rewrite", []), default_source="payload"
            ),
        )

    def apply(
        self,
        payload: Any,
        *,
        request: Any = None,
        response: Any = None,
        context: Any = None,
    ) -> Any:
        result = copy.deepcopy(payload)
        for path in self.remove:
            _remove_path(result, path)
        sources = {
            "payload": result,
            "request": request,
            "response": response,
            "context": context,
            "item": None,
        }
        for mutation in self.inject:
            if _path_exists(result, mutation.target):
                continue
            _apply_mutation(result, mutation, sources)
        for mutation in self.rewrite:
            _apply_mutation(result, mutation, sources)
        return result


def _compile_mutations(value: Any, *, default_source: str) -> tuple[FieldMutation, ...]:
    if isinstance(value, Mapping):
        configs = []
        for target, rule in value.items():
            if isinstance(rule, Mapping) and _VALUE_RULE_KEYS.intersection(rule):
                configs.append({"target": target, **rule})
            else:
                configs.append({"target": target, "value": rule})
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        configs = list(value)
    else:
        raise MappingConfigurationError("field mutations must be a list or object")
    return tuple(
        FieldMutation.from_config(config, default_source=default_source)
        for config in configs
    )


def _apply_mutation(
    result: Any,
    mutation: FieldMutation,
    sources: dict[str, Any],
) -> None:
    sources["payload"] = result
    value = mutation.value.evaluate(sources)
    if value is _MISSING:
        return
    _assign_path(result, mutation.target.parts, value, create=True)
    if mutation.remove_source and mutation.value.source == "payload":
        for source_path in mutation.value.paths:
            if source_path != mutation.target:
                _remove_path(result, source_path)


def transform_payload(
    payload: Any,
    config: Mapping[str, Any] | PayloadTransform | None,
    *,
    request: Any = None,
    response: Any = None,
    context: Any = None,
) -> Any:
    """Apply a validated transform without modifying the input payload."""

    return PayloadTransform.from_config(config).apply(
        payload,
        request=request,
        response=response,
        context=context,
    )


_USAGE_GROUPS = frozenset(
    {
        "tokens",
        "images",
        "input_media_seconds",
        "output_media_seconds",
        "characters",
        "counts",
        "tools",
        "dimensions",
    }
)


@dataclass(frozen=True)
class UsageField:
    target: str
    rule: ValueRule


@dataclass(frozen=True)
class UsageMapping:
    """Compiled extraction rules for normalized billable usage."""

    fields: tuple[UsageField, ...]
    default_requests: Decimal = Decimal("1")

    @classmethod
    def from_config(cls, config: Mapping[str, Any] | UsageMapping) -> UsageMapping:
        if isinstance(config, cls):
            return config
        if not isinstance(config, Mapping):
            raise MappingConfigurationError("usage mapping must be an object")
        if "fields" in config or "default_requests" in config:
            _strict_keys(
                config, frozenset({"fields", "default_requests"}), "usage mapping"
            )
            field_config = config.get("fields", {})
            default_requests = _finite_decimal(
                config.get("default_requests", 1), "default_requests", non_negative=True
            )
        else:
            field_config = config
            default_requests = Decimal("1")
        if not isinstance(field_config, Mapping):
            raise MappingConfigurationError("usage mapping fields must be an object")

        flattened: list[tuple[str, Any]] = []
        for target, rule in field_config.items():
            _flatten_usage_field(str(target), rule, flattened)
        seen: set[str] = set()
        compiled: list[UsageField] = []
        for target, raw_rule in flattened:
            _validate_usage_target(target)
            if target in seen:
                raise MappingConfigurationError(
                    "usage mapping contains a duplicate target"
                )
            seen.add(target)
            rule = ValueRule.from_config(raw_rule, default_source="response")
            if not target.startswith("dimensions."):
                if rule.aggregate in {"list", "join"}:
                    raise MappingConfigurationError(
                        "usage quantity cannot use a collection aggregate"
                    )
                if rule.literal is not _MISSING:
                    _finite_decimal(rule.literal, "usage quantity", non_negative=True)
                if rule.default is not _MISSING:
                    _finite_decimal(
                        rule.default, "usage quantity default", non_negative=True
                    )
            compiled.append(UsageField(target=target, rule=rule))
        return cls(fields=tuple(compiled), default_requests=default_requests)

    def extract(
        self,
        *,
        request: Any = None,
        response: Any = None,
        context: Any = None,
        payload: Any = None,
    ) -> NormalizedUsage:
        values: dict[str, Any] = {
            "requests": self.default_requests,
            "tokens": {},
            "images": {},
            "input_media_seconds": {},
            "output_media_seconds": {},
            "characters": {},
            "counts": {},
            "tools": {},
            "dimensions": {},
        }
        sources = {
            "request": request,
            "response": response,
            "context": context,
            "payload": payload,
            "item": None,
        }
        for field_rule in self.fields:
            value = field_rule.rule.evaluate(sources)
            if value is _MISSING:
                continue
            _store_usage_value(values, field_rule.target, value)
        try:
            return NormalizedUsage.from_mapping(values)
        except UsageValidationError as exc:
            raise MappingExtractionError("mapped usage is invalid") from exc


def _looks_like_value_rule(value: Any) -> bool:
    return isinstance(value, Mapping) and bool(_VALUE_RULE_KEYS.intersection(value))


def _flatten_usage_field(
    target: str,
    value: Any,
    output: list[tuple[str, Any]],
) -> None:
    if (
        target == "requests"
        or "." in target
        or not isinstance(value, Mapping)
        or _looks_like_value_rule(value)
    ):
        output.append((target, value))
        return
    if target not in _USAGE_GROUPS:
        output.append((target, value))
        return
    if not value:
        raise MappingConfigurationError("usage group cannot be empty")

    def visit(prefix: str, nested: Any) -> None:
        if isinstance(nested, Mapping) and not _looks_like_value_rule(nested):
            if not nested:
                raise MappingConfigurationError(
                    "usage group cannot contain an empty object"
                )
            for key, item in nested.items():
                name = str(key).strip()
                if not name:
                    raise MappingConfigurationError(
                        "usage bucket names cannot be empty"
                    )
                visit(f"{prefix}.{name}", item)
        else:
            output.append((prefix, nested))

    for key, nested in value.items():
        name = str(key).strip()
        if not name:
            raise MappingConfigurationError("usage bucket names cannot be empty")
        visit(f"{target}.{name}", nested)


def _validate_usage_target(target: str) -> None:
    if target == "requests":
        return
    root, separator, bucket = target.partition(".")
    if root not in _USAGE_GROUPS or not separator or not bucket:
        raise MappingConfigurationError("usage target is not supported")
    if any(not part for part in bucket.split(".")):
        raise MappingConfigurationError("usage target contains an empty bucket")


def _store_usage_value(values: dict[str, Any], target: str, value: Any) -> None:
    if target == "requests":
        values["requests"] = value
        return
    root, _, bucket = target.partition(".")
    if root != "dimensions":
        values[root][bucket] = value
        return
    current = values["dimensions"]
    parts = bucket.split(".")
    for part in parts[:-1]:
        existing = current.get(part)
        if existing is None:
            existing = {}
            current[part] = existing
        if not isinstance(existing, dict):
            raise MappingExtractionError("dimension targets overlap")
        current = existing
    current[parts[-1]] = copy.deepcopy(value)


def extract_usage(
    config: Mapping[str, Any] | UsageMapping,
    *,
    request: Any = None,
    response: Any = None,
    context: Any = None,
    payload: Any = None,
) -> NormalizedUsage:
    """Extract normalized usage from configured request and response fields."""

    return UsageMapping.from_config(config).extract(
        request=request,
        response=response,
        context=context,
        payload=payload,
    )


def merge_stream_usage(
    previous: NormalizedUsage | None,
    observation: NormalizedUsage,
    mode: StreamUsageMode | str,
) -> NormalizedUsage:
    """Merge a stream observation using delta-additive or monotonic snapshot semantics."""

    try:
        parsed_mode = (
            mode if isinstance(mode, StreamUsageMode) else StreamUsageMode(mode)
        )
    except ValueError as exc:
        raise MappingConfigurationError("stream usage mode is not supported") from exc
    if not isinstance(observation, NormalizedUsage):
        raise MappingExtractionError("stream usage observation is invalid")
    if previous is None:
        return observation
    if not isinstance(previous, NormalizedUsage):
        raise MappingExtractionError("previous stream usage is invalid")

    merge_quantity = (
        (lambda old, new: old + new) if parsed_mode is StreamUsageMode.DELTA else max
    )
    values: dict[str, Any] = {
        "requests": merge_quantity(previous.requests, observation.requests),
        "dimensions": _merge_dimension_values(
            previous.dimensions, observation.dimensions
        ),
    }
    for attribute in (
        "tokens",
        "images",
        "input_media_seconds",
        "output_media_seconds",
        "characters",
        "counts",
        "tools",
    ):
        old_values: Mapping[str, Decimal] = getattr(previous, attribute)
        new_values: Mapping[str, Decimal] = getattr(observation, attribute)
        values[attribute] = {
            key: (
                merge_quantity(old_values[key], new_values[key])
                if key in old_values and key in new_values
                else old_values.get(key, new_values.get(key))
            )
            for key in old_values.keys() | new_values.keys()
        }
    return NormalizedUsage.from_mapping(values)


def _merge_dimension_values(
    base: Mapping[str, Any], update: Mapping[str, Any]
) -> dict[str, Any]:
    merged = copy.deepcopy(dict(base))
    for key, value in update.items():
        if isinstance(value, Mapping) and isinstance(merged.get(key), Mapping):
            merged[key] = _merge_dimension_values(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


_CANONICAL_TASK_STATUSES = frozenset(
    {"pending", "submitted", "running", "succeeded", "failed", "cancelled", "expired"}
)


@dataclass(frozen=True)
class ExtractedArtifact:
    """Remote artifact metadata retained for authorized delivery."""

    source_url: str
    kind: str = "artifact"
    content_type: str | None = None
    size_bytes: int | None = None
    expires_at: datetime | str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ExtractedTask:
    task_id: str | None = None
    status: str | None = None
    result: Any = None
    artifacts: tuple[ExtractedArtifact, ...] = ()


_ARTIFACT_KEYS = frozenset(
    {
        "items",
        "url",
        "kind",
        "content_type",
        "size_bytes",
        "expires_at",
        "metadata",
        "required",
    }
)


@dataclass(frozen=True)
class ArtifactMapping:
    items: ValueRule
    url: ValueRule
    kind: ValueRule
    content_type: ValueRule
    size_bytes: ValueRule
    expires_at: ValueRule
    metadata: tuple[tuple[str, ValueRule], ...]
    required: bool = False

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> ArtifactMapping:
        if not isinstance(config, Mapping):
            raise MappingConfigurationError("artifact mapping must be an object")
        _strict_keys(config, _ARTIFACT_KEYS, "artifact mapping")
        if "url" not in config:
            raise MappingConfigurationError("artifact mapping requires a url rule")
        required = config.get("required", False)
        if not isinstance(required, bool):
            raise MappingConfigurationError(
                "artifact mapping required must be a boolean"
            )

        raw_items = config.get("items", {"source": "response", "path": "$"})
        items_rule = ValueRule.from_config(raw_items, default_source="response")
        if items_rule.aggregate == "only":
            items_rule = ValueRule(**{**items_rule.__dict__, "aggregate": "list"})
        raw_metadata = config.get("metadata", {})
        if not isinstance(raw_metadata, Mapping):
            raise MappingConfigurationError(
                "artifact metadata mapping must be an object"
            )
        metadata = tuple(
            (str(key), ValueRule.from_config(value, default_source="item"))
            for key, value in raw_metadata.items()
        )
        return cls(
            items=items_rule,
            url=ValueRule.from_config(config["url"], default_source="item"),
            kind=ValueRule.from_config(
                config.get("kind", {"value": "artifact"}), default_source="item"
            ),
            content_type=ValueRule.from_config(
                config.get("content_type", {"value": None}), default_source="item"
            ),
            size_bytes=ValueRule.from_config(
                config.get("size_bytes", {"value": None}), default_source="item"
            ),
            expires_at=ValueRule.from_config(
                config.get("expires_at", {"value": None}), default_source="item"
            ),
            metadata=metadata,
            required=required,
        )

    def extract(self, sources: dict[str, Any]) -> tuple[ExtractedArtifact, ...]:
        selected = self.items.evaluate(sources)
        if selected is _MISSING:
            if self.required:
                raise MappingExtractionError("required artifact collection is missing")
            return ()
        items = selected if isinstance(selected, list) else [selected]
        if len(items) == 1 and isinstance(items[0], list):
            items = items[0]
        artifacts: list[ExtractedArtifact] = []
        for item in items:
            item_sources = {**sources, "item": item}
            url = self.url.evaluate(item_sources)
            if url is _MISSING or url is None:
                if self.required:
                    raise MappingExtractionError("required artifact url is missing")
                continue
            if not isinstance(url, str) or not url.strip():
                raise MappingExtractionError("mapped artifact url is invalid")
            kind = self.kind.evaluate(item_sources)
            content_type = self.content_type.evaluate(item_sources)
            size_bytes = self.size_bytes.evaluate(item_sources)
            expires_at = self.expires_at.evaluate(item_sources)
            if not isinstance(kind, str) or not kind.strip():
                raise MappingExtractionError("mapped artifact kind is invalid")
            if content_type is _MISSING:
                content_type = None
            if content_type is not None and not isinstance(content_type, str):
                raise MappingExtractionError("mapped artifact content type is invalid")
            if size_bytes is _MISSING:
                size_bytes = None
            if size_bytes is not None:
                if isinstance(size_bytes, bool):
                    raise MappingExtractionError("mapped artifact size is invalid")
                try:
                    parsed_size = int(size_bytes)
                except (TypeError, ValueError, OverflowError) as exc:
                    raise MappingExtractionError(
                        "mapped artifact size is invalid"
                    ) from exc
                if parsed_size < 0 or parsed_size != _runtime_decimal(size_bytes):
                    raise MappingExtractionError("mapped artifact size is invalid")
                size_bytes = parsed_size
            if expires_at is _MISSING:
                expires_at = None
            if expires_at is not None and not isinstance(expires_at, (str, datetime)):
                raise MappingExtractionError("mapped artifact expiration is invalid")
            metadata = {}
            for key, rule in self.metadata:
                mapped = rule.evaluate(item_sources)
                if mapped is not _MISSING:
                    metadata[key] = copy.deepcopy(mapped)
            artifacts.append(
                ExtractedArtifact(
                    source_url=url.strip(),
                    kind=kind.strip(),
                    content_type=content_type,
                    size_bytes=size_bytes,
                    expires_at=expires_at,
                    metadata=metadata,
                )
            )
        if self.required and not artifacts:
            raise MappingExtractionError("required artifact collection is empty")
        return tuple(artifacts)


@dataclass(frozen=True)
class TaskMapping:
    """Compiled task identity, state, result, and artifact extraction."""

    task_id: ValueRule | None = None
    status: ValueRule | None = None
    result: ValueRule | None = None
    artifacts: tuple[ArtifactMapping, ...] = ()

    @classmethod
    def from_config(cls, config: Mapping[str, Any] | TaskMapping) -> TaskMapping:
        if isinstance(config, cls):
            return config
        if not isinstance(config, Mapping):
            raise MappingConfigurationError("task mapping must be an object")
        _strict_keys(
            config,
            frozenset({"task_id", "status", "result", "artifacts"}),
            "task mapping",
        )
        raw_artifacts = config.get("artifacts", [])
        if isinstance(raw_artifacts, Mapping):
            raw_artifacts = [raw_artifacts]
        if not isinstance(raw_artifacts, Sequence) or isinstance(
            raw_artifacts, (str, bytes)
        ):
            raise MappingConfigurationError("task artifacts must be a list or object")
        status = (
            ValueRule.from_config(config["status"], default_source="response")
            if "status" in config
            else None
        )
        if status is not None:
            for _, mapped in status.value_map:
                if (
                    not isinstance(mapped, str)
                    or mapped.lower() not in _CANONICAL_TASK_STATUSES
                ):
                    raise MappingConfigurationError(
                        "task status map contains an unsupported state"
                    )
        return cls(
            task_id=(
                ValueRule.from_config(config["task_id"], default_source="response")
                if "task_id" in config
                else None
            ),
            status=status,
            result=(
                ValueRule.from_config(config["result"], default_source="response")
                if "result" in config
                else None
            ),
            artifacts=tuple(
                ArtifactMapping.from_config(value) for value in raw_artifacts
            ),
        )

    def extract(
        self,
        *,
        request: Any = None,
        response: Any = None,
        context: Any = None,
    ) -> ExtractedTask:
        sources = {
            "request": request,
            "response": response,
            "context": context,
            "payload": response,
            "item": None,
        }
        task_id = self.task_id.evaluate(sources) if self.task_id else _MISSING
        if task_id is not _MISSING and task_id is not None:
            if isinstance(task_id, bool) or not isinstance(task_id, (str, int)):
                raise MappingExtractionError("mapped task id is invalid")
            task_id = str(task_id).strip()
            if not task_id:
                raise MappingExtractionError("mapped task id is invalid")
        else:
            task_id = None
        status = self.status.evaluate(sources) if self.status else _MISSING
        if status is not _MISSING and status is not None:
            if (
                not isinstance(status, str)
                or status.lower() not in _CANONICAL_TASK_STATUSES
            ):
                raise MappingExtractionError("mapped task status is invalid")
            status = status.lower()
        else:
            status = None
        result = self.result.evaluate(sources) if self.result else None
        if result is _MISSING:
            result = None
        artifacts: list[ExtractedArtifact] = []
        for mapping in self.artifacts:
            artifacts.extend(mapping.extract(sources))
        return ExtractedTask(
            task_id=task_id,
            status=status,
            result=copy.deepcopy(result),
            artifacts=tuple(artifacts),
        )


def extract_task(
    config: Mapping[str, Any] | TaskMapping,
    *,
    request: Any = None,
    response: Any = None,
    context: Any = None,
) -> ExtractedTask:
    """Extract normalized task data from a response."""

    return TaskMapping.from_config(config).extract(
        request=request,
        response=response,
        context=context,
    )


def extract_artifacts(
    config: Mapping[str, Any] | Sequence[Mapping[str, Any]] | ArtifactMapping,
    *,
    request: Any = None,
    response: Any = None,
    context: Any = None,
) -> tuple[ExtractedArtifact, ...]:
    """Extract artifact metadata independently of task identity and state."""

    if isinstance(config, ArtifactMapping):
        mappings = (config,)
    elif isinstance(config, Mapping):
        mappings = (ArtifactMapping.from_config(config),)
    elif isinstance(config, Sequence) and not isinstance(config, (str, bytes)):
        mappings = tuple(ArtifactMapping.from_config(item) for item in config)
    else:
        raise MappingConfigurationError("artifact mappings must be a list or object")
    sources = {
        "request": request,
        "response": response,
        "context": context,
        "payload": response,
        "item": None,
    }
    artifacts: list[ExtractedArtifact] = []
    for mapping in mappings:
        artifacts.extend(mapping.extract(sources))
    return tuple(artifacts)


_DEFAULT_PRIVATE_KEYS = frozenset(
    {
        "request_id",
        "task_id",
        "operation_id",
        "job_id",
        "trace_id",
        "upstream_id",
        "upstream_request_id",
        "upstream_task_id",
        "upstream_operation_id",
        "upstream_metadata",
        "model",
        "model_id",
        "model_name",
        "provider",
        "provider_id",
        "provider_name",
        "provider_metadata",
        "vendor",
        "vendor_id",
        "vendor_name",
        "endpoint",
        "endpoint_id",
        "artifact_url",
        "callback_url",
        "cancel_url",
        "download_url",
        "endpoint_url",
        "href",
        "job_url",
        "link",
        "operation_url",
        "output_url",
        "password",
        "passphrase",
        "poll_url",
        "result_url",
        "service_endpoint",
        "service_url",
        "status_url",
        "task_url",
        "uri",
        "url",
        "x_request_id",
    }
)
_PUBLIC_RULE_KEYS = frozenset(
    {
        "remove_keys",
        "remove_paths",
        "rewrite_keys",
        "rewrite",
        "artifact_paths",
        "max_depth",
        "max_nodes",
    }
)


def _normalized_key(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(value).lower()).strip("_")


@dataclass(frozen=True)
class PublicResponseRules:
    """Rules that prevent private execution metadata from reaching clients."""

    remove_keys: frozenset[str] = _DEFAULT_PRIVATE_KEYS
    remove_paths: tuple[DataPath, ...] = ()
    rewrite_keys: tuple[tuple[str, Any], ...] = ()
    rewrite: tuple[FieldMutation, ...] = ()
    artifact_paths: tuple[DataPath, ...] = ()
    max_depth: int = 64
    max_nodes: int = 100_000

    @classmethod
    def from_config(
        cls,
        config: Mapping[str, Any] | PublicResponseRules | None,
    ) -> PublicResponseRules:
        if isinstance(config, cls):
            return config
        if config is None:
            config = {}
        if not isinstance(config, Mapping):
            raise MappingConfigurationError("public response rules must be an object")
        _strict_keys(config, _PUBLIC_RULE_KEYS, "public response rules")
        remove_keys = config.get("remove_keys", [])
        if not isinstance(remove_keys, Sequence) or isinstance(
            remove_keys, (str, bytes)
        ):
            raise MappingConfigurationError("public remove_keys must be a list")
        normalized_remove = set(_DEFAULT_PRIVATE_KEYS)
        for key in remove_keys:
            normalized = _normalized_key(key)
            if not normalized:
                raise MappingConfigurationError("public remove key cannot be empty")
            normalized_remove.add(normalized)
        remove_paths = _compile_path_list(
            config.get("remove_paths", []), "public remove_paths"
        )
        artifact_paths = _compile_path_list(
            config.get("artifact_paths", []), "public artifact_paths"
        )
        rewrite_keys = config.get("rewrite_keys", {})
        if not isinstance(rewrite_keys, Mapping):
            raise MappingConfigurationError("public rewrite_keys must be an object")
        normalized_rewrites: list[tuple[str, Any]] = []
        for key, value in rewrite_keys.items():
            normalized = _normalized_key(key)
            if not normalized:
                raise MappingConfigurationError("public rewrite key cannot be empty")
            normalized_rewrites.append((normalized, copy.deepcopy(value)))
        rewrite = _compile_mutations(
            config.get("rewrite", []), default_source="payload"
        )
        if any(mutation.value.source != "payload" for mutation in rewrite):
            raise MappingConfigurationError(
                "public response rewrites may only read the scrubbed payload"
            )
        max_depth = config.get("max_depth", 64)
        max_nodes = config.get("max_nodes", 100_000)
        if (
            isinstance(max_depth, bool)
            or not isinstance(max_depth, int)
            or not 1 <= max_depth <= 256
        ):
            raise MappingConfigurationError(
                "public max_depth must be between 1 and 256"
            )
        if (
            isinstance(max_nodes, bool)
            or not isinstance(max_nodes, int)
            or not 1 <= max_nodes <= 1_000_000
        ):
            raise MappingConfigurationError(
                "public max_nodes must be between 1 and 1000000"
            )
        return cls(
            remove_keys=frozenset(normalized_remove),
            remove_paths=remove_paths,
            rewrite_keys=tuple(normalized_rewrites),
            rewrite=rewrite,
            artifact_paths=artifact_paths,
            max_depth=max_depth,
            max_nodes=max_nodes,
        )

    def scrub(
        self,
        response: Any,
        *,
        artifact_urls: Mapping[str, str] | Sequence[str] | None = None,
    ) -> Any:
        prepared = copy.deepcopy(response)
        _replace_artifact_urls(prepared, self.artifact_paths, artifact_urls)
        local_artifact_urls = frozenset(
            artifact_urls.values()
            if isinstance(artifact_urls, Mapping)
            else (artifact_urls or ())
        )
        rewrites = dict(self.rewrite_keys)
        counter = [0]
        result = _scrub_tree(
            prepared,
            remove_keys=self.remove_keys,
            rewrite_keys=rewrites,
            local_artifact_urls=local_artifact_urls,
            depth=0,
            max_depth=self.max_depth,
            max_nodes=self.max_nodes,
            counter=counter,
        )
        if result is _SCRUBBED:
            result = None
        for path in self.remove_paths:
            _remove_path(result, path)
        if self.rewrite:
            transform = PayloadTransform(rewrite=self.rewrite)
            result = transform.apply(result)
            # A rewrite may target a reserved identity/credential/link key. Reapply
            # the boundary so configured transformations cannot undo scrubbing or
            # override server-owned rewrite_keys.
            result = _scrub_tree(
                result,
                remove_keys=self.remove_keys,
                rewrite_keys=rewrites,
                local_artifact_urls=local_artifact_urls,
                depth=0,
                max_depth=self.max_depth,
                max_nodes=self.max_nodes,
                counter=[0],
            )
            if result is _SCRUBBED:
                result = None
            for path in self.remove_paths:
                _remove_path(result, path)
        return result


def _compile_path_list(value: Any, label: str) -> tuple[DataPath, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise MappingConfigurationError(f"{label} must be a list")
    paths = tuple(DataPath.parse(path) for path in value)
    if any(not path.parts for path in paths):
        raise MappingConfigurationError(f"{label} cannot contain the root")
    return paths


def _scrub_tree(
    value: Any,
    *,
    remove_keys: frozenset[str],
    rewrite_keys: Mapping[str, Any],
    local_artifact_urls: frozenset[str],
    depth: int,
    max_depth: int,
    max_nodes: int,
    counter: list[int],
) -> Any:
    counter[0] += 1
    if counter[0] > max_nodes or depth > max_depth:
        raise MappingExtractionError(
            "public response exceeds configured structural limits"
        )
    if isinstance(value, Mapping):
        result = {}
        for key, child in value.items():
            normalized = _normalized_key(key)
            if normalized in rewrite_keys:
                scrubbed = _scrub_tree(
                    copy.deepcopy(rewrite_keys[normalized]),
                    remove_keys=remove_keys,
                    rewrite_keys={},
                    local_artifact_urls=local_artifact_urls,
                    depth=depth + 1,
                    max_depth=max_depth,
                    max_nodes=max_nodes,
                    counter=counter,
                )
                if scrubbed is not _SCRUBBED:
                    result[key] = scrubbed
            elif not _private_response_key(
                normalized,
                child,
                remove_keys=remove_keys,
                local_artifact_urls=local_artifact_urls,
            ):
                scrubbed = _scrub_tree(
                    child,
                    remove_keys=remove_keys,
                    rewrite_keys=rewrite_keys,
                    local_artifact_urls=local_artifact_urls,
                    depth=depth + 1,
                    max_depth=max_depth,
                    max_nodes=max_nodes,
                    counter=counter,
                )
                if scrubbed is not _SCRUBBED:
                    result[key] = scrubbed
        return result
    if _is_sequence(value):
        result = []
        for child in value:
            scrubbed = _scrub_tree(
                child,
                remove_keys=remove_keys,
                rewrite_keys=rewrite_keys,
                local_artifact_urls=local_artifact_urls,
                depth=depth + 1,
                max_depth=max_depth,
                max_nodes=max_nodes,
                counter=counter,
            )
            if scrubbed is not _SCRUBBED:
                result.append(scrubbed)
        return result
    if _private_absolute_uri(value, local_artifact_urls):
        return _SCRUBBED
    return copy.deepcopy(value)


def _private_absolute_uri(value: Any, local_artifact_urls: frozenset[str]) -> bool:
    if not isinstance(value, str) or value in local_artifact_urls:
        return False
    stripped = value.strip()
    if stripped.startswith("//"):
        return True
    try:
        parsed = urlsplit(stripped)
    except ValueError:
        return False
    return parsed.scheme.lower() in {
        "ftp",
        "ftps",
        "gs",
        "http",
        "https",
        "oss",
        "s3",
        "ws",
        "wss",
    }


def _private_response_key(
    normalized: str,
    value: Any,
    *,
    remove_keys: frozenset[str],
    local_artifact_urls: frozenset[str],
) -> bool:
    if normalized in remove_keys:
        return not (isinstance(value, str) and value in local_artifact_urls)
    if requires_secret_backing(normalized):
        return True
    if normalized in {"href", "link", "uri", "url"} or normalized.endswith(
        ("_uri", "_url")
    ):
        return not (isinstance(value, str) and value in local_artifact_urls)
    if normalized.endswith(
        (
            "_correlation_id",
            "_job_id",
            "_operation_id",
            "_request_id",
            "_task_id",
            "_trace_id",
        )
    ):
        return True
    parts = normalized.split("_")
    if parts and parts[0] in {"provider", "upstream", "vendor"}:
        return True
    if (
        parts
        and parts[0] in {"remote", "service"}
        and set(parts)
        & {
            "endpoint",
            "id",
            "job",
            "metadata",
            "model",
            "operation",
            "provider",
            "request",
            "task",
            "uri",
            "url",
            "vendor",
        }
    ):
        return True
    return False


def _replace_artifact_urls(
    result: Any,
    paths: tuple[DataPath, ...],
    artifact_urls: Mapping[str, str] | Sequence[str] | None,
) -> None:
    if isinstance(artifact_urls, (str, bytes)):
        raise MappingExtractionError(
            "artifact URL replacements must be an object or list"
        )
    if isinstance(artifact_urls, Mapping):
        replacements = dict(artifact_urls)
        if any(
            not isinstance(old, str) or not isinstance(new, str) or not new
            for old, new in replacements.items()
        ):
            raise MappingExtractionError("artifact URL replacement is invalid")
        found: set[str] = set()
        if paths:
            matches = [
                match
                for path in paths
                for match in _matching_parents(result, path.parts)
            ]
            for parent, key in matches:
                old = parent[key]
                if not isinstance(old, str) or old not in replacements:
                    raise MappingExtractionError(
                        "artifact URL does not have a local replacement"
                    )
                parent[key] = replacements[old]
                found.add(old)
        else:
            _replace_exact_strings(result, replacements, found)
        if found != set(replacements):
            raise MappingExtractionError(
                "artifact URL replacement did not match the response"
            )
        return

    replacements = list(artifact_urls or [])
    if any(not isinstance(value, str) or not value for value in replacements):
        raise MappingExtractionError("artifact URL replacement is invalid")
    matches = [
        match for path in paths for match in _matching_parents(result, path.parts)
    ]
    if len(matches) != len(replacements):
        if matches or replacements:
            raise MappingExtractionError(
                "artifact URL replacement count does not match"
            )
        return
    for (parent, key), replacement in zip(matches, replacements, strict=True):
        if not isinstance(parent[key], str):
            raise MappingExtractionError("artifact URL field is not a string")
        parent[key] = replacement


def _replace_exact_strings(
    value: Any, replacements: Mapping[str, str], found: set[str]
) -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            if isinstance(child, str) and child in replacements:
                found.add(child)
                value[key] = replacements[child]
            else:
                _replace_exact_strings(child, replacements, found)
    elif isinstance(value, list):
        for index, child in enumerate(value):
            if isinstance(child, str) and child in replacements:
                found.add(child)
                value[index] = replacements[child]
            else:
                _replace_exact_strings(child, replacements, found)


def scrub_public_response(
    response: Any,
    config: Mapping[str, Any] | PublicResponseRules | None = None,
    *,
    artifact_urls: Mapping[str, str] | Sequence[str] | None = None,
) -> Any:
    """Return a public response with private metadata and remote URLs removed."""

    return PublicResponseRules.from_config(config).scrub(
        response,
        artifact_urls=artifact_urls,
    )


__all__ = [
    "ArtifactMapping",
    "DataPath",
    "ExtractedArtifact",
    "ExtractedTask",
    "FieldMutation",
    "MappingConfigurationError",
    "MappingExtractionError",
    "PayloadTransform",
    "PublicResponseRules",
    "StreamUsageMode",
    "TaskMapping",
    "UsageMapping",
    "ValueRule",
    "extract_artifacts",
    "extract_task",
    "extract_usage",
    "extract_value",
    "extract_values",
    "is_missing_value",
    "merge_stream_usage",
    "scrub_public_response",
    "transform_payload",
]
