"""Pure usage normalization and rule-based price calculation."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation, ROUND_CEILING, ROUND_FLOOR, ROUND_HALF_UP
from enum import Enum
from typing import Any, Mapping, Sequence


ZERO = Decimal("0")
ONE = Decimal("1")
MILLION = Decimal("1000000")
_MISSING = object()


class PricingConfigurationError(ValueError):
    """Raised when a pricing rule cannot be evaluated safely."""


class UsageValidationError(ValueError):
    """Raised when normalized usage contains an invalid quantity."""


class UsageMetric(str, Enum):
    REQUEST = "request"
    TOKEN = "token"
    IMAGE = "image"
    INPUT_MEDIA_SECOND = "input_media_second"
    OUTPUT_MEDIA_SECOND = "output_media_second"
    CHARACTER = "character"
    COUNT = "count"
    TOOL = "tool"

    @classmethod
    def parse(cls, value: UsageMetric | str) -> UsageMetric:
        if isinstance(value, cls):
            return value
        try:
            return cls(str(value))
        except ValueError as exc:
            allowed = ", ".join(item.value for item in cls)
            raise PricingConfigurationError(
                f"Unsupported usage metric {value!r}; expected one of: {allowed}"
            ) from exc


def _as_decimal(
    value: Any,
    *,
    label: str,
    allow_negative: bool = False,
) -> Decimal:
    if isinstance(value, bool) or value is None:
        raise UsageValidationError(f"{label} must be a finite number")
    try:
        result = value if isinstance(value, Decimal) else Decimal(str(value))
    except (InvalidOperation, TypeError, ValueError) as exc:
        raise UsageValidationError(f"{label} must be a finite number") from exc
    if not result.is_finite():
        raise UsageValidationError(f"{label} must be a finite number")
    if not allow_negative and result < ZERO:
        raise UsageValidationError(f"{label} cannot be negative")
    return result


def _flatten_quantities(
    values: Mapping[str, Any] | int | float | str | Decimal | None,
    *,
    label: str,
    prefix: str = "",
) -> dict[str, Decimal]:
    if values is None:
        return {}
    if not isinstance(values, Mapping):
        return {"default": _as_decimal(values, label=label)}

    flattened: dict[str, Decimal] = {}
    for raw_key, value in values.items():
        key = str(raw_key).strip()
        if not key:
            raise UsageValidationError(f"{label} bucket names cannot be empty")
        full_key = f"{prefix}.{key}" if prefix else key
        if isinstance(value, Mapping):
            flattened.update(_flatten_quantities(value, label=label, prefix=full_key))
        else:
            flattened[full_key] = _as_decimal(value, label=f"{label}.{full_key}")
    return flattened


def _copy_dimensions(values: Mapping[str, Any] | None) -> dict[str, Any]:
    if values is None:
        return {}
    if not isinstance(values, Mapping):
        raise UsageValidationError("dimensions must be an object")
    copied: dict[str, Any] = {}
    for key, value in values.items():
        normalized_key = str(key).strip()
        if not normalized_key:
            raise UsageValidationError("dimension names cannot be empty")
        copied[normalized_key] = (
            _copy_dimensions(value) if isinstance(value, Mapping) else value
        )
    return copied


def _merge_dimensions(
    base: Mapping[str, Any], override: Mapping[str, Any]
) -> dict[str, Any]:
    merged = _copy_dimensions(base)
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(merged.get(key), Mapping):
            merged[key] = _merge_dimensions(merged[key], value)
        else:
            merged[key] = (
                _copy_dimensions(value) if isinstance(value, Mapping) else value
            )
    return merged


def _json_decimal(value: Decimal, *, decimal_as_string: bool) -> str | int | float:
    if decimal_as_string:
        return str(value)
    if value == value.to_integral_value():
        return int(value)
    return float(value)


def _json_value(value: Any, *, decimal_as_string: bool) -> Any:
    if isinstance(value, Decimal):
        return _json_decimal(value, decimal_as_string=decimal_as_string)
    if isinstance(value, datetime):
        parsed = _parse_datetime(value, label="timestamp")
        assert parsed is not None
        return parsed.isoformat().replace("+00:00", "Z")
    if isinstance(value, Mapping):
        return {
            str(key): _json_value(item, decimal_as_string=decimal_as_string)
            for key, item in value.items()
        }
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [
            _json_value(item, decimal_as_string=decimal_as_string) for item in value
        ]
    return value


@dataclass(frozen=True)
class NormalizedUsage:
    """A provider-independent representation of billable usage.

    Bucket names are intentionally open-ended. Common token buckets are ``input``,
    ``output``, and ``cached_input``; media buckets normally name a modality. Nested
    mappings passed to a constructor are flattened with dots.
    """

    requests: Decimal | int | float | str = ONE
    tokens: Mapping[str, Any] = field(default_factory=dict)
    images: Mapping[str, Any] | int | float | str | Decimal = field(
        default_factory=dict
    )
    input_media_seconds: Mapping[str, Any] | int | float | str | Decimal = field(
        default_factory=dict
    )
    output_media_seconds: Mapping[str, Any] | int | float | str | Decimal = field(
        default_factory=dict
    )
    characters: Mapping[str, Any] | int | float | str | Decimal = field(
        default_factory=dict
    )
    counts: Mapping[str, Any] | int | float | str | Decimal = field(
        default_factory=dict
    )
    tools: Mapping[str, Any] | int | float | str | Decimal = field(default_factory=dict)
    dimensions: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "requests", _as_decimal(self.requests, label="requests")
        )
        for attribute in (
            "tokens",
            "images",
            "input_media_seconds",
            "output_media_seconds",
            "characters",
            "counts",
            "tools",
        ):
            object.__setattr__(
                self,
                attribute,
                _flatten_quantities(getattr(self, attribute), label=attribute),
            )
        object.__setattr__(self, "dimensions", _copy_dimensions(self.dimensions))

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> NormalizedUsage:
        """Build normalized usage from its JSON-compatible representation."""

        if not isinstance(values, Mapping):
            raise UsageValidationError("usage must be an object")
        return cls(
            requests=values.get("requests", values.get("request_count", ONE)),
            tokens=values.get("tokens", {}),
            images=values.get("images", values.get("image_count", {})),
            input_media_seconds=values.get("input_media_seconds", {}),
            output_media_seconds=values.get("output_media_seconds", {}),
            characters=values.get("characters", {}),
            counts=values.get("counts", {}),
            tools=values.get("tools", {}),
            dimensions=values.get("dimensions", {}),
        )

    @classmethod
    def from_legacy_metrics(
        cls,
        metrics: Mapping[str, Any] | None,
        *,
        requests: Decimal | int | float | str = ONE,
        dimensions: Mapping[str, Any] | None = None,
    ) -> NormalizedUsage:
        """Normalize the compact metrics already produced by existing invocations."""

        metrics = metrics or {}
        tokens = {}
        for source, bucket in (
            ("it", "input"),
            ("ot", "output"),
            ("ct", "cached_input"),
        ):
            if source in metrics:
                tokens[bucket] = metrics.get(source, 0) or 0
        counts = {"steps": metrics.get("steps", 0) or 0} if "steps" in metrics else {}
        return cls(
            requests=requests,
            tokens=tokens,
            counts=counts,
            dimensions=dimensions or {},
        )

    def quantity(self, metric: UsageMetric | str, bucket: str | None = None) -> Decimal:
        """Return one bucket, or the sum of all buckets when no bucket is supplied."""

        parsed_metric = UsageMetric.parse(metric)
        if parsed_metric is UsageMetric.REQUEST:
            if bucket is not None:
                return ZERO
            return self.requests

        attribute = {
            UsageMetric.TOKEN: "tokens",
            UsageMetric.IMAGE: "images",
            UsageMetric.INPUT_MEDIA_SECOND: "input_media_seconds",
            UsageMetric.OUTPUT_MEDIA_SECOND: "output_media_seconds",
            UsageMetric.CHARACTER: "characters",
            UsageMetric.COUNT: "counts",
            UsageMetric.TOOL: "tools",
        }[parsed_metric]
        quantities: Mapping[str, Decimal] = getattr(self, attribute)
        if bucket is None:
            return sum(quantities.values(), ZERO)
        return quantities.get(bucket, ZERO)

    def has_quantity(
        self, metric: UsageMetric | str, bucket: str | None = None
    ) -> bool:
        """Return whether a metric was observed, including an explicit zero.

        A missing provider usage metric is materially different from a reported
        zero.  Pricing rules must not become authoritative merely because their
        scope matches when the quantity they price was never observed.
        """

        parsed_metric = UsageMetric.parse(metric)
        if parsed_metric is UsageMetric.REQUEST:
            return bucket is None

        attribute = {
            UsageMetric.TOKEN: "tokens",
            UsageMetric.IMAGE: "images",
            UsageMetric.INPUT_MEDIA_SECOND: "input_media_seconds",
            UsageMetric.OUTPUT_MEDIA_SECOND: "output_media_seconds",
            UsageMetric.CHARACTER: "characters",
            UsageMetric.COUNT: "counts",
            UsageMetric.TOOL: "tools",
        }[parsed_metric]
        quantities: Mapping[str, Decimal] = getattr(self, attribute)
        return bool(quantities) if bucket is None else bucket in quantities

    def to_dict(self, *, decimal_as_string: bool = True) -> dict[str, Any]:
        """Return a JSON-compatible representation suitable for persistence."""

        return {
            "requests": _json_decimal(
                self.requests, decimal_as_string=decimal_as_string
            ),
            "tokens": _json_value(self.tokens, decimal_as_string=decimal_as_string),
            "images": _json_value(self.images, decimal_as_string=decimal_as_string),
            "input_media_seconds": _json_value(
                self.input_media_seconds, decimal_as_string=decimal_as_string
            ),
            "output_media_seconds": _json_value(
                self.output_media_seconds, decimal_as_string=decimal_as_string
            ),
            "characters": _json_value(
                self.characters, decimal_as_string=decimal_as_string
            ),
            "counts": _json_value(self.counts, decimal_as_string=decimal_as_string),
            "tools": _json_value(self.tools, decimal_as_string=decimal_as_string),
            "dimensions": _json_value(
                self.dimensions, decimal_as_string=decimal_as_string
            ),
        }


def _parse_datetime(value: datetime | str | None, *, label: str) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        result = value
    elif isinstance(value, str):
        normalized = value.strip()
        if normalized.endswith("Z"):
            normalized = f"{normalized[:-1]}+00:00"
        try:
            result = datetime.fromisoformat(normalized)
        except ValueError as exc:
            raise PricingConfigurationError(
                f"{label} must be an ISO-8601 timestamp"
            ) from exc
    else:
        raise PricingConfigurationError(f"{label} must be an ISO-8601 timestamp")
    if result.tzinfo is None:
        result = result.replace(tzinfo=timezone.utc)
    return result.astimezone(timezone.utc)


@dataclass(frozen=True)
class PricingContext:
    """Request attributes used to select applicable pricing rules."""

    cord: str | None = None
    path: str | None = None
    method: str | None = None
    dimensions: Mapping[str, Any] = field(default_factory=dict)
    at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def __post_init__(self) -> None:
        object.__setattr__(self, "dimensions", _copy_dimensions(self.dimensions))
        parsed_at = _parse_datetime(self.at, label="pricing context timestamp")
        assert parsed_at is not None
        object.__setattr__(self, "at", parsed_at)
        if self.method is not None:
            object.__setattr__(self, "method", self.method.upper())


@dataclass(frozen=True)
class RuleScope:
    cord: str | None = None
    path: str | None = None
    method: str | None = None

    def __post_init__(self) -> None:
        if self.method is not None:
            object.__setattr__(self, "method", self.method.upper())

    def matches(self, context: PricingContext) -> bool:
        return not (
            (self.cord is not None and self.cord != context.cord)
            or (self.path is not None and self.path != context.path)
            or (self.method is not None and self.method != context.method)
        )


_ROUNDING_MODES = {"exact", "ceil", "floor", "nearest"}


@dataclass(frozen=True)
class PricingRule:
    """The price for a normalized usage quantity under optional selectors."""

    metric: UsageMetric
    unit_price: Decimal
    bucket: str | None = None
    unit_size: Decimal = ONE
    conditions: Mapping[str, Any] = field(default_factory=dict)
    scope: RuleScope = field(default_factory=RuleScope)
    effective_from: datetime | None = None
    effective_to: datetime | None = None
    rounding: str = "exact"
    minimum_units: Decimal = ZERO
    maximum_units: Decimal | None = None
    rule_id: str | None = None
    match_group: str | None = None
    priority: int = 0
    fallback: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "metric", UsageMetric.parse(self.metric))
        try:
            unit_price = _as_decimal(self.unit_price, label="unit_price")
            unit_size = _as_decimal(self.unit_size, label="unit_size")
            minimum_units = _as_decimal(self.minimum_units, label="minimum_units")
            maximum_units = (
                _as_decimal(self.maximum_units, label="maximum_units")
                if self.maximum_units is not None
                else None
            )
        except UsageValidationError as exc:
            raise PricingConfigurationError(str(exc)) from exc
        if unit_size == ZERO:
            raise PricingConfigurationError("unit_size must be greater than zero")
        object.__setattr__(self, "unit_price", unit_price)
        object.__setattr__(self, "unit_size", unit_size)
        object.__setattr__(self, "minimum_units", minimum_units)
        object.__setattr__(self, "maximum_units", maximum_units)
        if maximum_units is not None and maximum_units < minimum_units:
            raise PricingConfigurationError(
                "maximum_units must be greater than or equal to minimum_units"
            )

        if self.metric is UsageMetric.REQUEST and self.bucket is not None:
            raise PricingConfigurationError("request rules cannot select a bucket")
        if self.bucket is not None:
            bucket = str(self.bucket).strip()
            if not bucket:
                raise PricingConfigurationError("bucket cannot be empty")
            object.__setattr__(self, "bucket", bucket)

        if self.match_group is not None:
            match_group = str(self.match_group).strip()
            if not match_group or len(match_group) > 128:
                raise PricingConfigurationError(
                    "match_group must contain 1-128 characters"
                )
            object.__setattr__(self, "match_group", match_group)
        if isinstance(self.priority, bool) or not isinstance(self.priority, int):
            raise PricingConfigurationError("priority must be an integer")
        if not -(2**31) <= self.priority < 2**31:
            raise PricingConfigurationError("priority is outside the supported range")
        if not isinstance(self.fallback, bool):
            raise PricingConfigurationError("fallback must be a boolean")
        if self.fallback and self.match_group is None:
            raise PricingConfigurationError("fallback requires match_group")

        if not isinstance(self.conditions, Mapping):
            raise PricingConfigurationError("conditions must be an object")
        try:
            conditions = _copy_dimensions(self.conditions)
        except UsageValidationError as exc:
            raise PricingConfigurationError(str(exc)) from exc
        _validate_conditions(conditions)
        object.__setattr__(self, "conditions", conditions)
        if not isinstance(self.scope, RuleScope):
            raise PricingConfigurationError("scope must be a RuleScope")

        effective_from = _parse_datetime(self.effective_from, label="effective_from")
        effective_to = _parse_datetime(self.effective_to, label="effective_to")
        if (
            effective_from is not None
            and effective_to is not None
            and effective_from >= effective_to
        ):
            raise PricingConfigurationError(
                "effective_from must be earlier than effective_to"
            )
        object.__setattr__(self, "effective_from", effective_from)
        object.__setattr__(self, "effective_to", effective_to)

        rounding = str(self.rounding).lower()
        if rounding not in _ROUNDING_MODES:
            allowed = ", ".join(sorted(_ROUNDING_MODES))
            raise PricingConfigurationError(f"rounding must be one of: {allowed}")
        object.__setattr__(self, "rounding", rounding)

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> PricingRule:
        if not isinstance(value, Mapping):
            raise PricingConfigurationError("each pricing rule must be an object")

        allowed_keys = {
            "id",
            "metric",
            "unit_price",
            "bucket",
            "unit_size",
            "conditions",
            "scope",
            "cord",
            "path",
            "method",
            "effective_from",
            "effective_to",
            "rounding",
            "minimum_units",
            "maximum_units",
            "match_group",
            "priority",
            "fallback",
        }
        unknown_keys = set(value) - allowed_keys
        if unknown_keys:
            raise PricingConfigurationError(
                f"Unsupported pricing rule field(s): {', '.join(sorted(map(str, unknown_keys)))}"
            )

        scope_value = value.get("scope", {})
        if scope_value is None:
            scope_value = {}
        if not isinstance(scope_value, Mapping):
            raise PricingConfigurationError("scope must be an object")
        unknown_scope_keys = set(scope_value) - {"cord", "path", "method"}
        if unknown_scope_keys:
            raise PricingConfigurationError(
                f"Unsupported scope field(s): {', '.join(sorted(map(str, unknown_scope_keys)))}"
            )
        scope = RuleScope(
            cord=_coalesce_scope(value, scope_value, "cord"),
            path=_coalesce_scope(value, scope_value, "path"),
            method=_coalesce_scope(value, scope_value, "method"),
        )
        if "metric" not in value:
            raise PricingConfigurationError("pricing rule is missing metric")
        if "unit_price" not in value:
            raise PricingConfigurationError("pricing rule is missing unit_price")
        return cls(
            metric=UsageMetric.parse(value["metric"]),
            unit_price=value["unit_price"],
            bucket=value.get("bucket"),
            unit_size=value.get("unit_size", ONE),
            conditions=value.get("conditions", {}),
            scope=scope,
            effective_from=value.get("effective_from"),
            effective_to=value.get("effective_to"),
            rounding=value.get("rounding", "exact"),
            minimum_units=value.get("minimum_units", ZERO),
            maximum_units=value.get("maximum_units"),
            rule_id=str(value["id"]) if value.get("id") is not None else None,
            match_group=value.get("match_group"),
            priority=value.get("priority", 0),
            fallback=value.get("fallback", False),
        )

    def matches(self, context: PricingContext) -> bool:
        if not self.scope.matches(context):
            return False
        if self.effective_from is not None and context.at < self.effective_from:
            return False
        if self.effective_to is not None and context.at >= self.effective_to:
            return False
        return all(
            _dimension_matches(context.dimensions, name, expected)
            for name, expected in self.conditions.items()
        )

    def billable_units(self, quantity: Decimal) -> Decimal:
        units = quantity / self.unit_size
        if self.rounding == "ceil":
            units = units.to_integral_value(rounding=ROUND_CEILING)
        elif self.rounding == "floor":
            units = units.to_integral_value(rounding=ROUND_FLOOR)
        elif self.rounding == "nearest":
            units = units.to_integral_value(rounding=ROUND_HALF_UP)
        if quantity > ZERO and units < self.minimum_units:
            units = self.minimum_units
        if self.maximum_units is not None and units > self.maximum_units:
            units = self.maximum_units
        return units


def _coalesce_scope(
    value: Mapping[str, Any], scope: Mapping[str, Any], key: str
) -> str | None:
    top_level = value.get(key)
    nested = scope.get(key)
    if top_level is not None and nested is not None and str(top_level) != str(nested):
        raise PricingConfigurationError(f"conflicting {key!r} scope values")
    selected = top_level if top_level is not None else nested
    if selected is None:
        return None
    result = str(selected).strip()
    if not result:
        raise PricingConfigurationError(f"scope {key!r} cannot be empty")
    return result


def _dimension_value(dimensions: Mapping[str, Any], name: str) -> Any:
    if name in dimensions:
        return dimensions[name]
    value: Any = dimensions
    for part in name.split("."):
        if not isinstance(value, Mapping) or part not in value:
            return _MISSING
        value = value[part]
    return value


def _dimension_matches(dimensions: Mapping[str, Any], name: str, expected: Any) -> bool:
    actual = _dimension_value(dimensions, name)
    if isinstance(expected, Mapping):
        normalized = {
            str(key).removeprefix("$"): value for key, value in expected.items()
        }
        return all(
            _operator_matches(actual, operator, operand)
            for operator, operand in normalized.items()
        )
    if isinstance(expected, Sequence) and not isinstance(
        expected, (str, bytes, bytearray)
    ):
        return actual is not _MISSING and actual in expected
    return actual is not _MISSING and actual == expected


def _operator_matches(actual: Any, operator: str, operand: Any) -> bool:
    if operator == "exists":
        return (actual is not _MISSING) is bool(operand)
    if actual is _MISSING:
        # Pricing selectors fail closed when their source dimension was not
        # observed.  Absence can be selected explicitly with ``exists: false``;
        # it must not accidentally qualify an inequality-based price tier.
        return False
    if operator == "eq":
        return actual == operand
    if operator == "ne":
        return actual != operand
    if operator in {"in", "not_in"}:
        if not isinstance(operand, Sequence) or isinstance(
            operand, (str, bytes, bytearray)
        ):
            raise PricingConfigurationError(f"{operator} condition must be an array")
        included = actual in operand
        return included if operator == "in" else not included
    try:
        if operator == "gt":
            return actual > operand
        if operator == "gte":
            return actual >= operand
        if operator == "lt":
            return actual < operand
        if operator == "lte":
            return actual <= operand
    except TypeError:
        return False
    raise PricingConfigurationError(f"Unsupported condition operator: {operator}")


_CONDITION_OPERATORS = {
    "eq",
    "ne",
    "in",
    "not_in",
    "gt",
    "gte",
    "lt",
    "lte",
    "exists",
}


def _validate_conditions(conditions: Mapping[str, Any]) -> None:
    for expected in conditions.values():
        if not isinstance(expected, Mapping):
            continue
        normalized = {
            str(key).removeprefix("$"): value for key, value in expected.items()
        }
        if not normalized:
            raise PricingConfigurationError(
                "condition operator objects cannot be empty"
            )
        unknown = set(normalized) - _CONDITION_OPERATORS
        if unknown:
            raise PricingConfigurationError(
                f"Unsupported condition operator(s): {', '.join(sorted(unknown))}"
            )
        for operator, operand in normalized.items():
            if operator == "exists" and not isinstance(operand, bool):
                raise PricingConfigurationError("exists condition must be a boolean")
            if operator in {"in", "not_in"} and (
                not isinstance(operand, Sequence)
                or isinstance(operand, (str, bytes, bytearray))
            ):
                raise PricingConfigurationError(
                    f"{operator} condition must be an array"
                )


@dataclass(frozen=True)
class PricingLineItem:
    rule_id: str
    metric: UsageMetric
    bucket: str | None
    quantity: Decimal
    billable_units: Decimal
    unit_price: Decimal
    amount: Decimal

    def to_dict(self, *, decimal_as_string: bool = True) -> dict[str, Any]:
        return {
            "rule_id": self.rule_id,
            "metric": self.metric.value,
            "bucket": self.bucket,
            "quantity": _json_decimal(
                self.quantity, decimal_as_string=decimal_as_string
            ),
            "billable_units": _json_decimal(
                self.billable_units, decimal_as_string=decimal_as_string
            ),
            "unit_price": _json_decimal(
                self.unit_price, decimal_as_string=decimal_as_string
            ),
            "amount": _json_decimal(self.amount, decimal_as_string=decimal_as_string),
        }


@dataclass(frozen=True)
class PricingResult:
    amount: Decimal = ZERO
    line_items: tuple[PricingLineItem, ...] = ()
    matched_rule_count: int = 0
    source: str = "none"
    missing_rule_count: int = 0

    @property
    def applied(self) -> bool:
        return self.matched_rule_count > 0

    @property
    def complete(self) -> bool:
        """Whether every context-applicable charge component was observed."""

        return self.missing_rule_count == 0

    def to_dict(self, *, decimal_as_string: bool = True) -> dict[str, Any]:
        """Return a JSON-compatible calculation result and line-item audit trail."""

        return {
            "amount": _json_decimal(self.amount, decimal_as_string=decimal_as_string),
            "applied": self.applied,
            "source": self.source,
            "matched_rule_count": self.matched_rule_count,
            "complete": self.complete,
            "missing_rule_count": self.missing_rule_count,
            "line_items": [
                item.to_dict(decimal_as_string=decimal_as_string)
                for item in self.line_items
            ],
        }


def parse_pricing_rules(
    values: Sequence[PricingRule | Mapping[str, Any]] | None,
) -> tuple[PricingRule, ...]:
    """Validate and freeze a JSON-compatible pricing-rule list."""

    if values is None:
        return ()
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes, bytearray)):
        raise PricingConfigurationError("pricing_rules must be an array")
    parsed = tuple(
        value if isinstance(value, PricingRule) else PricingRule.from_mapping(value)
        for value in values
    )
    _validate_match_groups(parsed)
    return parsed


def _validate_match_groups(rules: Sequence[PricingRule]) -> None:
    """Validate explicit mutually-exclusive tier groups.

    Ungrouped rules retain the historical additive behavior. Grouped rules form
    one charge component: the highest-priority matching tier wins, with a
    mandatory unconditional fallback when no selector matches.
    """

    grouped: dict[str, list[PricingRule]] = {}
    for rule in rules:
        if rule.match_group is not None:
            grouped.setdefault(rule.match_group, []).append(rule)

    for name, members in grouped.items():
        if len(members) < 2:
            raise PricingConfigurationError(
                f"match_group {name!r} requires at least one tier and one fallback"
            )
        fallbacks = [rule for rule in members if rule.fallback]
        if len(fallbacks) != 1:
            raise PricingConfigurationError(
                f"match_group {name!r} requires exactly one fallback rule"
            )
        if fallbacks[0].conditions:
            raise PricingConfigurationError(
                f"match_group {name!r} fallback rule cannot have conditions"
            )
        tiers = [rule for rule in members if not rule.fallback]
        if any(not rule.conditions for rule in tiers):
            raise PricingConfigurationError(
                f"match_group {name!r} tiers must have conditions"
            )
        priorities = [rule.priority for rule in tiers]
        if len(priorities) != len(set(priorities)):
            raise PricingConfigurationError(
                f"match_group {name!r} tier priorities must be unique"
            )

        first = members[0]
        signature = (
            first.metric,
            first.bucket,
            first.scope,
            first.effective_from,
            first.effective_to,
        )
        if any(
            (
                rule.metric,
                rule.bucket,
                rule.scope,
                rule.effective_from,
                rule.effective_to,
            )
            != signature
            for rule in members[1:]
        ):
            raise PricingConfigurationError(
                f"match_group {name!r} rules must share metric, bucket, scope, and effective window"
            )


def validate_conditional_pricing_coverage(
    rules: Sequence[PricingRule | Mapping[str, Any]],
) -> tuple[PricingRule, ...]:
    """Require a deterministic default for every conditional price component.

    Explicit ``match_group`` tiers already have a mandatory fallback.  An
    ungrouped conditional rule is additive, so it must have an unconditional
    rule for the same metric, bucket, scope, and effective window.  That rule is
    the component's baseline (and the conditional rule is a surcharge).  This
    prevents a request from reaching a funded upstream when no price rule can
    match its eventual dimensions.
    """

    parsed = parse_pricing_rules(rules)

    def signature(rule: PricingRule) -> tuple[Any, ...]:
        return (
            rule.metric,
            rule.bucket,
            rule.scope,
            rule.effective_from,
            rule.effective_to,
        )

    baselines = {
        signature(rule)
        for rule in parsed
        if rule.match_group is None and not rule.conditions
    }
    for rule in parsed:
        if (
            rule.match_group is None
            and rule.conditions
            and signature(rule) not in baselines
        ):
            identifier = f" {rule.rule_id!r}" if rule.rule_id else ""
            raise PricingConfigurationError(
                "conditional ungrouped pricing rule"
                f"{identifier} requires an unconditional rule with the same "
                "metric, bucket, scope, and effective window; use match_group "
                "with a fallback for mutually exclusive tiers"
            )
    return parsed


def price_usage(
    usage: NormalizedUsage,
    rules: Sequence[PricingRule | Mapping[str, Any]],
    context: PricingContext | None = None,
) -> PricingResult:
    """Apply additive rules and one winner per explicit match group."""

    parsed_rules = parse_pricing_rules(rules)
    context = context or PricingContext(dimensions=usage.dimensions)
    merged_dimensions = _merge_dimensions(usage.dimensions, context.dimensions)
    if merged_dimensions != context.dimensions:
        context = PricingContext(
            cord=context.cord,
            path=context.path,
            method=context.method,
            dimensions=merged_dimensions,
            at=context.at,
        )

    contextual: list[tuple[int, PricingRule]] = []
    for index, rule in enumerate(parsed_rules):
        if not rule.matches(context):
            continue
        contextual.append((index, rule))

    applicable = [
        (index, rule)
        for index, rule in contextual
        if usage.has_quantity(rule.metric, rule.bucket)
    ]
    missing_rule_count = sum(
        1
        for _index, rule in contextual
        if rule.match_group is None and not usage.has_quantity(rule.metric, rule.bucket)
    )
    contextual_groups = {
        rule.match_group for _index, rule in contextual if rule.match_group is not None
    }
    applicable_groups = {
        rule.match_group for _index, rule in applicable if rule.match_group is not None
    }
    missing_rule_count += len(contextual_groups - applicable_groups)

    selected_indexes = {index for index, rule in applicable if rule.match_group is None}
    grouped: dict[str, list[tuple[int, PricingRule]]] = {}
    for index, rule in applicable:
        if rule.match_group is not None:
            grouped.setdefault(rule.match_group, []).append((index, rule))
    for members in grouped.values():
        tiers = [item for item in members if not item[1].fallback]
        selected = max(tiers, key=lambda item: item[1].priority) if tiers else None
        if selected is None:
            selected = next(item for item in members if item[1].fallback)
        selected_indexes.add(selected[0])

    line_items: list[PricingLineItem] = []
    for index, rule in applicable:
        if index not in selected_indexes:
            continue
        quantity = usage.quantity(rule.metric, rule.bucket)
        billable_units = rule.billable_units(quantity)
        amount = billable_units * rule.unit_price
        line_items.append(
            PricingLineItem(
                rule_id=rule.rule_id or f"rule-{index + 1}",
                metric=rule.metric,
                bucket=rule.bucket,
                quantity=quantity,
                billable_units=billable_units,
                unit_price=rule.unit_price,
                amount=amount,
            )
        )
    return PricingResult(
        amount=sum((item.amount for item in line_items), ZERO),
        line_items=tuple(line_items),
        matched_rule_count=len(line_items),
        source="rules" if line_items or missing_rule_count else "none",
        missing_rule_count=missing_rule_count,
    )


def price_override(
    override: Any,
    usage: NormalizedUsage,
    context: PricingContext | None = None,
    *,
    combine_legacy_components: bool = False,
) -> PricingResult:
    """Price either a configured rule set or a legacy-only override.

    A row containing ``pricing_rules`` never mixes in its legacy columns. Callers
    that retain an older platform pricing path can inspect ``result.applied`` and
    perform that fallback with the complete legacy context they own.
    """

    rules = getattr(override, "pricing_rules", None)
    if rules is not None:
        return price_usage(usage, rules, context)
    if combine_legacy_components:
        return _price_combined_legacy_override(override, usage)
    return _price_legacy_override(override, usage)


def validate_legacy_pricing_rates(values: Mapping[str, Any]) -> None:
    """Reject unsafe legacy prices while preserving nullable fallback fields.

    Hosted pricing historically permits either side of token pricing to be null
    so the ordinary compute-derived price can fill it in.  Null therefore remains
    valid, but an explicitly configured rate must be finite and non-negative and
    a cache discount must remain within its fractional range.
    """

    for name in ("per_million_in", "per_million_out", "per_step", "per_request"):
        value = values.get(name)
        if value is None:
            continue
        try:
            _as_decimal(value, label=name)
        except UsageValidationError as exc:
            raise PricingConfigurationError(str(exc)) from exc

    cache_discount = values.get("cache_discount")
    if cache_discount is None:
        return
    try:
        discount = _as_decimal(cache_discount, label="cache_discount")
    except UsageValidationError as exc:
        raise PricingConfigurationError(str(exc)) from exc
    if discount > ONE:
        raise PricingConfigurationError("cache_discount must be between zero and one")


def _price_combined_legacy_override(
    override: Any, usage: NormalizedUsage
) -> PricingResult:
    """Add every observed legacy component without changing hosted precedence.

    External acceptance snapshots may deliberately combine a request minimum with
    completion-priced units. Missing provider observations leave the calculation
    incomplete, while every independently observed component remains auditable.
    """

    validate_legacy_pricing_rates(
        {
            name: getattr(override, name, None)
            for name in (
                "per_million_in",
                "per_million_out",
                "per_step",
                "per_request",
                "cache_discount",
            )
        }
    )

    line_items: list[PricingLineItem] = []
    missing_rule_count = 0
    per_million_in = getattr(override, "per_million_in", None)
    if per_million_in is not None:
        rate = _legacy_decimal(per_million_in, "per_million_in")
        if usage.has_quantity(UsageMetric.TOKEN, "input"):
            input_quantity = usage.quantity(UsageMetric.TOKEN, "input")
            line_items.append(
                _legacy_line(
                    "legacy-input-token",
                    UsageMetric.TOKEN,
                    "input",
                    input_quantity,
                    rate,
                )
            )
        else:
            input_quantity = ZERO
            missing_rule_count += 1

        cache_discount = getattr(override, "cache_discount", None)
        if cache_discount is not None:
            discount = _legacy_decimal(cache_discount, "cache_discount")
            if not ZERO <= discount <= ONE:
                raise PricingConfigurationError(
                    "cache_discount must be between zero and one"
                )
            if usage.has_quantity(UsageMetric.TOKEN, "cached_input"):
                cached = usage.quantity(UsageMetric.TOKEN, "cached_input")
                # Cached tokens cannot discount more input than was charged. A
                # hostile or inconsistent provider total therefore cannot turn
                # the input component, request baseline, or output usage negative.
                discounted = min(cached, input_quantity)
                if discounted:
                    line_items.append(
                        _legacy_line(
                            "legacy-cached-input-discount",
                            UsageMetric.TOKEN,
                            "cached_input",
                            discounted,
                            -(rate * discount),
                        )
                    )
            else:
                missing_rule_count += 1

    per_million_out = getattr(override, "per_million_out", None)
    if per_million_out is not None:
        rate = _legacy_decimal(per_million_out, "per_million_out")
        if usage.has_quantity(UsageMetric.TOKEN, "output"):
            line_items.append(
                _legacy_line(
                    "legacy-output-token",
                    UsageMetric.TOKEN,
                    "output",
                    usage.quantity(UsageMetric.TOKEN, "output"),
                    rate,
                )
            )
        else:
            missing_rule_count += 1

    per_step = getattr(override, "per_step", None)
    if per_step is not None:
        rate = _legacy_decimal(per_step, "per_step")
        if usage.has_quantity(UsageMetric.COUNT, "steps"):
            line_items.append(
                _legacy_line(
                    "legacy-step",
                    UsageMetric.COUNT,
                    "steps",
                    usage.quantity(UsageMetric.COUNT, "steps"),
                    rate,
                    unit_size=ONE,
                )
            )
        else:
            missing_rule_count += 1

    per_request = getattr(override, "per_request", None)
    if per_request is not None:
        line_items.append(
            _legacy_line(
                "legacy-request",
                UsageMetric.REQUEST,
                None,
                usage.requests,
                _legacy_decimal(per_request, "per_request"),
                unit_size=ONE,
            )
        )

    return PricingResult(
        amount=max(sum((item.amount for item in line_items), ZERO), ZERO),
        line_items=tuple(line_items),
        matched_rule_count=len(line_items),
        source="legacy" if line_items or missing_rule_count else "none",
        missing_rule_count=missing_rule_count,
    )


def _price_legacy_override(override: Any, usage: NormalizedUsage) -> PricingResult:
    per_million_in = getattr(override, "per_million_in", None)
    per_million_out = getattr(override, "per_million_out", None)
    if usage.tokens and (per_million_in is not None or per_million_out is not None):
        line_items: list[PricingLineItem] = []
        if per_million_in is not None:
            rate = _legacy_decimal(per_million_in, "per_million_in")
            quantity = usage.quantity(UsageMetric.TOKEN, "input")
            line_items.append(
                _legacy_line(
                    "legacy-input-token", UsageMetric.TOKEN, "input", quantity, rate
                )
            )
            cached = usage.quantity(UsageMetric.TOKEN, "cached_input")
            cache_discount = getattr(override, "cache_discount", None)
            if cached and cache_discount is not None:
                discount = _legacy_decimal(cache_discount, "cache_discount")
                if ZERO <= discount <= ONE:
                    discount_rate = -(rate * discount)
                    line_items.append(
                        _legacy_line(
                            "legacy-cached-input-discount",
                            UsageMetric.TOKEN,
                            "cached_input",
                            cached,
                            discount_rate,
                        )
                    )
        if per_million_out is not None:
            rate = _legacy_decimal(per_million_out, "per_million_out")
            quantity = usage.quantity(UsageMetric.TOKEN, "output")
            line_items.append(
                _legacy_line(
                    "legacy-output-token", UsageMetric.TOKEN, "output", quantity, rate
                )
            )
        return PricingResult(
            amount=sum((item.amount for item in line_items), ZERO),
            line_items=tuple(line_items),
            matched_rule_count=len(line_items),
            source="legacy",
        )

    per_step = getattr(override, "per_step", None)
    if per_step is not None and "steps" in usage.counts:
        rate = _legacy_decimal(per_step, "per_step")
        line = _legacy_line(
            "legacy-step",
            UsageMetric.COUNT,
            "steps",
            usage.quantity(UsageMetric.COUNT, "steps"),
            rate,
            unit_size=ONE,
        )
        return PricingResult(line.amount, (line,), 1, "legacy")

    per_request = getattr(override, "per_request", None)
    if per_request is not None:
        rate = _legacy_decimal(per_request, "per_request")
        line = _legacy_line(
            "legacy-request",
            UsageMetric.REQUEST,
            None,
            usage.requests,
            rate,
            unit_size=ONE,
        )
        return PricingResult(line.amount, (line,), 1, "legacy")
    return PricingResult()


def _legacy_decimal(value: Any, label: str) -> Decimal:
    try:
        return _as_decimal(value, label=label, allow_negative=True)
    except UsageValidationError as exc:
        raise PricingConfigurationError(str(exc)) from exc


def _legacy_line(
    rule_id: str,
    metric: UsageMetric,
    bucket: str | None,
    quantity: Decimal,
    unit_price: Decimal,
    *,
    unit_size: Decimal = MILLION,
) -> PricingLineItem:
    billable_units = quantity / unit_size
    return PricingLineItem(
        rule_id=rule_id,
        metric=metric,
        bucket=bucket,
        quantity=quantity,
        billable_units=billable_units,
        unit_price=unit_price,
        amount=billable_units * unit_price,
    )
