from datetime import datetime, timezone
from decimal import Decimal
import json
from types import SimpleNamespace

import pytest

from api.payment.pricing import (
    NormalizedUsage,
    PricingConfigurationError,
    PricingContext,
    PricingRule,
    UsageMetric,
    UsageValidationError,
    parse_pricing_rules,
    price_override,
    price_usage,
    validate_legacy_pricing_rates,
)


def rule(metric, price, **extra):
    return {"metric": metric, "unit_price": price, **extra}


def override(**values):
    defaults = {
        "per_request": None,
        "per_million_in": None,
        "per_million_out": None,
        "per_step": None,
        "cache_discount": None,
        "pricing_rules": None,
    }
    defaults.update(values)
    return SimpleNamespace(**defaults)


def test_normalized_usage_flattens_nested_buckets_and_uses_exact_decimals():
    usage = NormalizedUsage.from_mapping(
        {
            "request_count": "2",
            "tokens": {"input": 100, "output": "25.5"},
            "images": 3,
            "input_media_seconds": {"audio": "1.25"},
            "output_media_seconds": {"video": {"generated": 5}},
            "characters": {"input": 20},
            "counts": {"operation": 2},
            "tools": {"lookup": 4},
            "dimensions": {"output": {"resolution": "high"}},
        }
    )

    assert usage.requests == Decimal("2")
    assert usage.tokens == {"input": Decimal("100"), "output": Decimal("25.5")}
    assert usage.images == {"default": Decimal("3")}
    assert usage.output_media_seconds == {"video.generated": Decimal("5")}
    assert usage.quantity("token") == Decimal("125.5")
    assert usage.quantity("output_media_second", "video.generated") == Decimal("5")
    assert usage.dimensions == {"output": {"resolution": "high"}}


def test_normalized_usage_serialization_is_json_compatible_and_round_trips():
    usage = NormalizedUsage(
        requests=2,
        tokens={"input": "2.25"},
        output_media_seconds={"video": "3.5"},
        dimensions={"resolution": "high"},
    )

    persisted = usage.to_dict()
    public = usage.to_dict(decimal_as_string=False)

    assert json.loads(json.dumps(persisted)) == persisted
    assert persisted["tokens"]["input"] == "2.25"
    assert public["requests"] == 2
    assert public["tokens"]["input"] == 2.25
    assert NormalizedUsage.from_mapping(persisted) == usage


@pytest.mark.parametrize("bad_value", [-1, True, None, "NaN", "Infinity", object()])
def test_normalized_usage_rejects_invalid_quantities(bad_value):
    with pytest.raises(UsageValidationError):
        NormalizedUsage(tokens={"input": bad_value})


def test_price_usage_supports_every_normalized_metric():
    usage = NormalizedUsage(
        tokens={"input": 500_000, "output": 250_000},
        images={"output": 2},
        input_media_seconds={"audio": "2.5"},
        output_media_seconds={"video": 5},
        characters={"input": 25_000},
        counts={"operation": 2},
        tools={"lookup": 3},
    )
    rules = [
        rule("request", ".05", id="base"),
        rule("token", "2", bucket="input", unit_size=1_000_000),
        rule("token", "4", bucket="output", unit_size=1_000_000),
        rule("image", ".1", bucket="output"),
        rule("input_media_second", ".01", bucket="audio"),
        rule("output_media_second", ".2", bucket="video"),
        rule("character", ".1", bucket="input", unit_size=10_000),
        rule("count", ".05", bucket="operation"),
        rule("tool", ".01", bucket="lookup"),
    ]

    result = price_usage(usage, rules)

    assert result.applied is True
    assert result.source == "rules"
    assert result.matched_rule_count == 9
    assert result.amount == Decimal("3.655")
    assert [item.rule_id for item in result.line_items[:2]] == ["base", "rule-2"]


def test_bucketless_rule_prices_the_sum_of_a_metric():
    usage = NormalizedUsage(images={"generated": 2, "edited": 3})

    result = price_usage(usage, [rule("image", ".25")])

    assert result.amount == Decimal("1.25")
    assert result.line_items[0].quantity == Decimal("5")


def test_rule_scope_conditions_and_effective_window_all_have_to_match():
    usage = NormalizedUsage(
        output_media_seconds={"video": 10},
        dimensions={"resolution": "high", "mode": "fast"},
    )
    rules = [
        rule(
            "output_media_second",
            ".2",
            bucket="video",
            cord="generate",
            path="/video",
            method="post",
            conditions={"resolution": "high", "mode": {"in": ["fast", "balanced"]}},
            effective_from="2026-01-01T00:00:00Z",
            effective_to="2027-01-01T00:00:00Z",
        )
    ]
    matching = PricingContext(
        cord="generate",
        path="/video",
        method="POST",
        dimensions=usage.dimensions,
        at=datetime(2026, 6, 1, tzinfo=timezone.utc),
    )

    assert price_usage(usage, rules, matching).amount == Decimal("2.0")
    assert not price_usage(
        usage,
        rules,
        PricingContext(
            cord="other",
            path="/video",
            method="POST",
            dimensions=usage.dimensions,
            at=matching.at,
        ),
    ).applied
    assert not price_usage(
        usage,
        rules,
        PricingContext(
            cord="generate",
            path="/video",
            method="POST",
            dimensions={"resolution": "low", "mode": "fast"},
            at=matching.at,
        ),
    ).applied


def test_effective_from_is_inclusive_and_effective_to_is_exclusive():
    rule_values = [
        rule(
            "request",
            1,
            effective_from="2026-01-01T00:00:00+00:00",
            effective_to="2026-02-01T00:00:00+00:00",
        )
    ]
    usage = NormalizedUsage()

    assert price_usage(
        usage,
        rule_values,
        PricingContext(at=datetime(2026, 1, 1, tzinfo=timezone.utc)),
    ).applied
    assert not price_usage(
        usage,
        rule_values,
        PricingContext(at=datetime(2026, 2, 1, tzinfo=timezone.utc)),
    ).applied


def test_conditions_support_nested_dimensions_comparisons_and_existence():
    usage = NormalizedUsage(
        characters={"input": 100},
        dimensions={"output": {"width": 1920}, "mode": "quality"},
    )
    rules = [
        rule(
            "character",
            ".01",
            bucket="input",
            conditions={
                "output.width": {"gte": 1280, "lt": 3840},
                "mode": {"not_in": ["draft"]},
                "seed": {"exists": False},
            },
        )
    ]

    assert price_usage(usage, rules).amount == Decimal("1.00")


@pytest.mark.parametrize("operator", ["ne", "not_in", "gt", "gte", "lt", "lte"])
def test_missing_dimensions_only_match_an_explicit_absence_condition(operator):
    operand = ["draft"] if operator == "not_in" else "draft"
    usage = NormalizedUsage()

    assert not price_usage(
        usage,
        [rule("request", 1, conditions={"mode": {operator: operand}})],
    ).applied
    assert price_usage(
        usage,
        [rule("request", 1, conditions={"mode": {"exists": False}})],
    ).applied


@pytest.mark.parametrize(
    ("rounding", "expected_units"),
    [("exact", "1.2"), ("ceil", "2"), ("floor", "1"), ("nearest", "1")],
)
def test_rule_unit_rounding(rounding, expected_units):
    usage = NormalizedUsage(input_media_seconds={"audio": 12})
    result = price_usage(
        usage,
        [
            rule(
                "input_media_second",
                3,
                bucket="audio",
                unit_size=10,
                rounding=rounding,
            )
        ],
    )

    assert result.line_items[0].billable_units == Decimal(expected_units)


def test_minimum_units_applies_only_to_nonzero_usage():
    rules = [rule("tool", ".2", bucket="lookup", minimum_units=3)]

    used = price_usage(NormalizedUsage(tools={"lookup": 1}), rules)
    explicit_zero = price_usage(NormalizedUsage(tools={"lookup": 0}), rules)
    unobserved = price_usage(NormalizedUsage(), rules)

    assert used.amount == Decimal(".6")
    assert explicit_zero.amount == Decimal("0")
    assert explicit_zero.applied
    assert unobserved.amount == Decimal("0")
    assert not unobserved.applied


def test_scope_match_does_not_apply_when_metric_or_bucket_was_not_observed():
    usage = NormalizedUsage(tokens={"input": 0})
    rules = [
        rule("token", 2, bucket="input", scope={"cord": "chat"}),
        rule("token", 4, bucket="output", scope={"cord": "chat"}),
        rule("image", 1, scope={"cord": "chat"}),
    ]

    result = price_usage(usage, rules, PricingContext(cord="chat"))

    assert result.applied
    assert result.matched_rule_count == 1
    assert [item.bucket for item in result.line_items] == ["input"]
    assert result.amount == 0
    assert not result.complete
    assert result.missing_rule_count == 2


def test_missing_charge_component_cannot_hide_behind_an_applied_request_fee():
    result = price_usage(
        NormalizedUsage(requests=1),
        [
            rule("request", ".1"),
            rule("token", "2", bucket="output", unit_size=1_000_000),
        ],
    )

    assert result.applied
    assert result.amount == Decimal(".1")
    assert not result.complete
    assert result.missing_rule_count == 1


def test_match_group_selects_highest_priority_tier_and_has_a_fallback():
    rules = [
        rule(
            "output_media_second",
            ".14",
            bucket="video",
            match_group="video-resolution",
            priority=10,
            conditions={"height": {"gte": 720}},
        ),
        rule(
            "output_media_second",
            ".28",
            bucket="video",
            match_group="video-resolution",
            priority=20,
            conditions={"height": {"gte": 1080}},
        ),
        rule(
            "output_media_second",
            ".35",
            bucket="video",
            match_group="video-resolution",
            fallback=True,
        ),
    ]
    usage = NormalizedUsage(output_media_seconds={"video": 10})

    highest = price_usage(
        usage,
        rules,
        PricingContext(dimensions={"height": 1080}),
    )
    fallback = price_usage(
        usage,
        rules,
        PricingContext(dimensions={"height": "unrecognized"}),
    )

    assert highest.amount == Decimal("2.80")
    assert highest.matched_rule_count == 1
    assert highest.line_items[0].unit_price == Decimal(".28")
    assert fallback.amount == Decimal("3.50")
    assert fallback.matched_rule_count == 1
    assert fallback.line_items[0].unit_price == Decimal(".35")


def test_ungrouped_pricing_rules_remain_additive():
    result = price_usage(
        NormalizedUsage(requests=1, tokens={"output": 1_000_000}),
        [
            rule("request", ".1"),
            rule("token", "2", bucket="output", unit_size=1_000_000),
            rule("token", ".5", bucket="output", unit_size=1_000_000),
        ],
    )

    assert result.matched_rule_count == 3
    assert result.amount == Decimal("2.6")


@pytest.mark.parametrize(
    ("rules", "message"),
    [
        (
            [
                rule(
                    "request",
                    1,
                    match_group="tier",
                    priority=1,
                    conditions={"mode": "fast"},
                )
            ],
            "at least one tier and one fallback",
        ),
        (
            [
                rule(
                    "request",
                    1,
                    match_group="tier",
                    priority=1,
                    conditions={"mode": "fast"},
                ),
                rule(
                    "request",
                    2,
                    match_group="tier",
                    priority=1,
                    conditions={"mode": "slow"},
                ),
                rule("request", 3, match_group="tier", fallback=True),
            ],
            "priorities must be unique",
        ),
        (
            [
                rule(
                    "request",
                    1,
                    match_group="tier",
                    priority=1,
                    conditions={"mode": "fast"},
                ),
                rule(
                    "request",
                    2,
                    match_group="tier",
                    fallback=True,
                    conditions={"mode": {"exists": False}},
                ),
            ],
            "fallback rule cannot have conditions",
        ),
    ],
)
def test_match_group_configuration_fails_closed_before_pricing(rules, message):
    with pytest.raises(PricingConfigurationError, match=message):
        parse_pricing_rules(rules)


def test_maximum_units_caps_billable_usage():
    rules = [
        rule(
            "input_media_second",
            ".14",
            bucket="video",
            maximum_units=5,
        )
    ]

    result = price_usage(NormalizedUsage(input_media_seconds={"video": 18}), rules)

    assert result.amount == Decimal("0.70")
    assert result.line_items[0].quantity == 18
    assert result.line_items[0].billable_units == 5


def test_maximum_units_cannot_be_less_than_minimum_units():
    with pytest.raises(PricingConfigurationError, match="maximum_units"):
        parse_pricing_rules([rule("count", "1", minimum_units=4, maximum_units=3)])


@pytest.mark.parametrize(
    "values",
    [
        [{"unit_price": 1}],
        [{"metric": "token"}],
        [rule("unknown", 1)],
        [rule("request", 1, bucket="input")],
        [rule("request", 1, unit_size=0)],
        [rule("request", -1)],
        [rule("request", 1, effective_from="2026-02-01", effective_to="2026-01-01")],
        {"metric": "request", "unit_price": 1},
    ],
)
def test_invalid_rule_configuration_fails_closed(values):
    with pytest.raises(PricingConfigurationError):
        parse_pricing_rules(values)


@pytest.mark.parametrize(
    "bad_rule",
    [
        rule("request", 1, unexpected=True),
        rule("request", 1, scope={"unexpected": True}),
        rule("request", 1, conditions={"mode": {"equals": "fast"}}),
        rule("request", 1, conditions={"mode": {"in": "fast"}}),
        rule("request", 1, conditions={"mode": {"exists": "yes"}}),
    ],
)
def test_unknown_fields_and_malformed_conditions_fail_during_parsing(bad_rule):
    with pytest.raises(PricingConfigurationError):
        parse_pricing_rules([bad_rule])


def test_conflicting_flat_and_nested_scope_is_rejected():
    with pytest.raises(PricingConfigurationError, match="conflicting"):
        PricingRule.from_mapping(rule("request", 1, cord="one", scope={"cord": "two"}))


def test_legacy_request_override_is_unchanged():
    result = price_override(override(per_request=0.125), NormalizedUsage(requests=2))

    assert result.source == "legacy"
    assert result.amount == Decimal("0.250")
    assert result.line_items[0].metric is UsageMetric.REQUEST


def test_legacy_negative_rate_behavior_is_preserved():
    result = price_override(override(per_request="-.25"), NormalizedUsage())

    assert result.source == "legacy"
    assert result.amount == Decimal("-.25")


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("per_request", "-0.01"),
        ("per_million_in", "-0.01"),
        ("per_million_out", "-0.01"),
        ("per_step", "-0.01"),
        ("cache_discount", "-0.01"),
        ("cache_discount", "1.01"),
    ],
)
def test_external_combined_legacy_pricing_rejects_unsafe_rates(field, value):
    with pytest.raises(PricingConfigurationError, match=field):
        price_override(
            override(**{field: value}),
            NormalizedUsage(),
            combine_legacy_components=True,
        )


def test_legacy_pricing_validation_preserves_nullable_fallback_fields():
    validate_legacy_pricing_rates(
        {
            "per_request": None,
            "per_million_in": None,
            "per_million_out": None,
            "per_step": None,
            "cache_discount": None,
        }
    )


def test_legacy_step_override_takes_precedence_over_request():
    usage = NormalizedUsage(counts={"steps": 12})

    result = price_override(
        override(per_request=10, per_step=".025"),
        usage,
    )

    assert result.amount == Decimal(".300")
    assert result.line_items[0].bucket == "steps"


def test_legacy_token_override_preserves_cached_input_discount_formula():
    usage = NormalizedUsage(
        tokens={"input": 1_000_000, "cached_input": 250_000, "output": 500_000}
    )

    result = price_override(
        override(
            per_request=99, per_million_in=2, per_million_out=4, cache_discount=0.5
        ),
        usage,
    )

    assert result.amount == Decimal("3.750")
    assert result.line_items[1].amount == Decimal("-0.250")
    assert result.source == "legacy"


def test_combined_legacy_components_keep_request_floor_and_clamp_cache_discount():
    usage = NormalizedUsage(tokens={"input": 1_000_000, "cached_input": 5_000_000})

    result = price_override(
        override(per_request=".25", per_million_in=2, cache_discount=1),
        usage,
        combine_legacy_components=True,
    )

    assert result.complete
    assert result.amount == Decimal(".25")
    assert [item.rule_id for item in result.line_items] == [
        "legacy-input-token",
        "legacy-cached-input-discount",
        "legacy-request",
    ]
    assert result.line_items[1].quantity == Decimal("1000000")


def test_combined_legacy_request_floor_prices_when_usage_component_is_missing():
    result = price_override(
        override(per_request=".1", per_million_out=2),
        NormalizedUsage(requests=1),
        combine_legacy_components=True,
    )

    assert result.applied
    assert result.amount == Decimal(".1")
    assert not result.complete
    assert result.missing_rule_count == 1
    assert [item.rule_id for item in result.line_items] == ["legacy-request"]


def test_compact_legacy_metrics_only_create_buckets_that_were_observed():
    empty_usage = NormalizedUsage.from_legacy_metrics({})
    token_usage = NormalizedUsage.from_legacy_metrics({"it": 0, "ot": 5, "ct": 0})

    assert empty_usage.tokens == {}
    assert empty_usage.counts == {}
    assert token_usage.tokens == {
        "input": Decimal("0"),
        "output": Decimal("5"),
        "cached_input": Decimal("0"),
    }


def test_matching_new_rules_are_authoritative_over_legacy_fields():
    result = price_override(
        override(
            per_request=99,
            pricing_rules=[rule("request", ".5", cord="selected")],
        ),
        NormalizedUsage(),
        PricingContext(cord="selected"),
    )

    assert result.source == "rules"
    assert result.amount == Decimal(".5")


def test_nonmatching_scoped_rules_do_not_mix_in_legacy_fields():
    result = price_override(
        override(
            per_request=".25",
            pricing_rules=[rule("request", ".5", cord="selected")],
        ),
        NormalizedUsage(),
        PricingContext(cord="different"),
    )

    assert not result.applied
    assert result.source == "none"
    assert result.amount == Decimal("0")


def test_matching_zero_price_rule_does_not_fall_back_to_legacy_fields():
    result = price_override(
        override(
            per_request=10,
            pricing_rules=[rule("request", 0, scope={"path": "/free"})],
        ),
        NormalizedUsage(),
        PricingContext(path="/free"),
    )

    assert result.applied
    assert result.source == "rules"
    assert result.amount == Decimal("0")


@pytest.mark.parametrize("stored_rules", [{}, "", {"metric": "request"}])
def test_malformed_stored_rules_do_not_fall_back_to_legacy(stored_rules):
    with pytest.raises(PricingConfigurationError):
        price_override(
            override(per_request=10, pricing_rules=stored_rules),
            NormalizedUsage(),
        )


def test_usage_dimensions_are_used_when_context_does_not_override_them():
    usage = NormalizedUsage(dimensions={"mode": "batch"})
    context = PricingContext(cord="generate")
    rules = [rule("request", 1, cord="generate", conditions={"mode": "batch"})]

    assert price_usage(usage, rules, context).applied


def test_context_dimensions_merge_with_usage_dimensions_and_win_on_conflicts():
    usage = NormalizedUsage(
        dimensions={"mode": "batch", "output": {"resolution": "low", "format": "wide"}}
    )
    context = PricingContext(
        dimensions={"output": {"resolution": "high"}, "tier": "priority"}
    )
    rules = [
        rule(
            "request",
            1,
            conditions={
                "mode": "batch",
                "output.resolution": "high",
                "output.format": "wide",
                "tier": "priority",
            },
        )
    ]

    assert price_usage(usage, rules, context).applied


def test_pricing_result_serialization_includes_exact_line_item_audit_data():
    result = price_usage(
        NormalizedUsage(tokens={"input": 250_000}),
        [rule("token", "1.5", bucket="input", unit_size=1_000_000, id="input-rate")],
    )

    persisted = result.to_dict()
    public = result.to_dict(decimal_as_string=False)

    assert json.loads(json.dumps(persisted)) == persisted
    assert persisted == {
        "amount": "0.375",
        "applied": True,
        "source": "rules",
        "matched_rule_count": 1,
        "complete": True,
        "missing_rule_count": 0,
        "line_items": [
            {
                "rule_id": "input-rate",
                "metric": "token",
                "bucket": "input",
                "quantity": "250000",
                "billable_units": "0.25",
                "unit_price": "1.5",
                "amount": "0.375",
            }
        ],
    }
    assert public["amount"] == 0.375
