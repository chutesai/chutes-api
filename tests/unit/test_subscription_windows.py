from datetime import datetime, timezone

from api.invocation.util import (
    build_subscription_periods,
    get_fixed_four_hour_bucket_start,
    get_subscription_cycle_end,
    get_subscription_cycle_start,
)


def test_fixed_four_hour_bucket_is_not_rolling() -> None:
    now = datetime(2026, 3, 9, 7, 59, 59, tzinfo=timezone.utc)

    assert get_fixed_four_hour_bucket_start(now) == datetime(
        2026, 3, 9, 4, 0, 0, tzinfo=timezone.utc
    )


def test_subscription_cycle_start_uses_updated_at_anchor() -> None:
    updated_at = datetime(2026, 7, 7, 18, 47, 43, tzinfo=timezone.utc)
    now = datetime(2026, 8, 10, 12, 0, 0, tzinfo=timezone.utc)

    assert get_subscription_cycle_start(updated_at, now) == datetime(
        2026, 7, 7, 18, 47, 43, tzinfo=timezone.utc
    )


def test_subscription_period_keys_use_fixed_boundaries() -> None:
    updated_at = datetime(2026, 7, 7, 18, 47, 43, tzinfo=timezone.utc)
    now = datetime(2026, 8, 10, 12, 34, 56, tzinfo=timezone.utc)

    periods = build_subscription_periods(updated_at, now)

    assert periods["cycle_start"] == datetime(2026, 7, 7, 18, 47, 43, tzinfo=timezone.utc)
    assert periods["cycle_end"] == datetime(2026, 8, 7, 18, 47, 43, tzinfo=timezone.utc)
    assert periods["four_hour_start"] == datetime(2026, 8, 10, 12, 0, 0, tzinfo=timezone.utc)
    assert periods["monthly_period"] == f"m2:{int(periods['cycle_start'].timestamp())}"
    assert periods["four_hour_period"] == f"4h2:{int(periods['four_hour_start'].timestamp())}"


def test_subscription_cycle_caps_day_to_target_month_length() -> None:
    updated_at = datetime(2026, 1, 31, 18, 47, 43, tzinfo=timezone.utc)
    now = datetime(2026, 2, 28, 19, 0, 0, tzinfo=timezone.utc)

    assert get_subscription_cycle_start(updated_at, now) == datetime(
        2026, 1, 31, 18, 47, 43, tzinfo=timezone.utc
    )
    assert get_subscription_cycle_end(updated_at, now) == datetime(
        2026, 2, 28, 18, 47, 43, tzinfo=timezone.utc
    )


def test_subscription_periods_use_effective_anchor_date() -> None:
    effective_date = datetime(2026, 3, 15, 9, 30, 0, tzinfo=timezone.utc)
    now = datetime(2026, 4, 20, 12, 0, 0, tzinfo=timezone.utc)

    periods = build_subscription_periods(effective_date, now)

    assert periods["cycle_start"] == datetime(2026, 3, 15, 9, 30, 0, tzinfo=timezone.utc)
    assert periods["cycle_end"] == datetime(2026, 4, 15, 9, 30, 0, tzinfo=timezone.utc)


def test_subscription_anchor_is_clamped_to_march_1_2026() -> None:
    effective_date = datetime(2026, 2, 15, 9, 30, 0, tzinfo=timezone.utc)
    now = datetime(2026, 3, 9, 12, 0, 0, tzinfo=timezone.utc)

    periods = build_subscription_periods(effective_date, now)

    # Usage counting is floored to March 1.
    assert periods["anchor_date"] == datetime(2026, 3, 1, 0, 0, 0, tzinfo=timezone.utc)
    assert periods["cycle_start"] == datetime(2026, 3, 1, 0, 0, 0, tzinfo=timezone.utc)
    # Renewal/reset is based on the raw anchor, not the floor.
    assert periods["cycle_end"] == datetime(2026, 3, 15, 9, 30, 0, tzinfo=timezone.utc)


def test_cycle_end_uses_raw_anchor_not_floor() -> None:
    """
    A user whose cycle started before March 1 should see their actual renewal
    date, not one shifted by the usage floor.
    """
    updated_at = datetime(2026, 2, 24, 18, 47, 43, tzinfo=timezone.utc)
    now = datetime(2026, 3, 9, 12, 0, 0, tzinfo=timezone.utc)

    periods = build_subscription_periods(updated_at, now)

    # Usage starts from the floor.
    assert periods["cycle_start"] == datetime(2026, 3, 1, 0, 0, 0, tzinfo=timezone.utc)
    # Renewal is based on the raw anchor date.
    assert periods["cycle_end"] == datetime(2026, 3, 24, 18, 47, 43, tzinfo=timezone.utc)


def test_subscription_cycle_crosses_year_boundary() -> None:
    updated_at = datetime(2026, 12, 31, 23, 59, 59, tzinfo=timezone.utc)
    now = datetime(2027, 1, 15, 12, 0, 0, tzinfo=timezone.utc)

    periods = build_subscription_periods(updated_at, now)

    assert periods["cycle_start"] == datetime(2026, 12, 31, 23, 59, 59, tzinfo=timezone.utc)
    assert periods["cycle_end"] == datetime(2027, 1, 31, 23, 59, 59, tzinfo=timezone.utc)


def test_subscription_cycle_handles_leap_day_anchor() -> None:
    updated_at = datetime(2028, 2, 29, 6, 15, 0, tzinfo=timezone.utc)
    now = datetime(2028, 3, 10, 12, 0, 0, tzinfo=timezone.utc)

    periods = build_subscription_periods(updated_at, now)

    assert periods["cycle_start"] == datetime(2028, 2, 29, 6, 15, 0, tzinfo=timezone.utc)
    assert periods["cycle_end"] == datetime(2028, 3, 29, 6, 15, 0, tzinfo=timezone.utc)


def test_subscription_cycle_end_caps_from_leap_year_to_non_leap_year() -> None:
    updated_at = datetime(2028, 1, 31, 6, 15, 0, tzinfo=timezone.utc)
    now = datetime(2028, 2, 20, 12, 0, 0, tzinfo=timezone.utc)

    assert get_subscription_cycle_end(updated_at, now) == datetime(
        2028, 2, 29, 6, 15, 0, tzinfo=timezone.utc
    )


def test_fixed_four_hour_bucket_across_midnight_utc() -> None:
    now = datetime(2026, 3, 10, 0, 0, 1, tzinfo=timezone.utc)

    periods = build_subscription_periods(datetime(2026, 3, 9, 18, 0, 0, tzinfo=timezone.utc), now)

    assert periods["four_hour_start"] == datetime(2026, 3, 10, 0, 0, 0, tzinfo=timezone.utc)
    assert periods["four_hour_end"] == datetime(2026, 3, 10, 4, 0, 0, tzinfo=timezone.utc)
