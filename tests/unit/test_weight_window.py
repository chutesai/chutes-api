"""Unit tests for _seconds_until_next_weight_window."""

import pytest
from datetime import datetime, timezone
from unittest.mock import patch
from metasync.set_weights_on_metagraph import _seconds_until_next_weight_window


def _at(hour: int, minute: int, second: int = 0, microsecond: int = 0) -> datetime:
    return datetime(2026, 4, 13, hour, minute, second, microsecond, tzinfo=timezone.utc)


def _sleep(now: datetime) -> float:
    with patch("metasync.set_weights_on_metagraph.datetime") as mock_dt:
        mock_dt.now.return_value = now
        return _seconds_until_next_weight_window()


def test_mid_hour_returns_remaining_seconds():
    # 12:30:00 → next boundary is 13:00:00 → 1800s
    assert _sleep(_at(12, 30, 0)) == pytest.approx(1800.0)


def test_one_second_past_hour():
    # 12:00:01 → next boundary is 13:00:00 → 3599s
    assert _sleep(_at(12, 0, 1)) == pytest.approx(3599.0)


def test_one_second_before_hour():
    # 12:59:59 → next boundary is 13:00:00 → 1s
    assert _sleep(_at(12, 59, 59)) == pytest.approx(1.0)


def test_exactly_on_hour_boundary_returns_full_hour():
    # 12:00:00 exactly → next boundary is 13:00:00 → 3600s (not 0)
    assert _sleep(_at(12, 0, 0)) == pytest.approx(3600.0)


def test_microseconds_are_included():
    # 12:00:00.500000 → next boundary is 13:00:00 → 3599.5s
    assert _sleep(_at(12, 0, 0, microsecond=500_000)) == pytest.approx(3599.5, rel=1e-6)


def test_midnight_rollover():
    # 23:45:00 → next boundary is 00:00:00 next day → 900s
    assert _sleep(_at(23, 45, 0)) == pytest.approx(900.0)


def test_just_before_midnight():
    # 23:59:59 → next boundary is 00:00:00 → 1s
    assert _sleep(_at(23, 59, 59)) == pytest.approx(1.0)


def test_result_is_always_positive():
    for minute in range(0, 60, 7):
        for second in range(0, 60, 13):
            assert _sleep(_at(9, minute, second)) > 0


def test_result_never_exceeds_one_hour():
    for minute in range(0, 60, 7):
        for second in range(0, 60, 13):
            assert _sleep(_at(9, minute, second)) <= 3600.0
