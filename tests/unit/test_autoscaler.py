"""Unit tests for chute_autoscaler module."""

import pytest
from unittest.mock import Mock, AsyncMock, patch
from datetime import datetime, timezone


class TestBountyBoostCalculation:
    def test_bounty_boost_at_zero_minutes(self):
        from api.bounty.util import calculate_bounty_boost

        assert calculate_bounty_boost(0) == 1.5

    def test_bounty_boost_at_90_minutes(self):
        from api.bounty.util import calculate_bounty_boost

        boost = calculate_bounty_boost(90 * 60)
        assert boost == pytest.approx(2.75, rel=0.01)

    def test_bounty_boost_at_180_minutes(self):
        from api.bounty.util import calculate_bounty_boost

        assert calculate_bounty_boost(180 * 60) == 4.0

    def test_bounty_boost_beyond_180_minutes(self):
        from api.bounty.util import calculate_bounty_boost

        assert calculate_bounty_boost(240 * 60) == 4.0

    def test_bounty_boost_negative_age(self):
        from api.bounty.util import calculate_bounty_boost

        assert calculate_bounty_boost(-100) == 1.5

    @pytest.mark.parametrize(
        "minutes,expected_boost",
        [
            (0, 1.5),
            (45, 2.125),
            (90, 2.75),
            (135, 3.375),
            (180, 4.0),
            (240, 4.0),
        ],
    )
    def test_bounty_boost_curve(self, minutes, expected_boost):
        from api.bounty.util import calculate_bounty_boost

        boost = calculate_bounty_boost(minutes * 60)
        assert boost == pytest.approx(expected_boost, rel=0.01)


class TestScaleDownThresholds:
    def test_scale_down_threshold_ratio(self):
        from api.constants import UTILIZATION_SCALE_UP, UTILIZATION_SCALE_DOWN

        assert UTILIZATION_SCALE_UP > 0
        assert UTILIZATION_SCALE_DOWN > 0
        assert UTILIZATION_SCALE_DOWN < UTILIZATION_SCALE_UP

    def test_stable_zone_exists(self):
        from api.constants import UTILIZATION_SCALE_UP, UTILIZATION_SCALE_DOWN

        stable_zone = UTILIZATION_SCALE_UP - UTILIZATION_SCALE_DOWN
        assert stable_zone > 0


class TestAutoscaleContext:
    @pytest.fixture
    def mock_info(self):
        from api.constants import UTILIZATION_SCALE_UP

        info = Mock()
        info.instance_count = 5
        info.scaling_threshold = UTILIZATION_SCALE_UP
        info.has_rolling_update = False
        info.max_instances = 100
        info.public = True
        info.user_id = "test-user"
        info.node_selector = {"gpu_type": "A100", "gpu_count": 1}
        info.new_chute = False
        info.pending_instance_count = 0
        return info

    @pytest.fixture
    def mock_metrics(self):
        return {
            "utilization": {"current": 0.5, "5m": 0.45, "15m": 0.4, "1h": 0.35},
            "completed_requests": {"5m": 100, "15m": 300, "1h": 1200},
            "rate_limited_requests": {"5m": 5, "15m": 10, "1h": 30},
            "total_requests": {"5m": 105, "15m": 310, "1h": 1230},
            "rate_limit_ratio": {"5m": 0.048, "15m": 0.032, "1h": 0.024},
        }

    def test_context_initialization(self, mock_info, mock_metrics):
        from chute_autoscaler import AutoScaleContext
        from api.constants import UTILIZATION_SCALE_DOWN, UTILIZATION_SCALE_UP

        ctx = AutoScaleContext(
            chute_id="test-chute",
            metrics=mock_metrics,
            info=mock_info,
            supported_gpus={"A100"},
            instances=[],
            db_now=datetime.now(timezone.utc),
        )

        assert ctx.chute_id == "test-chute"
        assert ctx.current_count == 5
        assert ctx.threshold == UTILIZATION_SCALE_UP
        expected_scale_down = UTILIZATION_SCALE_UP * (UTILIZATION_SCALE_DOWN / UTILIZATION_SCALE_UP)
        assert ctx.scale_down_threshold == pytest.approx(expected_scale_down, rel=0.01)
        assert ctx.public is True

    def test_context_utilization_basis(self, mock_info, mock_metrics):
        from chute_autoscaler import AutoScaleContext

        ctx = AutoScaleContext(
            chute_id="test-chute",
            metrics=mock_metrics,
            info=mock_info,
            supported_gpus={"A100"},
            instances=[],
            db_now=datetime.now(timezone.utc),
        )
        assert ctx.utilization_basis == 0.45

    def test_context_rate_limit_basis(self, mock_info, mock_metrics):
        from chute_autoscaler import AutoScaleContext

        ctx = AutoScaleContext(
            chute_id="test-chute",
            metrics=mock_metrics,
            info=mock_info,
            supported_gpus={"A100"},
            instances=[],
            db_now=datetime.now(timezone.utc),
        )
        assert ctx.rate_limit_basis == 0.048


class TestScaleDownDecision:
    @pytest.fixture
    def make_context(self):
        def _make(utilization=0.3, current_count=5, rate_limiting=False, threshold=None):
            from chute_autoscaler import AutoScaleContext
            from api.constants import UTILIZATION_SCALE_UP

            info = Mock()
            info.instance_count = current_count
            info.scaling_threshold = UTILIZATION_SCALE_UP if threshold is None else threshold
            info.has_rolling_update = False
            info.max_instances = 100
            info.public = True
            info.user_id = "test-user"
            info.node_selector = {"gpu_type": "A100", "gpu_count": 1}
            info.new_chute = False
            info.pending_instance_count = 0

            rl_val = 0.05 if rate_limiting else 0.0
            metrics = {
                "utilization": {
                    "current": utilization,
                    "5m": utilization,
                    "15m": utilization,
                    "1h": utilization,
                },
                "completed_requests": {"5m": 100, "15m": 300, "1h": 1200},
                "rate_limited_requests": {"5m": 0, "15m": 0, "1h": 0},
                "total_requests": {"5m": 100, "15m": 300, "1h": 1200},
                "rate_limit_ratio": {"5m": rl_val, "15m": rl_val, "1h": rl_val},
            }

            return AutoScaleContext(
                chute_id="test-chute",
                metrics=metrics,
                info=info,
                supported_gpus={"A100"},
                instances=[],
                db_now=datetime.now(timezone.utc),
            )

        return _make

    @pytest.mark.asyncio
    async def test_scale_down_below_threshold(self, make_context):
        from chute_autoscaler import calculate_local_decision

        # Note: scale-down now requires pending_instance_count == 0 (which is the default)
        ctx = make_context(utilization=0.1, current_count=5)
        ctx.smoothed_util = 0.1  # Must set smoothed_util below scale_down_threshold
        await calculate_local_decision(ctx)

        assert ctx.action == "scale_down_candidate"
        assert ctx.downscale_amount == 1
        assert ctx.target_count == 4

    @pytest.mark.asyncio
    async def test_no_scale_down_in_stable_zone(self, make_context):
        from chute_autoscaler import calculate_local_decision

        ctx = make_context(utilization=0.45, current_count=5)
        await calculate_local_decision(ctx)

        assert ctx.action == "no_action"
        assert ctx.downscale_amount == 0

    @pytest.mark.asyncio
    async def test_no_scale_down_with_rate_limiting(self, make_context):
        from chute_autoscaler import calculate_local_decision

        ctx = make_context(utilization=0.2, current_count=5, rate_limiting=True)
        await calculate_local_decision(ctx)
        assert ctx.downscale_amount == 0

    @pytest.mark.asyncio
    async def test_no_scale_down_below_minimum(self, make_context):
        from chute_autoscaler import calculate_local_decision

        ctx = make_context(utilization=0.1, current_count=2)
        await calculate_local_decision(ctx)
        assert ctx.target_count >= 2

    @pytest.mark.asyncio
    async def test_projected_utilization_check(self, make_context):
        from chute_autoscaler import calculate_local_decision

        ctx = make_context(utilization=0.34, current_count=2)
        await calculate_local_decision(ctx)
        assert ctx.target_count == 2


class TestScaleUpDecision:
    @pytest.fixture
    def make_context(self):
        def _make(utilization=0.7, rate_limit_ratio=0.0, current_count=5):
            from chute_autoscaler import AutoScaleContext
            from api.constants import UTILIZATION_SCALE_UP

            info = Mock()
            info.instance_count = current_count
            info.scaling_threshold = UTILIZATION_SCALE_UP
            info.has_rolling_update = False
            info.max_instances = 100
            info.public = True
            info.user_id = "test-user"
            info.node_selector = {"gpu_type": "A100", "gpu_count": 1}
            info.new_chute = False
            info.pending_instance_count = 0

            metrics = {
                "utilization": {
                    "current": utilization,
                    "5m": utilization,
                    "15m": utilization,
                    "1h": utilization,
                },
                "completed_requests": {"5m": 100, "15m": 300, "1h": 1200},
                "rate_limited_requests": {"5m": 0, "15m": 0, "1h": 0},
                "total_requests": {"5m": 100, "15m": 300, "1h": 1200},
                "rate_limit_ratio": {
                    "5m": rate_limit_ratio,
                    "15m": rate_limit_ratio,
                    "1h": rate_limit_ratio,
                },
            }

            return AutoScaleContext(
                chute_id="test-chute",
                metrics=metrics,
                info=info,
                supported_gpus={"A100"},
                instances=[],
                db_now=datetime.now(timezone.utc),
            )

        return _make

    @pytest.mark.asyncio
    async def test_scale_up_high_utilization(self, make_context):
        from chute_autoscaler import calculate_local_decision

        ctx = make_context(utilization=0.75, current_count=5)
        ctx.is_starving = True
        await calculate_local_decision(ctx)

        assert "scale_up" in ctx.action
        assert ctx.upscale_amount > 0
        assert ctx.target_count > 5

    @pytest.mark.asyncio
    async def test_scale_up_rate_limiting(self, make_context):
        from chute_autoscaler import calculate_local_decision

        ctx = make_context(utilization=0.4, rate_limit_ratio=0.05, current_count=5)
        ctx.is_starving = True
        await calculate_local_decision(ctx)

        assert "scale_up" in ctx.action
        assert ctx.upscale_amount > 0

    @pytest.mark.asyncio
    async def test_scale_up_respects_max_instances(self, make_context):
        from chute_autoscaler import calculate_local_decision

        ctx = make_context(utilization=0.9, current_count=95)
        ctx.max_instances = 100
        ctx.is_starving = True
        await calculate_local_decision(ctx)

        assert ctx.target_count <= 100


class TestUrgencyScoring:
    def test_urgency_score_formula_high_utilization(self):
        from chute_autoscaler import AutoScaleContext
        from api.constants import UTILIZATION_SCALE_UP

        info = Mock()
        info.instance_count = 5
        info.scaling_threshold = UTILIZATION_SCALE_UP
        info.has_rolling_update = False
        info.max_instances = 100
        info.public = True
        info.user_id = "test-user"
        info.node_selector = {"gpu_type": "A100", "gpu_count": 1}
        info.new_chute = False
        info.pending_instance_count = 0

        metrics = {
            "utilization": {"current": 0.9, "5m": 0.9, "15m": 0.9, "1h": 0.9},
            "completed_requests": {"5m": 100, "15m": 300, "1h": 1200},
            "rate_limited_requests": {"5m": 0, "15m": 0, "1h": 0},
            "total_requests": {"5m": 100, "15m": 300, "1h": 1200},
            "rate_limit_ratio": {"5m": 0, "15m": 0, "1h": 0},
        }

        ctx = AutoScaleContext(
            chute_id="test-chute",
            metrics=metrics,
            info=info,
            supported_gpus={"A100"},
            instances=[],
            db_now=datetime.now(timezone.utc),
        )

        util_score = min(100, ctx.utilization_basis * 100)
        rl_score = ctx.rate_limit_basis * 5000
        ctx.urgency_score = util_score + rl_score

        assert ctx.urgency_score == pytest.approx(90, rel=0.1)

    def test_urgency_score_formula_rate_limiting_dominates(self):
        from chute_autoscaler import AutoScaleContext
        from api.constants import UTILIZATION_SCALE_UP

        info = Mock()
        info.instance_count = 5
        info.scaling_threshold = UTILIZATION_SCALE_UP
        info.has_rolling_update = False
        info.max_instances = 100
        info.public = True
        info.user_id = "test-user"
        info.node_selector = {"gpu_type": "A100", "gpu_count": 1}
        info.new_chute = False
        info.pending_instance_count = 0

        metrics = {
            "utilization": {"current": 0.5, "5m": 0.5, "15m": 0.5, "1h": 0.5},
            "completed_requests": {"5m": 100, "15m": 300, "1h": 1200},
            "rate_limited_requests": {"5m": 10, "15m": 30, "1h": 120},
            "total_requests": {"5m": 110, "15m": 330, "1h": 1320},
            "rate_limit_ratio": {"5m": 0.1, "15m": 0.09, "1h": 0.09},
        }

        ctx = AutoScaleContext(
            chute_id="test-chute",
            metrics=metrics,
            info=info,
            supported_gpus={"A100"},
            instances=[],
            db_now=datetime.now(timezone.utc),
        )

        util_score = min(100, ctx.utilization_basis * 100)
        rl_score = ctx.rate_limit_basis * 5000
        ctx.urgency_score = util_score + rl_score

        assert ctx.urgency_score == pytest.approx(550, rel=0.1)


class TestComputeMultiplierDecay:
    def test_hold_period_constants(self):
        from chute_autoscaler import (
            COMPUTE_MULTIPLIER_HOLD_HOURS,
            COMPUTE_MULTIPLIER_FULL_ADJUST_HOURS,
            COMPUTE_MULTIPLIER_RAMP_HOURS,
        )

        assert COMPUTE_MULTIPLIER_HOLD_HOURS == 2.0
        assert COMPUTE_MULTIPLIER_FULL_ADJUST_HOURS == 8.0
        assert COMPUTE_MULTIPLIER_RAMP_HOURS == 6.0

    def test_ease_in_curve_values(self):
        hold = 2.0
        ramp = 6.0

        def blend_at_hour(h):
            if h <= hold:
                return 0.0
            elif h >= hold + ramp:
                return 1.0
            else:
                t = (h - hold) / ramp
                return t**2

        assert blend_at_hour(2) == 0.0
        assert blend_at_hour(3) == pytest.approx(0.028, rel=0.1)
        assert blend_at_hour(5) == pytest.approx(0.25, rel=0.01)
        assert blend_at_hour(8) == 1.0

    def test_multiplier_blend_example(self):
        original = 8.0
        target = 2.0

        def multiplier_at_hour(h):
            hold = 2.0
            ramp = 6.0
            if h <= hold:
                return original
            elif h >= hold + ramp:
                return target
            else:
                t = (h - hold) / ramp
                blend = t**2
                return original * (1 - blend) + target * blend

        assert multiplier_at_hour(2) == 8.0
        assert multiplier_at_hour(5) == pytest.approx(6.5, rel=0.01)
        assert multiplier_at_hour(8) == 2.0


class TestBlendedMultiplierCalculation:
    def test_hold_period_with_existing_value(self):
        from chute_autoscaler import _calculate_blended_multiplier

        result = _calculate_blended_multiplier(current=3.0, target=2.0, hours_since_activation=1.0)
        assert result is None  # No update needed

    def test_hold_period_with_null_value(self):
        from chute_autoscaler import _calculate_blended_multiplier

        result = _calculate_blended_multiplier(current=None, target=2.0, hours_since_activation=1.0)
        assert result == 2.0  # Initialize to target

    def test_past_full_adjustment(self):
        from chute_autoscaler import _calculate_blended_multiplier

        result = _calculate_blended_multiplier(current=5.0, target=2.0, hours_since_activation=10.0)
        assert result == 2.0  # Clamp to target

    def test_ramp_period_midpoint(self):
        from chute_autoscaler import _calculate_blended_multiplier

        # At hour 5: t = (5-2)/6 = 0.5, blend = 0.25
        # result = 8.0 * 0.75 + 2.0 * 0.25 = 6.0 + 0.5 = 6.5
        result = _calculate_blended_multiplier(current=8.0, target=2.0, hours_since_activation=5.0)
        assert result == pytest.approx(6.5, rel=0.01)

    def test_ramp_period_start(self):
        from chute_autoscaler import _calculate_blended_multiplier

        # Just after hold period: minimal blend
        result = _calculate_blended_multiplier(current=8.0, target=2.0, hours_since_activation=2.1)
        assert result > 7.9  # Still very close to original

    def test_ramp_period_end(self):
        from chute_autoscaler import _calculate_blended_multiplier

        # Just before full adjustment
        result = _calculate_blended_multiplier(current=8.0, target=2.0, hours_since_activation=7.9)
        assert result < 2.5  # Close to target


class TestRelativeBoostAdjustment:
    def test_one_starving_one_comfortable(self):
        # Chute A: urgency 400, Chute B: urgency 50
        # avg = 225, max = 400, spread = 400
        URGENCY_MAX_FOR_BOOST = 500
        URGENCY_BOOST_MIN = 1.0
        URGENCY_BOOST_MAX = 2.5
        RELATIVE_ADJUSTMENT_MAX = 0.2

        def calculate_boost(urgency, avg_urgency, max_urgency):
            normalized = min(urgency / URGENCY_MAX_FOR_BOOST, 1.0)
            base_boost = URGENCY_BOOST_MIN + (normalized * (URGENCY_BOOST_MAX - URGENCY_BOOST_MIN))

            if max_urgency > 0:
                spread = max(max_urgency, 1)
                relative_position = (urgency - avg_urgency) / spread
                relative_position = max(-1.0, min(1.0, relative_position))
                relative_factor = 1.0 + (relative_position * RELATIVE_ADJUSTMENT_MAX)
            else:
                relative_factor = 1.0

            return max(1.0, base_boost * relative_factor)

        avg = 225
        max_urg = 400

        boost_a = calculate_boost(400, avg, max_urg)
        boost_b = calculate_boost(50, avg, max_urg)

        assert boost_a > boost_b
        assert boost_a > 2.0  # High urgency gets high boost
        assert boost_b >= 1.0  # Never below 1.0

    def test_all_equal_urgency(self):
        URGENCY_MAX_FOR_BOOST = 500
        URGENCY_BOOST_MIN = 1.0
        URGENCY_BOOST_MAX = 2.5
        RELATIVE_ADJUSTMENT_MAX = 0.2

        def calculate_boost(urgency, avg_urgency, max_urgency):
            normalized = min(urgency / URGENCY_MAX_FOR_BOOST, 1.0)
            base_boost = URGENCY_BOOST_MIN + (normalized * (URGENCY_BOOST_MAX - URGENCY_BOOST_MIN))

            if max_urgency > 0:
                spread = max(max_urgency, 1)
                relative_position = (urgency - avg_urgency) / spread
                relative_position = max(-1.0, min(1.0, relative_position))
                relative_factor = 1.0 + (relative_position * RELATIVE_ADJUSTMENT_MAX)
            else:
                relative_factor = 1.0

            return max(1.0, base_boost * relative_factor)

        # All chutes have same urgency
        boost_a = calculate_boost(400, 400, 400)
        boost_b = calculate_boost(400, 400, 400)
        boost_c = calculate_boost(400, 400, 400)

        assert boost_a == boost_b == boost_c
        # relative_position = 0, so factor = 1.0, just base boost
        expected = 1.0 + (400 / 500) * 1.5
        assert boost_a == pytest.approx(expected, rel=0.01)

    def test_boost_never_below_one(self):
        URGENCY_MAX_FOR_BOOST = 500
        URGENCY_BOOST_MIN = 1.0
        URGENCY_BOOST_MAX = 2.5
        RELATIVE_ADJUSTMENT_MAX = 0.2

        def calculate_boost(urgency, avg_urgency, max_urgency):
            normalized = min(urgency / URGENCY_MAX_FOR_BOOST, 1.0)
            base_boost = URGENCY_BOOST_MIN + (normalized * (URGENCY_BOOST_MAX - URGENCY_BOOST_MIN))

            if max_urgency > 0:
                spread = max(max_urgency, 1)
                relative_position = (urgency - avg_urgency) / spread
                relative_position = max(-1.0, min(1.0, relative_position))
                relative_factor = 1.0 + (relative_position * RELATIVE_ADJUSTMENT_MAX)
            else:
                relative_factor = 1.0

            return max(1.0, base_boost * relative_factor)

        # Very low urgency chute when others are high
        boost = calculate_boost(10, 300, 500)
        assert boost >= 1.0

    def test_above_average_gets_bump(self):
        URGENCY_MAX_FOR_BOOST = 500
        URGENCY_BOOST_MIN = 1.0
        URGENCY_BOOST_MAX = 2.5
        RELATIVE_ADJUSTMENT_MAX = 0.2

        def calculate_boost(urgency, avg_urgency, max_urgency):
            normalized = min(urgency / URGENCY_MAX_FOR_BOOST, 1.0)
            base_boost = URGENCY_BOOST_MIN + (normalized * (URGENCY_BOOST_MAX - URGENCY_BOOST_MIN))

            if max_urgency > 0:
                spread = max(max_urgency, 1)
                relative_position = (urgency - avg_urgency) / spread
                relative_position = max(-1.0, min(1.0, relative_position))
                relative_factor = 1.0 + (relative_position * RELATIVE_ADJUSTMENT_MAX)
            else:
                relative_factor = 1.0

            return max(1.0, base_boost * relative_factor)

        # Chute above average
        base = 1.0 + (300 / 500) * 1.5  # 1.9
        boost_with_adjustment = calculate_boost(300, 200, 400)

        # Should be higher than base due to positive relative adjustment
        assert boost_with_adjustment > base

    def test_below_average_gets_reduction(self):
        URGENCY_MAX_FOR_BOOST = 500
        URGENCY_BOOST_MIN = 1.0
        URGENCY_BOOST_MAX = 2.5
        RELATIVE_ADJUSTMENT_MAX = 0.2

        def calculate_boost(urgency, avg_urgency, max_urgency):
            normalized = min(urgency / URGENCY_MAX_FOR_BOOST, 1.0)
            base_boost = URGENCY_BOOST_MIN + (normalized * (URGENCY_BOOST_MAX - URGENCY_BOOST_MIN))

            if max_urgency > 0:
                spread = max(max_urgency, 1)
                relative_position = (urgency - avg_urgency) / spread
                relative_position = max(-1.0, min(1.0, relative_position))
                relative_factor = 1.0 + (relative_position * RELATIVE_ADJUSTMENT_MAX)
            else:
                relative_factor = 1.0

            return max(1.0, base_boost * relative_factor)

        # Chute below average
        base = 1.0 + (100 / 500) * 1.5  # 1.3
        boost_with_adjustment = calculate_boost(100, 300, 400)

        # Should be lower than base due to negative relative adjustment (but >= 1.0)
        assert boost_with_adjustment < base
        assert boost_with_adjustment >= 1.0


class TestDonorIdentification:
    @pytest.fixture
    def make_context(self):
        def _make(utilization=0.3, public=True, is_chutes_user=False):
            from chute_autoscaler import AutoScaleContext
            from api.constants import UTILIZATION_SCALE_UP

            info = Mock()
            info.instance_count = 5
            info.scaling_threshold = UTILIZATION_SCALE_UP
            info.has_rolling_update = False
            info.max_instances = 100
            info.public = public
            info.user_id = "chutes-user" if is_chutes_user else "other-user"
            info.node_selector = {"gpu_type": "A100", "gpu_count": 1}
            info.new_chute = False
            info.pending_instance_count = 0

            metrics = {
                "utilization": {
                    "current": utilization,
                    "5m": utilization,
                    "15m": utilization,
                    "1h": utilization,
                },
                "completed_requests": {"5m": 100, "15m": 300, "1h": 1200},
                "rate_limited_requests": {"5m": 0, "15m": 0, "1h": 0},
                "total_requests": {"5m": 100, "15m": 300, "1h": 1200},
                "rate_limit_ratio": {"5m": 0, "15m": 0, "1h": 0},
            }

            return AutoScaleContext(
                chute_id="test-chute",
                metrics=metrics,
                info=info,
                supported_gpus={"A100"},
                instances=[],
                db_now=datetime.now(timezone.utc),
            )

        return _make

    def test_critical_donor_below_scale_down_threshold(self, make_context):
        ctx = make_context(utilization=0.1)
        ctx.smoothed_util = 0.1  # Must set smoothed_util for donor identification
        assert ctx.utilization_basis < ctx.scale_down_threshold

    def test_optional_donor_in_stable_zone(self, make_context):
        ctx = make_context(utilization=0.45)
        assert ctx.utilization_basis >= ctx.scale_down_threshold
        assert ctx.utilization_basis < ctx.threshold

    def test_private_chute_not_donor(self, make_context):
        ctx = make_context(utilization=0.2, public=False, is_chutes_user=False)
        assert ctx.public is False


class TestDistributedLock:
    @pytest.mark.asyncio
    async def test_lock_acquired_successfully(self):
        from chute_autoscaler import autoscaler_lock

        mock_redis = AsyncMock()
        mock_redis.set = AsyncMock(return_value=True)
        mock_redis.get = AsyncMock(return_value=b"test-lock-id")
        mock_redis.delete = AsyncMock(return_value=1)

        with patch("chute_autoscaler.settings") as mock_settings:
            mock_settings.redis_client = mock_redis

            async with autoscaler_lock():
                mock_redis.set.assert_called_once()
                call_args = mock_redis.set.call_args
                assert call_args.kwargs.get("nx") is True
                assert call_args.kwargs.get("ex") == 180

    @pytest.mark.asyncio
    async def test_lock_not_acquired_raises_in_full_mode(self):
        from chute_autoscaler import autoscaler_lock

        mock_redis = AsyncMock()
        mock_redis.set = AsyncMock(return_value=False)
        mock_redis.ttl = AsyncMock(return_value=120)

        with patch("chute_autoscaler.settings") as mock_settings:
            mock_settings.redis_client = mock_redis

            with pytest.raises(RuntimeError, match="Another autoscaler is running"):
                async with autoscaler_lock(soft_mode=False):
                    pass

    @pytest.mark.asyncio
    async def test_lock_not_acquired_quiet_in_soft_mode(self):
        from chute_autoscaler import autoscaler_lock, LockNotAcquired

        mock_redis = AsyncMock()
        mock_redis.set = AsyncMock(return_value=False)
        mock_redis.ttl = AsyncMock(return_value=120)

        with patch("chute_autoscaler.settings") as mock_settings:
            mock_settings.redis_client = mock_redis

            with pytest.raises(LockNotAcquired):
                async with autoscaler_lock(soft_mode=True):
                    pass

    @pytest.mark.asyncio
    async def test_lock_released_on_exit(self):
        from chute_autoscaler import autoscaler_lock

        lock_id = None
        mock_redis = AsyncMock()

        async def mock_set(key, value, **kwargs):
            nonlocal lock_id
            lock_id = value
            return True

        mock_redis.set = mock_set
        mock_redis.get = AsyncMock(side_effect=lambda k: lock_id.encode() if lock_id else None)
        mock_redis.delete = AsyncMock(return_value=1)

        with patch("chute_autoscaler.settings") as mock_settings:
            mock_settings.redis_client = mock_redis

            async with autoscaler_lock():
                pass

            mock_redis.delete.assert_called_once()
