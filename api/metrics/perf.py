"""
Track LLM usage metrics in prometheus (in addition to DB).
"""

from loguru import logger
from typing import Optional
from functools import lru_cache
from api.metrics.aema import AdaptiveEMA
from api.util import memcache_get, memcache_set


@lru_cache()
def ptps_tracker():
    return AdaptiveEMA(key_prefix="ptps")


@lru_cache()
def otps_tracker():
    return AdaptiveEMA(key_prefix="otps")


class PerfTracker:
    """
    Keep a rolling moving average of seconds per token (for LLMs)
    or seconds per step (for diffusion models) for compute unit normalization
    so the compute units are immutable/not affected by performance changes.
    """

    def __init__(self, window_size: int = 10000, ttl_days: int = 7):
        self.window_size = window_size
        self.alpha = 2 / (window_size + 1)
        self.ttl_seconds = ttl_days * 24 * 3600

    def _keys(self, chute_id: str, metric: str) -> tuple:
        base = f"_mva:{chute_id}:{metric}"
        return (f"{base}:v", f"{base}:c", f"{base}:l")

    async def update_average(self, value: float, chute_id: str, metric: str) -> float:
        v_key, c_key, l_key = self._keys(chute_id, metric)
        try:
            old_avg_bytes = await memcache_get(v_key.encode())
            count_bytes = await memcache_get(c_key.encode())
            old_avg = float(old_avg_bytes) if old_avg_bytes else None
            count = int(count_bytes) if count_bytes else 0
            if old_avg is not None:
                new_avg = self.alpha * value + (1 - self.alpha) * old_avg
                new_count = min(count + 1, self.window_size)
            else:
                new_avg = value
                new_count = 1
            await memcache_set(v_key.encode(), str(new_avg).encode(), exptime=self.ttl_seconds)
            await memcache_set(c_key.encode(), str(new_count).encode(), exptime=self.ttl_seconds)
            await memcache_set(l_key.encode(), str(value).encode(), exptime=self.ttl_seconds)
            return new_avg
        except Exception as e:
            logger.debug(f"Memcache error: {e}")
            return value

    async def update_invocation_metrics(
        self,
        chute_id: str,
        duration: float,
        metrics: dict,
        public: bool = True,
    ) -> dict[str, float]:
        if duration <= 0:
            return {}
        updates = {}
        if not public:
            updates["p"] = True
        steps = metrics.get("steps")
        if steps and steps > 0:
            seconds_per_step = duration / steps
            avg_sps = await self.update_average(seconds_per_step, chute_id, "sps")
            updates["masps"] = round(avg_sps, 8)
        it = metrics.get("it", 0)
        ot = metrics.get("ot", 0)
        total_tokens = it + ot
        if total_tokens > 0:
            seconds_per_token = duration / total_tokens
            avg_spt = await self.update_average(seconds_per_token, chute_id, "spt")
            updates["maspt"] = round(avg_spt, 8)

            # Update prompt processing tokens per second and output tokens per second,
            # which we can only actually track for streamed requests, which means it
            # has a non-null TTFT.
            if metrics.get("ttft"):
                # Prompt tokens.
                pema = None
                oema = None
                try:
                    ptps = metrics["it"] / metrics["ttft"]
                    pema, _, _ = await ptps_tracker().update(chute_id, ptps)
                except Exception:
                    logger.warning(
                        "Failed to update adaptive EMA for prompt processing TPS: {str(exc)}"
                    )

                # Completion tokens.
                if ot:
                    try:
                        otps = metrics["ot"] / (duration - metrics["ttft"])
                        oema, _, _ = await otps_tracker().update(chute_id, otps)
                    except Exception:
                        logger.warning(
                            "Failed to update adaptive EMA for completion TPS: {str(exc)}"
                        )
                if pema and oema:
                    updates.update(
                        {
                            "ptps": pema,
                            "otps": oema,
                        }
                    )
            else:
                # Get the adaptive EMA values already stored (non-streamed responses).
                try:
                    prompt_stats = await ptps_tracker().get_info(chute_id)
                    output_stats = await otps_tracker().get_info(chute_id)
                    if (
                        prompt_stats
                        and output_stats
                        and prompt_stats["ready"]
                        and output_stats["ready"]
                    ):
                        updates.update(
                            {
                                "ptps": prompt_stats["ema"],
                                "otps": output_stats["ema"],
                            }
                        )
                except Exception as exc:
                    logger.warning(f"Failed to fetch adaptive EMA values: {str(exc)}")

            # Calculate the normalized compute for this item.
            if "ptps" in updates and "otps" in updates:
                normalized_input_time = it / updates["ptps"] if updates["ptps"] > 0 else 0
                normalized_output_time = ot / updates["otps"] if updates["otps"] > 0 else 0
                updates["nc"] = normalized_input_time + normalized_output_time
        return updates

    async def get_current(self, chute_id: str) -> dict[str, Optional[dict]]:
        result = {}
        for metric in ["sps", "spt"]:
            v_key, c_key, l_key = self._keys(chute_id, metric)
            try:
                v_bytes = await memcache_get(v_key.encode())
                c_bytes = await memcache_get(c_key.encode())
                l_bytes = await memcache_get(l_key.encode())
                if v_bytes:
                    result[metric] = {
                        "v": float(v_bytes),
                        "c": int(c_bytes) if c_bytes else 0,
                        "l": float(l_bytes) if l_bytes else None,
                    }
            except Exception:
                pass
        return result


PERF_TRACKER = PerfTracker()
