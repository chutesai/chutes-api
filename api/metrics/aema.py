"""
Helper for redis-based adaptive exponential moving average tracking.
Uses plain HGETALL/HSET instead of lua scripts to avoid evalsha timeouts.
"""

import time
from api.config import settings


class AdaptiveEMA:
    def __init__(
        self,
        key_prefix: str,
        target_window_seconds=600,
        min_alpha=0.0001,
        max_alpha=0.1,
        min_count=25,
    ):
        self.key_prefix = key_prefix
        self.target_window = target_window_seconds
        self.min_alpha = min_alpha
        self.max_alpha = max_alpha
        self.min_count = min_count

    async def update(self, key: str, new_value: float):
        """
        Update the adaptive EMA metrics using plain HGETALL + HSET.
        """
        client = settings.cm_redis_client
        redis_key = f"{self.key_prefix}:{key}"
        current_time = time.time()

        data = await client.hgetall(redis_key)

        if not data or b"ema" not in data:
            new_ema = new_value
            new_count = 1
            new_rate = 1.0
            alpha = self.max_alpha
        else:
            current_ema = float(data[b"ema"])
            current_count = int(data[b"count"])
            new_count = current_count + 1

            if b"last_time" in data:
                time_diff = current_time - float(data[b"last_time"])
                if time_diff > 0:
                    instant_rate = 1.0 / time_diff
                    if b"recent_rate" in data:
                        new_rate = 0.1 * instant_rate + 0.9 * float(data[b"recent_rate"])
                    else:
                        new_rate = instant_rate
                else:
                    new_rate = float(data[b"recent_rate"]) if b"recent_rate" in data else 1.0
            else:
                new_rate = 1.0

            target_samples = new_rate * self.target_window
            alpha = 1.0 / target_samples
            alpha = max(self.min_alpha, min(self.max_alpha, alpha))
            new_ema = (alpha * new_value) + ((1 - alpha) * current_ema)

        await client.hset(
            redis_key,
            mapping={
                "ema": new_ema,
                "count": new_count,
                "last_time": current_time,
                "recent_rate": new_rate,
                "last_alpha": alpha,
            },
        )

        if new_count >= self.min_count:
            return (new_ema, alpha, new_rate)
        else:
            return (None, alpha, new_rate)

    async def get_info(self, key: str):
        """
        Get current state information.
        """
        client = settings.cm_redis_client
        data = await client.hgetall(f"{self.key_prefix}:{key}")
        if not data:
            return None
        return {
            "ema": float(data[b"ema"]) if b"ema" in data else None,
            "count": int(data[b"count"]) if b"count" in data else 0,
            "recent_rate": float(data[b"recent_rate"]) if b"recent_rate" in data else 0,
            "last_alpha": float(data[b"last_alpha"]) if b"last_alpha" in data else None,
            "ready": int(data[b"count"]) >= self.min_count if b"count" in data else False,
        }
