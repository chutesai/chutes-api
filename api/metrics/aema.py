"""
Helper for redis-based adaptive exponential moving average tracking.
"""

import time
import uuid
import redis
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
        self.script_sha = None

        self.script = """
            local key = KEYS[1]
            local new_value = tonumber(ARGV[1])
            local current_time = tonumber(ARGV[2])
            local target_window = tonumber(ARGV[3])
            local min_alpha = tonumber(ARGV[4])
            local max_alpha = tonumber(ARGV[5])
            local min_count = tonumber(ARGV[6])

            local data = redis.call('HMGET', key, 'ema', 'count', 'last_time', 'recent_rate')
            local current_ema = data[1]
            local current_count = data[2]
            local last_time = data[3]
            local recent_rate = data[4]

            local new_ema, new_count, new_rate, alpha

            if not current_ema then
                new_ema = new_value
                new_count = 1
                new_rate = 1
                alpha = max_alpha
            else
                current_count = tonumber(current_count)
                new_count = current_count + 1
                if last_time then
                    local time_diff = current_time - tonumber(last_time)
                    if time_diff > 0 then
                        local instant_rate = 1.0 / time_diff
                        if recent_rate then
                            new_rate = 0.1 * instant_rate + 0.9 * tonumber(recent_rate)
                        else
                            new_rate = instant_rate
                        end
                    else
                        new_rate = tonumber(recent_rate) or 1
                    end
                else
                    new_rate = 1
                end
                local target_samples = new_rate * target_window
                alpha = 1.0 / target_samples
                if alpha < min_alpha then
                    alpha = min_alpha
                elseif alpha > max_alpha then
                    alpha = max_alpha
                end
                new_ema = (alpha * new_value) + ((1 - alpha) * tonumber(current_ema))
            end
            redis.call('HSET', key,
                'ema', new_ema,
                'count', new_count,
                'last_time', current_time,
                'recent_rate', new_rate,
                'last_alpha', alpha
            )
            if new_count >= min_count then
                return {new_ema, alpha, new_rate}
            else
                return {nil, alpha, new_rate}
            end
        """

    async def _ensure_script(self, client):
        """
        Load the lua script if it's not already loaded.
        """
        if not hasattr(client, "_aema_script_sha"):
            setattr(client, "_aema_script_sha", await client.script_load(self.script))

    async def update(self, key: str, new_value: float):
        """
        Update the adaptive EMA metrics.
        """
        client = settings.cm_redis_client[uuid.UUID(key).int % len(settings.cm_redis_client)]
        await self._ensure_script(client)
        try:
            result = await client.evalsha(
                client._aema_script_sha,
                1,
                f"{self.key_prefix}:{key}",
                new_value,
                time.time(),
                self.target_window,
                self.min_alpha,
                self.max_alpha,
                self.min_count,
            )
        except redis.NoScriptError:
            client._aema_script_sha = await client.script_load(self.script)
            result = await client.evalsha(
                client._aema_script_sha,
                1,
                f"{self.key_prefix}:{key}",
                new_value,
                time.time(),
                self.target_window,
                self.min_alpha,
                self.max_alpha,
                self.min_count,
            )
        if result[0] is not None:
            return (float(result[0]), float(result[1]), float(result[2]))
        else:
            return (None, float(result[1]), float(result[2]))

    async def get_info(self, key: str):
        """
        Get current state information.
        """
        client = settings.cm_redis_client[uuid.UUID(key).int % len(settings.cm_redis_client)]
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
