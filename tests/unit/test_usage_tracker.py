import asyncio

import orjson as json
import pytest

from api.payment import usage_tracker


class FakePipeline:
    def __init__(self, redis):
        self.redis = redis
        self.ops = []

    def hincrbyfloat(self, key, field, amount):
        self.ops.append(("hincrbyfloat", key, field, amount))

    def hincrby(self, key, field, amount):
        self.ops.append(("hincrby", key, field, amount))

    async def execute(self):
        for op, key, field, amount in self.ops:
            bucket = self.redis.hashes.setdefault(key, {})
            current = bucket.get(field, 0)
            bucket[field] = current + amount


class FakeRedis:
    def __init__(self, items):
        self.items = list(items)
        self.hashes = {}

    async def lpop(self, key, count=1):
        assert key == usage_tracker.QUEUE_KEY
        popped = self.items[:count]
        self.items = self.items[count:]
        return popped

    async def llen(self, key):
        assert key == usage_tracker.QUEUE_KEY
        return len(self.items)

    def pipeline(self):
        return FakePipeline(self)


@pytest.mark.asyncio
async def test_process_queue_items_tracks_app_usage_in_parallel_buckets(monkeypatch):
    minute_ts = 1712345670 - (1712345670 % 60)
    records = [
        json.dumps(
            {
                "u": "user-1",
                "c": "chute-1",
                "a": 1.5,
                "i": 10,
                "o": 20,
                "x": 3,
                "t": 0.25,
                "p": 2.0,
                "s": 1712345670,
                "d": "app-1",
            }
        ),
        json.dumps(
            {
                "u": "user-1",
                "c": "chute-1",
                "a": 0.5,
                "i": 1,
                "o": 2,
                "x": 0,
                "t": 0.5,
                "p": 0.75,
                "s": 1712345675,
            }
        ),
    ]
    redis = FakeRedis(records)
    sleep_calls = []

    async def fake_sleep(delay):
        sleep_calls.append(delay)

    monkeypatch.setattr(asyncio, "sleep", fake_sleep)
    usage_tracker.metrics["queue_size"] = 999
    usage_tracker.metrics["lag"] = 999.0

    processed = await usage_tracker.process_queue_items(redis, batch_size=100)

    assert processed == 2
    assert sleep_calls == [1]
    assert usage_tracker.metrics["queue_size"] == 0

    bucket_key = f"{usage_tracker.BUCKET_PREFIX}:{minute_ts}"
    assert redis.hashes[bucket_key] == {
        "user-1:chute-1:a": 2.0,
        "user-1:chute-1:n": 2,
        "user-1:chute-1:i": 11,
        "user-1:chute-1:o": 22,
        "user-1:chute-1:x": 3,
        "user-1:chute-1:t": 0.75,
        "user-1:chute-1:p": 2.75,
    }

    app_bucket_key = f"{usage_tracker.APP_BUCKET_PREFIX}:{minute_ts}"
    assert redis.hashes[app_bucket_key] == {
        "app-1:user-1:chute-1:a": 1.5,
        "app-1:user-1:chute-1:n": 1,
        "app-1:user-1:chute-1:i": 10,
        "app-1:user-1:chute-1:o": 20,
        "app-1:user-1:chute-1:x": 3,
        "app-1:user-1:chute-1:t": 0.25,
        "app-1:user-1:chute-1:p": 2.0,
    }
