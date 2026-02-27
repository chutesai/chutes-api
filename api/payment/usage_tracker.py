"""
Script to continuously pop usage records from redis queue and
update the actual database with usage data, deduct user balance.

Architecture:
1. Producer (util.py): RPUSH individual records to usage_queue
2. Consumer (this file):
   a. BRPOP from usage_queue
   b. HINCRBY/HINCRBYFLOAT to minute-bucket hash: usage_pending:{minute_ts}
      Fields: {user_id}:{chute_id}:amount, :count, :input_tokens, :output_tokens
   c. When minute rolls over, process completed buckets to DB
   d. DELETE processed bucket keys

Recovery on startup:
- Find all usage_pending:* keys
- Current minute = skip (still accumulating)
- Older minutes = process to DB (they're complete)

This gives us:
- Atomic increments (no race conditions)
- Only 1-2 Redis keys at any time
- Natural 60-second batching
- Simple recovery (completed minutes are self-contained)
"""

import gc
import sys
import time
import asyncio
import orjson as json
import api.database.orms  # noqa
import uvicorn
from datetime import datetime
from fastapi import FastAPI, Response, status
from collections import defaultdict
from sqlalchemy import text
from loguru import logger
from api.config import settings, SUBSCRIPTION_TIERS
from api.permissions import Permissioning
from api.database import get_session

QUEUE_KEY = "usage_queue"
BUCKET_PREFIX = "usage_pending"
PROCESSING_PREFIX = "usage_processing"
POLL_TIMEOUT = 5

# Health check tracking
last_heartbeat = time.time()
metrics = {"queue_size": 0, "lag": 0.0}
app = FastAPI()


@app.get("/health")
async def health_check():
    """
    Kubernetes readiness/liveness probe.
    Returns 503 if the main processing loop hasn't updated the heartbeat in > 60 seconds.
    """
    heartbeat_age = time.time() - last_heartbeat
    status_code = status.HTTP_200_OK

    if heartbeat_age > 60:
        status_code = status.HTTP_503_SERVICE_UNAVAILABLE

    return Response(
        content=json.dumps(
            {
                "status": "ok" if status_code == 200 else "stalled",
                "heartbeat_age": heartbeat_age,
                "queue_size": metrics["queue_size"],
                "lag": metrics["lag"],
            }
        ),
        status_code=status_code,
        media_type="application/json",
    )


def get_minute_ts(timestamp: int = None) -> int:
    if timestamp is None:
        timestamp = int(time.time())
    return timestamp - (timestamp % 60)


async def increment_bucket(redis, record: dict) -> None:
    """
    Increment counters in the minute-bucket hash for this record.

    Key: usage_pending:{minute_ts}
    Fields: {user_id}:{chute_id}:a (amount), :n (count), :i (input_tokens), :o (output_tokens), :x (cached_tokens), :t (compute_time)

    Record format (compact):
    - u: user_id, c: chute_id, a: amount, i: input_tokens, o: output_tokens, x: cached_tokens, t: compute_time, s: timestamp
    """
    user_id = record["u"]
    chute_id = record["c"]
    timestamp = int(record.get("s", time.time()))
    minute_ts = get_minute_ts(timestamp)

    bucket_key = f"{BUCKET_PREFIX}:{minute_ts}"
    field_prefix = f"{user_id}:{chute_id}"

    amount = float(record.get("a", 0))
    input_tokens = int(record.get("i", 0))
    output_tokens = int(record.get("o", 0))
    cached_tokens = int(record.get("x", 0))
    compute_time = float(record.get("t", 0))
    paygo_amount = float(record.get("p", 0))

    pipeline = redis.pipeline()
    pipeline.hincrbyfloat(bucket_key, f"{field_prefix}:a", amount)
    pipeline.hincrby(bucket_key, f"{field_prefix}:n", 1)
    pipeline.hincrby(bucket_key, f"{field_prefix}:i", input_tokens)
    pipeline.hincrby(bucket_key, f"{field_prefix}:o", output_tokens)
    pipeline.hincrby(bucket_key, f"{field_prefix}:x", cached_tokens)
    pipeline.hincrbyfloat(bucket_key, f"{field_prefix}:t", compute_time)
    pipeline.hincrbyfloat(bucket_key, f"{field_prefix}:p", paygo_amount)
    await pipeline.execute()


async def get_completed_buckets(redis, current_minute_ts: int) -> list[str]:
    """
    Find all bucket keys that are older than the current minute (i.e., complete).
    """
    keys = []
    cursor = 0
    pattern = f"{BUCKET_PREFIX}:*"
    while True:
        cursor, found_keys = await redis.scan(cursor, pattern, 1000)
        for key in found_keys:
            key_str = key.decode() if isinstance(key, bytes) else key
            # Extract timestamp from key
            try:
                bucket_ts = int(key_str.split(":")[-1])
                if bucket_ts < current_minute_ts:
                    keys.append(key_str)
            except (ValueError, IndexError):
                logger.warning(f"Invalid bucket key format: {key_str}")
        if cursor == 0:
            break
    return keys


async def get_stale_processing_buckets(redis) -> list[str]:
    """
    Find any usage_processing:* keys left over from a crashed worker.
    These need to be processed on startup.
    """
    keys = []
    cursor = 0
    pattern = f"{PROCESSING_PREFIX}:*"
    while True:
        cursor, found_keys = await redis.scan(cursor, pattern, 1000)
        for key in found_keys:
            key_str = key.decode() if isinstance(key, bytes) else key
            keys.append(key_str)
        if cursor == 0:
            break
    return keys


async def _warm_sub_cap_cache(aggregated: dict) -> None:
    """
    Reconcile subscription cap Redis keys from DB (source of truth).
    Overwrites any existing values so drift from the real-time INCRBYFLOAT
    path is corrected every ~60s.
    Runs as a background task to avoid blocking the main processing loop.
    """
    try:
        # Identify users with cap-eligible usage (paygo_amount > amount).
        sub_candidate_users = set()
        for (user_id, chute_id), m in aggregated.items():
            if m["p"] - m["a"] > 0:
                sub_candidate_users.add(user_id)

        if not sub_candidate_users:
            return

        # Batch-query which of these users have subscription quotas.
        async with get_session(readonly=True) as session:
            result = await session.execute(
                text("""
                    SELECT user_id FROM invocation_quotas
                    WHERE user_id = ANY(:user_ids)
                    AND chute_id = '*'
                    AND quota = ANY(:sub_quotas)
                """),
                {
                    "user_ids": list(sub_candidate_users),
                    "sub_quotas": list(SUBSCRIPTION_TIERS.keys())
                    + [q + 1 for q in SUBSCRIPTION_TIERS.keys()],
                },
            )
            sub_users = {row[0] for row in result}

        if not sub_users:
            return

        # Recompute totals from DB for full period and SET (overwrite) the cache keys.
        now = datetime.now()
        month_suffix = now.strftime("%Y%m")
        four_hour_bucket = int(time.time()) // (4 * 3600)

        sub_user_list = list(sub_users)
        async with get_session(readonly=True) as session:
            month_result = await session.execute(
                text("""
                    SELECT user_id, GREATEST(COALESCE(SUM(paygo_amount), 0) - COALESCE(SUM(amount), 0), 0)
                    FROM usage_data
                    WHERE user_id = ANY(:user_ids)
                    AND bucket >= date_trunc('month', now())
                    GROUP BY user_id
                """),
                {"user_ids": sub_user_list},
            )
            month_totals = {row[0]: float(row[1]) for row in month_result}

            four_hour_result = await session.execute(
                text("""
                    SELECT user_id, GREATEST(COALESCE(SUM(paygo_amount), 0) - COALESCE(SUM(amount), 0), 0)
                    FROM usage_data
                    WHERE user_id = ANY(:user_ids)
                    AND bucket >= now() - interval '4 hours'
                    GROUP BY user_id
                """),
                {"user_ids": sub_user_list},
            )
            four_hour_totals = {row[0]: float(row[1]) for row in four_hour_result}

        pipeline = settings.redis_client.pipeline()
        for user_id in sub_users:
            month_key = f"sub_cap_m:{month_suffix}:{user_id}"
            four_hour_key = f"sub_cap_4h:{four_hour_bucket}:{user_id}"
            pipeline.set(month_key, str(month_totals.get(user_id, 0.0)), ex=35 * 86400)
            pipeline.set(four_hour_key, str(four_hour_totals.get(user_id, 0.0)), ex=5 * 3600)
        await pipeline.execute()
        logger.info(f"Reconciled subscription cap cache for {len(sub_users)} users")
    except Exception as exc:
        logger.warning(f"Failed to reconcile subscription cap cache: {exc}")


async def process_bucket(redis, bucket_key: str, already_claimed: bool = False) -> None:
    """
    Process a completed minute bucket to the database.

    1. Atomically rename bucket to claim it (prevents double-processing by multiple workers)
    2. Read fields from the hash using HSCAN (streaming since there are boatloads of keys)
    3. Aggregate by user_id/chute_id
    4. Delete from Redis BEFORE commit to ensure we never double-charge
    5. Write to DB
    """
    bucket_ts = bucket_key.split(":")[-1]

    if already_claimed:
        # This is a stale usage_processing:* key from a crashed worker
        processing_key = bucket_key
    else:
        # Atomically claim the bucket by renaming it
        # If another worker already renamed it, this will fail and we skip
        processing_key = f"{PROCESSING_PREFIX}:{bucket_ts}"
        try:
            await redis.rename(bucket_key, processing_key)
        except Exception as e:
            # Key doesn't exist (already claimed by another worker) - skip
            logger.info(f"Bucket {bucket_key} already claimed by another worker: {e}")
            return

    # Extract minute timestamp from bucket key for the hour bucket
    hour_bucket = (int(bucket_ts) // 3600) * 3600

    # Parse fields into aggregated data using HSCAN to avoid loading everything at once
    # Fields are: {user_id}:{chute_id}:a (amount), :n (count), :i (input_tokens), :o (output_tokens), :x (cached_tokens), :t (compute_time)
    aggregated = defaultdict(
        lambda: {
            "a": 0.0,  # amount
            "n": 0,  # count
            "i": 0,  # input_tokens
            "o": 0,  # output_tokens
            "x": 0,  # cached_tokens
            "t": 0.0,  # compute_time
            "p": 0.0,  # paygo_amount
        }
    )

    cursor = 0
    field_count = 0
    while True:
        cursor, data = await redis.hscan(processing_key, cursor, count=1000)
        for field, value in data.items():
            field_str = field.decode() if isinstance(field, bytes) else field
            value_str = value.decode() if isinstance(value, bytes) else value

            parts = field_str.rsplit(":", 2)
            if len(parts) != 3:
                logger.warning(f"Invalid field format: {field_str}")
                continue

            user_id, chute_id, metric = parts[0], parts[1], parts[2]
            key = (user_id, chute_id)

            if metric == "a":
                aggregated[key]["a"] = float(value_str)
            elif metric == "n":
                aggregated[key]["n"] = int(value_str)
            elif metric == "i":
                aggregated[key]["i"] = int(value_str)
            elif metric == "o":
                aggregated[key]["o"] = int(value_str)
            elif metric == "x":
                aggregated[key]["x"] = int(value_str)
            elif metric == "t":
                aggregated[key]["t"] = float(value_str)
            elif metric == "p":
                aggregated[key]["p"] = float(value_str)

            field_count += 1

        if cursor == 0:
            break

    if not aggregated:
        await redis.delete(processing_key)
        return

    logger.info(f"Scanned {field_count} fields from {bucket_key} (processing as {processing_key})")
    for (user_id, chute_id), m in aggregated.items():
        logger.info(
            f"  {user_id}:{chute_id} -> amt={m['a']:.6f} n={m['n']} it={m['i']} ot={m['o']} ct={m['x']} t={m['t']:.4f}s p={m['p']:.6f}"
        )

    # Calculate user totals for balance deduction
    user_totals = defaultdict(float)
    for (user_id, chute_id), m in aggregated.items():
        user_totals[user_id] += m["a"]

    # Delete from Redis BEFORE commit - never double-charge
    await redis.delete(processing_key)

    # Write to database
    try:
        async with get_session() as session:
            # Batch upsert usage_data using unnest for efficiency
            sorted_items = sorted(aggregated.items(), key=lambda x: (x[0][0], x[0][1]))

            # This does all inserts in a single query instead of N queries
            usage_params = [
                {
                    "user_id": user_id,
                    "hour_bucket": hour_bucket,
                    "chute_id": chute_id,
                    "amount": m["a"],
                    "count": m["n"],
                    "input_tokens": m["i"],
                    "output_tokens": m["o"],
                    "cached_tokens": m["x"],
                    "compute_time": m["t"],
                    "paygo_amount": m["p"],
                }
                for (user_id, chute_id), m in sorted_items
            ]

            # Process in batches of 1000 to avoid query size limits
            batch_size = 1000
            for i in range(0, len(usage_params), batch_size):
                batch = usage_params[i : i + batch_size]
                stmt = text("""
                    INSERT INTO usage_data (user_id, bucket, chute_id, amount, count, input_tokens, output_tokens, cached_tokens, compute_time, paygo_amount)
                    SELECT
                        u.user_id,
                        to_timestamp(:hour_bucket),
                        d.chute_id,
                        d.amount,
                        d.count,
                        d.input_tokens,
                        d.output_tokens,
                        d.cached_tokens,
                        d.compute_time,
                        d.paygo_amount
                    FROM unnest(
                        CAST(:user_ids AS text[]),
                        CAST(:chute_ids AS text[]),
                        CAST(:amounts AS double precision[]),
                        CAST(:counts AS bigint[]),
                        CAST(:input_tokens AS bigint[]),
                        CAST(:output_tokens AS bigint[]),
                        CAST(:cached_tokens AS bigint[]),
                        CAST(:compute_times AS double precision[]),
                        CAST(:paygo_amounts AS double precision[])
                    ) AS d(user_id, chute_id, amount, count, input_tokens, output_tokens, cached_tokens, compute_time, paygo_amount)
                    JOIN users u ON u.user_id = d.user_id
                    ON CONFLICT (user_id, chute_id, bucket)
                    DO UPDATE SET
                        amount = (usage_data.amount + EXCLUDED.amount),
                        count = (usage_data.count + EXCLUDED.count),
                        input_tokens = (usage_data.input_tokens + EXCLUDED.input_tokens),
                        output_tokens = (usage_data.output_tokens + EXCLUDED.output_tokens),
                        cached_tokens = (COALESCE(usage_data.cached_tokens, 0) + EXCLUDED.cached_tokens),
                        compute_time = (COALESCE(usage_data.compute_time, 0) + EXCLUDED.compute_time),
                        paygo_amount = (COALESCE(usage_data.paygo_amount, 0) + EXCLUDED.paygo_amount)
                """)
                await session.execute(
                    stmt,
                    {
                        "hour_bucket": hour_bucket,
                        "user_ids": [p["user_id"] for p in batch],
                        "chute_ids": [p["chute_id"] for p in batch],
                        "amounts": [p["amount"] for p in batch],
                        "counts": [p["count"] for p in batch],
                        "input_tokens": [p["input_tokens"] for p in batch],
                        "output_tokens": [p["output_tokens"] for p in batch],
                        "cached_tokens": [p["cached_tokens"] for p in batch],
                        "compute_times": [p["compute_time"] for p in batch],
                        "paygo_amounts": [p["paygo_amount"] for p in batch],
                    },
                )

            logger.info(f"Upserted {len(usage_params)} usage_data rows for bucket {bucket_key}")

            # Batch update user balances using a single UPDATE with CASE
            # Filter out zero amounts and prepare for batch update
            nonzero_user_totals = {uid: amt for uid, amt in user_totals.items() if amt > 0}

            if nonzero_user_totals:
                # Get users who should NOT be charged (free accounts without invoice billing)
                free_users_result = await session.execute(
                    text("""
                        SELECT user_id FROM users
                        WHERE user_id = ANY(:user_ids)
                        AND (permissions_bitmask & :free_mask) = :free_mask
                        AND (permissions_bitmask & :invoice_mask) = 0
                    """),
                    {
                        "user_ids": list(nonzero_user_totals.keys()),
                        "free_mask": Permissioning.free_account.bitmask,
                        "invoice_mask": Permissioning.invoice_billing.bitmask,
                    },
                )
                free_user_ids = {row[0] for row in free_users_result}

                if free_user_ids:
                    logger.warning(
                        f"Skipping balance deduction for {len(free_user_ids)} free accounts"
                    )

                # Filter to only chargeable users
                chargeable = {
                    uid: amt for uid, amt in nonzero_user_totals.items() if uid not in free_user_ids
                }

                if chargeable:
                    # Sort users to prevent deadlocks when updating balances
                    # strictly ordering updates ensures consistent locking order across workers
                    sorted_user_ids = sorted(chargeable.keys())
                    sorted_amounts = [chargeable[uid] for uid in sorted_user_ids]

                    # Batch update balances in a single query
                    await session.execute(
                        text("""
                            UPDATE users
                            SET balance = balance - deductions.amount
                            FROM unnest(
                                CAST(:user_ids AS text[]),
                                CAST(:amounts AS double precision[])
                            ) AS deductions(user_id, amount)
                            WHERE users.user_id = deductions.user_id
                        """),
                        {
                            "user_ids": sorted_user_ids,
                            "amounts": sorted_amounts,
                        },
                    )
                    logger.info(
                        f"Deducted balance from {len(chargeable)} users, total=${sum(sorted_amounts)}"
                    )

            await session.commit()

        # Warm subscription cap cache in the background so it doesn't slow down the main loop.
        asyncio.create_task(_warm_sub_cap_cache(aggregated))

        logger.info(
            f"Processed bucket {bucket_key}: "
            f"{len(aggregated)} user/chute combos, "
            f"{len(user_totals)} users"
        )

    except Exception as exc:
        # DB failed but Redis key already deleted - data is lost (under-charge)
        # This is acceptable - better than double-charging
        logger.error(f"DB commit failed for bucket {bucket_key}: {exc}")
        raise


async def process_queue_items(redis, batch_size: int = 100) -> int:
    """
    Pop items from queue and increment their minute buckets.
    Uses batch processing and local aggregation to minimize Redis round trips.
    """
    global metrics

    items = await redis.lpop(QUEUE_KEY, count=batch_size)
    if not items:
        return 0

    # Calculate lag from the oldest item (first in list) and check remaining queue size
    try:
        first_record = json.loads(items[0])
        ts = float(first_record.get("s", 0)) or time.time()
        lag = time.time() - ts
    except Exception as exc:
        logger.error(f"Failed to determine queue lag: {str(exc)}")
        lag = 0.0

    remaining = await redis.llen(QUEUE_KEY)

    # Update global metrics
    metrics["queue_size"] = remaining
    metrics["lag"] = lag

    logger.info(f"Popped {len(items)} items. Queue size: {remaining}. Head lag: {lag:.2f}s")

    # Local aggregation to minimize HINCRBY calls
    # minute_ts -> user_id -> chute_id -> metrics
    updates = defaultdict(
        lambda: defaultdict(
            lambda: defaultdict(
                lambda: {"a": 0.0, "n": 0, "i": 0, "o": 0, "x": 0, "t": 0.0, "p": 0.0}
            )
        )
    )

    count = 0
    for item in items:
        if not item:
            continue
        try:
            record = json.loads(item)

            user_id = record["u"]
            chute_id = record["c"]
            timestamp = int(record.get("s", time.time()))
            minute_ts = get_minute_ts(timestamp)

            amount = float(record.get("a", 0))
            input_tokens = int(record.get("i", 0))
            output_tokens = int(record.get("o", 0))
            cached_tokens = int(record.get("x", 0))
            compute_time = float(record.get("t", 0))
            paygo_amount = float(record.get("p", 0))

            agg = updates[minute_ts][user_id][chute_id]
            agg["a"] += amount
            agg["n"] += 1
            agg["i"] += input_tokens
            agg["o"] += output_tokens
            agg["x"] += cached_tokens
            agg["t"] += compute_time
            agg["p"] += paygo_amount

            count += 1
        except Exception as exc:
            logger.error(f"Failed to process queue item: {exc}, raw={item}")

    # Push aggregated updates to Redis
    if count > 0:
        pipeline = redis.pipeline()
        for minute_ts, users in updates.items():
            bucket_key = f"{BUCKET_PREFIX}:{minute_ts}"
            for user_id, chutes in users.items():
                for chute_id, m in chutes.items():
                    field_prefix = f"{user_id}:{chute_id}"
                    if m["a"] != 0:
                        pipeline.hincrbyfloat(bucket_key, f"{field_prefix}:a", m["a"])
                    if m["n"] != 0:
                        pipeline.hincrby(bucket_key, f"{field_prefix}:n", m["n"])
                    if m["i"] != 0:
                        pipeline.hincrby(bucket_key, f"{field_prefix}:i", m["i"])
                    if m["o"] != 0:
                        pipeline.hincrby(bucket_key, f"{field_prefix}:o", m["o"])
                    if m["x"] != 0:
                        pipeline.hincrby(bucket_key, f"{field_prefix}:x", m["x"])
                    if m["t"] != 0:
                        pipeline.hincrbyfloat(bucket_key, f"{field_prefix}:t", m["t"])
                    if m["p"] != 0:
                        pipeline.hincrbyfloat(bucket_key, f"{field_prefix}:p", m["p"])
        await pipeline.execute()
    if count < batch_size:
        await asyncio.sleep(1)

    return count


async def process_usage_queue(batch_size: int = 100):
    """
    Main processing loop:
    1. On startup, process any completed minute buckets (recovery)
    2. Continuously pop from queue and increment minute buckets
    3. When minute rolls over, process completed buckets to DB
    """
    global last_heartbeat
    redis = settings.billing_redis_client.client
    last_minute_ts = get_minute_ts()

    # On startup, we first drain the queue to ensure the timestamp buckets are complete.
    logger.info("Draining queue before recovery...")
    drained_total = 0
    while True:
        drained = await process_queue_items(redis, batch_size=batch_size)
        drained_total += drained
        if drained < batch_size:
            break

    if drained_total > 0:
        logger.info(f"Drained {drained_total} items from queue into minute buckets")

    # Now, process any of  the stake keys from crashed workers.
    logger.info("Checking for stale processing buckets to recover...")
    stale_processing = await get_stale_processing_buckets(redis)
    if stale_processing:
        logger.warning(
            f"Recovering {len(stale_processing)} stale processing buckets from crashed worker"
        )
        for processing_key in stale_processing:
            try:
                await process_bucket(redis, processing_key, already_claimed=True)
            except Exception as exc:
                logger.error(f"Failed to recover stale processing bucket {processing_key}: {exc}")

    # And finally, process any completed buckets from before startup.
    logger.info("Checking for completed buckets to recover...")
    completed = await get_completed_buckets(redis, last_minute_ts)
    if completed:
        logger.warning(f"Recovering {len(completed)} completed buckets from previous run")
        for bucket_key in completed:
            try:
                await process_bucket(redis, bucket_key)
            except Exception as exc:
                logger.error(f"Failed to recover bucket {bucket_key}: {exc}")

    logger.info("Starting main processing loop")

    while True:
        last_heartbeat = time.time()
        try:
            current_minute_ts = get_minute_ts()
            if current_minute_ts > last_minute_ts:
                completed = await get_completed_buckets(redis, current_minute_ts)
                for bucket_key in completed:
                    try:
                        await process_bucket(redis, bucket_key)
                    except Exception as exc:
                        logger.error(f"Failed to process bucket {bucket_key}: {exc}")
                last_minute_ts = current_minute_ts

            # Process any items in the queue
            processed = await process_queue_items(redis, batch_size=batch_size)

            if processed > 0:
                logger.info(f"Processed {processed} queue items into minute buckets")
            else:
                # Queue is empty, wait for more items
                item = await redis.blpop(QUEUE_KEY, timeout=POLL_TIMEOUT)
                if item:
                    _, data = item
                    try:
                        record = json.loads(data)
                        await increment_bucket(redis, record)
                    except Exception as exc:
                        logger.error(f"Failed to process queue item: {exc}, raw={data}")

        except asyncio.TimeoutError:
            # BLPOP timeout or connection timeout - this is normal, just continue
            pass
        except Exception as exc:
            if "Timeout" in str(exc):
                # Redis timeout - normal when queue is idle
                pass
            else:
                logger.error(f"Error in usage queue processing: {exc}")
                await asyncio.sleep(5)


async def main():
    logger.info("Starting usage tracker...")

    # Start health check server
    config = uvicorn.Config(app, host="0.0.0.0", port=8000, log_level="error")
    server = uvicorn.Server(config)

    # Run server and processor concurrently, ensure both continue running.
    tasks = [asyncio.create_task(server.serve()), asyncio.create_task(process_usage_queue())]

    # This should never actually return, since both processes are expected to run forever.
    done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
    for task in pending:
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

    for task in done:
        try:
            task.result()
        except Exception as exc:
            logger.error(f"Task failed with exception: {exc}")
            raise

    logger.error("A critical task finished unexpectedly!")
    sys.exit(1)


if __name__ == "__main__":
    gc.set_threshold(5000, 50, 50)
    asyncio.run(main())
