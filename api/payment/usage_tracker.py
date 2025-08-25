"""
Script to continuously pop microtransactions out of redis and
update the actual database with usage data, deduct user balance.
"""

import time
import asyncio
import api.database.orms  # noqa
from sqlalchemy import select, text, bindparam, String
from loguru import logger
from api.user.schemas import User
from api.config import settings
from api.permissions import Permissioning
from api.database import get_session


async def process_balance_changes():
    current_time = int(time.time())
    redis = settings.redis_client

    # Fetch all pending payment data in redis.
    keys_to_process = []
    cursor = 0
    pattern = "balance:*"
    while True:
        cursor, keys = await redis.scan(cursor, pattern, 100)
        for key in keys:
            keys_to_process.append(key)
        if cursor == 0:
            break

    # Update the usage_data table with the pending micro-payment data, and deduct from user balance.
    async with get_session() as session:
        user_totals = {}
        for key in keys_to_process:
            parts = key.decode().split(":")
            user_id = parts[1]
            chute_id = parts[2]

            # Fetch the individual amounts, counts, and timestamps.
            values = await redis.hgetall(key)
            amount = float(values.get(b"amount", 0))
            count = int(values.get(b"count", 0))
            timestamp = int(values.get(b"timestamp", 0))
            input_tokens = int(values.get(b"input_tokens") or 0)
            output_tokens = int(values.get(b"output_tokens") or 0)
            if user_id not in user_totals:
                user_totals[user_id] = 0
            user_totals[user_id] += amount

            # The usage data is tracked at hour granularity.
            hour_bucket = (timestamp // 3600) * 3600

            # Upsert the data.
            logger.info(
                f"Updating usage data for {user_id=} {chute_id=} {hour_bucket=} {amount=} {count=}"
            )
            stmt = text("""
                INSERT INTO usage_data (user_id, bucket, chute_id, amount, count, input_tokens, output_tokens)
                SELECT :user_id, to_timestamp(:hour_bucket), :chute_id, :amount, :count, :input_tokens, :output_tokens
                WHERE EXISTS (SELECT 1 FROM users WHERE user_id = :user_id)
                ON CONFLICT (user_id, chute_id, bucket)
                DO UPDATE SET
                    amount = (usage_data.amount + :amount),
                    count = (usage_data.count + :count),
                    input_tokens = (usage_data.input_tokens + :input_tokens),
                    output_tokens = (usage_data.output_tokens + :output_tokens)
            """).bindparams(
                bindparam("user_id", type_=String),
                bindparam("chute_id", type_=String),
            )
            await session.execute(
                stmt,
                {
                    "user_id": user_id,
                    "hour_bucket": hour_bucket,
                    "chute_id": chute_id,
                    "amount": amount,
                    "count": count,
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                },
            )

        # Update user balances
        for user_id, amount in user_totals.items():
            user = (
                (await session.execute(select(User).where(User.user_id == user_id)))
                .unique()
                .scalar_one_or_none()
            )
            if user and not user.has_role(Permissioning.free_account):
                logger.info(f"Deducting from {user_id} [{user.username}]: {amount}")
                user.balance -= amount
            elif user:
                logger.warning(
                    f"Free account {user_id} [{user.username}], skipping deduction: {amount}"
                )

        # Delete processed keys from Redis, while in the transaction.
        # If the transaction fails, we lose the ability to deduct this
        # balance amount, but really that's fine since this loop is
        # performed frequently and it's better than the alternative, i.e.
        # double charging.
        pipeline = redis.pipeline()
        for key in keys_to_process:
            pipeline.delete(key)
        await pipeline.execute()
        await session.commit()

    # Update last processed timestamp
    await redis.set("last_processed_timestamp", current_time)


async def main():
    while True:
        try:
            await process_balance_changes()
            logger.success("Successfully updated usage data/balances, waiting for next interval...")
        except Exception as exc:
            logger.error(f"Error processing balance changes: {exc}")
        await asyncio.sleep(60)


if __name__ == "__main__":
    asyncio.run(main())
