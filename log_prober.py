import asyncio
from loguru import logger
from sqlalchemy import select
from api.config import settings
from api.database import get_session
import api.miner_client as miner_client
from api.instance.schemas import Instance


async def check_instance_logging_server(instance: Instance) -> bool:
    """
    Check a single instance's logging server.
    """
    logger.info(
        f"Checking {instance.instance_id=} of {instance.miner_hotkey=} {instance.chute_id=}"
    )
    log_port = None
    try:
        log_port = next(p for p in instance.port_mappings if p["internal_port"] == 8001)[
            "external_port"
        ]
        async with miner_client.get(
            instance.miner_hotkey,
            f"http://{instance.host}:{log_port}/logs",
            timeout=10,
            purpose="chutes",
        ) as resp:
            resp.raise_for_status()
            json_data = await resp.json()
            if "logs" not in json_data:
                raise ValueError("Missing 'logs' key in response")
            has_required_log = any(
                log.get("path") == "/tmp/_chute.log" for log in json_data["logs"]
            )
            if not has_required_log:
                raise ValueError("No log entry with path '/tmp/_chute.log' found")
            logger.success(
                f"✅ logging server running for {instance.instance_id=} of {instance.miner_hotkey=} for {instance.chute_id=} on http://{instance.host}:{log_port}"
            )
            return True
    except Exception:
        logger.error(
            f"❌ logging server check failure for {instance.instance_id=} of {instance.miner_hotkey=} for {instance.chute_id=} on http://{instance.host}:{log_port or '???'}"
        )
        return False


async def handle_check_result(instance_id: str, success: bool):
    """
    Handle the result of a check by updating Redis failure tracking.
    """
    redis_key = f"logserverfail:{instance_id}"
    if success:
        await settings.redis_client.delete(redis_key)
        return
    failure_count = await settings.redis_client.incr(redis_key)
    await settings.redis_client.expire(redis_key, 600)
    if failure_count >= 3:
        async with get_session() as session:
            instance = (
                (await session.execute(select(Instance).where(Instance.instance_id == instance_id)))
                .unique()
                .scalar_one_or_none()
            )
            if instance:
                logger.error(
                    f"❌ max consecutive logging server check failures encountered for {instance.instance_id=} of {instance.miner_hotkey=} for {instance.chute_id=}"
                )
                await session.delete(instance)
                await notify_deleted(instance, message="Failed 3 or more consecutive logging server probes.")
                await session.commit()


async def check_logging_servers(max_concurrent: int = 32):
    """
    Check all active instances' logging servers with concurrent execution.
    """
    semaphore = asyncio.Semaphore(max_concurrent)

    async def check_with_semaphore(instance: Instance):
        async with semaphore:
            success = await check_instance_logging_server(instance)
            await handle_check_result(instance.instance_id, success)
            return instance.instance_id, success

    async with get_session() as session:
        query = select(Instance).where(Instance.active.is_(True))
        result = await session.stream(query)
        instances = []
        async for row in result.unique():
            instances.append(row[0])
        logger.info(f"Checking {len(instances)} active instances")
        tasks = [check_with_semaphore(instance) for instance in instances]
        results = await asyncio.gather(*tasks)
        successful = sum(1 for _, success in results if success)
        failed = len(results) - successful
        logger.success("=" * 80)
        logger.success(f"Check complete: {successful} successful, {failed} failed")
        if failed > 0:
            failed_ids = [instance_id for instance_id, success in results if not success]
            logger.warning(f"Failed instances: {failed_ids}")


if __name__ == "__main__":
    asyncio.run(check_logging_servers())
