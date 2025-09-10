"""
Refresh balances (i.e. the materialized view which accounts for live private instances),
along with terminating any jobs/instances once balance reaches zero.
"""

import asyncio
from loguru import logger
from api.util import notify_deleted
from api.database import get_session
from sqlalchemy import text, select, func
import api.database.orms  # noqa
from api.instance.schemas import Instance
from api.job.schemas import Job
from api.user.schemas import User, UserCurrentBalance
from api.permissions import Permissioning


async def refresh_balance_view():
    """
    Refresh the materialized balance view.
    """
    logger.info("Refreshing user_current_balance materialized view...")
    async with get_session() as session:
        await session.execute(text("REFRESH MATERIALIZED VIEW CONCURRENTLY user_current_balance"))
        await session.commit()


async def terminate_jobs_for_zero_balance_users():
    """
    Find and terminate unfinished jobs for users with zero/negative balance.
    """
    logger.info("Looking for jobs to terminate for zero-balance users...")
    async with get_session() as session:
        result = await session.execute(
            select(Job)
            .join(User, Job.user_id == User.user_id)
            .join(UserCurrentBalance, User.user_id == UserCurrentBalance.user_id)
            .where(
                UserCurrentBalance.effective_balance <= 0,
                (
                    User.permissions_bitmask.op("&")(Permissioning.free_account.bitmask)
                    != Permissioning.free_account.bitmask
                ),
                (
                    User.permissions_bitmask.op("&")(Permissioning.invoice_billing.bitmask)
                    != Permissioning.invoice_billing.bitmask
                ),
                Job.finished_at.is_(None),
            )
        )
        jobs = result.scalars().all()
        if not jobs:
            logger.info("No jobs to terminate for zero-balance users!")
            return

        logger.info(f"Found {len(jobs)} jobs to terminate for zero-balance users...")
        for job in jobs:
            job.finished_at = func.now()
            job.status = "billing_termination"
            job.error_detail = "Job terminated due to insufficient balance"
            logger.warning(f"Terminated job {job.job_id} for user {job.user_id}")
            if job.instance:
                logger.warning(f"Deleting instance {job.instance.instance_id} for job {job.job_id}")
                await session.delete(job.instance)
                await session.execute(
                    text(
                        "UPDATE instance_audit SET deletion_reason = "
                        "'job has been terminated due to insufficient user balance' "
                        "WHERE instance_id = :instance_id"
                    ),
                    {"instance_id": job.instance.instance_id},
                )
                await notify_deleted(
                    job.instance, message="Job terminated due to insufficient user balance"
                )
            await session.commit()

        logger.success(f"Completed terminating {len(jobs)} jobs for zero-balance users")


async def shutdown_stale_instances():
    """
    Shut down instances that are beyond the billing stop time (they are no longer being used).
    """
    logger.info("Looking for instances past shutdown_after_seconds to terminate...")
    async with get_session() as session:
        instances = (
            (await session.execute(select(Instance).where(Instance.stop_billing_at <= func.now())))
            .unique()
            .scalars()
            .all()
        )
        for instance in instances:
            logger.warning(
                f"Shutting down {instance.instance_id=} {instance.miner_hotkey=} due to {instance.stop_billing_at=}"
            )
            await session.delete(instance)
            await session.execute(
                text(
                    "UPDATE instance_audit SET deletion_reason = "
                    "'user-defined/private chute instance has not been used since shutdown_after_seconds' "
                    "WHERE instance_id = :instance_id"
                ),
                {"instance_id": instance.instance_id},
            )
            await notify_deleted(
                instance, message="Instance has exceeded user defined shutdown_after_seconds"
            )
            await session.commit()


async def terminate_zero_balance_user_instances():
    """
    When a user no longer has any balance, shut down any
    private instances associated with the account.
    """
    logger.info("Looking for private instances to terminate for zero-balance users...")
    async with get_session() as session:
        result = await session.execute(
            select(Instance)
            .join(User, User.user_id == Instance.billed_to)
            .join(UserCurrentBalance, User.user_id == UserCurrentBalance.user_id)
            .where(Instance.billed_to == User.user_id)
            .where(
                UserCurrentBalance.effective_balance <= 0,
                (
                    User.permissions_bitmask.op("&")(Permissioning.free_account.bitmask)
                    != Permissioning.free_account.bitmask
                ),
                (
                    User.permissions_bitmask.op("&")(Permissioning.invoice_billing.bitmask)
                    != Permissioning.invoice_billing.bitmask
                ),
            )
        )
        instances = result.scalars().all()
        if not instances:
            logger.info("No private instances to terminate for zero-balance users")
            return

        logger.info(f"Found {len(instances)} private instances to terminate for zero-balance users")
        for instance in instances:
            logger.warning(
                f"Shutting down {instance.instance_id=} {instance.miner_hotkey=} "
                f"on private chute {instance.chute_id} due to zero balance"
            )
            await session.delete(instance)
            await session.execute(
                text(
                    "UPDATE instance_audit SET deletion_reason = "
                    "'user has zero/negative balance (private chute)' "
                    "WHERE instance_id = :instance_id"
                ),
                {"instance_id": instance.instance_id},
            )
            await notify_deleted(
                instance, message="Private instance terminated due to insufficient user balance"
            )
            await session.commit()

        logger.success(
            f"Completed terminating {len(instances)} private instances for zero-balance users"
        )


async def main():
    await refresh_balance_view()
    try:
        await shutdown_stale_instances()
        await refresh_balance_view()
        await terminate_jobs_for_zero_balance_users()
        await refresh_balance_view()
        await terminate_zero_balance_user_instances()
    finally:
        await refresh_balance_view()


if __name__ == "__main__":
    asyncio.run(main())
