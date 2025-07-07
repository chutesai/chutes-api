"""
Routes for jobs.
"""

import uuid
from fastapi import APIRouter, Depends, HTTPException, status, Request
from sqlalchemy import select, func, case, and_
from sqlalchemy.ext.asyncio import AsyncSession
from api.database import get_db_session
from api.chute.schemas import Chute
from api.chute.util import is_shared
from api.job.schemas import Job
from api.user.schemas import User, JobQuota
from api.user.service import get_current_user

router = APIRouter()


@router.post("/{chute_id}/{method}")
async def create_job(
    chute_id: str,
    method: str,
    request: Request,
    db: AsyncSession = Depends(get_db_session),
    current_user: User = Depends(get_current_user(raise_not_found=False)),
):
    # Load the chute.
    chute = (
        (
            await db.execute(
                select(Chute)
                .where(Chute.chute_id == chute_id)
                .where(Chute.jobs.op("@>")([{"name": method}]))
            )
        )
        .unique()
        .scalar_one_or_none()
    )
    if not chute or (
        not chute.public
        and chute.user_id != current_user.user_id
        and not await is_shared(chute_id, current_user.user_id)
    ):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Chute {chute_id} not found",
        )

    # User has quota for jobs?
    query = select(
        func.count(Job.id).label("total_jobs"),
        func.count(case((Job.chute_id == chute_id, Job.id))).label("chute_jobs"),
    ).where(
        and_(
            Job.user_id == current_user.user_id,
            Job.created_at >= func.date_trunc("day", func.now()),
        )
    )
    result = await db.execute(query)
    counts = result.one()
    total_job_count = counts.total_jobs
    chute_job_count = counts.chute_jobs

    job_quota = await JobQuota.get(current_user.user_id, chute_id)
    total_quota = await JobQuota.get(current_user.user_id, "global")
    if not job_quota or chute_job_count >= job_quota or total_job_count >= total_quota:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=f"Daily job quota exceeded: {chute_job_count=} {total_job_count=} of {job_quota=} and {total_quota=}",
        )

    # Create the job.
    job = Job(
        job_id=str(uuid.uuid4()),
        user_id=current_user.user_id,
        chute_id=chute_id,
        version=chute.version,
        chutes_version=chute.chutes_version,
        method=method,
        miner_uid=None,
        miner_hotkey=None,
        miner_coldkey=None,
        instance_id=None,
        active=False,
        verified=False,
        last_queried_at=None,
        job_args=await request.json(),
    )
    db.add(job)
    await db.commit()
    await db.refresh(job)

    # XXX Notify the miners.

    return job
