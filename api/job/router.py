"""
Routes for jobs.
"""

import io
import uuid
import orjson as json
from fastapi import APIRouter, Depends, HTTPException, status, Request
from fastapi import File, UploadFile
from sqlalchemy import text, select, func, case, and_
from sqlalchemy.ext.asyncio import AsyncSession
from api.config import settings
from api.database import get_db_session
from api.chute.schemas import Chute
from api.chute.util import is_shared
from api.job.schemas import Job
from api.job.response import JobResponse
from api.user.schemas import User, JobQuota
from api.user.service import get_current_user
from api.instance.util import load_job_from_jwt, create_job_jwt

router = APIRouter()


@router.post("/{chute_id}/{method}", response_model=JobResponse)
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

    # Notify the miners.
    await settings.redis_client.publish(
        "miner_broadcast",
        json.dumps(
            {
                "reason": "job_created",
                "data": {
                    "job_id": job.job_id,
                    "method": method,
                    "chute_id": chute_id,
                    "image_id": chute.image.image_id,
                },
            }
        ).decode(),
    )

    return job


@router.post("/{job_id}", response_model=JobResponse)
async def finish_job_and_get_upload_targets(
    job_id: str,
    token: str,
    request: Request,
    db: AsyncSession = Depends(get_db_session),
):
    """
    Mark a job as complete (which could be failed; "done" either way)
    """
    job = await load_job_from_jwt(db, job_id)
    payload = await request.json()
    job.finished_at = func.now()
    job.status = payload.pop("status", "error")
    output_filenames = payload.pop("output_filenames", [])
    if job.status == "complete" and output_filenames:
        job.status = "complete_pending_uploads"
    job.result = payload.pop("result", "error")
    job.error_detail = payload.pop("detail", None)

    # Create presigned URLs to upload all of the output files directly instead of through the API.
    upload_urls = []
    job.output_files = []
    for filename in output_filenames:
        date_str = job.created_at.strftime("%Y-%m-%d")
        s3_key = f"jobs/{job.chute_id}/{date_str}/{job_id}/outputs/{filename}"
        file_jwt = create_job_jwt(job_id, filename=filename)
        upload_url = f"https://api.{settings.base_domain}/jobs/{job_id}/upload?token={file_jwt}"
        job.output_files.append(
            {
                "filename": filename,
                "path": s3_key,
                "uploaded": False,
            }
        )
        upload_urls.append(upload_url)

    await db.commit()
    await db.refresh(job)
    job_response = JobResponse.from_orm(job)
    job_response.upload_urls = upload_urls
    return job_response


@router.put("/{job_id}/upload")
async def upload_job_file(
    job_id: str,
    token: str,
    file: UploadFile = File(...),
    db: AsyncSession = Depends(get_db_session),
):
    """
    Upload a job's output file.
    """
    job = await load_job_from_jwt(db, job_id, token, filename=file.filename)
    file_index = None
    s3_key = None
    for i, output_file in enumerate(job.output_files or []):
        if output_file["filename"] == file.filename:
            file_index = i
            s3_key = output_file["path"]
            break
    try:
        async with settings.s3_client() as s3:
            file_content = await file.read()
            await s3.upload_fileobj(
                io.BytesIO(file_content),
                settings.storage_bucket,
                s3_key,
                ExtraArgs={
                    "ContentType": file.content_type or "application/octet-stream",
                    "Metadata": {
                        "job_id": job_id,
                        "chute_id": job.chute_id,
                        "original_filename": file.filename,
                    },
                },
            )
            await db.execute(
                text("""
                    UPDATE jobs
                    SET output_files = jsonb_set(
                        output_files,
                        :path,
                        'true',
                        false
                    )
                    WHERE id = :job_id
                """),
                {"path": f"{{{file_index},uploaded}}", "job_id": job_id},
            )
            await db.commit()
            return {
                "status": "success",
                "filename": file.filename,
                "path": s3_key,
                "uploaded": True,
            }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to upload file: {str(e)}")


@router.put("/{job_id}", response_model=JobResponse)
async def complete_job(
    job_id: str,
    token: str,
    request: Request,
    db: AsyncSession = Depends(get_db_session),
):
    """
    Final update, which checks the file uploads to see which were successfully transferred etc.
    """
    job = await load_job_from_jwt(db, job_id)
    if not job.output_files:
        return job

    # Mark partial failures if some files failed to be uploaded.
    all_uploaded = all(f["uploaded"] for f in job.output_files)
    if not all_uploaded:
        failed_files = [f["filename"] for f in job.output_files if not f["uploaded"]]
        if job.status.startswith("complete"):
            job.status = "partial_failure"
            job.error_detail = f"Failed to upload files: {', '.join(failed_files)}"
        else:
            job.error_detail += f"\nFailed to upload files: {', '.join(failed_files)}"
    elif job.status.startswith("complete"):
        job.status = "complete"

    job.updated_at = func.now()

    await db.commit()
    await db.refresh(job)
    return job
