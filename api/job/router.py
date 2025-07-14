"""
Routes for jobs.
"""

import io
import uuid
import backoff
import traceback
import orjson as json
from loguru import logger
from fastapi import APIRouter, Depends, HTTPException, status, Request, Response, Form
from fastapi import File, UploadFile
from sqlalchemy import text, select, func, case, and_
from sqlalchemy.orm import selectinload
from sqlalchemy.ext.asyncio import AsyncSession
from api.config import settings
import api.miner_client as miner_client
from api.util import aes_encrypt, notify_job_deleted
from api.database import get_db_session
from api.chute.schemas import Chute, NodeSelector
from api.chute.util import is_shared
from api.job.schemas import Job
from api.job.response import JobResponse
from api.user.schemas import User, JobQuota
from api.user.service import get_current_user
from api.instance.util import load_job_from_jwt, create_job_jwt

router = APIRouter()


async def get_job_by_id(
    db: AsyncSession,
    job_id: str,
    current_user: User,
):
    job = (
        (
            await db.execute(
                select(Job).where(Job.job_id == job_id, Job.user_id == current_user.user_id)
            )
        )
        .unique()
        .scalar_one_or_none()
    )
    if not job:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Job not found, or does not belong to you",
        )
    return job


@backoff.on_exception(
    backoff.constant,
    Exception,
    jitter=None,
    interval=10,
    max_tries=7,
)
async def batch_delete_stored_files(job):
    """
    Helper to delete output files for a given job (in blob store) in batches.
    """
    if not job.output_files:
        return
    # output_files is now a dict, so we extract the paths from the values
    keys = [file_info["path"] for file_info in job.output_files.values()]
    async with settings.s3_client() as s3_client:
        for i in range(0, len(keys), 100):
            batch_keys = keys[i : i + 100]
            await s3_client.delete_objects(
                Bucket=settings.storage_bucket,
                Delete={
                    "Objects": [{"Key": key} for key in batch_keys],
                    "Quiet": True,
                },
            )


@router.post("/{chute_id}/{method}", response_model=JobResponse)
async def create_job(
    chute_id: str,
    method: str,
    request: Request,
    db: AsyncSession = Depends(get_db_session),
    current_user: User = Depends(get_current_user()),
):
    """
    Create a job.
    """
    # Load the chute.
    chute = (
        (
            await db.execute(
                select(Chute)
                .where(Chute.chute_id == chute_id)
                .where(Chute.jobs.op("@>")([{"name": method}]))
                .options(selectinload(Chute.instances))
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
        func.count(Job.job_id).label("total_jobs"),
        func.count(case((Job.chute_id == chute_id, Job.job_id))).label("chute_jobs"),
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

    # Cleverly determine compute multiplier, such that jobs have equal priority to normal chutes.
    node_selector = NodeSelector(**chute.node_selector)
    compute_multiplier = node_selector.compute_multiplier

    # Disk requirements?
    job_args = await request.json()
    disk_gb = job_args.get("_disk_gb")
    if disk_gb is not None:
        if not isinstance(disk_gb, (int, float)) or not 10 < disk_gb < 1000:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid disk size specified: {disk_gb}",
            )
    if not disk_gb:
        disk_gb = 10
    job_args["_disk_gb"] = int(disk_gb)

    # XXX for this version, we'll be.. not clever - ultimately needs a way
    # to calculate the maximum any particular GPU is getting at any point in time.
    if not set(node_selector.supported_gpus) - set(["h200"]):
        # 2025-07-08: all h200 chutes are at max capacity really, so we need
        # the multiplier to be quite aggressive. Each of those chutes have
        # 16-20 concurrency specified, meaning each node can have far more
        # compute units than just the baseline, i.e. they are getting
        # 16-20x the compute units at any given time.
        compute_multiplier *= 16

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
        job_args=job_args,
        miner_history=[],
        compute_multiplier=compute_multiplier,
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
                    "gpu_count": node_selector.gpu_count,
                    "compute_multiplier": compute_multiplier,
                    "exclude": [],
                    "disk_gb": disk_gb,
                },
            }
        ).decode(),
    )

    return job


@router.delete("/{job_id}", response_model=JobResponse)
async def delete_job(
    job_id: str,
    request: Request,
    db: AsyncSession = Depends(get_db_session),
    current_user: User = Depends(get_current_user()),
):
    """
    Delete a job.
    """
    job = (
        (
            await db.execute(
                select(Job).where(Job.job_id == job_id, Job.user_id == current_user.user_id)
            )
        )
        .unique()
        .scalar_one_or_none()
    )
    if not job:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Job {job_id} not found or does not belong to you",
        )
    await batch_delete_stored_files(job)
    await db.delete(job)

    # Notify the miner.
    miner_notified = False
    if job.instance:
        instance = job.instance
        try:
            payload = {"reason": "user initiated"}
            enc_payload = aes_encrypt(json.dumps(payload).encode(), instance.symmetric_key)
            path = aes_encrypt("/_shutdown", instance.symmetric_key, hex_encode=True)
            async with miner_client.post(
                instance.miner_hotkey,
                f"http://{instance.host}:{instance.port}/{path}",
                enc_payload,
                timeout=30.0,
            ) as resp:
                resp.raise_for_status()
                miner_notified = True
        except Exception as exc:
            logger.error(f"Error calling job shutdown endpoint: {exc}\n{traceback.format_exc()}")

    # If we didn't notify the specific miner running the job (if any), broadcast the event.
    if not miner_notified:
        await notify_job_deleted(job)

    return {
        "deleted": True,
        "job_id": job_id,
    }


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
    job = await load_job_from_jwt(db, job_id, token)
    if job.finished_at:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Job already finished",
        )

    payload = await request.json()
    job.status = payload.pop("status", "error")
    output_filenames = payload.pop("output_filenames", [])
    if job.status == "complete" and output_filenames:
        job.status = "complete_pending_uploads"
    job.result = payload.pop("result", "error")
    job.error_detail = payload.pop("detail", None)
    upload_urls = {}
    job.output_files = {}
    job_namespace = uuid.UUID(job_id)
    for filename in output_filenames:
        file_id = str(uuid.uuid5(job_namespace, filename))
        date_str = job.created_at.strftime("%Y-%m-%d")
        s3_key = f"jobs/{job.chute_id}/{date_str}/{job_id}/outputs/{filename}"
        file_jwt = create_job_jwt(job_id, filename=filename)
        upload_url = f"https://api.{settings.base_domain}/jobs/{job_id}/upload?token={file_jwt}"
        job.output_files[file_id] = {
            "filename": filename,
            "path": s3_key,
            "uploaded": False,
        }
        upload_urls[filename] = upload_url

    await db.commit()
    await db.refresh(job)
    job_response = JobResponse.from_orm(job)
    job_response.output_storage_urls = upload_urls
    return job_response


@router.put("/{job_id}/upload")
async def upload_job_file(
    job_id: str,
    token: str,
    file: UploadFile = File(...),
    path: str = Form(...),
    db: AsyncSession = Depends(get_db_session),
):
    """
    Upload a job's output file.
    """
    job = await load_job_from_jwt(db, job_id, token, filename=path)
    if job.finished_at:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Job already finished",
        )
    job_namespace = uuid.UUID(job_id)
    file_id = str(uuid.uuid5(job_namespace, path))
    if file_id not in job.output_files:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"File {path} not found in job output files",
        )

    s3_key = job.output_files[file_id]["path"]

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
                    "original_filename": path,
                },
            },
        )

        await db.execute(
            text("""
                UPDATE jobs
                SET output_files = jsonb_set(
                    output_files,
                    :path,
                    (output_files #> :path) || jsonb_build_object('uploaded', true),
                    true
                )
                WHERE job_id = :job_id
            """),
            {"path": [file_id], "job_id": job_id},
        )

        await db.commit()
        return {
            "status": "success",
            "filename": path,
            "file_id": file_id,
            "path": s3_key,
            "uploaded": True,
        }


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
    job = await load_job_from_jwt(db, job_id, token)
    if job.finished_at:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Job already finished",
        )
    if job.output_files:
        all_uploaded = all(file_info["uploaded"] for file_info in job.output_files.values())
        if not all_uploaded:
            failed_files = [
                file_info["filename"]
                for file_info in job.output_files.values()
                if not file_info["uploaded"]
            ]
            if job.status.startswith("complete"):
                job.status = "partial_failure"
                job.error_detail = f"Failed to upload files: {', '.join(failed_files)}"
            else:
                job.error_detail += f"\nFailed to upload files: {', '.join(failed_files)}"
        elif job.status.startswith("complete"):
            job.status = "complete"

    job.updated_at = func.now()
    job.finished_at = func.now()
    if job.instance:
        await db.delete(job.instance)
    await db.commit()
    await db.refresh(job)
    return job


@router.get("/{job_id}", response_model=JobResponse)
async def get_job(
    job_id: str,
    db: AsyncSession = Depends(get_db_session),
    current_user: User = Depends(get_current_user()),
):
    """
    Get a job.
    """
    job = (
        (
            await db.execute(
                select(Job).where(Job.job_id == job_id, Job.user_id == current_user.user_id)
            )
        )
        .unique()
        .scalar_one_or_none()
    )
    if not job:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Job not found, or does not belong to you",
        )
    return job


@router.get("/{job_id}/download", response_model=JobResponse)
async def download_output_file(
    job_id: str,
    filename: str,
    db: AsyncSession = Depends(get_db_session),
    current_user: User = Depends(get_current_user()),
):
    """
    Download a job's output file.
    """
    job = await get_job_by_id(db, job_id, current_user)
    job_namespace = uuid.UUID(job_id)
    file_id = str(uuid.uuid5(job_namespace, filename))
    if file_id not in job.output_files:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Job output file not found {job_id=} {filename=}",
        )
    output_file = job.output_files[file_id]
    data = io.BytesIO()
    async with settings.s3_client() as client:
        await client.download_fileobj(settings.storage_bucket, output_file["path"], data)
    return Response(
        content=data.getvalue(),
        media_type="application/octet-stream",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )
