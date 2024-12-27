"""
Invocations router.
"""

import pybase64 as base64
import pickle
import gzip
import orjson as json
import csv
import uuid
from datetime import date, datetime
from io import BytesIO, StringIO
from typing import Optional
from fastapi_cache.decorator import cache
from fastapi import APIRouter, Depends, HTTPException, status, Request, Response
from starlette.responses import StreamingResponse
from sqlalchemy import text, select
from sqlalchemy.ext.asyncio import AsyncSession
from api.config import settings
from api.chute.schemas import Chute
from api.chute.util import get_chute_by_id_or_name, invoke
from api.user.schemas import User
from api.user.service import get_current_user, chutes_user_id
from api.report.schemas import Report, ReportArgs
from api.database import get_db_session
from api.instance.util import discover_chute_targets
from api.permissions import Permissioning

router = APIRouter()
host_invocation_router = APIRouter()


@router.get("/exports/{year}/{month}/{day}/{hour}.csv")
async def get_export(
    year: int,
    month: int,
    day: int,
    hour: int,
) -> Response:
    """
    Get invocation exports for a particular hour.
    """
    # Sanity check the dates.
    valid = True
    if (
        (not 2024 <= year <= date.today().year)
        or not (1 <= month <= 12)
        or not (1 <= day <= 31)
        or not (0 <= hour <= 23)
    ):
        valid = False
    target_date = datetime(year, month, day, hour)
    today = date.today()
    current_hour = datetime.utcnow()
    if (
        target_date > datetime.utcnow()
        or target_date < datetime(2024, 12, 14, 0)
        or (target_date.date == today and hour == current_hour)
    ):
        valid = False
    if not valid:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Invocations export not found {year=} {month=} {day=} {hour=}",
        )

    exists = False
    key = f"invocations/{year}/{month:02d}/{day:02d}/{hour:02d}.csv"
    async with settings.s3_client() as s3:
        try:
            await s3.head_object(Bucket=settings.storage_bucket, Key=key)
            exists = True
        except Exception as exc:
            if exc.response["Error"]["Code"] != "404":
                raise
    if not exists:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Invocations export not found {year=} {month=} {day=} {hour=}",
        )

    data = BytesIO()
    async with settings.s3_client() as s3:
        await s3.download_fileobj(settings.storage_bucket, key, data)
    filename = key.replace("invocations/", "").replace("/", "-")
    return Response(
        content=data.getvalue(),
        media_type="text/csv",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@cache(expire=60)
@router.get("/exports/recent")
async def get_recent_export(
    hotkey: Optional[str] = None,
    limit: Optional[int] = 100,
    db: AsyncSession = Depends(get_db_session),
):
    """
    Get an export for recent data, which may not yet be in S3.
    """
    query = """
        SELECT
            invocation_id,
            chute_id,
            chute_user_id,
            function_name,
            image_id,
            image_user_id,
            instance_id,
            miner_uid,
            miner_hotkey,
            started_at,
            completed_at,
            error_message,
            compute_multiplier,
            bounty
        FROM partitioned_invocations
        WHERE started_at >= CURRENT_TIMESTAMP - INTERVAL '1 day'
    """
    if not limit or limit <= 0:
        limit = 100
    limit = min(limit, 10000)
    params = {"limit": limit}
    if hotkey:
        query += " AND miner_hotkey = :hotkey"
        params["hotkey"] = hotkey
    query += " ORDER BY started_at DESC LIMIT :limit"
    output = StringIO()
    writer = csv.writer(output)
    result = await db.execute(text(query), params)
    writer.writerow([col for col in result.keys()])
    writer.writerows(result)
    return Response(
        content=output.getvalue(),
        media_type="text/csv",
        headers={"Content-Disposition": 'attachment; filename="recent.csv"'},
    )


@router.post("/{invocation_id}/report")
async def report_invocation(
    invocation_id: str,
    report_args: ReportArgs,
    db: AsyncSession = Depends(get_db_session),
    current_user: User = Depends(get_current_user()),
):
    # Make sure the invocation exists and there isn't already a report.
    report_exists = (
        await db.execute(
            select(
                text(
                    "EXISTS (SELECT 1 FROM reports WHERE invocation_id = :invocation_id)"
                ).bindparams(invocation_id=invocation_id)
            )
        )
    ).scalar()
    if report_exists:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="A report has already been filed for this invocation",
        )
    invocation_exists = (
        await db.execute(
            select(
                text(
                    "EXISTS (SELECT 1 FROM invocations WHERE parent_invocation_id = :invocation_id AND user_id = :user_id)"
                ).bindparams(invocation_id=invocation_id, user_id=current_user.user_id)
            )
        )
    ).scalar()
    if not invocation_exists:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Invocation not found, or does not belong to you",
        )

    report = Report(
        invocation_id=invocation_id,
        user_id=current_user.user_id,
        reason=report_args.reason,
    )
    db.add(report)
    await db.commit()
    return {
        "status": f"report received for {invocation_id=}",
    }


async def _invoke(
    request: Request,
    db: AsyncSession,
    current_user: User,
):
    # This call will perform auth/access checks.
    chute = await get_chute_by_id_or_name(request.state.chute_id, db, current_user)
    if not chute:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="No matching chute found!"
        )

    # Check account balance.
    if current_user.balance <= 0 and not current_user.has_role(Permissioning.free_account):
        raise HTTPException(
            status_code=status.HTTP_402_PAYMENT_REQUIRED,
            detail=f"Account balance is ${current_user.balance}, please send tao to {current_user.payment_address}",
        )

    # Identify the cord that we'll trying to access by the public API path and method.
    selected_cord = None
    request_body = await request.json() if request.method in ("POST", "PUT", "PATCH") else {}
    request_params = request.query_params._dict if request.query_params else {}
    stream = request_body.get("stream", request_params.get("stream", False))
    for cord in chute.cords:
        public_path = cord.get("public_api_path", None)
        if public_path and public_path == request.url.path:
            if cord.get("public_api_method", "POST") == request.method and stream == cord.get(
                "stream"
            ):
                selected_cord = cord
                break
    if not selected_cord:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="No matching cord found!")

    # Wrap up the args/kwargs in the way the miner execution service expects them.
    args, kwargs = None, None
    if chute.standard_template in ("vllm", "tei") or selected_cord.get("passthrough", False):
        request_body = {"json": request_body, "params": request_params}
        args = base64.b64encode(gzip.compress(pickle.dumps(tuple()))).decode()
        kwargs = base64.b64encode(gzip.compress(pickle.dumps(request_body))).decode()
    else:
        args = base64.b64encode(gzip.compress(pickle.dumps((request_body,)))).decode()
        kwargs = base64.b64encode(gzip.compress(pickle.dumps({}))).decode()
    targets = await discover_chute_targets(db, chute.chute_id, max_wait=60)
    if not targets:
        chute_id = request.state.chute_id
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"No instances available (yet) for {chute_id=}",
        )

    # Initialize metrics.
    metrics = None
    if chute.standard_template == "vllm":
        metrics = {
            "ttft": None,
            "tps": 0.0,
            "tokens": 0,
        }
    elif chute.standard_template == "diffusion":
        metrics = {
            "sps": 0,
            "steps": request_body.get("num_inference_steps", 25.0),
        }

    # To stream, or not to stream.
    parent_invocation_id = str(uuid.uuid4())
    if stream:

        async def _stream_response():
            skip = False
            async for chunk in invoke(
                chute,
                current_user.user_id,
                selected_cord["path"],
                selected_cord["function"],
                stream,
                args,
                kwargs,
                targets,
                parent_invocation_id,
                metrics=metrics,
            ):
                if skip:
                    continue
                if chunk.startswith('data: {"result"'):
                    yield json.loads(chunk[6:])["result"]
                elif chunk.startswith('data: {"error"'):
                    error = json.loads(chunk[6:])["error"]
                    yield json.dumps(
                        {
                            "status": status.HTTP_500_INTERNAL_SERVER_ERROR,
                            "detail": error or "No result returned from upstream",
                        }
                    )
                    skip = True

        return StreamingResponse(
            _stream_response(), headers={"X-Chutes-InvocationID": parent_invocation_id}
        )

    # Non-streamed (which we actually do stream but we'll just return the first item)
    error = None
    response = None
    async for chunk in invoke(
        chute,
        current_user.user_id,
        selected_cord["path"],
        selected_cord["function"],
        stream,
        args,
        kwargs,
        targets,
        parent_invocation_id,
        metrics=metrics,
    ):
        if response:
            continue
        if chunk.startswith('data: {"result"'):
            result = json.loads(chunk[6:])["result"]
            if "bytes" in result:
                raw_data = BytesIO(base64.b64decode(result["bytes"].encode()))

                async def _streamfile():
                    yield raw_data.getvalue()

                response = StreamingResponse(
                    _streamfile(),
                    media_type=result["content_type"],
                    headers={"X-Chutes-InvocationID": parent_invocation_id},
                )
            elif "text" in result:
                response = Response(
                    content=result["text"],
                    media_type=result["content_type"],
                    headers={"X-Chutes-InvocationID": parent_invocation_id},
                )
            else:
                response = Response(
                    content=json.dumps(result.get("json", result)).decode(),
                    media_type="application/json",
                    headers={
                        "Content-type": "application/json",
                        "X-Chutes-InvocationID": parent_invocation_id,
                    },
                )
        elif chunk.startswith('data: {"error"'):
            error = json.loads(chunk[6:])["error"]
    if response:
        return response
    raise HTTPException(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        detail=error or "No result returned from upstream",
    )


@host_invocation_router.api_route(
    "{path:path}", methods=["GET", "POST", "PUT", "DELETE", "PATCH", "HEAD"]
)
async def hostname_invocation(
    request: Request,
    db: AsyncSession = Depends(get_db_session),
    current_user: User = Depends(get_current_user()),
):
    # MegaLLM handler.
    if request.state.chute_id == "__megallm__":
        payload = await request.json()
        model = payload.get("model")
        chute = None
        if model:
            chute = (
                (
                    await db.execute(
                        select(Chute).where(
                            Chute.user_id == await chutes_user_id(),
                            Chute.name == model,
                            Chute.public.is_(True),
                            Chute.standard_template == "vllm",
                        )
                    )
                )
                .unique()
                .scalar_one_or_none()
            )
        if not chute:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"model not found: {model}",
            )
        request.state.chute_id = chute.chute_id
        request.state.auth_object_id = chute.chute_id
    return await _invoke(request, db, current_user)
