"""
Invocations router.
"""

import re
import base64
import pickle
import gzip
import orjson as json
import csv
import uuid
import decimal
import traceback
from loguru import logger
from pydantic import BaseModel, ValidationError, Field
from datetime import date, datetime
from io import BytesIO, StringIO
from typing import Optional
from fastapi_cache.decorator import cache
from fastapi import APIRouter, Depends, HTTPException, status, Request, Response
from starlette.responses import StreamingResponse
from sqlalchemy import text, select
from sqlalchemy.ext.asyncio import AsyncSession
from api.config import settings
from api.chute.util import (
    invoke,
    get_one,
    get_llms,
    is_shared,
    count_prompt_tokens,
)
from api.user.schemas import User, InvocationQuota
from api.user.service import get_current_user
from api.report.schemas import Report, ReportArgs
from api.database import get_db_session, get_session, get_db_ro_session
from api.instance.util import get_chute_target_manager
from api.invocation.util import get_prompt_prefix_hashes
from api.util import check_vlm_payload
from api.permissions import Permissioning

router = APIRouter()
host_invocation_router = APIRouter()


class DiffusionInput(BaseModel):
    prompt: str
    negative_prompt: str = ""
    height: int = Field(default=1024, ge=128, le=2048)
    width: int = Field(default=1024, ge=128, le=2048)
    num_inference_steps: int = Field(default=25, ge=1, le=50)
    guidance_scale: float = Field(default=7.5, ge=1.0, le=20.0)
    seed: Optional[int] = Field(default=None, ge=0, le=2**32 - 1)
    img_guidance_scale: float = Field(default=7.5, ge=1.0, le=20.0)
    image_b64: Optional[list[str]] = Field(
        default=None, description="Base64 encoded images for image-to-image pipelines."
    )

    class Config:
        extra = "forbid"


@router.get("/usage")
async def get_usage(request: Request):
    """
    Get aggregated usage data, which is the amount of revenue
    we would be receiving if no usage was free.
    """
    cache_key = b"invocation_usage_data"
    if request:
        if cached := await settings.memcache.get(cache_key):
            return json.loads(cached)
    query = text(
        "SELECT chute_id, DATE(bucket) as date, sum(amount) as usd_amount, sum(count) as invocation_count "
        "from usage_data "
        "where bucket >= now() - interval '11 days' "
        "group by chute_id, date "
        "order by date desc, usd_amount desc"
    )
    async with get_session() as session:
        result = await session.execute(query)
        rv = []
        for chute_id, date, usd_amount, invocation_count in result:
            rv.append(
                {
                    "chute_id": chute_id,
                    "date": date,
                    "usd_amount": float(usd_amount),
                    "invocation_count": int(invocation_count),
                }
            )
        await settings.memcache.set(cache_key, json.dumps(rv))
        return rv


async def _cached_get_metrics(table, cache_key):
    if cached := await settings.memcache.get(cache_key):
        return json.loads(gzip.decompress(base64.b64decode(cached)))
    async with get_session() as session:
        result = await session.execute(text(f"SELECT * FROM {table}"))
        rows = result.mappings().all()
        rv = [dict(row) for row in rows]
        for row in rv:
            for key, value in row.items():
                if isinstance(value, decimal.Decimal):
                    row[key] = float(value)
        cache_value = base64.b64encode(gzip.compress(json.dumps(rv)))
        await settings.memcache.set(cache_key, cache_value, exptime=300)
        return rv


@router.get("/stats/llm")
async def get_llm_stats():
    return await _cached_get_metrics("vllm_metrics", b"llmstats")


@router.get("/stats/diffusion")
async def get_diffusion_stats():
    return await _cached_get_metrics("diffusion_metrics", b"diffstats")


@router.get("/exports/{year}/{month}/{day}/{hour_format}")
async def get_export(
    year: int,
    month: int,
    day: int,
    hour_format: str,
) -> Response:
    """
    Get invocation exports (and reports) for a particular hour.
    """
    format_match = re.match(r"^(\d+)((?:-(reports|jobs))?\.csv)$", hour_format)
    if not format_match:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail=f"Invalid format: {hour_format}"
        )
    hour = int(format_match.group(1))
    suffix = format_match.group(2)

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

    # Construct the S3 key based on whether this is a reports request
    key = f"invocations/{year}/{month:02d}/{day:02d}/{hour:02d}{suffix}"

    # Check if the file exists
    exists = False
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

    # Download and return the file.
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
    db: AsyncSession = Depends(get_db_ro_session),
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
            bounty,
            metrics
        FROM partitioned_invocations
        WHERE started_at >= CURRENT_TIMESTAMP - INTERVAL '1 day'
        AND completed_at IS NOT NULL
        AND error_message IS NULL
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
    current_user: User,
):
    # Check if the user has access.
    chute = await get_one(request.state.chute_id)
    if not chute or (not chute.public and chute.user_id != current_user.user_id):
        if not chute or not await is_shared(chute.chute_id, current_user.user_id):
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND, detail="No matching chute found!"
            )
    if chute.discount == 1.0:
        request.state.free_invocation = True

    # Check account balance.
    origin_ip = request.headers.get("x-forwarded-for", "").split(",")[0]

    # Check account quotas if not free/invoiced.
    if not (
        current_user.has_role(Permissioning.free_account)
        or current_user.has_role(Permissioning.invoice_billing)
        or request.state.free_invocation
    ):
        quota = await InvocationQuota.get(current_user.user_id, chute.chute_id)
        key = await InvocationQuota.quota_key(current_user.user_id, chute.chute_id)
        cached = await settings.quota_client.get(key)
        request_count = 0.0
        if cached:
            try:
                request_count = float(cached.decode())
            except ValueError:
                await settings.quota_client.delete(key)
        else:
            # Initialize the quota key with an expiration date (keys are daily)
            pipe = settings.quota_client.pipeline()
            pipe.incrbyfloat(key, 0.0)
            pipe.expire(key, 25 * 60 * 60)
            await pipe.execute()

        # Automatically switch to paygo when the quota is exceeded.
        if request_count >= quota:
            if current_user.balance <= 0 and not request.state.free_invocation:
                logger.warning(
                    f"Payment required: attempted invocation of {chute.name} "
                    f"from user {current_user.username} [{origin_ip}] with no balance "
                    f"and {request_count=} of {quota=}"
                )
                raise HTTPException(
                    status_code=status.HTTP_402_PAYMENT_REQUIRED,
                    detail=(
                        f"Quota exceeded and account balance is ${current_user.balance}, "
                        f"please pay with fiat or send tao to {current_user.payment_address}"
                    ),
                )
        else:
            # When within the quota, mark the invocation as "free" so no balance is deducted when finished.
            request.state.free_invocation = True

    # Identify the cord that we'll trying to access by the public API path and method.
    selected_cord = None
    request_body = await request.json() if request.method in ("POST", "PUT", "PATCH") else {}
    request_params = request.query_params._dict if request.query_params else {}
    stream = request_body.get("stream", request_params.get("stream", False))
    for cord in chute.cords:
        public_path = cord.get("public_api_path", None)
        if public_path and public_path == request.url.path:
            if cord.get("public_api_method", "POST") == request.method:
                if chute.standard_template != "vllm" or stream == cord.get("stream"):
                    selected_cord = cord
                    if cord.get("stream"):
                        stream = True
                    break
    if not selected_cord:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="No matching cord found!")

    # Wrap up the args/kwargs in the way the miner execution service expects them.
    args, kwargs = None, None
    prefix_hashes = None
    if chute.standard_template == "diffusion":
        request_body.pop("cord", None)
        request_body.pop("method", None)
        request_body.pop("model", None)
        steps = request_body.get("num_inference_steps")
        max_steps = 30 if chute.name == "FLUX-1.dev" else 50
        if steps and (isinstance(steps, int) or steps.isdigit()) and int(steps) > max_steps:
            request_body["num_inference_steps"] = int(max_steps)
        try:
            _ = DiffusionInput(**request_body)
        except ValidationError:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST, detail="bad request, naughty naughty"
            )
    elif chute.standard_template == "vllm":
        # Force usage metrics.
        if request_body.get("stream"):
            if "stream_options" not in request_body:
                request_body["stream_options"] = {}
            if not request_body["stream_options"].get("include_usage"):
                request_body["stream_options"]["include_usage"] = True
        if request_body.get("logprobs"):
            if not request_body.get("top_logprobs"):
                request_body["top_logprobs"] = 1

        # Custom temp for Dolphin.
        if chute.name in (
            "cognitivecomputations/Dolphin3.0-R1-Mistral-24B",
            "cognitivecomputations/Dolphin3.0-Mistral-24B",
        ):
            if "temperature" not in request_body:
                request_body["temperature"] = 0.05

        # Make sure the model name is correct.
        if (requested_model := request_body.get("model")) != chute.name:
            logger.warning(
                f"User requested model {requested_model} but chute name is: {chute.name}"
            )
            request_body["model"] = chute.name

        # Images must be https or base64.
        try:
            check_vlm_payload(request_body)
        except Exception as exc:
            if isinstance(exc, HTTPException):
                raise
            logger.error(f"Failed to check VLM request payload: {exc}")

        # Load prompt prefixes so we can do more intelligent routing.
        prefix_hashes = get_prompt_prefix_hashes(request_body)

    if chute.standard_template in ("vllm", "tei") or selected_cord.get("passthrough", False):
        request_body = {"json": request_body, "params": request_params}
        args = base64.b64encode(gzip.compress(pickle.dumps(tuple()))).decode()
        kwargs = base64.b64encode(gzip.compress(pickle.dumps(request_body))).decode()
    else:
        args = base64.b64encode(gzip.compress(pickle.dumps((request_body,)))).decode()
        kwargs = base64.b64encode(gzip.compress(pickle.dumps({}))).decode()
    async with get_session() as db:
        manager = await get_chute_target_manager(db, chute.chute_id, max_wait=0)
    if not manager or not manager.instances:
        chute_id = request.state.chute_id
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"No instances available (yet) for {chute_id=}",
        )

    # Initialize metrics.
    metrics = None
    if chute.standard_template == "vllm":
        if request.url.path.lstrip("/").startswith(("v1/chat", "v1/completion")):
            metrics = {
                "ttft": None,
                "tps": 0.0,
                "tokens": 0,
                "it": await count_prompt_tokens(request_body),
                "ot": 0,
            }
    elif chute.standard_template == "diffusion":
        steps = request_body.get("num_inference_steps", 25)
        if not isinstance(steps, int):
            try:
                steps = int(steps)
            except ValueError:
                steps = 25
        request_body["num_inference_steps"] = steps
        metrics = {
            "sps": 0,
            "steps": steps,
        }

    include_trace = request.headers.get("X-Chutes-Trace", "").lower() == "true"
    parent_invocation_id = str(uuid.uuid4())

    # Track unique requests.
    body_target = request_body
    if (
        chute.standard_template in ("vllm", "tei")
        or selected_cord.get("passthrough", False)
        and "json" in request_body
    ):
        body_target = request_body["json"]
    request_hash = None
    user_dupe_count = 0
    total_dupe_count = 0
    reroll = False
    try:
        raw_dump = json.dumps(body_target).decode()
        prompt_dump = None
        if "messages" in body_target:
            try:
                prompt_dump = "::".join(
                    [f"{m['role']}: {m['content']}" for m in body_target["messages"]]
                )
            except Exception as exc:
                logger.warning(f"Error generating prompt key for dupe tracking: {exc}")
        elif "prompt" in body_target and isinstance(body_target, str):
            prompt_dump = body_target["prompt"]
        request_hash_str = "::".join(
            [
                chute.name,
                request.url.path,
                raw_dump,
            ]
        ).encode()
        request_hash = str(uuid.uuid5(uuid.NAMESPACE_OID, request_hash_str)).replace("-", "")
        _prompt_hash = None
        if prompt_dump:
            prompt_hash_str = "::".join(
                [
                    chute.name,
                    request.url.path,
                    prompt_dump,
                ]
            ).encode()
            _prompt_hash = str(uuid.uuid5(uuid.NAMESPACE_OID, prompt_hash_str)).replace("-", "")

            # Check for rerolls, which are cheaper.
            rr_key = f"userreq:{current_user.user_id}{_prompt_hash}".encode()
            value = await settings.memcache.get(rr_key)
            if value is None:
                await settings.memcache.set(rr_key, b"0")
            count = await settings.memcache.incr(rr_key, 1)
            await settings.memcache.touch(rr_key, 60 * 60 * 3)
            if count > 1:
                reroll = True

        for _hash in (request_hash, _prompt_hash):
            if not _hash:
                continue
            req_key = f"req:{_hash}".encode()
            value = await settings.memcache.get(req_key)
            if value is None:
                await settings.memcache.set(req_key, b"0")
            count = await settings.memcache.incr(req_key, 1)
            await settings.memcache.touch(req_key, exptime=60 * 60 * 3)
            if count > 1 and _hash == request_hash:
                total_dupe_count = count

                # Check for user specific spam.
                ureq_key = f"userreq:{current_user.user_id}{_hash}".encode()
                value = await settings.memcache.get(ureq_key)
                if value is None:
                    await settings.memcache.set(ureq_key, b"0")
                user_dupe_count = await settings.memcache.incr(ureq_key, 1)
                await settings.memcache.touch(ureq_key, 60 * 60 * 3)

    except Exception as exc:
        logger.warning(f"Error updating request hash tracking: {exc}")

    # Handle cacheable requests.
    if total_dupe_count >= 1500:
        logger.warning(f"REQSPAM: {total_dupe_count=} for {request_hash=} on {chute.name=}")

    # And user spam.
    if (
        user_dupe_count >= 1000
        and not current_user.has_role(Permissioning.unlimited)
        and current_user.user_id
        not in [
            "8930c58d-00f6-57d3-bc62-156eb8b73026",
            "dff3e6bb-3a6b-5a2b-9c48-da3abcd5ca5f",
            "376536e8-674b-5e6f-b36e-c9168f0bf4a7",
            "b6bb1347-6ea5-556f-8b23-50b124f3ffc8",
            "5682c3e0-3635-58f7-b7f5-694962450dfc",
            "2104acf4-999e-5452-84f1-de82de35a7e7",
            "18c244ab-8a2e-5767-ae0e-5d20b50d05b5",
            "90fd1e31-84c9-5bc4-b628-ccc1e5dc75e6",
            "596a9bd6-2904-546a-a3d7-e2c5b271427b",
        ]
    ):
        logger.warning(
            f"USERSPAM: {current_user.username} sent {user_dupe_count} requests for {chute.name}"
        )
        if user_dupe_count > 5000:
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail="Stop spamming this prompt, please...",
            )

    # Handle streaming responses, either because the user asked for X-Chutes-Trace,
    # or in the case of LLMs with stream: true in request.
    if stream or include_trace:
        # We have to wait until we have the first chuck to determine whether or not we
        # should return a successful response, otherwise for example we could return
        # a 200 status but actually be overwhelmed and it should be a 429/503/etc.
        async def _buffered_stream_response():
            first_chunk_processed = False
            buffered_chunks = []

            try:
                async for chunk in invoke(
                    chute,
                    current_user,
                    selected_cord["path"],
                    selected_cord["function"],
                    stream,
                    args,
                    kwargs,
                    manager,
                    parent_invocation_id,
                    metrics=metrics,
                    request=request,
                    prefixes=prefix_hashes,
                    reroll=reroll,
                ):
                    if include_trace:
                        if not first_chunk_processed:
                            first_chunk_processed = True
                        yield chunk
                        continue

                    # Handle errors.
                    if chunk.startswith('data: {"error"'):
                        chunk_data = json.loads(chunk[6:])
                        error = chunk_data["error"]

                        # If the error occurred on the first chunk, we can raise an HTTP exception.
                        if not first_chunk_processed:
                            if error in ("infra_overload", "no_targets"):
                                raise HTTPException(
                                    status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                                    detail=chunk_data.get("detail") or error,
                                )
                            elif error == "bad_request":
                                raise HTTPException(
                                    status_code=status.HTTP_400_BAD_REQUEST,
                                    detail=chunk_data.get("detail") or error,
                                )
                            raise HTTPException(
                                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                                detail=error,
                            )
                        else:
                            # If we've already started streaming, at this point we can't change the response
                            # headers so we need to just include an error string in the stream.
                            yield json.dumps(
                                {
                                    "error": f"Unhandled exception during response stream: {error}",
                                }
                            )
                            return

                    # Normal result chunks.
                    elif chunk.startswith('data: {"result"'):
                        result_val = json.loads(chunk[6:])["result"]
                        if not first_chunk_processed:
                            first_chunk_processed = True
                            for buffered_chunk in buffered_chunks:
                                yield buffered_chunk
                            buffered_chunks = []
                        yield result_val

            except Exception as e:
                logger.error(f"Failed here: {e}\n{traceback.format_exc()}")
                if not first_chunk_processed:
                    if isinstance(e, HTTPException):
                        raise e
                    else:
                        raise HTTPException(
                            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e)
                        )
                else:
                    yield json.dumps(
                        {
                            "error": f"Unhandled exception during response stream: {e}",
                        }
                    )

        # Create the response generator, but wait for the first chunk before returning
        # the StreamingResponse object so we don't incorrectly give a 200 response
        # for failed requests.
        try:
            generator = _buffered_stream_response()
            first_chunk = await generator.__anext__()

            async def _stream_with_first_chunk():
                yield first_chunk
                async for chunk in generator:
                    yield chunk

            return StreamingResponse(
                _stream_with_first_chunk(),
                media_type="text/event-stream",
                headers={"X-Chutes-InvocationID": parent_invocation_id},
            )

        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Unhandled error generating response: {e}",
            )

    # Non-streamed (which we actually do stream but we'll just return the first item)
    error = None
    response = None
    async for chunk in invoke(
        chute,
        current_user,
        selected_cord["path"],
        selected_cord["function"],
        stream,
        args,
        kwargs,
        manager,
        parent_invocation_id,
        metrics=metrics,
        request=request,
        prefixes=prefix_hashes,
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
            chunk_data = json.loads(chunk[6:])
            error = chunk_data["error"]
            if error in ("infra_overload", "no_targets"):
                raise HTTPException(
                    status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                    detail=chunk_data.get("detail") or error,
                )
            elif error == "bad_request":
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=chunk_data.get("detail") or error,
                )
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=error,
            )

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
    current_user: User = Depends(get_current_user(raise_not_found=False)),
):
    # /v1/models endpoint for llm.chutes.ai is handled differently.
    if (
        request.state.chute_id == "__megallm__"
        and request.url.path == "/v1/models"
        and request.method.lower() == "get"
    ):
        return await get_llms()

    # The /v1/models endpoint can be checked with no auth, but otherwise we need users.
    if not current_user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required.",
        )

    # Mega LLM/diffusion request handler.
    if request.state.chute_id in ("__megallm__", "__megadiffuser__"):
        payload = await request.json()
        # MistralAI gated this model for some reason.......
        model = payload.get("model")
        if model == "mistralai/Mistral-Small-3.1-24B-Instruct-2503":
            payload["model"] = "chutesai/Mistral-Small-3.1-24B-Instruct-2503"

        # # Disable tools on kimi-k2 for now.
        # elif model == "moonshotai/Kimi-K2-Instruct":
        #     problematic = set(payload) & set(
        #         [
        #             "logit_bias",
        #             "response_format",
        #             "tools",
        #             "tool_choice",
        #             "regex",
        #             "grammar",
        #         ]
        #     )
        #     if problematic:
        #         raise HTTPException(
        #             status_code=status.HTTP_400_BAD_REQUEST,
        #             detail=f"{model} on chutes does not currently support {problematic}",
        #         )

        model = payload.get("model")
        chute = None
        template = "vllm" if "llm" in request.state.chute_id else "diffusion"
        if model:
            if (chute := await get_one(model)) is None:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"model not found: {model}",
                )
            if chute.standard_template != template or (
                not chute.public
                and (
                    chute.user_id != current_user.user_id
                    and not await is_shared(chute.chute_id, current_user.user_id)
                )
            ):
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"model not found: {model}",
                )
            request.state.chute_id = chute.chute_id
            request.state.auth_object_id = chute.chute_id

    # Model disabled temporarily?
    if await settings.redis_client.get(f"model_disabled:{request.state.chute_id}"):
        logger.warning(f"MODEL DISABLED: {request.state.chute_id}")
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="model is under maintenance",
        )

    return await _invoke(request, current_user)
