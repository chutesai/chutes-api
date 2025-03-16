"""
Invocations router.
"""

import base64
import pickle
import gzip
import orjson as json
import csv
import uuid
import time
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
from api.constants import LLM_PRICE_MULT_PER_MILLION
from api.chute.util import (
    invoke,
    get_one,
    get_vllm_models,
    count_prompt_tokens,
    TRACK_INVOCATION,
    UPDATE_INVOCATION,
    selector_hourly_price,
)
from api.util import rate_limit, ip_rate_limit, sse, now_str
from api.user.schemas import User
from api.user.service import get_current_user
from api.report.schemas import Report, ReportArgs
from api.database import get_db_session, get_session
from api.instance.util import get_chute_target_manager
from api.invocation.util import get_prompt_prefix_hashes
from api.invocation.cache import (
    cached_responder,
    append_stream,
    purge_stream,
    set_stream_expiration,
)
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
    is_reports = False
    if hour_format.endswith(".csv"):
        hour_part = hour_format[:-4]
        if hour_part.endswith("-reports"):
            hour_str = hour_part[:-8]
            is_reports = True
        else:
            hour_str = hour_part
            is_reports = False
        try:
            hour = int(hour_str)
        except ValueError:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid hour format: {hour_format}",
            )
    else:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid format: {hour_format}, must end with .csv",
        )

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
    if is_reports:
        key = f"invocations/{year}/{month:02d}/{day:02d}/{hour:02d}-reports.csv"
    else:
        key = f"invocations/{year}/{month:02d}/{day:02d}/{hour:02d}.csv"

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
    current_user: User,
):
    # This call will perform auth/access checks.
    chute = await get_one(request.state.chute_id)
    if not chute or (not chute.public and chute.user_id != current_user.user_id):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="No matching chute found!"
        )

    # Check account balance.
    origin_ip = request.headers.get("x-forwarded-for", "").split(",")[0]
    is_paid_account = not current_user.has_role(Permissioning.free_account)
    if (
        current_user.balance <= 0
        and is_paid_account
        and (not chute.discount or chute.discount < 1.0)
    ):
        logger.warning(
            f"Payment required: attempted invocation of {chute.name} from user {current_user.username} [{origin_ip}] with no balance"
        )
        raise HTTPException(
            status_code=status.HTTP_402_PAYMENT_REQUIRED,
            detail=f"Account balance is ${current_user.balance}, please send tao to {current_user.payment_address}",
        )

    # Rate limits.
    limit = settings.rate_limit_count
    overrides = current_user.rate_limit_overrides or {}
    override = overrides.get(chute.chute_id, overrides.get("*"))
    if override:
        limit = override
    else:
        # Temporary fallback manual overrides.
        if current_user.user_id == "5682c3e0-3635-58f7-b7f5-694962450dfc":
            limit = int(limit * 10)
        if current_user.user_id == "2104acf4-999e-5452-84f1-de82de35a7e7":
            limit = int(limit * 2.5)
        # Allow extra capacity for the models not on OpenRouter.
        if not chute.openrouter:
            limit *= 2
        if is_paid_account:
            limit *= 3
    await rate_limit(chute.chute_id, current_user, limit, settings.rate_limit_window)

    # IP address rate limits.
    if (
        current_user.user_id != "5682c3e0-3635-58f7-b7f5-694962450dfc"
        and current_user.user_id != "2104acf4-999e-5452-84f1-de82de35a7e7"
        and not request.state.squad_request
        and not override
    ):
        await ip_rate_limit(
            current_user, origin_ip, settings.ip_rate_limit_count, settings.ip_rate_limit_window
        )

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

        # SGLang chute we use for R1 uses the default overlap scheduler which does not support
        # these penalty params, and sampling params are causing crashes.
        if chute.name in ("deepseek-ai/DeepSeek-R1", "deepseek-ai/DeepSeek-V3"):
            for param in [
                "frequency_penalty",
                "presence_penalty",
                "repetition_penalty",
                "min_p",
                "top_p",
                "top_k",
            ]:
                request_body.pop(param, None)
            if (max_tokens := request_body.get("max_tokens")) is not None:
                try:
                    max_tokens = int(request_body["max_tokens"])
                    if max_tokens > 4096:
                        request_body["max_tokens"] = 4096
                except ValueError:
                    request_body["max_tokens"] = 4096

        # Make sure the model name is correct.
        if (requested_model := request_body.get("model")) != chute.name:
            logger.warning(
                f"User requested model {requested_model} but chute name is: {chute.name}"
            )
            request_body["model"] = chute.name

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
        manager = await get_chute_target_manager(db, chute.chute_id, max_wait=60)
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

    # Ready to query the miners finally :)
    logger.info(
        f"Calling {selected_cord['path']} of {chute.name} with up to {len(manager.instances)} "
        f"targets on behalf of {current_user.username} [{origin_ip}]"
    )

    include_trace = request.headers.get("X-Chutes-Trace", "").lower() == "true"
    parent_invocation_id = str(uuid.uuid4())
    request_hash = None

    # Track unique requests.
    body_target = request_body
    if (
        chute.standard_template in ("vllm", "tei")
        or selected_cord.get("passthrough", False)
        and "json" in request_body
    ):
        body_target = request_body["json"]
    _request_hash = None
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
        _request_hash = str(uuid.uuid5(uuid.NAMESPACE_OID, request_hash_str)).replace("-", "")
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

        for _hash in (_request_hash, _prompt_hash):
            if not _hash:
                continue
            req_key = f"req:{_hash}".encode()
            value = await settings.memcache.get(req_key)
            if value is None:
                await settings.memcache.set(req_key, b"0")
            count = await settings.memcache.incr(req_key, 1)
            if count > 1 and _hash == _request_hash:
                logger.info(f"Duplicate prompt received: {chute.name} {_hash} {count=}")
    except Exception as exc:
        logger.warning(f"Error updating request hash tracking: {exc}")

    # Handle cacheable requests.
    enable_cache = request.headers.get("X-Enable-Cache")
    request_hash = None
    if (
        (
            enable_cache
            or (
                current_user.user_id != "5682c3e0-3635-58f7-b7f5-694962450dfc"
                and not chute.openrouter
                and not request.headers.get("X-Disable-Cache")
            )
        )
        and metrics
        and "ttft" in metrics
    ):
        request_hash = _request_hash
        started_at = time.time()
        if (streamer := await cached_responder(request_hash, chute.name)) is not None:

            async def _send_from_cached_stream():
                invocation_id = str(uuid.uuid4())
                if include_trace:
                    yield sse(
                        {
                            "trace": {
                                "timestamp": now_str(),
                                "invocation_id": parent_invocation_id,
                                "chute_id": chute.chute_id,
                                "function": selected_cord["function"],
                                "message": f"responding from cache, {request_hash=}",
                            },
                        }
                    )
                any_chunk = False
                last_chunk = None
                async for chunk in streamer:
                    if not chunk:
                        continue
                    if include_trace:
                        yield {"result": chunk}
                    else:
                        yield chunk
                    if chunk.startswith(b"data: {"):
                        if metrics.get("ttft") is None:
                            metrics["ttft"] = time.time() - started_at
                        last_chunk = chunk
                    elif b"data:" in chunk:
                        any_chunk = True

                # Parse usage data.
                total_time = time.time() - started_at
                if last_chunk and b'"usage"' in last_chunk:
                    try:
                        usage_obj = json.loads(last_chunk[6:].decode())
                        usage = usage_obj.get("usage", {})
                        prompt_tokens = usage.get("prompt_tokens")
                        completion_tokens = usage.get("completion_tokens")
                        if prompt_tokens is not None and completion_tokens is not None:
                            metrics["it"] = max(0, prompt_tokens or 0)
                            metrics["ot"] = max(0, completion_tokens or 0)
                            metrics["tokens"] = metrics["ot"]
                            metrics["tps"] = metrics["ot"] / total_time
                            logger.info(f"LLMCACHE: metrics for chute={chute.name}: {metrics}")

                            if metrics["ot"] and not request.state.free_invocation:
                                tokens = metrics["ot"] + metrics["it"]
                                hourly_price = await selector_hourly_price(chute.node_selector)
                                discount = 0.0
                                if chute.discount and -3 < chute.discount <= 1:
                                    discount = chute.discount
                                balance_used = (
                                    tokens / 1000000.0 * hourly_price * LLM_PRICE_MULT_PER_MILLION
                                )
                                balance_used -= balance_used * discount
                                logger.info(
                                    f"LLMCACHE BALANCE: LLM token pricing: ${hourly_price * LLM_PRICE_MULT_PER_MILLION:.4f}/million for {chute.name}, {balance_used=} for {tokens=} {discount=}"
                                )
                                await set_stream_expiration(request_hash)

                                # User account balance deductions.
                                pipeline = settings.cm_redis_client.pipeline()
                                key = f"balance:{current_user.user_id}:{chute.chute_id}"
                                pipeline.hincrbyfloat(key, "amount", balance_used)
                                pipeline.hincrby(key, "count", 1)
                                pipeline.hset(key, "timestamp", int(time.time()))
                                await pipeline.execute()
                                logger.info(
                                    f"LLMCACHE Deducted (soon) ${balance_used:.12f} from {current_user.user_id=}"
                                )

                    except Exception as exc:
                        logger.warning(f"Error checking metrics: {exc}")

                if any_chunk:
                    # Track an invocation, just set the UID to < 0.
                    async with get_session() as session:
                        invocation_id = str(uuid.uuid4())
                        result = await session.execute(
                            TRACK_INVOCATION,
                            {
                                "parent_invocation_id": parent_invocation_id,
                                "invocation_id": invocation_id,
                                "function_name": selected_cord["function"],
                                "chute_id": chute.chute_id,
                                "chute_user_id": chute.user_id,
                                "user_id": current_user.user_id,
                                "image_id": chute.image_id,
                                "image_user_id": chute.image.user_id,
                                "instance_id": "00000000-0000-0000-0000-000000000000",
                                "miner_uid": -1,
                                "miner_hotkey": "00000000-0000-0000-0000-000000000000",
                                "request_path": selected_cord["path"],
                                "compute_multiplier": 0.0,
                            },
                        )
                        partition_suffix = result.scalar()
                        await session.execute(
                            text(UPDATE_INVOCATION.format(suffix=partition_suffix)),
                            {
                                "chute_id": chute.chute_id,
                                "invocation_id": invocation_id,
                                "response_path": None,
                                "metrics": json.dumps(metrics).decode(),
                            },
                        )
                        await session.commit()

            return StreamingResponse(
                _send_from_cached_stream(),
                media_type="text/event-stream",
                headers={"X-Chutes-InvocationID": parent_invocation_id},
            )

    if stream or include_trace:

        async def _stream_response():
            try:
                skip = False
                async for chunk in invoke(
                    chute,
                    current_user.user_id,
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
                    if include_trace:
                        yield chunk
                        continue
                    if skip:
                        continue
                    if chunk.startswith('data: {"result"'):
                        result_val = json.loads(chunk[6:])["result"]
                        yield result_val
                        if request_hash:
                            await append_stream(request_hash, result_val.encode())
                    elif chunk.startswith('data: {"error"'):
                        error = json.loads(chunk[6:])["error"]
                        yield json.dumps(
                            {
                                "status": status.HTTP_500_INTERNAL_SERVER_ERROR,
                                "detail": error or "No result returned from upstream",
                            }
                        )
                        skip = True
                        if request_hash:
                            await purge_stream(request_hash)
                if request_hash:
                    await append_stream(request_hash, b"[[__END__]]")
                    await set_stream_expiration(request_hash)

            except Exception:
                if request_hash:
                    await purge_stream(request_hash)
                raise

        return StreamingResponse(
            _stream_response(),
            media_type="text/event-stream",
            headers={"X-Chutes-InvocationID": parent_invocation_id},
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
    current_user: User = Depends(get_current_user(raise_not_found=False)),
):
    # /v1/models endpoint for llm.chutes.ai is handled differently.
    if (
        request.state.chute_id == "__megallm__"
        and request.url.path == "/v1/models"
        and request.method.lower() == "get"
    ):
        return await get_vllm_models(request)

    # The /v1/models endpoint can be checked with no auth, but otherwise we need users.
    if not current_user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required.",
        )

    # Mega LLM/diffusion request handler.
    if request.state.chute_id in ("__megallm__", "__megadiffuser__"):
        payload = await request.json()
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
                not chute.public and chute.user_id != current_user.user_id
            ):
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"model not found: {model}",
                )
            request.state.chute_id = chute.chute_id
            request.state.auth_object_id = chute.chute_id
    return await _invoke(request, current_user)
