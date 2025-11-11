"""
Application logic and utilities for chutes.
"""

import os
import aiohttp
import asyncio
import re
import uuid
import io
import time
import traceback
import orjson as json
import base64
import gzip
import math
import pickle
import random
from async_lru import alru_cache
from fastapi import Request, status
from loguru import logger
from transformers import AutoTokenizer
from typing import Optional
from sqlalchemy import and_, or_, text, exists, func
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.future import select
from sqlalchemy.orm import selectinload
from api.config import settings
from api.constants import (
    LLM_MIN_PRICE_IN,
    LLM_MIN_PRICE_OUT,
    LLM_PRICE_MULT_PER_MILLION_IN,
    LLM_PRICE_MULT_PER_MILLION_OUT,
    DIFFUSION_PRICE_MULT_PER_STEP,
)
from api.database import get_session, get_inv_session
from api.fmv.fetcher import get_fetcher
from api.exceptions import (
    InstanceRateLimit,
    BadRequest,
    KeyExchangeRequired,
    EmptyLLMResponse,
    InvalidResponse,
    InvalidCLLMV,
)
from api.util import (
    sse,
    now_str,
    semcomp,
    aes_encrypt,
    aes_decrypt,
    use_encryption_v2,
    use_encrypted_path,
    notify_deleted,
    image_supports_cllmv,
    has_legacy_private_billing,
)
from api.util import memcache_get, memcache_set, memcache_delete
from api.chute.schemas import Chute, NodeSelector, ChuteShare, LLMDetail
from api.user.schemas import User, InvocationQuota, InvocationDiscount, PriceOverride
from api.user.service import chutes_user_id
from api.miner_client import sign_request
from api.instance.schemas import Instance
from api.instance.util import LeastConnManager, update_shutdown_timestamp, invalidate_instance_cache
from api.gpu import COMPUTE_UNIT_PRICE_BASIS
from api.metrics.vllm import track_usage as track_vllm_usage
from api.metrics.perf import PERF_TRACKER
from api.metrics.capacity import (
    track_capacity,
    track_request_completed,
    track_request_rate_limited,
)
from cllmv import validate as cllmv_validate


# Tokenizer for input/output token estimation.
TOKENIZER = AutoTokenizer.from_pretrained(
    os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "..",
        "..",
        "tokenizer",
    )
)

REQUEST_SAMPLE_RATIO = 0.05
LLM_PATHS = {"chat_stream", "completion_stream", "chat", "completion"}

BASE_UNIFIED_INVOCATION_INSERT = """
INSERT INTO {table_name} (
    bounty,
    parent_invocation_id,
    invocation_id,
    chute_id,
    chute_user_id,
    function_name,
    user_id,
    image_id,
    image_user_id,
    instance_id,
    miner_uid,
    miner_hotkey,
    started_at,
    completed_at,
    error_message,
    compute_multiplier,
    metrics
) VALUES (
    0,
    :parent_invocation_id,
    :invocation_id,
    :chute_id,
    :chute_user_id,
    :function_name,
    :user_id,
    :image_id,
    :image_user_id,
    :instance_id,
    :miner_uid,
    :miner_hotkey,
    CURRENT_TIMESTAMP - make_interval(secs => :duration),
    CURRENT_TIMESTAMP,
    :error_message,
    :compute_multiplier,
    :metrics
)
"""
UNIFIED_INVOCATION_RV = """
RETURNING CEIL(EXTRACT(EPOCH FROM (completed_at - started_at))) * compute_multiplier AS total_compute_units
"""

UNIFIED_INVOCATION_INSERT_LEGACY = text(
    f"""{BASE_UNIFIED_INVOCATION_INSERT}
{UNIFIED_INVOCATION_RV}""".format(table_name="partitioned_invocations")
)
UNIFIED_INVOCATION_INSERT = text(
    f"""{BASE_UNIFIED_INVOCATION_INSERT}
ON CONFLICT (invocation_id, started_at)
    DO UPDATE SET invocation_id = EXCLUDED.invocation_id
{UNIFIED_INVOCATION_RV}""".format(table_name="invocations")
)


async def store_invocation(
    parent_invocation_id: str,
    invocation_id: str,
    chute_id: str,
    chute_user_id: str,
    function_name: str,
    user_id: str,
    image_id: str,
    image_user_id: str,
    instance_id: str,
    miner_uid: int,
    miner_hotkey: str,
    duration: float,
    compute_multiplier: float,
    error_message: Optional[str] = None,
    metrics: Optional[dict] = {},
    legacy: Optional[bool] = False,
):
    session_method = get_session if legacy else get_inv_session
    insert_sql = UNIFIED_INVOCATION_INSERT_LEGACY if legacy else UNIFIED_INVOCATION_INSERT
    async with session_method() as session:
        async with session.begin():
            result = await session.execute(
                insert_sql,
                {
                    "parent_invocation_id": parent_invocation_id,
                    "invocation_id": invocation_id,
                    "chute_id": chute_id,
                    "chute_user_id": chute_user_id,
                    "function_name": function_name,
                    "user_id": user_id,
                    "image_id": image_id,
                    "image_user_id": image_user_id,
                    "instance_id": instance_id,
                    "miner_uid": miner_uid,
                    "miner_hotkey": miner_hotkey,
                    "duration": duration,
                    "error_message": error_message,
                    "compute_multiplier": compute_multiplier,
                    "metrics": json.dumps(metrics).decode(),
                },
            )
            row = result.first()
            return row


async def safe_store_invocation(*args, **kwargs):
    try:
        await store_invocation(*args, **kwargs)
    except Exception as exc:
        logger.error(f"SAFE_STORE_INVOCATION: failed to insert new invocation record: {str(exc)}")


async def get_miner_session(instance: Instance, timeout: int = 600) -> aiohttp.ClientSession:
    """
    Get or create an aiohttp session for an instance.
    """
    return aiohttp.ClientSession(
        base_url=f"http://{instance.host}:{instance.port}",
        timeout=aiohttp.ClientTimeout(connect=5.0, total=timeout),
        read_bufsize=8 * 1024 * 1024,
    )


async def selector_hourly_price(node_selector) -> float:
    """
    Helper to quickly get the hourly price of a node selector, caching for subsequent calls.
    """
    node_selector = (
        NodeSelector(**node_selector) if isinstance(node_selector, dict) else node_selector
    )
    price = await node_selector.current_estimated_price()
    return price["usd"]["hour"]


async def get_chute_by_id_or_name(chute_id_or_name, db, current_user, load_instances: bool = False):
    """
    Helper to load a chute by ID or full chute name (optional username/chute name)
    """
    if not chute_id_or_name:
        return None

    name_match = re.match(
        r"/?(?:([a-zA-Z0-9_\.-]{3,15})/)?([a-z0-9][a-z0-9_\.\/-]*)$",
        chute_id_or_name.lstrip("/"),
        re.I,
    )
    if not name_match:
        return None
    query = select(Chute).join(User, Chute.user_id == User.user_id)

    # Perms check.
    if current_user:
        query = query.outerjoin(
            ChuteShare,
            and_(
                ChuteShare.chute_id == Chute.chute_id, ChuteShare.shared_to == current_user.user_id
            ),
        ).where(
            or_(
                Chute.public.is_(True),
                Chute.user_id == current_user.user_id,
                ChuteShare.shared_to == current_user.user_id,
                Chute.name.ilike("%/affine%"),
            )
        )
    else:
        query = query.where(or_(Chute.public.is_(True), Chute.name.ilike("%/affine%")))

    if load_instances:
        query = query.options(selectinload(Chute.instances))

    username = name_match.group(1)
    chute_name = name_match.group(2)
    chute_id_or_name = chute_id_or_name.lstrip("/")
    if not username and current_user:
        username = current_user.username

    conditions = []
    conditions.append(Chute.chute_id == chute_id_or_name)
    conditions.append(Chute.name.ilike(chute_id_or_name))
    conditions.append(Chute.name.ilike(chute_name))

    # User specific lookups.
    if current_user:
        conditions.extend(
            [
                and_(
                    User.username == current_user.username,
                    Chute.name.ilike(chute_name),
                ),
                and_(
                    User.username == current_user.username,
                    Chute.name.ilike(chute_id_or_name),
                ),
            ]
        )

    # Username/chute_name lookup (if username provided or defaulted)
    if username:
        conditions.append(
            and_(
                User.username == username,
                Chute.name.ilike(chute_name),
            )
        )

    # Public chute lookups by name/ID only
    conditions.extend(
        [
            and_(
                Chute.name.ilike(chute_id_or_name),
                Chute.public.is_(True),
            ),
            and_(
                Chute.name.ilike(chute_name),
                Chute.public.is_(True),
            ),
            and_(
                Chute.chute_id == chute_id_or_name,
                Chute.public.is_(True),
            ),
        ]
    )
    query = query.where(or_(*conditions))
    user_sort_id = current_user.user_id if current_user else await chutes_user_id()
    query = query.order_by((Chute.user_id == user_sort_id).desc()).limit(1)
    result = await db.execute(query)
    return result.unique().scalar_one_or_none()


@alru_cache(maxsize=100, ttl=30)
async def chute_id_by_slug(slug: str):
    """
    Check if a chute exists with the specified slug (which is a subdomain for standard apps).
    """
    async with get_session() as session:
        if chute_id := (
            await session.execute(select(Chute.chute_id).where(Chute.slug == slug))
        ).scalar_one_or_none():
            return chute_id
    return None


@alru_cache(maxsize=1000, ttl=180)
async def _get_one(name_or_id: str, nonce: int = None):
    """
    Load a chute by it's name or ID.
    """
    # Memcached lookup first.
    cache_key = f"_chute:{name_or_id}".encode()
    cached = await memcache_get(cache_key)
    if cached:
        try:
            return pickle.loads(cached)
        except Exception:
            await memcache_delete(cache_key)

    # Load from DB.
    chute_user = await chutes_user_id()
    async with get_session() as db:
        chute = (
            (
                await db.execute(
                    select(Chute)
                    .where(
                        or_(
                            Chute.name == name_or_id,
                            Chute.chute_id == name_or_id,
                        )
                    )
                    .order_by((Chute.user_id == chute_user).desc())
                    .limit(1)
                )
            )
            .unique()
            .scalar_one_or_none()
        )
        if chute:
            # Warm up the relationships for the serialization.
            _ = chute.image
            _ = chute.logo
            _ = chute.rolling_update
            _ = chute.user
            if chute.user:
                _ = chute.user.current_balance
            if chute.image:
                _ = chute.image.user
                _ = chute.image.logo
            serialized = pickle.dumps(chute)
            await memcache_set(cache_key, serialized, exptime=180)
        return chute


async def get_one(name_or_id: str, nonce: int = None):
    """
    Wrapper around the actual cached get_one with 30 second nonce to force refresh.
    """
    if not nonce:
        nonce = int(time.time())
        nonce -= nonce % 30
    return await _get_one(name_or_id, nonce=nonce)


@alru_cache(maxsize=5000, ttl=300)
async def is_shared(chute_id: str, user_id: str):
    """
    Check if a chute has been shared with a user.
    """
    cache_key = f"cshare:{chute_id}:{user_id}"
    cached = await memcache_get(cache_key)
    if cached:
        return cached == b"1"
    async with get_session() as db:
        query = select(
            exists().where(and_(ChuteShare.chute_id == chute_id, ChuteShare.shared_to == user_id))
        )
        result = await db.execute(query)
        shared = result.scalar()
        await memcache_set(cache_key, b"1" if shared else b"0", exptime=60)
        return shared


async def track_prefix_hashes(prefixes, instance_id):
    if not prefixes:
        return
    try:
        for _, prefix_hash in prefixes:
            await memcache_set(f"pfx:{prefix_hash}:{instance_id}".encode(), b"1", exptime=600)
            break  # XXX only track the largest prefix
    except Exception as exc:
        logger.warning(f"Error setting prefix hash cache: {exc}")


async def _invoke_one(
    chute: Chute,
    path: str,
    stream: bool,
    args: str,
    kwargs: str,
    target: Instance,
    started_at: float,
    metrics: dict = {},
    prefixes: list = None,
    manager: LeastConnManager = None,
):
    """
    Try invoking a chute/cord with a single instance.
    """
    # Call the miner's endpoint.
    path = path.lstrip("/")
    response = None
    payload = {"args": args, "kwargs": kwargs}

    # Set the 'p' private flag on invocations.
    private_billing = (
        not chute.public
        and not has_legacy_private_billing(chute)
        and chute.user_id != await chutes_user_id()
    )

    iv = None
    if use_encryption_v2(target.chutes_version):
        if not target.symmetric_key:
            raise KeyExchangeRequired(f"Instance {target.instance_id} requires new symmetric key.")
        payload = aes_encrypt(json.dumps(payload), target.symmetric_key)
        iv = bytes.fromhex(payload[:32])

    # Encrypted paths?
    plain_path = path.lstrip("/").rstrip("/")
    if use_encrypted_path(target.chutes_version):
        path = "/" + path.lstrip("/")
        encrypted_path = aes_encrypt(path.ljust(24, "?"), target.symmetric_key, hex_encode=True)
        path = encrypted_path

    session, response = None, None
    timeout = 600 if semcomp(target.chutes_version or "0.0.0", "0.3.59") < 0 else 900
    try:
        session = await get_miner_session(target, timeout=timeout)
        headers, payload_string = sign_request(miner_ss58=target.miner_hotkey, payload=payload)
        if iv:
            headers["X-Chutes-Serialized"] = "true"
        response = await session.post(
            f"/{path}",
            data=payload_string,
            headers=headers,
        )
        if response.status != 200:
            logger.info(
                f"Received response {response.status} from miner {target.miner_hotkey} instance_id={target.instance_id} of chute_id={target.chute_id}"
            )

        # Check if the instance restarted and is using encryption V2.
        if response.status == status.HTTP_426_UPGRADE_REQUIRED and iv:
            raise KeyExchangeRequired(
                f"Instance {target.instance_id} responded with 426, new key exchange required."
            )

        # Check if the instance is overwhelmed.
        if response.status == status.HTTP_429_TOO_MANY_REQUESTS:
            raise InstanceRateLimit(
                f"Instance {target.instance_id=} has returned a rate limit error!"
            )

        # Handle bad client requests.
        if response.status == status.HTTP_400_BAD_REQUEST:
            raise BadRequest("Invalid request: " + await response.text())

        if response.status == 451:
            logger.info(f"BAD ENCRYPTION: {await response.text()} from {payload=}")

        response.raise_for_status()

        # All good, send back the response.
        if stream:
            last_chunk = None
            any_chunks = False
            chunk_idx = 0
            async for raw_chunk in response.content:
                chunk = raw_chunk
                if iv:
                    chunk = aes_decrypt(raw_chunk, target.symmetric_key, iv)

                # Track time to first token and (approximate) token count; approximate
                # here because in speculative decoding multiple tokens may be returned.
                if (
                    chute.standard_template == "vllm"
                    and plain_path in LLM_PATHS
                    and chunk.startswith(b"data: {")
                    and b'content":""' not in chunk
                    and b'content": ""' not in chunk
                ):
                    if metrics["ttft"] is None:
                        metrics["ttft"] = round(time.time() - started_at, 3)
                    metrics["tokens"] += 1
                    chunk_idx += 1

                # Additional model response validation; model name/cllmv.
                if chute.standard_template == "vllm" and chunk.startswith(b"data: {"):
                    # Model name check.
                    data = None
                    try:
                        data = json.loads(chunk[6:])
                    except Exception:
                        ...
                    if (
                        isinstance(data, dict)
                        and data.get("model") != chute.name
                        and not data.get("error")
                    ):
                        logger.warning(
                            f"Model does not match chute name!: expected={chute.name} found {data.get('model')} -> {data=}"
                        )
                        raise EmptyLLMResponse(
                            f"BAD_RESPONSE {target.instance_id=} {chute.name} returned invalid chunk (model name)"
                        )

                    # CLLMV check.
                    if (
                        (random.random() <= 0.05 or chunk_idx <= 5)
                        and image_supports_cllmv(chute.image)
                        and target.version == chute.version
                        and chute.chute_id
                        not in (
                            "385aa551-6085-5c94-bda8-aa12609ef73b",
                            "64371741-a0a4-577c-923e-f545da8f6476",
                            "69f319a7-7f25-5cce-ae30-68b30fd5ec5d",
                            "f53ae961-7dcd-576f-badd-098f907d9bd2",
                            "110bf8d9-d07e-54dd-9c31-abe1a9919c7a",
                        )
                        and "model" in data
                        and not data.get("error")
                    ):
                        verification_token = data.get("chutes_verification")
                        text = None
                        has_tool_call = False
                        if (
                            "choices" in data
                            and isinstance(data["choices"], list)
                            and data["choices"]
                        ):
                            choice = data["choices"][0]
                            if isinstance(choice, dict):
                                if plain_path.startswith("chat"):
                                    if (
                                        "delta" in choice
                                        and choice["delta"]
                                        and isinstance(choice["delta"], dict)
                                    ):
                                        text = choice["delta"].get("content") or choice[
                                            "delta"
                                        ].get("reasoning_content")
                                        tool_calls = choice["delta"].get("tool_calls")
                                        if isinstance(tool_calls, list) and tool_calls:
                                            has_tool_call = True
                                else:
                                    text = choice.get("text")

                        # Verify the hash.
                        if (text or not has_tool_call) and (
                            not verification_token
                            or not cllmv_validate(
                                data.get("id") or "bad",
                                data.get("created") or 0,
                                text,
                                verification_token,
                                target.config_id,
                                chute.name,
                                chute.revision,
                            )
                        ):
                            logger.warning(
                                f"CLLMV FAILURE: STREAMED {target.instance_id=} {target.miner_hotkey=} {chute.name=}: {data=}"
                            )
                            raise InvalidCLLMV(
                                f"BAD_RESPONSE {target.instance_id=} {chute.name=} returned invalid chunk (failed cllmv check)"
                            )

                    last_chunk = chunk
                if b"data:" in chunk:
                    any_chunks = True

                yield chunk.decode()

            if chute.standard_template == "vllm" and plain_path in LLM_PATHS and metrics:
                if not any_chunks:
                    logger.warning(f"NO CHUNKS RETURNED: {chute.name} {target.instance_id=}")
                    raise EmptyLLMResponse(
                        f"EMPTY_STREAM {target.instance_id=} {chute.name} returned zero data chunks!"
                    )
                total_time = time.time() - started_at
                prompt_tokens = metrics.get("it", 0)
                completion_tokens = metrics.get("tokens", 0)

                # Sanity check on prompt token counts.
                if not metrics["it"]:
                    # Have to guess since this was done from the SDK and we aren't going to unpickle here.
                    raw_payload_size = len(json.dumps(payload))
                    metrics["it"] = len(raw_payload_size) / 3
                    prompt_tokens = metrics["it"]
                    logger.warning(f"Estimated the prompt tokens: {prompt_tokens} for {chute.name}")

                # Use usage data from the engine, but sanity check it...
                if last_chunk and b'"usage"' in last_chunk:
                    try:
                        usage_obj = json.loads(last_chunk[6:].decode())
                        usage = usage_obj.get("usage", {})
                        claimed_prompt_tokens = usage.get("prompt_tokens")

                        # Sanity check on prompt token counts.
                        if claimed_prompt_tokens > prompt_tokens * 10:
                            logger.warning(
                                f"Prompt tokens exceeded expectations [stream]: {claimed_prompt_tokens=} vs estimated={prompt_tokens} "
                                f"hotkey={target.miner_hotkey} instance_id={target.instance_id} chute={chute.name}"
                            )
                        else:
                            prompt_tokens = min(claimed_prompt_tokens, prompt_tokens)

                        # Sanity check on completion token counts.
                        claimed_completion_tokens = usage.get("completion_tokens")
                        if claimed_completion_tokens is not None:
                            # Some chutes do multi-token prediction, but even so let's make sure people don't do shenanigans.
                            if claimed_completion_tokens > completion_tokens * 10:
                                logger.warning(
                                    f"Completion tokens exceeded expectations [stream]: {claimed_completion_tokens=} vs estimated={completion_tokens} "
                                    f"hotkey={target.miner_hotkey} instance_id={target.instance_id} chute={chute.name}"
                                )
                            else:
                                completion_tokens = claimed_completion_tokens
                    except Exception as exc:
                        logger.warning(f"Error checking metrics: {exc}")

                metrics["it"] = max(0, prompt_tokens or 0)
                metrics["ot"] = max(0, completion_tokens or 0)
                metrics["ctps"] = round((metrics["it"] + metrics["ot"]) / total_time, 3)
                metrics["tps"] = round(metrics["ot"] / total_time, 3)
                metrics["tt"] = round(total_time, 3)
                if manager and manager.mean_count is not None:
                    metrics["mc"] = manager.mean_count

                # Moving average performance tracking to keep compute units immutable.
                ma_updates = await PERF_TRACKER.update_invocation_metrics(
                    chute_id=chute.chute_id,
                    duration=total_time,
                    metrics=metrics,
                    private_billing=private_billing,
                )
                metrics.update(ma_updates)

                if random.random() <= 0.1:
                    logger.info(f"Metrics for chute={chute.name} {metrics}")
                track_vllm_usage(chute.chute_id, target.miner_hotkey, total_time, metrics)
                await track_prefix_hashes(prefixes, target.instance_id)
        else:
            # Non-streamed responses, which may be encrypted with the new chutes encryption V2.
            headers = response.headers
            body_bytes = await response.read()
            data = {}
            if iv:
                # Encryption V2 always uses JSON, regardless of the underlying data type.
                response_data = json.loads(body_bytes)
                if "json" in response_data:
                    plaintext = aes_decrypt(response_data["json"], target.symmetric_key, iv)
                    if chute.standard_template == "vllm" and plaintext.startswith(
                        b'{"object":"error","message":"input_ids cannot be empty."'
                    ):
                        logger.warning(
                            f"Non-stream failed here: {chute.chute_id=} {target.instance_id=} {plaintext=}"
                        )
                        raise Exception(
                            "SGLang backend failure, input_ids null error response produced."
                        )
                    try:
                        data = {"content_type": "application/json", "json": json.loads(plaintext)}
                    except Exception as exc2:
                        logger.error(f"FAILED HERE: {str(exc2)} from {plaintext=}")
                        raise
                else:
                    # Response was a file or other response object.
                    plaintext = aes_decrypt(response_data["body"], target.symmetric_key, iv)
                    headers = response_data["headers"]
                    data = {
                        "content_type": response_data.get(
                            "media_type", headers.get("Content-Type", "text/plain")
                        ),
                        "bytes": base64.b64encode(plaintext).decode(),
                    }
            else:
                # Legacy response handling.
                content_type = response.headers.get("content-type")
                if content_type in (None, "application/json"):
                    json_data = await response.json()
                    data = {"content_type": content_type, "json": json_data}
                elif content_type.startswith("text/"):
                    text_data = await response.text()
                    data = {"content_type": content_type, "text": text_data}
                else:
                    raw_data = await response.read()
                    data = {
                        "content_type": content_type,
                        "bytes": base64.b64encode(raw_data).decode(),
                    }

            # Track metrics for the standard LLM/diffusion templates.
            total_time = time.time() - started_at
            if chute.standard_template == "vllm" and plain_path in LLM_PATHS:
                json_data = data.get("json")
                if json_data:
                    prompt_tokens = metrics.get("it", 0)
                    if not prompt_tokens:
                        # Have to guess since this was done from the SDK and we aren't going to unpickle here.
                        raw_payload_size = len(json.dumps(payload))
                        metrics["it"] = len(raw_payload_size) / 3
                        prompt_tokens = metrics["it"]
                        logger.warning(
                            f"Estimated the prompt tokens: {prompt_tokens} for {chute.name}"
                        )

                    # Make sure model name response is valid/matches chute name.
                    if "model" not in json_data or json_data.get("model") != chute.name:
                        logger.warning(
                            f"NOSTREAM_BADMODEL: {chute.name=} {chute.chute_id=} response had invalid/missing model: {target.instance_id=}: json_data['model']={json_data.get('model')}"
                        )
                        raise EmptyLLMResponse(
                            f"BAD_RESPONSE {target.instance_id=} {chute.name} returned invalid chunk (model name)"
                        )

                    # New verification hash.
                    if (
                        image_supports_cllmv(chute.image)
                        and target.version == chute.version
                        and "model" in json_data
                        and chute.chute_id
                        not in (
                            "385aa551-6085-5c94-bda8-aa12609ef73b",
                            "64371741-a0a4-577c-923e-f545da8f6476",
                            "69f319a7-7f25-5cce-ae30-68b30fd5ec5d",
                            "f53ae961-7dcd-576f-badd-098f907d9bd2",
                            "110bf8d9-d07e-54dd-9c31-abe1a9919c7a",
                        )
                    ):
                        verification_token = json_data.get("chutes_verification")
                        text = None
                        if json_data.get("choices"):
                            choice = json_data["choices"][0]
                            if "text" in choice and not plain_path.startswith("chat"):
                                text = choice["text"]
                            elif isinstance(choice.get("message"), dict):
                                text = choice["message"].get(
                                    "content", choice["message"].get("reasoning_content")
                                )
                        if not verification_token or not cllmv_validate(
                            json_data.get("id") or "bad",
                            json_data.get("created") or 0,
                            text,
                            verification_token,
                            target.config_id,
                            chute.name,
                            chute.revision,
                        ):
                            logger.warning(
                                f"CLLMV FAILURE: {target.instance_id=} {target.miner_hotkey=} {chute.name=}"
                            )
                            raise InvalidCLLMV(
                                f"BAD_RESPONSE {target.instance_id=} {chute.name=} returned invalid chunk (failed cllmv check)"
                            )
                        elif "affine" in chute.name.lower():
                            logger.success(
                                f"CLLMV success {target.instance_id=} {target.miner_hotkey=} {chute.name=} {chute.chute_id=}"
                            )

                    output_text = None
                    if plain_path == "chat":
                        try:
                            output_text = json_data["choices"][0]["message"]["content"] or ""
                            reasoning_content = json_data["choices"][0]["message"].get(
                                "reasoning_content"
                            )
                            if reasoning_content:
                                output_text += " " + reasoning_content
                        except Exception:
                            ...
                    else:
                        try:
                            output_text = json_data["choices"][0]["text"]
                        except (KeyError, IndexError):
                            ...
                    if not output_text:
                        output_text = json.dumps(json_data).decode()
                    completion_tokens = await count_str_tokens(output_text)
                    if (usage := json_data.get("usage")) is not None:
                        if claimed_completion_tokens := usage.get("completion_tokens", 0):
                            if claimed_completion_tokens > completion_tokens * 10:
                                logger.warning(
                                    f"Completion tokens exceeded expectations [nostream]: {claimed_completion_tokens=} vs estimated={completion_tokens} "
                                    f"hotkey={target.miner_hotkey} instance_id={target.instance_id} chute={chute.name}"
                                )
                            else:
                                completion_tokens = claimed_completion_tokens
                        if claimed_prompt_tokens := usage.get("prompt_tokens", 0):
                            if claimed_prompt_tokens > prompt_tokens * 10:
                                logger.warning(
                                    f"Prompt tokens exceeded expectations [nostream]: {claimed_prompt_tokens=} vs estimated={prompt_tokens} "
                                    f"hotkey={target.miner_hotkey} instance_id={target.instance_id} chute={chute.name}"
                                )
                            else:
                                prompt_tokens = claimed_prompt_tokens
                    else:
                        logger.warning(
                            f"Response from {target.instance_id=} {target.miner_hotkey=} of {chute.chute_id=} {chute.name=} did not include usage data!"
                        )
                        raise InvalidResponse(
                            f"BAD_RESPONSE {target.instance_id=} {chute.name=} returned invalid response (missing usage data)"
                        )

                    # Track metrics using either sane claimed usage metrics or estimates.
                    metrics["tokens"] = completion_tokens
                    metrics["it"] = prompt_tokens
                    metrics["ot"] = completion_tokens
                    metrics["ctps"] = round((metrics["it"] + metrics["ot"]) / total_time, 3)
                    metrics["tps"] = round(metrics["ot"] / total_time, 3)
                    metrics["tt"] = round(total_time, 3)
                    if manager and manager.mean_count is not None:
                        metrics["mc"] = manager.mean_count

                    # Moving average performance tracking to keep compute units immutable.
                    ma_updates = await PERF_TRACKER.update_invocation_metrics(
                        chute_id=chute.chute_id,
                        duration=total_time,
                        metrics=metrics,
                        private_billing=private_billing,
                    )
                    metrics.update(ma_updates)
                    if random.random() <= 0.1:
                        logger.info(f"Metrics for {chute.name}: {metrics}")
                    track_vllm_usage(chute.chute_id, target.miner_hotkey, total_time, metrics)
                    await track_prefix_hashes(prefixes, target.instance_id)
            elif (
                chute.standard_template == "diffusion"
                and path == "generate"
                and (metrics or {}).get("steps")
            ):
                delta = time.time() - started_at
                metrics["sps"] = int(metrics["steps"]) / delta

                # Moving average steps per second calc.
                ma_updates = await PERF_TRACKER.update_invocation_metrics(
                    chute_id=chute.chute_id,
                    duration=delta,
                    metrics=metrics,
                    private_billing=private_billing,
                )
                metrics.update(ma_updates)

            yield data
    finally:
        if response:
            try:
                async for _ in response.content:
                    pass
            except Exception:
                pass
            finally:
                try:
                    response.close()
                except Exception:
                    pass
        if session:
            try:
                await session.close()
            except Exception:
                pass


async def _s3_upload(data: io.BytesIO, path: str):
    """
    S3 upload helper.
    """
    try:
        async with settings.s3_client() as s3:
            await s3.upload_fileobj(data, settings.storage_bucket, path)
    except Exception as exc:
        logger.error(f"failed to store: {path} -> {exc}")


async def invoke(
    chute: Chute,
    user: User,
    path: str,
    function: str,
    stream: bool,
    args: str,
    kwargs: str,
    manager: LeastConnManager,
    parent_invocation_id: str,
    metrics: dict = {},
    request: Request = None,
    prefixes: list = None,
    reroll: bool = False,
):
    """
    Helper to actual perform function invocations, retrying when a target fails.
    """
    chute_id = chute.chute_id
    user_id = user.user_id
    yield sse(
        {
            "trace": {
                "timestamp": now_str(),
                "invocation_id": parent_invocation_id,
                "chute_id": chute_id,
                "function": function,
                "message": f"identified {len(manager.instances)} available targets",
            },
        }
    )

    infra_overload = False
    avoid = []
    for attempt_idx in range(5):
        async with manager.get_target(avoid=avoid, prefixes=prefixes) as (target, error_message):
            try:
                if attempt_idx == 0 and manager.mean_count is not None:
                    track_capacity(chute.chute_id, manager.mean_count, chute.concurrency or 1.0)
            except Exception as cap_err:
                logger.error(
                    f"Failed tracking chute capacity metrics: {cap_err}\n{traceback.format_exc()}"
                )

            if not target:
                if infra_overload or error_message == "infra_overload":
                    logger.warning(f"All miners are at max capacity: {chute.name=}")
                    yield sse(
                        {
                            "error": "infra_overload",
                            "detail": "Infrastructure is at maximum capacity, try again later",
                        }
                    )
                else:
                    if not error_message:
                        error_message = (
                            "Unhandled exception trying to load backend node to route request"
                        )
                    logger.warning(f"No targets found for {chute_id=}: {error_message=}")
                    yield sse({"error": error_message})
                return

            invocation_id = str(uuid.uuid4())
            started_at = time.time()
            multiplier = NodeSelector(**chute.node_selector).compute_multiplier
            if chute.boost:
                multiplier *= chute.boost

            try:
                yield sse(
                    {
                        "trace": {
                            "timestamp": now_str(),
                            "invocation_id": parent_invocation_id,
                            "child_id": invocation_id,
                            "chute_id": chute_id,
                            "function": function,
                            "message": f"attempting to query target={target.instance_id} uid={target.miner_uid} hotkey={target.miner_hotkey} coldkey={target.miner_coldkey}",
                        },
                    }
                )
                async for data in _invoke_one(
                    chute,
                    path,
                    stream,
                    args,
                    kwargs,
                    target,
                    request.state.started_at,
                    metrics,
                    prefixes,
                    manager,
                ):
                    try:
                        if "input_ids cannot be empty" in str(data):
                            logger.warning(
                                f"Failed here: {chute.chute_id=} {target.instance_id=} {data=}"
                            )
                    except Exception:
                        ...
                    yield sse({"result": data})

                # Store complete record in new invocations database, async.
                # XXX this is a different started_at from global request started_at, for compute units
                duration = time.time() - started_at
                asyncio.create_task(
                    safe_store_invocation(
                        parent_invocation_id,
                        invocation_id,
                        chute.chute_id,
                        chute.user_id,
                        function,
                        user_id,
                        chute.image_id,
                        chute.image.user_id,
                        target.instance_id,
                        target.miner_uid,
                        target.miner_hotkey,
                        duration,
                        multiplier,
                        error_message=None,
                        metrics=metrics,
                        legacy=False,
                    )
                )

                # Track in the legacy DB.
                compute_units = multiplier * math.ceil(duration)
                asyncio.create_task(
                    store_invocation(
                        parent_invocation_id,
                        invocation_id,
                        chute.chute_id,
                        chute.user_id,
                        function,
                        user_id,
                        chute.image_id,
                        chute.image.user_id,
                        target.instance_id,
                        target.miner_uid,
                        target.miner_hotkey,
                        duration,
                        multiplier,
                        error_message=None,
                        metrics=metrics,
                        legacy=True,
                    )
                )

                # Clear any consecutive failure flags.
                asyncio.create_task(
                    settings.redis_client.delete(f"consecutive_failures:{target.instance_id}")
                )

                # Update capacity tracking.
                track_request_completed(chute.chute_id)

                # Calculate the credits used and deduct from user's balance asynchronously.
                # For LLMs and Diffusion chutes, we use custom per token/image step pricing,
                # otherwise it's just based on time used.
                balance_used = 0.0
                override_applied = False
                if compute_units and not request.state.free_invocation:
                    hourly_price = await selector_hourly_price(chute.node_selector)

                    # Per megatoken pricing.
                    if chute.standard_template == "vllm" and metrics:
                        per_million_in, per_million_out = await get_mtoken_price(
                            user_id, chute.chute_id
                        )
                        balance_used = (metrics.get("it", 0) or 0) / 1000000.0 * per_million_in + (
                            metrics.get("ot", 0) or 0
                        ) / 1000000.0 * per_million_out
                        override_applied = True

                    elif (
                        price_override := await PriceOverride.get(user_id, chute.chute_id)
                    ) is not None:
                        if (
                            chute.standard_template == "diffusion"
                            and price_override.per_step is not None
                        ):
                            balance_used = (metrics.get("steps", 0) or 0) * price_override.per_step
                            override_applied = True

                        # Per request pricing (fallback if specific pricing not available)
                        elif price_override.per_request is not None:
                            balance_used = price_override.per_request
                            override_applied = True

                    # If no override was applied, use standard pricing
                    if not override_applied:
                        # Track any discounts.
                        discount = 0.0
                        # A negative discount just makes the chute more than our typical pricing,
                        # e.g. for chutes that have a concurrency of one and we can't really operate
                        # efficiently with the normal pricing.
                        if chute.discount and -3 < chute.discount <= 1:
                            discount = chute.discount

                        if discount < 1.0:
                            # Diffusion per step pricing.
                            if chute.standard_template == "diffusion":
                                balance_used = (
                                    (metrics.get("steps", 0) or 0)
                                    * hourly_price
                                    * DIFFUSION_PRICE_MULT_PER_STEP
                                )
                                balance_used -= balance_used * discount

                            default_balance_used = compute_units * COMPUTE_UNIT_PRICE_BASIS / 3600.0
                            default_balance_used -= default_balance_used * discount
                            if not balance_used:
                                logger.info(
                                    f"BALANCE: defaulting to time-based pricing: {default_balance_used=} for {chute.name=}"
                                )
                                balance_used = default_balance_used

                # Increment values in redis, which will be asynchronously processed to deduct from the actual balance.
                if balance_used and reroll and not override_applied:
                    # Also apply fractional balance to reroll.
                    balance_used = balance_used * settings.reroll_multiplier

                # User discounts.
                if balance_used and not override_applied:
                    user_discount = await InvocationDiscount.get(user_id, chute.chute_id)
                    if user_discount:
                        balance_used -= balance_used * user_discount

                # Don't charge for private instances.
                if (
                    not chute.public
                    and not has_legacy_private_billing(chute)
                    and chute.user_id != await chutes_user_id()
                ):
                    balance_used = 0

                # Ship the data over to usage tracker which actually deducts/aggregates balance/etc.
                try:
                    pipeline = settings.redis_client.pipeline()
                    key = f"balance:{user_id}:{chute.chute_id}"
                    pipeline.hincrbyfloat(key, "amount", balance_used)
                    pipeline.hincrby(key, "count", 1)
                    if chute.standard_template == "vllm" and metrics:
                        pipeline.hincrby(key, "input_tokens", metrics.get("it", 0))
                        pipeline.hincrby(key, "output_tokens", metrics.get("ot", 0))
                    pipeline.hset(key, "timestamp", int(time.time()))
                    await pipeline.execute()
                except Exception as exc:
                    logger.error(f"Error updating usage pipeline: {exc}")

                # Increment quota usage value.
                if (
                    request.state.free_invocation
                    and chute.discount < 1.0
                    and (
                        chute.public
                        or has_legacy_private_billing(chute)
                        or chute.user_id == await chutes_user_id()
                    )
                ):
                    try:
                        value = 1.0 if not reroll else settings.reroll_multiplier
                        if chute.name.endswith("-turbo"):
                            value *= 2.0
                        key = await InvocationQuota.quota_key(user.user_id, chute.chute_id)
                        _ = await settings.quota_client.incrbyfloat(key, value)
                    except Exception as exc:
                        logger.error(
                            f"Error updating quota usage for {user.user_id} chute {chute.chute_id}: {exc}"
                        )

                # For private chutes, push back the instance termination timestamp.
                if (
                    not chute.public
                    and not has_legacy_private_billing(chute)
                    and chute.user_id != await chutes_user_id()
                ):
                    await update_shutdown_timestamp(target.instance_id)

                yield sse(
                    {
                        "trace": {
                            "timestamp": now_str(),
                            "invocation_id": parent_invocation_id,
                            "child_id": invocation_id,
                            "chute_id": chute_id,
                            "function": function,
                            "message": f"successfully called {function=} on target={target.instance_id} uid={target.miner_uid} hotkey={target.miner_hotkey} coldkey={target.miner_coldkey}",
                        }
                    }
                )
                return
            except Exception as exc:
                avoid.append(target.instance_id)
                error_message = f"{exc}\n{traceback.format_exc()}"
                error_message = error_message.replace(
                    f"{target.host}:{target.port}", "[host redacted]"
                ).replace(target.host, "[host redacted]")

                error_detail = None
                if isinstance(exc, InstanceRateLimit):
                    error_message = "RATE_LIMIT"
                    infra_overload = True
                    track_request_rate_limited(chute.chute_id)
                elif isinstance(exc, BadRequest):
                    error_message = "BAD_REQUEST"
                    error_detail = str(exc)
                elif isinstance(exc, KeyExchangeRequired):
                    error_message = "KEY_EXCHANGE_REQUIRED"
                elif isinstance(exc, EmptyLLMResponse):
                    error_message = "EMPTY_STREAM"
                elif isinstance(exc, InvalidCLLMV):
                    error_message = "CLLMV_FAILURE"

                # Store complete record in new invocations database, async.
                duration = time.time() - started_at
                asyncio.create_task(
                    safe_store_invocation(
                        parent_invocation_id,
                        invocation_id,
                        chute.chute_id,
                        chute.user_id,
                        function,
                        user_id,
                        chute.image_id,
                        chute.image.user_id,
                        target.instance_id,
                        target.miner_uid,
                        target.miner_hotkey,
                        duration,
                        multiplier,
                        error_message=error_message,
                        legacy=False,
                    )
                )

                # Legacy invocations table storage.
                asyncio.create_task(
                    store_invocation(
                        parent_invocation_id,
                        invocation_id,
                        chute.chute_id,
                        chute.user_id,
                        function,
                        user_id,
                        chute.image_id,
                        chute.image.user_id,
                        target.instance_id,
                        target.miner_uid,
                        target.miner_hotkey,
                        duration,
                        multiplier,
                        error_message=error_message,
                        legacy=True,
                    )
                )

                async with get_session() as session:
                    # Handle the case where encryption V2 is in use and the instance needs a new key exchange.
                    if error_message == "KEY_EXCHANGE_REQUIRED":
                        # NOTE: Could probably just re-validate rather than deleting the instance, but this ensures no shenanigans are afoot.
                        delete_result = await session.execute(
                            text("DELETE FROM instances WHERE instance_id = :instance_id"),
                            {"instance_id": target.instance_id},
                        )
                        if delete_result.rowcount > 0:
                            await session.execute(
                                text(
                                    "UPDATE instance_audit SET deletion_reason = 'miner responded with 426 upgrade required, new symmetric key needed' WHERE instance_id = :instance_id"
                                ),
                                {"instance_id": target.instance_id},
                            )
                            await session.commit()
                            await invalidate_instance_cache(
                                target.chute_id, instance_id=target.instance_id
                            )
                            asyncio.create_task(
                                notify_deleted(
                                    target,
                                    message=f"Instance {target.instance_id} of miner {target.miner_hotkey} responded with a 426 error, indicating a new key exchange is required.",
                                )
                            )

                    elif error_message not in ("RATE_LIMIT", "BAD_REQUEST"):
                        # Handle consecutive failures (auto-delete instances).
                        consecutive_failures = await settings.redis_client.incr(
                            f"consecutive_failures:{target.instance_id}"
                        )
                        if (
                            consecutive_failures
                            and consecutive_failures >= settings.consecutive_failure_limit
                        ):
                            logger.warning(
                                f"CONSECUTIVE FAILURES: {target.instance_id}: {consecutive_failures=}"
                            )
                            delete_result = await session.execute(
                                text("DELETE FROM instances WHERE instance_id = :instance_id"),
                                {"instance_id": target.instance_id},
                            )
                            if delete_result.rowcount > 0:
                                await invalidate_instance_cache(
                                    target.chute_id, instance_id=target.instance_id
                                )
                                await session.execute(
                                    text(
                                        f"UPDATE instance_audit SET deletion_reason = 'max consecutive failures {consecutive_failures} reached' WHERE instance_id = :instance_id"
                                    ),
                                    {"instance_id": target.instance_id},
                                )
                                await session.commit()
                                asyncio.create_task(
                                    notify_deleted(
                                        target,
                                        message=f"Instance {target.instance_id} of miner {target.miner_hotkey} has reached the consecutive failure limit of {settings.consecutive_failure_limit} and has been deleted.",
                                    )
                                )

                if error_message == "BAD_REQUEST":
                    logger.warning(
                        f"instance_id={target.instance_id} [chute_id={target.chute_id}]: bad request {error_detail}"
                    )
                    yield sse(
                        {"error": "bad_request", "detail": f"Invalid request: {error_detail}"}
                    )
                    return

                yield sse(
                    {
                        "trace": {
                            "timestamp": now_str(),
                            "invocation_id": parent_invocation_id,
                            "child_id": invocation_id,
                            "chute_id": chute_id,
                            "function": function,
                            "message": f"error encountered while querying target={target.instance_id} uid={target.miner_uid} hotkey={target.miner_hotkey} coldkey={target.miner_coldkey}: exc={error_message}",
                        },
                    }
                )
                logger.error(
                    f"Error trying to call instance_id={target.instance_id} [chute_id={target.chute_id}]: {error_message}"
                )
    if infra_overload:
        logger.warning(f"All miners are at max capacity: {chute.name=}")
        yield sse(
            {
                "error": "infra_overload",
                "detail": "Infrastructure is at maximum capacity, try again later",
            }
        )
    else:
        logger.error(f"Failed to query any miners after {attempt_idx + 1} attempts")
        yield sse({"error": "exhausted all available targets to no avail"})


async def load_llm_details(chute, target):
    """
    Load the /v1/models endpoint for a chute from a single instance.
    """
    path = "/get_models"
    if use_encrypted_path(target.chutes_version):
        path = aes_encrypt(path.ljust(24, "?"), target.symmetric_key, hex_encode=True)
    payload = {
        "args": base64.b64encode(gzip.compress(pickle.dumps(tuple()))).decode(),
        "kwargs": base64.b64encode(gzip.compress(pickle.dumps({}))).decode(),
    }
    iv = None
    if use_encryption_v2(target.chutes_version):
        if not target.symmetric_key:
            raise KeyExchangeRequired(f"Instance {target.instance_id} requires new symmetric key.")
        payload = aes_encrypt(json.dumps(payload), target.symmetric_key)
        iv = bytes.fromhex(payload[:32])

    async with aiohttp.ClientSession(
        timeout=aiohttp.ClientTimeout(connect=5.0, total=60.0),
        read_bufsize=8 * 1024 * 1024,
        raise_for_status=True,
    ) as session:
        headers, payload_string = sign_request(miner_ss58=target.miner_hotkey, payload=payload)
        if iv:
            headers["X-Chutes-Serialized"] = "true"
        async with session.post(
            f"http://{target.host}:{target.port}/{path}", data=payload_string, headers=headers
        ) as resp:
            raw_data = await resp.json()
            logger.info(
                f"{target.chute_id=} {target.instance_id=} {target.miner_hotkey=}: {raw_data=}"
            )
            info = (
                raw_data
                if not iv
                else json.loads(aes_decrypt(raw_data["json"], target.symmetric_key, iv))
            )
            return info["data"][0]


async def get_mtoken_price(user_id: str, chute_id: str) -> tuple[float, float]:
    """
    Get the per-million token price for an LLM.
    """
    cache_key = f"mtokenprice:{user_id}:{chute_id}"
    cached = await memcache_get(cache_key)
    if cached is not None:
        try:
            parts = cached.decode().split(":")
            if len(parts) == 2:
                return float(parts[0]), float(parts[1])
            await memcache_delete(cache_key)
        except Exception:
            await memcache_delete(cache_key)

    # Inject the pricing information, first by checking price overrides, then
    # using the standard node-selector based calculation.
    per_million_in, per_million_out = None, None
    override = await PriceOverride.get(user_id, chute_id)
    user_discount = None
    if not override or override.user_id != user_id:
        user_discount = await InvocationDiscount.get(user_id, chute_id)
        if user_discount:
            logger.info(f"BALANCE: Applying additional user discount: {user_id=} {user_discount=}")
    chute = await get_one(chute_id)
    if override:
        if override.per_million_in is not None:
            per_million_in = override.per_million_in
            if user_discount:
                per_million_in -= per_million_in * user_discount
        if override.per_million_out is not None:
            per_million_out = override.per_million_out
            if user_discount:
                per_million_out -= per_million_out * user_discount

    # Standard compute-based calcs.
    if per_million_in is None or per_million_out is None:
        hourly = await selector_hourly_price(chute.node_selector)
        if per_million_in is None:
            per_million_in = max(hourly * LLM_PRICE_MULT_PER_MILLION_IN, LLM_MIN_PRICE_IN)
            if chute.discount:
                per_million_in -= per_million_in * chute.discount
                if (chute.concurrency or 1) < 16:
                    per_million_in *= 16.0 / (chute.concurrency or 1)
            if user_discount:
                per_million_in -= per_million_in * user_discount
        if per_million_out is None:
            per_million_out = max(hourly * LLM_PRICE_MULT_PER_MILLION_OUT, LLM_MIN_PRICE_OUT)
            if chute.discount:
                per_million_out -= per_million_out * chute.discount
                if (chute.concurrency or 1) < 16:
                    per_million_out *= 16.0 / (chute.concurrency or 1)
            if user_discount:
                per_million_out -= per_million_out * user_discount

    per_million_in = round(per_million_in, 2)
    per_million_out = round(per_million_out, 2)
    await memcache_set(cache_key, f"{per_million_in}:{per_million_out}".encode(), exptime=300)
    return per_million_in, per_million_out


async def get_and_store_llm_details(chute_id: str):
    """
    Load the data from /v1/models for a given LLM, cache it for later.
    """
    async with get_session() as session:
        chute = (
            (
                await session.execute(
                    select(Chute)
                    .where(Chute.chute_id == chute_id)
                    .options(selectinload(Chute.instances), selectinload(Chute.llm_detail))
                )
            )
            .unique()
            .scalar_one_or_none()
        )
        if not chute:
            logger.error(f"Chute not found: {chute_id}")
            return

        # Load the price per million tokens (in USD).
        per_million_in, per_million_out = await get_mtoken_price("global", chute_id)

        # Inject the tao price.
        price = {"input": {"usd": per_million_in}, "output": {"usd": per_million_out}}
        tao_usd = await get_fetcher().get_price("tao")
        if tao_usd:
            for key in ("input", "output"):
                price[key]["tao"] = price[key]["usd"] / tao_usd

        instances = [inst for inst in chute.instances if inst.active and inst.verified]
        random.shuffle(instances)

        # Try to fetch /v1/models from instances until one succeeds.
        model_info = None
        for instance in instances:
            try:
                model_info = await load_llm_details(chute, instance)
                model_info["id"] = chute.name
                model_info["chute_id"] = chute.chute_id
                model_info["price"] = price

                # OpenRouter format.
                model_info["pricing"] = {
                    "prompt": per_million_in,
                    "completion": per_million_out,
                }
                if chute.llm_detail and isinstance(chute.llm_detail.overrides, dict):
                    model_info.update(
                        {
                            k: v
                            for k, v in chute.llm_detail.overrides.items()
                            if k not in ("price", "pricing", "id", "root", "max_model_len")
                        }
                    )
                break
            except Exception as exc:
                logger.error(
                    f"Failed to load model info from {instance.instance_id=}: {exc=}\n{traceback.format_exc()}"
                )
        if not model_info:
            logger.error(f"Failed to populate model info from any instance for {chute_id=}")
            return None
        stmt = insert(LLMDetail).values(
            chute_id=chute_id, details=model_info, updated_at=func.now()
        )
        stmt = stmt.on_conflict_do_update(
            index_elements=["chute_id"],
            set_={"details": stmt.excluded.details, "updated_at": stmt.excluded.updated_at},
        )
        logger.success(f"Retrieved model info for {chute_id=}: {model_info=}")
        await session.execute(stmt)
        await session.commit()
        return model_info


async def refresh_all_llm_details():
    """
    Refresh LLM details for all LLMs.
    """
    async with get_session() as session:
        result = await session.execute(
            select(Chute.chute_id).where(
                Chute.standard_template == "vllm",
                Chute.user_id == await chutes_user_id(),
                Chute.chute_id != "561e4875-254d-588f-a36f-57c9cdef8961",
                Chute.public.is_(True),
            )
        )
        chute_ids = [row[0] for row in result]
    if not chute_ids:
        logger.info("No chutes found to refresh")
        return

    semaphore = asyncio.Semaphore(8)

    async def get_details_with_semaphore(chute_id: str):
        async with semaphore:
            try:
                return await get_and_store_llm_details(chute_id)
            except Exception as exc:
                logger.error(f"Failed to refresh LLM details for {chute_id}: {exc}")
                return None

    results = await asyncio.gather(
        *[get_details_with_semaphore(chute_id) for chute_id in chute_ids], return_exceptions=False
    )
    successful = [item for item in results if item is not None]
    logger.info(f"Refreshed LLM details successfully for {len(successful)}/{len(chute_ids)} chutes")
    return successful


async def get_llms(refresh: bool = False):
    """
    Get the combined /v1/models return value for chutes that are public and belong to chutes user.
    """
    if not refresh:
        cached = await settings.redis_client.get("all_llms")
        if cached:
            return json.loads(cached)
    else:
        await refresh_all_llm_details()

    async with get_session() as session:
        result = await session.execute(
            select(LLMDetail.details)
            .join(Chute, LLMDetail.chute_id == Chute.chute_id)
            .where(
                Chute.standard_template == "vllm",
                Chute.public.is_(True),
                Chute.user_id == await chutes_user_id(),
                Chute.chute_id != "561e4875-254d-588f-a36f-57c9cdef8961",
                LLMDetail.details.is_not(None),
            )
            .order_by(Chute.invocation_count.desc())
        )
        model_details = [row[0] for row in result if row[0] is not None]
        return_value = {"object": "list", "data": model_details}
        await settings.redis_client.set("all_llms", json.dumps(return_value), ex=300)
        return return_value


async def count_prompt_tokens(body):
    """
    Estimate the number of input tokens.
    """
    loop = asyncio.get_event_loop()
    try:
        if messages := body.get("messages"):
            if isinstance(messages, list):
                tokens = await loop.run_in_executor(
                    None,
                    TOKENIZER.apply_chat_template,
                    messages,
                )
                return len(tokens)
        if prompt := body.get("prompt"):
            return await count_str_tokens(prompt)
    except Exception as exc:
        logger.warning(f"Error estimating tokens: {exc}, defaulting to dumb method.")
    return int(len(repr(body).encode("utf-8", errors="ignore")) / 4)


async def count_str_tokens(output_str):
    """
    Estimate the number of output tokens.
    """
    loop = asyncio.get_event_loop()
    try:
        if isinstance(output_str, bytes):
            output_str = output_str.decode()
        tokens = await loop.run_in_executor(None, TOKENIZER, output_str)
        return max(0, len(tokens.input_ids) - 1)
    except Exception as exc:
        logger.warning(
            f"Error estimating tokens: {exc}, defaulting to dumb method: {output_str.__class__}"
        )
    return int(len(output_str) / 4)


async def update_llm_means():
    logger.info("Updating LLM miner mean metrics...")
    async with get_session() as session:
        await session.execute(text("DROP TABLE IF EXISTS llm_means_temp"))
        await session.execute(
            text("""
CREATE TABLE llm_means_temp AS
SELECT
    ins.chute_id,
    invocations.miner_hotkey,
    invocations.instance_id,
    AVG((metrics->>'tps')::float) as avg_tps,
    AVG((metrics->>'ot')::int) as avg_output_tokens
FROM invocations
JOIN instances ins ON invocations.instance_id = ins.instance_id
JOIN chutes c ON ins.chute_id = c.chute_id
WHERE
    started_at >= NOW() - INTERVAL '1 day'
    AND c.standard_template = 'vllm'
GROUP BY
    ins.chute_id,
    invocations.instance_id,
    invocations.miner_hotkey
ORDER BY
    ins.chute_id,
    avg_tps DESC
""")
        )
        await session.execute(text("DROP TABLE IF EXISTS llm_means"))
        await session.execute(text("ALTER TABLE llm_means_temp RENAME TO llm_means"))
        await session.commit()
