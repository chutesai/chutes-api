"""
Application logic and utilities for chutes.
"""

import os
import aiohttp
import asyncio
import re
import uuid
import random
import datetime
import io
import time
import traceback
import orjson as json
import base64
import gzip
import pickle
import backoff
from async_lru import alru_cache
from fastapi import Request, status
from loguru import logger
from transformers import AutoTokenizer
from sqlalchemy import and_, or_, text, update, String
from sqlalchemy.future import select
from sqlalchemy.orm import selectinload
from api.config import settings
from api.constants import (
    ENCRYPTED_HEADER,
    LLM_PRICE_MULT_PER_MILLION,
    DIFFUSION_PRICE_MULT_PER_STEP,
)
from api.database import get_session
from api.exceptions import InstanceRateLimit, BadRequest, KeyExchangeRequired
from api.util import sse, now_str, aes_encrypt, aes_decrypt, use_encryption_v2, use_encrypted_path
from api.chute.schemas import Chute, NodeSelector
from api.user.schemas import User
from api.user.service import chutes_user_id
from api.miner_client import sign_request
from api.instance.schemas import Instance
from api.instance.util import LeastConnManager, get_chute_target_manager
from api.gpu import COMPUTE_UNIT_PRICE_BASIS
from api.permissions import Permissioning
from api.metrics.vllm import track_usage as track_vllm_usage

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
TRACK_INVOCATION = text(
    """
INSERT INTO invocations (
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
    request_path,
    response_path,
    compute_multiplier,
    bounty
) VALUES (
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
    CURRENT_TIMESTAMP,
    NULL,
    NULL,
    :request_path,
    NULL,
    :compute_multiplier,
    0
) RETURNING to_char(date_trunc('week', started_at), 'IYYY_IW') AS suffix
"""
).columns(suffix=String)
UPDATE_INVOCATION = """
WITH removed_bounty AS (
    DELETE FROM bounties
    WHERE chute_id = :chute_id
    RETURNING bounty
)
UPDATE partitioned_invocations_{suffix} SET
    completed_at = CURRENT_TIMESTAMP,
    response_path = CAST(:response_path AS TEXT),
    bounty = COALESCE((SELECT bounty FROM removed_bounty), bounty),
    metrics = :metrics
WHERE invocation_id = :invocation_id
RETURNING CEIL(EXTRACT(EPOCH FROM (completed_at - started_at))) * compute_multiplier AS total_compute_units
"""
UPDATE_INVOCATION_ERROR = """
UPDATE partitioned_invocations_{suffix} SET
    completed_at = CURRENT_TIMESTAMP,
    error_message = CAST(:error_message AS TEXT)
WHERE invocation_id = :invocation_id
"""


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
    query = (
        select(Chute)
        .join(User, Chute.user_id == User.user_id)
        .where(or_(Chute.public.is_(True), Chute.user_id == current_user.user_id))
    )
    if load_instances:
        query = query.options(selectinload(Chute.instances))
    username = name_match.group(1) or current_user.username
    chute_name = name_match.group(2)
    chute_id_or_name = chute_id_or_name.lstrip("/")
    query = query.where(
        or_(
            and_(
                User.username == current_user.username,
                Chute.name == chute_name,
            ),
            and_(
                User.username == current_user.username,
                Chute.name == chute_id_or_name,
            ),
            and_(
                User.username == username,
                Chute.name == chute_name,
            ),
            Chute.chute_id == chute_id_or_name,
        )
    )
    result = await db.execute(query)
    return result.scalar_one_or_none()


@alru_cache(maxsize=100)
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


@alru_cache(maxsize=100)
async def get_one(name_or_id: str):
    """
    Load a chute by it's name or ID.l
    """
    chute_user = await chutes_user_id()
    async with get_session() as db:
        return (
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


async def _invoke_one(
    chute: Chute,
    path: str,
    stream: bool,
    args: str,
    kwargs: str,
    target: Instance,
    metrics: dict = {},
):
    """
    Try invoking a chute/cord with a single instance.
    """
    # Call the miner's endpoint.
    path = path.lstrip("/")
    response = None
    payload = {"args": args, "kwargs": kwargs}

    # Legacy chutes/encryption V1.
    iv = None
    legacy_encrypted = False
    if not use_encryption_v2(target.chutes_version):
        if settings.graval_url and random.random() <= 0.1:
            device_dicts = [node.graval_dict() for node in target.nodes]
            target_index = random.randint(0, len(device_dicts) - 1)
            target_device = device_dicts[target_index]
            seed = target.nodes[0].seed
            async with aiohttp.ClientSession(raise_for_status=True) as graval_session:
                try:
                    async with graval_session.post(
                        f"{settings.graval_url}/encrypt",
                        json={
                            "payload": {
                                "args": args,
                                "kwargs": kwargs,
                            },
                            "device_info": target_device,
                            "device_id": target_index,
                            "seed": seed,
                        },
                        timeout=3.0,
                    ) as resp:
                        payload = await resp.json()
                        legacy_encrypted = True
                except Exception as exc:
                    logger.error(
                        f"Error encrypting payload: {str(exc)}, sending plain text\n{traceback.format_exc()}"
                    )
    else:
        # Using encryption V2, make sure we have a key.
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
    try:
        session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(connect=5.0, total=600.0), read_bufsize=8 * 1024 * 1024
        )
        headers, payload_string = sign_request(miner_ss58=target.miner_hotkey, payload=payload)
        if iv:
            headers["X-Chutes-Serialized"] = "true"
        if legacy_encrypted:
            headers.update({ENCRYPTED_HEADER: "true"})
        iv_hex = iv.hex() if iv else None
        logger.debug(
            f"Attempting invocation of {chute.chute_id=} on {target.instance_id=} {legacy_encrypted=} {iv_hex=}"
        )
        started_at = time.time()
        response = await session.post(
            f"http://{target.host}:{target.port}/{path}",
            data=payload_string,
            headers=headers,
        )
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
            async for raw_chunk in response.content:
                chunk = raw_chunk
                if iv:
                    chunk = aes_decrypt(raw_chunk, target.symmetric_key, iv)
                if (
                    chute.standard_template == "vllm"
                    and plain_path in LLM_PATHS
                    and chunk.startswith(b"data: {")
                    and b'"content":""' not in chunk
                ):
                    if metrics["ttft"] is None:
                        metrics["ttft"] = time.time() - started_at
                    metrics["tokens"] += 1
                if chunk.startswith(b"data:") and not chunk.startswith(b"data: [DONE]"):
                    last_chunk = chunk

                yield chunk.decode()
            if chute.standard_template == "vllm" and plain_path in LLM_PATHS and metrics:
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
                        if claimed_prompt_tokens > prompt_tokens * 1.5:
                            logger.warning(
                                f"Prompt tokens exceeded expectations [stream]: {claimed_prompt_tokens=} vs estimated={prompt_tokens}"
                            )
                            # else:
                            prompt_tokens = min(claimed_prompt_tokens, prompt_tokens)

                        # Sanity check on completion token counts.
                        claimed_completion_tokens = usage.get("completion_tokens")
                        if claimed_completion_tokens is not None:
                            # Some chutes do multi-token prediction, but even so let's make sure people don't do shenanigans.
                            if claimed_completion_tokens > completion_tokens * 4:
                                logger.warning(
                                    f"Completion tokens exceeded expectations [stream]: {claimed_completion_tokens=} vs estimated={completion_tokens}"
                                )
                                # else:
                                completion_tokens = claimed_completion_tokens
                    except Exception as exc:
                        logger.warning(f"Error checking metrics: {exc}")

                metrics["it"] = max(0, prompt_tokens or 0)
                metrics["ot"] = max(0, completion_tokens or 0)
                metrics["tps"] = metrics["ot"] / total_time
                logger.info(
                    f"Metrics for chute={chute.name} miner={target.miner_hotkey} instance={target.instance_id}: {metrics}"
                )
                track_vllm_usage(chute.chute_id, target.miner_hotkey, total_time, metrics)
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

                    output_text = None
                    if plain_path == "chat":
                        try:
                            output_text = json_data["choices"][0]["message"]["content"]
                        except (KeyError, IndexError):
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
                            if claimed_completion_tokens > completion_tokens * 4:
                                logger.warning(
                                    f"Completion tokens exceeded expectations [nostream]: {claimed_completion_tokens=} vs estimated={completion_tokens}"
                                )
                            else:
                                completion_tokens = claimed_completion_tokens
                        if claimed_prompt_tokens := usage.get("prompt_tokens", 0):
                            if claimed_prompt_tokens > prompt_tokens * 1.5:
                                logger.warning(
                                    f"Prompt tokens exceeded expectations [nostream]: {claimed_prompt_tokens=} vs estimated={prompt_tokens}"
                                )
                            else:
                                prompt_tokens = claimed_prompt_tokens

                    # Track metrics using either sane claimed usage metrics or estimates.
                    metrics["tokens"] = completion_tokens
                    metrics["tps"] = completion_tokens / total_time
                    metrics["it"] = prompt_tokens
                    metrics["ot"] = completion_tokens
                    logger.info(
                        f"Metrics for [{chute.name}] miner={target.miner_hotkey} instance={target.instance_id}: {metrics}"
                    )
                    track_vllm_usage(chute.chute_id, target.miner_hotkey, total_time, metrics)
            elif (
                chute.standard_template == "diffusion"
                and path == "generate"
                and (metrics or {}).get("steps")
            ):
                metrics["sps"] = int(metrics["steps"]) / (time.time() - started_at)

            yield data
    finally:
        if response:
            await response.release()
        if session:
            await session.close()


async def _s3_upload(data: io.BytesIO, path: str):
    """
    S3 upload helper.
    """
    try:
        async with settings.s3_client() as s3:
            await s3.upload_fileobj(data, settings.storage_bucket, path)
    except Exception as exc:
        logger.error(f"failed to store: {path} -> {exc}")


async def _sample_request(chute_id, parent_invocation_id, args, kwargs):
    """
    Randomly sample and store request data.
    """
    request_path = None
    if random.random() <= REQUEST_SAMPLE_RATIO:
        today = datetime.date.today()
        request_path = f"invocations/{today.year}/{today.month}/{today.day}/{chute_id}/request-{parent_invocation_id}.json"
        asyncio.create_task(
            _s3_upload(io.BytesIO(json.dumps({"args": args, "kwargs": kwargs})), request_path)
        )
    return request_path


async def invoke(
    chute: Chute,
    user_id: str,
    path: str,
    function: str,
    stream: bool,
    args: str,
    kwargs: str,
    manager: LeastConnManager,
    parent_invocation_id: str,
    metrics: dict = {},
    request: Request = None,
):
    """
    Helper to actual perform function invocations, retrying when a target fails.
    """
    chute_id = chute.chute_id
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

    # Randomly sample for validation purposes.
    request_path = await _sample_request(chute_id, parent_invocation_id, args, kwargs)

    partition_suffix = None
    rate_limited = 0
    avoid = []
    for attempt_idx in range(5):
        async with manager.get_target(avoid=avoid) as target:
            if not target:
                logger.warning(f"No targets found for {chute_id=}, sleeping and re-trying...")
                await asyncio.sleep(0.5)
                continue

            invocation_id = str(uuid.uuid4())
            async with get_session() as session:
                result = await session.execute(
                    TRACK_INVOCATION,
                    {
                        "parent_invocation_id": parent_invocation_id,
                        "invocation_id": invocation_id,
                        "function_name": function,
                        "chute_id": chute.chute_id,
                        "chute_user_id": chute.user_id,
                        "user_id": user_id,
                        "image_id": chute.image_id,
                        "image_user_id": chute.image.user_id,
                        "instance_id": target.instance_id,
                        "miner_uid": target.miner_uid,
                        "miner_hotkey": target.miner_hotkey,
                        "request_path": request_path,
                        "compute_multiplier": NodeSelector(
                            **chute.node_selector
                        ).compute_multiplier,
                    },
                )
                partition_suffix = result.scalar()
                await session.commit()

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
                response_data = []
                async for data in _invoke_one(chute, path, stream, args, kwargs, target, metrics):
                    yield sse({"result": data})
                    if request_path:
                        response_data.append(data)

                async with get_session() as session:
                    # Save the response if we're randomly sampling this one.
                    response_path = None
                    if request_path:
                        response_path = request_path.replace("/request", "/response")
                        asyncio.create_task(
                            _s3_upload(io.BytesIO(json.dumps(response_data)), response_path)
                        )

                    # Mark the invocation as complete.
                    result = await session.execute(
                        text(UPDATE_INVOCATION.format(suffix=partition_suffix)),
                        {
                            "chute_id": chute_id,
                            "invocation_id": invocation_id,
                            "response_path": response_path,
                            "metrics": json.dumps(metrics).decode(),
                        },
                    )
                    try:
                        await settings.redis_client.delete(
                            f"consecutive_failures:{target.instance_id}"
                        )
                    except Exception as exc:
                        logger.warning(f"Error clearing consecutive failures: {exc}")

                    # Calculate the credits used and deduct from user's balance.
                    # For LLMs and Diffusion chutes, we use custom per token/image step pricing, otherwise
                    # it's just based on time used.
                    compute_units = result.scalar_one_or_none()
                    if compute_units and not request.state.free_invocation:
                        balance_used = None

                        # Track any discounts.
                        discount = 0.0
                        # A negative discount just makes the chute more than our typical pricing,
                        # e.g. for chutes that have a concurrency of one and we can't really operate
                        # efficiently with the normal pricing.
                        if chute.discount and -3 < chute.discount <= 1:
                            discount = chute.discount

                        if discount < 1.0:
                            # Calculate standard hourly price.
                            hourly_price = await selector_hourly_price(chute.node_selector)

                            # LLM per token pricing.
                            if chute.standard_template == "vllm" and metrics:
                                if output_tokens := metrics.get("ot"):
                                    tokens = output_tokens + metrics.get("it", 0)
                                    balance_used = (
                                        tokens
                                        / 1000000.0
                                        * hourly_price
                                        * LLM_PRICE_MULT_PER_MILLION
                                    )
                                    balance_used -= balance_used * discount
                                    logger.info(
                                        f"BALANCE: LLM token pricing: ${hourly_price * LLM_PRICE_MULT_PER_MILLION:.4f}/million for {chute.name}, {balance_used=} for {tokens=} {discount=}"
                                    )

                            # Diffusion per step pricing.
                            elif chute.standard_template == "diffusion":
                                if steps := metrics.get("steps"):
                                    balance_used = (
                                        steps * hourly_price * DIFFUSION_PRICE_MULT_PER_STEP
                                    )
                                    balance_used -= balance_used * discount
                                    logger.info(
                                        f"BALANCE: Diffusion step pricing: ${hourly_price * DIFFUSION_PRICE_MULT_PER_STEP:.4f}/step for {chute.name}, {balance_used=} {discount=}"
                                    )

                            # Sanity check and default pricing if not a standard template.
                            default_balance_used = compute_units * COMPUTE_UNIT_PRICE_BASIS / 3600
                            default_balance_used -= default_balance_used * discount
                            if balance_used and balance_used > default_balance_used:
                                logger.info(
                                    f"BALANCE: Balance per token/step exceeded default calculation for {chute.name}: {balance_used=} vs {default_balance_used=} {discount=}"
                                )
                                balance_used = default_balance_used
                            if balance_used is None:
                                balance_used = default_balance_used
                                logger.info(
                                    f"BALANCE: Defaulting to standard compute hourly pricing balance deduction for {chute.name}: {balance_used=} {discount=}"
                                )

                            # Deduct the balance used.
                            result = await session.execute(
                                update(User)
                                .where(User.user_id == user_id)
                                .where(
                                    User.permissions_bitmask.op("&")(
                                        Permissioning.free_account.bitmask
                                    )
                                    == 0
                                )
                                .values(balance=User.balance - balance_used)
                                .returning(User.balance)
                            )
                            new_balance = result.scalar_one_or_none()
                            if new_balance is not None:
                                logger.info(
                                    f"Deducted ${balance_used:.12f} from {user_id=}, new balance = ${new_balance:.12f}"
                                )
                        await session.commit()

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
                    await asyncio.sleep(0.5)
                    rate_limited += 1
                elif isinstance(exc, BadRequest):
                    error_message = "BAD_REQUEST"
                    error_detail = str(exc)
                elif isinstance(exc, KeyExchangeRequired):
                    error_message = "KEY_EXCHANGE_REQUIRED"

                async with get_session() as session:
                    await session.execute(
                        text(UPDATE_INVOCATION_ERROR.format(suffix=partition_suffix)),
                        {
                            "invocation_id": invocation_id,
                            "error_message": error_message,
                        },
                    )

                    # Handle the case where encryption V2 is in use and the instance needs a new key exchange.
                    if error_message == "KEY_EXCHANGE_REQUIRED":
                        # NOTE: Could probably just re-validate rather than deleting the instance, but this ensures no shenanigans are afoot.
                        await session.execute(
                            text("DELETE FROM instances WHERE instance_id = :instance_id"),
                            {"instance_id": target.instance_id},
                        )
                        await session.execute(
                            text(
                                "UPDATE instance_audit SET deletion_reason = 'miner responded with 426 upgrade required, new symmetric key needed' WHERE instance_id = :instance_id"
                            ),
                            {"instance_id": target.instance_id},
                        )
                        await session.commit()
                        event_data = {
                            "reason": "instance_deleted",
                            "message": f"Instance {target.instance_id} of miner {target.miner_hotkey} responded with a 426 error, indicating a new key exchange is required.",
                            "data": {
                                "chute_id": target.chute_id,
                                "instance_id": target.instance_id,
                                "miner_hotkey": target.miner_hotkey,
                            },
                        }
                        asyncio.create_task(
                            settings.redis_client.publish("events", json.dumps(event_data).decode())
                        )
                        event_data["filter_recipients"] = [target.miner_hotkey]
                        asyncio.create_task(
                            settings.redis_client.publish(
                                "miner_broadcast", json.dumps(event_data).decode()
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

                        if (
                            consecutive_failures
                            and consecutive_failures >= settings.consecutive_failure_limit
                        ):
                            await session.execute(
                                text("DELETE FROM instances WHERE instance_id = :instance_id"),
                                {"instance_id": target.instance_id},
                            )
                            await session.execute(
                                text(
                                    f"UPDATE instance_audit SET deletion_reason = 'max consecutive failures {consecutive_failures} reached' WHERE instance_id = :instance_id"
                                ),
                                {"instance_id": target.instance_id},
                            )
                            await session.commit()
                            event_data = {
                                "reason": "instance_deleted",
                                "message": f"Instance {target.instance_id} of miner {target.miner_hotkey} has reached the consecutive failure limit of {settings.consecutive_failure_limit} and has been deleted.",
                                "data": {
                                    "chute_id": target.chute_id,
                                    "instance_id": target.instance_id,
                                    "miner_hotkey": target.miner_hotkey,
                                },
                            }
                            asyncio.create_task(
                                settings.redis_client.publish(
                                    "events", json.dumps(event_data).decode()
                                )
                            )

                            # Miner notification.
                            event_data["filter_recipients"] = [target.miner_hotkey]
                            asyncio.create_task(
                                settings.redis_client.publish(
                                    "miner_broadcast", json.dumps(event_data).decode()
                                )
                            )
                if error_message == "BAD_REQUEST":
                    logger.warning(
                        f"instance_id={target.instance_id} [chute_id={target.chute_id}]: bad request {error_detail}"
                    )
                    yield sse({"error": f"Invalid request: {error_detail}"})
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
    if rate_limited == attempt_idx - 1:
        logger.warning(f"All miners are at max capacity: {rate_limited=} {chute.name=}")
        yield sse({"error": "rate_limit", "detail": "All miners are all maximum capacity"})
    else:
        logger.error(f"Failed to query all miners after {attempt_idx + 1} attempts")
        yield sse({"error": "exhausted all available targets to no avail"})


async def get_vllm_models(request: Request):
    """
    Get the combined /v1/models return value for chutes that are public and belong to chutes user.
    """
    cached = await settings.redis_client.get("vllm_models")
    if cached:
        return json.loads(cached)

    @backoff.on_exception(
        backoff.constant,
        Exception,
        jitter=None,
        interval=3,
        max_tries=5,
    )
    async def _get_models_data(chute):
        nonlocal request
        cached = await settings.redis_client.get(
            f"vllm_model_info:{chute.chute_id}:{chute.version}"
        )
        if cached:
            return json.loads(cached)
        async with get_session() as session:
            target_manager = await get_chute_target_manager(session, chute.chute_id, max_wait=0)
            if not target_manager or not target_manager.instances:
                return None
        args = base64.b64encode(gzip.compress(pickle.dumps(tuple()))).decode()
        kwargs = base64.b64encode(gzip.compress(pickle.dumps({}))).decode()
        inv_id = str(uuid.uuid4())
        result = None
        async for chunk in invoke(
            chute,
            await chutes_user_id(),
            "/get_models",
            "get_models",
            False,
            args,
            kwargs,
            target_manager,
            inv_id,
            metrics={},
            request=request,
        ):
            if chunk.startswith('data: {"result"'):
                result = json.loads(chunk[6:])["result"]
        await settings.redis_client.set(
            f"vllm_model_info:{chute.chute_id}:{chute.version}", json.dumps(result), ex=3600
        )
        return result

    async with get_session() as session:
        chutes = (
            (
                await session.execute(
                    select(Chute)
                    .where(
                        Chute.standard_template == "vllm",
                        Chute.chutes_version >= "0.2.15",
                        Chute.public.is_(True),
                        Chute.user_id == await chutes_user_id(),
                    )
                    .options(selectinload(Chute.instances))
                )
            )
            .unique()
            .scalars()
        )
        model_data = {"object": "list", "data": []}
        results = await asyncio.gather(*[_get_models_data(chute) for chute in chutes])
        for info in results:
            if not info:
                continue
            logger.info(f"Trying to add to vllm models data: {info}")
            data = info["json"]["data"][0]
            model_data["data"].append(data)
    await settings.redis_client.set("vllm_models", json.dumps(model_data), ex=300)
    return model_data


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
    return int(len(json.dumps(body)) / 4)


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
