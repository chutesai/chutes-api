"""
E2E encryption router — instance discovery with nonces and encrypted invocation relay.
"""

import math
import time
import uuid
import secrets
import asyncio
import random
import traceback
import httpx
import orjson as json
from datetime import datetime
from loguru import logger
from fastapi import APIRouter, Depends, Header, HTTPException, Request, status
from fastapi.responses import StreamingResponse, Response
from api.config import settings, get_subscription_tier
from api.user.service import get_current_user
from api.user.schemas import User, PriceOverride, InvocationDiscount, InvocationQuota
from api.chute.util import (
    get_one,
    is_shared,
    get_miner_session,
    get_mtoken_price,
    update_usage_data,
    safe_store_invocation,
    selector_hourly_price,
)
from api.chute.schemas import NodeSelector
from api.instance.util import (
    load_chute_targets,
    load_chute_target,
    is_instance_disabled,
    MANAGERS,
    update_shutdown_timestamp,
    clear_instance_disable_state,
)
from api.util import (
    encrypt_instance_request,
    decrypt_instance_response,
    has_legacy_private_billing,
)
from api.miner_client import sign_request
from api.rate_limit import rate_limit
from api.gpu import COMPUTE_UNIT_PRICE_BASIS
from api.constants import DIFFUSION_PRICE_MULT_PER_STEP
from api.user.service import chutes_user_id, subnet_role_accessible
from api.invocation.util import (
    resolve_rate_limit_headers,
    build_response_headers,
    check_quota_and_balance,
)
from api.metrics.capacity import track_request_completed, track_capacity

router = APIRouter()

# Lua script for atomic nonce consumption: verify instance_id match + delete
NONCE_CONSUME_LUA = """
local val = redis.call('HGET', KEYS[1], ARGV[1])
if val == false then return nil end
if val ~= ARGV[2] then return nil end
redis.call('HDEL', KEYS[1], ARGV[1])
return val
"""

NONCES_PER_INSTANCE = 10
NONCE_REDIS_TTL = 75
NONCE_CLIENT_TTL = 60
MAX_INSTANCES_RETURNED = 5


@router.get("/instances/{chute_id}")
async def get_e2e_instances(
    chute_id: str,
    current_user: User = Depends(get_current_user(raise_not_found=True, allow_api_key=True)),
    _rate_limit: None = Depends(rate_limit("e2e_instances", requests_per_minute=10)),
):
    """
    Discover E2E-capable instances for a chute and get nonces for invocation.
    """
    # Load chute and verify access.
    chute = await get_one(chute_id)
    if not chute:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Chute not found")
    if not (
        chute.public
        or chute.user_id == current_user.user_id
        or await is_shared(chute.chute_id, current_user.user_id)
        or subnet_role_accessible(chute, current_user)
    ):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Chute not found")

    # Load active instances.
    instances = await load_chute_targets(chute_id, nonce=0)
    if not instances:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No active instances found for this chute",
        )

    # Filter to E2E-capable, non-disabled instances.
    eligible = []
    for inst in instances:
        e2e_pubkey = (inst.extra or {}).get("e2e_pubkey")
        if not e2e_pubkey:
            continue
        if await is_instance_disabled(inst.instance_id):
            continue
        eligible.append(inst)

    if not eligible:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No E2E-capable instances available",
        )

    # Select random subset.
    selected = random.sample(eligible, min(len(eligible), MAX_INSTANCES_RETURNED))

    # Generate nonces and store in Redis hash.
    user_id = current_user.user_id
    hash_key = f"e2e_nonces:{user_id}:{chute_id}"
    redis = settings.redis_client

    # Build nonce mappings: token -> instance_id
    nonce_map = {}  # token -> instance_id
    instance_nonces = {}  # instance_id -> [tokens]
    for inst in selected:
        tokens = []
        for _ in range(NONCES_PER_INSTANCE):
            token = secrets.token_urlsafe(24)
            nonce_map[token] = inst.instance_id
            tokens.append(token)
        instance_nonces[inst.instance_id] = tokens

    # Store all nonces in a single HSET call, then set TTL.
    if nonce_map:
        await redis.hset(hash_key, mapping=nonce_map)
        await redis.expire(hash_key, NONCE_REDIS_TTL)

    now = int(time.time())
    result_instances = []
    for inst in selected:
        result_instances.append(
            {
                "instance_id": inst.instance_id,
                "e2e_pubkey": (inst.extra or {}).get("e2e_pubkey"),
                "nonces": instance_nonces[inst.instance_id],
            }
        )

    return {
        "instances": result_instances,
        "nonce_expires_in": NONCE_CLIENT_TTL,
        "nonce_expires_at": now + NONCE_CLIENT_TTL,
    }


@router.post("/invoke")
async def e2e_invoke(
    request: Request,
    current_user: User = Depends(get_current_user(raise_not_found=True, allow_api_key=True)),
    x_chute_id: str = Header(..., alias="X-Chute-Id"),
    x_instance_id: str = Header(..., alias="X-Instance-Id"),
    x_e2e_nonce: str = Header(..., alias="X-E2E-Nonce"),
    x_e2e_stream: str = Header("false", alias="X-E2E-Stream"),
    x_e2e_path: str = Header("/", alias="X-E2E-Path"),
):
    """
    Relay an E2E encrypted invocation to a specific instance.
    """
    user_id = current_user.user_id
    chute_id = x_chute_id
    instance_id = x_instance_id
    nonce_token = x_e2e_nonce
    is_stream = x_e2e_stream.lower() == "true"

    # Validate + consume nonce atomically via Lua script.
    hash_key = f"e2e_nonces:{user_id}:{chute_id}"
    redis = settings.redis_client
    result = await redis.eval(NONCE_CONSUME_LUA, 1, hash_key, nonce_token, instance_id)
    if result is None:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Invalid, expired, or already-used nonce",
        )

    # Load instance and verify it's valid.
    instance = await load_chute_target(instance_id)
    if not instance:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Instance not found",
        )
    if instance.chute_id != chute_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Instance does not belong to the specified chute",
        )
    if not instance.active or not instance.verified:
        raise HTTPException(
            status_code=status.HTTP_410_GONE,
            detail="Instance is no longer active",
        )

    # Load chute and verify access.
    chute = await get_one(chute_id)
    if not chute:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Chute not found")
    if not (
        chute.public
        or chute.user_id == current_user.user_id
        or await is_shared(chute.chute_id, current_user.user_id)
        or subnet_role_accessible(chute, current_user)
    ):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Chute not found")
    if chute.disabled:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="This chute is currently disabled.",
        )

    resolve_rate_limit_headers(request, current_user, chute)
    await check_quota_and_balance(request, current_user, chute)

    # Read raw E2E blob from request body.
    e2e_blob = await request.body()
    if not e2e_blob:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Empty request body",
        )

    # Transport-encrypt the raw E2E blob.
    encrypted_payload, _ = await asyncio.to_thread(encrypt_instance_request, e2e_blob, instance)
    # Encrypt the path for routing.
    request_path = x_e2e_path if x_e2e_path else "/"
    encrypted_path, _ = await asyncio.to_thread(
        encrypt_instance_request, request_path.ljust(24, "?"), instance, True
    )

    # Connection tracking (INCR/DECR, same as LeastConnManager.get_target).
    conn_id = str(uuid.uuid4())
    manager = MANAGERS.get(chute_id)
    if manager:
        try:
            key = f"cc:{chute_id}:{instance_id}"
            pipe = manager.redis_client.client.pipeline()
            pipe.incr(key)
            pipe.expire(key, manager.connection_expiry)
            await asyncio.wait_for(pipe.execute(), timeout=3.0)
        except Exception as e:
            logger.warning(f"E2E: Error tracking connection: {e}")

    invocation_id = str(uuid.uuid4())
    parent_invocation_id = str(uuid.uuid4())
    started_at = time.time()

    session = None
    pooled = True
    response = None
    streaming_started = False
    try:
        # Send to instance.
        session, pooled = await get_miner_session(instance, timeout=1800)
        headers, payload_string = sign_request(
            miner_ss58=instance.miner_hotkey, payload=encrypted_payload
        )
        headers["X-E2E-Encrypted"] = "true"
        if is_stream:
            headers["X-E2E-Stream"] = "true"

        e2e_timeout = httpx.Timeout(connect=10.0, read=1800.0, write=30.0, pool=10.0)
        if is_stream:
            response = await session.send(
                session.build_request(
                    "POST",
                    f"/{encrypted_path}",
                    content=payload_string,
                    headers=headers,
                    timeout=e2e_timeout,
                ),
                stream=True,
            )
        else:
            response = await session.post(
                f"/{encrypted_path}",
                content=payload_string,
                headers=headers,
                timeout=e2e_timeout,
            )

        # Handle transport-level errors.
        if response.status_code == 400:
            if is_stream:
                await response.aread()
            raise HTTPException(status_code=400, detail=response.text)
        if response.status_code == 429:
            if is_stream:
                await response.aread()
            raise HTTPException(
                status_code=429,
                detail="Instance is at maximum capacity, try again later",
            )
        if response.status_code == 426:
            if is_stream:
                await response.aread()
            raise HTTPException(
                status_code=502,
                detail="Instance requires key exchange, try a different instance",
            )
        if response.status_code != 200:
            if is_stream:
                await response.aread()
            raise HTTPException(
                status_code=response.status_code,
                detail=f"Instance returned status {response.status_code}",
            )

        multiplier = NodeSelector(**chute.node_selector).compute_multiplier
        if chute.boost:
            multiplier *= chute.boost

        if is_stream:
            streaming_started = True
            return StreamingResponse(
                _stream_e2e_response(
                    response,
                    session,
                    instance,
                    chute,
                    current_user,
                    multiplier,
                    started_at,
                    invocation_id,
                    parent_invocation_id,
                    manager,
                    conn_id,
                    request,
                    pooled,
                ),
                media_type="text/event-stream",
                headers=build_response_headers(request),
            )
        else:
            # Non-streaming: read full response, transport-decrypt, relay.
            raw_body = response.content
            decrypted = await asyncio.to_thread(decrypt_instance_response, raw_body, instance)

            # Time-based billing.
            duration = time.time() - started_at
            compute_units = multiplier * math.ceil(duration)
            await _do_billing(
                chute,
                current_user,
                instance,
                duration,
                compute_units,
                multiplier,
                None,
                invocation_id,
                parent_invocation_id,
                request,
                manager,
            )

            # Cleanup.
            asyncio.create_task(
                settings.redis_client.delete(f"consecutive_failures:{instance.instance_id}")
            )
            asyncio.create_task(clear_instance_disable_state(instance.instance_id))

            return Response(
                content=decrypted,
                media_type="application/octet-stream",
                headers=build_response_headers(request),
            )

    except HTTPException:
        raise
    except Exception as exc:
        logger.error(f"E2E invoke error: {exc}\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail="Internal error during E2E invocation")
    finally:
        if not streaming_started:
            # For streaming, cleanup happens in the generator.
            await _cleanup(session, response, manager, chute_id, instance_id, conn_id, pooled)


async def _stream_e2e_response(
    response,
    session,
    instance,
    chute,
    user,
    multiplier,
    started_at,
    invocation_id,
    parent_invocation_id,
    manager,
    conn_id,
    request,
    pooled,
):
    """
    Stream E2E response chunks, extracting usage events for billing.

    E2E streaming chunks are SSE events sent in plaintext over the mTLS
    tunnel (no per-chunk transport encryption). The events are already
    E2E-encrypted by the instance, so we relay them directly and only
    parse usage data for billing.
    """
    metrics = {}
    chunk_count = 0
    try:
        async for raw_chunk in response.aiter_lines():
            # aiter_lines() strips newlines; relay every line (including
            # empty ones) with a trailing \n to reconstruct the original
            # SSE framing (data: {...}\n\n).
            chunk_str = (
                raw_chunk.decode("utf-8", errors="replace")
                if isinstance(raw_chunk, bytes)
                else raw_chunk
            )

            # Parse non-empty SSE data lines to extract usage for billing.
            if chunk_str.startswith("data: "):
                try:
                    obj = json.loads(chunk_str[6:].encode())
                    if isinstance(obj, dict) and "usage" in obj:
                        usage = obj["usage"]
                        metrics["it"] = usage.get("prompt_tokens", 0)
                        metrics["ot"] = usage.get("completion_tokens", 0)
                        metrics["ct"] = (usage.get("prompt_tokens_details") or {}).get(
                            "cached_tokens", 0
                        )
                except Exception:
                    pass

            # Periodic disconnect check (every 5 data lines).
            if chunk_str:
                chunk_count += 1
            if chunk_count % 5 == 0 and chunk_count > 0 and await request.is_disconnected():
                logger.info(
                    f"E2E client disconnected mid-stream for {chute.name} {instance.instance_id=}"
                )
                await response.aclose()
                break

            # Relay the line with newline to preserve SSE framing.
            yield f"{chunk_str}\n".encode()

        # Billing after stream completes.
        duration = time.time() - started_at
        compute_units = multiplier * math.ceil(duration)
        await _do_billing(
            chute,
            user,
            instance,
            duration,
            compute_units,
            multiplier,
            metrics if metrics else None,
            invocation_id,
            parent_invocation_id,
            request,
            manager,
        )

        # Clear failure tracking on success.
        asyncio.create_task(
            settings.redis_client.delete(f"consecutive_failures:{instance.instance_id}")
        )
        asyncio.create_task(clear_instance_disable_state(instance.instance_id))

    except Exception as exc:
        logger.error(f"E2E stream error: {exc}\n{traceback.format_exc()}")
        raise
    finally:
        await _cleanup(
            session, response, manager, chute.chute_id, instance.instance_id, conn_id, pooled
        )


async def _do_billing(
    chute,
    user,
    instance,
    duration,
    compute_units,
    multiplier,
    metrics,
    invocation_id,
    parent_invocation_id,
    request,
    manager=None,
):
    """
    Handle billing for an E2E invocation.
    """
    user_id = user.user_id
    balance_used = 0.0
    override_applied = False
    free_invocation = getattr(request.state, "free_invocation", False)

    if compute_units:
        hourly_price = await selector_hourly_price(chute.node_selector)

        # Per megatoken pricing for vLLM chutes.
        if chute.standard_template == "vllm" and metrics and metrics.get("it"):
            per_million_in, per_million_out, cache_discount = await get_mtoken_price(
                user_id, chute.chute_id
            )
            prompt_tokens = metrics.get("it", 0) or 0
            output_tokens = metrics.get("ot", 0) or 0
            cached_tokens = metrics.get("ct", 0) or 0
            balance_used = (
                prompt_tokens / 1000000.0 * per_million_in
                - cached_tokens / 1000000.0 * per_million_in * cache_discount
                + output_tokens / 1000000.0 * per_million_out
            )
            override_applied = True

        elif (price_override := await PriceOverride.get(user_id, chute.chute_id)) is not None:
            if chute.standard_template == "diffusion" and price_override.per_step is not None:
                balance_used = (metrics.get("steps", 0) or 0) * price_override.per_step
                override_applied = True
            elif price_override.per_request is not None:
                balance_used = price_override.per_request
                override_applied = True

        # If no override was applied, use standard pricing.
        if not override_applied:
            discount = 0.0
            if chute.discount and -3 < chute.discount <= 1:
                discount = chute.discount
            if discount < 1.0:
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
                    balance_used = default_balance_used

        # User discounts.
        if balance_used and not override_applied:
            user_discount = await InvocationDiscount.get(user_id, chute.chute_id)
            if user_discount:
                balance_used -= balance_used * user_discount

    # Subscriber paygo discount (when quota exceeded).
    sub_paygo_discount = getattr(request.state, "subscriber_paygo_discount", 0.0)
    if balance_used and sub_paygo_discount and not override_applied:
        balance_used -= balance_used * sub_paygo_discount

    # Don't charge for private instances.
    if (
        not chute.public
        and not has_legacy_private_billing(chute)
        and chute.user_id != await chutes_user_id()
    ):
        balance_used = 0

    # Always track paygo-equivalent usage for subscription cap accounting.
    paygo_equivalent = balance_used

    # Free invocations are not charged but still count against subscription caps.
    if free_invocation:
        balance_used = 0

    # Keep subscription cap cache warm for near-real-time gating.
    if free_invocation and paygo_equivalent > 0:
        sub_quota = await InvocationQuota.get(user_id, chute.chute_id)
        if get_subscription_tier(sub_quota) is not None:
            month_key = f"sub_cap_m:{datetime.now().strftime('%Y%m')}:{user_id}"
            four_hour_bucket = int(time.time()) // (4 * 3600)
            four_hour_key = f"sub_cap_4h:{four_hour_bucket}:{user_id}"
            asyncio.create_task(settings.redis_client.incrbyfloat(month_key, paygo_equivalent))
            asyncio.create_task(settings.redis_client.expire(month_key, 35 * 86400))
            asyncio.create_task(settings.redis_client.incrbyfloat(four_hour_key, paygo_equivalent))
            asyncio.create_task(settings.redis_client.expire(four_hour_key, 5 * 3600))

    if metrics is None:
        metrics = {}
    metrics["b"] = balance_used

    # Store invocation record.
    asyncio.create_task(
        safe_store_invocation(
            parent_invocation_id,
            invocation_id,
            chute.chute_id,
            chute.user_id,
            "e2e_invoke",
            user_id,
            chute.image_id,
            chute.image.user_id,
            instance.instance_id,
            instance.miner_uid,
            instance.miner_hotkey,
            duration,
            multiplier,
            error_message=None,
            metrics=metrics,
        )
    )

    # Deduct balance.
    asyncio.create_task(
        update_usage_data(
            user_id,
            chute.chute_id,
            balance_used,
            metrics if chute.standard_template == "vllm" else None,
            compute_time=duration,
            paygo_amount=paygo_equivalent,
        )
    )

    # Increment quota usage value.
    if (
        free_invocation
        and (chute.discount or 0) < 1.0
        and (
            chute.public
            or has_legacy_private_billing(chute)
            or chute.user_id == await chutes_user_id()
        )
    ):
        key = await InvocationQuota.quota_key(user.user_id, chute.chute_id)
        asyncio.create_task(settings.redis_client.incrbyfloat(key, 1.0))

    # Prometheus metrics.
    track_request_completed(chute.chute_id)
    if manager and hasattr(manager, "mean_count"):
        try:
            instance_util = getattr(manager, "_last_instance_utilization", None)
            if instance_util is not None:
                instance_util = float(instance_util)
            await track_capacity(
                chute.chute_id,
                manager.mean_count or 0,
                chute.concurrency or 1,
                instance_utilization=instance_util,
            )
        except Exception:
            pass

    # Push back instance shutdown for private chutes.
    if (
        not chute.public
        and not has_legacy_private_billing(chute)
        and chute.user_id != await chutes_user_id()
    ):
        asyncio.create_task(update_shutdown_timestamp(instance.instance_id))


async def _cleanup(session, response, manager, chute_id, instance_id, conn_id, pooled=True):
    """
    Clean up httpx response and connection tracking.
    """
    if response:
        try:
            await response.aclose()
        except Exception:
            pass
    if not pooled and session:
        try:
            await session.aclose()
        except Exception:
            pass
    if manager:
        try:
            key = f"cc:{chute_id}:{instance_id}"

            async def _decr():
                val = await manager.redis_client.client.decr(key)
                if val < 0:
                    await manager.redis_client.client.set(key, 0, ex=manager.connection_expiry)

            await asyncio.shield(_decr())
        except Exception as e:
            logger.warning(f"E2E: Error cleaning up connection {conn_id}: {e}")
