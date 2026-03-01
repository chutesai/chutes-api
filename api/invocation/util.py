"""
Helpers for invocations.
"""

import os
import asyncio
import hashlib
import time
import aiohttp
import orjson as json
from datetime import date, datetime, timedelta, timezone
from typing import Dict
from async_lru import alru_cache
from loguru import logger
from fastapi import HTTPException, status
from api.gpu import COMPUTE_UNIT_PRICE_BASIS
from api.config import (
    settings,
    get_subscription_tier,
    SUBSCRIPTION_MONTHLY_CAP_MULTIPLIER,
    SUBSCRIPTION_4H_CAP_MULTIPLIER,
    SUBSCRIPTION_PAYGO_DISCOUNTS,
    FOUR_HOUR_CHUNKS_PER_MONTH,
)
from api.database import get_session, get_inv_session
from api.chute.schemas import NodeSelector
from api.permissions import Permissioning
from api.util import has_legacy_private_billing
from sqlalchemy import text

TOKEN_METRICS_QUERY = """
INSERT INTO vllm_metrics
SELECT * FROM get_llm_metrics('2025-01-30', DATE_TRUNC('day', NOW())::date)
ORDER BY date DESC, name;
"""

DIFFUSION_METRICS_QUERY = """
INSERT INTO diffusion_metrics
SELECT * FROM get_diffusion_metrics('2025-01-30', DATE_TRUNC('day', NOW())::date)
ORDER BY date DESC, name;
"""

PROMETHEUS_URL = os.getenv("PROMETHEUS_URL", "http://prometheus-server")


@alru_cache(maxsize=500, ttl=1200)
async def get_sponsored_chute_ids(user_id: str) -> frozenset[str]:
    """Get the set of chute IDs with active sponsorships for a given user."""
    redis_key = f"sponsored_chutes:{user_id}"
    cached = await settings.redis_client.get(redis_key)
    if cached is not None:
        return frozenset(json.loads(cached))

    query = text("""
        SELECT sc.chute_id
        FROM inference_sponsorships isp
        JOIN sponsorship_chutes sc ON isp.id = sc.sponsorship_id
        WHERE isp.user_id = :user_id
        AND isp.start_date <= CURRENT_DATE
        AND (isp.end_date IS NULL OR isp.end_date >= CURRENT_DATE)
    """)
    async with get_session(readonly=True) as session:
        result = await session.execute(query, {"user_id": user_id})
        chute_ids = frozenset(row[0] for row in result)

    await settings.redis_client.set(redis_key, json.dumps(list(chute_ids)), ex=1200)
    return chute_ids


@alru_cache(maxsize=1, ttl=1200)
async def get_all_sponsored_chute_ids() -> frozenset[str]:
    """Get the set of all chute IDs with any active sponsorship."""
    redis_key = "all_sponsored_chutes"
    cached = await settings.redis_client.get(redis_key)
    if cached is not None:
        return frozenset(json.loads(cached))

    query = text("""
        SELECT DISTINCT sc.chute_id
        FROM inference_sponsorships isp
        JOIN sponsorship_chutes sc ON isp.id = sc.sponsorship_id
        WHERE isp.start_date <= CURRENT_DATE
        AND (isp.end_date IS NULL OR isp.end_date >= CURRENT_DATE)
    """)
    async with get_session(readonly=True) as session:
        result = await session.execute(query)
        chute_ids = frozenset(row[0] for row in result)

    await settings.redis_client.set(redis_key, json.dumps(list(chute_ids)), ex=1200)
    return chute_ids


async def query_prometheus(
    queries: Dict[str, str], prometheus_url: str = PROMETHEUS_URL
) -> Dict[str, Dict[str, float]]:
    """
    Execute multiple Prometheus queries concurrently and return results keyed by chute_id.
    """
    results = {}

    async def query_single(session: aiohttp.ClientSession, name: str, query: str) -> tuple:
        try:
            async with session.get(
                f"{prometheus_url}/api/v1/query",
                params={"query": query},
                timeout=aiohttp.ClientTimeout(total=30),
            ) as response:
                response.raise_for_status()
                data = await response.json()
                if data["status"] == "success" and data["data"]["result"]:
                    chute_results = {}
                    for result in data["data"]["result"]:
                        chute_id = result["metric"].get("chute_id")
                        value = float(result["value"][1])
                        if chute_id:
                            chute_results[chute_id] = value
                    return (name, chute_results)
                return (name, {})
        except Exception as e:
            logger.warning(f"Error querying Prometheus for {name}: {e}")
            return (name, {})

    try:
        async with aiohttp.ClientSession() as session:
            import asyncio

            tasks = [query_single(session, name, query) for name, query in queries.items()]
            query_results = await asyncio.gather(*tasks)
            for name, result in query_results:
                results[name] = result
    except Exception as e:
        logger.error(f"Failed to query Prometheus: {e}")

    return results


async def gather_metrics(interval: str = "1 hour"):
    """
    Generate chute metrics from Prometheus (utilization, request counts, rate limits).
    Falls back to cached data if Prometheus is unavailable.
    """
    cache_key = f"gather_metrics_{interval}"
    cached = await settings.redis_client.get(cache_key)
    if cached:
        rows = json.loads(cached)
        for item in rows:
            yield item
        return

    # Map interval string to Prometheus duration
    interval_map = {"1 hour": "1h", "1 day": "1d", "1 week": "7d"}
    prom_interval = interval_map.get(interval, "1h")

    # Query Prometheus for metrics matching what autoscaler uses
    queries = {
        "utilization": f"avg by (chute_id) (avg_over_time(utilization[{prom_interval}]))",
        "completed": f"sum by (chute_id) (increase(requests_completed_total[{prom_interval}]))",
        "rate_limited": f"sum by (chute_id) (increase(requests_rate_limited_total[{prom_interval}]))",
        "usage_usd": f"sum by (chute_id) (increase(usage_usd_total[{prom_interval}]))",
        "compute_seconds": f"sum by (chute_id) (increase(compute_seconds_total[{prom_interval}]))",
    }

    prom_results = await query_prometheus(queries)

    # Get all chute IDs and their node_selectors from DB
    chute_data = {}
    async with get_session() as session:
        result = await session.execute(
            text(
                """
                SELECT c.chute_id, c.name, c.node_selector
                FROM chutes c
                WHERE EXISTS (
                   SELECT FROM instances i WHERE i.chute_id = c.chute_id
                )
                """
            )
        )
        for row in result:
            try:
                node_selector = NodeSelector(**row.node_selector)
                compute_multiplier = node_selector.compute_multiplier
            except Exception:
                compute_multiplier = 1.0
            chute_data[row.chute_id] = {
                "name": row.name,
                "compute_multiplier": compute_multiplier,
            }

    # Get active instance counts per chute
    instance_counts = {}
    async with get_session() as session:
        result = await session.execute(
            text(
                """
                SELECT chute_id, COUNT(*) as instance_count
                FROM instances
                WHERE active = true AND verified = true
                GROUP BY chute_id
                """
            )
        )
        for row in result:
            instance_counts[row.chute_id] = int(row.instance_count)

    # Build metrics for each chute
    items = []
    all_chute_ids = set(chute_data.keys())
    for chute_id in prom_results.get("completed", {}).keys():
        all_chute_ids.add(chute_id)

    now = datetime.now(timezone.utc)
    for chute_id in all_chute_ids:
        if chute_id not in chute_data:
            continue

        compute_multiplier = chute_data[chute_id]["compute_multiplier"]
        completed = prom_results.get("completed", {}).get(chute_id, 0)
        rate_limited = prom_results.get("rate_limited", {}).get(chute_id, 0)
        utilization = prom_results.get("utilization", {}).get(chute_id, 0)
        usage_usd = prom_results.get("usage_usd", {}).get(chute_id, 0)
        compute_seconds = prom_results.get("compute_seconds", {}).get(chute_id, 0)

        item = {
            "chute_id": chute_id,
            "end_date": now.isoformat(),
            "start_date": now.isoformat(),  # Approximate, Prometheus handles the range
            "compute_multiplier": compute_multiplier,
            "total_invocations": int(completed),
            "total_compute_time": compute_seconds,
            "error_count": 0,  # Could add error metric if available
            "rate_limit_count": int(rate_limited),
            "instance_count": instance_counts.get(chute_id, 0),
            "utilization": utilization,
            "per_second_price_usd": compute_multiplier * COMPUTE_UNIT_PRICE_BASIS / 3600,
            "total_usage_usd": usage_usd,
        }
        items.append(item)
        yield item

    if items:
        await settings.redis_client.set(cache_key, json.dumps(items), ex=120)


def get_prompt_prefix_hashes(payload: dict) -> list:
    """
    Given an LLM prompt, generate a list of prefix hashes that can be used
    in prefix-aware routing for higher KV cache hit rate. Exponential size,
    powers of 2, using only characters not tokens for performance, as well
    as md5 since collections don't really matter here, cache miss is fine.
    """
    if (prompt := payload.get("prompt")) is None:
        if (messages := payload.get("messages")) is None:
            return []
        if all([isinstance(v, dict) and isinstance(v.get("content"), str) for v in messages]):
            prompt = "".join([v["content"] for v in messages])
        else:
            return []
    if not prompt or len(prompt) <= 1024:
        return []
    size = 1024
    hashes = []
    while len(prompt) > size:
        hashes.append((size, hashlib.md5(prompt[:size].encode()).hexdigest()))
        size *= 2
    return hashes[::-1]


async def generate_invocation_history_metrics():
    """
    Generate all vllm/diffusion metrics through time.
    """
    async with get_inv_session() as session:
        await session.execute(text("TRUNCATE TABLE vllm_metrics RESTART IDENTITY"))
        await session.execute(text("TRUNCATE TABLE diffusion_metrics RESTART IDENTITY"))
        await session.execute(text(TOKEN_METRICS_QUERY))
        await session.execute(text(DIFFUSION_METRICS_QUERY))


DEFAULT_RATE_LIMIT = 60


async def _initialize_quota_cache(cache_key: str) -> None:
    await settings.redis_client.incrbyfloat(cache_key, 0.0)


def resolve_rate_limit_headers(request, current_user, chute):
    """
    Set rate limit and invoice billing headers on request.state.
    """
    request.state.invoice_billing = current_user.has_role(Permissioning.invoice_billing)
    if not chute.public:
        request.state.rl_user = "inf"
    else:
        overrides = current_user.rate_limit_overrides or {}
        chute_rl = overrides.get(chute.chute_id)
        global_rl = overrides.get("*")
        if chute_rl is not None:
            request.state.rl_chute = chute_rl
            request.state.rl_user = global_rl if global_rl is not None else DEFAULT_RATE_LIMIT
        else:
            request.state.rl_user = global_rl if global_rl is not None else DEFAULT_RATE_LIMIT


def build_response_headers(request, base_headers=None):
    """
    Build response headers dict with quota, rate limit, and invoice billing info.
    """
    headers = dict(base_headers or {})
    if getattr(request.state, "quota_total", None) is not None:
        headers["X-Chutes-Quota-Total"] = str(int(request.state.quota_total))
        headers["X-Chutes-Quota-Used"] = str(int(request.state.quota_used))
        remaining = max(0, request.state.quota_total - request.state.quota_used)
        headers["X-Chutes-Quota-Remaining"] = str(int(remaining))
    rl_user = getattr(request.state, "rl_user", None)
    rl_chute = getattr(request.state, "rl_chute", None)
    if rl_user is not None:
        headers["X-Chutes-RL-User"] = str(rl_user) if rl_user == "inf" else str(int(rl_user))
    if rl_chute is not None:
        headers["X-Chutes-RL-Chute"] = str(int(rl_chute))
    if getattr(request.state, "invoice_billing", False):
        headers["X-Chutes-Invoice-Billing"] = "true"
    return headers


async def get_subscription_usage(
    user_id: str, period: str, since_expr: str, cache_ttl: int
) -> float:
    """
    Get accumulated paygo-equivalent usage covered by subscription (not already paid via paygo).
    Tries Redis cache first, falls back to usage_data table query.
    period: cache key suffix, e.g. "m:202602" or "4h:123456"
    since_expr: SQL expression for the start bound, e.g. "date_trunc('month', now())"
    cache_ttl: TTL in seconds for the Redis cache entry
    """
    cache_key = f"sub_cap_{period}:{user_id}"
    cached = await settings.redis_client.get(cache_key)
    if cached is not None:
        return float(cached.decode() if isinstance(cached, bytes) else cached)

    # Cache miss — query usage_data table using DB-relative time
    # Cap usage = total paygo equivalent - what they actually paid
    async with get_session(readonly=True) as session:
        result = await session.execute(
            text(f"""
                SELECT COALESCE(
                    SUM(
                        GREATEST(COALESCE(ud.paygo_amount, 0) - COALESCE(ud.amount, 0), 0)
                    ),
                    0
                )
                FROM usage_data ud
                JOIN chutes c ON c.chute_id = ud.chute_id
                WHERE ud.user_id = :user_id
                AND ud.bucket >= {since_expr}
                AND c.public IS TRUE
            """),
            {"user_id": user_id},
        )
        usage = max(float(result.scalar() or 0.0), 0.0)

    await settings.redis_client.set(cache_key, str(usage), ex=cache_ttl)
    return usage


async def check_quota_and_balance(request, current_user, chute):
    """
    Enforce free-model limits, private chute owner balance, and subscriber
    invocation quotas (including subscription caps).  Sets
    request.state.free_invocation and quota state used by
    build_response_headers().

    Must be called AFTER resolve_rate_limit_headers().
    """
    from api.user.schemas import InvocationQuota
    from api.user.service import chutes_user_id

    quota_date = date.today()

    # Fully discounted chutes are free but have usage caps for unprivileged users.
    if chute.discount == 1.0:
        request.state.free_invocation = True

        if current_user.permissions_bitmask == 0:
            effective_balance = (
                current_user.current_balance.effective_balance
                if current_user.current_balance
                else 0.0
            )
            unlimited = False
            if effective_balance >= 10:
                unlimited = True
            else:
                quota = await InvocationQuota.get(current_user.user_id, "__anychute__")
                if quota > 2000:
                    unlimited = True
            if not unlimited:
                free_usage = 0
                try:
                    qkey = f"free_usage:{quota_date}:{current_user.user_id}"
                    free_usage = await settings.redis_client.incr(qkey)
                    if free_usage <= 3:
                        tomorrow = datetime.combine(quota_date, datetime.min.time()) + timedelta(
                            days=1
                        )
                        exp = max(int((tomorrow - datetime.now()).total_seconds()), 1)
                        await settings.redis_client.expire(qkey, exp)
                except Exception as exc:
                    logger.warning(
                        f"Error checking free usage for {current_user.user_id=}: {str(exc)}"
                    )
                if free_usage > 100:
                    logger.warning(
                        f"{current_user.user_id=} {current_user.username=} has hit daily free limit: {chute.name=} {effective_balance=}"
                    )
                    raise HTTPException(
                        status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                        detail="Free models limit reached for today - maintain >= $10 balance or upgrade subscription to pro to unlock more.",
                    )

    elif current_user.user_id == settings.or_free_user_id:
        sponsored_chutes = await get_sponsored_chute_ids(current_user.user_id)
        if chute.chute_id not in sponsored_chutes:
            logger.warning(
                f"Attempt to invoke {chute.chute_id=} {chute.name=} from openrouter free account."
            )
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail="Invalid free model, please select from the updated list of current chutes free models",
            )

    # Prevent calling private chutes when the owner has no balance.
    origin_ip = request.headers.get("x-forwarded-for", "").split(",")[0]
    if (
        not chute.public
        and not has_legacy_private_billing(chute)
        and chute.user_id != await chutes_user_id()
    ):
        owner_balance = (
            chute.user.current_balance.effective_balance if chute.user.current_balance else 0.0
        )
        if owner_balance <= 0:
            logger.warning(
                f"Preventing execution of chute {chute.name=} {chute.chute_id=}, "
                f"creator has insufficient balance {owner_balance=}"
            )
            raise HTTPException(
                status_code=status.HTTP_402_PAYMENT_REQUIRED,
                detail="Chute unavailable because the creator of this chute {chute.user_id=} has zero balance.",
            )
        request.state.free_invocation = True

    # Check account quotas if not free/invoiced.
    quota_date = date.today()
    if not (
        current_user.has_role(Permissioning.free_account)
        or current_user.has_role(Permissioning.invoice_billing)
        or request.state.free_invocation
    ):
        quota = await InvocationQuota.get(current_user.user_id, chute.chute_id)
        key = await InvocationQuota.quota_key(current_user.user_id, chute.chute_id)
        client_success, cached = await settings.redis_client.get_with_status(key)
        request_count = 0.0
        if cached is not None:
            try:
                request_count = float(cached.decode())
            except ValueError:
                await settings.redis_client.delete(key)
        elif client_success:
            asyncio.create_task(_initialize_quota_cache(key))

        # No quota for private/user-created chutes.
        effective_balance = (
            current_user.current_balance.effective_balance if current_user.current_balance else 0.0
        )
        if (
            not chute.public
            and not has_legacy_private_billing(chute)
            and chute.user_id != await chutes_user_id()
        ):
            quota = 0

        # Quota-200 users (one-time $5 payment) cannot use TEE models without balance.
        # $3/mo sub users (quota 300 or 301) cannot use premium chutes without balance.
        # In both cases, if the user has balance, force paygo (never free_invocation).
        force_paygo = False
        if quota == 200 and chute.tee:
            if effective_balance <= 0:
                raise HTTPException(
                    status_code=status.HTTP_402_PAYMENT_REQUIRED,
                    detail="TEE models require an active subscription or positive balance.",
                )
            force_paygo = True

        if get_subscription_tier(quota) == 3.0 and chute.chute_id in settings.premium_chute_ids:
            if effective_balance <= 0:
                raise HTTPException(
                    status_code=status.HTTP_402_PAYMENT_REQUIRED,
                    detail="This model requires a higher subscription tier or positive balance.",
                )
            force_paygo = True

        # Automatically switch to paygo when the quota is exceeded.
        if request_count >= quota:
            if effective_balance <= 0 and not request.state.free_invocation:
                logger.warning(
                    f"Payment required: attempted invocation of {chute.name} "
                    f"from user {current_user.username} [{origin_ip}] with no balance "
                    f"and {request_count=} of {quota=}"
                )
                error_kwargs = {
                    "status_code": status.HTTP_402_PAYMENT_REQUIRED,
                    "detail": {
                        "message": (
                            f"Quota exceeded and account balance is ${current_user.current_balance.effective_balance}, "
                            f"please pay with fiat or send tao to {current_user.payment_address}"
                        ),
                    },
                }
                if quota:
                    quota_reset = quota_date + timedelta(days=1)
                    quota_reset = datetime(
                        year=quota_reset.year,
                        month=quota_reset.month,
                        day=quota_reset.day,
                        tzinfo=timezone.utc,
                    ).isoformat()
                    error_kwargs["detail"]["quota_reset_timestamp"] = quota_reset

                raise HTTPException(**error_kwargs)
        else:
            # When within the quota, check subscription caps before marking as free.
            # force_paygo skips free_invocation entirely (TEE/premium restrictions).
            if force_paygo:
                if (fp_monthly_price := get_subscription_tier(quota)) is not None:
                    request.state.subscriber_paygo_discount = SUBSCRIPTION_PAYGO_DISCOUNTS.get(
                        fp_monthly_price, 0.0
                    )
            elif (monthly_price := get_subscription_tier(quota)) is not None:
                now = datetime.now()
                monthly_usage = await get_subscription_usage(
                    current_user.user_id,
                    f"m:{now.strftime('%Y%m')}",
                    "date_trunc('month', now())",
                    35 * 86400,  # 35 days TTL
                )
                monthly_cap = monthly_price * SUBSCRIPTION_MONTHLY_CAP_MULTIPLIER

                four_hour_bucket = int(time.time()) // (4 * 3600)
                four_hour_usage = await get_subscription_usage(
                    current_user.user_id,
                    f"4h:{four_hour_bucket}",
                    "now() - interval '4 hours'",
                    5 * 3600,  # 5 hours TTL
                )
                four_hour_cap = (
                    monthly_price / FOUR_HOUR_CHUNKS_PER_MONTH
                ) * SUBSCRIPTION_4H_CAP_MULTIPLIER

                if monthly_usage >= monthly_cap or four_hour_usage >= four_hour_cap:
                    # Cap exceeded — switch to paygo for remainder
                    exceeded = []
                    if monthly_usage >= monthly_cap:
                        exceeded.append(f"monthly ({monthly_usage:.4f}/{monthly_cap:.4f})")
                    if four_hour_usage >= four_hour_cap:
                        exceeded.append(f"4h ({four_hour_usage:.4f}/{four_hour_cap:.4f})")
                    logger.warning(
                        f"Subscription cap exceeded for {current_user.user_id} "
                        f"[{current_user.username}]: {', '.join(exceeded)}"
                    )
                    if effective_balance <= 0:
                        raise HTTPException(
                            status_code=status.HTTP_402_PAYMENT_REQUIRED,
                            detail="Subscription usage cap exceeded. Please add balance to continue.",
                        )
                    # Has balance — proceed as paygo with subscriber discount
                    request.state.subscriber_paygo_discount = SUBSCRIPTION_PAYGO_DISCOUNTS.get(
                        monthly_price, 0.0
                    )
                else:
                    request.state.free_invocation = True
            else:
                request.state.free_invocation = True

        # Store quota info for response headers.
        request.state.quota_total = quota
        request.state.quota_used = request_count
