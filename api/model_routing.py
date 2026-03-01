"""
Multi-model routing: failover, latency-based, and throughput-based selection.
"""

import time
import orjson as json
from sqlalchemy import select, func
from api.config import settings
from api.chute.schemas import Chute
from api.chute.util import get_one
from api.database import get_session
from api.instance.util import load_chute_target_ids, cm_redis_shard
from api.metrics.perf import otps_tracker, ptps_tracker, ttft_tracker
from api.model_alias.schemas import ModelAlias


ROUTING_SUFFIXES = (":latency", ":throughput")


def parse_model_parameter(model_str: str) -> tuple[str, str | None]:
    """
    Strip :latency or :throughput suffix from model string.
    Returns (model_str_without_suffix, routing_mode).
    routing_mode is None (failover), "latency", or "throughput".
    """
    model_str = model_str.strip()
    lower = model_str.lower()
    for suffix in ROUTING_SUFFIXES:
        if lower.endswith(suffix):
            return model_str[: -len(suffix)], suffix[1:]  # strip the colon
    return model_str, None


def _dedupe_keep_order(items: list[str]) -> list[str]:
    """Remove duplicates while preserving original order."""
    seen = set()
    out = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        out.append(item)
    return out


async def get_user_alias(user_id: str, alias: str) -> list[str] | None:
    """
    Look up a user's model alias. Redis-cached with 120s TTL.
    Returns ordered list of chute_ids, or None if alias doesn't exist.
    """
    cache_key = f"malias:v2:{user_id}:{alias.lower()}"
    cached = await settings.redis_client.get(cache_key)
    if cached is not None:
        if cached == b"__none__":
            return None
        return json.loads(cached)

    async with get_session() as session:
        result = await session.execute(
            select(ModelAlias.chute_ids).where(
                ModelAlias.user_id == user_id,
                func.lower(ModelAlias.alias) == alias.lower(),
            )
        )
        row = result.scalar_one_or_none()

    if row is not None:
        await settings.redis_client.set(cache_key, json.dumps(row), ex=120)
        return row
    else:
        await settings.redis_client.set(cache_key, b"__none__", ex=120)
        return None


async def check_chute_availability(chute_id: str) -> bool:
    """
    Lightweight check: does this chute have at least one instance with capacity?
    Uses Redis connection tracking keys; falls back to load_chute_target_ids for cold chutes.
    """
    instance_ids = await settings.redis_client.smembers(f"cc_inst:{chute_id}")
    if not instance_ids:
        nonce = int(time.time())
        nonce -= nonce % 30
        db_ids = await load_chute_target_ids(chute_id, nonce=nonce)
        return len(db_ids) > 0

    conc_raw = await settings.redis_client.get(f"cc_conc:{chute_id}")
    concurrency = int(conc_raw) if conc_raw else 1

    keys = [
        f"cc:{chute_id}:{iid.decode() if isinstance(iid, bytes) else iid}" for iid in instance_ids
    ]
    values = await cm_redis_shard(chute_id).mget(keys)
    for v in values:
        if int(v or 0) < concurrency:
            return True

    return False


async def get_chute_perf(chute_id: str) -> dict[str, float | None]:
    """
    Get current otps, ptps, and ttft EMA values for a chute.
    """
    otps_info = await otps_tracker().get_info(chute_id)
    ptps_info = await ptps_tracker().get_info(chute_id)
    ttft_info = await ttft_tracker().get_info(chute_id)
    return {
        "otps": otps_info["ema"] if otps_info and otps_info.get("ready") else None,
        "ptps": ptps_info["ema"] if ptps_info and ptps_info.get("ready") else None,
        "ttft": ttft_info["ema"] if ttft_info and ttft_info.get("ema") is not None else None,
    }


async def _load_chutes_map(chute_ids: list[str]) -> dict[str, Chute]:
    """Load chute objects for a list of IDs/names, returning a map of id->Chute."""
    result = {}
    for cid in chute_ids:
        chute = await get_one(cid)
        if chute is not None:
            result[cid] = chute
    return result


async def _rank_failover(chute_ids: list[str], chutes_map: dict[str, Chute]) -> list[Chute]:
    """
    Failover ranking: available chutes first (in order), then at-capacity chutes
    that have instances (in order). Chutes with no instances at all are excluded.
    """
    available = []
    at_capacity = []

    for cid in chute_ids:
        chute = chutes_map.get(cid)
        if chute is None:
            continue
        if await check_chute_availability(chute.chute_id):
            available.append(chute)
        else:
            nonce = int(time.time())
            nonce -= nonce % 30
            ids = await load_chute_target_ids(chute.chute_id, nonce=nonce)
            if ids:
                at_capacity.append(chute)

    return available + at_capacity


async def _rank_by_metric(
    chute_ids: list[str],
    chutes_map: dict[str, Chute],
    metric: str,
    ascending: bool = False,
) -> list[Chute]:
    """
    Rank chutes by metric value among available chutes.
    ascending=True for lower-is-better metrics (ttft), False for higher-is-better (otps).
    Chutes without metrics are appended in original order after ranked ones.
    """
    scored: list[tuple[float, Chute]] = []
    unscored: list[Chute] = []

    for cid in chute_ids:
        chute = chutes_map.get(cid)
        if chute is None:
            continue
        if not await check_chute_availability(chute.chute_id):
            continue
        perf = await get_chute_perf(chute.chute_id)
        score = perf.get(metric)
        if score is None:
            unscored.append(chute)
        else:
            scored.append((score, chute))

    scored.sort(key=lambda x: x[0], reverse=not ascending)
    ranked = [chute for _, chute in scored] + unscored

    # If nothing was available, fall back to failover ordering.
    if not ranked:
        return await _rank_failover(chute_ids, chutes_map)
    return ranked


def _check_chute_access(chute: Chute, template: str, user_id: str) -> bool:
    """Check that chute matches template. Access checks happen downstream."""
    return chute.standard_template == template


async def resolve_model_parameter(
    model_str: str, user_id: str, template: str
) -> tuple[list[Chute], str | None]:
    """
    Main entry point for multi-model resolution.
    Returns (ranked_chutes, routing_mode).
    ranked_chutes is an ordered list — caller should try each in sequence,
    falling back to the next on infra_overload.

    Resolution order:
    1. Try exact get_one(model_str) first — handles names with colons/commas
    2. Strip :latency/:throughput suffix
    3. If contains comma -> comma-separated list of chute names
    4. Else try get_one(stripped) for single-chute lookup
    5. Else look up as user alias -> expand to ordered chute_ids list
    """
    # 1. Always try exact match first — colons and commas can appear in real model names.
    exact = await get_one(model_str)
    if exact is not None and _check_chute_access(exact, template, user_id):
        return [exact], None

    raw_model, routing_mode = parse_model_parameter(model_str)

    chute_ids: list[str] | None = None

    if "," in raw_model:
        tokens = [s.strip() for s in raw_model.split(",") if s.strip()]
        expanded: list[str] = []
        for token in tokens:
            # Prefer direct model lookup over alias when names collide.
            if await get_one(token) is not None:
                expanded.append(token)
                continue
            alias_ids = await get_user_alias(user_id, token)
            if alias_ids is not None:
                expanded.extend(alias_ids)
            else:
                expanded.append(token)
        chute_ids = _dedupe_keep_order(expanded)
    else:
        # Try single lookup on suffix-stripped name.
        if routing_mode is not None:
            chute = await get_one(raw_model)
            if chute is not None and _check_chute_access(chute, template, user_id):
                return [chute], routing_mode

        # Try as alias.
        alias_ids = await get_user_alias(user_id, raw_model)
        if alias_ids is not None:
            chute_ids = alias_ids
        else:
            return [], routing_mode

    if not chute_ids:
        return [], routing_mode

    chutes_map = await _load_chutes_map(chute_ids)

    valid_ids = [
        cid
        for cid in chute_ids
        if cid in chutes_map and _check_chute_access(chutes_map[cid], template, user_id)
    ]
    if not valid_ids:
        return [], routing_mode

    valid_map = {cid: chutes_map[cid] for cid in valid_ids}

    if routing_mode == "throughput":
        ranked = await _rank_by_metric(valid_ids, valid_map, "otps")
    elif routing_mode == "latency":
        ranked = await _rank_by_metric(valid_ids, valid_map, "ttft", ascending=True)
    else:
        ranked = await _rank_failover(valid_ids, valid_map)

    return ranked, routing_mode
