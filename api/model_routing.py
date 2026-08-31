"""
Multi-model routing: failover, latency-based, and throughput-based selection.
"""

import time
import orjson as json
from sqlalchemy import func, or_, select
from api.config import settings
from api.chute.schemas import Chute
from api.chute.standard_templates import standard_template_matches
from api.chute.util import get_one
from api.database import get_session
from api.instance.util import load_chute_target_ids
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

    async with get_session(readonly=True) as session:
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
    chute = await get_one(chute_id)
    if chute and chute.execution_backend == "external":
        if chute.disabled:
            return False
        from api.external_backend.schemas import (
            ExternalBackendAccount,
            ExternalChuteBinding,
        )

        async with get_session(readonly=True) as session:
            result = await session.execute(
                select(ExternalChuteBinding.binding_id)
                .join(
                    ExternalBackendAccount,
                    ExternalBackendAccount.account_id
                    == ExternalChuteBinding.account_id,
                )
                .where(
                    ExternalChuteBinding.chute_id == chute.chute_id,
                    ExternalChuteBinding.enabled.is_(True),
                    ExternalBackendAccount.enabled.is_(True),
                )
                .limit(1)
            )
            return result.scalar_one_or_none() is not None

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
    values = await settings.cm_redis_client.mget(keys)
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


async def _get_routing_chute(
    name_or_id: str,
    *,
    user_id: str,
    execution_backend: str | None = None,
) -> Chute | None:
    """Load an exact routing candidate, optionally constrained by backend."""

    if execution_backend is None:
        return await get_one(name_or_id)
    async with get_session(readonly=True) as session:
        return (
            (
                await session.execute(
                    select(Chute)
                    .where(
                        or_(
                            Chute.name == name_or_id,
                            Chute.chute_id == name_or_id,
                        ),
                        Chute.execution_backend == execution_backend,
                        Chute.disabled.is_not(True),
                    )
                    .order_by(
                        (Chute.user_id == user_id).desc(),
                        Chute.public.desc(),
                        Chute.created_at.desc(),
                    )
                    .limit(1)
                )
            )
            .unique()
            .scalar_one_or_none()
        )


async def _load_chutes_map(
    chute_ids: list[str],
    *,
    user_id: str,
    execution_backend: str | None = None,
) -> dict[str, Chute]:
    """Load chute objects for a list of IDs/names, returning a map of id->Chute."""
    result = {}
    for cid in chute_ids:
        chute = await _get_routing_chute(
            cid,
            user_id=user_id,
            execution_backend=execution_backend,
        )
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
        elif chute.execution_backend == "external":
            # External availability is entirely binding/account based. Never
            # fall through to hosted target discovery for this backend.
            continue
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
        if chute.execution_backend == "external":
            # Provider-neutral external routes do not expose hosted OTPS/TTFT
            # telemetry. Keep them available as unscored fallbacks.
            unscored.append(chute)
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
    return standard_template_matches(
        chute.standard_template,
        template,
        execution_backend=getattr(chute, "execution_backend", "hosted"),
    )


async def resolve_model_parameter(
    model_str: str,
    user_id: str,
    template: str,
    *,
    execution_backend: str | None = None,
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
    exact = await _get_routing_chute(
        model_str,
        user_id=user_id,
        execution_backend=execution_backend,
    )
    if exact is not None and _check_chute_access(exact, template, user_id):
        return [exact], None

    raw_model, routing_mode = parse_model_parameter(model_str)

    chute_ids: list[str] | None = None

    if "," in raw_model:
        tokens = [s.strip() for s in raw_model.split(",") if s.strip()]
        expanded: list[str] = []
        for token in tokens:
            # Prefer direct model lookup over alias when names collide.
            if (
                await _get_routing_chute(
                    token,
                    user_id=user_id,
                    execution_backend=execution_backend,
                )
                is not None
            ):
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
            chute = await _get_routing_chute(
                raw_model,
                user_id=user_id,
                execution_backend=execution_backend,
            )
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

    chutes_map = await _load_chutes_map(
        chute_ids,
        user_id=user_id,
        execution_backend=execution_backend,
    )

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


async def resolve_exact_external_models(
    model_str: str, user_id: str, template: str
) -> list[Chute]:
    """Resolve exact external candidates before legacy hosted name rewrites.

    Access is intentionally checked by the common invocation filter. Keeping all
    exact candidates here preserves sharing and subnet-role access semantics and
    prevents an inaccessible duplicate from hiding an accessible one.
    """

    from api.external_backend.schemas import (
        ExternalBackendAccount,
        ExternalChuteBinding,
    )

    async with get_session(readonly=True) as session:
        candidates = (
            await session.execute(
                select(Chute)
                .join(
                    ExternalChuteBinding,
                    ExternalChuteBinding.chute_id == Chute.chute_id,
                )
                .join(
                    ExternalBackendAccount,
                    ExternalBackendAccount.account_id
                    == ExternalChuteBinding.account_id,
                )
                .where(
                    or_(Chute.name == model_str, Chute.chute_id == model_str),
                    Chute.execution_backend == "external",
                    Chute.disabled.is_(False),
                    ExternalChuteBinding.enabled.is_(True),
                    ExternalBackendAccount.enabled.is_(True),
                )
                .order_by(
                    (Chute.user_id == user_id).desc(),
                    Chute.public.desc(),
                    Chute.created_at.desc(),
                )
            )
        ).unique().scalars().all()
    return [
        candidate
        for candidate in candidates
        if not getattr(candidate, "disabled", False)
        and _check_chute_access(candidate, template, user_id)
    ]
