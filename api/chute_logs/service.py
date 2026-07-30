"""Core logic for the chute log shipper (validator side).

Responsibilities:
  * authenticate a shipment by the VM's registry mTLS leaf, binding it to the
    launch-config owner (cross-miner injection is rejected here);
  * decide whether the guest should keep capturing or stop (the cutoff);
  * dedupe + enrich + push lines to the dedicated Loki store;
  * read lines back for the owner / miner-CLI paths, always with a server-forced
    matcher so a private chute's logs never leak (see spec §6);
  * get/set/clear the per-chute debug override.
"""

import hashlib
import json
import re
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from typing import Dict, List, Optional

from cryptography.hazmat.primitives.serialization import Encoding
from cryptography.x509 import Certificate
from loguru import logger
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from api.chute_logs import loki
from api.chute_logs.exceptions import LogCaptureNotAuthorized, UnknownLaunchConfig
from api.chute_logs.schemas import LogShipmentArgs, StoredLogLine
from api.chute.schemas import Chute
from api.config import settings
from api.database import get_session
from api.instance.schemas import LaunchConfig
from api.server.exceptions import AttestationError
from api.server.schemas import Server
from api.server.util import verify_leaf_cert_signed_by_ca

# Redis keys.
_DEBUG_KEY = "pod_logs_debug:{chute_id}"
_WATERMARK_KEY = "chute_logs:wm:{config_id}"
# Watermark / debug override lifetime: comfortably past the Loki retention window.
_WATERMARK_TTL = 26 * 3600
_DEFAULT_DEBUG_TTL = 24 * 3600


@dataclass(frozen=True)
class LogCaptureContext:
    """Authenticated, immutable identity for a chute's log capture (safe to cache per boot)."""

    config_id: str
    chute_id: str
    user_id: str
    miner_hotkey: str


# ---------------------------------------------------------------------------
# Shipment authentication — cached per (config_id, cert)
# ---------------------------------------------------------------------------
_AUTH_KEY = "chute_logs:auth:{config_id}:{cert_hash}"


async def authenticate_shipment(config_id: str, client_cert: Certificate) -> LogCaptureContext:
    """Resolve identity for (config_id, cert), cached in Redis. Raises pure domain errors
    (``UnknownLaunchConfig`` / ``LogCaptureNotAuthorized``) — the dependency maps them to HTTP.

    The expensive part — leaf verification + the launch-config/chute/server lookups — then runs
    once per (config, cert) for the whole fleet instead of per batch per pod. An in-process cache
    barely hits across 50-100 load-balanced pods, so the cache is shared in Redis. Only successful
    resolutions are stored; 403/404 raise and are never cached, so a rejected cert always re-resolves.
    """
    cert_hash = hashlib.sha256(client_cert.public_bytes(Encoding.DER)).hexdigest()
    key = _AUTH_KEY.format(config_id=config_id, cert_hash=cert_hash)
    try:
        cached = await settings.redis_client.get(key)
        if cached is not None:
            return LogCaptureContext(**json.loads(cached))
    except Exception as exc:  # pragma: no cover - a redis hiccup must not block auth
        logger.warning(f"chute-logs: auth cache read failed for {config_id}: {exc}")

    async with get_session(readonly=True) as db:
        ctx = await _authenticate(db, config_id, client_cert)

    try:
        await settings.redis_client.set(
            key, json.dumps(asdict(ctx)), ex=settings.chute_logs_auth_cache_seconds
        )
    except Exception as exc:  # pragma: no cover
        logger.warning(f"chute-logs: auth cache write failed for {config_id}: {exc}")
    return ctx


async def _authenticate(
    db: AsyncSession, config_id: str, client_cert: Certificate
) -> LogCaptureContext:
    """Resolve + authenticate a log shipment for ``config_id``.

    The VM mTLS client leaf uses a CN shared across all VMs, so VM identity comes from
    *which CA signed the leaf* (each server records its own per-boot CA at provision time),
    not from the subject. We look up the launch config's owning miner, then accept the
    shipment only if the leaf verifies against one of that miner's registered per-boot VM
    CAs. This makes cross-miner injection impossible: a leaf signed by another miner's VM CA
    cannot verify against this miner's CAs.

    Miner-scoped rather than IP-pinned on purpose: pre-registration the config is not yet
    bound to a server, so the pod may run on any of the miner's VMs — the miner's CA set is
    the right granularity.

    Raises (pure domain errors — the transport boundary maps them, see exceptions.py):
        UnknownLaunchConfig (→ 404) — ``config_id`` unknown (the guest treats this as terminal).
        LogCaptureNotAuthorized (→ 403) — no matching CA, or the leaf fails to verify
            (also terminal for the guest).
    """
    launch_config = await db.scalar(select(LaunchConfig).where(LaunchConfig.config_id == config_id))
    if launch_config is None:
        raise UnknownLaunchConfig()

    chute = await db.scalar(select(Chute).where(Chute.chute_id == launch_config.chute_id))
    if chute is None:
        # Config with no chute is a broken/racing state; nothing to authorize against.
        raise UnknownLaunchConfig("Chute not found.")

    servers = (
        (
            await db.execute(
                select(Server).where(
                    Server.miner_hotkey == launch_config.miner_hotkey,
                    Server.vm_root_ca_cert.isnot(None),
                )
            )
        )
        .scalars()
        .all()
    )

    for server in servers:
        ca = server.vm_root_ca_certificate
        if ca is None:
            continue
        try:
            verify_leaf_cert_signed_by_ca(client_cert, ca)
            return LogCaptureContext(
                config_id=config_id,
                chute_id=chute.chute_id,
                user_id=chute.user_id,
                miner_hotkey=launch_config.miner_hotkey,
            )
        except AttestationError:
            continue

    logger.warning(
        f"chute-logs: mTLS leaf did not verify against any CA for miner "
        f"{launch_config.miner_hotkey} (config {config_id})"
    )
    raise LogCaptureNotAuthorized()


# ---------------------------------------------------------------------------
# Cutoff — binary: keep capturing until the launch is terminal (unless debugging)
# ---------------------------------------------------------------------------
def _is_terminal(launch_config: Optional[LaunchConfig]) -> bool:
    """Has the launch reached a terminal state — activated, failed, or deleted?"""
    if launch_config is None:
        return True  # deleted mid-stream
    instance = launch_config.instance
    activated = instance is not None and instance.activated_at is not None
    return activated or launch_config.failed_at is not None


async def should_stop_capture(ctx: LogCaptureContext) -> bool:
    """Whether to tell the guest to stop shipping (HTTP 204).

    Binary and lifecycle-driven: stop once the launch config is terminal, unless the per-chute
    debug override is on. Computed fresh per batch (cheap: one Redis debug read + one indexed
    launch-config lookup); no cache, no wall-clock cap. A config that never terminalizes keeps
    capturing on purpose — that anomaly's logs are what we want (ingest is bounded per-batch and
    by Loki retention, so nothing runs away).
    """
    if await is_debug_enabled(ctx.chute_id):
        return False
    async with get_session(readonly=True) as db:
        launch_config = await db.scalar(
            select(LaunchConfig).where(LaunchConfig.config_id == ctx.config_id)
        )
        return _is_terminal(launch_config)


# ---------------------------------------------------------------------------
# Ingest
# ---------------------------------------------------------------------------
_TS_RE = re.compile(
    r"^(\d{4}-\d{2}-\d{2}[Tt]\d{2}:\d{2}:\d{2})(?:\.(\d{1,9}))?([Zz]|[+-]\d{2}:?\d{2})$"
)


def rfc3339nano_to_unix_ns(ts: str) -> Optional[int]:
    """Convert an RFC3339 (nanosecond) timestamp to integer unix nanoseconds.

    Not a Loki concern — this is the canonical ns the ingest path derives once and uses for
    *both* dedup (the Redis watermark comparison) and the Loki push value. ``datetime.fromisoformat``
    only handles microseconds, so the fractional part is parsed separately to keep full nanosecond
    precision. Returns None for an unparseable timestamp (the caller skips that line).
    """
    match = _TS_RE.match(ts.strip())
    if not match:
        return None
    base, frac, offset = match.groups()
    offset = "+00:00" if offset in ("Z", "z") else offset
    if len(offset) == 5:  # +HHMM -> +HH:MM
        offset = f"{offset[:3]}:{offset[3:]}"
    try:
        dt = datetime.fromisoformat(f"{base}{offset}")
    except ValueError:
        return None
    seconds = int(dt.timestamp())
    nanos = int((frac or "").ljust(9, "0")[:9])
    return seconds * 1_000_000_000 + nanos


def _truncate(text: str, max_bytes: int) -> str:
    encoded = text.encode("utf-8", errors="replace")
    if len(encoded) <= max_bytes:
        return text
    return encoded[:max_bytes].decode("utf-8", errors="ignore")


async def ingest(
    args: LogShipmentArgs,
    ctx: LogCaptureContext,
    server_ip: str,
) -> int:
    """Dedupe, enrich, and push a shipment's lines to Loki. Returns the count stored.

    Dedupe is a Redis high-watermark on ``(config_id, max ts ns)`` — idempotent across
    guest retries and independent of Loki's own entry dedup. When ``LOKI_URL`` is unset
    the lines are accepted and dropped (the guest is a no-op until Loki is provisioned).
    """
    if not args.logs:
        return 0

    config_id = ctx.config_id

    # Resolve + normalize timestamps, honoring the per-shipment line cap.
    prepared: List[tuple] = []  # (ns, ts, stream, log)
    for line in args.logs[: settings.chute_logs_max_lines_per_shipment]:
        ns = rfc3339nano_to_unix_ns(line.ts)
        if ns is None:
            continue
        stream = line.stream if line.stream in ("stdout", "stderr") else "stdout"
        prepared.append(
            (ns, line.ts, stream, _truncate(line.log, settings.chute_logs_max_line_bytes))
        )
    if not prepared:
        return 0

    # High-watermark dedupe.
    wm_key = _WATERMARK_KEY.format(config_id=config_id)
    watermark = 0
    try:
        raw = await settings.redis_client.get(wm_key)
        if raw is not None:
            watermark = int(raw)
    except Exception as exc:  # pragma: no cover - redis hiccup shouldn't drop logs
        logger.warning(f"chute-logs: watermark read failed for {config_id}: {exc}")

    fresh = [item for item in prepared if item[0] > watermark]
    if not fresh:
        return 0
    max_ns = max(item[0] for item in fresh)

    if settings.loki_url:
        await _push_lines(fresh, ctx, server_ip, args.deployment_id)
    else:
        logger.debug(f"chute-logs: LOKI_URL unset, dropping {len(fresh)} lines for {config_id}")

    try:
        await settings.redis_client.set(wm_key, str(max_ns), ex=_WATERMARK_TTL)
    except Exception as exc:  # pragma: no cover
        logger.warning(f"chute-logs: watermark write failed for {config_id}: {exc}")

    return len(fresh)


async def _push_lines(
    fresh: List[tuple],
    ctx: LogCaptureContext,
    server_ip: str,
    deployment_id: str,
) -> None:
    """Group by stream label and push to Loki. High-cardinality ids ride in the JSON line."""
    base = {
        "config_id": ctx.config_id,
        "chute_id": ctx.chute_id,
        "user_id": ctx.user_id,
        "miner_hotkey": ctx.miner_hotkey,
        "server_ip": server_ip,
        "deployment_id": deployment_id,
    }
    by_stream: Dict[str, List[List[str]]] = {}
    for ns, ts, stream, log in fresh:
        record = dict(base)
        record.update({"ts": ts, "stream": stream, "log": log})
        by_stream.setdefault(stream, []).append([str(ns), json.dumps(record)])

    streams = [
        {
            "stream": {"app": loki.APP_LABEL, "stream": stream},
            "values": values,
        }
        for stream, values in by_stream.items()
    ]
    await loki.LokiClient().push(streams)


# ---------------------------------------------------------------------------
# Reads (always server-forced — see spec §6)
# ---------------------------------------------------------------------------
def _escape(value: str) -> str:
    return value.replace("\\", "\\\\").replace('"', '\\"')


def _build_query(config_id: str, forced: Dict[str, str]) -> str:
    query = f'{{app="{loki.APP_LABEL}"}} | json | config_id="{_escape(config_id)}"'
    for key, value in forced.items():
        query += f' | {key}="{_escape(value)}"'
    return query


async def read_config_logs(
    config_id: str, forced: Dict[str, str], limit: int = 5000
) -> List[StoredLogLine]:
    """Read a config's stored logs, forcing ``forced`` label matchers into the query.

    ``forced`` is built by the caller from the authenticated principal (e.g.
    ``{"user_id": <owner>}``) so the query can never widen past what the caller may see.
    Returns [] when Loki is unconfigured.
    """
    if not settings.loki_url:
        return []
    now_ns = int(datetime.now(timezone.utc).timestamp() * 1_000_000_000)
    start_ns = now_ns - (_WATERMARK_TTL * 1_000_000_000)
    end_ns = now_ns + 60 * 1_000_000_000
    query = _build_query(config_id, forced)
    rows = await loki.LokiClient().query_range(query, start_ns, end_ns, limit=limit)
    out: List[StoredLogLine] = []
    for ts_ns, record in rows:
        out.append(
            StoredLogLine(
                ts=record.get("ts") or ts_ns,
                stream=record.get("stream") or "stdout",
                log=record.get("log") or "",
            )
        )
    return out


# ---------------------------------------------------------------------------
# Per-chute debug override
# ---------------------------------------------------------------------------
async def is_debug_enabled(chute_id: str) -> bool:
    try:
        return bool(await settings.redis_client.exists(_DEBUG_KEY.format(chute_id=chute_id)))
    except Exception as exc:  # pragma: no cover
        logger.warning(f"chute-logs: debug-flag read failed for {chute_id}: {exc}")
        return False


async def set_debug(chute_id: str, ttl_seconds: int = _DEFAULT_DEBUG_TTL) -> None:
    await settings.redis_client.set(_DEBUG_KEY.format(chute_id=chute_id), "1", ex=ttl_seconds)


async def clear_debug(chute_id: str) -> None:
    await settings.redis_client.delete(_DEBUG_KEY.format(chute_id=chute_id))
