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

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, List, Optional

from async_lru import alru_cache
from cryptography import x509
from cryptography.hazmat.primitives.serialization import Encoding
from cryptography.x509 import Certificate
from fastapi import Depends, HTTPException, status
from loguru import logger
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from api.chute_logs import loki
from api.chute_logs.schemas import LogShipmentArgs, StoredLogLine
from api.chute.schemas import Chute
from api.config import settings
from api.database import get_session
from api.instance.schemas import LaunchConfig
from api.server.exceptions import AttestationError
from api.server.schemas import Server
from api.server.util import extract_client_cert, verify_leaf_cert_signed_by_ca

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
    created_at_iso: str  # launch-config creation time, for the capture-expiry backstop


@dataclass(frozen=True)
class LogCaptureState:
    """Mutable, short-TTL-cached cutoff state for a config."""

    outcome: str  # running | failed | activated
    stop: bool


# ---------------------------------------------------------------------------
# Authentication — FastAPI dependency, cached per (config_id, cert)
# ---------------------------------------------------------------------------
def resolve_log_context():
    """FastAPI dependency: authenticate an incoming log shipment and resolve its LogCaptureContext.

    Composes ``extract_client_cert`` and resolves identity via CA match (below). The result
    is memoized per (config_id, cert) so a burst of batches doesn't re-verify the leaf or
    re-hit the DB on every POST.
    """

    async def _dep(
        config_id: str,
        client_cert: Certificate = Depends(extract_client_cert()),
    ) -> LogCaptureContext:
        cert_pem = client_cert.public_bytes(Encoding.PEM).decode()
        return await _authenticate_cached(config_id, cert_pem)

    return _dep


@alru_cache(maxsize=4096, ttl=settings.chute_logs_auth_cache_seconds)
async def _authenticate_cached(config_id: str, cert_pem: str) -> LogCaptureContext:
    """Cached auth core. alru_cache does not cache exceptions, so 403/404 always re-resolve."""
    leaf = x509.load_pem_x509_certificate(cert_pem.encode())
    async with get_session(readonly=True) as db:
        return await _authenticate(db, config_id, leaf)


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

    Raises:
        HTTPException 404 — ``config_id`` unknown (the guest treats this as terminal).
        HTTPException 403 — no leaf, no matching CA, or the leaf fails to verify
            (also terminal for the guest).
    """
    launch_config = await db.scalar(select(LaunchConfig).where(LaunchConfig.config_id == config_id))
    if launch_config is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Unknown launch config.")

    chute = await db.scalar(select(Chute).where(Chute.chute_id == launch_config.chute_id))
    if chute is None:
        # Config with no chute is a broken/racing state; nothing to authorize against.
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Chute not found.")

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
                created_at_iso=launch_config.created_at.isoformat()
                if launch_config.created_at
                else "",
            )
        except AttestationError:
            continue

    logger.warning(
        f"chute-logs: mTLS leaf did not verify against any CA for miner "
        f"{launch_config.miner_hotkey} (config {config_id})"
    )
    raise HTTPException(
        status_code=status.HTTP_403_FORBIDDEN,
        detail="Log shipment mTLS leaf does not match a registered VM CA for this launch config.",
    )


# ---------------------------------------------------------------------------
# Cutoff / outcome — short-TTL cached mutable state
# ---------------------------------------------------------------------------
def _capture_expired(created_at_iso: str) -> bool:
    if not created_at_iso:
        return False
    created = datetime.fromisoformat(created_at_iso)
    if created.tzinfo is None:
        created = created.replace(tzinfo=timezone.utc)
    age = (datetime.now(timezone.utc) - created).total_seconds()
    return age >= settings.chute_logs_max_capture_seconds


def _outcome(launch_config: LaunchConfig) -> str:
    """Derive the ingest-time outcome from launch-config / instance state."""
    instance = launch_config.instance
    if instance is not None and instance.activated_at is not None:
        return "activated"
    if launch_config.failed_at is not None:
        return "failed"
    return "running"


def _compute_stop(launch_config: LaunchConfig, created_at_iso: str, debug: bool) -> bool:
    """Cutoff decision: True → tell the guest to stop (HTTP 204).

    Default: stop once the instance is activated. The per-chute debug override keeps capture
    going past activation, but the max-capture backstop always wins so a never-activating,
    never-terminating pod cannot stream forever.
    """
    if _capture_expired(created_at_iso):
        return True
    if debug:
        return False
    instance = launch_config.instance
    if instance is not None and instance.activated_at is not None:
        return True
    if launch_config.failed_at is not None:
        # Pod is terminal; guest will stop on its own — signal stop after this batch.
        return True
    return False


async def _compute_state(
    db: AsyncSession, config_id: str, chute_id: str, created_at_iso: str
) -> LogCaptureState:
    launch_config = await db.scalar(select(LaunchConfig).where(LaunchConfig.config_id == config_id))
    if launch_config is None:
        # Disappeared mid-stream — treat as terminal.
        return LogCaptureState(outcome="failed", stop=True)
    debug = await is_debug_enabled(chute_id)
    return LogCaptureState(
        outcome=_outcome(launch_config),
        stop=_compute_stop(launch_config, created_at_iso, debug),
    )


@alru_cache(maxsize=8192, ttl=settings.chute_logs_state_cache_seconds)
async def _capture_state_cached(
    config_id: str, chute_id: str, created_at_iso: str
) -> LogCaptureState:
    async with get_session(readonly=True) as db:
        return await _compute_state(db, config_id, chute_id, created_at_iso)


async def get_capture_state(ctx: LogCaptureContext) -> LogCaptureState:
    """Current outcome + cutoff for a chute's log capture, cached briefly to collapse per-poll bursts."""
    return await _capture_state_cached(ctx.config_id, ctx.chute_id, ctx.created_at_iso)


# ---------------------------------------------------------------------------
# Ingest
# ---------------------------------------------------------------------------
def _truncate(text: str, max_bytes: int) -> str:
    encoded = text.encode("utf-8", errors="replace")
    if len(encoded) <= max_bytes:
        return text
    return encoded[:max_bytes].decode("utf-8", errors="ignore")


async def ingest(
    args: LogShipmentArgs,
    ctx: LogCaptureContext,
    outcome: str,
    server_ip: Optional[str],
) -> int:
    """Dedupe, enrich, and push a shipment's lines to Loki. Returns the count stored.

    Dedupe is a Redis high-watermark on ``(config_id, max ts ns)`` — idempotent across
    guest retries and independent of Loki's own entry dedup. When ``LOKI_URL`` is unset
    the lines are accepted and dropped (the guest is a no-op until Loki is provisioned).
    """
    if not args.logs:
        return 0

    config_id = ctx.config_id
    server_ip = server_ip or ""

    # Resolve + normalize timestamps, honoring the per-shipment line cap.
    prepared: List[tuple] = []  # (ns, ts, stream, log)
    for line in args.logs[: settings.chute_logs_max_lines_per_shipment]:
        ns = loki.rfc3339nano_to_unix_ns(line.ts)
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
        await _push_lines(fresh, ctx, outcome, server_ip, args.deployment_id)
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
    outcome: str,
    server_ip: str,
    deployment_id: Optional[str],
) -> None:
    """Group by stream label and push to Loki. High-cardinality ids ride in the JSON line."""
    import json

    base = {
        "config_id": ctx.config_id,
        "chute_id": ctx.chute_id,
        "user_id": ctx.user_id,
        "miner_hotkey": ctx.miner_hotkey,
        "server_ip": server_ip,
        "deployment_id": deployment_id or "",
        "outcome": outcome,
    }
    by_stream: Dict[str, List[List[str]]] = {}
    for ns, ts, stream, log in fresh:
        record = dict(base)
        record.update({"ts": ts, "stream": stream, "log": log})
        by_stream.setdefault(stream, []).append([str(ns), json.dumps(record)])

    streams = [
        {
            "stream": {"app": loki.APP_LABEL, "outcome": outcome, "stream": stream},
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
