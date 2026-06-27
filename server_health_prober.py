"""
Stale-inventory sweep for TEE servers.

Probes each TEE server's unauthenticated liveness endpoint
(GET http://<ip>:8080/status/health -> 200 {"status": "ok"}), then materializes
both `last_health_at` (timestamp of the last successful probe) and `health_status`
(healthy / stale / unknown) on the servers row in a single bulk update per sweep.

A failed probe never touches `last_health_at` and never deletes anything — a server
flips to `stale` purely because its last success has aged past the threshold.
"""

import gc
import asyncio
import traceback
from datetime import datetime, timezone
import httpx as _httpx
import api.database.orms  # noqa
from loguru import logger
from sqlalchemy import select, text
from api.config import settings
from api.database import get_session
from api.constants import ServerHealthStatus
from api.server.schemas import Server

HEALTH_PORT = 8080
PROBE_TIMEOUT = _httpx.Timeout(connect=5.0, read=10.0, write=10.0, pool=5.0)


async def probe(ip: str) -> bool:
    """
    Hit a single server's /status/health endpoint. Returns True only on 200 + {"status": "ok"}.
    """
    try:
        async with _httpx.AsyncClient(timeout=PROBE_TIMEOUT) as client:
            resp = await client.get(f"http://{ip}:{HEALTH_PORT}/status/health")
            return resp.status_code == 200 and resp.json().get("status") == "ok"
    except Exception as exc:
        logger.debug(f"Health probe failed for {ip}: {exc}")
        return False


def _status_for(last_health_at, now, threshold_seconds: int) -> str:
    """
    Derive health status from the (post-probe) last_health_at timestamp.
    """
    if last_health_at is None:
        return ServerHealthStatus.UNKNOWN.value
    age = (now - last_health_at).total_seconds()
    if age <= threshold_seconds:
        return ServerHealthStatus.HEALTHY.value
    return ServerHealthStatus.STALE.value


async def sweep(max_concurrent: int = None):
    """
    Probe all TEE servers concurrently and bulk-update their health columns.
    """
    max_concurrent = max_concurrent or settings.server_health_max_concurrent
    threshold = settings.server_health_stale_threshold_seconds
    semaphore = asyncio.Semaphore(max_concurrent)
    now = datetime.now(timezone.utc)

    async with get_session(readonly=True) as session:
        result = await session.execute(
            select(Server.server_id, Server.ip, Server.last_health_at).where(Server.is_tee.is_(True))
        )
        servers = result.all()

    logger.info(f"Probing {len(servers)} TEE servers (threshold={threshold}s)")

    async def probe_one(server_id: str, ip: str, last_health_at):
        async with semaphore:
            reachable = await probe(ip)
            new_last = now if reachable else last_health_at
            status = _status_for(new_last, now, threshold)
            return {"server_id": server_id, "last_health_at": new_last, "health_status": status}

    updates = await asyncio.gather(
        *[probe_one(s.server_id, s.ip, s.last_health_at) for s in servers]
    )

    if not updates:
        logger.info("No TEE servers to update.")
        return

    # Single bulk update: unnest parallel arrays and join on server_id.
    ids = [u["server_id"] for u in updates]
    timestamps = [u["last_health_at"] for u in updates]
    statuses = [u["health_status"] for u in updates]
    async with get_session() as session:
        await session.execute(
            text(
                """
                UPDATE servers AS s
                SET last_health_at = v.last_health_at,
                    health_status = v.health_status
                FROM (
                    SELECT * FROM unnest(
                        CAST(:ids AS text[]),
                        CAST(:timestamps AS timestamptz[]),
                        CAST(:statuses AS text[])
                    ) AS t(server_id, last_health_at, health_status)
                ) AS v
                WHERE s.server_id = v.server_id
                """
            ),
            {"ids": ids, "timestamps": timestamps, "statuses": statuses},
        )

    counts = {s.value: 0 for s in ServerHealthStatus}
    for u in updates:
        counts[u["health_status"]] += 1
    logger.success("=" * 80)
    logger.success(
        f"Health sweep complete: "
        f"\t{counts[ServerHealthStatus.HEALTHY.value]} healthy, "
        f"\t{counts[ServerHealthStatus.STALE.value]} stale, "
        f"\t{counts[ServerHealthStatus.UNKNOWN.value]} unknown"
    )


if __name__ == "__main__":
    gc.set_threshold(5000, 50, 50)
    try:
        asyncio.run(sweep())
    except Exception:
        logger.error(f"Health sweep failed:\n{traceback.format_exc()}")
        raise
