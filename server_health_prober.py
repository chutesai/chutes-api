"""
Stale-inventory sweep for TEE servers.

Probes each TEE server's attestation-proxy health endpoint
(GET https://<ip>:30443/health -> 200 {"status": "healthy"}), then materializes
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

PROBE_TIMEOUT = _httpx.Timeout(connect=5.0, read=10.0, write=10.0, pool=5.0)


async def probe(server: Server) -> bool:
    """
    Hit a server's attestation-proxy health endpoint. Returns True only on
    200 + {"status": "healthy"}. The cert is self-signed (CN=attestation-service),
    so TLS verification is disabled.
    """
    try:
        async with _httpx.AsyncClient(timeout=PROBE_TIMEOUT, verify=False) as client:
            resp = await client.get(server.health_check_url)
            return resp.status_code == 200 and resp.json().get("status") == "healthy"
    except Exception as exc:
        # Some httpx errors (e.g. ConnectTimeout) stringify to "", so include the type name.
        logger.debug(
            f"Health probe failed for {server.server_id} ({server.health_check_url}): "
            f"{type(exc).__name__}: {exc}"
        )
        return False


async def sweep(max_concurrent: int = None):
    """
    Probe all TEE servers concurrently and bulk-update their health columns.
    """
    max_concurrent = max_concurrent or settings.server_health_max_concurrent
    semaphore = asyncio.Semaphore(max_concurrent)
    now = datetime.now(timezone.utc)

    async with get_session(readonly=True) as session:
        result = await session.execute(select(Server).where(Server.is_tee.is_(True)))
        servers = result.scalars().all()

    logger.info(
        f"Probing {len(servers)} TEE servers "
        f"(degraded>{settings.server_health_degraded_threshold_seconds}s, "
        f"offline>{settings.server_health_offline_threshold_seconds}s)"
    )

    async def probe_one(server: Server) -> bool:
        async with semaphore:
            reachable = await probe(server)
            if reachable:
                server.last_health_at = now  # in-memory, so the count below reflects it
            return reachable

    reachable_flags = await asyncio.gather(*[probe_one(s) for s in servers])

    # The only persisted state is last_health_at; health_status is derived live from it.
    reachable_ids = [s.server_id for s, ok in zip(servers, reachable_flags) if ok]
    if reachable_ids:
        async with get_session() as session:
            await session.execute(
                text("UPDATE servers SET last_health_at = :now WHERE server_id = ANY(:ids)"),
                {"now": now, "ids": reachable_ids},
            )

    counts = {s.value: 0 for s in ServerHealthStatus}
    for server in servers:
        counts[server.health_status.value] += 1
    logger.success("=" * 80)
    logger.success(
        f"Health sweep complete: "
        f"\t{counts[ServerHealthStatus.HEALTHY.value]} healthy, "
        f"\t{counts[ServerHealthStatus.DEGRADED.value]} degraded, "
        f"\t{counts[ServerHealthStatus.OFFLINE.value]} offline, "
        f"\t{counts[ServerHealthStatus.UNKNOWN.value]} unknown"
    )


if __name__ == "__main__":
    gc.set_threshold(5000, 50, 50)
    try:
        asyncio.run(sweep())
    except Exception:
        logger.error(f"Health sweep failed:\n{traceback.format_exc()}")
        raise
