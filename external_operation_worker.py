"""Dedicated process entrypoint for external task polling and settlement maintenance."""

from __future__ import annotations

import asyncio
import os
import signal

import api.logging_bootstrap  # noqa: F401  # configure service logging first
import api.database.orms  # noqa: F401
from aiohttp import web
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from sqlalchemy import text

from api.config import settings
from api.database import get_session
from api.external_backend.polling import (
    ExternalOperationPoller,
    start_external_operation_poller,
    stop_external_operation_poller,
)


def _health_port() -> int:
    try:
        port = int(os.getenv("EXTERNAL_WORKER_HEALTH_PORT", "8000"))
    except ValueError as exc:
        raise RuntimeError("EXTERNAL_WORKER_HEALTH_PORT must be an integer") from exc
    if not 1 <= port <= 65535:
        raise RuntimeError("EXTERNAL_WORKER_HEALTH_PORT must be between 1 and 65535")
    return port


async def _start_health_server(
    poller: ExternalOperationPoller,
) -> web.AppRunner:
    async def health(_: web.Request) -> web.Response:
        database_ok = False
        redis_ok = False
        if poller.running:
            try:
                async with asyncio.timeout(2.0):
                    async with get_session(readonly=True) as db:
                        database_ok = (
                            await db.execute(text("SELECT 1"))
                        ).scalar_one() == 1
                    redis_ok = bool(await settings.redis_client.client.ping())
            except Exception:
                database_ok = False
                redis_ok = False
        healthy = poller.running and database_ok and redis_ok
        return web.json_response(
            {
                "status": "ok" if healthy else "unavailable",
                "poller_running": poller.running,
                "database": database_ok,
                "redis": redis_ok,
            },
            status=200 if healthy else 503,
        )

    async def metrics(_: web.Request) -> web.Response:
        return web.Response(
            body=generate_latest(),
            headers={"Content-Type": CONTENT_TYPE_LATEST},
        )

    app = web.Application(client_max_size=1024)
    app.router.add_get("/health", health)
    app.router.add_get("/metrics", metrics)
    app.router.add_get("/_metrics", metrics)
    runner = web.AppRunner(app, access_log=None)
    await runner.setup()
    site = web.TCPSite(
        runner,
        host=os.getenv("EXTERNAL_WORKER_HEALTH_HOST", "0.0.0.0"),
        port=_health_port(),
    )
    await site.start()
    return runner


async def main() -> None:
    stop = asyncio.Event()
    loop = asyncio.get_running_loop()
    for signal_name in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(signal_name, stop.set)
    poller = start_external_operation_poller()
    if not poller.running:
        raise RuntimeError("EXTERNAL_POLLER_ENABLED must be true for the worker")
    health_runner: web.AppRunner | None = None
    try:
        health_runner = await _start_health_server(poller)
        await stop.wait()
    finally:
        try:
            if health_runner is not None:
                await health_runner.cleanup()
        finally:
            await stop_external_operation_poller()


if __name__ == "__main__":
    asyncio.run(main())
