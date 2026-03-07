import socket
import asyncio
import inspect
import time
import traceback
import concurrent.futures
from typing import Any, Optional
from loguru import logger
from redis.exceptions import (
    RedisError,
    TimeoutError,
    ConnectionError,
    ResponseError,
    AuthenticationError,
    BusyLoadingError,
)
import redis.asyncio as redis
from collections.abc import AsyncIterable, Iterable


# Exceptions we allow without panic (redis is a cache after all...)
FAIL_OPEN_EXCEPTIONS = (
    # Redis-level errors
    RedisError,
    TimeoutError,
    ConnectionError,
    ResponseError,
    AuthenticationError,
    BusyLoadingError,
    # Socket / network / DNS
    socket.timeout,
    socket.error,
    socket.gaierror,
    OSError,
    # Async / concurrency timeouts
    asyncio.TimeoutError,
    concurrent.futures.TimeoutError,
    # Connection / pool / teardown issues
    BrokenPipeError,
    ConnectionResetError,
    RuntimeError,
)


def is_async_iterable(obj) -> bool:
    return isinstance(obj, AsyncIterable) or hasattr(obj, "__aiter__")


def is_sync_iterable(obj) -> bool:
    if isinstance(obj, (str, bytes, dict, list, tuple, set)):
        return False
    return isinstance(obj, Iterable) or hasattr(obj, "__iter__")


def is_pipeline(obj) -> bool:
    """Detect Redis pipelines without checking by name."""
    # Redis Pipeline classes all have these attributes
    return hasattr(obj, "execute") and hasattr(obj, "command_stack")


def pool_stats(pool) -> str:
    """Return a lightweight snapshot of pool usage for diagnostics."""
    try:
        in_use = len(getattr(pool, "_in_use_connections", []))
        available = len(getattr(pool, "_available_connections", []))
        max_conns = getattr(pool, "_max_connections", None) or getattr(
            pool, "max_connections", None
        )
        return f"in_use={in_use} available={available} max={max_conns}"
    except Exception:
        return "pool_stats_unavailable"


def wrap_pipeline(pipe, default=None, timeout: float = 0.5, owner=None):
    """Make pipeline.execute() fail-open. If owner (SafeRedis) is provided,
    pipeline failures/successes contribute to failover tracking."""
    loop = asyncio.get_running_loop()
    start = loop.time()
    orig_execute = pipe.execute

    async def safe_execute(*args, **kwargs):
        task = asyncio.ensure_future(orig_execute(*args, **kwargs))
        try:
            value = await asyncio.wait_for(asyncio.shield(task), timeout)
            elapsed = loop.time() - start
            if owner is not None:
                owner._record_success()
            if elapsed > 0.25:
                logger.debug(f"SafeRedis: slow pipleine elapsed={elapsed * 1000:.1f}ms")
            return value
        except asyncio.TimeoutError:
            task.add_done_callback(lambda t: t.exception() if not t.cancelled() else None)
            if owner is not None:
                owner._record_failure()
            logger.error("SafeRedis: pipeline.execute fail-open wait_for asyncio.TimeoutError")
        except FAIL_OPEN_EXCEPTIONS as exc:
            if owner is not None:
                owner._record_failure()
            error_detail = str(exc)
            if not error_detail.strip():
                error_detail = traceback.format_exc()
            logger.error(f"SafeRedis: pipeline.execute fail-open: {error_detail}")
        return []

    pipe.execute = safe_execute
    return pipe


class SafeIterator:
    def __init__(self, it, default=None):
        self._it = it
        self._default = default
        self._failed = False

    def __iter__(self):
        return self

    def __next__(self):
        if self._failed:
            raise StopIteration
        try:
            return next(self._it)
        except FAIL_OPEN_EXCEPTIONS as exc:
            error_detail = str(exc)
            if not error_detail.strip():
                error_detail = traceback.format_exc()
            logger.error(f"SafeRedis: iter fail-open: {error_detail}")
            self._failed = True
            raise StopIteration


class SafeAsyncIterator:
    def __init__(self, it, default=None):
        self._it = it
        self._default = default
        self._failed = False

    def __aiter__(self):
        return self

    async def __anext__(self):
        if self._failed:
            raise StopAsyncIteration
        try:
            return await self._it.__anext__()
        except FAIL_OPEN_EXCEPTIONS as exc:
            error_detail = str(exc)
            if not error_detail.strip():
                error_detail = traceback.format_exc()
            logger.error(f"SafeRedis: async-iter fail-open: {error_detail}")
            self._failed = True
            raise StopAsyncIteration


class SafeRedis:
    def __init__(
        self,
        host: str = "172.16.0.100",
        port: int = 1700,
        password: Optional[str] = "secret",
        db: int = 0,
        *,
        default: Any = None,
        socket_connect_timeout: float = 0.2,
        socket_timeout: float = 0.5,
        op_timeout: float = 0.5,
        max_connections: int = 8,
        socket_keepalive: bool = True,
        health_check_interval: int = 30,
        retry_on_timeout: bool = False,
        retry: Any = None,
        consecutive_failure_limit: int = 5,
        pool_reset_cooldown: float = 5.0,
        fallback_host: Optional[str] = None,
        primary_probe_interval: float = 30.0,
        **kwargs,
    ):
        self.default = default
        self.timeout = op_timeout
        self._consecutive_failures = 0
        self._consecutive_failure_limit = consecutive_failure_limit
        self._pool_reset_cooldown = pool_reset_cooldown
        self._last_pool_reset: float = 0.0
        self._primary_host = host
        self._fallback_host = fallback_host
        self._active_host = host
        self._on_primary = True
        self._primary_probe_interval = primary_probe_interval
        self._primary_probe_task: Optional[asyncio.Task] = None
        self._redis_kwargs = dict(
            host=host,
            port=port,
            db=db,
            password=password,
            socket_connect_timeout=socket_connect_timeout,
            socket_timeout=socket_timeout,
            max_connections=max_connections,
            socket_keepalive=socket_keepalive,
            health_check_interval=health_check_interval,
            retry_on_timeout=retry_on_timeout,
            retry=retry,
            **kwargs,
        )
        self.client = redis.Redis(**self._redis_kwargs)

    def _record_success(self):
        if self._consecutive_failures > 0:
            logger.info(
                f"SafeRedis: connection recovered on host={self._active_host} "
                f"after {self._consecutive_failures} consecutive failures"
            )
            self._consecutive_failures = 0

    def _record_failure(self):
        self._consecutive_failures += 1
        if self._consecutive_failures >= self._consecutive_failure_limit:
            self._maybe_reset_pool()

    def _maybe_reset_pool(self):
        now = time.monotonic()
        if now - self._last_pool_reset < self._pool_reset_cooldown:
            return
        self._last_pool_reset = now
        port = self._redis_kwargs.get("port", "?")
        db = self._redis_kwargs.get("db", "?")

        # Decide which host to reconnect to.
        if self._on_primary and self._fallback_host:
            new_host = self._fallback_host
            self._on_primary = False
            logger.warning(
                f"SafeRedis: {self._consecutive_failures} consecutive failures on "
                f"primary={self._primary_host} port={port} db={db}, "
                f"failing over to fallback={new_host}"
            )
        elif not self._on_primary:
            # Already on fallback and still failing — try primary again.
            new_host = self._primary_host
            self._on_primary = True
            logger.warning(
                f"SafeRedis: {self._consecutive_failures} consecutive failures on "
                f"fallback={self._active_host} port={port} db={db}, "
                f"trying primary={new_host} again"
            )
        else:
            # No fallback configured, just reset the pool on the same host.
            new_host = self._active_host
            logger.warning(
                f"SafeRedis: {self._consecutive_failures} consecutive failures on "
                f"host={self._active_host} port={port} db={db}, resetting connection pool"
            )

        self._active_host = new_host
        try:
            old_pool = self.client.connection_pool
            self.client = redis.Redis(**{**self._redis_kwargs, "host": new_host})
            asyncio.ensure_future(self._close_old_pool(old_pool))
        except Exception:
            logger.error(f"SafeRedis: pool reset failed: {traceback.format_exc()}")
        self._consecutive_failures = 0

        # If we just moved off primary, start probing for primary recovery.
        if not self._on_primary and self._fallback_host:
            self._start_primary_probe()

    def _start_primary_probe(self):
        if self._primary_probe_task and not self._primary_probe_task.done():
            return
        self._primary_probe_task = asyncio.ensure_future(self._probe_primary_loop())

    async def _probe_primary_loop(self):
        port = self._redis_kwargs.get("port", "?")
        db = self._redis_kwargs.get("db", "?")
        logger.info(
            f"SafeRedis: starting primary probe for {self._primary_host}:{port} db={db} "
            f"every {self._primary_probe_interval}s"
        )
        while not self._on_primary:
            await asyncio.sleep(self._primary_probe_interval)
            if self._on_primary:
                break
            probe = None
            try:
                probe = redis.Redis(
                    **{
                        **self._redis_kwargs,
                        "host": self._primary_host,
                        "max_connections": 1,
                    }
                )
                pong = await asyncio.wait_for(probe.ping(), timeout=2.0)
                if pong:
                    logger.info(
                        f"SafeRedis: primary {self._primary_host}:{port} db={db} is back, switching over"
                    )
                    old_pool = self.client.connection_pool
                    self._active_host = self._primary_host
                    self._on_primary = True
                    self.client = redis.Redis(**{**self._redis_kwargs, "host": self._primary_host})
                    asyncio.ensure_future(self._close_old_pool(old_pool))
                    self._consecutive_failures = 0
                    break
            except Exception:
                logger.debug(
                    f"SafeRedis: primary probe {self._primary_host}:{port} still unreachable"
                )
            finally:
                if probe is not None:
                    try:
                        await probe.aclose()
                    except Exception:
                        pass
        logger.info(f"SafeRedis: primary probe loop ended, on_primary={self._on_primary}")

    @staticmethod
    async def _close_old_pool(pool):
        try:
            await pool.disconnect()
        except Exception:
            pass

    async def get_with_status(self, key):
        try:
            result = await self.client.get(key)
            self._record_success()
            return True, result
        except FAIL_OPEN_EXCEPTIONS as exc:
            self._record_failure()
            error_detail = str(exc)
            if not error_detail.strip():
                error_detail = traceback.format_exc()
            logger.error(f"SafeRedis: fail-open on get (call): {error_detail}")
            return False, None

    def __getattr__(self, name):
        name_lower = name.lower()

        attr = getattr(self.client, name)

        if not callable(attr):
            return attr

        def wrapper(*args, **kwargs):
            try:
                result = attr(*args, **kwargs)
            except FAIL_OPEN_EXCEPTIONS as exc:
                self._record_failure()
                error_detail = str(exc)
                if not error_detail.strip():
                    error_detail = traceback.format_exc()
                logger.error(f"SafeRedis: fail-open on {name} (call): {error_detail}")
                if name_lower == "scan":
                    return (0, [])
                return self.default

            if is_pipeline(result):
                return wrap_pipeline(result, self.default, timeout=self.timeout * 3, owner=self)

            if inspect.isawaitable(result):

                async def safe_coro():
                    timeout = 30.0 if name_lower == "scan" else self.timeout
                    loop = asyncio.get_running_loop()
                    start = loop.time()
                    task = asyncio.ensure_future(result)
                    try:
                        value = await asyncio.wait_for(asyncio.shield(task), timeout)
                        elapsed = loop.time() - start
                        self._record_success()
                        if elapsed > 0.25:
                            logger.debug(
                                f"SafeRedis: slow call {name} elapsed={elapsed * 1000:.1f}ms "
                                f"pool=({pool_stats(self.client.connection_pool)})"
                            )
                        return value
                    except asyncio.TimeoutError:
                        elapsed = loop.time() - start
                        self._record_failure()
                        task.add_done_callback(
                            lambda t: t.exception() if not t.cancelled() else None
                        )
                        logger.error(
                            f"SafeRedis: timeout on {name} (shielded, task orphaned) "
                            f"elapsed={elapsed * 1000:.1f}ms pool=({pool_stats(self.client.connection_pool)})"
                        )
                        return self.default
                    except FAIL_OPEN_EXCEPTIONS as exc:
                        elapsed = loop.time() - start
                        self._record_failure()
                        error_detail = str(exc)
                        if not error_detail.strip():
                            error_detail = traceback.format_exc()
                        logger.error(
                            f"SafeRedis: fail-open on {name} (await): {error_detail} "
                            f"elapsed={elapsed * 1000:.1f}ms pool=({pool_stats(self.client.connection_pool)})"
                        )
                        return self.default

                return safe_coro()

            if is_async_iterable(result):
                return SafeAsyncIterator(result, default=self.default)
            if is_sync_iterable(result):
                return SafeIterator(result, default=self.default)
            return result

        return wrapper
