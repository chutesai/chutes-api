import socket
import asyncio
import inspect
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


def wrap_pipeline(pipe, default=None):
    """Make pipeline.execute() fail-open."""
    orig_execute = pipe.execute

    async def safe_execute(*args, **kwargs):
        try:
            return await orig_execute(*args, **kwargs)
        except FAIL_OPEN_EXCEPTIONS as exc:
            logger.error(f"SafeRedis: pipeline.execute fail-open: {exc}")
            return []  # pipelines return lists normally

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
            logger.error(f"SafeRedis iter fail-open: {exc}")
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
            logger.error(f"SafeRedis async-iter fail-open: {exc}")
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
        max_connections: int = 8,
        socket_keepalive: bool = True,
        socket_keepalive_options: Optional[dict] = None,
        health_check_interval: int = 30,
        retry_on_timeout: bool = False,
        retry: Any = None,
        **kwargs,
    ):
        if socket_keepalive_options is None:
            socket_keepalive_options = {
                socket.IPPROTO_TCP: {
                    socket.TCP_KEEPIDLE: 60,
                    socket.TCP_KEEPINTVL: 15,
                    socket.TCP_KEEPCNT: 4,
                }
            }

        self.default = default

        self.client = redis.Redis(
            host=host,
            port=port,
            db=db,
            password=password,
            socket_connect_timeout=socket_connect_timeout,
            socket_timeout=socket_timeout,
            max_connections=max_connections,
            socket_keepalive=socket_keepalive,
            socket_keepalive_options=socket_keepalive_options,
            health_check_interval=health_check_interval,
            retry_on_timeout=retry_on_timeout,
            retry=retry,
            **kwargs,
        )

    def __getattr__(self, name):
        attr = getattr(self.client, name)

        if not callable(attr):
            return attr

        def wrapper(*args, **kwargs):
            try:
                result = attr(*args, **kwargs)
            except FAIL_OPEN_EXCEPTIONS as exc:
                logger.error(f"SafeRedis fail-open on {name} (call): {exc}")
                return self.default

            if inspect.isawaitable(result):

                async def safe_coro():
                    try:
                        return await result
                    except FAIL_OPEN_EXCEPTIONS as exc:
                        logger.error(f"SafeRedis fail-open on {name} (await): {exc}")
                        return self.default

                return safe_coro()

            if is_pipeline(result):
                return wrap_pipeline(result, self.default)
            if is_async_iterable(result):
                return SafeAsyncIterator(result, default=self.default)
            if is_sync_iterable(result):
                return SafeIterator(result, default=self.default)
            return result

        return wrapper
