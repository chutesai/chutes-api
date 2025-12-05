import asyncio
import socket
import concurrent.futures
from typing import Any, Callable
from loguru import logger
import aiomcache


FAIL_OPEN_EXCEPTIONS = (
    ConnectionError,
    TimeoutError,
    socket.timeout,
    socket.error,
    socket.gaierror,
    OSError,
    asyncio.TimeoutError,
    concurrent.futures.TimeoutError,
    BrokenPipeError,
    ConnectionResetError,
    RuntimeError,
)


class SafeMemcached:
    """
    Fail-open wrapper for aiomcache.Client.
    """

    def __init__(
        self,
        host: str = "172.16.0.100",
        port: int = 22122,
        *,
        pool_size: int = 4,
        default: Any = None,
    ):
        self.default = default
        self._client = aiomcache.Client(
            host=host.encode() if isinstance(host, str) else host,
            port=port,
            pool_size=pool_size,
        )

    @property
    def client(self) -> aiomcache.Client:
        return self._client

    def __getattr__(self, name: str) -> Callable[..., Any]:
        attr = getattr(self._client, name)

        if not callable(attr):
            return attr

        async def safe_call(*args, **kwargs):
            try:
                result = attr(*args, **kwargs)
                if asyncio.iscoroutine(result):
                    try:
                        return await asyncio.wait_for(result, 0.5)
                    except FAIL_OPEN_EXCEPTIONS as exc:
                        logger.error(f"SafeMemcached fail-open on {name}(await): {exc}")
                        return self.default
                return result
            except FAIL_OPEN_EXCEPTIONS as exc:
                logger.error(f"SafeMemcached fail-open on {name}(call): {exc}")
                return self.default

        return safe_call

    async def close(self):
        try:
            await self._client.close()
        except FAIL_OPEN_EXCEPTIONS as exc:
            logger.warning(f"SafeMemcached close() fail-open: {exc}")
