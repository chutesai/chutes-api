"""Instance connection helpers — httpx + HTTP/2 with TLS cert verification."""

import ssl
import socket
import asyncio
import httpx
import httpcore
from collections import OrderedDict
from cryptography import x509
from cryptography.x509.oid import NameOID

from api.util import semcomp

_POOL_MAX = 2048

# Aggressive TCP keepalive: detect dead peers in ~40s.
# 15s idle before first probe, then probe every 5s, give up after 5 failures.
_KEEPALIVE_SOCK_OPTS = [
    (socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1),
    (socket.IPPROTO_TCP, socket.TCP_KEEPIDLE, 15),
    (socket.IPPROTO_TCP, socket.TCP_KEEPINTVL, 5),
    (socket.IPPROTO_TCP, socket.TCP_KEEPCNT, 5),
]

# LRU caches keyed by instance_id — oldest entries evicted when full.
_ssl_cache: OrderedDict[str, tuple[ssl.SSLContext, str]] = OrderedDict()
_client_cache: OrderedDict[str, httpx.AsyncClient] = OrderedDict()


def _should_pool(instance) -> bool:
    """Only pool/cache when the instance has TLS (cacert) AND chutes_version >= 0.5.5."""
    return bool(instance.cacert) and semcomp(instance.chutes_version or "0.0.0", "0.5.5") >= 0


def _get_ssl_and_cn(instance) -> tuple[ssl.SSLContext, str]:
    """Get or create cached SSL context + CN for an instance."""
    iid = str(instance.instance_id)
    if iid in _ssl_cache:
        _ssl_cache.move_to_end(iid)
        return _ssl_cache[iid]

    ctx = ssl.create_default_context()
    extra = instance.extra or {}
    # Use CA cert for chain verification when available, fall back to server cert.
    ca_pem = extra.get("ca_cert") or instance.cacert
    ctx.load_verify_locations(cadata=ca_pem)

    # Load mTLS client cert if available.
    # Client key is sent unencrypted (no passphrase) from the miner.
    client_cert_pem = extra.get("client_cert")
    client_key_pem = extra.get("client_key")
    if client_cert_pem and client_key_pem:
        import tempfile
        import os

        with tempfile.NamedTemporaryFile(mode="wb", suffix=".pem", delete=False) as cf:
            cf.write(client_cert_pem.encode())
            cert_tmp = cf.name
        with tempfile.NamedTemporaryFile(mode="wb", suffix=".pem", delete=False) as kf:
            kf.write(client_key_pem.encode())
            key_tmp = kf.name
        try:
            ctx.load_cert_chain(certfile=cert_tmp, keyfile=key_tmp)
        finally:
            os.unlink(cert_tmp)
            os.unlink(key_tmp)

    cert = x509.load_pem_x509_certificate(instance.cacert.encode())
    cn = cert.subject.get_attributes_for_oid(NameOID.COMMON_NAME)[0].value
    _ssl_cache[iid] = (ctx, cn)
    if len(_ssl_cache) > _POOL_MAX:
        _ssl_cache.popitem(last=False)
    return ctx, cn


async def _graceful_close(client: httpx.AsyncClient) -> None:
    """Close an httpx client gracefully, giving HTTP/2 GOAWAY + TLS close_notify time."""
    try:
        await asyncio.wait_for(client.aclose(), timeout=5.0)
    except Exception:
        pass


def evict_instance_ssl(instance_id: str):
    """Remove cached SSL context and client when an instance is destroyed."""
    iid = str(instance_id)
    _ssl_cache.pop(iid, None)
    client = _client_cache.pop(iid, None)
    if client and not client.is_closed:
        try:
            loop = asyncio.get_running_loop()
            loop.create_task(_graceful_close(client))
        except RuntimeError:
            pass


def get_instance_url(instance, port: int | None = None) -> str:
    """Build the correct URL (https with CN or http with IP) for an instance."""
    p = port or instance.port
    if instance.cacert:
        _, cn = _get_ssl_and_cn(instance)
        return f"https://{cn}:{p}"
    return f"http://{instance.host}:{p}"


class _InstanceNetworkBackend(httpcore.AsyncNetworkBackend):
    """Resolves cert CN hostnames to instance IPs without external DNS lookups.

    httpx uses the URL hostname for TLS SNI and cert verification, then calls
    connect_tcp(hostname, port) for the actual TCP connection. We intercept
    connect_tcp and remap the CN hostname to the real IP. This means:
      - TLS SNI = hostname (correct, matches cert CN)
      - Cert verification = hostname vs cert CN (correct)
      - TCP connection = actual instance IP (correct, no DNS needed)
    """

    def __init__(self, hostname: str, ip: str):
        self._hostname = hostname
        self._ip = ip
        self._backend = httpcore.AnyIOBackend()

    async def connect_tcp(self, host, port, timeout=None, local_address=None, socket_options=None):
        actual_host = self._ip if host == self._hostname else host
        return await self._backend.connect_tcp(
            actual_host,
            port,
            timeout=timeout,
            local_address=local_address,
            socket_options=socket_options,
        )

    async def connect_unix_socket(self, path, timeout=None, socket_options=None):
        return await self._backend.connect_unix_socket(
            path,
            timeout=timeout,
            socket_options=socket_options,
        )

    async def sleep(self, seconds):
        await self._backend.sleep(seconds)


class _AsyncStreamWrapper(httpx.AsyncByteStream):
    """Wrap an httpcore async stream as an httpx AsyncByteStream."""

    def __init__(self, core_stream):
        self._stream = core_stream

    async def __aiter__(self):
        async for chunk in self._stream:
            yield chunk

    async def aclose(self):
        await self._stream.aclose()


class _CoreTransport(httpx.AsyncBaseTransport):
    """Wrap a raw httpcore pool so httpx <-> httpcore request mapping works.

    httpx 0.28+ passes string-based URLs to transports, but httpcore 1.0
    expects bytes-based URLs. httpx.AsyncHTTPTransport handles this internally
    but doesn't expose the network_backend param we need for custom DNS.
    """

    def __init__(self, pool: httpcore.AsyncConnectionPool):
        self._pool = pool

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        core_request = httpcore.Request(
            method=request.method.encode() if isinstance(request.method, str) else request.method,
            url=httpcore.URL(
                scheme=request.url.raw_scheme,
                host=request.url.raw_host,
                port=request.url.port,
                target=request.url.raw_path,
            ),
            headers=request.headers.raw,
            content=request.stream,
        )
        core_response = await self._pool.handle_async_request(core_request)
        return httpx.Response(
            status_code=core_response.status,
            headers=core_response.headers,
            stream=_AsyncStreamWrapper(core_response.stream),
            extensions=core_response.extensions,
        )

    async def aclose(self) -> None:
        await self._pool.aclose()


async def get_instance_client(instance, timeout: int = 600) -> tuple[httpx.AsyncClient, bool]:
    """Get or create an httpx AsyncClient for an instance.

    Returns (client, pooled) — caller must close the client when done if not pooled.
    Only HTTPS instances with chutes_version >= 0.5.5 are pooled (HTTP/2 multiplexing).

    For pooled clients the timeout baked into the client is generous (read=None);
    callers should pass per-request timeouts to .post()/.get() etc.
    For ephemeral clients the caller's timeout is set on the client directly.
    """
    pooled = _should_pool(instance)
    iid = str(instance.instance_id)

    if pooled and iid in _client_cache:
        client = _client_cache[iid]
        if not client.is_closed:
            _client_cache.move_to_end(iid)
            return client, True
        _client_cache.pop(iid, None)

    # Pooled clients use read=None so per-request timeouts can override.
    # Ephemeral clients bake in the caller's timeout directly.
    read_timeout = None if pooled else (float(timeout) if timeout else None)

    if instance.cacert:
        ssl_ctx, cn = _get_ssl_and_cn(instance)
        # Build httpcore pool with our custom resolver that maps CN → IP.
        # keepalive_expiry=75 matches the chute-side idle timeouts and avoids
        # httpcore silently dropping connections after 5s (the default), which
        # causes SSL shutdown timeouts on the peer because close_notify is
        # never sent for idle-evicted connections.
        pool = httpcore.AsyncConnectionPool(
            ssl_context=ssl_ctx,
            http2=True,
            network_backend=_InstanceNetworkBackend(hostname=cn, ip=instance.host),
            socket_options=_KEEPALIVE_SOCK_OPTS,
            keepalive_expiry=75,
        )
        client = httpx.AsyncClient(
            transport=_CoreTransport(pool),
            base_url=f"https://{cn}:{instance.port}",
            timeout=httpx.Timeout(connect=10.0, read=read_timeout, write=30.0, pool=10.0),
        )
    else:
        client = httpx.AsyncClient(
            base_url=f"http://{instance.host}:{instance.port}",
            timeout=httpx.Timeout(connect=10.0, read=read_timeout, write=30.0, pool=10.0),
        )

    if pooled:
        _client_cache[iid] = client
        if len(_client_cache) > _POOL_MAX:
            _, evicted = _client_cache.popitem(last=False)
            if evicted and not evicted.is_closed:
                try:
                    loop = asyncio.get_running_loop()
                    loop.create_task(_graceful_close(evicted))
                except RuntimeError:
                    pass

    return client, pooled
