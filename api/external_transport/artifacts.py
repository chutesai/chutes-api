"""Streaming relay for remote artifacts without local blob persistence."""

from __future__ import annotations

import asyncio
from typing import AsyncIterator, Mapping

import aiohttp

from .errors import (
    RedirectRejectedError,
    RequestRejectedError,
    ResponseTooLargeError,
    UpstreamConnectionError,
    UpstreamTimeoutError,
)
from .models import ArtifactProfile, SecretResolver
from .security import (
    create_connector,
    render_secret_headers,
    sanitize_artifact_request_headers,
    sanitize_response_headers,
    strip_headers_for_cross_origin_redirect,
    validate_profile_headers,
    validate_redirect,
    validate_url,
)


_REDIRECT_STATUSES = frozenset({301, 302, 303, 307, 308})


class ArtifactResponse:
    """Metadata and a one-shot byte iterator for a remote artifact."""

    def __init__(
        self,
        *,
        status_code: int,
        headers: Mapping[str, str],
        chunk_bytes: int,
        max_bytes: int,
        response: aiohttp.ClientResponse | None = None,
        session: aiohttp.ClientSession | None = None,
    ) -> None:
        self.status_code = status_code
        self.headers = headers
        self._chunk_bytes = chunk_bytes
        self._max_bytes = max_bytes
        self._response = response
        self._session = session
        self._started = False
        self._closed = response is None

    async def iter_bytes(self, *, max_bytes: int | None = None) -> AsyncIterator[bytes]:
        """Relay the upstream body once, applying normal consumer backpressure."""

        if self._started:
            raise RuntimeError("artifact response can only be consumed once")
        self._started = True
        if self._response is None:
            return
        effective_max = self._max_bytes
        if max_bytes is not None:
            if (
                isinstance(max_bytes, bool)
                or not isinstance(max_bytes, int)
                or max_bytes < 0
            ):
                raise ValueError("artifact iterator max_bytes must be non-negative")
            effective_max = min(effective_max, max_bytes)
        total = 0
        try:
            async for chunk in self._response.content.iter_chunked(self._chunk_bytes):
                total += len(chunk)
                if total > effective_max:
                    raise ResponseTooLargeError(
                        "artifact response exceeds configured byte limit"
                    )
                yield chunk
        except asyncio.TimeoutError as exc:
            raise UpstreamTimeoutError("artifact stream timed out") from exc
        except aiohttp.ClientError as exc:
            raise UpstreamConnectionError("artifact stream ended unexpectedly") from exc
        finally:
            await self.aclose()

    async def aclose(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._response is not None:
            self._response.close()
        if self._session is not None:
            await self._session.close()

    async def __aenter__(self) -> ArtifactResponse:
        return self

    async def __aexit__(self, *_args: object) -> None:
        await self.aclose()


class ArtifactRelay:
    """Open GET or HEAD requests under an explicit artifact security profile."""

    def __init__(
        self,
        profile: ArtifactProfile,
        *,
        secret_resolver: SecretResolver | None = None,
    ) -> None:
        self._profile = profile
        self._secret_resolver = secret_resolver

    async def open(
        self,
        url: str,
        *,
        method: str = "GET",
        request_headers: Mapping[str, str] | None = None,
    ) -> ArtifactResponse:
        method = method.upper()
        if method not in {"GET", "HEAD"}:
            raise RequestRejectedError("artifact relay supports only GET and HEAD")
        current_url = validate_url(url, self._profile.network)
        current_headers = sanitize_artifact_request_headers(request_headers or {})
        current_headers.update(validate_profile_headers(self._profile.static_headers))
        secret_headers, _secret_names = await render_secret_headers(
            self._profile.secret_headers, self._secret_resolver
        )
        current_headers.update(secret_headers)
        current_headers.setdefault("accept-encoding", "identity")
        timeout = aiohttp.ClientTimeout(
            total=self._profile.timeout.total,
            connect=self._profile.timeout.connect,
            sock_connect=self._profile.timeout.socket_connect,
            sock_read=self._profile.timeout.socket_read,
        )
        session = aiohttp.ClientSession(
            connector=create_connector(self._profile.network),
            timeout=timeout,
            auto_decompress=False,
            cookie_jar=aiohttp.DummyCookieJar(),
            skip_auto_headers={"User-Agent"},
            trust_env=False,
        )
        response: aiohttp.ClientResponse | None = None
        handed_off = False
        try:
            redirect_count = 0
            while True:
                response = await session.request(
                    method,
                    current_url,
                    headers=current_headers,
                    allow_redirects=False,
                )
                if response.status not in _REDIRECT_STATUSES:
                    break
                if redirect_count >= self._profile.redirects.max_redirects:
                    raise RedirectRejectedError("artifact redirect limit exceeded")
                next_url, cross_origin = validate_redirect(
                    current_url,
                    response.headers.get("Location", ""),
                    self._profile.network,
                    self._profile.redirects,
                )
                response.release()
                response = None
                if cross_origin:
                    current_headers = strip_headers_for_cross_origin_redirect(
                        current_headers
                    )
                current_url = next_url
                redirect_count += 1

            declared = response.headers.get("Content-Length")
            if declared:
                try:
                    if int(declared) > self._profile.max_bytes:
                        raise ResponseTooLargeError(
                            "artifact response exceeds configured byte limit"
                        )
                except ValueError:
                    pass
            headers = sanitize_response_headers(
                response.headers,
                frozenset(),
                artifact=True,
            )
            if method == "HEAD":
                status_code = response.status
                response.release()
                response = None
                await session.close()
                return ArtifactResponse(
                    status_code=status_code,
                    headers=headers,
                    chunk_bytes=self._profile.stream_chunk_bytes,
                    max_bytes=self._profile.max_bytes,
                )
            result = ArtifactResponse(
                status_code=response.status,
                headers=headers,
                chunk_bytes=self._profile.stream_chunk_bytes,
                max_bytes=self._profile.max_bytes,
                response=response,
                session=session,
            )
            response = None
            handed_off = True
            return result
        except asyncio.TimeoutError as exc:
            raise UpstreamTimeoutError("artifact request timed out") from exc
        except aiohttp.ClientError as exc:
            raise UpstreamConnectionError("artifact request failed") from exc
        finally:
            if response is not None and not response.closed:
                response.close()
            if not handed_off and not session.closed:
                await session.close()
