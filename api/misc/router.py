"""
Router for misc. stuff, e.g. score proxy.
"""

import aiohttp
from loguru import logger
from urllib.parse import urlparse
from fastapi import APIRouter, Request, Response, HTTPException, status, Query
from fastapi.responses import StreamingResponse
from typing import AsyncIterator

router = APIRouter()

ALLOWED_DOMAINS = [
    "scoredata.me",
]


def is_url_allowed(url: str) -> bool:
    parsed = urlparse(url)
    return any(domain in parsed.netloc for domain in ALLOWED_DOMAINS)


@router.get("/proxy")
async def proxy(
    url: str,
    request: Request,
    stream: bool = Query(False, description="Stream the response for large files/videos"),
):
    if url == "ping":
        return {"pong": True}

    if not url.startswith(("http://", "https://")) or not is_url_allowed(url):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Invalid or unauthorized URL.",
        )

    # Configure headers to forward.
    headers_to_forward = {}
    skip_headers = {
        "host",
        "connection",
        "content-length",
        "content-encoding",
        "transfer-encoding",
        "upgrade",
    }
    for header_name, header_value in request.headers.items():
        if header_name.lower() not in skip_headers:
            headers_to_forward[header_name] = header_value
    timeout = aiohttp.ClientTimeout(connect=10.0, total=300.0)

    # Headers to forward from the upstream response
    forward_response_headers = [
        "content-type",
        "content-length",
        "accept-ranges",
        "content-range",
        "cache-control",
        "etag",
        "last-modified",
        "expires",
        "date",
    ]

    try:
        if stream:
            session = aiohttp.ClientSession(timeout=timeout)
            response = await session.get(url, headers=headers_to_forward)
            if not response.ok:
                await response.close()
                await session.close()
                raise HTTPException(
                    status_code=response.status,
                    detail=f"Upstream server returned {response.status}",
                )
            response_headers = {}
            for header in forward_response_headers:
                if header in response.headers:
                    response_headers[header] = response.headers[header]

            async def stream_content() -> AsyncIterator[bytes]:
                try:
                    async for chunk in response.content.iter_chunked(8192):
                        yield chunk
                finally:
                    await response.close()
                    await session.close()

            return StreamingResponse(
                stream_content(),
                status_code=response.status,
                headers=response_headers,
                media_type=response_headers.get("content-type", "application/octet-stream"),
            )

        else:
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.get(url, headers=headers_to_forward) as response:
                    content = await response.read()

                    response_headers = {}
                    for header in forward_response_headers:
                        if header in response.headers:
                            response_headers[header] = response.headers[header]

                    return Response(
                        content=content, status_code=response.status, headers=response_headers
                    )

    except HTTPException:
        raise
    except aiohttp.ClientTimeout:
        logger.error(f"WHITELIST_PROXY: upstream gateway timeout: {url=} {stream=}")
        raise HTTPException(
            status_code=status.HTTP_504_GATEWAY_TIMEOUT, detail="Upstream server timeout"
        )
    except aiohttp.ClientError as e:
        logger.error(
            f"WHITELIST_PROXY: upstream gateway request failed: {url=} {stream=} exception={str(e)}"
        )
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Upstream server returned error: {str(e)}",
        )
    except Exception as e:
        logger.error(
            f"WHITELIST_PROXY: unhandled exception proxying upstream request: {url=} {stream=} exception={str(e)}"
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Unhandled exception proxying request: {str(e)}",
        )
