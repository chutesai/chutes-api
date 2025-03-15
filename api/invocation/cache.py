import asyncio
import time
from loguru import logger
from api.config import settings


CACHE_PREFIX = "llm_sc:"


async def stream_response(prompt_id, model):
    """
    Yield chunks from a redis stream directly instead of routing to miner.
    """
    started_at = time.time()
    last_offset = None
    chunks = 0
    finished = False
    while not finished and time.time() - started_at <= 600:
        stream_result = None
        try:
            stream_result = await settings.llm_cache_client.xrange(
                f"{CACHE_PREFIX}{prompt_id}", last_offset or "-", "+"
            )
        except Exception as exc:
            logger.warning(f"Error fetching stream result: {exc}")
        if not stream_result:
            await asyncio.sleep(0.02)
            continue
        for offset, data in stream_result:
            last_offset = offset.decode()
            parts = last_offset.split("-")
            last_offset = parts[0] + "-" + str(int(parts[1]) + 1)
            if data[b"chunk"] == b"[[__END__]]":
                finished = True
                break
            chunks += data[b"chunk"].count(b"data: ")
            yield data[b"chunk"]

    delta = time.time() - started_at
    logger.success(
        "LLMCACHE: \N{GRINNING FACE} "
        + f"{prompt_id=} of {model=} streamed {chunks} chunks from redis in {delta} at {chunks / delta} tps"
    )


async def append_stream(prompt_id, chunk):
    """
    Add chunk data to an open redis stream.
    """
    try:
        await settings.llm_cache_client.xadd(f"{CACHE_PREFIX}{prompt_id}", {"chunk": chunk})
    except Exception as exc:
        logger.error(f"Error appending cache stream: {exc}")
        await purge_stream(prompt_id)


async def set_stream_expiration(prompt_id):
    """
    Set an expiration for a prompt.
    """
    await settings.llm_cache_client.expire(f"{CACHE_PREFIX}{prompt_id}", 300)


async def cached_responder(prompt_id, model) -> dict:
    """
    Send output from redis cache for duplicate outputs.
    """
    stream_length = await settings.llm_cache_client.xlen(f"{CACHE_PREFIX}{prompt_id}")
    if stream_length > 0:
        logger.info(f"LLMCACHE: found stream for {prompt_id=} of {model=}")
        return stream_response(prompt_id, model)
    return None


async def purge_stream(prompt_id):
    """
    Delete a stream cache.
    """
    try:
        await settings.llm_cache_client.delete(f"{CACHE_PREFIX}{prompt_id}")
    except Exception as exc:
        logger.error(f"Error purging cache stream: {exc}")
