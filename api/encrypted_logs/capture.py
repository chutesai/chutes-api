"""
Background startup log capture for private chute instances.

Captures logs from instance creation through activation only (startup window).
Logs are encrypted with the chute owner's Sr25519 public key and stored in
Redis streams with 4h TTL so users can debug failed warmups.
"""

import asyncio
import time
import pybase64
import orjson as json
from loguru import logger
from scalecodec.utils.ss58 import ss58_decode
from sqlalchemy.future import select

import api.miner_client as miner_client
from api.config import settings
from api.database import get_session
from api.encrypted_logs.crypto import (
    generate_ephemeral_keypair,
    encrypt_log_chunk,
    validate_ristretto_point,
)

# TTL for encrypted log streams in Redis (4 hours)
LOG_STREAM_TTL_SECONDS = 4 * 60 * 60

# Max capture duration (30 minutes — well beyond normal startup)
MAX_CAPTURE_SECONDS = 30 * 60

# Max entries per stream
LOG_STREAM_MAXLEN = 5000

# How many log lines to batch before encrypting
BATCH_SIZE = 10

# Sentinel string that indicates the instance has activated
ACTIVATION_SENTINEL = "Instance activated:"

# Redis key prefixes
STREAM_KEY_PREFIX = "encrypted_logs:"
SESSION_KEY_PREFIX = "encrypted_log_sessions:"


def _stream_key(instance_id: str) -> str:
    return f"{STREAM_KEY_PREFIX}{instance_id}"


def _sessions_key(chute_id: str) -> str:
    return f"{SESSION_KEY_PREFIX}{chute_id}"


def _user_pubkey_from_ss58(ss58_address: str) -> bytes:
    """Decode an SS58 address to raw 32-byte public key."""
    hex_str = ss58_decode(ss58_address)
    return bytes.fromhex(hex_str)


async def start_encrypted_log_capture(
    instance_id: str,
    config_id: str,
    chute_id: str,
    owner_ss58: str,
    host: str,
    log_port: int,
    miner_hotkey: str,
):
    """
    Start a background task to capture and encrypt startup logs for a private
    chute instance. Captures from creation through activation only.

    Should be called as: asyncio.create_task(start_encrypted_log_capture(...))
    """
    try:
        user_pubkey = _user_pubkey_from_ss58(owner_ss58)
        if not validate_ristretto_point(user_pubkey):
            logger.warning(
                f"Owner pubkey for {chute_id} is not a valid Ristretto point, "
                f"skipping encrypted log capture"
            )
            return

        ephemeral_scalar, ephemeral_pubkey = generate_ephemeral_keypair()

        # Register session so the client can find and decrypt it later
        session_info = {
            "instance_id": instance_id,
            "config_id": config_id,
            "chute_id": chute_id,
            "ephemeral_pubkey": pybase64.b64encode(ephemeral_pubkey).decode(),
            "stream_key": _stream_key(instance_id),
            "started_at": time.time(),
        }
        await settings.redis_client.client.rpush(
            _sessions_key(chute_id),
            json.dumps(session_info).decode(),
        )
        await settings.redis_client.client.expire(_sessions_key(chute_id), LOG_STREAM_TTL_SECONDS)

        logger.info(
            f"Starting encrypted startup log capture for instance {instance_id} chute {chute_id}"
        )

        await _capture_startup_logs(
            instance_id=instance_id,
            config_id=config_id,
            host=host,
            log_port=log_port,
            miner_hotkey=miner_hotkey,
            user_pubkey=user_pubkey,
            ephemeral_scalar=ephemeral_scalar,
            ephemeral_pubkey=ephemeral_pubkey,
        )
    except asyncio.CancelledError:
        pass
    except Exception as exc:
        logger.warning(f"Encrypted log capture failed for {instance_id}: {exc}")
    finally:
        _set_stream_ttl(instance_id)


def _set_stream_ttl(instance_id: str):
    """Ensure stream TTL is set (fire and forget)."""

    async def _do():
        try:
            key = _stream_key(instance_id)
            ttl = await settings.redis_client.client.ttl(key)
            if ttl == -1:
                await settings.redis_client.client.expire(key, LOG_STREAM_TTL_SECONDS)
        except Exception:
            pass

    asyncio.create_task(_do())


async def _is_instance_activated(config_id: str) -> bool:
    """Check if the instance has been activated by querying the database."""
    from api.instance.schemas import Instance

    try:
        async with get_session(readonly=True) as session:
            instance = (
                (await session.execute(select(Instance).where(Instance.config_id == config_id)))
                .unique()
                .scalar_one_or_none()
            )
            if not instance:
                return True  # Instance gone — stop capturing
            return instance.activated_at is not None
    except Exception:
        return False


async def _capture_startup_logs(
    instance_id: str,
    config_id: str,
    host: str,
    log_port: int,
    miner_hotkey: str,
    user_pubkey: bytes,
    ephemeral_scalar: bytes,
    ephemeral_pubkey: bytes,
):
    """
    Stream startup logs from the instance, encrypt, and store in Redis stream.
    Stops when:
      - Instance activates (ACTIVATION_SENTINEL in logs or activated_at set in DB)
      - Instance disappears
      - MAX_CAPTURE_SECONDS deadline reached
      - Log stream ends
    """
    import httpx

    stream_key = _stream_key(instance_id)
    deadline = time.monotonic() + MAX_CAPTURE_SECONDS

    headers, _ = miner_client.sign_request(miner_hotkey, purpose="chutes")
    client = httpx.AsyncClient(
        base_url=f"http://{host}:{log_port}",
        timeout=httpx.Timeout(connect=10.0, read=None, write=30.0, pool=10.0),
    )

    try:
        batch = []
        line_buf = ""
        async with client.stream(
            "GET",
            "/logs/stream",
            headers=headers,
            params={"backfill": "10000"},
        ) as resp:
            if resp.status_code != 200:
                logger.warning(f"Startup log stream for {instance_id} returned {resp.status_code}")
                return

            check_counter = 0
            async for chunk in resp.aiter_bytes():
                if time.monotonic() >= deadline:
                    logger.info(f"Startup log capture deadline reached for {instance_id}")
                    break

                line_buf += chunk.decode("utf-8", errors="replace")
                while "\n" in line_buf:
                    line, line_buf = line_buf.split("\n", 1)
                    line = line.strip()
                    if not line or line == ".":
                        continue

                    # Check for activation sentinel
                    if ACTIVATION_SENTINEL in line:
                        # Store any remaining batch, then stop
                        if line.startswith("data: "):
                            try:
                                log_msg = json.loads(line[6:]).get("log", "")
                                if log_msg and log_msg.strip() and len(log_msg.strip()) > 1:
                                    batch.append(log_msg)
                            except Exception:
                                pass
                        if batch:
                            await _store_batch(
                                stream_key,
                                batch,
                                user_pubkey,
                                ephemeral_scalar,
                                ephemeral_pubkey,
                            )
                        logger.info(
                            f"Instance {instance_id} activated, stopping startup log capture"
                        )
                        return

                    # Parse log line
                    if line.startswith("data: "):
                        data_str = line[6:].strip()
                        if not data_str:
                            continue
                        try:
                            data = json.loads(data_str)
                            log_msg = data.get("log", "")
                            if log_msg and log_msg.strip() and len(log_msg.strip()) > 1:
                                batch.append(log_msg)
                        except Exception:
                            continue

                    # Encrypt and store batch when full
                    if len(batch) >= BATCH_SIZE:
                        await _store_batch(
                            stream_key,
                            batch,
                            user_pubkey,
                            ephemeral_scalar,
                            ephemeral_pubkey,
                        )
                        batch = []

                # Periodically check DB for activation (every ~50 chunks)
                check_counter += 1
                if check_counter % 50 == 0:
                    if await _is_instance_activated(config_id):
                        if batch:
                            await _store_batch(
                                stream_key,
                                batch,
                                user_pubkey,
                                ephemeral_scalar,
                                ephemeral_pubkey,
                            )
                        logger.info(
                            f"Instance {instance_id} activated (DB check), "
                            f"stopping startup log capture"
                        )
                        return

        # Stream ended — store remaining batch
        if batch:
            await _store_batch(
                stream_key,
                batch,
                user_pubkey,
                ephemeral_scalar,
                ephemeral_pubkey,
            )
    finally:
        await client.aclose()


async def _store_batch(
    stream_key: str,
    batch: list[str],
    user_pubkey: bytes,
    ephemeral_scalar: bytes,
    ephemeral_pubkey: bytes,
):
    """Encrypt a batch of log lines and add to the Redis stream."""
    plaintext = "\n".join(batch).encode("utf-8")
    encrypted = encrypt_log_chunk(plaintext, user_pubkey, ephemeral_scalar, ephemeral_pubkey)

    try:
        await settings.redis_client.client.xadd(
            stream_key,
            {"data": pybase64.b64encode(encrypted)},
            maxlen=LOG_STREAM_MAXLEN,
        )
        # Ensure TTL is set
        ttl = await settings.redis_client.client.ttl(stream_key)
        if ttl == -1:
            await settings.redis_client.client.expire(stream_key, LOG_STREAM_TTL_SECONDS)
    except Exception as exc:
        logger.debug(f"Failed to store encrypted log batch: {exc}")


async def get_encrypted_log_sessions(chute_id: str) -> list[dict]:
    """Get all encrypted log sessions for a chute that still have data."""
    key = _sessions_key(chute_id)
    try:
        raw_sessions = await settings.redis_client.client.lrange(key, 0, -1)
        sessions = []
        for raw in raw_sessions:
            session = json.loads(raw)
            stream_key = session.get("stream_key", "")
            stream_len = await settings.redis_client.client.xlen(stream_key)
            if stream_len > 0:
                session["chunk_count"] = stream_len
                sessions.append(session)
        return sessions
    except Exception as exc:
        logger.warning(f"Failed to get encrypted log sessions: {exc}")
        return []


async def get_encrypted_log_chunks(instance_id: str, count: int = 5000) -> list[bytes]:
    """Read encrypted log chunks from a Redis stream."""
    stream_key = _stream_key(instance_id)
    try:
        entries = await settings.redis_client.client.xrange(stream_key, count=count)
        return [entry[1][b"data"] for entry in entries]
    except Exception as exc:
        logger.warning(f"Failed to read encrypted logs: {exc}")
        return []
