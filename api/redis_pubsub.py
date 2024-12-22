"""
Redis pubsub classes and methods.
"""

import os
import asyncio
import orjson as json
import redis.asyncio as redis
from datetime import datetime
import api.database.orms  # noqa
from loguru import logger


class RedisListener:
    """
    Redis pubsub subscriber.
    """

    def __init__(self, socket_server, channel: str):
        self.sio = socket_server
        self.channel = channel
        self.is_running = False
        self.pubsub = None
        self.last_reconnect = datetime.now()
        self.reconnect_attempts = 0
        self.max_reconnect_attempts = 10
        self.base_delay = 1
        self.max_delay = 30
        self.redis = redis.Redis.from_url(os.getenv("REDIS_URL", "redis://127.0.0.1:6379/0"))

    async def start(self):
        """
        Start the listener, handling connection/timeout errors.
        """
        self.is_running = True
        while self.is_running:
            try:
                if not self.pubsub:
                    self.pubsub = self.redis.pubsub()
                    await self.pubsub.subscribe(self.channel)
                    logger.info(f"Subscribed to channel: {self.channel}")
                    self.reconnect_attempts = 0
                await self._listen()
            except (redis.ConnectionError, redis.TimeoutError) as e:
                await self._handle_connection_error(e)
            except Exception as e:
                logger.error(f"Unexpected error in redis listener: {e}")
                await self._handle_connection_error(e)

    async def stop(self):
        """
        Gracefully stop the listener.
        """
        self.is_running = False
        if self.pubsub:
            await self.pubsub.unsubscribe(self.channel)
            await self.pubsub.close()
            self.pubsub = None
        logger.info("Redis listener stopped")

    async def _listen(self):
        """
        Main listening loop.
        """
        async for message in self.pubsub.listen():
            if not self.is_running:
                break
            if message["type"] == "message":
                try:
                    data = json.loads(message["data"].decode())
                    if (
                        hasattr(self.sio, "reverse_map")
                        and (send_to := data.get("filter_recipients")) is not None
                    ):
                        for hotkey in send_to:
                            if (room := self.sio.reverse_map.get(hotkey)) is not None:
                                logger.debug(f"Notifying {hotkey=}: {data}")
                                await self.sio.emit(self.channel, data, room=room)
                    else:
                        logger.debug(f"Broadcasting: {data}")
                        await self.sio.emit(self.channel, data)
                except Exception as exc:
                    logger.error(f"Error processing message: {exc}")

    async def _handle_connection_error(self, error):
        """
        Handle connection errors with exponential backoff.
        """
        self.reconnect_attempts += 1
        if self.reconnect_attempts > self.max_reconnect_attempts:
            logger.error("Max reconnection attempts reached. Stopping listener.")
            await self.stop()
            return
        delay = min(self.base_delay * (2 ** (self.reconnect_attempts - 1)), self.max_delay)
        logger.warning(
            f"Redis connection error: {error}, attempt {self.reconnect_attempts}/{self.max_reconnect_attempts}, retrying in {delay} seconds..."
        )
        if self.pubsub:
            try:
                await self.pubsub.close()
            except Exception as exc:
                logger.warning(f"Redis pubsub close error: {exc}")
                pass
            self.pubsub = None
        await asyncio.sleep(delay)
