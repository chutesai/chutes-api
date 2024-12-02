"""
Socket.IO poowered websocket server for continuous bi-directional vali/miner comms.
"""

import asyncio
import socketio
import redis.asyncio as redis
import api.constants as cst
import orjson as json
from typing import Optional
from datetime import datetime
from fastapi import Request, FastAPI, HTTPException
import api.database.orms  # noqa
from api.config import settings
from api.user.router import get_current_user
from typing import Dict
from loguru import logger


class SyntheticRequest(Request):
    """
    Synthetic requests to allow re-using our existing
    authentication logic within socket.io.
    """

    def __init__(self, headers: Dict[str, str]):
        scope = {
            "type": "http",
            "method": "GET",
            "headers": [(k.lower().encode(), v.encode()) for k, v in headers.items()],
            "query_string": b"",
        }
        super().__init__(scope)


class RedisListener:
    """
    Redis pubsub subscriber.
    """

    def __init__(self, socket_server, channel: str = "miner_broadcast"):
        self.sio = socket_server
        self.channel = channel
        self.pubsub: Optional[redis.client.PubSub] = None
        self.is_running = False
        self.last_reconnect = datetime.now()
        self.reconnect_attempts = 0
        self.max_reconnect_attempts = 10
        self.base_delay = 1
        self.max_delay = 30

    async def start(self):
        """
        Start the listener, handling connection/timeout errors.
        """
        self.is_running = True
        while self.is_running:
            try:
                if not self.pubsub:
                    self.pubsub = settings.redis_client.pubsub()
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
                    logger.debug(f"Broadcasting to miners: {data}")
                    await self.sio.emit("miner_broadcast", data)
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


sio = socketio.AsyncServer(async_mode="asgi", cors_allowed_origins="*")
fastapi_app = FastAPI()
app = socketio.ASGIApp(sio, fastapi_app)
authenticated_sockets = {}


@fastapi_app.on_event("startup")
async def initialize_socket_app():
    """
    Start our redis subscriber when the server starts.
    """
    fastapi_app.state.redis_listener = RedisListener(sio)
    asyncio.create_task(fastapi_app.state.redis_listener.start())


@fastapi_app.on_event("shutdown")
async def shutdown():
    """
    Shut down the redis listener when the app shuts down.
    """
    if hasattr(fastapi_app.state, "redis_listener"):
        await fastapi_app.state.redis_listener.stop()


@sio.event
async def connect(session_id: str, _):
    """
    New connection established.
    """
    logger.info(f"New socket.io connection from {session_id=}")


@sio.event
async def disconnect(session_id):
    """
    Client disconnect.
    """
    if session_id in authenticated_sockets:
        hotkey = authenticated_sockets[session_id]
        logger.info(f"Disconnected authenticated miner: {hotkey}")
        authenticated_sockets.pop(session_id)


@sio.event
async def authenticate(session_id: str, headers: Dict[str, str]) -> bool:
    """
    Authentication request from a client (miner).  Headers here aren't
    really headers since this is socket.io, but we'll treat them as such.
    """
    try:
        request = SyntheticRequest(headers)
        _ = await get_current_user(registered_to=settings.netuid, purpose="sockets")(
            request=request,
            hotkey=headers.get(cst.HOTKEY_HEADER),
            signature=headers.get(cst.SIGNATURE_HEADER),
            nonce=headers.get(cst.NONCE_HEADER),
        )
        hotkey = headers.get(cst.HOTKEY_HEADER)
        logger.info(f"Successfully authenticated miner {hotkey=}, {session_id=}")
        authenticated_sockets[session_id] = hotkey
        await sio.emit("auth_success", {"message": "Authenticated"}, to=session_id)
        return True
    except HTTPException as e:
        error_msg = f"Authentication failed: {e.detail}"
        logger.warning(error_msg)
        await sio.emit("auth_failed", {"error": error_msg}, to=session_id)
    except Exception as e:
        error_msg = f"Unexpected error: {str(e)}"
        logger.error(error_msg)
        await sio.emit("auth_failed", {"error": error_msg}, to=session_id)
    await sio.disconnect(session_id)
    return False


@sio.event
async def miner_message(session_id, data):
    """
    Placeholder for miner originated messages, not really used (yet).
    """
    if session_id not in authenticated_sockets:
        logger.warning(f"Unauthenticated message from {session_id}")
        await sio.disconnect(session_id)
        return
    hotkey = authenticated_sockets[session_id]
    logger.debug(f"Received message from miner {hotkey=}: {data=}")
