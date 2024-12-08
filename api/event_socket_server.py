"""
Socket.IO poowered websocket server for event broadcasting.
"""

import asyncio
import socketio
from loguru import logger
from fastapi import FastAPI
import api.database.orms  # noqa
from api.redis_pubsub import RedisListener


sio = socketio.AsyncServer(async_mode="asgi", cors_allowed_origins="*")
fastapi_app = FastAPI()
app = socketio.ASGIApp(sio, fastapi_app)


@fastapi_app.on_event("startup")
async def initialize_socket_app():
    """
    Start our redis subscriber when the server starts.
    """
    fastapi_app.state.redis_listener = RedisListener(sio, "events")
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
    logger.info(f"Disconnected {session_id=}")
