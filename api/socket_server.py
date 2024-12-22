"""
Socket.IO poowered websocket server for continuous bi-directional vali/miner comms.
"""

import asyncio
import socketio
import api.constants as cst
from typing import Dict
from loguru import logger
from fastapi import FastAPI, HTTPException
import api.database.orms  # noqa
from api.config import settings
from api.user.router import get_current_user
from api.socket_shared import SyntheticRequest
from api.redis_pubsub import RedisListener


sio = socketio.AsyncServer(async_mode="asgi", cors_allowed_origins="*")
fastapi_app = FastAPI()
app = socketio.ASGIApp(sio, fastapi_app)
sio.session_map = {}
sio.reverse_map = {}


@fastapi_app.on_event("startup")
async def initialize_socket_app():
    """
    Start our redis subscriber when the server starts.
    """
    fastapi_app.state.redis_listener = RedisListener(sio, "miner_broadcast")
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
    if (hotkey := sio.session_map.pop(session_id, None)) is not None:
        sio.reverse_map.pop(hotkey, None)
        logger.info(f"Disconnected authenticated miner: {hotkey}")


@sio.event
async def authenticate(session_id: str, headers: Dict[str, str]) -> bool:
    """
    Authentication request from a client (miner).  Headers here aren't
    really headers since this is socket.io, but we'll treat them as such.
    """
    try:
        request = SyntheticRequest(headers)
        _ = await get_current_user(
            raise_not_found=False, registered_to=settings.netuid, purpose="sockets"
        )(
            request=request,
            hotkey=headers.get(cst.HOTKEY_HEADER),
            signature=headers.get(cst.SIGNATURE_HEADER),
            nonce=headers.get(cst.NONCE_HEADER),
        )
        hotkey = headers.get(cst.HOTKEY_HEADER)
        logger.info(f"Successfully authenticated miner {hotkey=}, {session_id=}")
        sio.session_map[session_id] = hotkey
        sio.reverse_map[hotkey] = session_id
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
    if (hotkey := sio.session_map.get(session_id)) is None:
        logger.warning(f"Unauthenticated message from {session_id}")
        await sio.disconnect(session_id)
        return
    logger.debug(f"Received message from miner {hotkey=}: {data=}")
