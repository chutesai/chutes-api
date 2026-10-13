"""Support debug-override endpoints for the chute log store (mounted at ``/logs``).

The mTLS *ingest* endpoint lives on the instance router
(``POST /instances/launch_config/{config_id}/logs``) so all config-keyed lifecycle
operations stay together.

Owner and miner reads of stored Loki lines live on this same router. Clients never
supply raw LogQL; ``service.read_config_logs`` forces the matcher server-side.
"""

from typing import Optional

from fastapi import APIRouter, Depends, Header, HTTPException, Query, status
import jwt
from sqlalchemy import select

from api.chute_logs import service
from api.chute_logs.exceptions import LogCaptureError
from api.chute_logs.read_access import (
    authorize_miner_read,
    authorize_user_read,
    decode_launch_jwt,
    get_catalog,
    get_log_reader,
    lines_before_activation,
    require_instance,
    require_launch_config,
)
from api.chute.schemas import Chute
from api.database import get_session
from api.permissions import Permissioning
from api.user.service import get_current_user, require_role

router = APIRouter()

# Hotkey signature is bound to purpose "logs", same as the live pod stream.
logs_user = get_current_user(purpose="logs")

# Access control is declarative: every debug route depends on this gate, so the
# chutes_support requirement lives in the signature and is enforced before the handler
# runs — nothing relies on the body remembering to call a check. No `purpose`: support
# reaches these via bearer token (API key / JWT), which ignores it; it only matters for
# hotkey-signature auth.
_support = require_role(Permissioning.chutes_support)


async def _chute_must_exist(chute_id: str) -> str:
    """Dependency: 404 if the chute_id is unknown (resource validation, not access control)."""
    async with get_session(readonly=True) as session:
        if not await session.scalar(select(Chute.chute_id).where(Chute.chute_id == chute_id)):
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Chute not found.")
    return chute_id


@router.get("/debug/{chute_id}", dependencies=[Depends(_support), Depends(_chute_must_exist)])
async def get_debug_override(chute_id: str):
    """Report whether the per-chute log-capture debug override is enabled."""
    return {"chute_id": chute_id, "enabled": await service.is_debug_enabled(chute_id)}


@router.put("/debug/{chute_id}", dependencies=[Depends(_support), Depends(_chute_must_exist)])
async def set_debug_override(
    chute_id: str,
    ttl_seconds: Optional[int] = Query(None, ge=60, le=7 * 24 * 3600),
):
    """Enable log capture past instance activation for one chute (debugging).

    NOTE: this captures post-activation chute stdout for the window it is enabled.
    """
    if ttl_seconds is None:
        await service.set_debug(chute_id)
    else:
        await service.set_debug(chute_id, ttl_seconds)
    return {"chute_id": chute_id, "enabled": True}


@router.delete("/debug/{chute_id}", dependencies=[Depends(_support), Depends(_chute_must_exist)])
async def clear_debug_override(chute_id: str):
    """Disable the per-chute log-capture debug override."""
    await service.clear_debug(chute_id)
    return {"chute_id": chute_id, "enabled": False}


@router.get("/instances/{instance_id}")
async def read_instance_logs(
    instance_id: str,
    current_user=Depends(logs_user),
    catalog=Depends(get_catalog),
    read_logs=Depends(get_log_reader),
):
    try:
        instance = require_instance(await catalog.get_instance(instance_id))
        chute = await catalog.get_chute(instance.chute_id)
        shared = await catalog.is_shared_with(chute.chute_id, current_user.user_id)
        authorize_user_read(chute, current_user, shared)
    except LogCaptureError as exc:
        raise HTTPException(status_code=exc.http_status, detail=exc.message) from exc
    return await read_logs(instance.config_id, {"user_id": chute.user_id})


@router.get("/miner")
async def read_miner_logs(
    authorization: str | None = Header(None),
    catalog=Depends(get_catalog),
    read_logs=Depends(get_log_reader),
):
    # JWT sub selects the launch row. This route does not re-check the caller hotkey.
    if not authorization:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired launch token.",
        )
    token = authorization.strip().split(" ")[-1]
    try:
        payload = decode_launch_jwt(token)
    except jwt.InvalidTokenError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired launch token.",
        ) from None
    config_id = payload.get("sub")
    try:
        launch = require_launch_config(await catalog.get_launch_config(config_id))
        chute = await catalog.get_chute(launch.chute_id)
        authorize_miner_read(chute)
    except LogCaptureError as exc:
        raise HTTPException(status_code=exc.http_status, detail=exc.message) from exc
    # UNCLEAR: failed_at set, activated_at null, and no instance row. The live miner
    # stream returns 422 in that state. This read names activated_at only, so a public
    # config here returns the stored lines with no time cutoff. Raise with PaperMoney.
    instance = launch.instance
    activated_at = None if instance is None else instance.activated_at
    lines = await read_logs(config_id, {"miner_hotkey": launch.miner_hotkey})
    return lines_before_activation(lines, activated_at)
