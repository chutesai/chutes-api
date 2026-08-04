"""Support debug-override endpoints for the chute log store (mounted at ``/logs``).

The mTLS *ingest* endpoint lives on the instance router
(``POST /instances/launch_config/{config_id}/logs``) so all config-keyed lifecycle
operations stay together.

The owner-facing read is intentionally **not exposed yet**: reads go through the
internal Grafana until a standalone, security-reviewed follow-up wires the
user-facing read (see spec §5). The primitive it will use —
``service.read_config_logs``, server-forced to the owner's ``user_id`` (spec §6) —
already exists and is unit-tested, so the follow-up only needs to add the route and
its auth. Loki is never client-reachable; clients never supply raw LogQL.
"""

from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy import select

from api.chute_logs import service
from api.chute.schemas import Chute
from api.database import get_session
from api.permissions import Permissioning
from api.user.service import require_role

router = APIRouter()

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
