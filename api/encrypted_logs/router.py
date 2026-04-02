"""API endpoints for retrieving encrypted startup logs."""

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select

from api.chute.schemas import Chute
from api.database import get_db_session
from api.encrypted_logs.capture import get_encrypted_log_chunks, get_encrypted_log_sessions
from api.user.schemas import User
from api.user.service import get_current_user

router = APIRouter()


@router.get("/{chute_id}/sessions")
async def list_encrypted_log_sessions(
    chute_id: str,
    db: AsyncSession = Depends(get_db_session),
    current_user: User = Depends(get_current_user(purpose="chutes")),
):
    """List encrypted log capture sessions for a chute (owner only)."""
    chute = (
        (await db.execute(select(Chute).where(Chute.chute_id == chute_id)))
        .unique()
        .scalar_one_or_none()
    )
    if not chute or chute.user_id != current_user.user_id:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Chute not found, or does not belong to you",
        )

    sessions = await get_encrypted_log_sessions(chute_id)
    return sessions


@router.get("/{chute_id}/sessions/{instance_id}/chunks")
async def list_encrypted_log_chunks(
    chute_id: str,
    instance_id: str,
    db: AsyncSession = Depends(get_db_session),
    current_user: User = Depends(get_current_user(purpose="chutes")),
):
    """Fetch encrypted log chunks for a specific instance (owner only).

    Returns a list of base64-encoded encrypted chunks. The client must
    use the ephemeral_pubkey from the session metadata + their private
    key to ECDH-decrypt each chunk.
    """
    chute = (
        (await db.execute(select(Chute).where(Chute.chute_id == chute_id)))
        .unique()
        .scalar_one_or_none()
    )
    if not chute or chute.user_id != current_user.user_id:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Chute not found, or does not belong to you",
        )

    # Verify instance_id actually belongs to this chute's sessions.
    sessions = await get_encrypted_log_sessions(chute_id)
    if not any(s["instance_id"] == instance_id for s in sessions):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No encrypted logs found for this instance",
        )

    chunks = await get_encrypted_log_chunks(instance_id)
    if not chunks:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No encrypted logs found for this instance",
        )

    return {
        "instance_id": instance_id,
        "chute_id": chute_id,
        "chunks": [c.decode() if isinstance(c, bytes) else c for c in chunks],
    }
