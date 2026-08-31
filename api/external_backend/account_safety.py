"""Mutation safeguards for accounts used by retained upstream operations."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Mapping

from fastapi import HTTPException, status
from sqlalchemy import or_, select, text
from sqlalchemy.ext.asyncio import AsyncSession

from api.external_backend.schemas import (
    ExternalOperation,
    ExternalOperationStatus,
)


_ACTIVE_STATUSES = frozenset(
    {
        ExternalOperationStatus.PENDING.value,
        ExternalOperationStatus.SUBMITTED.value,
        ExternalOperationStatus.RUNNING.value,
    }
)


def _timestamp(value: object) -> datetime | None:
    if isinstance(value, datetime):
        parsed = value
    elif isinstance(value, str):
        normalized = value[:-1] + "+00:00" if value.endswith("Z") else value
        try:
            parsed = datetime.fromisoformat(normalized)
        except ValueError:
            return None
    else:
        return None
    return parsed.replace(tzinfo=parsed.tzinfo or timezone.utc)


def operation_blocks_account_mutation(
    operation: object,
    *,
    now: datetime,
    artifact_relay_invalidated_at: datetime | None = None,
) -> bool:
    """Return whether current account transport config is still needed."""

    status_value = getattr(operation, "status", None)
    if isinstance(status_value, ExternalOperationStatus):
        status_value = status_value.value
    # Every live mode still depends on the transport configuration. In particular,
    # sync/stream work can resolve a secret after the operation row is inserted.
    if status_value in _ACTIVE_STATUSES:
        return True

    if artifact_relay_invalidated_at is not None:
        created_at = _timestamp(getattr(operation, "created_at", None))
        # The emergency cutoff is authoritative only for work known to predate it.
        # Missing or malformed historical timestamps remain fail-closed.
        if created_at is not None and created_at <= artifact_relay_invalidated_at:
            return False

    descriptor = getattr(operation, "result_descriptor", None)
    if not isinstance(descriptor, Mapping):
        return False
    artifacts = descriptor.get("artifacts")
    if not isinstance(artifacts, list) or not artifacts:
        return False

    operation_expiration = _timestamp(getattr(operation, "expires_at", None))
    for artifact in artifacts:
        if not isinstance(artifact, Mapping):
            # Malformed retained data must not make a safety guard fail open.
            return True
        raw_expiration = artifact.get("expires_at")
        expiration = (
            _timestamp(raw_expiration)
            if raw_expiration is not None
            else operation_expiration
        )
        if raw_expiration is not None and expiration is None:
            return True
        if expiration is None or expiration > now:
            return True
    return False


async def ensure_account_transport_mutation_safe(
    db: AsyncSession,
    account_id: str,
) -> None:
    """Lock an account and reject changes needed by live or relayed work."""

    # An operation insert takes a key-share lock for this foreign key. Locking the
    # account first closes the race between this check and the eventual update.
    locked_account = await db.execute(
        text(
            "SELECT artifact_relay_invalidated_at FROM external_backend_accounts "
            "WHERE account_id = :account_id FOR UPDATE"
        ),
        {"account_id": account_id},
    )
    artifact_relay_invalidated_at = _timestamp(locked_account.scalar_one_or_none())
    now = _timestamp((await db.execute(text("SELECT clock_timestamp()"))).scalar_one())
    if now is None:
        raise RuntimeError("database clock did not return a timestamp")
    operations = (
        (
            await db.execute(
                select(ExternalOperation).where(
                    ExternalOperation.account_id == account_id,
                    or_(
                        ExternalOperation.status.in_(_ACTIVE_STATUSES),
                        ExternalOperation.result_descriptor.is_not(None),
                    ),
                )
            )
        )
        .unique()
        .scalars()
        .all()
    )
    if any(
        operation_blocks_account_mutation(
            item,
            now=now,
            artifact_relay_invalidated_at=artifact_relay_invalidated_at,
        )
        for item in operations
    ):
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=(
                "Account transport configuration cannot be changed while it is "
                "needed by active operations or unexpired artifacts."
            ),
        )


__all__ = [
    "ensure_account_transport_mutation_safe",
    "operation_blocks_account_mutation",
]
