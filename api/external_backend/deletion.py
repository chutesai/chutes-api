"""Deletion safeguards for externally executed Chutes."""

from fastapi import HTTPException, status
from sqlalchemy import select, text
from sqlalchemy.ext.asyncio import AsyncSession

from api.external_backend.schemas import ExternalOperation, ExternalOperationStatus


ACTIVE_EXTERNAL_OPERATION_STATUSES = (
    ExternalOperationStatus.PENDING.value,
    ExternalOperationStatus.SUBMITTED.value,
    ExternalOperationStatus.RUNNING.value,
)


async def ensure_no_active_external_operations(
    db: AsyncSession,
    chute_id: str,
) -> None:
    """Lock a Chute and reject deletion while any upstream work is still active."""

    # Lock the binding before its parent Chute, matching invocation acceptance.
    # The eventual Chute lock conflicts with operation FK inserts, so no operation
    # can race between the active-work check and deletion.
    await db.execute(
        text(
            "SELECT binding_id FROM external_chute_bindings "
            "WHERE chute_id = :chute_id FOR UPDATE"
        ),
        {"chute_id": chute_id},
    )
    await db.execute(
        text("SELECT chute_id FROM chutes WHERE chute_id = :chute_id FOR UPDATE"),
        {"chute_id": chute_id},
    )
    active_operation_id = (
        await db.execute(
            select(ExternalOperation.operation_id)
            .where(
                ExternalOperation.chute_id == chute_id,
                ExternalOperation.status.in_(ACTIVE_EXTERNAL_OPERATION_STATUSES),
            )
            .limit(1)
        )
    ).scalar_one_or_none()
    if active_operation_id is not None:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=(
                "This Chute cannot be deleted while an external operation is "
                "pending, submitted, or running."
            ),
        )


__all__ = [
    "ACTIVE_EXTERNAL_OPERATION_STATUSES",
    "ensure_no_active_external_operations",
]
