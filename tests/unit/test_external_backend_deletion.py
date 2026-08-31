from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import HTTPException

from api.external_backend.deletion import (
    ACTIVE_EXTERNAL_OPERATION_STATUSES,
    ensure_no_active_external_operations,
)
from api.external_backend.schemas import ExternalOperationStatus


def _result(value):
    result = MagicMock()
    result.scalar_one_or_none.return_value = value
    return result


def test_active_operation_statuses_only_include_live_work():
    assert ACTIVE_EXTERNAL_OPERATION_STATUSES == (
        ExternalOperationStatus.PENDING.value,
        ExternalOperationStatus.SUBMITTED.value,
        ExternalOperationStatus.RUNNING.value,
    )


@pytest.mark.asyncio
async def test_deletion_guard_rejects_active_operation():
    db = AsyncMock()
    db.execute = AsyncMock(
        side_effect=[MagicMock(), MagicMock(), _result("operation-id")]
    )

    with pytest.raises(HTTPException) as exc_info:
        await ensure_no_active_external_operations(db, "chute-id")

    assert exc_info.value.status_code == 409
    assert "pending, submitted, or running" in exc_info.value.detail
    assert db.execute.await_count == 3


@pytest.mark.asyncio
async def test_deletion_guard_allows_no_active_operation():
    db = AsyncMock()
    db.execute = AsyncMock(side_effect=[MagicMock(), MagicMock(), _result(None)])

    await ensure_no_active_external_operations(db, "chute-id")

    assert db.execute.await_count == 3
