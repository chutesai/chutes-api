from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import HTTPException

from api.external_backend.account_safety import (
    ensure_account_transport_mutation_safe,
    operation_blocks_account_mutation,
)


NOW = datetime(2030, 1, 1, tzinfo=timezone.utc)


def _operation(**overrides):
    values = {
        "operation_mode": "sync",
        "status": "succeeded",
        "result_descriptor": None,
        "expires_at": None,
        "created_at": NOW - timedelta(days=1),
    }
    values.update(overrides)
    return SimpleNamespace(**values)


@pytest.mark.parametrize("mode", ["sync", "stream", "task", "realtime"])
@pytest.mark.parametrize("status", ["pending", "submitted", "running"])
def test_live_operations_block_structural_account_changes(mode, status):
    assert operation_blocks_account_mutation(
        _operation(operation_mode=mode, status=status), now=NOW
    )


def test_terminal_task_without_artifacts_does_not_block_account_changes():
    assert not operation_blocks_account_mutation(
        _operation(operation_mode="task", status="succeeded"), now=NOW
    )


def test_unexpired_artifact_uses_artifact_expiration_then_operation_fallback():
    assert operation_blocks_account_mutation(
        _operation(
            result_descriptor={"artifacts": [{"reference": "https://asset.test/a"}]},
            expires_at=NOW + timedelta(minutes=5),
        ),
        now=NOW,
    )
    assert not operation_blocks_account_mutation(
        _operation(
            result_descriptor={
                "artifacts": [
                    {
                        "reference": "https://asset.test/a",
                        "expires_at": (NOW - timedelta(seconds=1)).isoformat(),
                    }
                ]
            },
            expires_at=NOW + timedelta(minutes=5),
        ),
        now=NOW,
    )


def test_artifact_without_expiration_blocks_because_relay_remains_available():
    assert operation_blocks_account_mutation(
        _operation(
            result_descriptor={"artifacts": [{"reference": "https://asset.test/a"}]}
        ),
        now=NOW,
    )


def test_force_rotation_cutoff_releases_only_older_terminal_artifacts():
    operation = _operation(
        result_descriptor={"artifacts": [{"reference": "https://asset.test/a"}]},
        created_at=NOW - timedelta(seconds=1),
    )
    assert not operation_blocks_account_mutation(
        operation, now=NOW, artifact_relay_invalidated_at=NOW
    )
    operation.created_at = NOW + timedelta(seconds=1)
    assert operation_blocks_account_mutation(
        operation, now=NOW, artifact_relay_invalidated_at=NOW
    )

    operation.status = "running"
    operation.created_at = NOW - timedelta(seconds=1)
    assert operation_blocks_account_mutation(
        operation, now=NOW, artifact_relay_invalidated_at=NOW
    )


def _result(operations):
    result = MagicMock()
    result.unique.return_value.scalars.return_value.all.return_value = operations
    return result


def _scalar(value):
    result = MagicMock()
    result.scalar_one.return_value = value
    return result


@pytest.mark.asyncio
async def test_account_guard_locks_row_and_rejects_blocking_operation():
    db = AsyncMock()
    db.execute = AsyncMock(
        side_effect=[
            MagicMock(),
            _scalar(NOW),
            _result([_operation(operation_mode="task", status="running")]),
        ]
    )

    with pytest.raises(HTTPException) as exc_info:
        await ensure_account_transport_mutation_safe(db, "account-id")

    assert exc_info.value.status_code == 409
    assert db.execute.await_count == 3


@pytest.mark.asyncio
async def test_account_guard_allows_rotation_when_no_operation_needs_transport():
    db = AsyncMock()
    db.execute = AsyncMock(side_effect=[MagicMock(), _scalar(NOW), _result([])])

    await ensure_account_transport_mutation_safe(db, "account-id")

    assert db.execute.await_count == 3
