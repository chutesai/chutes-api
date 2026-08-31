import inspect
import sys
from datetime import UTC, datetime, timedelta
from types import ModuleType, SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from fastapi import HTTPException


# These management tests do not exercise the optional substrate client.
_substrate_package = ModuleType("async_substrate_interface")
_substrate_module = ModuleType("async_substrate_interface.async_substrate")
_substrate_sync_module = ModuleType("async_substrate_interface.sync_substrate")
_substrate_module.AsyncSubstrateInterface = object
_substrate_sync_module.SubstrateInterface = object
_substrate_package.AsyncSubstrateInterface = object
sys.modules.setdefault("async_substrate_interface", _substrate_package)
sys.modules.setdefault("async_substrate_interface.async_substrate", _substrate_module)
sys.modules.setdefault(
    "async_substrate_interface.sync_substrate", _substrate_sync_module
)
_ss58_module = ModuleType("scalecodec.utils.ss58")
_ss58_module.is_valid_ss58_address = lambda _value: False
_ss58_module.ss58_decode = lambda _value: ""
sys.modules.setdefault("scalecodec.utils.ss58", _ss58_module)

from api.external_backend import admin_router  # noqa: E402
from api.external_backend.schemas import (  # noqa: E402
    ExternalAccountBulkCancelRequest,
    ExternalChuteBindingUpdate,
    ExternalCredentialForceRotateRequest,
    ExternalSettlementRetryRequest,
    ExternalSettlementStatus,
    ExternalSettlementWriteOffRequest,
)
from api.permissions import Permissioning  # noqa: E402
from api.user.schemas import User  # noqa: E402


@pytest.mark.asyncio
async def test_external_provisioner_rejects_non_system_billing_admin(monkeypatch):
    billing_admin = User(
        user_id="billing-admin-id",
        username="billing-admin",
        permissions_bitmask=Permissioning.billing_admin.bitmask,
    )
    assert billing_admin.has_role(Permissioning.billing_admin)
    monkeypatch.setattr(
        admin_router,
        "chutes_user_id",
        AsyncMock(return_value="chutes-system-id"),
    )

    with pytest.raises(HTTPException) as exc_info:
        await admin_router.require_external_provisioner(billing_admin)

    assert exc_info.value.status_code == 403
    assert exc_info.value.detail == (
        "Only the Chutes system account may provision external backends."
    )


@pytest.mark.asyncio
async def test_external_provisioner_accepts_exact_system_user_id(monkeypatch):
    system_user = User(
        user_id="chutes-system-id",
        username="chutes",
        permissions_bitmask=0,
    )
    monkeypatch.setattr(
        admin_router,
        "chutes_user_id",
        AsyncMock(return_value="chutes-system-id"),
    )

    assert await admin_router.require_external_provisioner(system_user) is system_user


def test_only_provisioning_posts_use_system_user_dependency():
    for endpoint in (
        admin_router.create_account,
        admin_router.create_binding,
        admin_router.create_external_chute,
    ):
        dependency = inspect.signature(endpoint).parameters["current_user"].default
        assert dependency.dependency is admin_router.require_external_provisioner

    for endpoint in (
        admin_router.retry_quarantined_settlement,
        admin_router.write_off_quarantined_settlement,
    ):
        dependency = inspect.signature(endpoint).parameters["current_user"].default
        assert dependency.dependency is admin_router._require_management_role


@pytest.mark.asyncio
async def test_update_binding_locks_binding_then_account_then_chute(monkeypatch):
    calls = []
    binding = SimpleNamespace(
        binding_id="binding-id",
        chute_id="chute-id",
        account_id="account-id",
        routes=[],
        enabled=False,
        updated_at=None,
    )
    account = SimpleNamespace(account_id="account-id", enabled=True)
    chute = SimpleNamespace(
        chute_id="chute-id",
        name="external-chute",
        cords=[],
        disabled=True,
    )

    async def binding_for_user(_db, _user_id, _binding_id, *, for_update=False):
        calls.append(("binding", for_update))
        return binding

    async def account_for_user(
        _db, _user_id, _account_id, *, for_update=False, for_share=False
    ):
        calls.append(("account", for_share))
        return account

    async def chute_for_user(_db, _user_id, _chute_id, *, for_update=False):
        calls.append(("chute", for_update))
        return chute

    monkeypatch.setattr(admin_router, "_binding_for_user", binding_for_user)
    monkeypatch.setattr(admin_router, "_account_for_user", account_for_user)
    monkeypatch.setattr(admin_router, "_external_chute_for_user", chute_for_user)
    monkeypatch.setattr(admin_router, "_commit_or_conflict", AsyncMock())
    monkeypatch.setattr(admin_router, "_invalidate_chute", AsyncMock())
    db = SimpleNamespace(refresh=AsyncMock())

    result = await admin_router.update_binding(
        "binding-id",
        ExternalChuteBindingUpdate(enabled=True),
        db=db,
        current_user=SimpleNamespace(user_id="user-id"),
    )

    assert result is binding
    assert calls == [("binding", True), ("account", True), ("chute", True)]
    assert binding.enabled is True
    assert chute.disabled is False


@pytest.mark.asyncio
async def test_lock_helpers_target_only_binding_and_chute_rows():
    binding = SimpleNamespace(binding_id="binding-id")
    chute = SimpleNamespace(chute_id="chute-id")
    statements = []

    class Result:
        def __init__(self, value):
            self.value = value

        def unique(self):
            return self

        def scalar_one_or_none(self):
            return self.value

    db = SimpleNamespace(
        execute=AsyncMock(
            side_effect=lambda statement: (
                statements.append(statement)
                or Result(binding if len(statements) == 1 else chute)
            )
        )
    )

    await admin_router._binding_for_user(db, "user-id", "binding-id", for_update=True)
    await admin_router._external_chute_for_user(
        db, "user-id", "chute-id", for_update=True
    )

    assert [table.name for table in statements[0]._for_update_arg.of] == [
        "external_chute_bindings"
    ]
    assert [table.name for table in statements[1]._for_update_arg.of] == ["chutes"]
    assert statements[1].get_execution_options()["populate_existing"] is True


def _quarantined_operation():
    return SimpleNamespace(
        operation_id="operation-id",
        status="failed",
        settlement_status=ExternalSettlementStatus.QUARANTINED.value,
        settlement_metadata={
            "billable": True,
            "settlement_attempts": 8,
            "pricing": {"source": "rules"},
        },
        usage={"requests": "1", "tokens": {"output": "bad"}},
        next_poll_at=None,
        settled_at=None,
        updated_at=None,
    )


def _legacy_pricing_snapshot(per_request):
    return {
        "source": "legacy",
        "legacy": {
            "per_request": per_request,
            "per_million_in": None,
            "per_million_out": None,
            "per_step": None,
            "cache_discount": None,
        },
        "context": {
            "cord": "/generate",
            "path": "/chutes/chute-id/generate",
            "method": "POST",
            "dimensions": {},
            "at": "2030-01-01T00:00:00+00:00",
        },
        "accepted_at": "2030-01-01T00:00:00+00:00",
        "billing_chute_id": "chute-id",
        "free_invocation": False,
        "balance_exempt": False,
        "invoice_billing": False,
        "increment_invocation_quota": False,
    }


def _task_route_snapshot(*, cancellable: bool) -> dict:
    operation_config = {}
    if cancellable:
        operation_config["cancel"] = {
            "path_template": "/tasks/{task_id}/cancel",
            "method": "post",
        }
    return {
        "cord_path": "/generate",
        "upstream_resource_id": "video-model",
        "operation_mode": "task",
        "protocol": "generic-json",
        "path_template": "/tasks",
        "method": "post",
        "request_config": {"body_mode": "json"},
        "response_config": {"codec": "json"},
        "operation_config": operation_config,
        "capabilities": {"mode": "task"},
    }


@pytest.mark.asyncio
async def test_billing_admin_retry_can_correct_usage_with_audited_hashes(monkeypatch):
    operation = _quarantined_operation()
    db = SimpleNamespace(refresh=AsyncMock())
    monkeypatch.setattr(
        admin_router,
        "_lock_quarantined_settlement",
        AsyncMock(return_value=operation),
    )
    monkeypatch.setattr(admin_router, "_commit_or_conflict", AsyncMock())

    result = await admin_router.retry_quarantined_settlement(
        "operation-id",
        ExternalSettlementRetryRequest(
            reason="provider usage was malformed; corrected from invoice",
            usage={"requests": 1, "tokens": {"output": 42}},
        ),
        db=db,
        current_user=SimpleNamespace(user_id="billing-admin-id"),
    )

    assert result is operation
    assert operation.settlement_status == ExternalSettlementStatus.FAILED.value
    assert operation.next_poll_at is not None
    assert operation.usage["tokens"] == {"output": "42.0"}
    assert operation.settlement_metadata["settlement_attempts"] == 0
    action = operation.settlement_metadata["operator_actions"][-1]
    assert action["action"] == "retry"
    assert action["actor_user_id"] == "billing-admin-id"
    assert (
        action["usage_correction"]["before_sha256"]
        != (action["usage_correction"]["after_sha256"])
    )


@pytest.mark.asyncio
async def test_billing_admin_can_replace_broken_pricing_with_hash_and_charge_cap(
    monkeypatch,
):
    operation = _quarantined_operation()
    previous = _legacy_pricing_snapshot(None)
    corrected_terms = {
        "source": "legacy",
        "legacy": {
            **previous["legacy"],
            "per_request": "2.50",
        },
    }
    corrected = {
        **previous,
        **corrected_terms,
    }
    operation.usage = {"requests": "1"}
    operation.settlement_metadata["pricing"] = previous
    db = SimpleNamespace(refresh=AsyncMock())
    monkeypatch.setattr(
        admin_router,
        "_lock_quarantined_settlement",
        AsyncMock(return_value=operation),
    )
    monkeypatch.setattr(admin_router, "_commit_or_conflict", AsyncMock())

    await admin_router.retry_quarantined_settlement(
        "operation-id",
        ExternalSettlementRetryRequest(
            reason="corrected from the accepted customer rate card",
            pricing_snapshot=corrected_terms,
            expected_pricing_sha256=admin_router._usage_sha256(previous),
            customer_authorized_max_amount="2.50",
        ),
        db=db,
        current_user=SimpleNamespace(user_id="billing-admin-id"),
    )

    assert operation.settlement_metadata["pricing"] == corrected
    assert operation.settlement_metadata[
        "settlement_pricing_correction_max_amount"
    ] == ("2.50")
    correction = operation.settlement_metadata["operator_actions"][-1][
        "pricing_correction"
    ]
    assert correction["before_sha256"] == admin_router._usage_sha256(previous)
    assert correction["after_sha256"] == admin_router._usage_sha256(corrected)
    assert correction["calculated_amount"] == "2.50"


@pytest.mark.asyncio
async def test_billing_admin_pricing_correction_requires_runtime_charge_cap(
    monkeypatch,
):
    operation = _quarantined_operation()
    previous = _legacy_pricing_snapshot(None)
    operation.usage = {"requests": "1"}
    operation.settlement_metadata["pricing"] = previous
    monkeypatch.setattr(
        admin_router,
        "_lock_quarantined_settlement",
        AsyncMock(return_value=operation),
    )

    args = ExternalSettlementRetryRequest.model_construct(
        reason="invalid caller bypassed request-model validation",
        usage=None,
        pricing_snapshot={"source": "legacy", "legacy": {"per_request": "2.50"}},
        expected_pricing_sha256=admin_router._usage_sha256(previous),
        customer_authorized_max_amount=None,
    )
    with pytest.raises(admin_router.HTTPException) as exc_info:
        await admin_router.retry_quarantined_settlement(
            "operation-id",
            args,
            db=SimpleNamespace(),
            current_user=SimpleNamespace(user_id="billing-admin-id"),
        )

    assert exc_info.value.status_code == 422
    assert "customer-authorized maximum" in exc_info.value.detail


@pytest.mark.asyncio
async def test_billing_admin_pricing_correction_cannot_change_acceptance_context(
    monkeypatch,
):
    operation = _quarantined_operation()
    previous = _legacy_pricing_snapshot(None)
    corrected = _legacy_pricing_snapshot("2.50")
    corrected["billing_chute_id"] = "other-chute-id"
    operation.usage = {"requests": "1"}
    operation.settlement_metadata["pricing"] = previous
    monkeypatch.setattr(
        admin_router,
        "_lock_quarantined_settlement",
        AsyncMock(return_value=operation),
    )

    with pytest.raises(admin_router.HTTPException) as exc_info:
        await admin_router.retry_quarantined_settlement(
            "operation-id",
            ExternalSettlementRetryRequest(
                reason="attempted identity change",
                pricing_snapshot=corrected,
                expected_pricing_sha256=admin_router._usage_sha256(previous),
                customer_authorized_max_amount="2.50",
            ),
            db=SimpleNamespace(),
            current_user=SimpleNamespace(user_id="billing-admin-id"),
        )

    assert exc_info.value.status_code == 422


@pytest.mark.asyncio
async def test_billing_admin_write_off_is_explicit_and_audited(monkeypatch):
    operation = _quarantined_operation()
    db = SimpleNamespace(refresh=AsyncMock())
    monkeypatch.setattr(
        admin_router,
        "_lock_quarantined_settlement",
        AsyncMock(return_value=operation),
    )
    monkeypatch.setattr(admin_router, "_commit_or_conflict", AsyncMock())

    result = await admin_router.write_off_quarantined_settlement(
        "operation-id",
        ExternalSettlementWriteOffRequest(reason="documented provider credit"),
        db=db,
        current_user=SimpleNamespace(user_id="billing-admin-id"),
    )

    assert result is operation
    assert operation.settlement_status == ExternalSettlementStatus.NOT_BILLABLE.value
    assert operation.settled_at is not None
    assert operation.settlement_metadata["original_billable"] is True
    assert operation.settlement_metadata["billable"] is False
    action = operation.settlement_metadata["operator_actions"][-1]
    assert action["action"] == "write_off"
    assert action["reason"] == "documented provider credit"


@pytest.mark.asyncio
async def test_operator_resolution_rejects_an_immutable_outbox(monkeypatch):
    operation = _quarantined_operation()
    db = SimpleNamespace(
        execute=AsyncMock(return_value=SimpleNamespace()),
        get=AsyncMock(return_value=operation),
    )
    monkeypatch.setattr(
        admin_router,
        "external_usage_event_exists",
        AsyncMock(return_value=True),
    )

    with pytest.raises(admin_router.HTTPException) as exc_info:
        await admin_router._lock_quarantined_settlement(db, "operation-id")

    assert exc_info.value.status_code == 409
    assert "immutable usage charge" in exc_info.value.detail


@pytest.mark.asyncio
async def test_bulk_cancel_defers_pending_wakes_tasks_and_flags_local_sessions(
    monkeypatch,
):
    now = datetime(2030, 1, 1, tzinfo=UTC)
    recovery_deadline = now + timedelta(minutes=5)
    pending = SimpleNamespace(
        status="pending",
        operation_mode="task",
        route_snapshot=_task_route_snapshot(cancellable=True),
        settlement_metadata={},
        next_poll_at=recovery_deadline,
        updated_at=None,
    )
    task = SimpleNamespace(
        status="running",
        operation_mode="task",
        route_snapshot=_task_route_snapshot(cancellable=True),
        settlement_metadata={},
        next_poll_at=now + timedelta(minutes=1),
        updated_at=None,
    )
    stream = SimpleNamespace(
        status="running",
        operation_mode="stream",
        settlement_metadata={"operator_actions": [{} for _ in range(32)]},
        next_poll_at=recovery_deadline,
        updated_at=None,
    )
    non_cancellable_task = SimpleNamespace(
        status="pending",
        operation_mode="task",
        route_snapshot=_task_route_snapshot(cancellable=False),
        settlement_metadata={},
        next_poll_at=recovery_deadline,
        updated_at=None,
    )
    sync = SimpleNamespace(
        status="running",
        operation_mode="sync",
        route_snapshot={},
        settlement_metadata={},
        next_poll_at=recovery_deadline,
        updated_at=None,
    )
    monkeypatch.setattr(
        admin_router,
        "_active_operations_for_account",
        AsyncMock(return_value=[pending, task, stream, non_cancellable_task, sync]),
    )
    monkeypatch.setattr(
        admin_router,
        "_lock_account_operation_governance_scopes",
        AsyncMock(),
    )

    counts = await admin_router._request_account_cancellation(
        SimpleNamespace(),
        account_id="account-id",
        action={"action_id": "action-id", "action": "bulk_cancel"},
        now=now,
    )

    assert counts == {
        "cancel_requested": 3,
        "pending_deferred": 1,
        "task_woken": 1,
        "local_sessions": 1,
        "not_cancellable": 2,
    }
    assert pending.next_poll_at is recovery_deadline
    assert task.next_poll_at is now
    assert stream.next_poll_at is recovery_deadline
    assert all(
        operation.settlement_metadata["cancel_requested"] is True
        for operation in (pending, task, stream)
    )
    assert len(stream.settlement_metadata["operator_actions"]) == 32
    assert stream.settlement_metadata["operator_actions"][-1]["action_id"] == (
        "action-id"
    )
    assert non_cancellable_task.settlement_metadata == {}
    assert non_cancellable_task.next_poll_at is recovery_deadline
    assert sync.settlement_metadata == {}


@pytest.mark.asyncio
async def test_bulk_governance_scopes_lock_sorted_users_before_account():
    statements: list[tuple[str, dict | None]] = []

    class Rows:
        def all(self):
            return []

    class Session:
        async def execute(self, statement, parameters=None):
            statements.append((str(statement), parameters))
            return Rows()

    await admin_router._lock_account_operation_governance_scopes(
        Session(),
        "account-id",
        [
            SimpleNamespace(user_id="user-b"),
            SimpleNamespace(user_id="user-a"),
            SimpleNamespace(user_id="user-b"),
        ],
    )

    lock_parameters = [
        parameters
        for sql, parameters in statements
        if "FOR UPDATE" in sql and parameters is not None
    ]
    assert lock_parameters == [
        {"scope_type": "user", "scope_id": "user-a"},
        {"scope_type": "user", "scope_id": "user-b"},
        {"scope_type": "account", "scope_id": "account-id"},
    ]


@pytest.mark.asyncio
async def test_bulk_cancel_locks_operations_before_governance_scopes(monkeypatch):
    call_order = []
    operations = [SimpleNamespace(user_id="user-a")]

    async def active(_db, _account_id):
        call_order.append("operations")
        return operations

    async def scopes(_db, _account_id, locked_operations):
        call_order.append("governance")
        assert locked_operations is operations

    monkeypatch.setattr(admin_router, "_active_operations_for_account", active)
    monkeypatch.setattr(
        admin_router, "_lock_account_operation_governance_scopes", scopes
    )
    monkeypatch.setattr(
        admin_router, "_operation_supports_emergency_cancel", lambda _operation: False
    )

    await admin_router._request_account_cancellation(
        SimpleNamespace(),
        account_id="account-id",
        action={"action_id": "action-id", "action": "bulk_cancel"},
        now=datetime(2030, 1, 1, tzinfo=UTC),
    )

    assert call_order == ["operations", "governance"]


@pytest.mark.asyncio
async def test_account_bulk_cancel_persists_account_audit(monkeypatch):
    now = datetime(2030, 1, 1, tzinfo=UTC)
    account = SimpleNamespace(
        account_id="account-id", management_metadata={}, updated_at=None
    )
    monkeypatch.setattr(
        admin_router, "_account_for_user", AsyncMock(return_value=account)
    )
    monkeypatch.setattr(admin_router, "_database_now", AsyncMock(return_value=now))
    cancel = AsyncMock(
        return_value={
            "cancel_requested": 2,
            "pending_deferred": 1,
            "task_woken": 1,
            "local_sessions": 0,
            "not_cancellable": 0,
        }
    )
    monkeypatch.setattr(admin_router, "_request_account_cancellation", cancel)
    monkeypatch.setattr(admin_router, "_commit_or_conflict", AsyncMock())

    result = await admin_router.cancel_account_operations(
        "account-id",
        ExternalAccountBulkCancelRequest(reason="provider incident"),
        db=SimpleNamespace(),
        current_user=SimpleNamespace(user_id="billing-admin-id"),
    )

    assert result["cancel_requested"] == 2
    action = account.management_metadata["operator_actions"][-1]
    assert action["action"] == "bulk_cancel"
    assert action["actor_user_id"] == "billing-admin-id"
    assert action["reason"] == "provider incident"


@pytest.mark.asyncio
async def test_force_rotation_disables_admission_and_invalidates_relays(monkeypatch):
    now = datetime(2030, 1, 1, tzinfo=UTC)
    account = SimpleNamespace(
        account_id="account-id",
        credential_references={"primary": "secret://secret-id"},
        management_metadata={},
        enabled=True,
        artifact_relay_invalidated_at=None,
        updated_at=None,
    )
    chute = SimpleNamespace(
        chute_id="chute-id", name="external-model", slug="model-slug", disabled=False
    )
    binding = SimpleNamespace(chute_id="chute-id", chute=chute)
    monkeypatch.setattr(
        admin_router, "_account_for_user", AsyncMock(return_value=account)
    )
    monkeypatch.setattr(admin_router, "_database_now", AsyncMock(return_value=now))
    monkeypatch.setattr(
        admin_router,
        "_request_account_cancellation",
        AsyncMock(
            return_value={
                "cancel_requested": 1,
                "pending_deferred": 0,
                "task_woken": 1,
                "local_sessions": 0,
                "not_cancellable": 0,
            }
        ),
    )
    store = AsyncMock(return_value=dict(account.credential_references))
    monkeypatch.setattr(admin_router, "_store_credentials", store)
    monkeypatch.setattr(
        admin_router, "_bindings_for_account", AsyncMock(return_value=[binding])
    )
    monkeypatch.setattr(admin_router, "_commit_or_conflict", AsyncMock())
    invalidate = AsyncMock()
    monkeypatch.setattr(admin_router, "_invalidate_chute", invalidate)

    result = await admin_router.force_rotate_account_credentials(
        "account-id",
        ExternalCredentialForceRotateRequest(
            reason="credential exposure",
            credentials={"primary": "replacement-value"},
        ),
        db=SimpleNamespace(),
        current_user=SimpleNamespace(user_id="billing-admin-id"),
    )

    assert result["account_disabled"] is True
    assert result["artifact_relays_invalidated_at"] is now
    assert account.enabled is False
    assert account.artifact_relay_invalidated_at is now
    assert chute.disabled is True
    invalidate.assert_awaited_once_with("chute-id", "external-model", "model-slug")
    action = account.management_metadata["operator_actions"][-1]
    assert action["action"] == "force_credential_rotation"
    assert action["rotated_credential_count"] == 1
    assert "replacement-value" not in repr(action)
    assert store.await_args.kwargs["credentials"]["primary"].get_secret_value() == (
        "replacement-value"
    )


@pytest.mark.asyncio
async def test_force_rotation_rejects_new_credential_names(monkeypatch):
    account = SimpleNamespace(
        account_id="account-id",
        credential_references={"primary": "secret://secret-id"},
    )
    monkeypatch.setattr(
        admin_router, "_account_for_user", AsyncMock(return_value=account)
    )

    with pytest.raises(admin_router.HTTPException) as exc_info:
        await admin_router.force_rotate_account_credentials(
            "account-id",
            ExternalCredentialForceRotateRequest(
                reason="credential exposure",
                credentials={"new-name": "replacement-value"},
            ),
            db=SimpleNamespace(),
            current_user=SimpleNamespace(user_id="billing-admin-id"),
        )

    assert exc_info.value.status_code == 422
    assert "existing credential" in exc_info.value.detail
