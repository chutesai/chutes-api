"""User-owned management endpoints for externally executed Chutes."""

from __future__ import annotations

import hashlib
import re
import uuid
from datetime import datetime, timezone
from typing import Any, Sequence

import orjson
from fastapi import APIRouter, Depends, HTTPException, Response, status
from loguru import logger
from pydantic import SecretStr
from slugify import slugify
from sqlalchemy import delete, exists, func, select, text
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import noload, selectinload

from api.chute.schemas import Chute, Cord
from api.database import get_db_session
from api.external_backend.account_safety import ensure_account_transport_mutation_safe
from api.external_backend.deletion import ensure_no_active_external_operations
from api.external_backend.governance import lock_governance_state_rows
from api.external_backend.schemas import (
    ExternalAccountBulkCancelRequest,
    ExternalAccountBulkCancelResponse,
    ExternalAuthHeaderTemplate,
    ExternalBackendAccount,
    ExternalBackendAccountCreate,
    ExternalBackendAccountResponse,
    ExternalBackendAccountUpdate,
    ExternalChuteBinding,
    ExternalChuteBindingCreate,
    ExternalChuteBindingResponse,
    ExternalChuteBindingUpdate,
    ExternalChuteCreate,
    ExternalChuteResponse,
    ExternalChuteUpdate,
    ExternalCord,
    ExternalCredentialForceRotateRequest,
    ExternalCredentialForceRotateResponse,
    ExternalOperation,
    ExternalOperationMode,
    ExternalOperationResponse,
    ExternalOperationStatus,
    ExternalRouteConfig,
    ExternalSettlementRetryRequest,
    ExternalSettlementStatus,
    ExternalSettlementWriteOffRequest,
)
from api.external_backend.model_compat import external_chute_version
from api.external_backend.validation import (
    RouteConfigurationError,
    validate_route_configuration,
)
from api.payment.pricing import (
    NormalizedUsage,
    PricingConfigurationError,
    parse_pricing_rules,
)
from api.payment.util import encrypt_secret
from api.permissions import Permissioning
from api.secret.schemas import Secret
from api.user.schemas import PriceOverride, User
from api.user.service import chutes_user_id, get_current_user, require_role

from .billing_outbox import external_usage_event_exists


router = APIRouter()
_require_management_role = require_role(Permissioning.billing_admin)


async def require_external_provisioner(
    current_user: User = Depends(get_current_user()),
) -> User:
    """Allow external-provider provisioning only for the canonical system user."""

    system_user_id = await chutes_user_id()
    if system_user_id is None or current_user.user_id != system_user_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Only the Chutes system account may provision external backends.",
        )
    return current_user


_TERMINAL_OPERATION_STATUSES = frozenset(
    {
        ExternalOperationStatus.SUCCEEDED.value,
        ExternalOperationStatus.FAILED.value,
        ExternalOperationStatus.CANCELLED.value,
        ExternalOperationStatus.EXPIRED.value,
    }
)
_MAX_OPERATOR_ACTIONS = 32
_ACTIVE_OPERATION_STATUSES = frozenset(
    {
        ExternalOperationStatus.PENDING.value,
        ExternalOperationStatus.SUBMITTED.value,
        ExternalOperationStatus.RUNNING.value,
    }
)
_IMMUTABLE_PRICING_SNAPSHOT_FIELDS = frozenset(
    {
        "accepted_at",
        "balance_exempt",
        "billing_chute_id",
        "context",
        "free_invocation",
        "increment_invocation_quota",
        "invoice_billing",
    }
)


def _secret_id(user_id: str, account_id: str, name: str) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_OID, f"{user_id}:{account_id}:{name}"))


def _secret_reference(secret_id: str) -> str:
    return f"secret://{secret_id}"


def _dump_templates(
    templates: list[ExternalAuthHeaderTemplate],
) -> list[dict[str, Any]]:
    return [template.model_dump(mode="json") for template in templates]


def _dump_routes(routes: list[ExternalRouteConfig]) -> list[dict[str, Any]]:
    return [route.model_dump(mode="json") for route in routes]


def _validate_template_references(
    templates: list[ExternalAuthHeaderTemplate], credential_names: set[str]
) -> None:
    referenced = {
        reference
        for template in templates
        for reference in template.references.values()
    }
    if not credential_names or not referenced.issubset(credential_names):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Auth header templates must reference configured credential names.",
        )


def _validate_binding_routes(chute: Chute, routes: list[ExternalRouteConfig]) -> None:
    cords = [
        cord if isinstance(cord, Cord) else Cord.model_validate(cord)
        for cord in chute.cords
    ]
    cord_paths = [cord.path for cord in cords]
    route_paths = [route.cord_path for route in routes]
    if (
        len(cord_paths) != len(set(cord_paths))
        or len(route_paths) != len(set(route_paths))
        or set(cord_paths) != set(route_paths)
    ):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Routes must match the Chute cord paths.",
        )
    selectors = [
        (cord.public_api_path, cord.public_api_method.upper(), bool(cord.stream))
        for cord in cords
    ]
    if len(selectors) != len(set(selectors)):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Public Cord path, method, and stream selectors must be unique.",
        )


def _validate_route_configurations(
    account: ExternalBackendAccount,
    routes: list[ExternalRouteConfig] | list[dict[str, Any]],
    cords: list[ExternalCord] | list[Cord] | list[dict[str, Any]],
    pricing_rules: list[dict[str, Any]],
) -> None:
    cord_by_path = {
        (cord.get("path") if isinstance(cord, dict) else cord.path): cord
        for cord in cords
    }
    for raw_route in routes:
        route = (
            raw_route
            if isinstance(raw_route, ExternalRouteConfig)
            else ExternalRouteConfig.model_validate(raw_route)
        )
        try:
            validate_route_configuration(
                account,
                route,
                cord=cord_by_path.get(route.cord_path),
                pricing_rules=pricing_rules,
            )
        except RouteConfigurationError as exc:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=f"Invalid route configuration for {route.cord_path}: {exc}",
            ) from exc


def _version_for_external_chute(
    *,
    account_id: str,
    standard_template: str | None,
    cords: list[ExternalCord] | list[Cord] | list[dict[str, Any]],
    routes: list[ExternalRouteConfig] | list[dict[str, Any]],
    pricing_rules: list[dict[str, Any]],
) -> str:
    return external_chute_version(
        account_id=account_id,
        standard_template=standard_template,
        cords=cords,
        routes=routes,
        pricing_rules=pricing_rules,
    )


def _to_persisted_cords(cords: list[ExternalCord]) -> list[Cord]:
    return [Cord.model_validate(cord.model_dump(mode="python")) for cord in cords]


async def _account_for_user(
    db: AsyncSession,
    user_id: str,
    account_id: str,
    *,
    for_update: bool = False,
    for_share: bool = False,
) -> ExternalBackendAccount:
    query = select(ExternalBackendAccount).where(
        ExternalBackendAccount.account_id == account_id,
        ExternalBackendAccount.user_id == user_id,
    )
    if for_update and for_share:
        raise ValueError("account lock mode is ambiguous")
    if for_update or for_share:
        query = query.options(noload(ExternalBackendAccount.user)).with_for_update(
            read=for_share,
            of=ExternalBackendAccount,
        )
    account = (await db.execute(query)).unique().scalar_one_or_none()
    if account is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Account not found."
        )
    return account


async def _binding_for_user(
    db: AsyncSession,
    user_id: str,
    binding_id: str,
    *,
    for_update: bool = False,
) -> ExternalChuteBinding:
    query = (
        select(ExternalChuteBinding)
        .join(
            ExternalBackendAccount,
            ExternalBackendAccount.account_id == ExternalChuteBinding.account_id,
        )
        .where(
            ExternalChuteBinding.binding_id == binding_id,
            ExternalBackendAccount.user_id == user_id,
        )
    )
    if for_update:
        # Invocation acceptance locks the binding before its Chute. Management
        # mutations use the same order so neither side can deadlock while the ORM
        # flushes updates to both rows.
        query = query.with_for_update(of=ExternalChuteBinding)
    binding = (await db.execute(query)).unique().scalar_one_or_none()
    if binding is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Binding not found."
        )
    return binding


async def _external_chute_for_user(
    db: AsyncSession,
    user_id: str,
    chute_id: str,
    *,
    for_update: bool = False,
) -> Chute:
    query = (
        select(Chute)
        .where(
            Chute.chute_id == chute_id,
            Chute.user_id == user_id,
            Chute.execution_backend == "external",
        )
        .options(selectinload(Chute.external_binding))
    )
    if for_update:
        query = query.with_for_update(of=Chute).execution_options(
            populate_existing=True
        )
    chute = (await db.execute(query)).unique().scalar_one_or_none()
    if chute is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="External Chute not found.",
        )
    return chute


async def _price_override(db: AsyncSession, chute_id: str) -> PriceOverride | None:
    return (
        await db.execute(
            select(PriceOverride).where(
                PriceOverride.user_id == "*",
                PriceOverride.chute_id == chute_id,
            )
        )
    ).scalar_one_or_none()


def _chute_response(
    chute: Chute, price_override: PriceOverride | None
) -> ExternalChuteResponse:
    return ExternalChuteResponse.model_validate(
        {
            "chute_id": chute.chute_id,
            "user_id": chute.user_id,
            "name": chute.name,
            "tagline": chute.tagline or "",
            "readme": chute.readme or "",
            "tool_description": chute.tool_description,
            "logo_id": chute.logo_id,
            "public": chute.public,
            "standard_template": chute.standard_template,
            "slug": chute.slug,
            "version": chute.version,
            "execution_backend": chute.execution_backend,
            "cords": chute.cords,
            "binding": chute.external_binding,
            "pricing_rules": list(getattr(price_override, "pricing_rules", None) or []),
            "created_at": chute.created_at,
            "updated_at": chute.updated_at,
        }
    )


async def _store_credentials(
    db: AsyncSession,
    *,
    user_id: str,
    account_id: str,
    credentials: dict[str, SecretStr],
    references: dict[str, str] | None = None,
) -> dict[str, str]:
    updated = dict(references or {})
    for name, secret_value in credentials.items():
        secret_id = _secret_id(user_id, account_id, name)
        encrypted_value = await encrypt_secret(secret_value.get_secret_value())
        secret = await db.get(Secret, secret_id)
        if secret is None:
            db.add(
                Secret(
                    secret_id=secret_id,
                    user_id=user_id,
                    purpose=account_id,
                    kind="external_backend",
                    key=name,
                    value=encrypted_value,
                )
            )
        else:
            if (
                secret.user_id != user_id
                or secret.purpose != account_id
                or secret.kind != "external_backend"
            ):
                raise HTTPException(
                    status_code=status.HTTP_409_CONFLICT,
                    detail="Credential reference conflicts with another resource.",
                )
            secret.key = name
            secret.value = encrypted_value
        updated[name] = _secret_reference(secret_id)
    return updated


async def _commit_or_conflict(db: AsyncSession, detail: str) -> None:
    try:
        await db.commit()
    except IntegrityError as exc:
        await db.rollback()
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT, detail=detail
        ) from exc


def _usage_sha256(value: Any) -> str:
    return hashlib.sha256(orjson.dumps(value, option=orjson.OPT_SORT_KEYS)).hexdigest()


def _append_operator_action(metadata: dict[str, Any], action: dict[str, Any]) -> None:
    existing = metadata.get("operator_actions", [])
    actions = list(existing) if isinstance(existing, list) else []
    actions.append(action)
    metadata["operator_actions"] = actions[-_MAX_OPERATOR_ACTIONS:]


async def _database_now(db: AsyncSession) -> datetime:
    value = (await db.execute(text("SELECT clock_timestamp()"))).scalar_one()
    if not isinstance(value, datetime):
        raise RuntimeError("database clock did not return a timestamp")
    return value.replace(tzinfo=value.tzinfo or timezone.utc)


async def _active_operations_for_account(
    db: AsyncSession, account_id: str
) -> list[ExternalOperation]:
    return (
        (
            await db.execute(
                select(ExternalOperation)
                .options(
                    noload(ExternalOperation.user),
                    noload(ExternalOperation.account),
                    noload(ExternalOperation.binding),
                )
                .where(
                    ExternalOperation.account_id == account_id,
                    ExternalOperation.status.in_(_ACTIVE_OPERATION_STATUSES),
                )
                .order_by(
                    ExternalOperation.user_id,
                    ExternalOperation.operation_id,
                )
                .with_for_update(of=ExternalOperation)
            )
        )
        .unique()
        .scalars()
        .all()
    )


async def _lock_account_operation_governance_scopes(
    db: AsyncSession,
    account_id: str,
    operations: Sequence[ExternalOperation],
) -> None:
    """Pre-lock every bulk-mutation scope in the trigger's global order.

    The caller already holds the backend-account row, which prevents a new
    operation from acquiring its foreign-key key-share lock, and has locked all
    active operation rows before entering here. Governance rows then follow in
    sorted user-before-account order, matching retention, poll recovery, and
    ordinary operation mutation. Flush-time triggers can only reacquire locks
    this transaction already owns.
    """

    await lock_governance_state_rows(
        db,
        user_ids=(operation.user_id for operation in operations),
        account_ids=(account_id,),
    )


def _operation_supports_emergency_cancel(operation: ExternalOperation) -> bool:
    if operation.operation_mode in {
        ExternalOperationMode.STREAM.value,
        ExternalOperationMode.REALTIME.value,
    }:
        return True
    if operation.operation_mode != ExternalOperationMode.TASK.value:
        return False
    try:
        route = ExternalRouteConfig.model_validate(operation.route_snapshot)
        operation_config = dict(route.operation_config or {})
        task_config = operation_config.get("task") or {}
        if not isinstance(task_config, dict):
            return False
        cancel_config = operation_config.get("cancel") or task_config.get("cancel")
        return isinstance(cancel_config, dict)
    except Exception:
        # Emergency management must not turn malformed or non-cancellable work
        # into a billable terminal failure. Leave it on its existing poll path.
        return False


async def _request_account_cancellation(
    db: AsyncSession,
    *,
    account_id: str,
    action: dict[str, Any],
    now: datetime,
    audit_not_cancellable: bool = False,
) -> dict[str, int]:
    operations = await _active_operations_for_account(db, account_id)
    await _lock_account_operation_governance_scopes(db, account_id, operations)
    counts = {
        "cancel_requested": 0,
        "pending_deferred": 0,
        "task_woken": 0,
        "local_sessions": 0,
        "not_cancellable": 0,
    }
    for operation in operations:
        if not _operation_supports_emergency_cancel(operation):
            if audit_not_cancellable:
                metadata = dict(operation.settlement_metadata or {})
                _append_operator_action(metadata, action)
                operation.settlement_metadata = metadata
                operation.updated_at = func.now()
            counts["not_cancellable"] += 1
            continue
        metadata = dict(operation.settlement_metadata or {})
        metadata["cancel_requested"] = True
        metadata["cancel_requested_at"] = now.isoformat()
        _append_operator_action(metadata, action)
        operation.settlement_metadata = metadata
        operation.updated_at = func.now()
        counts["cancel_requested"] += 1
        if operation.status == ExternalOperationStatus.PENDING.value:
            # The submitting process still owns this row. Preserve its recovery
            # deadline so cancellation cannot orphan accepted upstream work.
            counts["pending_deferred"] += 1
        elif operation.operation_mode == ExternalOperationMode.TASK.value:
            operation.next_poll_at = now
            counts["task_woken"] += 1
        elif operation.operation_mode in {
            ExternalOperationMode.STREAM.value,
            ExternalOperationMode.REALTIME.value,
        }:
            # The local usage monitor observes the persisted flag and tears down
            # the live upstream session without relying on a provider cancel API.
            counts["local_sessions"] += 1
    return counts


async def _bindings_for_account(
    db: AsyncSession, account_id: str
) -> list[ExternalChuteBinding]:
    return (
        (
            await db.execute(
                select(ExternalChuteBinding).where(
                    ExternalChuteBinding.account_id == account_id
                )
            )
        )
        .unique()
        .scalars()
        .all()
    )


async def _lock_quarantined_settlement(
    db: AsyncSession, operation_id: str
) -> ExternalOperation:
    # This lock order matches settlement and outbox delivery, so an operator
    # decision cannot cross a late immutable charge or successful settlement.
    await db.execute(
        text("SELECT pg_advisory_xact_lock(hashtextextended(:key, 0))").bindparams(
            key=f"external-settlement:{operation_id}"
        )
    )
    operation = await db.get(ExternalOperation, operation_id, with_for_update=True)
    if operation is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="External operation not found.",
        )
    if (
        operation.status not in _TERMINAL_OPERATION_STATUSES
        or operation.settlement_status != ExternalSettlementStatus.QUARANTINED.value
    ):
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Only a quarantined terminal settlement can be resolved.",
        )
    if await external_usage_event_exists(db, operation_id):
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="An immutable usage charge already exists for this operation.",
        )
    return operation


async def _invalidate_price_override_cache(chute_id: str) -> None:
    # Keep the hosted execution dependency out of management-router import time.
    from api.chute.util import invalidate_price_override_cache

    await invalidate_price_override_cache(chute_id)


async def _invalidate_chute(
    chute_id: str, chute_name: str, chute_slug: str | None = None
) -> None:
    # Keep the hosted execution dependency out of management-router import time.
    from api.chute.util import invalidate_chute_cache

    await invalidate_chute_cache(chute_id, chute_name, chute_slug)


@router.post(
    "/operations/{operation_id}/settlement/retry",
    response_model=ExternalOperationResponse,
)
async def retry_quarantined_settlement(
    operation_id: str,
    args: ExternalSettlementRetryRequest,
    db: AsyncSession = Depends(get_db_session),
    current_user: User = Depends(_require_management_role),
):
    """Release one operator-reviewed settlement back to the durable reconciler."""

    operation = await _lock_quarantined_settlement(db, operation_id)
    now = datetime.now(timezone.utc)
    action: dict[str, Any] = {
        "action_id": str(uuid.uuid4()),
        "action": "retry",
        "actor_user_id": current_user.user_id,
        "at": now.isoformat(),
        "reason": args.reason,
        "previous_settlement_status": operation.settlement_status,
    }
    if args.usage is not None:
        before_usage = operation.usage or {}
        corrected_usage = NormalizedUsage.from_mapping(
            args.usage.model_dump(mode="json")
        ).to_dict()
        action["usage_correction"] = {
            "before_sha256": _usage_sha256(before_usage),
            "after_sha256": _usage_sha256(corrected_usage),
        }
        operation.usage = corrected_usage

    metadata = dict(operation.settlement_metadata or {})
    if args.pricing_snapshot is not None:
        previous_pricing = metadata.get("pricing")
        if not isinstance(previous_pricing, dict):
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="The accepted pricing snapshot is unavailable.",
            )
        previous_hash = _usage_sha256(previous_pricing)
        if args.expected_pricing_sha256 != previous_hash:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="The accepted pricing snapshot changed; review it again.",
            )
        corrected_pricing = dict(args.pricing_snapshot)
        if any(
            field in corrected_pricing
            and corrected_pricing.get(field) != previous_pricing.get(field)
            for field in _IMMUTABLE_PRICING_SNAPSHOT_FIELDS
        ):
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=(
                    "A pricing correction cannot change the accepted request context "
                    "or billing identity."
                ),
            )
        for field in _IMMUTABLE_PRICING_SNAPSHOT_FIELDS:
            if field in previous_pricing:
                corrected_pricing[field] = previous_pricing[field]
            else:
                corrected_pricing.pop(field, None)
        if corrected_pricing.get("source") not in {"legacy", "rules"}:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="The corrected pricing snapshot source is invalid.",
            )
        try:
            from api.external_backend.service import _pricing_result

            corrected_usage = NormalizedUsage.from_mapping(operation.usage or {})
            corrected_result = _pricing_result(corrected_pricing, corrected_usage)
        except (PricingConfigurationError, ValueError, TypeError) as exc:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="The corrected pricing snapshot is invalid.",
            ) from exc
        authorized_maximum = args.customer_authorized_max_amount
        if authorized_maximum is None:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=("A pricing correction requires a customer-authorized maximum."),
            )
        if (
            not corrected_result.applied
            or not corrected_result.complete
            or corrected_result.amount < 0
            or corrected_result.amount > authorized_maximum
        ):
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=(
                    "The corrected pricing must completely price the persisted usage "
                    "within the customer-authorized maximum."
                ),
            )
        corrected_hash = _usage_sha256(corrected_pricing)
        action["pricing_correction"] = {
            "before_sha256": previous_hash,
            "after_sha256": corrected_hash,
            "customer_authorized_max_amount": str(authorized_maximum),
            "calculated_amount": str(corrected_result.amount),
        }
        metadata["pricing"] = corrected_pricing
        metadata["settlement_pricing_correction_max_amount"] = str(authorized_maximum)

    attempts = metadata.get("settlement_retry_generation", 0)
    attempts = attempts if isinstance(attempts, int) and attempts >= 0 else 0
    metadata["settlement_retry_generation"] = attempts + 1
    metadata["settlement_attempts"] = 0
    metadata["settlement_operator_retry_at"] = now.isoformat()
    metadata.pop("settlement_next_attempt_at", None)
    _append_operator_action(metadata, action)
    operation.settlement_metadata = metadata
    operation.settlement_status = ExternalSettlementStatus.FAILED.value
    operation.next_poll_at = now
    operation.settled_at = None
    operation.updated_at = func.now()
    await _commit_or_conflict(db, "Unable to retry this settlement.")
    await db.refresh(operation)
    return operation


@router.post(
    "/operations/{operation_id}/settlement/write-off",
    response_model=ExternalOperationResponse,
)
async def write_off_quarantined_settlement(
    operation_id: str,
    args: ExternalSettlementWriteOffRequest,
    db: AsyncSession = Depends(get_db_session),
    current_user: User = Depends(_require_management_role),
):
    """Resolve a reviewed unpriceable operation without silently losing spend."""

    operation = await _lock_quarantined_settlement(db, operation_id)
    now = datetime.now(timezone.utc)
    action = {
        "action_id": str(uuid.uuid4()),
        "action": "write_off",
        "actor_user_id": current_user.user_id,
        "at": now.isoformat(),
        "reason": args.reason,
        "previous_settlement_status": operation.settlement_status,
    }
    metadata = dict(operation.settlement_metadata or {})
    metadata["original_billable"] = metadata.get("billable")
    metadata["billable"] = False
    metadata["settlement_write_off_at"] = now.isoformat()
    metadata["settlement_write_off_action_id"] = action["action_id"]
    metadata.pop("settlement_next_attempt_at", None)
    _append_operator_action(metadata, action)
    operation.settlement_metadata = metadata
    operation.settlement_status = ExternalSettlementStatus.NOT_BILLABLE.value
    operation.next_poll_at = None
    operation.settled_at = now
    operation.updated_at = func.now()
    await _commit_or_conflict(db, "Unable to write off this settlement.")
    await db.refresh(operation)
    return operation


@router.get("/accounts", response_model=list[ExternalBackendAccountResponse])
async def list_accounts(
    db: AsyncSession = Depends(get_db_session),
    current_user: User = Depends(get_current_user()),
):
    result = await db.execute(
        select(ExternalBackendAccount)
        .where(ExternalBackendAccount.user_id == current_user.user_id)
        .order_by(ExternalBackendAccount.created_at.desc())
    )
    return result.unique().scalars().all()


@router.post(
    "/accounts",
    response_model=ExternalBackendAccountResponse,
    status_code=status.HTTP_201_CREATED,
)
async def create_account(
    args: ExternalBackendAccountCreate,
    db: AsyncSession = Depends(get_db_session),
    current_user: User = Depends(require_external_provisioner),
):
    account_id = str(uuid.uuid4())
    references = await _store_credentials(
        db,
        user_id=current_user.user_id,
        account_id=account_id,
        credentials=args.credentials,
    )
    account = ExternalBackendAccount(
        account_id=account_id,
        user_id=current_user.user_id,
        name=args.name,
        adapter=args.adapter,
        base_url=str(args.base_url),
        credential_references=references,
        auth_header_templates=_dump_templates(args.auth_header_templates),
        connection_config=args.connection_config,
        enabled=args.enabled,
    )
    db.add(account)
    await _commit_or_conflict(db, "An account with this name already exists.")
    await db.refresh(account)
    return account


@router.get("/accounts/{account_id}", response_model=ExternalBackendAccountResponse)
async def get_account(
    account_id: str,
    db: AsyncSession = Depends(get_db_session),
    current_user: User = Depends(get_current_user()),
):
    return await _account_for_user(db, current_user.user_id, account_id)


@router.patch("/accounts/{account_id}", response_model=ExternalBackendAccountResponse)
async def update_account(
    account_id: str,
    args: ExternalBackendAccountUpdate,
    db: AsyncSession = Depends(get_db_session),
    current_user: User = Depends(_require_management_role),
):
    # Lock before deriving credential/template state so concurrent PATCHes cannot
    # overwrite one another after waiting on the transport-mutation guard.
    account = await _account_for_user(
        db, current_user.user_id, account_id, for_update=True
    )
    references = dict(account.credential_references)
    updated_names = set(args.credentials or {})
    removed_names = set(args.remove_credentials)
    missing_names = removed_names - set(references)
    if missing_names:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Unknown credential name(s): {', '.join(sorted(missing_names))}.",
        )
    final_names = (set(references) | updated_names) - removed_names
    templates = (
        args.auth_header_templates
        if args.auth_header_templates is not None
        else [
            ExternalAuthHeaderTemplate.model_validate(value)
            for value in account.auth_header_templates
        ]
    )
    _validate_template_references(templates, final_names)

    # Accepted tasks, realtime sessions, and unexpired artifact relays resolve
    # credentials at execution time. Rotating a value can therefore strand work
    # just as surely as removing or renaming the credential.
    structural_change = bool(removed_names or updated_names)
    if "base_url" in args.model_fields_set:
        structural_change = structural_change or (
            str(args.base_url).rstrip("/") != account.base_url.rstrip("/")
        )
    if "connection_config" in args.model_fields_set:
        structural_change = (
            structural_change or args.connection_config != account.connection_config
        )
    if "auth_header_templates" in args.model_fields_set:
        structural_change = structural_change or _dump_templates(templates) != list(
            account.auth_header_templates
        )
    if structural_change:
        await ensure_account_transport_mutation_safe(db, account.account_id)
    enabled_changed = (
        "enabled" in args.model_fields_set and args.enabled != account.enabled
    )

    if args.credentials:
        references = await _store_credentials(
            db,
            user_id=current_user.user_id,
            account_id=account.account_id,
            credentials=args.credentials,
            references=references,
        )
    if removed_names:
        await db.execute(
            delete(Secret).where(
                Secret.user_id == current_user.user_id,
                Secret.purpose == account.account_id,
                Secret.kind == "external_backend",
                Secret.key.in_(removed_names),
            )
        )
        for name in removed_names:
            references.pop(name, None)

    if "name" in args.model_fields_set:
        account.name = args.name
    if "base_url" in args.model_fields_set:
        account.base_url = str(args.base_url)
    if "connection_config" in args.model_fields_set:
        account.connection_config = args.connection_config
    if "enabled" in args.model_fields_set:
        account.enabled = args.enabled
    account.credential_references = references
    account.auth_header_templates = _dump_templates(templates)
    account.updated_at = func.now()

    bindings: list[ExternalChuteBinding] = []
    if structural_change or enabled_changed:
        bindings = (
            (
                await db.execute(
                    select(ExternalChuteBinding).where(
                        ExternalChuteBinding.account_id == account.account_id
                    )
                )
            )
            .unique()
            .scalars()
            .all()
        )
    if structural_change:
        for binding in bindings:
            price_override = await _price_override(db, binding.chute_id)
            _validate_route_configurations(
                account,
                binding.routes,
                binding.chute.cords,
                list(getattr(price_override, "pricing_rules", None) or []),
            )
    invalidated_chutes: list[tuple[str, str, str | None]] = []
    if enabled_changed:
        for binding in bindings:
            binding.chute.disabled = not (account.enabled and binding.enabled)
            invalidated_chutes.append(
                (
                    binding.chute_id,
                    binding.chute.name,
                    getattr(binding.chute, "slug", None),
                )
            )

    await _commit_or_conflict(db, "Unable to update this account.")
    await db.refresh(account)
    for chute_id, chute_name, chute_slug in invalidated_chutes:
        await _invalidate_chute(chute_id, chute_name, chute_slug)
    return account


@router.post(
    "/accounts/{account_id}/operations/cancel",
    response_model=ExternalAccountBulkCancelResponse,
)
async def cancel_account_operations(
    account_id: str,
    args: ExternalAccountBulkCancelRequest,
    db: AsyncSession = Depends(get_db_session),
    current_user: User = Depends(_require_management_role),
):
    """Request cancellation of every operation active at the account lock point."""

    account = await _account_for_user(
        db, current_user.user_id, account_id, for_update=True
    )
    now = await _database_now(db)
    action = {
        "action_id": str(uuid.uuid4()),
        "action": "bulk_cancel",
        "actor_user_id": current_user.user_id,
        "at": now.isoformat(),
        "reason": args.reason,
    }
    counts = await _request_account_cancellation(
        db,
        account_id=account.account_id,
        action=action,
        now=now,
    )
    management_metadata = dict(getattr(account, "management_metadata", None) or {})
    _append_operator_action(management_metadata, action)
    account.management_metadata = management_metadata
    account.updated_at = func.now()
    await _commit_or_conflict(db, "Unable to cancel this account's operations.")
    logger.warning(
        "External account bulk cancellation action_id={} account_id={} "
        "actor_user_id={} cancel_requested={} not_cancellable={}",
        action["action_id"],
        account.account_id,
        current_user.user_id,
        counts["cancel_requested"],
        counts["not_cancellable"],
    )
    return {
        "account_id": account.account_id,
        "action_id": action["action_id"],
        **counts,
    }


@router.post(
    "/accounts/{account_id}/credentials/force-rotate",
    response_model=ExternalCredentialForceRotateResponse,
)
async def force_rotate_account_credentials(
    account_id: str,
    args: ExternalCredentialForceRotateRequest,
    db: AsyncSession = Depends(get_db_session),
    current_user: User = Depends(_require_management_role),
):
    """Disable admissions and rotate existing credential names during an incident."""

    account = await _account_for_user(
        db, current_user.user_id, account_id, for_update=True
    )
    references = dict(account.credential_references or {})
    unknown_names = set(args.credentials) - set(references)
    if unknown_names:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=(
                "Force rotation accepts only existing credential name(s); unknown: "
                f"{', '.join(sorted(unknown_names))}."
            ),
        )

    now = await _database_now(db)
    action = {
        "action_id": str(uuid.uuid4()),
        "action": "force_credential_rotation",
        "actor_user_id": current_user.user_id,
        "at": now.isoformat(),
        "reason": args.reason,
        "rotated_credential_count": len(args.credentials),
    }
    counts = await _request_account_cancellation(
        db,
        account_id=account.account_id,
        action=action,
        now=now,
        audit_not_cancellable=True,
    )
    references = await _store_credentials(
        db,
        user_id=current_user.user_id,
        account_id=account.account_id,
        credentials=args.credentials,
        references=references,
    )
    account.credential_references = references
    account.enabled = False
    account.artifact_relay_invalidated_at = now
    management_metadata = dict(getattr(account, "management_metadata", None) or {})
    _append_operator_action(management_metadata, action)
    account.management_metadata = management_metadata
    account.updated_at = func.now()

    invalidated_chutes: list[tuple[str, str, str | None]] = []
    for binding in await _bindings_for_account(db, account.account_id):
        binding.chute.disabled = True
        invalidated_chutes.append(
            (
                binding.chute_id,
                binding.chute.name,
                getattr(binding.chute, "slug", None),
            )
        )

    await _commit_or_conflict(db, "Unable to force-rotate account credentials.")
    for chute_id, chute_name, chute_slug in invalidated_chutes:
        await _invalidate_chute(chute_id, chute_name, chute_slug)
    logger.warning(
        "External account credential force-rotation action_id={} account_id={} "
        "actor_user_id={} rotated_count={} cancel_requested={} not_cancellable={}",
        action["action_id"],
        account.account_id,
        current_user.user_id,
        len(args.credentials),
        counts["cancel_requested"],
        counts["not_cancellable"],
    )
    return {
        "account_id": account.account_id,
        "action_id": action["action_id"],
        **counts,
        "credential_names": sorted(args.credentials),
        "account_disabled": True,
        "artifact_relays_invalidated_at": now,
    }


@router.delete("/accounts/{account_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_account(
    account_id: str,
    db: AsyncSession = Depends(get_db_session),
    current_user: User = Depends(_require_management_role),
):
    account = await _account_for_user(
        db, current_user.user_id, account_id, for_update=True
    )
    binding_count = (
        await db.execute(
            select(func.count())
            .select_from(ExternalChuteBinding)
            .where(ExternalChuteBinding.account_id == account.account_id)
        )
    ).scalar_one()
    if binding_count:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Delete or move this account's Chute bindings first.",
        )
    # Historical terminal operations retain their usage, charge, and public result
    # metadata but no longer need provider credentials. Active work and unexpired
    # relays are rejected by the shared transport guard; the database detaches safe
    # history from this account when the account and its managed Secrets are removed.
    await ensure_account_transport_mutation_safe(db, account.account_id)
    await db.execute(
        delete(Secret).where(
            Secret.user_id == current_user.user_id,
            Secret.purpose == account.account_id,
            Secret.kind == "external_backend",
        )
    )
    await db.delete(account)
    await _commit_or_conflict(db, "This account is still in use and cannot be deleted.")
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.get("/bindings", response_model=list[ExternalChuteBindingResponse])
async def list_bindings(
    db: AsyncSession = Depends(get_db_session),
    current_user: User = Depends(get_current_user()),
):
    result = await db.execute(
        select(ExternalChuteBinding)
        .join(
            ExternalBackendAccount,
            ExternalBackendAccount.account_id == ExternalChuteBinding.account_id,
        )
        .where(ExternalBackendAccount.user_id == current_user.user_id)
        .order_by(ExternalChuteBinding.created_at.desc())
    )
    return result.unique().scalars().all()


@router.post(
    "/bindings",
    response_model=ExternalChuteBindingResponse,
    status_code=status.HTTP_201_CREATED,
)
async def create_binding(
    args: ExternalChuteBindingCreate,
    db: AsyncSession = Depends(get_db_session),
    current_user: User = Depends(require_external_provisioner),
):
    account = await _account_for_user(
        db, current_user.user_id, args.account_id, for_share=True
    )
    if not account.enabled:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="The selected account is disabled.",
        )
    chute = await _external_chute_for_user(db, current_user.user_id, args.chute_id)
    if chute.external_binding is not None:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="This Chute already has an external binding.",
        )
    price_override = await _price_override(db, chute.chute_id)
    try:
        parsed_rules = parse_pricing_rules(
            getattr(price_override, "pricing_rules", None) if price_override else None
        )
    except PricingConfigurationError as exc:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="This Chute has invalid pricing rules.",
        ) from exc
    if not parsed_rules:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Pricing rules must be configured before adding a binding.",
        )
    _validate_binding_routes(chute, args.routes)
    _validate_route_configurations(
        account,
        args.routes,
        chute.cords,
        price_override.pricing_rules,
    )
    binding = ExternalChuteBinding(
        binding_id=str(uuid.uuid4()),
        chute_id=chute.chute_id,
        account_id=args.account_id,
        routes=_dump_routes(args.routes),
        enabled=args.enabled,
    )
    db.add(binding)
    chute.disabled = not args.enabled
    chute.version = _version_for_external_chute(
        account_id=account.account_id,
        standard_template=chute.standard_template,
        cords=chute.cords,
        routes=args.routes,
        pricing_rules=price_override.pricing_rules,
    )
    await _commit_or_conflict(db, "Unable to create this binding.")
    await db.refresh(binding)
    await _invalidate_chute(chute.chute_id, chute.name, getattr(chute, "slug", None))
    return binding


@router.get("/bindings/{binding_id}", response_model=ExternalChuteBindingResponse)
async def get_binding(
    binding_id: str,
    db: AsyncSession = Depends(get_db_session),
    current_user: User = Depends(get_current_user()),
):
    return await _binding_for_user(db, current_user.user_id, binding_id)


@router.patch("/bindings/{binding_id}", response_model=ExternalChuteBindingResponse)
async def update_binding(
    binding_id: str,
    args: ExternalChuteBindingUpdate,
    db: AsyncSession = Depends(get_db_session),
    current_user: User = Depends(_require_management_role),
):
    binding = await _binding_for_user(
        db, current_user.user_id, binding_id, for_update=True
    )
    account = await _account_for_user(
        db,
        current_user.user_id,
        args.account_id or binding.account_id,
        for_share=True,
    )
    chute = await _external_chute_for_user(
        db, current_user.user_id, binding.chute_id, for_update=True
    )
    configuration_changed = False
    if args.account_id is not None:
        if not account.enabled:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="The selected account is disabled.",
            )
        binding.account_id = args.account_id
        configuration_changed = True
    if args.routes is not None:
        _validate_binding_routes(chute, args.routes)
        binding.routes = _dump_routes(args.routes)
        configuration_changed = True
    if args.enabled is not None:
        binding.enabled = args.enabled
    if args.account_id is not None or args.enabled is not None:
        chute.disabled = not (account.enabled and binding.enabled)
    if configuration_changed:
        price_override = await _price_override(db, binding.chute_id)
        try:
            parsed_rules = parse_pricing_rules(
                getattr(price_override, "pricing_rules", None)
                if price_override
                else None
            )
        except PricingConfigurationError as exc:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="This Chute has invalid pricing rules.",
            ) from exc
        if not parsed_rules:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="Pricing rules must be configured before updating a binding.",
            )
        _validate_route_configurations(
            account,
            binding.routes,
            chute.cords,
            price_override.pricing_rules,
        )
        chute.version = _version_for_external_chute(
            account_id=binding.account_id,
            standard_template=chute.standard_template,
            cords=chute.cords,
            routes=binding.routes,
            pricing_rules=price_override.pricing_rules,
        )
    binding.updated_at = func.now()
    await _commit_or_conflict(db, "Unable to update this binding.")
    await db.refresh(binding)
    await _invalidate_chute(binding.chute_id, chute.name, getattr(chute, "slug", None))
    return binding


@router.delete("/bindings/{binding_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_binding(
    binding_id: str,
    db: AsyncSession = Depends(get_db_session),
    current_user: User = Depends(_require_management_role),
):
    binding = await _binding_for_user(
        db, current_user.user_id, binding_id, for_update=True
    )
    chute = await _external_chute_for_user(
        db, current_user.user_id, binding.chute_id, for_update=True
    )
    chute_id = binding.chute_id
    chute_name = chute.name
    chute.disabled = True
    await db.delete(binding)
    await db.commit()
    await _invalidate_chute(chute_id, chute_name, getattr(chute, "slug", None))
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.get("/chutes", response_model=list[ExternalChuteResponse])
async def list_external_chutes(
    db: AsyncSession = Depends(get_db_session),
    current_user: User = Depends(get_current_user()),
):
    chutes = (
        (
            await db.execute(
                select(Chute)
                .where(
                    Chute.user_id == current_user.user_id,
                    Chute.execution_backend == "external",
                )
                .options(selectinload(Chute.external_binding))
                .order_by(Chute.created_at.desc())
            )
        )
        .unique()
        .scalars()
        .all()
    )
    if not chutes:
        return []
    prices = (
        (
            await db.execute(
                select(PriceOverride).where(
                    PriceOverride.user_id == "*",
                    PriceOverride.chute_id.in_([chute.chute_id for chute in chutes]),
                )
            )
        )
        .scalars()
        .all()
    )
    by_chute = {price.chute_id: price for price in prices}
    return [_chute_response(chute, by_chute.get(chute.chute_id)) for chute in chutes]


async def _available_slug(
    db: AsyncSession, username: str, name: str, chute_id: str
) -> str:
    base = re.sub(
        r"[^a-z0-9-]+$",
        "-",
        slugify(f"{username}-{name}", max_length=58).lower(),
    ).strip("-")
    base = base or f"external-{chute_id[:8]}"
    if not (await db.execute(select(exists().where(Chute.slug == base)))).scalar():
        return base
    return f"{base[:52].rstrip('-')}-{chute_id[:5]}"


@router.post(
    "/chutes",
    response_model=ExternalChuteResponse,
    status_code=status.HTTP_201_CREATED,
)
async def create_external_chute(
    args: ExternalChuteCreate,
    db: AsyncSession = Depends(get_db_session),
    current_user: User = Depends(require_external_provisioner),
):
    account = await _account_for_user(
        db, current_user.user_id, args.account_id, for_share=True
    )
    if not account.enabled:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="The selected account is disabled.",
        )
    if args.public and not current_user.has_role(Permissioning.public_model_deployment):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You do not have permission to create a public Chute.",
        )
    existing = (
        await db.execute(
            select(Chute.chute_id).where(
                Chute.user_id == current_user.user_id,
                func.lower(Chute.name) == args.name.lower(),
            )
        )
    ).scalar_one_or_none()
    if existing is not None:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="A Chute with this name already exists.",
        )

    chute_id = str(
        uuid.uuid5(uuid.NAMESPACE_OID, f"{current_user.username}::chute::{args.name}")
    )
    version = _version_for_external_chute(
        account_id=account.account_id,
        standard_template=args.standard_template,
        cords=args.cords,
        routes=args.routes,
        pricing_rules=args.pricing_rules,
    )
    persisted_cords = _to_persisted_cords(args.cords)
    _validate_route_configurations(
        account,
        args.routes,
        args.cords,
        args.pricing_rules,
    )
    try:
        chute = Chute(
            chute_id=chute_id,
            user_id=current_user.user_id,
            execution_backend="external",
            image_id=None,
            name=args.name,
            tagline=args.tagline,
            readme=args.readme,
            tool_description=args.tool_description,
            logo_id=args.logo_id,
            public=args.public,
            standard_template=args.standard_template,
            cords=persisted_cords,
            jobs=[],
            node_selector={},
            slug=await _available_slug(db, current_user.username, args.name, chute_id),
            code=None,
            filename=None,
            ref_str=None,
            version=version,
            chutes_version=None,
            allow_external_egress=False,
            encrypted_fs=False,
            tee=False,
            lock_modules=False,
            disabled=False,
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Invalid Chute configuration: {exc}",
        ) from exc

    binding = ExternalChuteBinding(
        binding_id=str(uuid.uuid4()),
        chute_id=chute_id,
        account_id=account.account_id,
        routes=_dump_routes(args.routes),
        enabled=True,
    )
    price_override = PriceOverride(
        user_id="*",
        chute_id=chute_id,
        pricing_rules=args.pricing_rules,
    )
    db.add_all([chute, binding, price_override])
    await _commit_or_conflict(db, "Unable to create this external Chute.")
    await db.refresh(chute)
    await db.refresh(binding)
    chute.external_binding = binding
    await _invalidate_chute(chute.chute_id, chute.name, chute.slug)
    await _invalidate_price_override_cache(chute.chute_id)
    return _chute_response(chute, price_override)


@router.get("/chutes/{chute_id}", response_model=ExternalChuteResponse)
async def get_external_chute(
    chute_id: str,
    db: AsyncSession = Depends(get_db_session),
    current_user: User = Depends(get_current_user()),
):
    chute = await _external_chute_for_user(db, current_user.user_id, chute_id)
    return _chute_response(chute, await _price_override(db, chute.chute_id))


@router.patch("/chutes/{chute_id}", response_model=ExternalChuteResponse)
async def update_external_chute(
    chute_id: str,
    args: ExternalChuteUpdate,
    db: AsyncSession = Depends(get_db_session),
    current_user: User = Depends(_require_management_role),
):
    chute = await _external_chute_for_user(db, current_user.user_id, chute_id)
    binding = chute.external_binding
    if binding is None:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="This external Chute has no binding.",
        )
    binding = await _binding_for_user(
        db, current_user.user_id, binding.binding_id, for_update=True
    )
    if args.public and not current_user.has_role(Permissioning.public_model_deployment):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You do not have permission to make this Chute public.",
        )
    account = await _account_for_user(
        db,
        current_user.user_id,
        args.account_id or binding.account_id,
        for_share=True,
    )
    chute = await _external_chute_for_user(
        db, current_user.user_id, chute_id, for_update=True
    )
    if args.account_id is not None:
        if not account.enabled:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="The selected account is disabled.",
            )
        binding.account_id = account.account_id
    if args.cords is not None and args.routes is not None:
        chute.cords = _to_persisted_cords(args.cords)
        binding.routes = _dump_routes(args.routes)
    for field_name in (
        "tagline",
        "readme",
        "tool_description",
        "logo_id",
        "public",
        "standard_template",
    ):
        if field_name in args.model_fields_set:
            setattr(chute, field_name, getattr(args, field_name))
    if args.enabled is not None:
        binding.enabled = args.enabled
    if args.account_id is not None or args.enabled is not None:
        chute.disabled = not (account.enabled and binding.enabled)

    price_override = await _price_override(db, chute.chute_id)
    if args.pricing_rules is not None:
        if price_override is None:
            price_override = PriceOverride(user_id="*", chute_id=chute.chute_id)
            db.add(price_override)
        price_override.pricing_rules = args.pricing_rules
    if price_override is None or not price_override.pricing_rules:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="External Chutes require pricing rules.",
        )

    _validate_route_configurations(
        account,
        binding.routes,
        chute.cords,
        price_override.pricing_rules,
    )

    chute.version = _version_for_external_chute(
        account_id=binding.account_id,
        standard_template=chute.standard_template,
        cords=chute.cords,
        routes=binding.routes,
        pricing_rules=price_override.pricing_rules,
    )
    chute.updated_at = func.now()
    binding.updated_at = func.now()
    await _commit_or_conflict(db, "Unable to update this external Chute.")
    await db.refresh(chute)
    await db.refresh(binding)
    chute.external_binding = binding
    await _invalidate_chute(chute.chute_id, chute.name, chute.slug)
    await _invalidate_price_override_cache(chute.chute_id)
    return _chute_response(chute, price_override)


@router.delete("/chutes/{chute_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_external_chute(
    chute_id: str,
    db: AsyncSession = Depends(get_db_session),
    current_user: User = Depends(_require_management_role),
):
    chute = await _external_chute_for_user(db, current_user.user_id, chute_id)
    await ensure_no_active_external_operations(db, chute.chute_id)
    chute_name = chute.name
    chute_slug = chute.slug
    await db.execute(
        delete(PriceOverride).where(PriceOverride.chute_id == chute.chute_id)
    )
    await db.delete(chute)
    await db.commit()
    await _invalidate_chute(chute_id, chute_name, chute_slug)
    await _invalidate_price_override_cache(chute_id)
    return Response(status_code=status.HTTP_204_NO_CONTENT)
