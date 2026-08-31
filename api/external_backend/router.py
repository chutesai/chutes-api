"""Public operation status, cancellation, and remote-result relay endpoints."""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from typing import Any, Optional

from fastapi import APIRouter, Depends, HTTPException, Request, Response
from loguru import logger
from sqlalchemy import func, select, text
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import joinedload
from starlette.responses import StreamingResponse

from api.database import get_db_session, get_session
from api.external_transport import (
    ArtifactRelay,
    ExternalTransportError,
    ResponseTooLargeError,
)
from api.external_transport.security import (
    public_artifact_content_type,
    public_artifact_disposition,
)
from api.pagination import PaginatedResponse
from api.user.schemas import User
from api.user.service import get_current_user

from .config import ExternalConfigurationError, build_artifact_profile
from .artifact_limits import (
    ArtifactByteReservation,
    ArtifactRelayLimitError,
    acquire_artifact_relay,
    finalize_artifact_bytes,
    reserve_artifact_bytes,
)
from .metrics import artifact_bytes, artifact_requests
from .model_compat import public_charge_line_items
from .public_urls import artifact_url, operation_url
from .schemas import (
    ExternalOperation,
    ExternalOperationMode,
    ExternalOperationStatus,
    ExternalRouteConfig,
)
from .service import build_secret_resolver


router = APIRouter()


def _timestamp(value: Any) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.replace(tzinfo=value.tzinfo or timezone.utc)
    if not isinstance(value, str):
        return None
    normalized = value[:-1] + "+00:00" if value.endswith("Z") else value
    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError:
        return None
    return parsed.replace(tzinfo=parsed.tzinfo or timezone.utc)


def _artifact_relay_cutoff(operation: ExternalOperation) -> datetime | None:
    # Callers that expose artifacts load the account explicitly. Reading through
    # ``__dict__`` prevents an accidental async lazy-load in response rendering.
    account = getattr(operation, "__dict__", {}).get("account")
    cutoff = _timestamp(getattr(account, "artifact_relay_invalidated_at", None))
    created_at = _timestamp(getattr(operation, "created_at", None))
    if cutoff is None or created_at is None or created_at > cutoff:
        return None
    return cutoff


def _effective_artifact_expiration(
    operation: ExternalOperation, artifact: dict[str, Any]
) -> datetime | None:
    expiration = _timestamp(artifact.get("expires_at")) or _timestamp(
        operation.expires_at
    )
    cutoff = _artifact_relay_cutoff(operation)
    if cutoff is not None and (expiration is None or cutoff < expiration):
        return cutoff
    return expiration


def _is_expired(
    operation: ExternalOperation,
    artifact: dict[str, Any],
    *,
    now: datetime | None = None,
) -> bool:
    expiration = _effective_artifact_expiration(operation, artifact)
    if not expiration:
        return False
    return expiration <= (now or datetime.now(timezone.utc))


def _public_operation(operation: ExternalOperation) -> dict[str, Any]:
    descriptor = operation.result_descriptor or {}
    descriptor_metadata = descriptor.get("metadata") or {}
    inline_result = (
        descriptor_metadata.get("inline_result")
        if isinstance(descriptor_metadata, dict)
        else None
    )
    artifacts = []
    for index, artifact in enumerate(descriptor.get("artifacts") or []):
        expiration = _effective_artifact_expiration(operation, artifact)
        artifacts.append(
            {
                "index": index,
                "kind": artifact.get("kind") or "artifact",
                "content_type": artifact.get("content_type"),
                "size_bytes": artifact.get("size_bytes"),
                "expires_at": expiration,
                "available": not _is_expired(operation, artifact),
                "url": artifact_url(operation.operation_id, index),
            }
        )
    usage = None
    if operation.usage:
        usage = {
            key: value
            for key, value in operation.usage.items()
            if key
            in {
                "requests",
                "tokens",
                "images",
                "input_media_seconds",
                "output_media_seconds",
                "characters",
                "counts",
                "tools",
            }
        }
    settlement = operation.settlement_metadata or {}
    pricing = settlement.get("result") or {}
    return {
        "id": operation.operation_id,
        "status_url": operation_url(operation.operation_id),
        "chute_id": operation.chute_id,
        "cord": operation.cord_path,
        "mode": operation.operation_mode,
        "status": operation.status,
        "created_at": operation.created_at,
        "submitted_at": operation.submitted_at,
        "started_at": operation.started_at,
        "finished_at": operation.finished_at,
        "expires_at": operation.expires_at,
        "usage": usage,
        "charge": (
            {
                "amount": pricing.get("charged_amount", pricing.get("amount")),
                "line_items": public_charge_line_items(pricing.get("line_items", [])),
            }
            if pricing
            else None
        ),
        "result": {
            "status": descriptor.get("status", "complete"),
            "artifacts": artifacts,
            **({"data": inline_result} if inline_result is not None else {}),
        }
        if descriptor
        else None,
        "error": (
            {
                "code": (operation.error or {}).get("code", "operation_failed"),
                "message": (operation.error or {}).get(
                    "message", "The operation could not be completed."
                ),
                "retryable": bool((operation.error or {}).get("retryable", False)),
            }
            if operation.error
            else None
        ),
    }


async def _owned_operation(
    db: AsyncSession,
    operation_id: str,
    current_user: User,
    *,
    load_account: bool = False,
    for_update: bool = False,
) -> ExternalOperation:
    query = select(ExternalOperation).where(
        ExternalOperation.operation_id == operation_id,
        ExternalOperation.user_id == current_user.user_id,
    )
    if load_account:
        query = query.options(joinedload(ExternalOperation.account))
    if for_update:
        query = query.with_for_update(of=ExternalOperation)
    operation = (await db.execute(query)).unique().scalar_one_or_none()
    if not operation:
        raise HTTPException(status_code=404, detail="Operation not found.")
    return operation


@router.get("/operations", response_model=PaginatedResponse)
async def list_operations(
    response: Response,
    page: Optional[int] = 0,
    limit: Optional[int] = 25,
    operation_status: Optional[str] = None,
    chute_id: Optional[str] = None,
    db: AsyncSession = Depends(get_db_session),
    current_user: User = Depends(get_current_user()),
):
    """List the caller's external operations without private upstream identifiers."""

    page = max(0, page or 0)
    limit = min(100, max(1, limit or 25))
    query = (
        select(ExternalOperation)
        .options(joinedload(ExternalOperation.account))
        .where(ExternalOperation.user_id == current_user.user_id)
    )
    if operation_status:
        if operation_status not in {item.value for item in ExternalOperationStatus}:
            raise HTTPException(status_code=400, detail="Invalid operation status.")
        query = query.where(ExternalOperation.status == operation_status)
    if chute_id:
        query = query.where(ExternalOperation.chute_id == chute_id)
    total = (
        await db.execute(select(func.count()).select_from(query.subquery()))
    ).scalar_one()
    items = (
        (
            await db.execute(
                query.order_by(ExternalOperation.created_at.desc())
                .offset(page * limit)
                .limit(limit)
            )
        )
        .scalars()
        .all()
    )
    response.headers["Cache-Control"] = "private, no-store"
    response.headers["X-Content-Type-Options"] = "nosniff"
    return {
        "total": total,
        "page": page,
        "limit": limit,
        "items": [_public_operation(item) for item in items],
    }


@router.get("/operations/{operation_id}")
async def get_operation(
    operation_id: str,
    response: Response,
    db: AsyncSession = Depends(get_db_session),
    current_user: User = Depends(get_current_user()),
):
    """Read current local state; task results remain stored by the external service."""

    response.headers["Cache-Control"] = "private, no-store"
    response.headers["X-Content-Type-Options"] = "nosniff"
    return _public_operation(
        await _owned_operation(db, operation_id, current_user, load_account=True)
    )


@router.post("/operations/{operation_id}/cancel", status_code=202)
async def cancel_operation(
    operation_id: str,
    response: Response,
    db: AsyncSession = Depends(get_db_session),
    current_user: User = Depends(get_current_user()),
):
    """Request explicit cancellation without coupling it to client disconnection."""

    operation = await _owned_operation(db, operation_id, current_user, for_update=True)
    if operation.status == ExternalOperationStatus.PENDING.value:
        # Submission owns the row until it either attaches the upstream identity
        # or reaches its persisted dispatch-recovery deadline. Pulling that
        # deadline forward could orphan accepted work and make it unbillable.
        raise HTTPException(
            status_code=409,
            detail="Cancellation is unavailable until submission completes.",
        )
    if operation.status in {
        ExternalOperationStatus.SUCCEEDED.value,
        ExternalOperationStatus.FAILED.value,
        ExternalOperationStatus.CANCELLED.value,
        ExternalOperationStatus.EXPIRED.value,
    }:
        raise HTTPException(status_code=409, detail="Operation is already terminal.")
    route = ExternalRouteConfig.model_validate(operation.route_snapshot)
    operation_config = dict(route.operation_config or {})
    task_config = operation_config.get("task") or {}
    cancel_config = operation_config.get("cancel") or task_config.get("cancel")
    local_session_cancel = route.operation_mode in {
        ExternalOperationMode.STREAM,
        ExternalOperationMode.REALTIME,
    }
    task_cancel = route.operation_mode is ExternalOperationMode.TASK and isinstance(
        cancel_config, dict
    )
    if not local_session_cancel and not task_cancel:
        raise HTTPException(status_code=409, detail="Cancellation is not supported.")
    settlement = dict(operation.settlement_metadata or {})
    settlement["cancel_requested"] = True
    settlement["cancel_requested_at"] = datetime.now(timezone.utc).isoformat()
    operation.settlement_metadata = settlement
    if route.operation_mode is ExternalOperationMode.TASK:
        operation.next_poll_at = datetime.now(timezone.utc)
    await db.flush()
    response.headers["Cache-Control"] = "private, no-store"
    response.headers["X-Content-Type-Options"] = "nosniff"
    return {
        "id": operation.operation_id,
        "status": operation.status,
        "cancellation": "requested",
    }


@router.api_route(
    "/operations/{operation_id}/artifacts/{artifact_index}",
    methods=["GET", "HEAD"],
)
async def relay_artifact(
    operation_id: str,
    artifact_index: int,
    request: Request,
    current_user: User = Depends(get_current_user()),
):
    """Relay a result from its upstream cache with Range support and no local blob copy."""

    # Serialize admission with account mutation, validate the retained descriptor,
    # and cache credential values before releasing the short database transaction.
    # The subsequent byte stream never holds a database connection.
    connection_config: dict[str, Any] = {}
    async with get_session() as db:
        locked = (
            await db.execute(
                text(
                    """
                    SELECT operation.operation_id
                    FROM external_operations AS operation
                    JOIN external_backend_accounts AS account
                      ON account.account_id = operation.account_id
                    WHERE operation.operation_id = :operation_id
                      AND operation.user_id = :user_id
                    FOR SHARE OF account
                    """
                ),
                {
                    "operation_id": operation_id,
                    "user_id": current_user.user_id,
                },
            )
        ).scalar_one_or_none()
        if locked is None:
            raise HTTPException(status_code=404, detail="Operation not found.")
        database_now = _timestamp(
            (await db.execute(text("SELECT clock_timestamp()"))).scalar_one()
        )
        if database_now is None:
            raise HTTPException(
                status_code=503, detail="Artifact is temporarily unavailable."
            )
        operation = await _owned_operation(
            db, operation_id, current_user, load_account=True
        )
        descriptor = operation.result_descriptor or {}
        artifacts = descriptor.get("artifacts") or []
        if artifact_index < 0 or artifact_index >= len(artifacts):
            raise HTTPException(status_code=404, detail="Artifact not found.")
        artifact = artifacts[artifact_index]
        if _is_expired(operation, artifact, now=database_now):
            raise HTTPException(status_code=410, detail="Artifact has expired.")
        reference = artifact.get("reference")
        if not isinstance(reference, str) or not reference:
            raise HTTPException(status_code=404, detail="Artifact not found.")
        try:
            route = ExternalRouteConfig.model_validate(operation.route_snapshot)
            profile = build_artifact_profile(operation.account, route, reference)
            secret_resolver = build_secret_resolver(operation.account)
            for secret_reference in set(
                reference
                for header in profile.secret_headers
                for reference in header.references.values()
            ):
                await secret_resolver(secret_reference)
            connection_config = dict(operation.account.connection_config or {})
        except Exception as exc:
            raise HTTPException(
                status_code=410, detail="Artifact is unavailable."
            ) from exc
    try:
        relay_lease = await acquire_artifact_relay(
            current_user.user_id,
            connection_config,
            lease_seconds=profile.timeout.total + 60.0,
        )
    except ArtifactRelayLimitError as exc:
        artifact_requests.labels(outcome=exc.reason).inc()
        raise HTTPException(
            status_code=503 if exc.reason == "unavailable" else 429,
            detail="Artifact relay capacity is temporarily unavailable.",
        ) from exc
    try:
        upstream = await ArtifactRelay(profile, secret_resolver=secret_resolver).open(
            reference,
            method=request.method,
            request_headers=dict(request.headers),
        )
    except ExternalConfigurationError as exc:
        await relay_lease.release()
        artifact_requests.labels(outcome="configuration_error").inc()
        raise HTTPException(status_code=410, detail="Artifact is unavailable.") from exc
    except ExternalTransportError as exc:
        await relay_lease.release()
        artifact_requests.labels(outcome="transport_error").inc()
        raise HTTPException(
            status_code=502, detail="Artifact is temporarily unavailable."
        ) from exc

    if upstream.status_code in {403, 404, 410}:
        await upstream.aclose()
        await relay_lease.release()
        artifact_requests.labels(outcome="expired").inc()
        raise HTTPException(status_code=410, detail="Artifact has expired.")
    headers = {
        name: value
        for name, value in upstream.headers.items()
        if name.lower() not in {"cache-control", "content-type"}
    }
    headers["Cache-Control"] = "private, no-store, no-transform"
    headers["X-Content-Type-Options"] = "nosniff"
    public_content_type = public_artifact_content_type(artifact.get("content_type"))
    headers["Content-Type"] = public_content_type
    headers["Content-Disposition"] = public_artifact_disposition(
        public_content_type, artifact_index
    )
    if upstream.status_code == 416:
        await upstream.aclose()
        await relay_lease.release()
        artifact_requests.labels(outcome="range_rejected").inc()
        return Response(status_code=416, headers=headers)
    if upstream.status_code not in {200, 206}:
        await upstream.aclose()
        await relay_lease.release()
        artifact_requests.labels(outcome="upstream_error").inc()
        raise HTTPException(
            status_code=502, detail="Artifact is temporarily unavailable."
        )
    if request.method == "HEAD":
        await upstream.aclose()
        await relay_lease.release()
        artifact_requests.labels(outcome="head").inc()
        return Response(status_code=upstream.status_code, headers=headers)

    raw_length = upstream.headers.get("content-length")
    expected_bytes: int | None = None
    if raw_length is not None:
        try:
            expected_bytes = int(raw_length)
        except (TypeError, ValueError, OverflowError):
            expected_bytes = None
        if expected_bytes is not None and expected_bytes < 0:
            expected_bytes = None
    try:
        async with get_session() as db:
            byte_reservation = await reserve_artifact_bytes(
                db,
                operation_id=operation_id,
                user_id=current_user.user_id,
                connection_config=connection_config,
                # Unknown-length responses cannot exceed the transport profile,
                # so reserve that bound instead of the entire per-operation cap.
                expected_bytes=(
                    expected_bytes if expected_bytes is not None else profile.max_bytes
                ),
                reservation_ttl_seconds=profile.timeout.total + 60.0,
            )
    except ArtifactRelayLimitError as exc:
        await upstream.aclose()
        await relay_lease.release()
        artifact_requests.labels(outcome=exc.reason).inc()
        raise HTTPException(
            status_code=429,
            detail="Artifact relay byte limit was reached.",
        ) from exc

    async def finalize_relay(
        reservation: ArtifactByteReservation, transferred_bytes: int
    ) -> None:
        try:
            await upstream.aclose()
        finally:
            await relay_lease.release()
        try:
            async with get_session() as db:
                await finalize_artifact_bytes(
                    db,
                    reservation,
                    transferred_bytes=transferred_bytes,
                )
        except Exception:
            # A retained reservation is conservative and expires at the transport
            # recovery deadline. Never turn cleanup telemetry into a response leak.
            logger.exception(
                "Failed to reconcile artifact byte reservation for operation {}",
                operation_id,
            )

    async def relay_bytes():
        transferred_bytes = 0
        try:
            async for chunk in upstream.iter_bytes(
                max_bytes=byte_reservation.max_bytes
            ):
                transferred_bytes += len(chunk)
                artifact_bytes.inc(len(chunk))
                yield chunk
        except ResponseTooLargeError:
            artifact_requests.labels(outcome="byte_limit").inc()
            raise
        finally:
            cleanup = asyncio.create_task(
                finalize_relay(byte_reservation, transferred_bytes)
            )
            try:
                await asyncio.shield(cleanup)
            except asyncio.CancelledError:
                await cleanup
                raise

    artifact_requests.labels(outcome="streamed").inc()
    return StreamingResponse(
        relay_bytes(),
        status_code=upstream.status_code,
        media_type=public_content_type,
        headers=headers,
    )


__all__ = ["router"]
