"""Transactional delivery of externally executed usage into platform billing."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from typing import Any

from loguru import logger
from sqlalchemy import select, text
from sqlalchemy.dialects.postgresql import insert

from api.database import get_session
from .metrics import billing_delivery_attempts
from .schemas import (
    ExternalOperation,
    ExternalSettlementStatus,
    ExternalUsageOutbox,
)


class ExternalUsageDeliveryError(RuntimeError):
    """A durable usage event could not be applied to billing."""


@dataclass(frozen=True, slots=True)
class ExternalUsageEvent:
    event_id: str
    operation_id: str
    user_id: str
    chute_id: str
    app_id: str | None
    amount: Decimal
    paygo_amount: Decimal
    input_tokens: Decimal
    output_tokens: Decimal
    cached_tokens: Decimal
    compute_time: float
    track_task_completion: bool
    free_invocation: bool
    increment_invocation_quota: bool
    occurred_at: datetime


@dataclass(frozen=True, slots=True)
class ExternalUsageDeliveryReceipt:
    event_id: str
    operation_id: str
    user_id: str
    chute_id: str
    amount: Decimal
    paygo_amount: Decimal
    compute_time: float
    track_task_completion: bool
    free_invocation: bool
    increment_invocation_quota: bool


def _decimal(value: Any, field_name: str) -> Decimal:
    try:
        result = value if isinstance(value, Decimal) else Decimal(str(value))
    except Exception as exc:
        raise ExternalUsageDeliveryError(
            f"external usage event has an invalid {field_name}"
        ) from exc
    if not result.is_finite() or result < 0:
        raise ExternalUsageDeliveryError(
            f"external usage event has an invalid {field_name}"
        )
    return result


def _event_values(event: ExternalUsageEvent) -> dict[str, Any]:
    if event.occurred_at.tzinfo is None:
        raise ExternalUsageDeliveryError(
            "external usage timestamp must include a timezone"
        )
    return {
        "event_id": event.event_id,
        "operation_id": event.operation_id,
        "user_id": event.user_id,
        "chute_id": event.chute_id,
        "app_id": event.app_id,
        "amount": _decimal(event.amount, "amount"),
        "paygo_amount": _decimal(event.paygo_amount, "paygo amount"),
        "input_tokens": _decimal(event.input_tokens, "input tokens"),
        "output_tokens": _decimal(event.output_tokens, "output tokens"),
        "cached_tokens": _decimal(event.cached_tokens, "cached tokens"),
        "compute_time": float(_decimal(event.compute_time, "compute time")),
        "track_task_completion": bool(event.track_task_completion),
        "free_invocation": bool(event.free_invocation),
        "increment_invocation_quota": bool(event.increment_invocation_quota),
        "occurred_at": event.occurred_at,
        "attempts": 0,
        "next_attempt_at": event.occurred_at,
    }


async def enqueue_external_usage_event(session: Any, event: ExternalUsageEvent) -> bool:
    """Persist an immutable event in the caller's operation transaction."""

    statement = (
        insert(ExternalUsageOutbox)
        .values(**_event_values(event))
        .on_conflict_do_nothing(index_elements=[ExternalUsageOutbox.operation_id])
        .returning(ExternalUsageOutbox.event_id)
    )
    return (await session.execute(statement)).scalar_one_or_none() is not None


async def external_usage_event_exists(session: Any, operation_id: str) -> bool:
    """Return whether the operation already has an immutable pending charge."""

    event_id = (
        await session.execute(
            select(ExternalUsageOutbox.event_id).where(
                ExternalUsageOutbox.operation_id == operation_id
            )
        )
    ).scalar_one_or_none()
    return event_id is not None


def _hour_bucket(value: datetime) -> datetime:
    if value.tzinfo is None:
        raise ExternalUsageDeliveryError(
            "external usage timestamp must include a timezone"
        )
    return value.astimezone(timezone.utc).replace(minute=0, second=0, microsecond=0)


async def _apply_usage(session: Any, event: ExternalUsageOutbox) -> None:
    """Apply one locked event; the caller owns the surrounding transaction."""

    amount = _decimal(event.amount, "amount")
    paygo_amount = _decimal(event.paygo_amount, "paygo amount")
    input_tokens = _decimal(event.input_tokens, "input tokens")
    output_tokens = _decimal(event.output_tokens, "output tokens")
    cached_tokens = _decimal(event.cached_tokens, "cached tokens")
    compute_time = float(_decimal(event.compute_time, "compute time"))
    amount_value = float(amount)
    paygo_amount_value = float(paygo_amount)
    bucket = _hour_bucket(event.occurred_at)

    await session.execute(
        text(
            """
            INSERT INTO usage_data (
                user_id, bucket, chute_id, amount, count, input_tokens,
                output_tokens, cached_tokens, compute_time, paygo_amount
            ) VALUES (
                :user_id, :bucket, :chute_id, :amount, 1, :input_tokens,
                :output_tokens, :cached_tokens, :compute_time, :paygo_amount
            )
            ON CONFLICT (user_id, chute_id, bucket)
            DO UPDATE SET
                amount = usage_data.amount + EXCLUDED.amount,
                count = usage_data.count + EXCLUDED.count,
                input_tokens = COALESCE(usage_data.input_tokens, 0) + EXCLUDED.input_tokens,
                output_tokens = COALESCE(usage_data.output_tokens, 0) + EXCLUDED.output_tokens,
                cached_tokens = COALESCE(usage_data.cached_tokens, 0) + EXCLUDED.cached_tokens,
                compute_time = COALESCE(usage_data.compute_time, 0) + EXCLUDED.compute_time,
                paygo_amount = COALESCE(usage_data.paygo_amount, 0) + EXCLUDED.paygo_amount
            """
        ),
        {
            "user_id": event.user_id,
            "bucket": bucket,
            "chute_id": event.chute_id,
            "amount": amount_value,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "cached_tokens": cached_tokens,
            "compute_time": compute_time,
            "paygo_amount": paygo_amount_value,
        },
    )

    if event.app_id:
        await session.execute(
            text(
                """
                INSERT INTO app_usage_data (
                    app_id, user_id, bucket, chute_id, amount, count,
                    input_tokens, output_tokens, cached_tokens, compute_time,
                    paygo_amount
                ) VALUES (
                    :app_id, :user_id, :bucket, :chute_id, :amount, 1,
                    :input_tokens, :output_tokens, :cached_tokens, :compute_time,
                    :paygo_amount
                )
                ON CONFLICT (app_id, user_id, chute_id, bucket)
                DO UPDATE SET
                    amount = app_usage_data.amount + EXCLUDED.amount,
                    count = app_usage_data.count + EXCLUDED.count,
                    input_tokens = COALESCE(app_usage_data.input_tokens, 0) + EXCLUDED.input_tokens,
                    output_tokens = COALESCE(app_usage_data.output_tokens, 0) + EXCLUDED.output_tokens,
                    cached_tokens = COALESCE(app_usage_data.cached_tokens, 0) + EXCLUDED.cached_tokens,
                    compute_time = COALESCE(app_usage_data.compute_time, 0) + EXCLUDED.compute_time,
                    paygo_amount = COALESCE(app_usage_data.paygo_amount, 0) + EXCLUDED.paygo_amount
                """
            ),
            {
                "app_id": event.app_id,
                "user_id": event.user_id,
                "bucket": bucket,
                "chute_id": event.chute_id,
                "amount": amount_value,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "cached_tokens": cached_tokens,
                "compute_time": compute_time,
                "paygo_amount": paygo_amount_value,
            },
        )

    # Match the hosted billing lock order: usage aggregates first, then user.
    # If the user disappeared, raising here rolls all aggregate changes back.
    user_exists = (
        await session.execute(
            text("SELECT 1 FROM users WHERE user_id = :user_id FOR UPDATE"),
            {"user_id": event.user_id},
        )
    ).scalar_one_or_none()
    if user_exists is None:
        raise ExternalUsageDeliveryError("billing user is unavailable")

    # Acceptance freezes the billing decision into the immutable event amount.
    # A later role grant/revocation must not make accepted provider spend free or
    # retroactively charge an invocation which was accepted as free.
    if amount > 0:
        await session.execute(
            text(
                "UPDATE users SET balance = balance - :amount WHERE user_id = :user_id"
            ),
            {"amount": amount_value, "user_id": event.user_id},
        )


def _receipt(event: ExternalUsageOutbox) -> ExternalUsageDeliveryReceipt:
    return ExternalUsageDeliveryReceipt(
        event_id=event.event_id,
        operation_id=event.operation_id,
        user_id=event.user_id,
        chute_id=event.chute_id,
        amount=_decimal(event.amount, "amount"),
        paygo_amount=_decimal(event.paygo_amount, "paygo amount"),
        compute_time=float(_decimal(event.compute_time, "compute time")),
        track_task_completion=bool(event.track_task_completion),
        free_invocation=bool(event.free_invocation),
        increment_invocation_quota=bool(event.increment_invocation_quota),
    )


async def _record_delivery_failure(operation_id: str, error: Exception) -> None:
    """Persist bounded retry state without storing database or credential details."""

    try:
        async with get_session() as session:
            event = (
                await session.execute(
                    select(ExternalUsageOutbox)
                    .where(ExternalUsageOutbox.operation_id == operation_id)
                    .with_for_update()
                )
            ).scalar_one_or_none()
            if event is None:
                return
            attempts = max(0, int(event.attempts or 0)) + 1
            delay = min(3600.0, 5.0 * (2 ** min(attempts - 1, 10)))
            event.attempts = attempts
            event.next_attempt_at = datetime.now(timezone.utc) + timedelta(
                seconds=delay
            )
            event.last_error_code = type(error).__name__[:128]
    except Exception:
        logger.exception(
            "Failed to record billing outbox retry for external operation {}",
            operation_id,
        )


async def deliver_external_usage_event(
    operation_id: str,
) -> ExternalUsageDeliveryReceipt | None:
    """Apply and acknowledge one event in the same database transaction.

    Returning ``None`` means another worker already completed delivery. A crash at
    any point before commit rolls back the usage row, balance deduction, operation
    transition, and outbox deletion together.
    """

    try:
        receipt: ExternalUsageDeliveryReceipt | None = None
        async with get_session() as session:
            await session.execute(
                text(
                    "SELECT pg_advisory_xact_lock(hashtextextended(:key, 0))"
                ).bindparams(key=f"external-settlement:{operation_id}")
            )
            operation = await session.get(
                ExternalOperation, operation_id, with_for_update=True
            )
            if operation is None:
                raise ExternalUsageDeliveryError("external operation is unavailable")
            event = (
                await session.execute(
                    select(ExternalUsageOutbox)
                    .where(ExternalUsageOutbox.operation_id == operation_id)
                    .with_for_update()
                )
            ).scalar_one_or_none()
            if event is None:
                if (
                    operation.settlement_status
                    == ExternalSettlementStatus.SETTLED.value
                ):
                    return None
                raise ExternalUsageDeliveryError("external usage event is unavailable")
            if operation.settlement_status == ExternalSettlementStatus.SETTLED.value:
                # A committed delivery deletes its event in the same transaction.
                # Refuse an impossible reintroduced event rather than double-charge.
                raise ExternalUsageDeliveryError(
                    "settled operation still has an external usage event"
                )

            await _apply_usage(session, event)
            receipt = _receipt(event)
            now = datetime.now(timezone.utc)
            metadata = dict(operation.settlement_metadata or {})
            metadata["settlement_delivery_at"] = now.isoformat()
            metadata["settlement_delivery"] = "applied"
            metadata.pop("settlement_next_attempt_at", None)
            operation.settlement_metadata = metadata
            operation.settlement_status = ExternalSettlementStatus.SETTLED.value
            operation.settled_at = now
            operation.next_poll_at = None
            await session.delete(event)
        billing_delivery_attempts.labels(outcome="applied").inc()
        return receipt
    except Exception as exc:
        billing_delivery_attempts.labels(outcome="failed").inc()
        await _record_delivery_failure(operation_id, exc)
        raise


__all__ = [
    "ExternalUsageDeliveryError",
    "ExternalUsageDeliveryReceipt",
    "ExternalUsageEvent",
    "deliver_external_usage_event",
    "enqueue_external_usage_event",
    "external_usage_event_exists",
]
