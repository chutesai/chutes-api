"""Cross-replica request, concurrency, and byte caps for artifact relays."""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Mapping

from loguru import logger
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from api.config import settings

from .governance import GovernancePolicy, governance_policy
from .schemas import ExternalOperation


_ACQUIRE_LUA = """
redis.call('ZREMRANGEBYSCORE', KEYS[1], '-inf', ARGV[1])
if redis.call('ZCARD', KEYS[1]) >= tonumber(ARGV[3]) then
    return 'concurrency'
end
local requests = redis.call('INCR', KEYS[2])
if requests == 1 then
    redis.call('EXPIRE', KEYS[2], 120)
end
if requests > tonumber(ARGV[4]) then
    return 'rate'
end
redis.call('ZADD', KEYS[1], ARGV[2], ARGV[5])
local active_ttl = redis.call('TTL', KEYS[1])
if active_ttl < tonumber(ARGV[6]) then
    redis.call('EXPIRE', KEYS[1], ARGV[6])
end
return 'ok'
"""


class ArtifactRelayLimitError(RuntimeError):
    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


@dataclass(slots=True)
class ArtifactRelayLease:
    user_id: str
    token: str
    _released: bool = False

    async def release(self) -> None:
        if self._released:
            return
        self._released = True
        try:
            await settings.redis_client.zrem(
                f"external_artifact:active:{self.user_id}", self.token
            )
        except Exception:
            # The lease has a transport-derived expiry. A telemetry outage while
            # releasing it should conservatively retain capacity, not corrupt an
            # otherwise successful artifact response.
            logger.warning("External artifact relay lease release was deferred")


@dataclass(frozen=True, slots=True)
class ArtifactByteReservation:
    """A durable byte-budget claim which is reconciled after streaming."""

    operation_id: str
    user_id: str
    token: str
    max_bytes: int


async def acquire_artifact_relay(
    user_id: str,
    connection_config: Mapping[str, Any] | None,
    *,
    lease_seconds: float,
) -> ArtifactRelayLease:
    policy = governance_policy(connection_config)
    if (
        isinstance(lease_seconds, bool)
        or not isinstance(lease_seconds, (int, float))
        or not 10 <= float(lease_seconds) <= 86520
    ):
        raise ArtifactRelayLimitError("unavailable")
    now = time.time()
    lease_ttl = int(float(lease_seconds) + 0.999)
    token = str(uuid.uuid4())
    window = int(now // 60)
    try:
        result = await settings.redis_client.client.eval(
            _ACQUIRE_LUA,
            2,
            f"external_artifact:active:{user_id}",
            f"external_artifact:rate:{user_id}:{window}",
            now,
            now + lease_ttl,
            policy.artifact_max_concurrent_per_user,
            policy.artifact_requests_per_minute,
            token,
            lease_ttl,
        )
    except Exception as exc:
        raise ArtifactRelayLimitError("unavailable") from exc
    normalized = result.decode() if isinstance(result, bytes) else result
    if normalized != "ok":
        raise ArtifactRelayLimitError(str(normalized or "unavailable"))
    return ArtifactRelayLease(user_id=user_id, token=token)


def _reserved_bytes(metadata: Mapping[str, Any] | None) -> int:
    if not isinstance(metadata, Mapping):
        return 0
    value = metadata.get("artifact_relay_bytes_reserved", 0)
    return (
        value
        if isinstance(value, int) and not isinstance(value, bool) and value >= 0
        else 0
    )


def _reservation_records(
    metadata: Mapping[str, Any] | None,
) -> dict[str, dict[str, int]]:
    if not isinstance(metadata, Mapping):
        return {}
    raw = metadata.get("artifact_relay_reservations")
    if not isinstance(raw, Mapping):
        return {}
    records: dict[str, dict[str, int]] = {}
    for raw_token, raw_record in raw.items():
        if not isinstance(raw_token, str) or not isinstance(raw_record, Mapping):
            continue
        values: dict[str, int] = {}
        valid = True
        for field in ("bytes", "bucket", "expires_at"):
            value = raw_record.get(field)
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                valid = False
                break
            values[field] = value
        if valid:
            records[raw_token] = values
    return records


def _refund_reservation(
    metadata: dict[str, Any],
    refund: int,
) -> None:
    if refund <= 0:
        return
    total = _reserved_bytes(metadata)
    metadata["artifact_relay_bytes_reserved"] = max(0, total - refund)


def _prune_expired_reservations(
    metadata: dict[str, Any], *, now_epoch: int
) -> tuple[dict[str, dict[str, int]], tuple[dict[str, int], ...]]:
    records = _reservation_records(metadata)
    active: dict[str, dict[str, int]] = {}
    expired: list[dict[str, int]] = []
    for token, record in records.items():
        if record["expires_at"] <= now_epoch:
            _refund_reservation(metadata, record["bytes"])
            expired.append(record)
        else:
            active[token] = record
    metadata["artifact_relay_reservations"] = active
    return active, tuple(expired)


def _bucket_timestamp(bucket: int) -> datetime:
    return datetime.fromtimestamp(bucket * 60, tz=timezone.utc)


async def _adjust_artifact_bucket(
    db: AsyncSession,
    *,
    user_id: str,
    bucket: int,
    delta: int,
) -> None:
    if delta == 0:
        return
    if delta > 0:
        await db.execute(
            text(
                """
                INSERT INTO external_governance_buckets (
                    scope_type, scope_id, bucket_start, artifact_relay_bytes
                ) VALUES (
                    'user', :user_id, :bucket_start, :delta
                )
                ON CONFLICT (scope_type, scope_id, bucket_start)
                DO UPDATE SET
                    artifact_relay_bytes =
                        external_governance_buckets.artifact_relay_bytes
                        + EXCLUDED.artifact_relay_bytes,
                    updated_at = NOW()
                """
            ),
            {
                "user_id": user_id,
                "bucket_start": _bucket_timestamp(bucket),
                "delta": delta,
            },
        )
        return
    await db.execute(
        text(
            """
            UPDATE external_governance_buckets
               SET artifact_relay_bytes = GREATEST(
                       artifact_relay_bytes - :refund,
                       0
                   ),
                   updated_at = NOW()
             WHERE scope_type = 'user'
               AND scope_id = :user_id
               AND bucket_start = :bucket_start
            """
        ),
        {
            "user_id": user_id,
            "bucket_start": _bucket_timestamp(bucket),
            "refund": -delta,
        },
    )


async def reserve_artifact_bytes(
    db: AsyncSession,
    *,
    operation_id: str,
    user_id: str,
    connection_config: Mapping[str, Any] | None,
    expected_bytes: int | None,
    reservation_ttl_seconds: float = 3600.0,
) -> ArtifactByteReservation:
    """Atomically reserve relay capacity; unknown lengths consume the remaining cap."""

    if (
        isinstance(reservation_ttl_seconds, bool)
        or not isinstance(reservation_ttl_seconds, (int, float))
        or not 10 <= float(reservation_ttl_seconds) <= 86520
    ):
        raise ArtifactRelayLimitError("unavailable")

    policy: GovernancePolicy = governance_policy(connection_config)
    await db.execute(
        text("SELECT pg_advisory_xact_lock(hashtextextended(:key, 0))"),
        {"key": f"external-artifact-user:{user_id}"},
    )
    operation = await db.get(ExternalOperation, operation_id, with_for_update=True)
    if operation is None or operation.user_id != user_id:
        raise ArtifactRelayLimitError("unavailable")
    now = (await db.execute(text("SELECT clock_timestamp()"))).scalar_one()
    if not isinstance(now, datetime):
        raise ArtifactRelayLimitError("unavailable")
    now = now.replace(tzinfo=now.tzinfo or timezone.utc)
    now_epoch = int(now.timestamp())
    metadata = dict(operation.settlement_metadata or {})
    reservations, expired = _prune_expired_reservations(metadata, now_epoch=now_epoch)
    if expired:
        # Flush the operation first: its governance trigger takes user/account
        # state rows before bucket rows. Artifact accounting follows that same
        # order, avoiding a state-row/bucket-row inversion with admission.
        operation.settlement_metadata = metadata
        await db.flush()
        expired_by_bucket: dict[int, int] = {}
        for record in expired:
            expired_by_bucket[record["bucket"]] = (
                expired_by_bucket.get(record["bucket"], 0) + record["bytes"]
            )
        for bucket, refund in expired_by_bucket.items():
            await _adjust_artifact_bucket(
                db,
                user_id=user_id,
                bucket=bucket,
                delta=-refund,
            )
    operation_reserved = _reserved_bytes(metadata)
    remaining = policy.artifact_max_bytes_per_operation - operation_reserved
    requested = remaining if expected_bytes is None else expected_bytes
    if requested < 0 or requested > remaining:
        raise ArtifactRelayLimitError("operation_bytes")
    current_bucket = int(now.timestamp()) // 60
    # Relay time, not operation creation time, defines the rolling exposure.
    # The compact minute ledger stays bounded independently of retained
    # operation history, including artifacts produced more than a day ago.
    daily = (
        await db.execute(
            text(
                """
                SELECT COALESCE(SUM(artifact_relay_bytes), 0)
                  FROM external_governance_buckets
                 WHERE scope_type = 'user'
                   AND scope_id = :user_id
                   AND bucket_start >= date_trunc('minute', :since)
                """
            ),
            {
                "user_id": user_id,
                "since": now - timedelta(hours=24),
            },
        )
    ).scalar_one()
    if requested > policy.artifact_max_daily_bytes_per_user - int(daily or 0):
        raise ArtifactRelayLimitError("daily_bytes")
    metadata["artifact_relay_bytes_reserved"] = operation_reserved + requested
    token = str(uuid.uuid4())
    reservations[token] = {
        "bytes": requested,
        "bucket": current_bucket,
        "expires_at": now_epoch + int(float(reservation_ttl_seconds) + 0.999),
    }
    metadata["artifact_relay_reservations"] = reservations
    count = metadata.get("artifact_relay_requests", 0)
    metadata["artifact_relay_requests"] = (
        count + 1 if isinstance(count, int) and count >= 0 else 1
    )
    operation.settlement_metadata = metadata
    await db.flush()
    await _adjust_artifact_bucket(
        db,
        user_id=user_id,
        bucket=current_bucket,
        delta=requested,
    )
    return ArtifactByteReservation(
        operation_id=operation_id,
        user_id=user_id,
        token=token,
        max_bytes=requested,
    )


async def finalize_artifact_bytes(
    db: AsyncSession,
    reservation: ArtifactByteReservation,
    *,
    transferred_bytes: int,
) -> None:
    """Charge actual relayed bytes and release the unused reservation."""

    if (
        not isinstance(transferred_bytes, int)
        or isinstance(transferred_bytes, bool)
        or transferred_bytes < 0
    ):
        raise ArtifactRelayLimitError("unavailable")
    await db.execute(
        text("SELECT pg_advisory_xact_lock(hashtextextended(:key, 0))"),
        {"key": f"external-artifact-user:{reservation.user_id}"},
    )
    operation = await db.get(
        ExternalOperation, reservation.operation_id, with_for_update=True
    )
    if operation is None or operation.user_id != reservation.user_id:
        return
    metadata = dict(operation.settlement_metadata or {})
    reservations = _reservation_records(metadata)
    record = reservations.pop(reservation.token, None)
    if record is None:
        return
    reserved = min(record["bytes"], reservation.max_bytes)
    actual = min(transferred_bytes, reserved)
    refund = reserved - actual
    _refund_reservation(metadata, refund)
    metadata["artifact_relay_reservations"] = reservations
    prior_transferred = metadata.get("artifact_relay_bytes_transferred", 0)
    if not isinstance(prior_transferred, int) or isinstance(prior_transferred, bool):
        prior_transferred = 0
    metadata["artifact_relay_bytes_transferred"] = max(0, prior_transferred) + actual
    operation.settlement_metadata = metadata
    await db.flush()
    await _adjust_artifact_bucket(
        db,
        user_id=reservation.user_id,
        bucket=record["bucket"],
        delta=-refund,
    )


__all__ = [
    "ArtifactByteReservation",
    "ArtifactRelayLease",
    "ArtifactRelayLimitError",
    "acquire_artifact_relay",
    "finalize_artifact_bytes",
    "reserve_artifact_bytes",
]
