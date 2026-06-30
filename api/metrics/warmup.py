"""
Track chute warmup demand and time-to-hot in Prometheus.

The "demand signal" is the moment something asks for a chute to become hot -- either an
explicit /warmup request from a user, or an automatically created bounty for a public chute.
The "time-to-hot" is the latency from that demand to the first instance going active.

These are metrics (time-series trends), so they live in Prometheus/GMP rather than the DB --
GMP retains them for 24 months with automatic rollup. The request->hot correlation timestamp is
held transiently in Redis (the bounty path already stores its creation time there), never in
Postgres.
"""

import time
from enum import Enum

from loguru import logger
from prometheus_client import Counter, Histogram


class WarmupTrigger(str, Enum):
    """What caused a warmup demand. Used as the `trigger` label on the warmup metrics."""

    EXPLICIT = "explicit_warmup"  # a user hit GET /warmup/{chute}
    BOUNTY = "bounty"  # a bounty was created for a public chute


warmup_requests = Counter(
    "chute_warmup_requests_total",
    "Total warmup demand signals (explicit /warmup calls and bounty creations)",
    ["chute_id", "trigger"],
)
warmup_seconds = Histogram(
    "chute_warmup_seconds",
    "Time from warmup demand to the first instance going hot, in seconds",
    ["chute_id", "trigger"],
    # Warmups routinely span seconds to tens of minutes (cold miners pulling images),
    # so use buckets that cover sub-second through a couple hours.
    buckets=(5, 15, 30, 60, 120, 300, 600, 900, 1800, 3600, 7200),
)


def track_warmup_request(chute_id: str, trigger: WarmupTrigger) -> None:
    """
    Record a warmup demand signal.
    """
    warmup_requests.labels(chute_id=chute_id, trigger=trigger.value).inc()


def track_warmup_seconds(chute_id: str, trigger: WarmupTrigger, seconds: float) -> None:
    """
    Record the time from warmup demand to an instance going hot.
    """
    if seconds is None or seconds < 0:
        return
    warmup_seconds.labels(chute_id=chute_id, trigger=trigger.value).observe(seconds)


def track_warmup_seconds_since(chute_id: str, trigger: WarmupTrigger, requested_at_raw) -> None:
    """
    Observe request->hot for a warmup whose start time was stamped in Redis.

    `requested_at_raw` is the value read back from Redis, or None when nothing was stamped (no-op).
    float() parses both str and bytes, so we don't care about the client's decode_responses setting
    (matching how the rest of the codebase consumes redis values). A missing/garbage value is logged
    and skipped, never raised, so a metrics problem can't break the activation path.
    """
    if requested_at_raw is None:
        return
    try:
        seconds = time.time() - float(requested_at_raw)
    except (TypeError, ValueError) as exc:
        logger.warning(
            f"warmup: could not parse requested_at {requested_at_raw!r} for chute_id={chute_id}: {exc}"
        )
        return
    track_warmup_seconds(chute_id, trigger, seconds)
