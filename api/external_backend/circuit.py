"""Distributed, provider-neutral admission circuit for failing accounts."""

from __future__ import annotations

from dataclasses import dataclass

from api.config import settings
from loguru import logger

from .metrics import circuit_events


_FAILURE_LUA = """
local count = redis.call('INCR', KEYS[1])
if count == 1 then
    redis.call('EXPIRE', KEYS[1], ARGV[2])
end
if count >= tonumber(ARGV[1]) then
    redis.call('SET', KEYS[2], ARGV[3], 'EX', ARGV[2])
    return 1
end
return 0
"""


def _key(account_id: str, suffix: str) -> str:
    return f"external_circuit:{account_id}:{suffix}"


@dataclass(frozen=True)
class _CircuitConfig:
    auth_failure_threshold: int
    service_failure_threshold: int
    cooldown_seconds: int


# Settings validates these ranges during process startup. Freeze the three values
# here so recording a successful upstream response never parses or validates
# operator configuration on the response path.
_CONFIG = _CircuitConfig(
    auth_failure_threshold=settings.external_circuit_auth_failure_threshold,
    service_failure_threshold=settings.external_circuit_service_failure_threshold,
    cooldown_seconds=settings.external_circuit_cooldown_seconds,
)


def _raw_redis_client():
    """Return the underlying client without SafeRedis's fail-open wrapper.

    Circuit state is itself optional and remains fail-open, but this module must
    observe backend errors so it can emit an error metric.  Calling methods on
    ``SafeRedis`` would turn connection failures into ordinary ``None`` values
    before they reach the exception handlers below.
    """

    return settings.redis_client.client


async def circuit_is_open(account_id: str) -> bool:
    try:
        value = await _raw_redis_client().get(_key(account_id, "open"))
    except Exception:
        # Redis-backed governance augments the durable account kill switch. A
        # telemetry outage must not turn every healthy upstream into a 503.
        circuit_events.labels(reason="backend", action="error").inc()
        logger.warning("External admission circuit state is temporarily unavailable")
        return False
    return bool(value)


async def record_upstream_result(
    account_id: str, *, status_code: int | None = None, transport_error: bool = False
) -> None:
    """Update admission state; accepted task polling never consults this circuit."""

    reason: str | None = None
    threshold = 0
    try:
        redis = _raw_redis_client()
        if status_code in {401, 403}:
            reason, threshold = "auth", _CONFIG.auth_failure_threshold
        elif transport_error or (status_code is not None and 500 <= status_code <= 599):
            reason, threshold = "service", _CONFIG.service_failure_threshold
        else:
            await redis.delete(
                _key(account_id, "auth"),
                _key(account_id, "service"),
                _key(account_id, "open"),
            )
            circuit_events.labels(reason="healthy", action="reset").inc()
            return
        opened = await redis.eval(
            _FAILURE_LUA,
            2,
            _key(account_id, reason),
            _key(account_id, "open"),
            threshold,
            _CONFIG.cooldown_seconds,
            reason,
        )
    except Exception:
        circuit_events.labels(reason="backend", action="error").inc()
        logger.warning("External admission circuit update is temporarily unavailable")
        return
    circuit_events.labels(reason=reason, action="failure").inc()
    if opened:
        circuit_events.labels(reason=reason, action="opened").inc()
        logger.error(
            "External account {} admission circuit opened for {} seconds after {} failures",
            account_id,
            _CONFIG.cooldown_seconds,
            reason,
        )


__all__ = ["circuit_is_open", "record_upstream_result"]
