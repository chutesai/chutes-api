"""Shared lifecycle deadlines and crash-safe usage checkpoints for sessions."""

from __future__ import annotations

import asyncio
import copy
import inspect
import math
from datetime import datetime, timedelta
from typing import Any, Awaitable, Callable, Mapping

from loguru import logger


SESSION_RECOVERY_GRACE_SECONDS = 60.0
USAGE_CHECKPOINT_INTERVAL_SECONDS = 5.0


class UsageCheckpointLoop:
    """Persist changing session usage on a bounded cadence.

    Streaming observations can arrive once per token or message, so writes are driven
    by one fixed timer instead of the observation path. The final settlement remains
    authoritative; these snapshots exist for a different worker to recover useful
    partial usage after an abrupt process loss.
    """

    def __init__(
        self,
        *,
        operation_id: str,
        read_usage: Callable[[], Mapping[str, Any]],
        persist_usage: Callable[[Mapping[str, Any]], Awaitable[None] | None],
        initial_usage: Mapping[str, Any],
        interval_seconds: float = USAGE_CHECKPOINT_INTERVAL_SECONDS,
        always_persist: bool = False,
    ) -> None:
        if (
            isinstance(interval_seconds, bool)
            or not isinstance(interval_seconds, (int, float))
            or not math.isfinite(float(interval_seconds))
            or interval_seconds <= 0
        ):
            raise ValueError("usage checkpoint interval must be positive and finite")
        self._operation_id = operation_id
        self._read_usage = read_usage
        self._persist_usage = persist_usage
        self._interval_seconds = float(interval_seconds)
        self._always_persist = bool(always_persist)
        self._last_usage = copy.deepcopy(dict(initial_usage))
        self._stop = asyncio.Event()
        self._task: asyncio.Task[None] | None = None

    def start(self) -> asyncio.Task[None]:
        """Start the single periodic writer for this session."""

        if self._task is None:
            self._task = asyncio.create_task(
                self._run(),
                name=f"external-usage-checkpoint:{self._operation_id}",
            )
        return self._task

    async def checkpoint_now(self) -> bool:
        """Persist a changed snapshot, returning whether a write was issued."""

        current = copy.deepcopy(dict(self._read_usage()))
        if current == self._last_usage and not self._always_persist:
            return False
        result = self._persist_usage(current)
        if inspect.isawaitable(result):
            await result
        self._last_usage = current
        return True

    async def _run(self) -> None:
        while not self._stop.is_set():
            try:
                await asyncio.wait_for(
                    self._stop.wait(), timeout=self._interval_seconds
                )
            except asyncio.TimeoutError:
                try:
                    await self.checkpoint_now()
                except asyncio.CancelledError:
                    raise
                except Exception:
                    # A checkpoint is a crash-recovery aid. A transient persistence
                    # failure must not tear down an otherwise healthy paid session;
                    # the next timer tick retries and final settlement stays canonical.
                    logger.exception(
                        "Failed to checkpoint usage for external operation {}",
                        self._operation_id,
                    )

    async def stop(self) -> None:
        """Stop the timer without a final write; settlement owns the final value."""

        self._stop.set()
        task = self._task
        if task is None:
            return
        try:
            await task
        finally:
            self._task = None


class UsageBudgetMonitor:
    """Periodically terminate a metered session once observed cost exhausts a cap."""

    def __init__(
        self,
        *,
        operation_id: str,
        read_usage: Callable[[], Mapping[str, Any]],
        check_usage: Callable[[Mapping[str, Any]], Awaitable[tuple[bool, str | None]]],
        on_exceeded: Callable[[str], Awaitable[None] | None],
        interval_seconds: float = USAGE_CHECKPOINT_INTERVAL_SECONDS,
        max_check_failures: int = 3,
    ) -> None:
        if (
            isinstance(interval_seconds, bool)
            or not isinstance(interval_seconds, (int, float))
            or not math.isfinite(float(interval_seconds))
            or interval_seconds <= 0
        ):
            raise ValueError("usage budget interval must be positive and finite")
        if (
            isinstance(max_check_failures, bool)
            or not isinstance(max_check_failures, int)
            or not 1 <= max_check_failures <= 100
        ):
            raise ValueError("usage budget failure limit must be between 1 and 100")
        self._operation_id = operation_id
        self._read_usage = read_usage
        self._check_usage = check_usage
        self._on_exceeded = on_exceeded
        self._interval_seconds = float(interval_seconds)
        self._max_check_failures = max_check_failures
        self._stop = asyncio.Event()
        self._task: asyncio.Task[None] | None = None
        self.exceeded = False
        self.reason: str | None = None

    def start(self) -> asyncio.Task[None]:
        if self._task is None:
            self._task = asyncio.create_task(
                self._run(), name=f"external-budget-monitor:{self._operation_id}"
            )
        return self._task

    async def _run(self) -> None:
        consecutive_failures = 0
        while not self._stop.is_set():
            try:
                await asyncio.wait_for(
                    self._stop.wait(), timeout=self._interval_seconds
                )
                continue
            except asyncio.TimeoutError:
                pass
            try:
                allowed, reason = await self._check_usage(
                    copy.deepcopy(dict(self._read_usage()))
                )
                consecutive_failures = 0
                if allowed:
                    continue
                self.exceeded = True
                self.reason = reason or "budget"
                result = self._on_exceeded(self.reason)
                if inspect.isawaitable(result):
                    await result
                return
            except asyncio.CancelledError:
                raise
            except Exception:
                consecutive_failures += 1
                logger.exception(
                    "Failed to enforce usage budget for external operation {}",
                    self._operation_id,
                )
                if consecutive_failures < self._max_check_failures:
                    continue
                self.exceeded = True
                self.reason = "budget_unavailable"
                result = self._on_exceeded(self.reason)
                if inspect.isawaitable(result):
                    await result
                return

    async def stop(self) -> None:
        self._stop.set()
        task = self._task
        if task is None or task is asyncio.current_task():
            return
        try:
            await task
        finally:
            self._task = None


def session_recovery_deadline(
    started_at: datetime,
    maximum_duration_seconds: float,
) -> datetime:
    """Return a conservative reaping deadline after the transport's own limit."""

    if (
        isinstance(maximum_duration_seconds, bool)
        or not isinstance(maximum_duration_seconds, (int, float))
        or not math.isfinite(float(maximum_duration_seconds))
        or maximum_duration_seconds <= 0
    ):
        raise ValueError("session maximum duration must be a positive finite number")
    return started_at + timedelta(
        seconds=float(maximum_duration_seconds) + SESSION_RECOVERY_GRACE_SECONDS
    )


__all__ = [
    "SESSION_RECOVERY_GRACE_SECONDS",
    "USAGE_CHECKPOINT_INTERVAL_SECONDS",
    "UsageCheckpointLoop",
    "UsageBudgetMonitor",
    "session_recovery_deadline",
]
