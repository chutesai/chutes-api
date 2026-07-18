"""
Structured logging configuration for in-cluster deployments.

When LOG_FORMAT=json (set in-cluster via the chart), loguru's default sink is replaced
with a single stdout sink that emits one JSON object per line, AND every other logging
path is routed through loguru so it produces the same single-line JSON instead of
multi-line plaintext:
  - stdlib `logging` (libraries, uvicorn/gunicorn access + error) via an intercept handler
  - uncaught exceptions in the main thread, worker threads, and the asyncio event loop

The node Fluent Bit DaemonSet collects and enriches container stdout with the full
kubernetes.* metadata, so the application does not stamp any cluster metadata itself.

Each JSON line is FLAT (logger.bind(...) context lands at the top level) and includes a
top-level "text" field holding the rendered human-readable line (including any traceback),
so client-side `kubectl logs ... | jq -r .text` reproduces a clean, readable log.

Tracebacks are formatted with backtrace=True / diagnose=False: full call frames but NO
local-variable values, so no secrets/PII leak into stdout/OpenSearch and lines stay compact
enough to clear Fluent Bit's Skip_Long_Lines (~32 KB) limit.

When LOG_FORMAT is unset (local dev) this is a no-op -- loguru keeps its default
human-readable stderr sink.
"""

import os
import sys
import json
import asyncio
import logging
import threading
import traceback
from enum import Enum
from contextvars import ContextVar
from typing import Any

from loguru import logger

_HUMAN = "{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {name}:{function}:{line} - {message}"


# ---------------------------------------------------------------------------
# Ambient request identity (contextvars).
#
# Identity fields (ip, server_id, instance_id, miner_hotkey, ...) are added incrementally,
# each as soon as it is resolved -- the request dependency contributes ip/hotkey/path params,
# then deeper resolvers add server_id/instance_id -- and accumulate into a single per-request
# context that flows implicitly to every log line, including bare `logger.*` calls at deep
# raise sites. This is what lets code log a failure where it is detected (with the specific
# reason) without threading server/instance identity down through call arguments. Isolation
# is automatic: each request runs in its own asyncio Task with a copied context, so updates
# never leak between requests.
#
# The JSON sink merges these into each record below via setdefault, so an explicit
# logger.bind(...) / *_logger(...) field always wins over the ambient value.
# The default is None (not a shared {}): each bind writes a brand-new dict, so there is no
# process-wide mutable object any request could accidentally mutate in place and leak.
_log_context: ContextVar[dict | None] = ContextVar("log_context", default=None)


def update_log_context(**fields: Any) -> None:
    """Merge identity fields into the current request's ambient logging context.

    Additive and repeatable: call it at each point an identity becomes known (the request
    dependency, then deeper resolvers) and the fields accumulate for the rest of the request;
    passing a key again overwrites it. Drops None values. Every subsequent log line in this
    request/task carries the accumulated fields, so identity need not be threaded through
    call args.
    """
    vals = {k: v for k, v in fields.items() if v is not None}
    if not vals:
        return
    current = _log_context.get()
    # Always set a NEW dict -- never mutate the current one in place (it may be the copied
    # context a parent task shares with sibling tasks).
    _log_context.set({**current, **vals} if current else dict(vals))


class _InterceptHandler(logging.Handler):
    """Forward stdlib logging records to loguru, preserving level and exc_info."""

    def emit(self, record: logging.LogRecord) -> None:
        try:
            level = logger.level(record.levelname).name
        except ValueError:
            level = record.levelno
        # Walk out of this handler and the stdlib logging machinery so {name}/{function}/{line}
        # point at the originating call site rather than logging.callHandlers. Start at this
        # frame (depth 0) and skip every frame inside logging/__init__.py. (Canonical loguru
        # intercept recipe; the older logging.currentframe()+2 form lands on callHandlers.)
        frame, depth = sys._getframe(), 0
        while frame and (depth == 0 or frame.f_code.co_filename == logging.__file__):
            frame = frame.f_back
            depth += 1
        # Preserve the stdlib channel name (e.g. "uvicorn.access", "sqlalchemy.engine") --
        # the frame walk makes {name}/{function}/{line} point at the calling module, which
        # loses the logger the record actually came from. The sink surfaces _channel as the
        # top-level `logger` field.
        logger.bind(_channel=record.name).opt(depth=depth, exception=record.exc_info).log(
            level, record.getMessage()
        )


def _json_logging_enabled() -> bool:
    return os.environ.get("LOG_FORMAT", "").lower() == "json"


def configure_structured_logging() -> None:
    """Install the JSON stdout sink and route stdlib logging + synchronous exception hooks
    through it. Safe to call once at import time -- before any event loop exists -- which is how
    api.logging_bootstrap invokes it. The asyncio event-loop exception handler is NOT installed
    here (no running loop yet); call install_asyncio_exception_handler() from an async entrypoint
    for that. No-op unless LOG_FORMAT=json.
    """
    if not _json_logging_enabled():
        return

    def _sink(message):
        r = message.record
        extra = dict(r["extra"])
        # Merge ambient request identity (contextvars). setdefault so an explicit
        # logger.bind(...) field on the record always wins over the ambient value.
        ambient = _log_context.get()
        if ambient:
            for k, v in ambient.items():
                extra.setdefault(k, v)
        # Intercepted stdlib records carry the original channel name; native loguru calls
        # fall back to the module name ({name}).
        channel = extra.pop("_channel", None)
        payload = {
            "text": str(message).rstrip("\n"),  # human-readable line (+ trace) for `jq -r .text`
            "timestamp": r["time"].isoformat(),
            "level": r["level"].name,
            "logger": channel or r["name"],
            "function": r["function"],
            "line": r["line"],
            "message": r["message"],
            **extra,  # logger.bind(...) context, flat at top level
        }
        if r["exception"]:
            # Full formatted traceback including frames. r["exception"] is a
            # (type, value, traceback) namedtuple. diagnose=False (on the sink) keeps the
            # auto-appended trace in `text` free of local-variable values; this explicit
            # field never includes them either.
            payload["exception"] = "".join(traceback.format_exception(*r["exception"]))
        sys.stdout.write(json.dumps(payload, default=str) + "\n")

    level = os.environ.get("LOG_LEVEL", "INFO")
    logger.remove()
    logger.add(_sink, level=level, format=_HUMAN, backtrace=True, diagnose=False)

    # Route stdlib logging through loguru so everything is single-line JSON with the same
    # schema. basicConfig(force=True) installs the intercept handler on the ROOT logger, which
    # captures every logger that propagates -- the default -- so dependencies we've never heard
    # of are covered automatically. We then sweep already-registered loggers, drop any handlers
    # they installed, and force propagation, so libraries that manage their own output (uvicorn,
    # gunicorn, ...) also funnel into the single sink. No per-logger allow-list to keep in sync.
    logging.basicConfig(handlers=[_InterceptHandler()], level=0, force=True)
    for name in list(logging.root.manager.loggerDict):
        lg = logging.getLogger(name)
        lg.handlers = []
        lg.propagate = True

    # Uncaught exceptions -> one JSON line with a full trace, rather than raw multi-line text.
    def _excepthook(exc_type, exc_value, exc_tb):
        logger.opt(exception=(exc_type, exc_value, exc_tb)).critical("Uncaught exception")

    sys.excepthook = _excepthook

    def _thread_excepthook(args):
        logger.opt(exception=(args.exc_type, args.exc_value, args.exc_traceback)).critical(
            "Uncaught thread exception"
        )

    threading.excepthook = _thread_excepthook


def install_asyncio_exception_handler() -> None:
    """Route uncaught asyncio event-loop exceptions through the JSON sink. Must be called from
    within a running event loop (e.g. a FastAPI lifespan or async main), since there is no loop
    at import time -- this is the one piece configure_structured_logging() cannot do. Idempotent
    and a no-op unless LOG_FORMAT=json or when called with no running loop.
    """
    if not _json_logging_enabled():
        return

    def _aioloop_handler(loop, ctx):
        # ctx["exception"] may be absent for non-exception loop errors (e.g. transport msgs).
        logger.opt(exception=ctx.get("exception")).critical(
            "Uncaught asyncio exception: {}", ctx.get("message")
        )

    try:
        asyncio.get_running_loop().set_exception_handler(_aioloop_handler)
    except RuntimeError:
        pass


# ---------------------------------------------------------------------------
# Structured lifecycle logging helpers.
#
# The JSON sink above spreads every logger.bind(...) field at the top level of the emitted line, so
# binding chute_id/instance_id/etc. makes them first-class searchable fields in OpenSearch -- a
# single `chute_id:abc123` query reconstructs a chute's whole lifecycle across the chute, launch
# config, instance, bounty and autoscaler subsystems. These helpers build a logger bound with the
# canonical field names from an ORM object (dropping None and the sink's reserved keys) and tag the
# lifecycle stage with a bounded `event` vocabulary. They live here (not in a domain module) because
# they are cross-cutting and depend only on loguru, so foundational modules like ORM schemas can
# import them without an import cycle.
#
# loguru's bind() returns a NEW logger; it does not mutate the global one. Callers MUST use the
# returned object (`log = instance_logger(instance, ...); log.info(...)`).
# ---------------------------------------------------------------------------

# Keys the JSON sink populates itself (_sink above); binding any of these would collide, so we drop
# them from bound context.
_RESERVED_LOG_KEYS = frozenset(
    {"text", "timestamp", "level", "logger", "function", "line", "message", "exception", "_channel"}
)


class LogType(str, Enum):
    """Coarse `log_type` tag grouping the fine-grained `event` values below by subsystem/record type.

    These are loosely-coupled domains, not ordered stages (a user's chute can run on an image built by
    another user), so this is a plain classifier, not a pipeline position. Lets support narrow a
    user's activity to one kind of thing without enumerating every event, e.g.
    `user_id:abc123 AND log_type:image` for just their build activity vs. the full `user_id:abc123`.
    """

    IMAGE = "image"
    CHUTE = "chute"
    LAUNCH_CONFIG = "launch_config"
    INSTANCE = "instance"
    BOUNTY = "bounty"
    AUTOSCALER = "autoscaler"
    SERVER = "server"


class LifecycleEvent(str, Enum):
    """Bounded `event` tags for chute lifecycle log lines (searchable/alertable in OpenSearch)."""

    # Chute
    CHUTE_CREATE = "chute_create"
    CHUTE_UPDATE = "chute_update"
    CHUTE_DELETE = "chute_delete"
    CHUTE_WARMUP = "chute_warmup"
    # Launch config (deployment attempt)
    LAUNCH_CONFIG_CREATE = "launch_config_create"
    LAUNCH_CONFIG_RETRIEVE = "launch_config_retrieve"
    LAUNCH_CONFIG_VERIFY = "launch_config_verify"
    LAUNCH_CONFIG_FAIL = "launch_config_fail"
    # Instance
    INSTANCE_CREATE = "instance_create"
    INSTANCE_VERIFY = "instance_verify"
    INSTANCE_ACTIVATE = "instance_activate"
    INSTANCE_DISABLE = "instance_disable"
    INSTANCE_DELETE = "instance_delete"
    # Bounty
    BOUNTY_CREATE = "bounty_create"
    BOUNTY_CLAIM = "bounty_claim"
    # Image build (forge) -- keyed by image_id; a chute ties to its build via chute.image_id
    IMAGE_CREATE = "image_create"
    IMAGE_BUILD_START = "image_build_start"
    IMAGE_BUILD_COMPLETE = "image_build_complete"
    IMAGE_BUILD_FAIL = "image_build_fail"
    IMAGE_BUILD_STATUS = "image_build_status"
    # Autoscaler
    SCALE_UP = "scale_up"
    SCALE_DOWN = "scale_down"
    # Server / TEE attestation -- keyed by server_id (or ip pre-registration)
    SERVER_REGISTER = "server_register"
    SERVER_VERIFY = "server_verify"
    SERVER_DELETE = "server_delete"
    BOOT_ATTESTATION = "boot_attestation"
    RUNTIME_ATTESTATION = "runtime_attestation"
    ATTESTATION_FAIL = "attestation_fail"


# Map an event to its coarse log_type by name prefix, so bare bound_logger(event=...) call sites
# (which don't go through a typed helper) still get a `log_type` for free -- no per-site bookkeeping.
_EVENT_NAME_PREFIX_LOG_TYPE = (
    ("CHUTE_", LogType.CHUTE),
    ("IMAGE_", LogType.IMAGE),
    ("LAUNCH_CONFIG_", LogType.LAUNCH_CONFIG),
    ("INSTANCE_", LogType.INSTANCE),
    ("BOUNTY_", LogType.BOUNTY),
    ("SCALE_", LogType.AUTOSCALER),
    ("SERVER_", LogType.SERVER),
    ("BOOT_", LogType.SERVER),
    ("RUNTIME_", LogType.SERVER),
    ("ATTESTATION_", LogType.SERVER),
)


def _log_type_for_event(event: LifecycleEvent):
    for prefix, log_type in _EVENT_NAME_PREFIX_LOG_TYPE:
        if event.name.startswith(prefix):
            return log_type
    return None


def bound_logger(event: LifecycleEvent = None, log_type: LogType = None, **fields):
    """
    Return a loguru logger bound with the given lifecycle fields (dropping None + reserved keys).

    Binds a coarse `log_type` tag alongside the fine-grained `event`: an explicit `log_type` wins,
    else it is derived from the event's prefix. Use this for call sites that only have ids in scope;
    prefer the typed helpers below when an ORM object is available. The returned logger must be used
    for the subsequent log call.
    """
    ctx = {k: v for k, v in fields.items() if v is not None and k not in _RESERVED_LOG_KEYS}
    if event is not None:
        ctx["event"] = event.value if isinstance(event, Enum) else event
        if log_type is None and isinstance(event, LifecycleEvent):
            log_type = _log_type_for_event(event)
    if log_type is not None:
        ctx["log_type"] = log_type.value if isinstance(log_type, Enum) else log_type
    return logger.bind(**ctx)


def chute_logger(chute, *, event: LifecycleEvent = None, **extra):
    """Logger bound with a chute's canonical fields (incl. image_id, to tie a chute to its build)."""
    return bound_logger(
        event=event,
        log_type=LogType.CHUTE,
        chute_id=getattr(chute, "chute_id", None),
        user_id=getattr(chute, "user_id", None),
        version=getattr(chute, "version", None),
        image_id=getattr(chute, "image_id", None),
        **extra,
    )


def image_logger(image, *, event: LifecycleEvent = None, **extra):
    """Logger bound with an image's canonical fields (the build/forge lifecycle keys on image_id)."""
    return bound_logger(
        event=event,
        log_type=LogType.IMAGE,
        image_id=getattr(image, "image_id", None),
        user_id=getattr(image, "user_id", None),
        image_name=getattr(image, "name", None),
        image_tag=getattr(image, "tag", None),
        **extra,
    )


def instance_logger(instance, *, event: LifecycleEvent = None, **extra):
    """Logger bound with an instance's canonical fields."""
    return bound_logger(
        event=event,
        log_type=LogType.INSTANCE,
        instance_id=getattr(instance, "instance_id", None),
        chute_id=getattr(instance, "chute_id", None),
        config_id=getattr(instance, "config_id", None),
        miner_hotkey=getattr(instance, "miner_hotkey", None),
        miner_uid=getattr(instance, "miner_uid", None),
        version=getattr(instance, "version", None),
        **extra,
    )


def server_logger(server, *, event: LifecycleEvent = None, **extra):
    """Logger bound with a TEE server's canonical fields for attestation tracing.

    Accepts a Server ORM object (or any object exposing these attributes). Use for
    boot/runtime/registration attestation log lines so they can be traced in OpenSearch
    by server_id / ip / miner_hotkey.
    """
    return bound_logger(
        event=event,
        log_type=LogType.SERVER,
        server_id=getattr(server, "server_id", None),
        server_name=getattr(server, "name", None),
        ip=getattr(server, "ip", None),
        miner_hotkey=getattr(server, "miner_hotkey", None),
        version=getattr(server, "version", None),
        **extra,
    )


def launch_config_logger(launch_config, *, event: LifecycleEvent = None, **extra):
    """Logger bound with a launch config's canonical fields."""
    return bound_logger(
        event=event,
        log_type=LogType.LAUNCH_CONFIG,
        config_id=getattr(launch_config, "config_id", None),
        chute_id=getattr(launch_config, "chute_id", None),
        job_id=getattr(launch_config, "job_id", None),
        miner_hotkey=getattr(launch_config, "miner_hotkey", None),
        miner_uid=getattr(launch_config, "miner_uid", None),
        **extra,
    )
