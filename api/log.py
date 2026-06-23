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

from loguru import logger

_HUMAN = "{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {name}:{function}:{line} - {message}"


class _InterceptHandler(logging.Handler):
    """Forward stdlib logging records to loguru, preserving level and exc_info."""

    def emit(self, record: logging.LogRecord) -> None:
        try:
            level = logger.level(record.levelname).name
        except ValueError:
            level = record.levelno
        # Walk out of the logging machinery so {name}/{function}/{line} point at the
        # real caller rather than this handler.
        frame, depth = logging.currentframe(), 2
        while frame and frame.f_code.co_filename == logging.__file__:
            frame = frame.f_back
            depth += 1
        # Preserve the stdlib channel name (e.g. "uvicorn.access", "sqlalchemy.engine") --
        # the frame walk makes {name}/{function}/{line} point at the calling module, which
        # loses the logger the record actually came from. The sink surfaces _channel as the
        # top-level `logger` field.
        logger.bind(_channel=record.name).opt(depth=depth, exception=record.exc_info).log(
            level, record.getMessage()
        )


def configure_structured_logging() -> None:
    if os.environ.get("LOG_FORMAT", "").lower() != "json":
        return

    def _sink(message):
        r = message.record
        extra = dict(r["extra"])
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

    def _aioloop_handler(loop, ctx):
        # ctx["exception"] may be absent for non-exception loop errors (e.g. transport msgs).
        logger.opt(exception=ctx.get("exception")).critical(
            "Uncaught asyncio exception: {}", ctx.get("message")
        )

    # All entrypoints call this from within an async context, so there's a running loop to
    # attach to. The guard keeps it safe for any synchronous caller (the main-thread and
    # thread excepthooks still cover those).
    try:
        asyncio.get_running_loop().set_exception_handler(_aioloop_handler)
    except RuntimeError:
        pass
