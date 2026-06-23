"""
Structured logging configuration for in-cluster deployments.

When LOG_FORMAT=json (set in-cluster via the chart), loguru's default sink is
replaced with a single stdout sink that emits one JSON object per line. The node
Fluent Bit DaemonSet already collects and enriches container stdout with the full
kubernetes.* metadata, so the application does not stamp any cluster metadata itself.

Each JSON line is FLAT (bound fields from logger.bind(...) land at the top level)
and includes a top-level "text" field holding the rendered human-readable line, so
client-side `kubectl logs ... | jq -r .text` reproduces a clean log.

When LOG_FORMAT is unset (local dev) this is a no-op -- loguru keeps its default
human-readable stderr sink.
"""

import os
import sys
import json
from loguru import logger

_HUMAN = "{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {name}:{function}:{line} - {message}"


def configure_structured_logging() -> None:
    if os.environ.get("LOG_FORMAT", "").lower() != "json":
        return

    def _sink(message):
        r = message.record
        payload = {
            "text": str(message).rstrip("\n"),  # human-readable line for `jq -r .text`
            "timestamp": r["time"].isoformat(),
            "level": r["level"].name,
            "logger": r["name"],
            "function": r["function"],
            "line": r["line"],
            "message": r["message"],
            **r["extra"],  # logger.bind(...) context, flat at top level
        }
        if r["exception"]:
            payload["exception"] = str(r["exception"])
        sys.stdout.write(json.dumps(payload, default=str) + "\n")

    logger.remove()
    logger.add(_sink, level=os.environ.get("LOG_LEVEL", "INFO"), format=_HUMAN)
