"""
Structured logging configuration for in-cluster deployments.

Adds a JSON file sink for Fluent Bit collection alongside the default
human-readable stderr sink. Safe to call in all entrypoints -- the JSON
sink only activates when the /var/log/app emptyDir volume is mounted.
"""

import os
from loguru import logger


def configure_structured_logging() -> None:
    structured_log_path = os.environ.get("STRUCTURED_LOG_PATH", "/var/log/app/structured.log")
    log_dir = os.path.dirname(structured_log_path)
    if os.path.isdir(log_dir):
        logger.add(
            structured_log_path,
            serialize=True,
            rotation="100 MB",
            retention="1 day",
            compression="gz",
            enqueue=True,
        )
