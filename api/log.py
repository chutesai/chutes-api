"""
Structured logging configuration for in-cluster deployments.

Adds a JSON file sink for Fluent Bit collection alongside the default
human-readable stderr sink. Safe to call in all entrypoints -- the JSON
sink only activates when the /var/log/app emptyDir volume is mounted.

Rotation/retention are tunable per-deployment via env vars:
  STRUCTURED_LOG_ROTATION  default "100 MB"  (e.g. "500 MB" for high-traffic services)
  STRUCTURED_LOG_RETENTION default 3          (number of rotated files to keep on disk)

Count-based retention is intentional: time-based retention ("1 day") can accumulate
hundreds of rotated files for high-traffic services, exhausting ephemeral storage.
"""

import os
from loguru import logger


def configure_structured_logging() -> None:
    structured_log_path = os.environ.get("STRUCTURED_LOG_PATH", "/var/log/app/structured.log")
    log_dir = os.path.dirname(structured_log_path)
    if not os.path.isdir(log_dir):
        return
    rotation = os.environ.get("STRUCTURED_LOG_ROTATION", "100 MB")
    retention = int(os.environ.get("STRUCTURED_LOG_RETENTION", "10"))
    logger.add(
        structured_log_path,
        serialize=True,
        rotation=rotation,
        retention=retention,
        compression="gz",
        enqueue=True,
    )
