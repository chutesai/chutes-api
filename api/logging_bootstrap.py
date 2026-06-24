"""
Import-for-side-effect: configure structured logging as early as possible.

Long-running entrypoints import this module FIRST -- before any module that logs at import
time (e.g. api.config._load_tee_measurements, api.instance.router) -- so the entire stdout
stream is single-line JSON from the very first app log line when LOG_FORMAT=json.

It is done as an import rather than a function call so it can sit ahead of the other imports
without tripping flake8/ruff's E402 (module-level import not at top of file). No-op when
LOG_FORMAT is unset (local dev keeps human-readable stderr).

Note: output emitted before the Python process starts -- e.g. `uv`/pip build lines from
`uv run` -- cannot be captured here; that requires running a prebuilt env / invoking python
directly rather than through `uv run`.
"""

from api.log import configure_structured_logging

configure_structured_logging()
