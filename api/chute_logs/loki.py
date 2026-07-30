"""Minimal async Loki client for the chute log store.

The validator pushes chute log lines straight to a dedicated in-namespace Loki
(``LOKI_URL``) via its HTTP push API and reads them back with ``query_range``.
This traffic must NEVER go through the API's own stdout — that would leak chute
logs into the ops monitoring store via Fluent Bit (see the spec). Loki is not
client-reachable; only this client and the internal Grafana talk to it.

Label discipline: labels are low-cardinality only (``app``,
``stream``). High-cardinality identifiers (config_id, chute_id, user_id, …) ride
inside the JSON log line and are filtered via LogQL ``| json | field="…"``.

Connection safety: ``LokiClient`` is a process-wide singleton — ``LokiClient()``
always returns the same instance, which owns the ONE bounded ``httpx.AsyncClient``
(one connection pool) for the process. Enforcing the singleton in the class (not a
convention at the call site) means no caller can accidentally construct a second
pool and blow up the file-descriptor count under load; the pool is closed once, on
shutdown, via :meth:`LokiClient.aclose`.
"""

import json
import threading
from typing import Dict, List, Optional, Tuple

import httpx
from loguru import logger

from api.config import settings

# App label shared by every chute-log stream; the read/grafana matchers anchor on it.
APP_LABEL = "chute-log-shipper"

_PUSH_PATH = "/loki/api/v1/push"
_QUERY_RANGE_PATH = "/loki/api/v1/query_range"

# Bound the single pool so a stuck/slow Loki can never exhaust the API's FDs.
_DEFAULT_LIMITS = httpx.Limits(max_connections=20, max_keepalive_connections=10)


class LokiClient:
    """Process-wide singleton owning the ONE bounded httpx pool for Loki.

    Call it directly — ``LokiClient()`` always returns the same instance, so there
    is structurally never more than one connection pool no matter how many call
    sites exist. Construction is lazy: the pool is opened on first use (only reached
    behind an ``if settings.loki_url`` guard) and closed once, on shutdown, via
    :meth:`aclose`.

    Tests: set ``LokiClient._instance`` to a fake (e.g. ``monkeypatch.setattr``) to
    inject behaviour without opening a real pool. The default ``None`` is restored
    between tests, so no client state leaks across them.
    """

    _instance: Optional["LokiClient"] = None
    _lock = threading.Lock()

    def __new__(cls) -> "LokiClient":
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    self = super().__new__(cls)
                    self._connect()
                    cls._instance = self
        return cls._instance

    def _connect(self) -> None:
        """Open the single connection pool. Runs exactly once, from ``__new__``."""
        self._tenant_id = settings.loki_tenant_id
        self._http = httpx.AsyncClient(
            base_url=settings.loki_url.rstrip("/"),
            timeout=httpx.Timeout(settings.loki_timeout_seconds),
            limits=_DEFAULT_LIMITS,
        )

    def _headers(self) -> Dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self._tenant_id:
            headers["X-Scope-OrgID"] = self._tenant_id
        return headers

    async def push(self, streams: List[Dict]) -> None:
        """Push one or more Loki streams. Raises httpx.HTTPError on transport failure."""
        resp = await self._http.post(
            _PUSH_PATH, headers=self._headers(), content=json.dumps({"streams": streams})
        )
        if resp.status_code >= 300:
            # Surface the body so a 400 (bad labels / out-of-order) is diagnosable.
            raise httpx.HTTPStatusError(
                f"Loki push failed ({resp.status_code}): {resp.text[:500]}",
                request=resp.request,
                response=resp,
            )

    async def query_range(
        self,
        logql: str,
        start_ns: int,
        end_ns: int,
        limit: int = 5000,
    ) -> List[Tuple[str, Dict]]:
        """Run a LogQL query_range, returning ``[(ts_ns, parsed_line_json), …]`` sorted ascending.

        Each stored line is a JSON object (see service._push_lines), so we json-decode it;
        a line that does not decode is returned as ``{"log": <raw>}``.
        """
        params = {
            "query": logql,
            "start": str(start_ns),
            "end": str(end_ns),
            "limit": str(limit),
            "direction": "forward",
        }
        resp = await self._http.get(_QUERY_RANGE_PATH, headers=self._headers(), params=params)
        resp.raise_for_status()
        payload = resp.json()
        out: List[Tuple[str, Dict]] = []
        for stream in payload.get("data", {}).get("result", []):
            for ts_ns, line in stream.get("values", []):
                try:
                    parsed = json.loads(line)
                    if not isinstance(parsed, dict):
                        parsed = {"log": line}
                except (json.JSONDecodeError, TypeError):
                    parsed = {"log": line}
                out.append((ts_ns, parsed))
        out.sort(key=lambda item: int(item[0]))
        return out

    @classmethod
    async def aclose(cls) -> None:  # pragma: no cover - lifecycle helper
        """Close the shared pool on shutdown (no-op if never created)."""
        inst = cls._instance
        if inst is not None and getattr(inst, "_http", None) is not None:
            try:
                await inst._http.aclose()
            except Exception as exc:
                logger.warning(f"error closing loki client: {exc}")
        cls._instance = None
