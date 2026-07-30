"""Wire + internal models for the chute log shipper (validator side).

The guest wire contract is deliberately minimal (sek8s spec, frozen 2026-07-27,
+ 2026-07-29 deltas): the body carries only ``deployment_id`` plus the log lines.
Everything security-relevant (config_id, chute_id, miner_hotkey, user_id,
server_ip) is derived server-side from the request path + verified mTLS leaf +
proxy — never self-asserted by the guest.
"""

from typing import List

from pydantic import BaseModel, Field


class LogLine(BaseModel):
    """One CRI log line as shipped by the guest."""

    ts: str = Field(
        ..., description="RFC3339 nanosecond timestamp, e.g. 2026-07-27T00:00:00.123456789Z"
    )
    stream: str = Field("stdout", description="stdout | stderr")
    log: str = Field("", description="The log line content")


class LogShipmentArgs(BaseModel):
    """Request body for POST /instances/launch_config/{config_id}/logs.

    ``deployment_id`` is the sole top-level metadata field: it is non-security
    correlation data (Grafana filtering only) that the validator cannot derive
    from ``config_id`` in the pre-registration window, so the guest supplies it
    from the ``chutes/deployment-id`` pod label. It is never used for auth.
    """

    deployment_id: str = Field(
        "",
        description="From the chutes/deployment-id pod label; empty if absent. Never used for auth.",
    )
    logs: List[LogLine] = Field(default_factory=list)


class StoredLogLine(BaseModel):
    """A log line as returned by the owner / miner-CLI read paths."""

    ts: str
    stream: str
    log: str
