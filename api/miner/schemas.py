"""
Pydantic schemas for miner API responses.
"""

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field

from api.server.schemas import Server


class MinerServerGpu(BaseModel):
    """GPU info within a miner server."""

    uuid: str
    gpu_identifier: str
    device_index: int
    verified_at: str | None = None
    verification_error: str | None = None


class MinerServer(BaseModel):
    """Server with nested GPU info for miner inventory."""

    server_id: str
    name: str
    ip: str
    is_tee: bool
    version: str | None = None
    maintenance_pending: bool = False
    created_at: str | None = None
    updated_at: str | None = None
    gpus: list[MinerServerGpu] = Field(default_factory=list)

    @classmethod
    def from_server(cls, server: Server) -> "MinerServer":
        """
        Build from a Server ORM instance. Server must have nodes (GPUs) joined.
        """
        return cls(
            server_id=server.server_id,
            name=server.name,
            ip=server.ip,
            is_tee=server.is_tee,
            version=server.version,
            maintenance_pending=server.in_maintenance,
            created_at=server.created_at.isoformat() if server.created_at else None,
            updated_at=server.updated_at.isoformat() if server.updated_at else None,
            gpus=[
                MinerServerGpu(
                    uuid=n.uuid,
                    gpu_identifier=n.gpu_identifier,
                    device_index=n.device_index,
                    verified_at=n.verified_at.isoformat() if n.verified_at else None,
                    verification_error=n.verification_error,
                )
                for n in server.nodes
            ],
        )


class MinerServersResponse(BaseModel):
    """Response containing the miner's server inventory."""

    servers: list[MinerServer] = Field(default_factory=list)

    @classmethod
    def from_servers(cls, servers: list[Server]) -> "MinerServersResponse":
        """
        Build from a list of Server ORM instances. Each server must have nodes (GPUs) joined.
        """
        return cls(servers=[MinerServer.from_server(s) for s in servers])


class MinerInventoryEntry(BaseModel):
    """A single GPU in the miner's inventory, with the chute it is currently serving."""

    gpu_id: str
    last_verified_at: datetime | None = None
    verification_error: str | None = None
    active: bool | None = None
    chute_id: str
    chute_name: str


class ActiveInstance(BaseModel):
    """An active, verified instance anywhere on the platform (used for preemption decisions)."""

    instance_id: str
    miner_hotkey: str
    chute_id: str
    activated_at: str | None = Field(None, description="ISO-8601 timestamp.")
    compute_multiplier: float


class MinerStatsEntry(BaseModel):
    """
    Instance-based stats for one miner over an interval.

    When the endpoint is called with ``per_chute=true`` the rows are broken down by
    chute and ``chute_id`` is populated; otherwise they are aggregated per miner and
    ``chute_id`` is absent.
    """

    miner_hotkey: str
    chute_id: str | None = None
    total_instances: int
    bounty_count: int
    compute_seconds: float
    compute_units: float
    invocation_count: int = Field(
        ...,
        description="Deprecated: mirrors total_instances, retained for compatibility.",
    )
    total_bounty: float | None = Field(
        None,
        description=(
            "Deprecated: mirrors bounty_count, retained for compatibility. "
            "Only present when per_chute is false."
        ),
    )


class MinerBountyEntry(BaseModel):
    """Deprecated per-miner bounty totals, retained for backwards compatibility."""

    miner_hotkey: str
    total_bounty: float


class MinerIntervalStats(BaseModel):
    """Stats for a single interval bucket."""

    instance_stats: list[MinerStatsEntry] = Field(default_factory=list)
    bounties: list[MinerBountyEntry] = Field(
        default_factory=list,
        description="Deprecated, and always empty when per_chute is true.",
    )
    compute_units: list[MinerStatsEntry] = Field(
        default_factory=list, description="Deprecated: identical to instance_stats."
    )


class MinerStatsResponse(BaseModel):
    """Miner stats bucketed by interval."""

    past_hour: MinerIntervalStats
    past_day: MinerIntervalStats
    all: MinerIntervalStats = Field(..., description="Trailing 7 days, despite the name.")


class MinerScoreRawValues(BaseModel):
    """Un-normalized scoring inputs for a single miner."""

    total_instances: float
    bounty_score: float
    instance_seconds: float
    instance_compute_units: float


class MinerScoresResponse(BaseModel):
    """
    Scoring data keyed by miner hotkey.

    When the ``hotkey`` query parameter is supplied both maps are filtered down to that
    single key, whose value is null if the miner has no score.
    """

    raw_values: dict[str, MinerScoreRawValues | None] = Field(default_factory=dict)
    final_scores: dict[str, float | None] = Field(
        default_factory=dict, description="Normalized to sum to 1.0 across all miners."
    )


class UniqueChuteHistoryEntry(BaseModel):
    """One hourly datapoint of a miner's chute inventory."""

    time: str = Field(..., description="ISO-8601 timestamp for the top of the hour.")
    count: int = Field(..., description="Unique chutes served at this timepoint.")
    total_count: int = Field(..., description="Total instances served at this timepoint.")


class MinerMetagraphNode(BaseModel):
    """A node registered on the subnet metagraph."""

    hotkey: str
    netuid: int
    checksum: str
    coldkey: str
    node_id: int | None = None
    incentive: float | None = None
    stake: float | None = None
    tao_stake: float | None = None
    alpha_stake: float | None = None
    trust: float | None = None
    vtrust: float | None = None
    last_updated: int | None = None
    ip: str | None = None
    ip_type: int | None = None
    port: int | None = None
    protocol: int | None = None
    real_host: str | None = None
    real_port: int | None = None
    synced_at: datetime | None = None
    blacklist_reason: str | None = None


class MinerChuteNodeSelector(BaseModel):
    """
    Node selector for a chute, as served to miners.

    ``compute_multiplier`` and ``supported_gpus`` are derived rather than stored, and
    are merged in before the chute is returned.
    """

    gpu_count: int | None = None
    min_vram_gb_per_gpu: int | None = None
    max_hourly_price_per_gpu: float | None = None
    include: list[str] | None = None
    exclude: list[str] | None = None
    dynamic: bool | None = None
    compute_multiplier: float | None = None
    supported_gpus: list[str] = Field(default_factory=list)


class MinerChute(BaseModel):
    """
    A chute as served to miners.

    Note that ``code`` is deliberately a placeholder here -- miner inventory never
    exposes chute source, which runtimes obtain via their launch config instead.
    """

    chute_id: str
    user_id: str
    name: str | None = None
    tagline: str | None = None
    readme: str | None = None
    tool_description: str | None = None
    image_id: str | None = None
    image: str = Field(..., description="Fully qualified image ref, e.g. user/name:tag.")
    logo_id: str | None = None
    public: bool | None = None
    standard_template: str | None = None
    cords: list[dict[str, Any]] = Field(default_factory=list)
    jobs: list[dict[str, Any]] | None = None
    node_selector: MinerChuteNodeSelector
    slug: str | None = None
    code: str = Field(..., description="Placeholder only; never the real chute source.")
    filename: str
    ref_str: str
    version: str | None = None
    concurrency: int | None = None
    boost: float | None = None
    chutes_version: str | None = None
    revision: str | None = None
    openrouter: bool | None = None
    discount: float | None = None
    created_at: datetime | None = None
    updated_at: datetime | None = None
    max_instances: int | None = None
    scaling_threshold: float | None = None
    shutdown_after_seconds: int | None = None
    allow_external_egress: bool | None = None
    encrypted_fs: bool | None = None
    tee: bool | None = None
    lock_modules: bool | None = None
    immutable: bool | None = None
    disabled: bool | None = None
    invocation_count: int | None = None
    supported_gpus: list[str] = Field(default_factory=list)
    preemptible: bool
    effective_compute_multiplier: float = Field(
        ...,
        description="Multiplier a miner would earn by activating an instance right now.",
    )
    compute_multiplier_factors: dict[str, float] = Field(
        default_factory=dict,
        description="Breakdown of the bonuses composing effective_compute_multiplier.",
    )
    bounty: int | None = Field(None, description="Current bounty amount, if any.")
