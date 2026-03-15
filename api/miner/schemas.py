"""
Pydantic schemas for miner API responses.
"""

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
