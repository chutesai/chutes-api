"""
ORM definitions for instances (deployments of chutes and/or inventory announcements).
"""

import secrets
from loguru import logger
from pydantic import BaseModel, Field, constr
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.sql import func
from sqlalchemy.orm import relationship, deferred
from sqlalchemy import (
    Column,
    String,
    DateTime,
    Boolean,
    ForeignKey,
    Integer,
    Index,
    Table,
    Numeric,
    Double,
    UniqueConstraint,
    event,
)
from typing import Optional
from api.database import Base, generate_uuid
from api.metrics import launch_config as launch_config_metrics

# Association table.
instance_nodes = Table(
    "instance_nodes",
    Base.metadata,
    Column("instance_id", String, ForeignKey("instances.instance_id", ondelete="CASCADE")),
    Column("node_id", String, ForeignKey("nodes.uuid", ondelete="NO ACTION")),
    UniqueConstraint("instance_id", "node_id", name="uq_instance_node"),
    UniqueConstraint("node_id", name="uq_inode"),
)


class InstanceArgs(BaseModel):
    node_ids: list[str]
    host: str
    port: int


class ActivateArgs(BaseModel):
    active: bool


class PortMap(BaseModel):
    internal_port: int = Field(..., ge=22, le=65535)
    external_port: int = Field(..., ge=22, le=65535)
    proto: str = constr(pattern=r"^(tcp|udp|http)$")
    default: Optional[bool] = Field(default=None)


class LaunchConfigArgs(BaseModel):
    gpus: list[dict]
    host: str
    port_mappings: list[PortMap]
    fsv: Optional[str] = None
    egress: Optional[bool] = None
    lock_modules: Optional[bool] = None
    netnanny_hash: Optional[str] = None
    run_path: Optional[str] = None
    py_dirs: Optional[list[str]] = None
    rint_commitment: Optional[str] = None
    rint_nonce: Optional[str] = None
    rint_pubkey: Optional[str] = None
    tls_cert: Optional[str] = None
    tls_cert_sig: Optional[str] = None
    tls_ca_cert: Optional[str] = None
    tls_client_cert: Optional[str] = None
    tls_client_key: Optional[str] = None
    tls_client_key_password: Optional[str] = None
    e2e_pubkey: Optional[str] = None
    cllmv_session_init: Optional[str] = None
    env: str
    code: Optional[str] = None
    run_code: Optional[str] = None
    inspecto: Optional[str] = None


class TeeLaunchConfigArgs(LaunchConfigArgs):
    deployment_id: str


class LegacyTeeLaunchConfigArgs(LaunchConfigArgs):
    gpu_evidence: list[dict]


class Instance(Base):
    __tablename__ = "instances"
    instance_id = Column(String, primary_key=True, default=generate_uuid)
    host = Column(String, nullable=False)
    port = Column(Integer, nullable=False)
    chute_id = Column(String, ForeignKey("chutes.chute_id", ondelete="CASCADE"), nullable=False)
    version = Column(String, nullable=False)
    miner_uid = Column(Integer, nullable=False)
    miner_hotkey = Column(String, nullable=False)
    miner_coldkey = Column(String, nullable=False)
    region = Column(String)
    active = Column(Boolean, default=False)
    verified = Column(Boolean, default=False)
    last_queried_at = Column(DateTime(timezone=True))
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True))
    activated_at = Column(DateTime(timezone=True), nullable=True)
    last_verified_at = Column(DateTime(timezone=True))
    stop_billing_at = Column(DateTime, nullable=True)
    billed_to = Column(String, ForeignKey("users.user_id", ondelete="CASCADE"), nullable=True)
    verification_error = Column(String, nullable=True)
    consecutive_failures = Column(Integer, default=0)
    chutes_version = Column(String, nullable=True)
    symmetric_key = Column(String, default=lambda: secrets.token_bytes(16).hex())
    config_id = Column(
        String,
        ForeignKey("launch_configs.config_id", ondelete="SET NULL"),
        nullable=True,
        unique=True,
    )
    # Chute deployment ID; set when instance claims TEE launch config and is verified.
    deployment_id = Column(
        String,
        nullable=True,
    )
    cacert = Column(String, nullable=True)
    port_mappings = Column(JSONB, nullable=True)
    inspecto = Column(String, nullable=True)
    env_creation = deferred(Column(JSONB, nullable=True))
    bounty = Column(Boolean, default=False)

    # Runtime integrity (runint) commitment and nonce
    rint_commitment = Column(String, nullable=True)
    rint_nonce = Column(String, nullable=True)
    # ECDH session encryption: miner's pubkey and derived session key
    rint_pubkey = Column(String, nullable=True)
    rint_session_key = Column(String, nullable=True)

    # Flexible extra data (aegis v4 fields, future extensions)
    extra = Column(JSONB, nullable=True)

    # Hourly rate charged to customer, which may differ from the hourly rate of the actual
    # GPUs used for this instance due to node selector. For example, if a chute supports
    # both H100 and A100, the user is only charged the A100 rate since the miners *could*
    # have run it on A100s, regardless of whether or not they did so.
    hourly_rate = Column(Double, nullable=True)
    compute_multiplier = Column(Double, nullable=True)

    nodes = relationship("Node", secondary=instance_nodes, back_populates="instance")
    chute = relationship("Chute", back_populates="instances")
    job = relationship("Job", back_populates="instance", uselist=False)
    config = relationship("LaunchConfig", back_populates="instance", lazy="joined")
    billed_user = relationship("User", back_populates="instances")

    __table_args__ = (
        Index(
            "idx_chute_active_lastq",
            "chute_id",
            "active",
            "verified",
            "last_queried_at",
        ),
        UniqueConstraint("host", "port", name="unique_host_port"),
    )


class LaunchConfig(Base):
    __tablename__ = "launch_configs"
    config_id = Column(String, primary_key=True, default=generate_uuid)
    seed = Column(Numeric, nullable=False)
    env_key = Column(String, nullable=False)
    chute_id = Column(String, ForeignKey("chutes.chute_id", ondelete="CASCADE"), nullable=False)
    job_id = Column(
        String,
        ForeignKey("jobs.job_id", ondelete="CASCADE"),
        nullable=True,
    )
    host = Column(String, nullable=True)
    port = Column(Integer, nullable=True)
    env_type = Column(String, nullable=True)
    miner_uid = Column(Integer, nullable=False)
    miner_hotkey = Column(String, nullable=False)
    miner_coldkey = Column(String, nullable=False)
    created_at = Column(DateTime, server_default=func.now())
    retrieved_at = Column(DateTime, nullable=True)
    verified_at = Column(DateTime, nullable=True)
    failed_at = Column(DateTime, nullable=True)
    verification_error = Column(String, nullable=True)
    nonce = Column(String, nullable=True)

    instance = relationship("Instance", back_populates="config", uselist=False, lazy="joined")
    job = relationship("Job", back_populates="launch_config")

    __table_args__ = (UniqueConstraint("job_id", name="uq_job_launch_config"),)


# ---------------------------------------------------------------------------
# Deployment-funnel metrics, driven entirely by ORM events on LaunchConfig so the Prometheus
# counters stay in lock-step with the table without any call-site instrumentation to remember.
# Each lifecycle phase corresponds to a column transition:
#   created        -> after_insert
#   retrieved      -> retrieved_at set (NULL -> value)
#   verified       -> verified_at set (NULL -> value)
#   failed         -> verification_error set (-> bucketed reason)
# These 'set' events fire only on Python-level assignment, NOT on ORM load/refresh, so reading an
# existing row never double-counts. Each phase is reached at most once per row, so a simple
# "value is not None" guard is sufficient. The one raw-SQL failure path (invalid GPU/nodes config)
# bypasses the ORM and is counted explicitly at its call site in api/instance/router.py.
# ---------------------------------------------------------------------------


def _safe_metric(fn, *args):
    """Run a metric update, never letting a metrics error break a deployment/verification path.

    A failure here is not expected and points at a problem in the metrics layer itself, so log it
    (WARNING) for visibility -- the stable "failed to record launch_config metric" prefix is a good
    thing to alert on -- rather than swallowing it silently.
    """
    try:
        fn(*args)
    except Exception as exc:
        logger.warning(
            f"failed to record launch_config metric via {getattr(fn, '__name__', fn)!r}: {exc}"
        )


@event.listens_for(LaunchConfig, "after_insert")
def _track_launch_config_attempt(mapper, connection, target):
    _safe_metric(launch_config_metrics.track_attempt, target.chute_id)


@event.listens_for(LaunchConfig.retrieved_at, "set")
def _track_launch_config_retrieved(target, value, oldvalue, initiator):
    if value is not None:
        _safe_metric(launch_config_metrics.track_retrieved, target.chute_id)


@event.listens_for(LaunchConfig.verified_at, "set")
def _track_launch_config_verified(target, value, oldvalue, initiator):
    if value is not None:
        _safe_metric(launch_config_metrics.track_verified, target.chute_id)


@event.listens_for(LaunchConfig.verification_error, "set")
def _track_launch_config_failure(target, value, oldvalue, initiator):
    if value is not None:
        _safe_metric(launch_config_metrics.track_failure, target.chute_id, value)
