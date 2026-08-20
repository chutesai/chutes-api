"""
ORM definitions for servers and TDX attestations.
"""

from datetime import datetime, timezone
from cryptography import x509
from cryptography.hazmat.backends import default_backend
from cryptography.x509 import Certificate
from pydantic import BaseModel, Field, field_validator
from sqlalchemy.sql import func
from sqlalchemy.orm import relationship
from sqlalchemy import (
    Column,
    Integer,
    String,
    DateTime,
    Boolean,
    CheckConstraint,
    ForeignKey,
    Text,
    Index,
    ForeignKeyConstraint,
    UniqueConstraint,
    case,
)
from sqlalchemy.ext.hybrid import hybrid_property
from sqlalchemy.dialects.postgresql import JSONB
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from api.config import settings
from api.database import Base, generate_uuid
from api.constants import (
    ServerHealthStatus,
    ATTESTATION_PROXY_PORT,
    ATTESTATION_PROXY_HEALTH_PATH,
)
from api.node.schemas import NodeArgs
from api.constants import SUPPORTED_LUKS_VOLUMES


class TeeInstanceEvidence(BaseModel):
    """TEE evidence for a single instance: TDX quote, GPU evidence (per-GPU dicts), and server certificate."""

    quote: str = Field(..., description="Base64-encoded TDX quote")
    gpu_evidence: List[Dict[str, Any]] = Field(
        ...,
        description="Per-GPU evidence: list of dicts (each GPU's evidence/certificate already structured; evidence fields are base64 where applicable)",
    )
    instance_id: Optional[str] = Field(
        None, description="Instance ID (present when part of a chute's evidence list)"
    )
    certificate: str = Field(
        ..., description="Base64-encoded DER format TLS certificate from the server"
    )
    signature: Optional[str] = Field(
        None,
        description="Base64-encoded RSA-PKCS1v15-SHA256 signature of attested_body, signed with the host TLS private key. Present on attestation proxy >= 0.2.0. Verifying this against the certificate public key proves the responder holds the attested private key.",
    )
    attested_body: Optional[str] = Field(
        None,
        description="Base64-encoded raw response body from the attestation proxy—the exact bytes covered by signature. Present when signature is present.",
    )


class NonceResponse(BaseModel):
    """Response model for nonce generation."""

    nonce: str
    expires_at: str


class BootAttestationArgs(BaseModel):
    """Request model for boot attestation."""

    quote: str = Field(..., description="Base64 encoded TDX quote")
    miner_hotkey: str = Field(..., description="Miner hotkey that owns this VM")
    vm_name: str = Field(..., description="VM name/identifier")
    first_boot: bool = Field(
        False,
        description="True when the VM detected a fresh (re-downloaded) image via its LUKS2 header token",
    )


class BootAttestationResponse(BaseModel):
    """Response model for successful boot attestation."""

    key: str
    luks_quote_nonce: Optional[str] = None
    root_next: Optional[str] = Field(
        None,
        description="New root passphrase the VM should rotate to (None for pre-1.4.0 VMs)",
    )
    root_confirm_nonce: Optional[str] = Field(
        None,
        description="Single-use nonce for confirming root passphrase rotation via POST /luks/confirm",
    )
    vm_auth_ss58: Optional[str] = None


class RuntimeAttestationArgs(BaseModel):
    """Request model for runtime attestation."""

    quote: str = Field(..., description="Base64 encoded TDX quote")


class RuntimeAttestationResponse(BaseModel):
    """Response model for runtime attestation."""

    attestation_id: str
    verified_at: str
    status: str


@dataclass
class LuksVolumeRotation:
    """Internal result of rotating a single LUKS volume's passphrase (not an API model)."""

    current: Optional[str]
    """Current active passphrase. None on first boot — VM should run luksFormat."""
    next: str
    """New pending passphrase the VM should add as a LUKS key slot."""

    @property
    def is_first_boot(self) -> bool:
        return self.current is None


@dataclass
class StorageProvisionResult:
    """Internal result of storage provisioning (not an API model).

    The secrets a VM receives when it (re)provisions storage on boot: the rotated per-volume
    LUKS passphrases, the k3s encryption key (base64), and the single-use nonce it uses to
    confirm the rotation succeeded. Returned by both POST /provision (process_provision_request)
    and the legacy POST /luks/attest (process_luks_attest_request), which share the underlying
    _issue_storage_secrets helper.
    """

    volumes: Dict[str, "LuksVolumeRotation"]
    confirm_nonce: str
    k3s_encryption_key: str


@dataclass
class LuksConfirmResult:
    """Internal result of process_luks_confirm (not an API model)."""

    volumes: Dict[str, dict]
    """Per-volume outcome: {"result": "promoted"|"discarded"|"no_pending"}."""


@dataclass
class AttestationAuth:
    """Authorization presented with an attestation: the raw proof that the caller may use the
    matched measurement. Threaded to ``verify_quote``'s rc gate, which VERIFIES it -- the gate never
    trusts a caller-supplied "already authenticated" flag; it proves possession from the material
    carried here. Two modes, each matched to its environment and each self-verifying:

      * ``signed`` (boot/provision, initramfs): ``rc_signature`` is a hex RSA PKCS#1 v1.5 / SHA-256
        signature over the server-issued nonce; the gate verifies it against the measurement's
        ``authorized_signing_keys``. No hotkey/sr25519 (unavailable in the measured initramfs).
      * ``hotkey`` (register/runtime, userspace): the miner's STANDARD request auth material --
        ``miner_hotkey`` + the ``X-Chutes-Signature`` over ``get_signing_message(hotkey, nonce,
        body_sha256, purpose)``. The gate re-runs that exact verification (``nonce_is_valid`` +
        ``Keypair.verify``) and then checks ``miner_hotkey`` against ``authorized_hotkeys``. This is
        the same signature ``get_current_user`` checks, re-verified so the gate depends on no
        upstream assumption.

    An empty value proves nothing and may only use published (non-rc) measurements -- rc fails
    closed. See ``api.server.util.authorize_rc_measurement``.
    """

    # signed mode: hex RSA PKCS#1 v1.5 / SHA-256 signature over the nonce by an operator key.
    rc_signature: Optional[str] = None
    # hotkey mode: the miner's standard request-auth material, re-verified by the gate.
    miner_hotkey: Optional[str] = None
    hotkey_signature: Optional[str] = None  # hex sr25519 X-Chutes-Signature
    hotkey_nonce: Optional[str] = None  # X-Chutes-Nonce
    body_sha256: Optional[str] = None  # request.state.body_sha256 (payload hash)
    purpose: Optional[str] = None  # the endpoint's get_current_user purpose

    @classmethod
    def signed(cls, rc_signature: Optional[str]) -> "AttestationAuth":
        # Boot/provision proof is purely the RSA signature; the identity is whichever authorized
        # operator key verifies it, so no hotkey is carried.
        return cls(rc_signature=rc_signature)

    @classmethod
    def hotkey_signed(
        cls,
        miner_hotkey: Optional[str],
        *,
        signature: Optional[str] = None,
        nonce: Optional[str] = None,
        body_sha256: Optional[str] = None,
        purpose: Optional[str] = None,
    ) -> "AttestationAuth":
        # Register/runtime proof is the standard request hotkey signature; the gate re-verifies it
        # (it is NOT trusted just because it was carried here).
        return cls(
            miner_hotkey=miner_hotkey,
            hotkey_signature=signature,
            hotkey_nonce=nonce,
            body_sha256=body_sha256,
            purpose=purpose,
        )


class LuksAttestRequest(BaseModel):
    """Request model for POST /luks/attest."""

    quote: str = Field(..., description="Base64-encoded TDX quote (runtime type, RTMR3 extended)")
    volumes: List[str] = Field(..., description="Volume names to rotate passphrases for")

    @field_validator("volumes")
    @classmethod
    def validate_volumes(cls, v: List[str]) -> List[str]:
        if not v:
            raise ValueError("volumes must be non-empty")
        invalid = [vol for vol in v if vol not in SUPPORTED_LUKS_VOLUMES]
        if invalid:
            raise ValueError(
                f"Invalid volume name(s): {invalid}. Supported: {list(SUPPORTED_LUKS_VOLUMES)}"
            )
        return v


class LuksVolumeInfo(BaseModel):
    """Passphrase info for a single volume in the luks/attest response."""

    current: Optional[str] = Field(
        None,
        description="Current passphrase (None on first boot — VM must luksFormat before luksOpen)",
    )
    next: str = Field(
        ..., description="New pending passphrase the VM should add as a LUKS key slot"
    )


class LuksAttestResponse(BaseModel):
    """Response model for POST /luks/attest."""

    volumes: Dict[str, LuksVolumeInfo]
    confirm_nonce: str = Field(..., description="Single-use nonce for the confirm endpoint")
    k3s_encryption_key: str = Field(..., description="k3s encryption key (base64)")


class LuksVolumeConfirmStatus(BaseModel):
    """Confirm status for a single volume."""

    rotated: bool = Field(
        ...,
        description="True if passphrase rotation succeeded for this volume; False to discard pending",
    )


class LuksConfirmRequest(BaseModel):
    """Request model for POST /luks/confirm."""

    volumes: Dict[str, LuksVolumeConfirmStatus] = Field(
        ..., description="Per-volume rotation result reported by the VM"
    )


class LuksConfirmResponse(BaseModel):
    """Response model for POST /luks/confirm."""

    status: str
    volumes: Dict[str, Any]


class ProvisionRequest(BaseModel):
    """
    Request model for POST /servers/{vm_name}/provision.

    The runtime (RTMR3-attested) provisioning entry point for new VMs. The VM presents its
    root CA as the mTLS client cert; the quote's REPORTDATA binds SHA256(that cert's pubkey),
    so the CA identity is recorded from this call. Mirrors the luks/attest body today (quote
    + volumes) and is the extensible home for future provisioning inputs.
    """

    quote: str = Field(..., description="Base64-encoded TDX quote (runtime type, RTMR3 extended)")
    volumes: List[str] = Field(..., description="Volume names to rotate passphrases for")

    @field_validator("volumes")
    @classmethod
    def validate_volumes(cls, v: List[str]) -> List[str]:
        if not v:
            raise ValueError("volumes must be non-empty")
        invalid = [vol for vol in v if vol not in SUPPORTED_LUKS_VOLUMES]
        if invalid:
            raise ValueError(
                f"Invalid volume name(s): {invalid}. Supported: {list(SUPPORTED_LUKS_VOLUMES)}"
            )
        return v


class ProvisionResponse(BaseModel):
    """
    Response model for POST /servers/{vm_name}/provision.

    Carries the storage-provisioning secrets today (rotated volume passphrases, k3s
    encryption key, confirm nonce); shaped to extend with future provisioning outputs.
    """

    volumes: Dict[str, LuksVolumeInfo]
    confirm_nonce: str = Field(..., description="Single-use nonce for POST /provision/confirm")
    k3s_encryption_key: str = Field(..., description="k3s encryption key (base64)")


class GpuAttestationArgs(BaseModel):
    evidence: str = Field(..., description="Base64 encoded GPU evidence")


class GpuAttestationResponse(BaseModel):
    attestation_id: str
    verified_at: str
    gpu_info: Dict[str, Any]  # GPU details from evidence


class ServerArgs(BaseModel):
    """Request model for server registration."""

    host: str = Field(..., description="Public IP address or DNS Name of the server")
    id: str = Field(..., description="Server ID (e.g. k8s node uid)")
    name: Optional[str] = Field(None, description="Server name (defaults to server id if omitted)")
    gpus: list[NodeArgs] = Field(..., description="GPU info for this server")


class TeeChuteEvidence(BaseModel):
    """TEE evidence for a chute: list of evidence per instance (from instance evidence endpoints)."""

    evidence: List[TeeInstanceEvidence] = Field(
        ..., description="TEE evidence for each instance of the chute"
    )
    failed_instance_ids: List[str] = Field(
        default_factory=list,
        description="Instance IDs for which evidence could not be retrieved (instances still exist but evidence fetch failed)",
    )


class MaintenanceReason(BaseModel):
    """A single reason why maintenance eligibility was denied."""

    reason: str
    current_version: Optional[str] = None
    target_version: Optional[str] = None
    window_id: Optional[str] = None
    current_slots: Optional[int] = None
    limit: Optional[int] = None
    blocking: Optional[List[dict]] = None


class SoleSurvivorBlock(BaseModel):
    """An instance that is the sole active instance for its chute."""

    chute_id: str
    instance_id: str


class PreflightResult(BaseModel):
    """Result of a maintenance preflight eligibility check."""

    eligible: bool
    denial_reasons: List[MaintenanceReason] = Field(default_factory=list)
    blocking_chute_ids: List[SoleSurvivorBlock] = Field(default_factory=list)
    current_slots: int = 0
    limit: int = 1


class UpgradeWindowInfo(BaseModel):
    """Summary of an upgrade window for API responses."""

    id: str
    target_measurement_version: str
    upgrade_window_start: str
    upgrade_window_end: str
    max_concurrent_per_miner: int = 1


class ConfirmMaintenanceResult(BaseModel):
    """Result of confirming maintenance on a server."""

    server_id: str
    purged_instance_ids: List[str] = Field(default_factory=list)
    window: UpgradeWindowInfo


class ServerUpgradeStatus(BaseModel):
    """A TEE server and its version relative to the upgrade target."""

    server_id: str
    name: Optional[str] = None
    version: Optional[str] = None
    needs_upgrade: bool
    in_maintenance: bool


class MaintenancePolicyResponse(BaseModel):
    """Response for GET /servers/maintenance/policy."""

    active_window: Optional[UpgradeWindowInfo] = None
    window_open: bool = False
    current_slots: int = 0
    servers: List[ServerUpgradeStatus] = Field(default_factory=list)


class TeeMeasurementResponse(BaseModel):
    """Public response model for a single accepted TEE measurement configuration."""

    version: str
    name: str
    mrtd: str
    boot_rtmrs: Dict[str, str]
    runtime_rtmrs: Dict[str, str]
    expected_gpus: List[str]
    gpu_count: int


class VmBootRecord(Base):
    """The pre-server initramfs boot record for a VM -- one row per boot (append; history retained).

    A VM boots in fully-measured initramfs, before the miner registers it via POST /servers, so
    there is no Server row yet. Each row captures one boot's full initramfs lifecycle:
      * ``boot_quote``      -- the boot attestation quote, set on /boot/attestation (row insert)
      * ``provision_quote`` -- the runtime provisioning quote, set when /provision updates this row
      * ``vm_root_ca_cert`` -- the per-boot VM root CA recorded in measured initramfs (/provision)
      * ``measurement_version``, ``server_ip``, and (on failed boots) ``verification_error``

    /provision updates the VM's most recent boot row (correlating the two initramfs calls of the
    same boot), so a successful boot ends up with both quotes + the CA in a single row -- the boot
    vs provision distinction is which quote column is set, so no phase discriminator is needed. The
    *current* CA is the ``vm_root_ca_cert`` of the VM's latest row that has one; ``register_server``
    syncs it onto ``servers.vm_root_ca_cert`` (the copy every mTLS consumer reads). Ephemeral
    per-boot auth keys live in ``VmAuthKey``, deliberately not here -- no history of throwaway keys.

    (Formerly ``boot_attestations`` -- broadened in place; existing rows are preserved.)
    """

    __tablename__ = "vm_boot_records"

    attestation_id = Column(String, primary_key=True, default=generate_uuid)
    boot_quote = Column(Text, nullable=False)  # base64 boot quote; every row is a boot attestation
    provision_quote = Column(
        Text, nullable=True
    )  # base64 runtime quote (/provision), set on update
    # The luks_quote_nonce issued at /boot/attestation and consumed at /provision -- ties the two
    # calls of one boot deterministically to this row.
    provision_nonce = Column(String, nullable=True)
    server_ip = Column(String, nullable=True)
    miner_hotkey = Column(String, nullable=True)
    vm_name = Column(String, nullable=True)
    vm_root_ca_cert = Column(
        Text, nullable=True
    )  # per-boot VM root CA (PEM), recorded at /provision
    verification_error = Column(String, nullable=True)
    measurement_version = Column(
        String, nullable=True
    )  # Matched TEE measurement config version (audit trail); NULL if verification failed
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    verified_at = Column(DateTime(timezone=True), nullable=True)

    __table_args__ = (
        Index("idx_boot_server_id", "server_ip"),
        Index("idx_boot_created", "created_at"),
        Index("idx_boot_verified", "verified_at"),
        Index("idx_boot_miner_vm", "miner_hotkey", "vm_name"),
    )


class TeeUpgradeWindow(Base):
    """Validator-managed maintenance window: one row per coordinated TEE image cutover."""

    __tablename__ = "tee_upgrade_windows"

    id = Column(String, primary_key=True, default=generate_uuid)
    upgrade_window_start = Column(DateTime(timezone=True), nullable=False)
    upgrade_window_end = Column(DateTime(timezone=True), nullable=False)
    target_measurement_version = Column(Text, nullable=False)
    max_concurrent_per_miner = Column(Integer, nullable=False, default=1, server_default="1")
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)

    pending_servers = relationship(
        "Server",
        back_populates="pending_upgrade_window",
        foreign_keys="Server.maintenance_pending_window_id",
    )

    __table_args__ = (
        UniqueConstraint("target_measurement_version", name="uq_tee_upgrade_target"),
        CheckConstraint("upgrade_window_end > upgrade_window_start", name="chk_window_bounds"),
        Index("idx_tee_upgrade_window_bounds", "upgrade_window_start", "upgrade_window_end"),
    )


class Server(Base):
    """Main server entity (created after boot via CLI)."""

    __tablename__ = "servers"

    server_id = Column(String, primary_key=True)  # Provided by client (e.g. k8s node uid)
    ip = Column(String, nullable=False)  # Links to boot attestations
    miner_hotkey = Column(String, nullable=False)
    name = Column(
        String, nullable=False
    )  # Stable identity for LUKS linkage (unique with miner_hotkey)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    netuid = Column(Integer, nullable=False, default=64, server_default="64")

    is_tee = Column(Boolean, default=False, server_default="false")

    # Maintenance: set at confirm, cleared on successful boot completion or lazily when window closes.
    maintenance_pending_window_id = Column(
        String,
        ForeignKey("tee_upgrade_windows.id", ondelete="SET NULL"),
        nullable=True,
    )
    # Current attested measurement version, updated on every successful boot attestation.
    version = Column(Text, nullable=True)

    # Per-VM root CA cert recorded via POST /servers/{vm_name}/provision (from the mTLS
    # client cert of the RTMR3-attested runtime call). NULL means the VM has not yet
    # provisioned (pre-migration or old image) -> legacy auth path.
    vm_root_ca_cert = Column(Text, nullable=True)

    # Timestamp of the last successful TEE /status/health probe; stamped by server_health_prober.py.
    # NULL = never seen healthy. health_status below is derived from this, live.
    last_health_at = Column(DateTime(timezone=True), nullable=True)

    @property
    def vm_root_ca_certificate(self) -> Optional[Certificate]:
        """
        Parsed form of vm_root_ca_cert; None when the VM has not provisioned a CA.

        The raw column stays the PEM string (it is written straight from the mTLS client cert
        and consumed as-is by ssl_context.load_verify_locations(cadata=...)); this property is
        for the consumers that need an x509.Certificate (leaf verification). vm_root_ca_cert is
        always written from a valid cert, so a malformed value here is a data-integrity bug and
        is allowed to raise rather than be masked as an auth failure.
        """
        if not self.vm_root_ca_cert:
            return None
        return x509.load_pem_x509_certificate(self.vm_root_ca_cert.encode(), default_backend())

    @property
    def in_maintenance(self) -> bool:
        return self.maintenance_pending_window_id is not None

    @property
    def health_check_url(self) -> str:
        """Attestation-proxy health endpoint (HTTPS, self-signed) — the workload-readiness signal."""
        return f"https://{self.ip}:{ATTESTATION_PROXY_PORT}{ATTESTATION_PROXY_HEALTH_PATH}"

    @hybrid_property
    def health_status(self) -> ServerHealthStatus:
        """
        Liveness derived live from last_health_at and the configured thresholds:
        healthy -> degraded (no comms past the degraded threshold) -> offline (past the offline threshold).
        Never seen healthy => unknown. Recomputed on every read, so time-based transitions
        (degraded -> offline) happen with no write; the prober only stamps last_health_at on success.
        """
        if self.last_health_at is None:
            return ServerHealthStatus.UNKNOWN
        age = (datetime.now(timezone.utc) - self.last_health_at).total_seconds()
        if age >= settings.server_health_offline_threshold_seconds:
            return ServerHealthStatus.OFFLINE
        if age >= settings.server_health_degraded_threshold_seconds:
            return ServerHealthStatus.DEGRADED
        return ServerHealthStatus.HEALTHY

    @health_status.expression
    def health_status(cls):
        """SQL form of the above so it's queryable: Server.health_status.in_(('degraded', 'offline'))."""
        age = func.extract("epoch", func.now() - cls.last_health_at)
        return case(
            (cls.last_health_at.is_(None), ServerHealthStatus.UNKNOWN.value),
            (
                age >= settings.server_health_offline_threshold_seconds,
                ServerHealthStatus.OFFLINE.value,
            ),
            (
                age >= settings.server_health_degraded_threshold_seconds,
                ServerHealthStatus.DEGRADED.value,
            ),
            else_=ServerHealthStatus.HEALTHY.value,
        )

    # Relationships
    nodes = relationship("Node", back_populates="server", cascade="all, delete-orphan")
    runtime_attestations = relationship(
        "ServerAttestation", back_populates="server", cascade="all, delete-orphan"
    )
    miner = relationship("MetagraphNode", back_populates="servers")
    pending_upgrade_window = relationship(
        "TeeUpgradeWindow",
        back_populates="pending_servers",
        foreign_keys=[maintenance_pending_window_id],
    )

    __table_args__ = (
        Index("idx_server_miner", "miner_hotkey"),
        Index("idx_servers_miner_name", "miner_hotkey", "name", unique=True),
        Index(
            "idx_servers_maintenance_pending",
            "miner_hotkey",
            postgresql_where=maintenance_pending_window_id.isnot(None),
        ),
        Index("idx_servers_last_health", "last_health_at"),
        ForeignKeyConstraint(
            ["netuid", "miner_hotkey"], ["metagraph_nodes.netuid", "metagraph_nodes.hotkey"]
        ),
    )


class ServerAttestation(Base):
    """Track runtime attestations (post-registration)."""

    __tablename__ = "server_attestations"

    attestation_id = Column(String, primary_key=True, default=generate_uuid)
    server_id = Column(String, ForeignKey("servers.server_id", ondelete="CASCADE"), nullable=False)
    quote_data = Column(Text, nullable=True)  # Base64 encoded quote
    verification_error = Column(String, nullable=True)
    measurement_version = Column(
        String, nullable=True
    )  # Matched TEE measurement config version (audit trail); NULL if verification failed
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    verified_at = Column(DateTime(timezone=True), nullable=True)

    server = relationship("Server", back_populates="runtime_attestations")

    __table_args__ = (
        Index("idx_attestation_server", "server_id"),
        Index("idx_attestation_created", "created_at"),
        Index("idx_attestation_verified", "verified_at"),
    )


class VmCacheConfig(Base):
    """Track LUKS volume encryption passphrases by VM configuration (JSONB: volume name -> encrypted passphrase)."""

    __tablename__ = "vm_cache_configs"

    miner_hotkey = Column(String, primary_key=True)
    vm_name = Column(String, primary_key=True)
    volume_passphrases = Column(JSONB, nullable=False, default=dict)
    k3s_encryption_key = Column(Text, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    last_boot_at = Column(DateTime(timezone=True), nullable=True)

    __table_args__ = (
        Index("idx_vm_cache_miner", "miner_hotkey"),
        Index("idx_vm_cache_last_boot", "last_boot_at"),
    )


class VmAuthKey(Base):
    """Per-VM ephemeral SR25519 auth key, rotated on every successful boot attestation.

    Lifecycle: created/replaced on each boot attestation; independent of VmCacheConfig
    which persists across reboots. The auth_seed is Fernet-encrypted at rest.
    """

    __tablename__ = "vm_auth_keys"

    miner_hotkey = Column(String, primary_key=True)
    vm_name = Column(String, primary_key=True)
    auth_seed = Column(Text, nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    __table_args__ = (Index("idx_vm_auth_keys_miner", "miner_hotkey"),)
