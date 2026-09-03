"""
FastAPI routes for server management and TDX attestation.
"""

from typing import Dict, Any, List, Optional
import orjson as json
from fastapi import APIRouter, Depends, HTTPException, Request, status, Header, Query
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.exc import IntegrityError, DatabaseError
from loguru import logger
from api.request_context import bind_request_context
from cryptography.x509 import Certificate

from api.database import get_db_session
from api.config import settings
from api.node.util import check_node_inventory
from api.user.schemas import User
from api.user.service import get_current_user
from api.constants import (
    HOTKEY_HEADER,
    NoncePurpose,
    HOST_PROFILE_MAX_BYTES,
    HOST_PROFILE_SUBMISSIONS_PER_HOTKEY,
    HOST_PROFILE_SUBMISSIONS_GLOBAL,
    HOST_PROFILE_WINDOW_SECONDS,
    HostProfileStatus,
    HOST_PROFILES_RATE_LIMIT_PER_MINUTE,
    TDX_PREFLIGHT_PER_HOTKEY,
    TDX_PREFLIGHT_GLOBAL,
    TDX_HOST_PROFILE_STATUS_PER_HOTKEY,
    TDX_HOST_PROFILE_STATUS_GLOBAL,
)
from api.rate_limit import rate_limit, rate_limit_miner

from api.server.schemas import (
    HotkeyAuth,
    ProvisionRequest,
    ProvisionResponse,
    BootAttestationArgs,
    ServerArgs,
    Server,
    NonceResponse,
    BootAttestationResponse,
    LuksAttestRequest,
    LuksAttestResponse,
    LuksVolumeInfo,
    LuksConfirmRequest,
    LuksConfirmResponse,
    PreflightResult,
    ConfirmMaintenanceResult,
    MaintenancePolicyResponse,
    ServerUpgradeStatus,
    TeeUpgradeWindow,
    UpgradeWindowInfo,
    TeeMeasurementResponse,
    HostProfileResponse,
    HostProfile,
    HostProfileSubmissionResponse,
    HostProfileStatusResponse,
    HostProfileMeasurement,
    PreflightResponse,
)
from api.server.service import (
    BootAttestationResult,
    create_nonce,
    process_boot_attestation,
    register_server,
    process_provision_request,
    check_server_ownership,
    get_server_by_name_or_id,
    update_server_name,
    delete_server,
    validate_boot_nonce,
    require_luks_quote_nonce,
    require_confirm_nonce,
    process_luks_attest_request,
    process_luks_confirm,
    get_latest_upgrade_window,
    is_window_open,
    preflight_maintenance,
    confirm_maintenance,
    _count_active_maintenance_slots,
)
from api.server.util import (
    list_host_profile_records,
    require_hotkey_auth,
    deny_blacklisted_miner,
    resolve_host_profile_status,
    host_profile_status,
    measurements_for_fingerprint,
    extract_hotkey_auth,
    extract_client_cert_hash,
    require_mtls_proxy,
    require_attestation_proxy,
    require_cvm_proxy,
    gate_legacy_attestation,
    extract_client_cert,
)
from api.server.exceptions import (
    AttestationError,
    ServerNotFoundError,
    ServerRegistrationError,
)
from api.util import is_valid_host, semcomp


router = APIRouter(dependencies=[Depends(bind_request_context)])

HOST_PROFILE_STATUS_DETAIL = {
    HostProfileStatus.ACCEPTED: (
        "This host class has had measurements generated and is retained for attestation. See GET "
        "/servers/tdx/host_profiles for the (version, rc) images that cover it."
    ),
    HostProfileStatus.PENDING: (
        "Host profile stored; measurements will be generated and published."
    ),
    HostProfileStatus.UNKNOWN: (
        "This host class has no measurements and no submission on file. Submit the profile to "
        "request measurements."
    ),
}


# Anonymous Boot Attestation Endpoints (Pre-registration)


@router.get("/nonce", response_model=NonceResponse)
async def get_nonce(
    request: Request,
    miner_hotkey: Optional[str] = None,
    _mtls=Depends(gate_legacy_attestation()),
):
    """
    Generate a nonce for boot attestation.

    This endpoint is called by VMs during boot before any registration.
    No authentication required as the VM doesn't exist in the system yet.

    miner_hotkey is OPTIONAL for backwards compatibility: older VM initramfs fetch the nonce
    without it. When supplied it is bound into the nonce, and boot attestation enforces that
    args.miner_hotkey matches the value stored here (preventing cross-miner nonce reuse). VMs
    that omit it get an unbound nonce and the binding check is skipped.

    TODO: make miner_hotkey required when all VMs >= 1.4.0
    """
    try:
        server_ip = request.state.client_ip
        nonce_info = await create_nonce(
            server_ip, purpose=NoncePurpose.BOOT, miner_hotkey=miner_hotkey
        )

        return NonceResponse(nonce=nonce_info["nonce"], expires_at=nonce_info["expires_at"])
    except Exception as e:
        logger.error(f"Failed to generate boot nonce: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Could not issue a boot nonce (validator-side error). Retry shortly.",
        )


@router.post("/boot/attestation", response_model=BootAttestationResponse)
async def verify_boot_attestation(
    request: Request,
    args: BootAttestationArgs,
    db: AsyncSession = Depends(get_db_session),
    _mtls=Depends(require_mtls_proxy()),
    nonce: str = Depends(validate_boot_nonce()),
    expected_cert_hash=Depends(extract_client_cert_hash()),
    auth: Optional[HotkeyAuth] = Depends(extract_hotkey_auth()),
    _blacklist=Depends(deny_blacklisted_miner()),
):
    """
    Verify boot attestation and return LUKS passphrase.

    Both VM generations reach this route, so the hotkey proof is EXTRACTED rather than required:
    a presented signature is verified here and 401s if it does not hold, but its absence is left
    for the handler to judge once the quote names the attested image. See
    process_boot_attestation, which requires a proof from any image whose measured initramfs
    ships the signer.

    Verifies the TDX quote against the expected boot measurements and returns the LUKS passphrase
    for disk decryption if valid. For VMs >= 1.3.0 it also returns a luks_quote_nonce for the
    following runtime call (POST /provision on 1.4.0+, POST /luks/attest on 1.3.x); for 1.4.0+ it
    additionally returns root_next + root_confirm_nonce and the VM's ephemeral auth SS58.
    """
    try:
        server_ip = request.state.client_ip

        result: BootAttestationResult = await process_boot_attestation(
            db,
            server_ip,
            args,
            nonce,
            expected_cert_hash,
            auth=auth,
        )

        return BootAttestationResponse(
            key=result.root_key,
            luks_quote_nonce=result.luks_quote_nonce,
            root_next=result.root_next,
            root_confirm_nonce=result.root_confirm_nonce,
            vm_auth_ss58=result.vm_auth_ss58,
        )
    except AttestationError as e:
        # Includes NonceError (400) and all quote/GPU errors. The failure was already logged
        # at its detection site (with ambient identity); the boundary only maps to HTTP.
        raise HTTPException(status_code=e.http_status, detail=e.message)
    except Exception as e:
        logger.error(f"Unexpected error in boot attestation: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=(
                "Boot attestation failed with an unexpected validator-side error. Retry; if it "
                "persists, contact the Chutes team with the VM name and time of this attempt."
            ),
        )


@router.post("/{vm_name}/luks/attest", response_model=LuksAttestResponse)
async def attest_luks(
    vm_name: str,
    body: LuksAttestRequest,
    db: AsyncSession = Depends(get_db_session),
    hotkey: str | None = Header(None, alias=HOTKEY_HEADER),
    _mtls=Depends(require_attestation_proxy()),
    expected_cert_hash=Depends(extract_client_cert_hash()),
    validated_nonce: str = Depends(require_luks_quote_nonce),
):
    """
    Rotate LUKS passphrases for new-format VMs (version >= 1.3.0).

    DEPRECATED: superseded by POST /provision, which does the same storage rotation and
    additionally records the VM root CA identity. Kept unchanged for legacy in-field VMs;
    retire once the fleet upgrades.

    The VM embeds the luks_quote_nonce (received in the boot attestation response)
    in a TDX quote after extending RTMR3 in initramfs. require_luks_quote_nonce
    validates and consumes the nonce; the handler then calls verify_quote which
    checks the TDX signature and all RTMR measurements including RTMR3. Returns
    rotated passphrases, the k3s encryption key, and a confirm nonce.
    """
    try:
        result = await process_luks_attest_request(
            db, hotkey, vm_name, body, validated_nonce, expected_cert_hash
        )
        return LuksAttestResponse(
            volumes={
                vol: LuksVolumeInfo(current=r.current, next=r.next)
                for vol, r in result.volumes.items()
            },
            confirm_nonce=result.confirm_nonce,
            k3s_encryption_key=result.k3s_encryption_key,
        )
    except AttestationError as e:
        # verify_quote logged the failure at its detection site (with ambient identity);
        # the boundary only maps to HTTP.
        raise HTTPException(status_code=e.http_status, detail=e.message)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error in LUKS attest: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="LUKS attestation failed",
        )


@router.post("/{vm_name}/luks/confirm", response_model=LuksConfirmResponse)
async def confirm_luks_rotation(
    vm_name: str,
    body: LuksConfirmRequest,
    db: AsyncSession = Depends(get_db_session),
    hotkey: str | None = Header(None, alias=HOTKEY_HEADER),
    _mtls=Depends(gate_legacy_attestation()),
    _=Depends(require_confirm_nonce),
):
    """
    Confirm or discard pending LUKS passphrase rotation results.

    DEPRECATED for the storage-rotation flow: new VMs use POST /provision/confirm (same
    shared logic). Kept for legacy in-field VMs; note the boot-phase root-passphrase confirm
    still uses this route until /luks/* is fully retired.

    The VM reports per-volume success/failure. require_confirm_nonce validates
    and consumes the nonce before the handler runs. Volumes with rotated=True
    have pending passphrases promoted to current; rotated=False discards pending.
    """
    try:
        result = await process_luks_confirm(db, hotkey, vm_name, body)
        return LuksConfirmResponse(status="confirmed", volumes=result.volumes)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error in LUKS confirm: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="LUKS confirm failed",
        )


@router.post("/{vm_name}/provision", response_model=ProvisionResponse)
async def provision(
    vm_name: str,
    body: ProvisionRequest,
    db: AsyncSession = Depends(get_db_session),
    hotkey: str | None = Header(None, alias=HOTKEY_HEADER),
    _mtls=Depends(require_cvm_proxy()),
    client_cert: Certificate = Depends(extract_client_cert()),
    validated_nonce: str = Depends(require_luks_quote_nonce),
    auth: HotkeyAuth = Depends(require_hotkey_auth()),
):
    """
    Provision a new VM at runtime: record its root CA identity and issue storage secrets.

    Only 1.4.0+ VMs reach this route (it is behind require_cvm_proxy), and every one of them
    signs, so require_hotkey_auth demands a proven hotkey at the door -- no backwards-compatible
    "unsigned is acceptable" case exists here, unlike boot attestation. The proven identity is
    what the rc gate in verify_quote is given.

    The RTMR3-attested runtime entry point for new VMs (supersedes /luks/attest going
    forward). The VM presents its per-boot root CA as the mTLS client cert; the quote's
    REPORTDATA binds SHA256(that cert's pubkey), so the same cert_hash check that guards
    /luks/attest also proves CA possession — no bespoke quote logic is needed.
    require_luks_quote_nonce validates and consumes the runtime nonce; the handler verifies
    the quote (signature + all RTMR measurements incl. RTMR3), records
    server.vm_root_ca_cert (idempotent), and returns rotated passphrases, the k3s encryption
    key, and a confirm nonce.
    """
    try:
        result = await process_provision_request(
            db,
            hotkey,
            vm_name,
            body,
            validated_nonce,
            client_cert,
            auth=auth,
        )
        return ProvisionResponse(
            volumes={
                vol: LuksVolumeInfo(current=r.current, next=r.next)
                for vol, r in result.volumes.items()
            },
            confirm_nonce=result.confirm_nonce,
            k3s_encryption_key=result.k3s_encryption_key,
        )
    except AttestationError as e:
        # Failure already logged at its detection site (with ambient identity); the boundary
        # only maps the internal exception to HTTP.
        raise HTTPException(status_code=e.http_status, detail=e.message)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error in provision for {vm_name}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Provisioning failed",
        )


@router.post("/{vm_name}/provision/confirm", response_model=LuksConfirmResponse)
async def provision_confirm(
    vm_name: str,
    body: LuksConfirmRequest,
    db: AsyncSession = Depends(get_db_session),
    hotkey: str | None = Header(None, alias=HOTKEY_HEADER),
    _mtls=Depends(require_cvm_proxy()),
    _=Depends(require_confirm_nonce),
    _auth=Depends(require_hotkey_auth()),
):
    """
    Confirm or discard pending storage-passphrase rotations from POST /provision.

    Delegates to the same shared logic as the legacy /luks/confirm (process_luks_confirm):
    require_confirm_nonce validates and consumes the nonce, then volumes with rotated=True
    have pending passphrases promoted to current and rotated=False discards pending.
    """
    try:
        result = await process_luks_confirm(db, hotkey, vm_name, body)
        return LuksConfirmResponse(status="confirmed", volumes=result.volumes)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error in provision confirm for {vm_name}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Provision confirm failed",
        )


# Server Management Endpoints (Post-boot via CLI)
# ToDo: Not sure we will want to keep this, ideally want to integrate with miner add-node command
@router.post("/", response_model=Dict[str, str], status_code=status.HTTP_201_CREATED)
async def create_server(
    args: ServerArgs,
    db: AsyncSession = Depends(get_db_session),
    hotkey: str | None = Header(None, alias=HOTKEY_HEADER),
    _: User = Depends(get_current_user(raise_not_found=False, registered_to=settings.netuid)),
    _blacklist=Depends(deny_blacklisted_miner()),
):
    """
    Register a new server.

    This is called via CLI after the server has booted and decrypted its disk.
    Links the server to any existing boot attestation history via server ip.
    """
    try:
        gpu_uuids = [gpu.uuid for gpu in args.gpus]
        existing_nodes = await check_node_inventory(db, gpu_uuids)
        if existing_nodes:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=f"Nodes already exist in inventory, please contact chutes team to resolve: {existing_nodes}",
            )

        valid_host = await is_valid_host(args.host)
        if not valid_host:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid verification host provided.",
            )

        # TEE servers require globally unique IPs (across TEE and non-TEE)
        existing_server = (
            await db.execute(select(Server).where(Server.ip == args.host))
        ).scalar_one_or_none()
        if existing_server:
            logger.error(
                f"TEE server registration rejected: IP {args.host} already registered to server_id={existing_server.server_id} name={existing_server.name} miner_hotkey={existing_server.miner_hotkey}; requesting miner_hotkey={hotkey}"
            )
            if existing_server.miner_hotkey == hotkey:
                detail = (
                    f"IP {args.host} is already registered to your server {existing_server.server_id} ({existing_server.name}). "
                    "IPs must be unique across all servers. Use GET /miner/servers to review your inventory."
                )
            else:
                detail = "Conflict with an existing server. Please contact support to resolve."
            raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=detail)

        await register_server(db, args, hotkey)

        return {"message": "Server registered successfully."}

    except ServerRegistrationError as e:
        logger.error(
            f"Server registration failed: server_id={args.id} host={args.host} miner_hotkey={hotkey} error={e.detail}"
        )
        raise e
    except AttestationError as e:
        # register_server already emitted the structured, server-bound failure log; here we
        # only map the domain error to its HTTP response (correct status + safe message).
        # Must precede `except HTTPException` so it does NOT fall through to the generic 500.
        raise HTTPException(status_code=e.http_status, detail=e.message)
    except HTTPException:
        # Re-raise HTTPExceptions (like blacklist, node conflicts, invalid host) as-is
        raise
    except (IntegrityError, DatabaseError) as e:
        # Handle database errors that might occur before register_server is called
        # (e.g., in check_node_inventory)
        logger.error(
            f"Database error in server registration: server_id={args.id} host={args.host} miner_hotkey={hotkey} error={str(e)}",
            exc_info=True,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Server registration failed - database error. Please contact support with your server ID and miner hotkey.",
        )
    except Exception as e:
        logger.error(
            f"Unexpected error in server registration: server_id={args.id} host={args.host} miner_hotkey={hotkey} error={str(e)}",
            exc_info=True,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Server registration failed",
        )


TEE_MEASUREMENTS_CACHE_KEY = "tee_measurements"
TEE_MEASUREMENTS_CACHE_TTL = 3600  # 60 minutes; measurements only change on new releases


@router.get("/tee/measurements", response_model=List[TeeMeasurementResponse])
async def get_tee_measurements():
    """
    Return the list of currently accepted TEE measurement configurations.

    These are the reference values (MRTD + RTMRs) that the platform accepts
    during boot and runtime attestation. Clients can use these to independently
    verify that a server is running approved software before trusting it.
    No authentication required — public transparency endpoint.
    """
    cached = await settings.redis_client.get(TEE_MEASUREMENTS_CACHE_KEY)
    if cached:
        return json.loads(cached)

    try:
        measurements = settings.tee_measurements
    except Exception as e:
        logger.error(f"TEE measurement config is invalid: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to read TEE measurements",
        )

    # Exclude release-candidate (rc) measurements
    result = [
        TeeMeasurementResponse(
            version=m.version,
            name=m.name,
            mrtd=m.mrtd,
            boot_rtmrs=m.boot_rtmrs,
            runtime_rtmrs=m.runtime_rtmrs,
            expected_gpus=m.expected_gpus,
            gpu_count=m.gpu_count,
            fingerprint=m.fingerprint,
        )
        for m in measurements
        if not m.rc
    ]
    await settings.redis_client.set(
        TEE_MEASUREMENTS_CACHE_KEY,
        json.dumps([r.model_dump() for r in result]).decode(),
        ex=TEE_MEASUREMENTS_CACHE_TTL,
    )
    return result


# Per-variant, so a cached "with pending" is never served to a default request.
HOST_PROFILES_CACHE_KEY = "tdx_host_profiles:{variant}"


@router.get("/tdx/host_profiles", response_model=List[HostProfileResponse])
async def list_host_profiles(
    db: AsyncSession = Depends(get_db_session),
    include_pending: bool = Query(
        False,
        description="Also return host classes awaiting measurement generation.",
    ),
    _: None = Depends(rate_limit("tdx_host_profiles", HOST_PROFILES_RATE_LIMIT_PER_MINUTE)),
):
    """
    Return host profiles: the platform inputs a measurement is built from.

    By default, only host classes that HAVE measurements. Join `fingerprint` to
    GET /servers/tee/measurements, regenerate RTMR0 from the inputs here, and compare -- that makes
    every published measurement independently reproducible, and a quote holder can see which host
    class their own RTMR0 corresponds to. A third party needs no flags and can never be handed an
    unverified claim.

    `include_pending=true` also returns host classes awaiting generation. That is the measurement
    generator's queue: a profile becomes measured only once measurements are generated for it, and
    generation has to fetch it first. Each entry's `measured` flag says which set it is in.

    Public and unauthenticated, so the generation side needs no database credentials. Every entry
    is host-class data: the machine-identifying fields are dropped at submission and never stored,
    and the submitter's `miner_hotkey` is a column this query never selects.
    """
    cache_key = HOST_PROFILES_CACHE_KEY.format(variant="all" if include_pending else "measured")
    cached = await settings.redis_client.get(cache_key)
    if cached:
        return json.loads(cached)

    try:
        profiles = await list_host_profile_records(db, include_pending=include_pending)
    except Exception as exc:
        logger.error(f"Failed to list host profiles: {exc}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Failed to read host profiles, please try again later.",
        )

    await settings.redis_client.set(
        cache_key,
        json.dumps(profiles).decode(),
        ex=TEE_MEASUREMENTS_CACHE_TTL,
    )
    return profiles


@router.post("/tdx/host_profiles", response_model=HostProfileSubmissionResponse)
async def submit_host_profile(
    request: Request,
    profile: HostProfile,
    db: AsyncSession = Depends(get_db_session),
    hotkey: str | None = Header(None, alias=HOTKEY_HEADER),
    _: User = Depends(
        rate_limit_miner(
            "host_profile_submit",
            HOST_PROFILE_SUBMISSIONS_PER_HOTKEY,
            window_seconds=HOST_PROFILE_WINDOW_SECONDS,
            global_limit=HOST_PROFILE_SUBMISSIONS_GLOBAL,
        )
    ),
):
    """
    Register a host class (sek8s `discover-profile.sh` output) so Chutes can measure it.

    One of four distinct operations: this REGISTERS, POST /servers/tdx/host_profiles/status
    RESOLVES a class (is it known, and for which images), POST /servers/tdx/preflight CHECKS whether
    one specific image can boot, and GET /servers/tdx/host_profiles LISTS the generated set. A miner
    reaches this only when the status lookup reports the class is not on file. The API owns the
    fingerprint -- the miner sends raw platform metadata and gets back the class's retention
    lifecycle, which only ever advances:

      * `accepted` -- a measurement has already been generated for this class; retained from here on
      * `pending`  -- parked in object storage, awaiting its first measurement generation

    A real submission is always stored, so this never returns `unknown`. It answers only whether the
    class has been measured at all; whether the caller's specific image can boot is preflight's job.
    Signed by the miner hotkey, so the signature covers the request body.
    """
    raw_body = await request.body()
    if len(raw_body) > HOST_PROFILE_MAX_BYTES:
        raise HTTPException(
            status_code=status.HTTP_413_CONTENT_TOO_LARGE,
            detail=f"Host profile exceeds {HOST_PROFILE_MAX_BYTES} bytes.",
        )

    try:
        fingerprint, profile_status, stored = await resolve_host_profile_status(
            db=db,
            profile=profile,
            hotkey=hotkey,
        )
    except Exception as exc:
        logger.error(f"Failed to resolve host profile status from {hotkey=}: {exc}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Failed to store host profile, please try again later.",
        )

    return HostProfileSubmissionResponse(
        fingerprint=fingerprint,
        status=profile_status,
        stored=stored,
        detail=HOST_PROFILE_STATUS_DETAIL[profile_status],
    )


@router.post("/tdx/host_profiles/status", response_model=HostProfileStatusResponse)
async def tdx_host_profile_status(
    request: Request,
    profile: HostProfile,
    db: AsyncSession = Depends(get_db_session),
    hotkey: str | None = Header(None, alias=HOTKEY_HEADER),
    _: User = Depends(
        rate_limit_miner(
            "tdx_host_profile_status",
            TDX_HOST_PROFILE_STATUS_PER_HOTKEY,
            window_seconds=HOST_PROFILE_WINDOW_SECONDS,
            global_limit=TDX_HOST_PROFILE_STATUS_GLOBAL,
        )
    ),
):
    """
    Is this host class known, and which VM images cover it? The check `chutes-cvm host verify` runs.

    Deliberately version-free: a miner verifies a host BEFORE downloading any image, so this asks
    only about the topology -- "can this host run anything at all, and if so what". Whether one
    specific image can boot is POST /servers/tdx/preflight, which the launch path runs against the
    version it actually holds.

      * `measurements` non-empty -> the class is attestable; those are the images it can launch.
      * empty, `status: pending`  -> registered, awaiting measurement generation. Nothing to do but
        retry later; re-submitting will not speed it up.
      * empty, `status: unknown`  -> never submitted. Register it with
        POST /servers/tdx/host_profiles.

    A POST because the API owns the fingerprint: a caller cannot ask about "my topology" without
    handing over the profile to be fingerprinted. Stores nothing -- the read-only counterpart to the
    submission route. Signed by the miner hotkey, so the signature covers the request body.
    """
    raw_body = await request.body()
    if len(raw_body) > HOST_PROFILE_MAX_BYTES:
        raise HTTPException(
            status_code=status.HTTP_413_CONTENT_TOO_LARGE,
            detail=f"Host profile exceeds {HOST_PROFILE_MAX_BYTES} bytes.",
        )

    fingerprint = profile.fingerprint
    try:
        profile_status = await host_profile_status(db, fingerprint)
    except Exception as exc:
        logger.error(f"Failed to read host profile status from {hotkey=}: {exc}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Failed to look up host profile, please try again later.",
        )

    measurements = measurements_for_fingerprint(fingerprint)
    if measurements:
        covered = ", ".join(f"{m['version']}{' (rc)' if m['rc'] else ''}" for m in measurements)
        detail = (
            f"This host class is measured; published images that cover it: {covered}. Check a "
            "specific image with POST /servers/tdx/preflight."
        )
    elif profile_status is HostProfileStatus.PENDING:
        detail = (
            "This host class is registered and awaiting measurement generation. Nothing to do -- "
            "retry once Chutes publishes its measurements."
        )
    elif profile_status is HostProfileStatus.ACCEPTED:
        detail = (
            "This host class has been measured before, but no published image currently covers it. "
            "Retry once Chutes publishes measurements for a current image."
        )
    else:
        detail = (
            "This host class is not on file. Register it with POST /servers/tdx/host_profiles, "
            "then retry once Chutes publishes its measurements."
        )

    return HostProfileStatusResponse(
        fingerprint=fingerprint,
        status=profile_status,
        measurements=[HostProfileMeasurement(**m) for m in measurements],
        detail=detail,
    )


@router.post("/tdx/preflight", response_model=PreflightResponse)
async def tdx_preflight(
    request: Request,
    profile: HostProfile,
    version: str = Query(..., description="VM image version the caller intends to boot."),
    rc: bool = Query(False, description="Whether that image is a release-candidate (debug) build."),
    hotkey: str | None = Header(None, alias=HOTKEY_HEADER),
    _: User = Depends(
        rate_limit_miner(
            "tdx_preflight",
            TDX_PREFLIGHT_PER_HOTKEY,
            window_seconds=HOST_PROFILE_WINDOW_SECONDS,
            global_limit=TDX_PREFLIGHT_GLOBAL,
        )
    ),
):
    """
    Can this exact image boot on this host? The one call a launch/upgrade preflight makes.

    A check against both a host profile and the measurement set, so it hangs off neither. The API
    fingerprints the submitted profile and answers whether a published measurement for the caller's
    `(version, rc)` carries that fingerprint -- the whole launch decision in one boolean:

      * `launchable: true`  -> the VM will attest; launch.
      * `launchable: false` -> register the class (POST /servers/tdx/host_profiles), then retry once
        Chutes publishes its measurement.

    Stores nothing and reveals nothing beyond the answer -- not the full measurement set, not
    whether a profile row exists. Signed by the miner hotkey, so the signature covers the body.
    """
    raw_body = await request.body()
    if len(raw_body) > HOST_PROFILE_MAX_BYTES:
        raise HTTPException(
            status_code=status.HTTP_413_CONTENT_TOO_LARGE,
            detail=f"Host profile exceeds {HOST_PROFILE_MAX_BYTES} bytes.",
        )

    fingerprint = profile.fingerprint
    launchable = any(
        m["version"] == version and bool(m["rc"]) == rc
        for m in measurements_for_fingerprint(fingerprint)
    )
    label = f"{version}{' (rc)' if rc else ''}"
    detail = (
        f"A published measurement for {label} covers this host class; it can launch."
        if launchable
        else (
            f"No published measurement for {label} covers this host class yet. Register it with "
            "POST /servers/tdx/host_profiles, then retry once Chutes publishes the measurement."
        )
    )
    return PreflightResponse(fingerprint=fingerprint, launchable=launchable, detail=detail)


@router.get("/signing-keys")
async def get_signing_keys():
    """
    Return the signed key bundle used by booting VMs to fetch and verify cosign/Helm keys.

    Each entry in 'keys' is a base64-encoded public key; the corresponding entry in
    'signatures' is a base64-encoded detached PGP signature produced by the root signing key.
    Intentionally public — no mTLS required. Independent third parties and auditors can
    fetch these public keys to verify TDX quotes without needing to be a VM client.
    """
    bundle = settings.signing_keys_bundle
    if bundle is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Signing keys bundle not available",
        )
    return bundle


@router.get("/maintenance/policy", response_model=MaintenancePolicyResponse)
async def get_maintenance_policy(
    db: AsyncSession = Depends(get_db_session),
    hotkey: str | None = Header(None, alias=HOTKEY_HEADER),
    _: User = Depends(
        get_current_user(purpose="tee", raise_not_found=False, registered_to=settings.netuid)
    ),
):
    """Return the upgrade window, concurrency limits, and the miner's server version status.

    When a window is currently open, ``window_open`` is true and ``active_window`` describes it.
    When no window is open, we fall back to the most recently created window (``window_open``
    false) so miners can still see which servers remain out of date against the last target.
    """
    if not hotkey:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Hotkey header required")

    # The latest window is always the relevant upgrade target; window_open just says whether
    # it is currently open. Reporting it even when closed lets miners see which servers are
    # out of date (we enforce a minimum boot version even outside of an open window).
    window = await get_latest_upgrade_window(db)
    window_open = window is not None and is_window_open(window)

    window_info: UpgradeWindowInfo | None = None
    current_slots = 0
    server_statuses: list[ServerUpgradeStatus] = []

    if window is not None:
        target_version = window.target_measurement_version
        window_info = UpgradeWindowInfo(
            id=window.id,
            target_measurement_version=target_version,
            upgrade_window_start=str(window.upgrade_window_start),
            upgrade_window_end=str(window.upgrade_window_end),
            max_concurrent_per_miner=window.max_concurrent_per_miner,
        )
        current_slots = await _count_active_maintenance_slots(db, hotkey, window)

        tee_servers = (
            (
                await db.execute(
                    select(Server).where(Server.miner_hotkey == hotkey, Server.is_tee.is_(True))
                )
            )
            .scalars()
            .all()
        )

        for srv in tee_servers:
            server_statuses.append(
                ServerUpgradeStatus(
                    server_id=srv.server_id,
                    name=srv.name,
                    version=srv.version,
                    needs_upgrade=srv.version is None or semcomp(srv.version, target_version) < 0,
                    in_maintenance=srv.in_maintenance,
                )
            )

    return MaintenancePolicyResponse(
        active_window=window_info,
        window_open=window_open,
        current_slots=current_slots,
        servers=server_statuses,
    )


@router.patch("/{server_id}", response_model=Dict[str, Any])
async def patch_server_name(
    server_id: str,
    server_name: str = Query(..., description="New VM name to set"),
    db: AsyncSession = Depends(get_db_session),
    hotkey: str | None = Header(None, alias=HOTKEY_HEADER),
    _: User = Depends(
        get_current_user(purpose="tee", raise_not_found=False, registered_to=settings.netuid)
    ),
):
    """
    Update name for an existing server. Path is server_id; query param is the new name.
    The server row is updated when hotkey and server_id match.
    """
    if not hotkey:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Hotkey header required",
        )
    try:
        server = await update_server_name(db, hotkey, server_id, server_name)
        return {
            "name": server.name,
            "ip": server.ip,
            "created_at": server.created_at.isoformat(),
            "updated_at": server.updated_at.isoformat() if server.updated_at else None,
        }
    except ServerNotFoundError as e:
        raise e
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to patch server name: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to patch server name",
        )


@router.get("/{server_id}", response_model=Dict[str, Any])
async def get_server_details(
    server_id: str,
    db: AsyncSession = Depends(get_db_session),
    hotkey: str | None = Header(None, alias=HOTKEY_HEADER),
    _: User = Depends(
        get_current_user(purpose="tee", raise_not_found=False, registered_to=settings.netuid)
    ),
):
    """
    Get details for a specific server by miner hotkey and server id.
    """
    try:
        server = await check_server_ownership(db, server_id, hotkey)

        response: dict = {
            "server_id": server.server_id,
            "name": server.name,
            "ip": server.ip,
            "version": server.version,
            "maintenance_pending_window_id": server.maintenance_pending_window_id,
            "created_at": server.created_at.isoformat(),
            "updated_at": server.updated_at.isoformat() if server.updated_at else None,
        }
        if server.in_maintenance:
            window = await db.get(TeeUpgradeWindow, server.maintenance_pending_window_id)
            if window is not None:
                response["target_version"] = window.target_measurement_version

        return response

    except ServerNotFoundError as e:
        raise e
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get server details: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get server details",
        )


@router.get("/{server_name_or_id}/maintenance/preflight", response_model=PreflightResult)
async def get_maintenance_preflight(
    server_name_or_id: str,
    db: AsyncSession = Depends(get_db_session),
    hotkey: str | None = Header(None, alias=HOTKEY_HEADER),
    _: User = Depends(
        get_current_user(purpose="tee", raise_not_found=False, registered_to=settings.netuid)
    ),
):
    """Check maintenance eligibility for a server without entering maintenance."""
    server = await get_server_by_name_or_id(db, hotkey, server_name_or_id)
    return await preflight_maintenance(db, server, hotkey)


@router.put("/{server_name_or_id}/maintenance", response_model=ConfirmMaintenanceResult)
async def put_confirm_maintenance(
    server_name_or_id: str,
    db: AsyncSession = Depends(get_db_session),
    hotkey: str | None = Header(None, alias=HOTKEY_HEADER),
    _: User = Depends(
        get_current_user(purpose="tee", raise_not_found=False, registered_to=settings.netuid)
    ),
):
    """Enter maintenance: purge instances and mark server for upgrade."""
    server = await get_server_by_name_or_id(db, hotkey, server_name_or_id)
    return await confirm_maintenance(db, server, hotkey)


@router.delete("/{server_name_or_id}", response_model=Dict[str, str])
async def remove_server(
    server_name_or_id: str,
    db: AsyncSession = Depends(get_db_session),
    hotkey: str | None = Header(None, alias=HOTKEY_HEADER),
    _: User = Depends(
        get_current_user(purpose="tee", raise_not_found=False, registered_to=settings.netuid)
    ),
):
    """
    Remove a server by miner hotkey and server id or VM name (path param server_name_or_id).
    """
    try:
        server = await get_server_by_name_or_id(db, hotkey, server_name_or_id)
        await delete_server(db, server.server_id, hotkey)

        return {"name": server.name, "message": "Server removed successfully"}

    except ServerNotFoundError as e:
        raise e
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to remove server: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to remove server",
        )


# Runtime Attestation Endpoints (Post-registration)


@router.get("/{server_id}/nonce", response_model=NonceResponse)
async def get_runtime_nonce(
    request: Request,
    server_id: str,
    db: AsyncSession = Depends(get_db_session),
    hotkey: str | None = Header(None, alias=HOTKEY_HEADER),
    _: User = Depends(
        get_current_user(purpose="tee", raise_not_found=False, registered_to=settings.netuid)
    ),
):
    """
    Generate a nonce for runtime attestation.
    """
    try:
        server = await check_server_ownership(db, server_id, hotkey)

        actual_ip = request.state.client_ip
        if server.ip != actual_ip:
            logger.warning(
                f"Runtime nonce IP mismatch: server_id={server_id} registered_ip={server.ip} request_ip={actual_ip}"
            )
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Request source IP does not match the registered server IP.",
            )

        nonce_info = await create_nonce(server.ip, purpose=NoncePurpose.RUNTIME)

        return NonceResponse(nonce=nonce_info["nonce"], expires_at=nonce_info["expires_at"])

    except ServerNotFoundError as e:
        raise e
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to generate runtime nonce: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to generate nonce",
        )
