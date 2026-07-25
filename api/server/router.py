"""
FastAPI routes for server management and TDX attestation.
"""

from typing import Dict, Any, List
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
from api.metagraph import get_miner_by_hotkey
from api.constants import HOTKEY_HEADER, NoncePurpose

from api.server.schemas import (
    ProvisionRequest,
    ProvisionResponse,
    BootAttestationArgs,
    RuntimeAttestationArgs,
    ServerArgs,
    Server,
    NonceResponse,
    BootAttestationResponse,
    RuntimeAttestationResponse,
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
    process_runtime_attestation,
    get_server_attestation_status,
    delete_server,
    validate_request_nonce,
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
    extract_client_cert_hash,
    require_mtls_domain,
    extract_client_cert,
)
from api.server.exceptions import (
    AttestationError,
    ServerNotFoundError,
    ServerRegistrationError,
)
from api.miner.util import is_miner_blacklisted
from api.util import is_valid_host, semcomp


router = APIRouter(dependencies=[Depends(bind_request_context)])


# Anonymous Boot Attestation Endpoints (Pre-registration)


@router.get("/nonce", response_model=NonceResponse)
async def get_nonce(
    request: Request,
    miner_hotkey: str,
    _mtls=Depends(require_mtls_domain()),
):
    """
    Generate a nonce for boot attestation.

    This endpoint is called by VMs during boot before any registration.
    No authentication required as the VM doesn't exist in the system yet.

    miner_hotkey is required and bound into the nonce. Boot attestation will reject
    any request whose args.miner_hotkey does not match the value stored here,
    preventing cross-miner nonce reuse. VMs that do not supply this param are
    considered legacy and are rejected — use the measurement version enforcement
    to drive upgrades.
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
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to generate nonce"
        )


@router.post("/boot/attestation", response_model=BootAttestationResponse)
async def verify_boot_attestation(
    request: Request,
    args: BootAttestationArgs,
    db: AsyncSession = Depends(get_db_session),
    _mtls=Depends(require_mtls_domain()),
    nonce: str = Depends(validate_boot_nonce()),
    expected_cert_hash=Depends(extract_client_cert_hash()),
):
    """
    Verify boot attestation and return LUKS passphrase.

    This endpoint verifies the TDX quote against expected boot measurements
    and returns the LUKS passphrase for disk decryption if valid.
    For VMs running version >= 1.3.0, also returns a luks_quote_nonce for
    the subsequent POST /luks/attest call.
    """
    try:
        server_ip = request.state.client_ip

        # Verify the miner hotkey is actually registered on the subnet.
        miner_node = await get_miner_by_hotkey(args.miner_hotkey, db)
        if not miner_node:
            logger.warning(
                f"Boot attestation rejected: miner hotkey {args.miner_hotkey} is not registered on subnet {settings.netuid}"
            )
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Miner hotkey is not registered on the subnet",
            )

        result: BootAttestationResult = await process_boot_attestation(
            db, server_ip, args, nonce, expected_cert_hash
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
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Boot attestation failed"
        )


@router.post("/{vm_name}/luks/attest", response_model=LuksAttestResponse)
async def attest_luks(
    vm_name: str,
    body: LuksAttestRequest,
    db: AsyncSession = Depends(get_db_session),
    hotkey: str | None = Header(None, alias=HOTKEY_HEADER),
    _mtls=Depends(require_mtls_domain()),
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
    _mtls=Depends(require_mtls_domain()),
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
    _mtls=Depends(require_mtls_domain()),
    client_cert: Certificate = Depends(extract_client_cert()),
    validated_nonce: str = Depends(require_luks_quote_nonce),
):
    """
    Provision a new VM at runtime: record its root CA identity and issue storage secrets.

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
            db, hotkey, vm_name, body, validated_nonce, client_cert
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
    _mtls=Depends(require_mtls_domain()),
    _=Depends(require_confirm_nonce),
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
):
    """
    Register a new server.

    This is called via CLI after the server has booted and decrypted its disk.
    Links the server to any existing boot attestation history via server ip.
    """
    try:
        reason = await is_miner_blacklisted(db, hotkey)
        if reason:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=reason,
            )

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
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Server registration failed"
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
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to get server details"
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
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to remove server"
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
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to generate nonce"
        )


@router.post("/{server_id}/attestation", response_model=RuntimeAttestationResponse)
async def verify_runtime_attestation(
    request: Request,
    server_id: str,
    args: RuntimeAttestationArgs,
    db: AsyncSession = Depends(get_db_session),
    hotkey: str | None = Header(None, alias=HOTKEY_HEADER),
    _: User = Depends(
        get_current_user(purpose="tee", raise_not_found=False, registered_to=settings.netuid)
    ),
    nonce: str = Depends(validate_request_nonce(NoncePurpose.RUNTIME)),
    expected_cert_hash=Depends(extract_client_cert_hash()),
):
    """
    Verify runtime attestation with full measurement validation.
    """
    try:
        server = await check_server_ownership(db, server_id, hotkey)
        actual_ip = request.state.client_ip
        result = await process_runtime_attestation(
            db, server.server_id, actual_ip, args, hotkey, nonce, expected_cert_hash
        )

        return RuntimeAttestationResponse(
            attestation_id=result["attestation_id"],
            verified_at=result["verified_at"],
            status=result["status"],
        )

    except ServerNotFoundError as e:
        raise e
    except AttestationError as e:
        # Includes NonceError (400) and all quote/GPU errors. Already logged at the detection
        # site (with ambient server identity); the boundary only maps to HTTP.
        raise HTTPException(status_code=e.http_status, detail=e.message)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error in runtime attestation: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Runtime attestation failed"
        )


# ToDo: Also likely to remove this
@router.get("/{server_id}/attestation/status", response_model=Dict[str, Any])
async def get_attestation_status(
    server_id: str,
    db: AsyncSession = Depends(get_db_session),
    hotkey: str | None = Header(None, alias=HOTKEY_HEADER),
    _: User = Depends(
        get_current_user(purpose="tee", raise_not_found=False, registered_to=settings.netuid)
    ),
):
    """
    Get current attestation status for a server by miner hotkey and server id.
    """
    try:
        server = await check_server_ownership(db, server_id, hotkey)
        status_info = await get_server_attestation_status(db, server.server_id, hotkey)
        status_info["name"] = server.name
        return status_info

    except ServerNotFoundError as e:
        raise e
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get attestation status: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get attestation status",
        )
