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

from api.database import get_db_session
from api.config import settings
from api.node.util import check_node_inventory
from api.user.schemas import User
from api.user.service import get_current_user
from api.constants import HOTKEY_HEADER, NoncePurpose, SUPPORTED_LUKS_VOLUMES

from api.server.schemas import (
    BootAttestationArgs,
    RuntimeAttestationArgs,
    ServerArgs,
    Server,
    NonceResponse,
    BootAttestationResponse,
    RuntimeAttestationResponse,
    LuksPassphraseRequest,
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
    create_nonce,
    process_boot_attestation,
    register_server,
    check_server_ownership,
    get_server_by_name_or_id,
    update_server_name,
    process_runtime_attestation,
    get_server_attestation_status,
    delete_server,
    validate_request_nonce,
    process_luks_passphrase_request,
    require_luks_quote_nonce,
    require_confirm_nonce,
    process_luks_attest_request,
    process_luks_confirm,
    get_active_upgrade_window,
    preflight_maintenance,
    confirm_maintenance,
    _count_active_maintenance_slots,
)
from api.server.util import (
    extract_client_cert_hash,
    get_luks_passphrase,
)
from api.server.exceptions import (
    AttestationError,
    NonceError,
    ServerNotFoundError,
    ServerRegistrationError,
)
from api.miner.util import is_miner_blacklisted
from api.util import extract_ip, is_valid_host, semcomp


router = APIRouter()


# Anonymous Boot Attestation Endpoints (Pre-registration)


@router.get("/nonce", response_model=NonceResponse)
async def get_nonce(request: Request):
    """
    Generate a nonce for boot attestation.

    This endpoint is called by VMs during boot before any registration.
    No authentication required as the VM doesn't exist in the system yet.
    """
    try:
        server_ip = extract_ip(request)
        nonce_info = await create_nonce(server_ip, purpose=NoncePurpose.BOOT)

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
    nonce=Depends(validate_request_nonce(NoncePurpose.BOOT)),
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
        server_ip = extract_ip(request)
        boot_token, luks_quote_nonce = await process_boot_attestation(
            db, server_ip, args, nonce, expected_cert_hash
        )

        return BootAttestationResponse(
            key=get_luks_passphrase(),
            boot_token=boot_token,
            luks_quote_nonce=luks_quote_nonce,
        )
    except NonceError as e:
        logger.warning(f"Boot attestation nonce error: {str(e)}")
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except AttestationError as e:
        logger.warning(f"Boot attestation failed: {str(e)}")
        raise e
    except Exception as e:
        logger.error(f"Unexpected error in boot attestation: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Boot attestation failed"
        )


def _validate_luks_request(
    boot_token: str | None,
    hotkey: str | None,
    body: LuksPassphraseRequest,
) -> None:
    """Validate LUKS POST request: boot token, hotkey, volumes, rekey. Raises HTTPException on invalid."""
    if not boot_token:
        detail = "Boot token is required (X-Boot-Token header)"
        logger.warning(f"LUKS request validation failed: {detail}")
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=detail)
    if not hotkey:
        detail = "Hotkey is required"
        logger.warning(f"LUKS request validation failed: {detail}")
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=detail)
    if not body.volumes:
        detail = "volumes is required and must be non-empty"
        logger.warning(f"LUKS request validation failed: {detail}")
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=detail)
    invalid_volumes = [v for v in body.volumes if v not in SUPPORTED_LUKS_VOLUMES]
    if invalid_volumes:
        detail = (
            f"Invalid volume name(s): {invalid_volumes}. Supported: {list(SUPPORTED_LUKS_VOLUMES)}"
        )
        logger.warning(f"LUKS request validation failed: {detail}")
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=detail)
    if body.rekey is not None:
        not_in_volumes = [v for v in body.rekey if v not in body.volumes]
        if not_in_volumes:
            detail = f"rekey must be a subset of volumes; not in volumes: {not_in_volumes}"
            logger.warning(f"LUKS request validation failed: {detail}")
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=detail)
        invalid_rekey = [v for v in body.rekey if v not in SUPPORTED_LUKS_VOLUMES]
        if invalid_rekey:
            detail = f"Invalid rekey volume name(s): {invalid_rekey}. Supported: {list(SUPPORTED_LUKS_VOLUMES)}"
            logger.warning(f"LUKS request validation failed: {detail}")
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=detail)


@router.post("/{vm_name}/luks", response_model=Dict[str, str])
async def sync_luks_passphrases(
    vm_name: str,
    body: LuksPassphraseRequest,
    db: AsyncSession = Depends(get_db_session),
    hotkey: str | None = Header(None, alias=HOTKEY_HEADER),
    boot_token: str | None = Header(None, alias="X-Boot-Token"),
):
    """
    Sync LUKS passphrases for legacy VMs (version < 1.3.0).

    VM sends its volume list; API returns keys for existing volumes, creates keys
    for new volumes, rekeys volumes in the rekey list, and prunes stored keys for
    volumes not in the list. Boot token is validated and consumed on success.
    """
    try:
        _validate_luks_request(boot_token, hotkey, body)
        result = await process_luks_passphrase_request(
            db,
            boot_token,
            hotkey,
            vm_name,
            body.volumes,
            rekey_volume_names=body.rekey,
        )
        return result
    except NonceError as e:
        logger.warning(f"Boot token validation error: {str(e)}")
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail=str(e))
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error in LUKS POST: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to sync/create LUKS passphrases",
        )


@router.post("/{vm_name}/luks/attest", response_model=LuksAttestResponse)
async def attest_luks(
    vm_name: str,
    body: LuksAttestRequest,
    db: AsyncSession = Depends(get_db_session),
    hotkey: str | None = Header(None, alias=HOTKEY_HEADER),
    expected_cert_hash=Depends(extract_client_cert_hash()),
    validated_nonce: str = Depends(require_luks_quote_nonce),
):
    """
    Rotate LUKS passphrases for new-format VMs (version >= 1.3.0).

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
        logger.warning(f"LUKS attest quote verification failed: {str(e)}")
        raise e
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
    _=Depends(require_confirm_nonce),
):
    """
    Confirm or discard pending LUKS passphrase rotation results.

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
    ]
    await settings.redis_client.set(
        TEE_MEASUREMENTS_CACHE_KEY,
        json.dumps([r.model_dump() for r in result]).decode(),
        ex=TEE_MEASUREMENTS_CACHE_TTL,
    )
    return result


@router.get("/maintenance/policy", response_model=MaintenancePolicyResponse)
async def get_maintenance_policy(
    db: AsyncSession = Depends(get_db_session),
    hotkey: str | None = Header(None, alias=HOTKEY_HEADER),
    _: User = Depends(
        get_current_user(purpose="tee", raise_not_found=False, registered_to=settings.netuid)
    ),
):
    """Return the active upgrade window, concurrency limits, and the miner's pending servers."""
    if not hotkey:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Hotkey header required")

    active_window = await get_active_upgrade_window(db)
    window_info: UpgradeWindowInfo | None = None
    current_slots = 0
    server_statuses: list[ServerUpgradeStatus] = []

    if active_window is not None:
        target_version = active_window.target_measurement_version
        window_info = UpgradeWindowInfo(
            id=active_window.id,
            target_measurement_version=target_version,
            upgrade_window_start=str(active_window.upgrade_window_start),
            upgrade_window_end=str(active_window.upgrade_window_end),
            max_concurrent_per_miner=active_window.max_concurrent_per_miner,
        )
        current_slots = await _count_active_maintenance_slots(db, hotkey, active_window)

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

        actual_ip = extract_ip(request)
        if server.ip != actual_ip:
            raise Exception()

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
    nonce=Depends(validate_request_nonce(NoncePurpose.RUNTIME)),
    expected_cert_hash=Depends(extract_client_cert_hash()),
):
    """
    Verify runtime attestation with full measurement validation.
    """
    try:
        server = await check_server_ownership(db, server_id, hotkey)
        actual_ip = extract_ip(request)
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
    except HTTPException:
        raise
    except NonceError as e:
        logger.warning(f"Runtime attestation nonce error: {str(e)}")
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except AttestationError as e:
        logger.warning(f"Runtime attestation failed: {str(e)}")
        raise e
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
