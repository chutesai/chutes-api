"""
FastAPI routes for server management and TDX attestation.
"""

from typing import List, Dict, Any
from fastapi import APIRouter, Depends, HTTPException, Request, status, Header, Query
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
    NonceResponse,
    BootAttestationResponse,
    RuntimeAttestationResponse,
    LuksPassphraseRequest,
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
    list_servers,
    delete_server,
    validate_request_nonce,
    process_luks_passphrase_request,
)
from api.server.util import extract_client_cert_hash, get_luks_passphrase
from api.server.exceptions import (
    AttestationError,
    NonceError,
    ServerNotFoundError,
    ServerRegistrationError,
)
from api.miner.util import is_miner_blacklisted
from api.util import extract_ip, is_valid_host


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
    """
    try:
        server_ip = extract_ip(request)
        boot_token = await process_boot_attestation(db, server_ip, args, nonce, expected_cert_hash)

        return BootAttestationResponse(key=get_luks_passphrase(), boot_token=boot_token)
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
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Boot token is required (X-Boot-Token header)",
        )
    if not hotkey:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Hotkey is required",
        )
    if not body.volumes:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="volumes is required and must be non-empty",
        )
    invalid_volumes = [v for v in body.volumes if v not in SUPPORTED_LUKS_VOLUMES]
    if invalid_volumes:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid volume name(s): {invalid_volumes}. Supported: {list(SUPPORTED_LUKS_VOLUMES)}",
        )
    if body.rekey is not None:
        not_in_volumes = [v for v in body.rekey if v not in body.volumes]
        if not_in_volumes:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"rekey must be a subset of volumes; not in volumes: {not_in_volumes}",
            )
        invalid_rekey = [v for v in body.rekey if v not in SUPPORTED_LUKS_VOLUMES]
        if invalid_rekey:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid rekey volume name(s): {invalid_rekey}. Supported: {list(SUPPORTED_LUKS_VOLUMES)}",
            )


@router.post("/{vm_name}/luks", response_model=Dict[str, str])
async def sync_luks_passphrases(
    vm_name: str,
    body: LuksPassphraseRequest,
    db: AsyncSession = Depends(get_db_session),
    hotkey: str | None = Header(None, alias=HOTKEY_HEADER),
    boot_token: str | None = Header(None, alias="X-Boot-Token"),
):
    """
    Sync LUKS passphrases: VM sends volume list; API returns keys for existing volumes,
    creates keys for new volumes, rekeys volumes in rekey list, and prunes stored keys
    for volumes not in the list. Boot token is consumed after successful POST.
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


# ToDo: Maybe don't need to expose this
@router.get("/", response_model=List[Dict[str, Any]])
async def list_user_servers(
    db: AsyncSession = Depends(get_db_session),
    hotkey: str | None = Header(None, alias=HOTKEY_HEADER),
    _: User = Depends(
        get_current_user(purpose="tee", raise_not_found=False, registered_to=settings.netuid)
    ),
):
    """
    List all servers for the authenticated miner.
    """
    try:
        servers = await list_servers(db, hotkey)

        return [
            {
                "server_id": server.server_id,
                "name": server.name,
                "ip": server.ip,
                "created_at": server.created_at.isoformat(),
                "updated_at": server.updated_at.isoformat() if server.updated_at else None,
            }
            for server in servers
        ]

    except Exception as e:
        logger.error(f"Failed to list servers: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to list servers"
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

        return {
            "server_id": server.server_id,
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
        logger.error(f"Failed to get server details: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to get server details"
        )


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
