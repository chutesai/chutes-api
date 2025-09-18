"""
FastAPI routes for server management and TDX attestation.
"""

from typing import List, Dict, Any
from fastapi import APIRouter, Depends, HTTPException, Request, status, Header
from sqlalchemy.ext.asyncio import AsyncSession
from loguru import logger

from api.database import get_db_session
from api.config import settings
from api.user.schemas import User
from api.user.service import get_current_user
from api.constants import HOTKEY_HEADER

from api.server.schemas import (
    BootAttestationArgs, RuntimeAttestationArgs, ServerArgs,
    NonceResponse, BootAttestationResponse, RuntimeAttestationResponse,
    Server
)
from api.server.service import (
    create_nonce, process_boot_attestation, register_server,
    check_server_ownership, process_runtime_attestation, get_server_attestation_status,
    list_servers, delete_server, validate_request_nonce
)
from api.server.exceptions import (
    AttestationError, NonceError, ServerNotFoundError, ServerRegistrationError
)
from api.util import extract_ip


router = APIRouter()


# Anonymous Boot Attestation Endpoints (Pre-registration)

@router.get("/nonce", response_model=NonceResponse)
async def get_nonce(
    request: Request
):
    """
    Generate a nonce for boot attestation.
    
    This endpoint is called by VMs during boot before any registration.
    No authentication required as the VM doesn't exist in the system yet.
    """
    try:
        server_ip = extract_ip(request)
        nonce_info = await create_nonce(server_ip)
        
        return NonceResponse(
            nonce=nonce_info["nonce"],
            expires_at=nonce_info["expires_at"]
        )
    except Exception as e:
        logger.error(f"Failed to generate boot nonce: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to generate nonce"
        )


@router.post("/boot/attestation", response_model=BootAttestationResponse)
async def verify_boot_attestation(
    request: Request,
    args: BootAttestationArgs,
    db: AsyncSession = Depends(get_db_session),
    nonce = Depends(validate_request_nonce())
):
    """
    Verify boot attestation and return LUKS passphrase.
    
    This endpoint verifies the TDX quote against expected boot measurements
    and returns the LUKS passphrase for disk decryption if valid.
    """
    try:
        server_ip = extract_ip(request)
        result = await process_boot_attestation(db, server_ip, args, nonce)
        
        return BootAttestationResponse(
            luks_passphrase=result["luks_passphrase"],
            attestation_id=result["attestation_id"],
            verified_at=result["verified_at"]
        )
        
    except NonceError as e:
        logger.warning(f"Boot attestation nonce error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except AttestationError as e:
        logger.warning(f"Boot attestation failed: {str(e)}")
        raise e
    except Exception as e:
        logger.error(f"Unexpected error in boot attestation: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Boot attestation failed"
        )

# Server Management Endpoints (Post-boot via CLI)
# ToDo: Not sure we will want to keep this, ideally want to integrate with miner add-node command
@router.post("/", response_model=Dict[str, str], status_code=status.HTTP_201_CREATED)
async def create_server(
    request: Request,
    args: ServerArgs,
    db: AsyncSession = Depends(get_db_session),
    hotkey: str | None = Header(None, alias=HOTKEY_HEADER),
    _: User = Depends(get_current_user(raise_not_found=False, registered_to=settings.netuid)),
    nonce = Depends(validate_request_nonce())
):
    """
    Register a new server.
    
    This is called via CLI after the server has booted and decrypted its disk.
    Links the server to any existing boot attestation history via server ip.
    """
    try:
        actual_ip = extract_ip(request)
        server = await register_server(db, actual_ip, args, hotkey, nonce)
        
        return {
            "server_id": server.server_id,
            "message": "Server registered successfully"
        }
        
    except ServerRegistrationError as e:
        logger.warning(f"Server registration failed: {str(e)}")
        raise e
    except Exception as e:
        logger.error(f"Unexpected error in server registration: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Server registration failed"
        )

# ToDo: Maybe don't need to expose this
@router.get("/", response_model=List[Dict[str, Any]])
async def list_user_servers(
    db: AsyncSession = Depends(get_db_session),
    hotkey: str | None = Header(None, alias=HOTKEY_HEADER),
    _: User = Depends(get_current_user(raise_not_found=False, registered_to=settings.netuid))
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
                "active": server.active,
                "created_at": server.created_at.isoformat(),
                "updated_at": server.updated_at.isoformat() if server.updated_at else None,
                "metadata": server.metadata
            }
            for server in servers
        ]
        
    except Exception as e:
        logger.error(f"Failed to list servers: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to list servers"
        )


@router.get("/{server_id}", response_model=Dict[str, Any])
async def get_server_details(
    server_id: str,
    db: AsyncSession = Depends(get_db_session),
    hotkey: str | None = Header(None, alias=HOTKEY_HEADER),
    _: User = Depends(get_current_user(raise_not_found=False, registered_to=settings.netuid))
):
    """
    Get details for a specific server.
    """
    try:
        server = await check_server_ownership(db, server_id, hotkey)
        
        return {
            "server_id": server.server_id,
            "name": server.name,
            "ip": server.ip,
            "active": server.active,
            "created_at": server.created_at.isoformat(),
            "updated_at": server.updated_at.isoformat() if server.updated_at else None,
            "metadata": server.metadata,
            "expected_measurements": server.expected_measurements
        }
        
    except ServerNotFoundError as e:
        raise e
    except Exception as e:
        logger.error(f"Failed to get server details: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get server details"
        )


@router.delete("/{server_id}", response_model=Dict[str, str])
async def remove_server(
    server_id: str,
    db: AsyncSession = Depends(get_db_session),
    hotkey: str | None = Header(None, alias=HOTKEY_HEADER),
    _: User = Depends(get_current_user(raise_not_found=False, registered_to=settings.netuid))
):
    """
    Remove a server (mark as inactive).
    """
    try:
        await delete_server(db, server_id, hotkey)
        
        return {
            "server_id": server_id,
            "message": "Server removed successfully"
        }
        
    except ServerNotFoundError as e:
        raise e
    except Exception as e:
        logger.error(f"Failed to remove server: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to remove server"
        )


# Runtime Attestation Endpoints (Post-registration)

@router.get("/{server_id}/nonce", response_model=NonceResponse)
async def get_runtime_nonce(
    request: Request,
    server_id: str,
    db: AsyncSession = Depends(get_db_session),
    hotkey: str | None = Header(None, alias=HOTKEY_HEADER),
    _: User = Depends(get_current_user(raise_not_found=False, registered_to=settings.netuid))
):
    """
    Generate a nonce for runtime attestation.
    """
    try:
        # Verify server ownership
        server = await check_server_ownership(db, server_id, hotkey)
        
        actual_ip = extract_ip(request)
        if server.ip != actual_ip:
            raise Exception()

        nonce_info = await create_nonce(server.ip)
        
        return NonceResponse(
            nonce=nonce_info["nonce"],
            expires_at=nonce_info["expires_at"]
        )
        
    except ServerNotFoundError as e:
        raise e
    except Exception as e:
        logger.error(f"Failed to generate runtime nonce: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to generate nonce"
        )


@router.post("/{server_id}/attestation", response_model=RuntimeAttestationResponse)
async def verify_runtime_attestation(
    request: Request,
    server_id: str,
    args: RuntimeAttestationArgs,
    db: AsyncSession = Depends(get_db_session),
    hotkey: str | None = Header(None, alias=HOTKEY_HEADER),
    _: User = Depends(get_current_user(raise_not_found=False, registered_to=settings.netuid)),
    nonce = Depends(validate_request_nonce())
):
    """
    Verify runtime attestation with full measurement validation.
    """
    try:
        actual_ip = extract_ip(request)
        result = await process_runtime_attestation(db, server_id, actual_ip, args, hotkey)
        
        return RuntimeAttestationResponse(
            attestation_id=result["attestation_id"],
            verified_at=result["verified_at"],
            status=result["status"]
        )
        
    except ServerNotFoundError as e:
        raise e
    except NonceError as e:
        logger.warning(f"Runtime attestation nonce error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except AttestationError as e:
        logger.warning(f"Runtime attestation failed: {str(e)}")
        raise e
    except Exception as e:
        logger.error(f"Unexpected error in runtime attestation: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Runtime attestation failed"
        )

# ToDo: Also likely to remove this
@router.get("/{server_id}/attestation/status", response_model=Dict[str, Any])
async def get_attestation_status(
    server_id: str,
    db: AsyncSession = Depends(get_db_session),
    hotkey: str | None = Header(None, alias=HOTKEY_HEADER),
    _: User = Depends(get_current_user(raise_not_found=False, registered_to=settings.netuid))
):
    """
    Get current attestation status for a server.
    """
    try:
        status_info = await get_server_attestation_status(db, server_id, hotkey)
        return status_info
        
    except ServerNotFoundError as e:
        raise e
    except Exception as e:
        logger.error(f"Failed to get attestation status: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get attestation status"
        )