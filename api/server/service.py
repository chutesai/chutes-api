"""
Core server management and TDX attestation logic.
"""

import base64
from datetime import datetime, timezone, timedelta
from typing import Optional, Dict, Any
from loguru import logger
from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.exc import IntegrityError

from api.config import settings
from api.server.quote import BootTdxQuote, RuntimeTdxQuote, TdxQuote, TdxVerificationResult
from api.server.schemas import (
    Server, ServerAttestation, BootAttestation,
    BootAttestationArgs, RuntimeAttestationArgs, ServerArgs
)
from api.server.exceptions import (
    InvalidQuoteError, MeasurementMismatchError, 
    NonceError, ServerNotFoundError, ServerRegistrationError
)
from api.server.util import (
    extract_nonce, verify_measurements, get_luks_passphrase,
    generate_nonce, get_nonce_expiry_seconds, verify_quote_signature
)


async def create_nonce(
    attestation_type: str, 
    server_id: Optional[str] = None
) -> Dict[str, str]:
    """
    Create a new attestation nonce using Redis.
    
    Args:
        attestation_type: 'boot' or 'runtime'
        server_id: Optional server ID for runtime attestations
        
    Returns:
        Dictionary with nonce and expiry info
    """
    nonce = generate_nonce()
    expiry_seconds = get_nonce_expiry_seconds()
    
    # Use Redis to store nonce with TTL
    redis_key = f"nonce:{nonce}"
    redis_value = f"{attestation_type}:{server_id or 'boot'}"
    
    await settings.redis_client.setex(redis_key, expiry_seconds, redis_value)
    
    expires_at = datetime.now(timezone.utc).replace(microsecond=0) + timedelta(seconds=expiry_seconds)
    
    logger.info(f"Created {attestation_type} nonce: {nonce[:8]}... for server {server_id}")
    
    return {
        "nonce": nonce,
        "expires_at": expires_at.isoformat()
    }


async def validate_and_consume_nonce(
    nonce_value: str, 
    attestation_type: str,
    server_id: Optional[str] = None
) -> None:
    """
    Validate and consume a nonce using Redis.
    
    Args:
        nonce_value: Nonce to validate
        attestation_type: Expected attestation type
        server_id: Expected server ID (for runtime attestations)
        
    Raises:
        NonceError: If nonce is invalid, expired, or already used
    """
    redis_key = f"nonce:{nonce_value}"
    
    # Get and delete nonce atomically
    redis_value = await settings.redis_client.get(redis_key)
    
    if not redis_value:
        raise NonceError("Nonce not found or expired")
    
    # Parse the stored value
    try:
        stored_type, stored_server = redis_value.decode().split(":", 1)
    except (ValueError, AttributeError):
        raise NonceError("Invalid nonce format")
    
    # Validate attestation type
    if stored_type != attestation_type:
        raise NonceError(f"Nonce type mismatch: expected {attestation_type}, got {stored_type}")
    
    # Validate server ID
    expected_server = server_id or 'boot'
    if stored_server != expected_server:
        raise NonceError(f"Nonce server mismatch: expected {expected_server}, got {stored_server}")
    
    # Consume the nonce by deleting it
    deleted = await settings.redis_client.delete(redis_key)
    if not deleted:
        raise NonceError("Nonce was already consumed")
    
    logger.info(f"Validated and consumed nonce: {nonce_value[:8]}...")

async def verify_quote(quote: TdxQuote, server: Optional[Server] = None) -> TdxVerificationResult:
    # Validate nonce
    nonce = extract_nonce(quote)
    server_id = server.server_id if server else None
    await validate_and_consume_nonce(nonce, quote.quote_type, server_id)

    result = await verify_quote_signature(quote)

    verify_measurements(quote)

    return result

async def process_boot_attestation(
    db: AsyncSession, 
    args: BootAttestationArgs
) -> Dict[str, str]:
    """
    Process a boot attestation request.
    
    Args:
        db: Database session
        args: Boot attestation arguments
        
    Returns:
        Dictionary containing LUKS passphrase and attestation info
        
    Raises:
        NonceError: If nonce validation fails
        InvalidQuoteError: If quote is invalid
        MeasurementMismatchError: If measurements don't match
    """
    logger.info(f"Processing boot attestation for vm id:: {args.vm_id}")
    
    # Parse and verify quote
    nonce = None
    try:# Verify quote signature

        quote = BootTdxQuote.from_base64(args.quote)
        nonce = extract_nonce(quote)
        result = await verify_quote(quote)
        
        # Create boot attestation record
        boot_attestation = BootAttestation(
            quote_data=args.quote,
            vm_id=args.vm_id,
            mrtd=quote.mrtd,
            verification_result=result.to_dict(),
            verified=True,
            nonce_used=nonce,
            verified_at=func.now()
        )
        
        db.add(boot_attestation)
        await db.commit()
        await db.refresh(boot_attestation)
        
        logger.success(f"Boot attestation successful: {boot_attestation.attestation_id}")
        
        return {
            "luks_passphrase": get_luks_passphrase(),
            "attestation_id": boot_attestation.attestation_id,
            "verified_at": boot_attestation.verified_at.isoformat()
        }
        
    except (InvalidQuoteError, MeasurementMismatchError) as e:
        # Create failed attestation record
        boot_attestation = BootAttestation(
            quote_data=args.quote,
            vm_id=args.vm_id,
            verified=False,
            verification_error=str(e.detail),
            nonce_used=nonce
        )
        
        db.add(boot_attestation)
        await db.commit()
        
        logger.error(f"Boot attestation failed: {str(e)}")
        raise


async def register_server(
    db: AsyncSession, 
    args: ServerArgs, 
    miner_hotkey: str
) -> Server:
    """
    Register a new server.
    
    Args:
        db: Database session
        args: Server registration arguments
        miner_hotkey: Miner hotkey from authentication
        
    Returns:
        Created server object
        
    Raises:
        ServerRegistrationError: If registration fails
    """
    try:
        server = Server(
            name=args.name,
            vm_id=args.vm_id,
            miner_hotkey=miner_hotkey,
            metadata=args.metadata
        )
        
        db.add(server)
        await db.commit()
        await db.refresh(server)
        
        logger.success(f"Registered server: {server.server_id} for miner: {miner_hotkey}")
        return server
        
    except IntegrityError as e:
        await db.rollback()
        logger.error(f"Server registration failed: {str(e)}")
        raise ServerRegistrationError("Server registration failed - duplicate vm id or other constraint violation")
    except Exception as e:
        await db.rollback()
        logger.error(f"Unexpected error during server registration: {str(e)}")
        raise ServerRegistrationError(f"Server registration failed: {str(e)}")


async def check_server_ownership(db: AsyncSession, server_id: str, miner_hotkey: str) -> Server:
    """
    Get a server by ID, ensuring it belongs to the authenticated miner.
    
    Args:
        db: Database session
        server_id: Server ID
        miner_hotkey: Authenticated miner hotkey
        
    Returns:
        Server object
        
    Raises:
        ServerNotFoundError: If server not found or doesn't belong to miner
    """
    query = select(Server).where(
        Server.server_id == server_id,
        Server.miner_hotkey == miner_hotkey
    )
    
    result = await db.execute(query)
    server = result.scalar_one_or_none()
    
    if not server:
        raise ServerNotFoundError(server_id)
    
    return server


async def process_runtime_attestation(
    db: AsyncSession,
    server_id: str,
    args: RuntimeAttestationArgs,
    miner_hotkey: str
) -> Dict[str, str]:
    """
    Process a runtime attestation request.
    
    Args:
        db: Database session
        server_id: Server ID
        args: Runtime attestation arguments
        miner_hotkey: Authenticated miner hotkey
        
    Returns:
        Dictionary containing attestation status info
        
    Raises:
        ServerNotFoundError: If server not found
        NonceError: If nonce validation fails
        InvalidQuoteError: If quote is invalid
        MeasurementMismatchError: If measurements don't match
    """
    logger.info(f"Processing runtime attestation for server: {server_id}")
    
    # Get server and verify ownership
    server = await check_server_ownership(db, server_id, miner_hotkey)
    
    # Parse and verify quote
    nonce = None
    try:
        # Verify quote signature
        quote = RuntimeTdxQuote.from_base64(args.quote)
        nonce = extract_nonce(quote)
        result = await verify_quote(quote, server)
        
        # Create runtime attestation record
        attestation = ServerAttestation(
            server_id=server_id,
            quote_data=args.quote,
            mrtd=quote.mrtd,
            rtmrs=quote.rtmrs,
            verification_result=result.to_dict(),
            verified=True,
            nonce_used=nonce,
            verified_at=func.now()
        )
        
        db.add(attestation)
        await db.commit()
        await db.refresh(attestation)
        
        logger.success(f"Runtime attestation successful: {attestation.attestation_id}")
        
        return {
            "attestation_id": attestation.attestation_id,
            "verified_at": attestation.verified_at.isoformat(),
            "status": "verified"
        }
        
    except (InvalidQuoteError, MeasurementMismatchError) as e:
        # Create failed attestation record
        attestation = ServerAttestation(
            server_id=server_id,
            quote_data=args.quote,
            verified=False,
            verification_error=str(e.detail),
            nonce_used=nonce
        )
        
        db.add(attestation)
        await db.commit()
        
        logger.error(f"Runtime attestation failed: {str(e)}")
        raise


async def get_server_attestation_status(
    db: AsyncSession,
    server_id: str,
    miner_hotkey: str
) -> Dict[str, Any]:
    """
    Get the current attestation status for a server.
    
    Args:
        db: Database session
        server_id: Server ID
        miner_hotkey: Authenticated miner hotkey
        
    Returns:
        Dictionary containing attestation status
    """
    # Verify server ownership
    server = await check_server_ownership(db, server_id, miner_hotkey)
    
    # Get latest attestation
    query = select(ServerAttestation).where(
        ServerAttestation.server_id == server_id
    ).order_by(ServerAttestation.created_at.desc()).limit(1)
    
    result = await db.execute(query)
    latest_attestation = result.scalar_one_or_none()
    
    status = {
        "server_id": server_id,
        "server_name": server.name,
        "active": server.active,
        "last_attestation": None,
        "attestation_status": "never_attested"
    }
    
    if latest_attestation:
        status["last_attestation"] = {
            "attestation_id": latest_attestation.attestation_id,
            "verified": latest_attestation.verified,
            "created_at": latest_attestation.created_at.isoformat(),
            "verified_at": latest_attestation.verified_at.isoformat() if latest_attestation.verified_at else None,
            "verification_error": latest_attestation.verification_error
        }
        status["attestation_status"] = "verified" if latest_attestation.verified else "failed"
    
    return status


async def list_servers(db: AsyncSession, miner_hotkey: str) -> list[Server]:
    """
    List all servers for a miner.
    
    Args:
        db: Database session
        miner_hotkey: Authenticated miner hotkey
        
    Returns:
        List of server objects
    """
    query = select(Server).where(
        Server.miner_hotkey == miner_hotkey,
        Server.active == True
    ).order_by(Server.created_at.desc())
    
    result = await db.execute(query)
    servers = result.scalars().all()
    
    logger.info(f"Found {len(servers)} servers for miner: {miner_hotkey}")
    return servers


async def delete_server(db: AsyncSession, server_id: str, miner_hotkey: str) -> bool:
    """
    Delete a server (mark as inactive).
    
    Args:
        db: Database session
        server_id: Server ID
        miner_hotkey: Authenticated miner hotkey
        
    Returns:
        True if deleted successfully
        
    Raises:
        ServerNotFoundError: If server not found
    """
    server = await check_server_ownership(db, server_id, miner_hotkey)
    
    # Mark as inactive instead of hard delete to preserve attestation history
    server.active = False
    server.updated_at = func.now()
    
    await db.commit()
    
    logger.info(f"Marked server as inactive: {server_id}")
    return True