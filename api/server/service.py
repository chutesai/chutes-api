"""
Core server management and TDX attestation logic.
"""

import asyncio
import base64
from datetime import datetime, timezone, timedelta
import json
import tempfile
from typing import Dict, Any
from fastapi import HTTPException, Header, Request, status
from loguru import logger
from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.exc import IntegrityError

from api.config import settings
from api.constants import NONCE_HEADER
from api.server.quote import BootTdxQuote, RuntimeTdxQuote, TdxQuote, TdxVerificationResult
from api.server.schemas import (
    Server,
    ServerAttestation,
    BootAttestation,
    BootAttestationArgs,
    RuntimeAttestationArgs,
    ServerArgs,
)
from api.server.exceptions import (
    GpuEvidenceError,
    InvalidGpuEvidenceError,
    InvalidQuoteError,
    MeasurementMismatchError,
    NonceError,
    ServerNotFoundError,
    ServerRegistrationError,
)
from api.server.util import (
    extract_nonce,
    verify_measurements,
    get_luks_passphrase,
    generate_nonce,
    get_nonce_expiry_seconds,
    verify_quote_signature,
)
from api.util import extract_ip


async def create_nonce(server_ip: str) -> Dict[str, str]:
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
    redis_value = f"{server_ip}"

    await settings.redis_client.setex(redis_key, expiry_seconds, redis_value)

    expires_at = datetime.now(timezone.utc).replace(microsecond=0) + timedelta(
        seconds=expiry_seconds
    )

    logger.info(f"Created nonce: {nonce[:8]}... for server {server_ip}")

    return {"nonce": nonce, "expires_at": expires_at.isoformat()}


async def validate_and_consume_nonce(nonce_value: str, server_ip: str) -> None:
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
        stored_server = redis_value.decode()
    except (ValueError, AttributeError):
        raise NonceError("Invalid nonce format")

    # Validate server ID
    expected_server = server_ip
    if stored_server != expected_server:
        raise NonceError(f"Nonce server mismatch: expected {expected_server}, got {stored_server}")

    # Consume the nonce by deleting it
    deleted = await settings.redis_client.delete(redis_key)
    if not deleted:
        raise NonceError("Nonce was already consumed")

    logger.info(f"Validated and consumed nonce: {nonce_value[:8]}...")


async def validate_request_nonce():
    async def _validate_request_nonce(
        request: Request, nonce: str | None = Header(None, alias=NONCE_HEADER)
    ):
        server_ip = extract_ip(request)

        try:
            await validate_and_consume_nonce(nonce, server_ip)

            return nonce
        except NonceError as e:
            logger.error(f"Request nonce validation failed: {e}")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid nonce supplied"
            )

    return _validate_request_nonce


async def verify_quote(quote: TdxQuote, expected_nonce: str) -> TdxVerificationResult:
    # Validate nonce
    nonce = extract_nonce(quote)
    if nonce != expected_nonce:
        raise NonceError("Quote nonce does not match expected nonce.")

    result = await verify_quote_signature(quote)

    verify_measurements(quote)

    return result


async def verify_gpu_evidence(evidence: list[Dict[str, str]], expected_nonce: str) -> None:
    try:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json") as fp:
            json.dump(evidence, fp)
            fp.flush()

            verify_gpus_cmd = ["chutes-nvattest", "--nonce", expected_nonce, "--evidence", fp.name]

            process = await asyncio.create_subprocess_exec(*verify_gpus_cmd)

            await asyncio.gather(process.wait())

            if process.returncode != 0:
                raise InvalidGpuEvidenceError()

            logger.info("GPU evidence verified successfully.")

    except FileNotFoundError as e:
        logger.error(f"Failed to verify GPU evidence.  chutes-nvattest command not found?:\n{e}")
        raise GpuEvidenceError("Failed to verify GPU evidence.")
    except Exception as e:
        logger.error(f"Unexepected exception encoutnered verifying GPU evidence:\n{e}")
        raise GpuEvidenceError("Encountered an unexpected exception verifying GPU evidence.")


async def process_boot_attestation(
    db: AsyncSession, server_ip: str, args: BootAttestationArgs, nonce: str
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
    logger.info(f"Processing boot attestation for server:: {server_ip}")

    # Parse and verify quote
    try:  # Verify quote signature
        quote = BootTdxQuote.from_base64(args.quote)
        result = await verify_quote(quote, nonce)

        # Create boot attestation record
        boot_attestation = BootAttestation(
            quote_data=args.quote,
            server_ip=server_ip,
            mrtd=quote.mrtd,
            verification_result=result.to_dict(),
            verified=True,
            nonce_used=nonce,
            verified_at=func.now(),
        )

        db.add(boot_attestation)
        await db.commit()
        await db.refresh(boot_attestation)

        logger.success(f"Boot attestation successful: {boot_attestation.attestation_id}")

        return {
            "luks_passphrase": get_luks_passphrase(),
            "attestation_id": boot_attestation.attestation_id,
            "verified_at": boot_attestation.verified_at.isoformat(),
        }

    except (InvalidQuoteError, MeasurementMismatchError) as e:
        # Create failed attestation record
        boot_attestation = BootAttestation(
            quote_data=args.quote,
            server_ip=server_ip,
            verified=False,
            verification_error=str(e.detail),
            nonce_used=nonce,
        )

        db.add(boot_attestation)
        await db.commit()

        logger.error(f"Boot attestation failed: {str(e)}")
        raise


async def register_server(
    db: AsyncSession, actual_ip: str, args: ServerArgs, miner_hotkey: str, expected_nonce: str
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
        quote = RuntimeTdxQuote.from_base64(args.quote)
        await verify_quote(quote, expected_nonce)

        gpu_evidence = json.loads(base64.b64decode(args.evidence))
        await verify_gpu_evidence(gpu_evidence, expected_nonce)

        server = Server(name=args.name, ip=actual_ip, miner_hotkey=miner_hotkey)

        db.add(server)
        await db.commit()
        await db.refresh(server)

        logger.success(f"Registered server: {server.server_id} for miner: {miner_hotkey}")
        return server
    except (InvalidQuoteError, MeasurementMismatchError) as e:
        await db.rollback()
        logger.error(f"Server registration failed:\n{e}")
        raise ServerRegistrationError("Server registartion failed: invalid quote")
    except IntegrityError as e:
        await db.rollback()
        logger.error(f"Server registration failed: {str(e)}")
        raise ServerRegistrationError("Server registration failed - constraint violation")
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
    query = select(Server).where(Server.server_id == server_id, Server.miner_hotkey == miner_hotkey)

    result = await db.execute(query)
    server = result.scalar_one_or_none()

    if not server:
        raise ServerNotFoundError(server_id)

    return server


async def process_runtime_attestation(
    db: AsyncSession,
    server_id: str,
    actual_ip: str,
    args: RuntimeAttestationArgs,
    miner_hotkey: str,
    expected_nonce: str,
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

    if server.ip != actual_ip:
        raise Exception()

    # Parse and verify quote
    try:
        # Verify quote signature
        quote = RuntimeTdxQuote.from_base64(args.quote)
        result = await verify_quote(quote, expected_nonce)

        # Create runtime attestation record
        attestation = ServerAttestation(
            server_id=server_id,
            quote_data=args.quote,
            mrtd=quote.mrtd,
            rtmrs=quote.rtmrs,
            verification_result=result.to_dict(),
            verified=True,
            nonce_used=expected_nonce,
            verified_at=func.now(),
        )

        db.add(attestation)
        await db.commit()
        await db.refresh(attestation)

        logger.success(f"Runtime attestation successful: {attestation.attestation_id}")

        return {
            "attestation_id": attestation.attestation_id,
            "verified_at": attestation.verified_at.isoformat(),
            "status": "verified",
        }

    except (InvalidQuoteError, MeasurementMismatchError) as e:
        # Create failed attestation record
        attestation = ServerAttestation(
            server_id=server_id,
            quote_data=args.quote,
            verified=False,
            verification_error=str(e.detail),
            nonce_used=expected_nonce,
        )

        db.add(attestation)
        await db.commit()

        logger.error(f"Runtime attestation failed: {str(e)}")
        raise


async def get_server_attestation_status(
    db: AsyncSession, server_id: str, miner_hotkey: str
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
    query = (
        select(ServerAttestation)
        .where(ServerAttestation.server_id == server_id)
        .order_by(ServerAttestation.created_at.desc())
        .limit(1)
    )

    result = await db.execute(query)
    latest_attestation = result.scalar_one_or_none()

    status = {
        "server_id": server_id,
        "server_name": server.name,
        "last_attestation": None,
        "attestation_status": "never_attested",
    }

    if latest_attestation:
        status["last_attestation"] = {
            "attestation_id": latest_attestation.attestation_id,
            "verified": latest_attestation.verified,
            "created_at": latest_attestation.created_at.isoformat(),
            "verified_at": latest_attestation.verified_at.isoformat()
            if latest_attestation.verified_at
            else None,
            "verification_error": latest_attestation.verification_error,
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
    query = (
        select(Server).where(Server.miner_hotkey == miner_hotkey).order_by(Server.created_at.desc())
    )

    result = await db.execute(query)
    servers = result.scalars().all()

    logger.info(f"Found {len(servers)} servers for miner: {miner_hotkey}")
    return servers


async def delete_server(db: AsyncSession, server_id: str, miner_hotkey: str) -> bool:
    """
    Delete a server.

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

    # NOTE: Do we want to do a soft delete to keep attestation history?
    db.delete(server)

    await db.commit()

    logger.info(f"Deleted server: {server_id} [{server.name}({server.ip})]")
    return True
