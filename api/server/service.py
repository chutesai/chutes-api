"""
Core server management and TDX attestation logic.
"""

import asyncio
import pybase64 as base64
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
from api.constants import NONCE_HEADER, NoncePurpose
from api.gpu import SUPPORTED_GPUS
from api.node.util import _track_nodes
from api.server.client import TeeServerClient
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
    AttestationError,
    GetEvidenceError,
    GpuEvidenceError,
    InvalidClientCertError,
    InvalidGpuEvidenceError,
    InvalidQuoteError,
    MeasurementMismatchError,
    NonceError,
    ServerNotFoundError,
    ServerRegistrationError,
)
from api.server.util import (
    _track_server,
    extract_report_data,
    verify_measurements,
    generate_nonce,
    get_nonce_expiry_seconds,
    verify_quote_signature,
    verify_result,
    get_cache_passphrase,
    create_or_update_cache_passphrase,
)
from api.util import extract_ip


async def create_nonce(server_ip: str, purpose: NoncePurpose) -> Dict[str, str]:
    """
    Create a new attestation nonce using Redis.

    Args:
        server_ip: IP address of the server/instance requesting the nonce
        purpose: Purpose of the nonce (NoncePurpose enum value)

    Returns:
        Dictionary with nonce and expiry info
    """
    nonce = generate_nonce()
    expiry_seconds = get_nonce_expiry_seconds()

    # Use Redis to store nonce with TTL
    # Store as JSON to include both server_ip and purpose
    redis_key = f"nonce:{nonce}"
    redis_value = json.dumps({"server_ip": server_ip, "purpose": purpose.value})

    await settings.redis_client.setex(redis_key, expiry_seconds, redis_value)

    expires_at = datetime.now(timezone.utc).replace(microsecond=0) + timedelta(
        seconds=expiry_seconds
    )

    logger.info(f"Created nonce: {nonce[:8]}... for server {server_ip} with purpose {purpose}")

    return {"nonce": nonce, "expires_at": expires_at.isoformat()}


async def validate_and_consume_nonce(nonce_value: str, server_ip: str, purpose: NoncePurpose) -> None:
    """
    Validate and consume a nonce using Redis.

    Args:
        nonce_value: Nonce to validate
        server_ip: Expected server IP address
        purpose: Expected purpose for the nonce (NoncePurpose enum value)

    Raises:
        NonceError: If nonce is invalid, expired, already used, or purpose/server mismatch
    """
    redis_key = f"nonce:{nonce_value}"

    # Get and delete nonce atomically
    redis_value = await settings.redis_client.get(redis_key)

    if not redis_value:
        raise NonceError("Nonce not found or expired")

    # Parse the stored value
    try:
        stored_data = json.loads(redis_value.decode())
        
        # Handle legacy format (just server_ip as string) for backward compatibility
        if isinstance(stored_data, str):
            stored_server = stored_data
            stored_purpose = None
        else:
            stored_server = stored_data.get("server_ip")
            stored_purpose = stored_data.get("purpose")
    except (ValueError, AttributeError, json.JSONDecodeError):
        raise NonceError("Invalid nonce format")

    # Validate server IP
    if stored_server != server_ip:
        raise NonceError(f"Nonce server mismatch: expected {server_ip}, got {stored_server}")

    # Validate purpose (if stored nonce has a purpose, it must match)
    if stored_purpose and stored_purpose != purpose.value:
        raise NonceError(
            f"Nonce purpose mismatch: expected {purpose.value}, got {stored_purpose}. "
            f"Nonces are purpose-specific and cannot be reused across different operations."
        )

    # Consume the nonce by deleting it
    deleted = await settings.redis_client.delete(redis_key)
    if not deleted:
        raise NonceError("Nonce was already consumed")

    logger.info(f"Validated and consumed nonce: {nonce_value[:8]}... for purpose {purpose}")


def validate_request_nonce(purpose: NoncePurpose):
    """
    Create a nonce validator dependency that validates nonces for a specific purpose.
    
    Args:
        purpose: The expected purpose for the nonce (NoncePurpose enum value)
    
    Returns:
        A FastAPI dependency function that validates the nonce
    """
    async def _validate_request_nonce(
        request: Request, nonce: str | None = Header(None, alias=NONCE_HEADER)
    ):
        server_ip = extract_ip(request)

        try:
            await validate_and_consume_nonce(nonce, server_ip, purpose)

            return nonce
        except NonceError as e:
            logger.error(f"Request nonce validation failed: {e}")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid nonce supplied"
            )

    return _validate_request_nonce


async def verify_quote(
    quote: TdxQuote, expected_nonce: str, expected_cert_hash: str
) -> TdxVerificationResult:
    # Validate nonce
    nonce, cert_hash = extract_report_data(quote)

    if nonce != expected_nonce:
        logger.info(f"Nonce error:  {nonce} =/= {expected_nonce}")
        raise NonceError("Quote nonce does not match expected nonce.")

    if cert_hash != expected_cert_hash:
        raise InvalidClientCertError()

    # Verify the quote using DCAP
    result = await verify_quote_signature(quote)
    # Verify the quote against the result to ensure it was parsed properly
    verify_result(quote, result)
    # Verify the quote against configured MRTD/RMTRs
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


async def generate_and_store_boot_token(miner_hotkey: str, vm_name: str) -> str:
    """
    Generate and store a boot token for a verified VM.

    Args:
        miner_hotkey: Miner hotkey that owns this VM
        vm_name: VM name/identifier

    Returns:
        Boot token string
    """
    boot_token = generate_nonce()
    redis_key = f"boot_token:{boot_token}"
    # Store boot token with miner_hotkey:vm_name (10 minute TTL)
    boot_token_value = f"{miner_hotkey}:{vm_name}"
    await settings.redis_client.setex(redis_key, 10 * 60, boot_token_value)
    logger.info(f"Generated boot token for VM {vm_name} (miner: {miner_hotkey})")

    return boot_token


async def process_boot_attestation(
    db: AsyncSession,
    server_ip: str,
    args: BootAttestationArgs,
    nonce: str,
    expected_cert_hash: str,
) -> str:
    """
    Process a boot attestation request.

    Args:
        db: Database session
        server_ip: Server IP address
        args: Boot attestation arguments (includes miner_hotkey and vm_name)
        nonce: Validated nonce
        expected_cert_hash: Expected certificate hash

    Returns:
        Boot token for subsequent cache passphrase retrieval

    Raises:
        NonceError: If nonce validation fails
        InvalidQuoteError: If quote is invalid
        MeasurementMismatchError: If measurements don't match
    """
    logger.info(
        f"Processing boot attestation for VM {args.vm_name} (miner: {args.miner_hotkey}, IP: {server_ip})"
    )

    # Parse and verify quote
    try:  # Verify quote signature
        quote = BootTdxQuote.from_base64(args.quote)
        await verify_quote(quote, nonce, expected_cert_hash)

        # Create boot attestation record
        boot_attestation = BootAttestation(
            quote_data=args.quote,
            server_ip=server_ip,
            created_at=func.now(),
            verified_at=func.now(),
        )

        db.add(boot_attestation)
        await db.commit()
        await db.refresh(boot_attestation)

        logger.success(f"Boot attestation successful: {boot_attestation.attestation_id}")

        # Generate boot token for this verified VM
        boot_token = await generate_and_store_boot_token(args.miner_hotkey, args.vm_name)

        return boot_token

    except (InvalidQuoteError, MeasurementMismatchError) as e:
        # Create failed attestation record
        boot_attestation = BootAttestation(
            quote_data=args.quote,
            server_ip=server_ip,
            verification_error=str(e.detail),
            created_at=func.now(),
        )

        db.add(boot_attestation)
        await db.commit()

        logger.error(f"Boot attestation failed: {str(e)}")
        raise


async def register_server(db: AsyncSession, args: ServerArgs, miner_hotkey: str):
    try:
        server = await _track_server(db, args.id, args.host, miner_hotkey, is_tee=True)

        # Set the attributes we can't get from pynvml
        for gpu in args.gpus:
            gpu_info = SUPPORTED_GPUS[gpu.gpu_identifier]
            for key in ["processors", "max_threads_per_processor"]:
                setattr(gpu, key, gpu_info.get(key))

        # Start verification process
        await verify_server(db, server, miner_hotkey)

        # Track nodes once verified
        await _track_nodes(db, miner_hotkey, server.server_id, args.gpus, "0", func.now())

    except AttestationError as e:
        # Preserve the specific error message from the AttestationError
        error_detail = e.detail if hasattr(e, "detail") else str(e)
        logger.error(
            f"Server registration failed - attestation error: server_id={args.id} host={args.host} miner_hotkey={miner_hotkey} error={error_detail}"
        )
        raise ServerRegistrationError(f"Server registration failed - {error_detail}")
    except IntegrityError as e:
        await db.rollback()
        logger.error(
            f"Server registration failed - IntegrityError: server_id={args.id} host={args.host} miner_hotkey={miner_hotkey} error={str(e)}"
        )
        raise ServerRegistrationError(
            "Server registration failed - database constraint violation. This may indicate a duplicate server ID, invalid miner configuration, or other database conflict. Please contact support with your server ID and miner hotkey."
        )
    except Exception as e:
        await db.rollback()
        logger.error(
            f"Unexpected error during server registration: server_id={args.id} host={args.host} miner_hotkey={miner_hotkey} error={str(e)}",
            exc_info=True,
        )
        raise ServerRegistrationError(
            "Server registration failed - unexpected error occurred. Please contact support with your server ID and miner hotkey."
        )


async def verify_server(db: AsyncSession, server: Server, miner_hotkey: str) -> None:
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
    failure_reason = ""
    quote = None
    try:
        client = TeeServerClient(server)

        nonce = generate_nonce()
        logger.info(
            f"Verifying server server_id={server.server_id} ip={server.ip} miner_hotkey={miner_hotkey} with nonce {nonce}"
        )
        quote, gpu_evidence, expected_cert_hash = await client.get_evidence(nonce)

        await verify_quote(quote, nonce, expected_cert_hash)

        await verify_gpu_evidence(gpu_evidence, nonce)

        logger.success(
            f"Verified server server_id={server.server_id} ip={server.ip} for miner: {miner_hotkey}"
        )

        # Create attestation record
        server_attestation = ServerAttestation(
            quote_data=base64.b64encode(quote.raw_bytes).decode("utf-8"),
            server_id=server.server_id,
            created_at=func.now(),
            verified_at=func.now(),
        )

        db.add(server_attestation)
        await db.commit()
        await db.refresh(server_attestation)

    except GetEvidenceError as e:
        failure_reason = "Failed to get attestation evidence."
        logger.error(
            f"Server verification failed - GetEvidenceError: server_id={server.server_id} ip={server.ip} miner_hotkey={miner_hotkey} error={e.detail}"
        )
        raise e
    except (InvalidQuoteError, MeasurementMismatchError) as e:
        logger.error(
            f"Server verification failed - quote error: server_id={server.server_id} ip={server.ip} miner_hotkey={miner_hotkey} error={e.detail}"
        )
        failure_reason = "Server verification failed: invalid quote"
        raise e
    except InvalidGpuEvidenceError as e:
        logger.error(
            f"Server verification failed - invalid GPU evidence: server_id={server.server_id} ip={server.ip} miner_hotkey={miner_hotkey} error={e.detail}"
        )
        failure_reason = "Server verification failed: invalid GPU evidence"
        raise e
    except GpuEvidenceError as e:
        logger.error(
            f"Server verification failed - GPU evidence error: server_id={server.server_id} ip={server.ip} miner_hotkey={miner_hotkey} error={e.detail}"
        )
        failure_reason = "Server verification failed: Failed to verify GPU evidence"
        raise e
    except Exception as e:
        logger.error(
            f"Unexpected error during server verification: server_id={server.server_id} ip={server.ip} miner_hotkey={miner_hotkey} error={str(e)}"
        )
        failure_reason = "Unexpected error during server verification."
        raise e
    finally:
        if failure_reason:
            # Create attestation record
            server_attestation = ServerAttestation(
                quote_data=base64.b64encode(quote.raw_bytes).decode("utf-8") if quote else None,
                server_id=server.server_id,
                verification_error=failure_reason,
                created_at=func.now(),
            )

            db.add(server_attestation)
            await db.commit()
            await db.refresh(server_attestation)


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
    expected_cert_hash: str,
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
        result = await verify_quote(quote, expected_nonce, expected_cert_hash)

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
    _ = await check_server_ownership(db, server_id, miner_hotkey)

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

    await db.delete(server)
    await db.commit()

    logger.info(f"Deleted server: {server_id}")
    return True


async def validate_boot_token_and_get_vm_identity(boot_token: str) -> tuple[str, str]:
    """
    Validate boot token and return the VM identity (miner_hotkey, vm_name).

    Args:
        boot_token: Boot token from initial attestation

    Returns:
        Tuple of (miner_hotkey, vm_name)

    Raises:
        NonceError: If boot token is invalid or expired
    """
    # Validate boot token
    redis_key = f"boot_token:{boot_token}"
    redis_value = await settings.redis_client.get(redis_key)

    if not redis_value:
        raise NonceError("Boot token not found or expired")

    # Parse miner_hotkey:vm_name from the stored value
    try:
        boot_token_value = redis_value.decode()
        miner_hotkey, vm_name = boot_token_value.split(":", 1)
    except (ValueError, AttributeError) as e:
        logger.error(f"Failed to parse boot token value: {e}")
        raise NonceError("Invalid boot token format")

    logger.info(f"Validated boot token for VM {vm_name} (miner: {miner_hotkey})")

    return miner_hotkey, vm_name


async def get_cache_passphrase_with_token(
    db: AsyncSession, boot_token: str, hotkey: str, vm_name: str
) -> str:
    """
    Validate boot token and retrieve existing cache passphrase.

    This is called when the cache volume is already encrypted and needs the passphrase.

    Args:
        db: Database session
        boot_token: Boot token from initial attestation
        hotkey: Miner hotkey (must match boot token)
        vm_name: VM name (must match boot token)

    Returns:
        Cache volume passphrase (decrypted)

    Raises:
        NonceError: If boot token is invalid or expired, or if hotkey/vm_name don't match
        ValueError: If no passphrase exists (shouldn't happen for encrypted volumes)
    """
    # Validate token and get VM identity from boot token
    token_hotkey, token_vm_name = await validate_boot_token_and_get_vm_identity(boot_token)

    # Verify that the provided hotkey and vm_name match what's in the boot token
    if token_hotkey != hotkey:
        logger.warning(f"Hotkey mismatch: expected {token_hotkey}, got {hotkey}")
        raise NonceError("Hotkey does not match boot token")
    if token_vm_name != vm_name:
        logger.warning(f"VM name mismatch: expected {token_vm_name}, got {vm_name}")
        raise NonceError("VM name does not match boot token")

    # Get existing passphrase
    passphrase = await get_cache_passphrase(db, hotkey, vm_name)

    # Consume the boot token after successful retrieval
    redis_key = f"boot_token:{boot_token}"
    await settings.redis_client.delete(redis_key)

    return passphrase


async def create_cache_passphrase_with_token(
    db: AsyncSession, boot_token: str, hotkey: str, vm_name: str
) -> str:
    """
    Validate boot token and create/override cache passphrase.

    This is called when the cache volume is unencrypted (new or wiped).

    Args:
        db: Database session
        boot_token: Boot token from initial attestation
        hotkey: Miner hotkey (must match boot token)
        vm_name: VM name (must match boot token)

    Returns:
        Cache volume passphrase (decrypted)

    Raises:
        NonceError: If boot token is invalid or expired, or if hotkey/vm_name don't match
    """
    # Validate token and get VM identity from boot token
    token_hotkey, token_vm_name = await validate_boot_token_and_get_vm_identity(boot_token)

    # Verify that the provided hotkey and vm_name match what's in the boot token
    if token_hotkey != hotkey:
        logger.warning(f"Hotkey mismatch: expected {token_hotkey}, got {hotkey}")
        raise NonceError("Hotkey does not match boot token")
    if token_vm_name != vm_name:
        logger.warning(f"VM name mismatch: expected {token_vm_name}, got {vm_name}")
        raise NonceError("VM name does not match boot token")

    # Create or override passphrase
    passphrase = await create_or_update_cache_passphrase(db, hotkey, vm_name)

    # Consume the boot token after successful creation
    redis_key = f"boot_token:{boot_token}"
    await settings.redis_client.delete(redis_key)

    return passphrase
