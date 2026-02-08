"""
Core server management and TDX attestation logic.
"""

import asyncio
import pybase64 as base64
from datetime import datetime, timezone, timedelta
import json
import tempfile
from typing import Dict, Any, Optional
from fastapi import HTTPException, Header, Request, status
from loguru import logger
from sqlalchemy import or_, select, func
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
    ChuteNotTeeError,
    InstanceNotFoundError,
)
from api.server.util import (
    _track_server,
    extract_report_data,
    verify_measurements,
    get_matching_measurement_config,
    generate_nonce,
    get_nonce_expiry_seconds,
    verify_quote_signature,
    verify_result,
    sync_server_luks_passphrases,
    delete_luks_passphrases_for_server,
    get_public_key_hash,
    cert_to_base64_der,
    validate_user_nonce,
)
from api.instance.schemas import Instance
from api.chute.schemas import Chute
from api.node.schemas import Node
from sqlalchemy.orm import joinedload
from api.server.schemas import TeeInstanceEvidence
from api.node.schemas import NodeArgs
from api.util import extract_ip, semcomp


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


async def validate_and_consume_nonce(
    nonce_value: str, server_ip: str, purpose: NoncePurpose
) -> None:
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


def validate_gpus_for_measurements(quote: TdxQuote, gpus: list[NodeArgs]) -> None:
    """
    Validate that the provided GPUs match the expected GPUs for this measurement configuration.

    Looks up the measurement configuration using the quote's RTMR0.

    Args:
        quote: Verified TDX quote (must have been verified via verify_quote)
        gpus: List of GPU nodes being registered

    Raises:
        MeasurementMismatchError: If GPUs don't match measurement configuration expectations
    """
    # Look up measurement configuration by full MRTD + RTMRs (same as verify_measurements)
    measurement_config = get_matching_measurement_config(quote)

    # Extract GPU identifiers
    provided_gpu_ids = {gpu.gpu_identifier.lower() for gpu in gpus}
    expected_gpu_ids = set(measurement_config.expected_gpus)

    # Check that all provided GPUs are in expected list
    unexpected_gpus = provided_gpu_ids - expected_gpu_ids
    if unexpected_gpus:
        raise MeasurementMismatchError(
            f"GPU mismatch for measurement config '{measurement_config.name}': "
            f"Expected GPUs {expected_gpu_ids}, but got {unexpected_gpus}"
        )

    # Check GPU count if specified
    if measurement_config.gpu_count and len(gpus) != measurement_config.gpu_count:
        raise MeasurementMismatchError(
            f"GPU count mismatch for measurement config '{measurement_config.name}': "
            f"Expected {measurement_config.gpu_count} GPUs, but got {len(gpus)}"
        )

    logger.info(
        f"GPU validation passed for measurement config '{measurement_config.name}': "
        f"{len(gpus)} GPUs of types {provided_gpu_ids}"
    )


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

        measurement_config = get_matching_measurement_config(quote)
        # Create boot attestation record
        boot_attestation = BootAttestation(
            quote_data=args.quote,
            server_ip=server_ip,
            measurement_version=measurement_config.version,
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
        # Create failed attestation record; set measurement_version if quote matched a config
        measurement_version = None
        try:
            quote = BootTdxQuote.from_base64(args.quote)
            measurement_config = get_matching_measurement_config(quote)
            measurement_version = measurement_config.version
        except (InvalidQuoteError, MeasurementMismatchError):
            pass
        boot_attestation = BootAttestation(
            quote_data=args.quote,
            server_ip=server_ip,
            verification_error=str(e.detail),
            measurement_version=measurement_version,
            created_at=func.now(),
        )

        db.add(boot_attestation)
        await db.commit()

        logger.error(f"Boot attestation failed: {str(e)}")
        raise


async def register_server(db: AsyncSession, args: ServerArgs, miner_hotkey: str):
    try:
        server = await _track_server(db, args.id, args.name, args.host, miner_hotkey, is_tee=True)

        # Set the attributes we can't get from pynvml
        for gpu in args.gpus:
            gpu_info = SUPPORTED_GPUS[gpu.gpu_identifier]
            for key in ["processors", "max_threads_per_processor"]:
                setattr(gpu, key, gpu_info.get(key))

        # Start verification process (pass GPUs for validation)
        await verify_server(db, server, miner_hotkey, gpus=args.gpus)

        # Track nodes once verified
        await _track_nodes(db, miner_hotkey, server.server_id, args.gpus, "0", func.now())

    except AttestationError as e:
        # Preserve the specific error message from the AttestationError
        error_detail = e.detail if hasattr(e, "detail") else str(e)
        logger.error(
            f"Server registration failed - attestation error: name={args.name} host={args.host} miner_hotkey={miner_hotkey} error={error_detail}"
        )
        raise ServerRegistrationError(f"Server registration failed - {error_detail}")
    except IntegrityError as e:
        await db.rollback()
        logger.error(
            f"Server registration failed - IntegrityError: name={args.name} host={args.host} miner_hotkey={miner_hotkey} error={str(e)}"
        )
        raise ServerRegistrationError(
            "Server registration failed - database constraint violation. This may indicate a duplicate server ID, invalid miner configuration, or other database conflict. Please contact support with your server ID and miner hotkey."
        )
    except Exception as e:
        await db.rollback()
        logger.error(
            f"Unexpected error during server registration: name={args.name} host={args.host} miner_hotkey={miner_hotkey} error={str(e)}",
            exc_info=True,
        )
        raise ServerRegistrationError(
            "Server registration failed - unexpected error occurred. Please contact support with your server ID and miner hotkey."
        )


async def verify_server(
    db: AsyncSession, server: Server, miner_hotkey: str, gpus: list[NodeArgs]
) -> None:
    """
    Verify server attestation and validate GPUs match measurement configuration.

    Args:
        db: Database session
        server: Server to verify
        miner_hotkey: Miner hotkey
        gpus: List of GPUs to validate against measurement configuration
    """
    failure_reason = ""
    quote = None
    try:
        client = TeeServerClient(server)

        nonce = generate_nonce()
        logger.info(
            f"Verifying server server_id={server.server_id} ip={server.ip} miner_hotkey={miner_hotkey} with nonce {nonce}"
        )
        quote, gpu_evidence, cert = await client.get_server_evidence(nonce)
        expected_cert_hash = get_public_key_hash(cert)

        # Verify quote measurements (matches by full MRTD + RTMRs; multiple configs may share RTMR0)
        await verify_quote(quote, nonce, expected_cert_hash)

        # Verify GPU evidence
        await verify_gpu_evidence(gpu_evidence, nonce)

        # Validate GPUs match measurement configuration
        validate_gpus_for_measurements(quote, gpus)

        measurement_config = get_matching_measurement_config(quote)

        logger.success(
            f"Verified server server_id={server.server_id} ip={server.ip} for miner: {miner_hotkey}"
        )

        # Create attestation record (measurement_version for audit trail; server version = latest attestation)
        server_attestation = ServerAttestation(
            quote_data=base64.b64encode(quote.raw_bytes).decode("utf-8"),
            server_id=server.server_id,
            created_at=func.now(),
            verified_at=func.now(),
            measurement_version=measurement_config.version,
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
            measurement_version = None
            if quote:
                try:
                    measurement_config = get_matching_measurement_config(quote)
                    measurement_version = measurement_config.version
                except MeasurementMismatchError:
                    pass
            server_attestation = ServerAttestation(
                quote_data=base64.b64encode(quote.raw_bytes).decode("utf-8") if quote else None,
                server_id=server.server_id,
                verification_error=failure_reason,
                created_at=func.now(),
                measurement_version=measurement_version,
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


async def get_server_by_name(db: AsyncSession, miner_hotkey: str, server_name: str) -> Server:
    """
    Get a server by miner hotkey and VM name (stable identity for API paths).

    Args:
        db: Database session
        miner_hotkey: Miner hotkey (must match authenticated user)
        vm_name: VM name

    Returns:
        Server object

    Raises:
        ServerNotFoundError: If server not found
    """
    query = select(Server).where(Server.miner_hotkey == miner_hotkey, Server.name == server_name)
    result = await db.execute(query)
    server = result.scalar_one_or_none()
    if not server:
        raise ServerNotFoundError(f"{server_name}")
    return server


async def get_server_by_name_or_id(
    db: AsyncSession, miner_hotkey: str, server_name_or_id: str
) -> Server:
    """
    Get a server by miner hotkey and either VM name or server id.

    Args:
        db: Database session
        miner_hotkey: Miner hotkey (must match authenticated user)
        server_name_or_id: VM name or server_id

    Returns:
        Server object

    Raises:
        ServerNotFoundError: If server not found
    """
    query = select(Server).where(
        Server.miner_hotkey == miner_hotkey,
        or_(
            Server.name == server_name_or_id,
            Server.server_id == server_name_or_id,
        ),
    )
    result = await db.execute(query)
    server = result.scalar_one_or_none()
    if not server:
        raise ServerNotFoundError(server_name_or_id)
    return server


async def update_server_name(
    db: AsyncSession, miner_hotkey: str, server_id: str, server_name: str
) -> Server:
    """
    Update name for an existing server (by server_id). Used to sync names for
    servers that existed before the name schema change.

    Args:
        db: Database session
        miner_hotkey: Authenticated miner hotkey (must own the server)
        server_id: Server ID (e.g. k8s node uid)
        server_name: New VM name to set (unique per miner)

    Returns:
        Updated Server

    Raises:
        ServerNotFoundError: If server not found or not owned by miner
        HTTPException 409: If new_vm_name is already used by another server of this miner
    """
    server = await check_server_ownership(db, server_id, miner_hotkey)
    if server.name == server_name:
        return server
    server.name = server_name
    try:
        await db.commit()
        await db.refresh(server)
    except IntegrityError:
        await db.rollback()
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"name '{server_name}' already in use by another server for this miner",
        )
    logger.info(f"Updated server {server_id} name to {server_name}")
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
        await verify_quote(quote, expected_nonce, expected_cert_hash)

        # Create runtime attestation record
        measurement_config = get_matching_measurement_config(quote)
        attestation = ServerAttestation(
            server_id=server_id,
            quote_data=args.quote,
            verification_error=None,
            measurement_version=measurement_config.version,
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
        measurement_version = None
        try:
            quote_parsed = RuntimeTdxQuote.from_base64(args.quote)
            measurement_config = get_matching_measurement_config(quote_parsed)
            measurement_version = measurement_config.version
        except (InvalidQuoteError, MeasurementMismatchError):
            pass
        attestation = ServerAttestation(
            server_id=server_id,
            quote_data=args.quote,
            verification_error=str(e.detail),
            measurement_version=measurement_version,
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
        verified = latest_attestation.verification_error is None
        status["last_attestation"] = {
            "attestation_id": latest_attestation.attestation_id,
            "verified": verified,
            "created_at": latest_attestation.created_at.isoformat(),
            "verified_at": latest_attestation.verified_at.isoformat()
            if latest_attestation.verified_at
            else None,
            "verification_error": latest_attestation.verification_error,
        }
        status["attestation_status"] = "verified" if verified else "failed"

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

    await delete_luks_passphrases_for_server(db, server.miner_hotkey, server.name)
    await db.delete(server)
    await db.commit()

    logger.info(f"Deleted server: {server_id}")
    return True


async def _get_boot_token_context(boot_token: str) -> tuple[str, str]:
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

    logger.info(f"Retrieved boot token for VM {vm_name} (miner: {miner_hotkey})")

    return miner_hotkey, vm_name


async def _validate_boot_token_for_luks(boot_token: str, hotkey: str, vm_name: str) -> None:
    """Validate boot token and verify hotkey/vm_name match. Raises NonceError on failure."""
    token_hotkey, token_vm_name = await _get_boot_token_context(boot_token)
    if token_hotkey != hotkey:
        logger.warning(f"Hotkey mismatch: expected {token_hotkey}, got {hotkey}")
        raise NonceError("Hotkey does not match boot token")
    if token_vm_name != vm_name:
        logger.warning(f"VM name mismatch: expected {token_vm_name}, got {vm_name}")
        raise NonceError("VM name does not match boot token")


async def _consume_boot_token(boot_token: str) -> None:
    redis_key = f"boot_token:{boot_token}"
    await settings.redis_client.delete(redis_key)


async def process_luks_passphrase_request(
    db: AsyncSession,
    boot_token: str,
    hotkey: str,
    vm_name: str,
    volume_names: list,
    rekey_volume_names: Optional[list] = None,
) -> Dict[str, str]:
    """Validate boot token and run LUKS sync (ensure keys for volumes, prune others, rekey optional). Consumes token."""
    await _validate_boot_token_for_luks(boot_token, hotkey, vm_name)
    result = await sync_server_luks_passphrases(
        db, hotkey, vm_name, volume_names, rekey_volume_names=rekey_volume_names
    )
    await _consume_boot_token(boot_token)
    return result


async def get_instance_server(db: AsyncSession, instance_id: str) -> tuple[Server, Instance]:
    """
    Get the TEE server and instance for evidence/attestation (instance has chute, nodes, server loaded).

    Args:
        db: Database session
        instance_id: Instance ID

    Returns:
        (Server, Instance). Use instance.deployment_id or instance.instance_id for proxy routing.

    Raises:
        InstanceNotFoundError: If instance not found
        ChuteNotTeeError: If the instance's chute is not TEE-enabled
    """
    # Load instance with chute, nodes and their servers
    query = (
        select(Instance)
        .where(Instance.instance_id == instance_id)
        .options(joinedload(Instance.chute), joinedload(Instance.nodes).joinedload(Node.server))
    )
    result = await db.execute(query)
    instance = result.unique().scalar_one_or_none()

    if not instance:
        raise InstanceNotFoundError(instance_id)

    # Check if chute is TEE-enabled (TEE chutes can only run on TEE servers)
    if not instance.chute.tee:
        raise ChuteNotTeeError(instance.chute.chute_id)

    # Instance always has nodes, get server from first node
    node = instance.nodes[0]
    server = node.server

    return (server, instance)


async def _get_instance_evidence(
    server: Server, deployment_id: str, nonce: str
) -> TeeInstanceEvidence:
    """
    Get TEE instance evidence via the chute's evidence endpoint (third-party flow).
    Caller supplies nonce; we call chute-service-{deployment_id}/evidence?nonce=...
    Verification flow (no caller nonce) uses get_chute_evidence(deployment_id) → verify endpoint.
    """
    client = TeeServerClient(server)
    quote, gpu_evidence, cert = await client.get_chute_evidence(deployment_id, nonce=nonce)

    quote_base64 = base64.b64encode(quote.raw_bytes).decode("utf-8")
    cert_base64 = cert_to_base64_der(cert)

    return TeeInstanceEvidence(
        quote=quote_base64, gpu_evidence=gpu_evidence, certificate=cert_base64
    )


async def get_instance_evidence(
    db: AsyncSession, instance_id: str, nonce: str
) -> TeeInstanceEvidence:
    """
    Get TEE evidence for a specific instance (instance evidence endpoint flow).
    Requires instance.deployment_id (set when TEE launch config is claimed and verified).
    Runtime evidence is only supported for chutes_version >= 0.6.0.
    """
    validate_user_nonce(nonce)
    server, instance = await get_instance_server(db, instance_id)
    if semcomp(instance.chutes_version or "0.0.0", "0.6.0") < 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Instances requires chutes_version >= 0.6.0 to retrieve evidence.",
        )
    if not instance.deployment_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Instance has no deployment_id; evidence is only available after TEE verification",
        )
    return await _get_instance_evidence(server, instance.deployment_id, nonce)


async def get_chute_instances_evidence(
    db: AsyncSession, chute_id: str, nonce: str
) -> tuple[list[TeeInstanceEvidence], list[str]]:
    """
    Get TEE evidence for all instances of a chute (chute evidence endpoint flow).
    Returns (evidence_list, failed_instance_ids). Failed instance IDs are included
    so the user knows those instances still exist; access is already enforced at this point.
    Runtime evidence is only supported for chutes_version >= 0.6.0.
    """
    validate_user_nonce(nonce)

    query = select(Chute).where(Chute.chute_id == chute_id)
    result = await db.execute(query)
    chute = result.scalar_one_or_none()

    if not chute:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail=f"Chute {chute_id} not found"
        )

    if not chute.tee:
        raise ChuteNotTeeError(chute_id)

    if semcomp(chute.chutes_version or "0.0.0", "0.6.0") < 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Instances requires chutes_version >= 0.6.0 to retrieve evidence.",
        )

    instances_query = (
        select(Instance)
        .where(
            Instance.chute_id == chute_id,
            Instance.active.is_(True),
            Instance.verified.is_(True),
        )
        .options(joinedload(Instance.nodes).joinedload(Node.server))
    )
    instances_result = await db.execute(instances_query)
    instances = instances_result.unique().scalars().all()

    evidence_list: list[TeeInstanceEvidence] = []
    failed_instance_ids: list[str] = []
    for instance in instances:
        if not instance.deployment_id:
            failed_instance_ids.append(instance.instance_id)
            continue
        try:
            node = instance.nodes[0]
            server = node.server
            evidence = await _get_instance_evidence(server, instance.deployment_id, nonce)
            evidence_list.append(
                TeeInstanceEvidence(
                    quote=evidence.quote,
                    gpu_evidence=evidence.gpu_evidence,
                    instance_id=instance.instance_id,
                    certificate=evidence.certificate,
                )
            )
        except GetEvidenceError as e:
            logger.error(f"Failed to get evidence for instance {instance.instance_id}: {str(e)}")
            failed_instance_ids.append(instance.instance_id)

    return (evidence_list, failed_instance_ids)
