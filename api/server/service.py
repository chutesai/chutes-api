"""
Core server management and TDX attestation logic.
"""

import pybase64 as base64
from datetime import datetime, timezone, timedelta
import json
import secrets
from typing import Dict, Any, Optional
from fastapi import HTTPException, Header, Request, status
from loguru import logger
from sqlalchemy import delete, or_, select, func
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.exc import IntegrityError

from api.config import settings
from api.constants import NONCE_HEADER, NoncePurpose, HOTKEY_HEADER, LUKS_STORAGE_VOLUME
from api.gpu import SUPPORTED_GPUS
from api.node.util import _track_nodes
from api.server.client import TeeServerClient
from api.server.quote import BootTdxQuote, RuntimeTdxQuote, TdxQuote
from api.server.schemas import (
    Server,
    ServerAttestation,
    BootAttestation,
    BootAttestationArgs,
    RuntimeAttestationArgs,
    ServerArgs,
    TeeUpgradeWindow,
    MaintenanceReason,
    SoleSurvivorBlock,
    PreflightResult,
    UpgradeWindowInfo,
    ConfirmMaintenanceResult,
    LuksAttestRequest,
    LuksAttestResult,
    LuksConfirmRequest,
    LuksConfirmResult,
)
from api.server.exceptions import (
    AttestationError,
    GetEvidenceError,
    GpuEvidenceError,
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
    _get_vm_cache_config,
    get_matching_measurement_config,
    generate_nonce,
    get_nonce_expiry_seconds,
    verify_quote,
    verify_gpu_evidence,
    sync_server_luks_passphrases,
    rotate_luks_passphrases,
    generate_confirm_nonce,
    generate_luks_quote_nonce,
    encrypt_passphrase,
    decrypt_passphrase,
    get_public_key_hash,
    cert_to_base64_der,
    validate_user_nonce,
)
from api.instance.schemas import Instance, instance_nodes
from api.instance.util import purge_and_notify
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

    redis_value = await settings.redis_client.getdel(redis_key)

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


async def require_luks_quote_nonce(
    vm_name: str,
    hotkey: str | None = Header(None, alias=HOTKEY_HEADER),
    quote_nonce: str | None = Header(None, alias="X-Quote-Nonce"),
) -> str:
    """
    FastAPI dependency for POST /luks/attest (new VMs, version >= 1.3.0).

    Mirrors validate_request_nonce: GETDEL luks_quote_nonce:{hotkey}:{vm_name},
    verify X-Quote-Nonce header matches stored value. Returns the validated nonce
    so the handler can pass it to verify_quote (which checks signature + all RTMRs
    including RTMR3 extended in initramfs).
    """
    if not quote_nonce or not hotkey:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Quote nonce (X-Quote-Nonce) and hotkey are required",
        )
    redis_key = f"luks_quote_nonce:{hotkey}:{vm_name}"
    stored = await settings.redis_client.getdel(redis_key)
    if not stored:
        logger.warning(f"LUKS quote nonce not found or expired for VM {vm_name} (hotkey: {hotkey})")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Quote nonce not found or expired",
        )
    if stored.decode() != quote_nonce:
        logger.warning(f"LUKS quote nonce mismatch for VM {vm_name} (hotkey: {hotkey})")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Quote nonce mismatch",
        )
    return quote_nonce


async def require_confirm_nonce(
    vm_name: str,
    hotkey: str | None = Header(None, alias=HOTKEY_HEADER),
    confirm_nonce: str | None = Header(None, alias="X-Confirm-Nonce"),
) -> None:
    """
    FastAPI dependency for POST /luks/confirm.

    GETDEL confirm:{hotkey}:{vm_name}, verify X-Confirm-Nonce header matches
    stored value. Raises 401 on mismatch or missing/expired nonce.
    """
    if not confirm_nonce or not hotkey:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Confirm nonce (X-Confirm-Nonce) and hotkey are required",
        )
    redis_key = f"confirm:{hotkey}:{vm_name}"
    stored = await settings.redis_client.getdel(redis_key)
    if not stored:
        logger.warning(f"Confirm nonce not found or expired for VM {vm_name} (hotkey: {hotkey})")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Confirm nonce not found or expired",
        )
    if stored.decode() != confirm_nonce:
        logger.warning(f"Confirm nonce mismatch for VM {vm_name} (hotkey: {hotkey})")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Confirm nonce mismatch",
        )


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
) -> tuple[Optional[str], Optional[str]]:
    """
    Process a boot attestation request.

    Args:
        db: Database session
        server_ip: Server IP address
        args: Boot attestation arguments (includes miner_hotkey and vm_name)
        nonce: Validated nonce
        expected_cert_hash: Expected certificate hash

    Returns:
        Tuple of (boot_token, luks_quote_nonce). Exactly one is set depending on
        the VM's measurement version: boot_token for version < 1.3.0 (legacy POST
        /luks flow), luks_quote_nonce for version >= 1.3.0 (POST /luks/attest flow).

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
            miner_hotkey=args.miner_hotkey,
            vm_name=args.vm_name,
            measurement_version=measurement_config.version,
            created_at=func.now(),
            verified_at=func.now(),
        )

        db.add(boot_attestation)
        await db.commit()
        await db.refresh(boot_attestation)

        logger.success(f"Boot attestation successful: {boot_attestation.attestation_id}")

        await _handle_boot_version_update(
            db, args.miner_hotkey, args.vm_name, measurement_config.version
        )

        # Version-gate: legacy VMs (< 1.3.0) get a boot token for POST /luks;
        # new VMs (>= 1.3.0) get a luks_quote_nonce for POST /luks/attest instead.
        boot_token: Optional[str] = None
        luks_quote_nonce: Optional[str] = None
        if semcomp(measurement_config.version, "1.3.0") >= 0:
            luks_quote_nonce = await generate_luks_quote_nonce(args.miner_hotkey, args.vm_name)
        else:
            boot_token = await generate_and_store_boot_token(args.miner_hotkey, args.vm_name)

        return boot_token, luks_quote_nonce

    except (InvalidQuoteError, MeasurementMismatchError) as e:
        # Create failed attestation record; set measurement_version if quote matched a config
        measurement_version = None
        try:
            quote = BootTdxQuote.from_base64(args.quote)
            measurement_config = get_matching_measurement_config(quote)
            measurement_version = measurement_config.version
        except (InvalidQuoteError, MeasurementMismatchError):
            pass
        if measurement_version is None:
            logger.warning(
                "Boot attestation failed with no matching measurement config (measurement_version will be NULL). "
            )
        boot_attestation = BootAttestation(
            quote_data=args.quote,
            server_ip=server_ip,
            miner_hotkey=args.miner_hotkey,
            vm_name=args.vm_name,
            verification_error=str(e.detail),
            measurement_version=measurement_version,
            created_at=func.now(),
        )

        db.add(boot_attestation)
        await db.commit()

        logger.error(f"Boot attestation failed: {str(e)}")
        raise


async def _handle_boot_version_update(
    db: AsyncSession, miner_hotkey: str, vm_name: str, measurement_version: str
) -> None:
    """Update server.version on every successful boot; clear maintenance slot if target met."""
    try:
        server = await get_server_by_name(db, miner_hotkey, vm_name)
    except ServerNotFoundError:
        return

    server.version = measurement_version

    if server.in_maintenance:
        window = await db.get(TeeUpgradeWindow, server.maintenance_pending_window_id)
        if (
            window is not None
            and semcomp(measurement_version, window.target_measurement_version) >= 0
        ):
            logger.info(
                f"Maintenance complete for server {server.server_id}: "
                f"version {measurement_version} meets target {window.target_measurement_version}"
            )
            server.maintenance_pending_window_id = None
        elif window is not None:
            logger.warning(
                f"Boot attestation for server {server.server_id} has version {measurement_version} "
                f"but target is {window.target_measurement_version}; maintenance not complete"
            )
        else:
            logger.warning(
                f"Server {server.server_id} has stale maintenance_pending_window_id "
                f"pointing to missing window; clearing"
            )
            server.maintenance_pending_window_id = None

    await db.commit()


async def register_server(db: AsyncSession, args: ServerArgs, miner_hotkey: str):
    """
    Register a TEE server: create Server, verify attestation (creating a ServerAttestation
    record on success or failure), then track nodes. ServerAttestation is always inserted
    by verify_server for audit trail.
    """
    try:
        server = await _track_server(
            db, args.id, args.name or args.id, args.host, miner_hotkey, is_tee=True
        )

        # Set the attributes we can't get from pynvml
        for gpu in args.gpus:
            gpu_info = SUPPORTED_GPUS[gpu.gpu_identifier]
            for key in ["processors", "max_threads_per_processor"]:
                setattr(gpu, key, gpu_info.get(key))

        # Start verification process (pass GPUs for validation)
        measurement_version = await verify_server(db, server, miner_hotkey, gpus=args.gpus)

        if measurement_version is not None:
            server.version = measurement_version
            await db.commit()

        # Track nodes once verified
        await _track_nodes(db, miner_hotkey, server.server_id, args.gpus, "0", func.now())

    except AttestationError as e:
        # Clean up orphan server: _track_server committed before verify_server failed.
        await db.rollback()
        await db.execute(delete(Server).where(Server.server_id == args.id))
        await db.commit()
        error_detail = e.detail if hasattr(e, "detail") else str(e)
        logger.error(
            f"Server registration failed - attestation error: name={args.name or args.id} host={args.host} miner_hotkey={miner_hotkey} error={error_detail}"
        )
        raise ServerRegistrationError(f"Server registration failed - {error_detail}")
    except IntegrityError as e:
        await db.rollback()
        # Clean up orphan server when IntegrityError came from _track_nodes.
        # If from _track_server (duplicate server), this is a no-op.
        await db.execute(delete(Server).where(Server.server_id == args.id))
        await db.commit()
        logger.error(
            f"Server registration failed - IntegrityError: name={args.name or args.id} host={args.host} miner_hotkey={miner_hotkey} error={str(e)}"
        )
        raise ServerRegistrationError(
            "Server registration failed - database constraint violation. This may indicate a duplicate server ID, invalid miner configuration, or other database conflict. Please contact support with your server ID and miner hotkey."
        )
    except Exception as e:
        await db.rollback()
        # Clean up orphan server if failure occurred after _track_server.
        await db.execute(delete(Server).where(Server.server_id == args.id))
        await db.commit()
        logger.error(
            f"Unexpected error during server registration: name={args.name or args.id} host={args.host} miner_hotkey={miner_hotkey} error={str(e)}",
            exc_info=True,
        )
        raise ServerRegistrationError(
            "Server registration failed - unexpected error occurred. Please contact support with your server ID and miner hotkey."
        )


async def verify_server(
    db: AsyncSession, server: Server, miner_hotkey: str, gpus: list[NodeArgs]
) -> Optional[str]:
    """
    Verify server attestation and validate GPUs match measurement configuration.

    Returns the measurement_version string on success, None on failure.
    """
    failure_reason = ""
    quote = None
    measurement_config = None
    try:
        client = TeeServerClient(server)

        nonce = generate_nonce()
        logger.info(
            f"Verifying server server_id={server.server_id} ip={server.ip} miner_hotkey={miner_hotkey} with nonce {nonce}"
        )
        quote, gpu_evidence, cert = await client.get_server_evidence(nonce)
        measurement_config = get_matching_measurement_config(quote)
        expected_cert_hash = get_public_key_hash(cert)

        # Verify quote measurements (matches by full MRTD + RTMRs; multiple configs may share RTMR0)
        await verify_quote(quote, nonce, expected_cert_hash)

        # Verify GPU evidence
        await verify_gpu_evidence(gpu_evidence, nonce)

        # Validate GPUs match measurement configuration
        validate_gpus_for_measurements(quote, gpus)

        logger.success(
            f"Verified server server_id={server.server_id} ip={server.ip} for miner: {miner_hotkey}"
        )

        # Create attestation record (measurement_version for audit trail; server version = latest attestation).
        # Commit here so we have a durable record for this run even if _track_nodes or later steps fail.
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

        return measurement_config.version

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
            measurement_version = measurement_config.version if measurement_config else None
            server_attestation = ServerAttestation(
                quote_data=base64.b64encode(quote.raw_bytes).decode("utf-8") if quote else None,
                server_id=server.server_id,
                verification_error=failure_reason,
                created_at=func.now(),
                measurement_version=measurement_version,
            )

            try:
                db.add(server_attestation)
                await db.commit()
                await db.refresh(server_attestation)
                logger.info(
                    f"Persisted failed server attestation for server_id={server.server_id} (reason: {failure_reason})"
                )
            except Exception:
                logger.exception(
                    f"Failed to persist failed attestation record for server_id={server.server_id}; "
                    "attestation history will be incomplete",
                    exc_info=True,
                )
                raise


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


async def process_luks_attest_request(
    db: AsyncSession,
    hotkey: str,
    vm_name: str,
    body: LuksAttestRequest,
    quote_nonce: str,
    expected_cert_hash: str,
) -> LuksAttestResult:
    """
    Process POST /luks/attest for new-format VMs (version >= 1.3.0).

    The quote nonce has already been validated and consumed by require_luks_quote_nonce.
    Verifies the TDX quote (signature + all RTMR measurements including RTMR3), rotates
    passphrases, manages the k3s encryption key, and issues a confirm nonce.
    """
    quote = RuntimeTdxQuote.from_base64(body.quote)
    await verify_quote(quote, quote_nonce, expected_cert_hash)

    volumes_data, vm_config = await rotate_luks_passphrases(db, hotkey, vm_name, body.volumes)

    # Derive k3s key lifecycle from DB state: if storage had no current passphrase
    # (first boot) or no k3s key is stored yet, generate a new one.
    storage_rotation = volumes_data.get(LUKS_STORAGE_VOLUME)
    if (
        storage_rotation is None or storage_rotation.is_first_boot
    ) or not vm_config.k3s_encryption_key:
        k3s_bytes = secrets.token_bytes(32)
        k3s_b64 = base64.b64encode(k3s_bytes).decode()
        vm_config.k3s_encryption_key = encrypt_passphrase(k3s_b64)
        await db.commit()
    else:
        k3s_b64 = decrypt_passphrase(vm_config.k3s_encryption_key)

    confirm_nonce = await generate_confirm_nonce(hotkey, vm_name)

    return LuksAttestResult(
        volumes=volumes_data,
        confirm_nonce=confirm_nonce,
        k3s_encryption_key=k3s_b64,
    )


async def process_luks_confirm(
    db: AsyncSession,
    hotkey: str,
    vm_name: str,
    body: LuksConfirmRequest,
) -> LuksConfirmResult:
    """
    Process POST /luks/confirm.

    The confirm nonce has already been validated and consumed by require_confirm_nonce.
    Promotes pending passphrases to current for volumes that rotated successfully,
    or discards them for volumes that failed.
    """
    vm_config = await _get_vm_cache_config(db, hotkey, vm_name)
    if vm_config is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No LUKS config found for VM {vm_name}",
        )

    stored: Dict[str, str] = dict(vm_config.volume_passphrases or {})
    confirmed_volumes: Dict[str, dict] = {}

    for vol, vol_status in body.volumes.items():
        pending_key = f"pending_{vol}"
        if vol_status.rotated:
            if pending_key in stored:
                stored[vol] = stored.pop(pending_key)
                confirmed_volumes[vol] = {"result": "promoted"}
                logger.info(
                    f"LUKS confirm: promoted pending passphrase for volume {vol} (VM: {vm_name})"
                )
            else:
                confirmed_volumes[vol] = {"result": "no_pending"}
                logger.warning(
                    f"LUKS confirm: rotated=True but no pending passphrase for volume {vol} (VM: {vm_name})"
                )
        else:
            stored.pop(pending_key, None)
            confirmed_volumes[vol] = {"result": "discarded"}
            logger.info(
                f"LUKS confirm: discarded pending passphrase for volume {vol} (VM: {vm_name})"
            )

    vm_config.volume_passphrases = stored
    await db.commit()

    return LuksConfirmResult(volumes=confirmed_volumes)


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


# ---------------------------------------------------------------------------
# TEE Maintenance Window
# ---------------------------------------------------------------------------


async def get_active_upgrade_window(
    db: AsyncSession,
) -> Optional[TeeUpgradeWindow]:
    """Return the active tee_upgrade_windows row (start <= now <= end), or None."""
    now = func.now()
    query = (
        select(TeeUpgradeWindow)
        .where(
            TeeUpgradeWindow.upgrade_window_start <= now,
            TeeUpgradeWindow.upgrade_window_end >= now,
        )
        .order_by(TeeUpgradeWindow.created_at.desc())
    )
    result = await db.execute(query)
    rows = result.scalars().all()
    if not rows:
        return None
    if len(rows) > 1:
        logger.warning(
            f"Multiple overlapping tee_upgrade_windows rows active ({len(rows)}); "
            f"using most recently created: {rows[0].id}"
        )
    return rows[0]


async def _get_instances_on_server(db: AsyncSession, server_id: str) -> list[Instance]:
    """Return all instances hosted on a server via Instance → instance_nodes → Node → Server."""
    query = (
        select(Instance)
        .join(instance_nodes, Instance.instance_id == instance_nodes.c.instance_id)
        .join(Node, instance_nodes.c.node_id == Node.uuid)
        .where(Node.server_id == server_id)
        .distinct()
    )
    result = await db.execute(query)
    return list(result.scalars().all())


async def _find_sole_survivor_chutes(
    db: AsyncSession, instances: list[Instance]
) -> list[SoleSurvivorBlock]:
    """For each instance, check if it is the only active instance globally for its chute.

    Returns a list of SoleSurvivorBlock for blocking sole survivors.
    """
    blocking: list[SoleSurvivorBlock] = []
    seen_chutes: set[str] = set()
    for inst in instances:
        if inst.chute_id in seen_chutes:
            continue
        seen_chutes.add(inst.chute_id)
        count_query = (
            select(func.count())
            .select_from(Instance)
            .where(
                Instance.chute_id == inst.chute_id,
                Instance.active.is_(True),
                Instance.instance_id != inst.instance_id,
            )
        )
        result = await db.execute(count_query)
        other_active = result.scalar() or 0
        if other_active == 0:
            blocking.append(SoleSurvivorBlock(chute_id=inst.chute_id, instance_id=inst.instance_id))
    return blocking


async def _count_active_maintenance_slots(
    db: AsyncSession, miner_hotkey: str, active_window: TeeUpgradeWindow
) -> int:
    """Count servers for this miner that are in maintenance for the given active window."""
    query = (
        select(func.count())
        .select_from(Server)
        .where(
            Server.miner_hotkey == miner_hotkey,
            Server.maintenance_pending_window_id == active_window.id,
        )
    )
    result = await db.execute(query)
    return result.scalar() or 0


async def preflight_maintenance(
    db: AsyncSession, server: Server, miner_hotkey: str
) -> PreflightResult:
    """Read-only eligibility check for entering maintenance on a server."""
    denial_reasons: list[MaintenanceReason] = []
    blocking: list[SoleSurvivorBlock] = []
    limit = 1
    current_slots = 0
    active_window: Optional[TeeUpgradeWindow] = None

    if not server.is_tee:
        denial_reasons.append(MaintenanceReason(reason="not_tee"))

    if not denial_reasons:
        active_window = await get_active_upgrade_window(db)
        if active_window is None:
            denial_reasons.append(MaintenanceReason(reason="no_active_window"))

    if active_window is not None:
        limit = active_window.max_concurrent_per_miner

        if (
            server.version is not None
            and semcomp(server.version, active_window.target_measurement_version) >= 0
        ):
            denial_reasons.append(
                MaintenanceReason(
                    reason="already_at_target",
                    current_version=server.version,
                    target_version=active_window.target_measurement_version,
                )
            )

        if server.in_maintenance:
            if server.maintenance_pending_window_id == active_window.id:
                denial_reasons.append(
                    MaintenanceReason(
                        reason="maintenance_pending",
                        current_version=server.version,
                        target_version=active_window.target_measurement_version,
                        window_id=active_window.id,
                    )
                )
            else:
                server.maintenance_pending_window_id = None

        current_slots = await _count_active_maintenance_slots(db, miner_hotkey, active_window)
        if current_slots >= limit:
            denial_reasons.append(
                MaintenanceReason(
                    reason="concurrency_cap",
                    current_slots=current_slots,
                    limit=limit,
                )
            )

        instances = await _get_instances_on_server(db, server.server_id)
        blocking = await _find_sole_survivor_chutes(db, instances)
        if blocking:
            denial_reasons.append(
                MaintenanceReason(
                    reason="sole_survivor",
                    blocking=[b.model_dump() for b in blocking],
                )
            )

    return PreflightResult(
        eligible=len(denial_reasons) == 0,
        denial_reasons=denial_reasons,
        blocking_chute_ids=blocking,
        current_slots=current_slots,
        limit=limit,
    )


async def confirm_maintenance(
    db: AsyncSession, server: Server, miner_hotkey: str
) -> ConfirmMaintenanceResult:
    """Enter maintenance: re-validate, set pending window, and auto-purge instances.

    Raises HTTPException (409/403) on failure.
    """
    preflight = await preflight_maintenance(db, server, miner_hotkey)
    if not preflight.eligible:
        reason_codes = {r.reason for r in preflight.denial_reasons}
        conflict_reasons = {"sole_survivor", "concurrency_cap", "maintenance_pending"}
        if reason_codes & conflict_reasons:
            status_code = status.HTTP_409_CONFLICT
        else:
            status_code = status.HTTP_403_FORBIDDEN
        raise HTTPException(status_code=status_code, detail=preflight.model_dump())

    active_window = await get_active_upgrade_window(db)
    server.maintenance_pending_window_id = active_window.id
    await db.commit()
    await db.refresh(server)

    instances = await _get_instances_on_server(db, server.server_id)
    purged_ids: list[str] = []
    for inst in instances:
        try:
            await purge_and_notify(
                inst,
                reason="maintenance - server entering TEE upgrade window",
                valid_termination=True,
            )
            purged_ids.append(inst.instance_id)
        except Exception:
            logger.error(
                f"Failed to purge instance {inst.instance_id} during maintenance", exc_info=True
            )

    return ConfirmMaintenanceResult(
        server_id=server.server_id,
        purged_instance_ids=purged_ids,
        window=UpgradeWindowInfo(
            id=active_window.id,
            target_measurement_version=active_window.target_measurement_version,
            upgrade_window_start=str(active_window.upgrade_window_start),
            upgrade_window_end=str(active_window.upgrade_window_end),
            max_concurrent_per_miner=active_window.max_concurrent_per_miner,
        ),
    )
