"""
TDX quote parsing, crypto operations, and server helper functions.
"""

import secrets
from typing import Dict
from sqlalchemy import select
from sqlalchemy.sql import func
from sqlalchemy.ext.asyncio import AsyncSession
from urllib.parse import unquote
from aiohttp import ClientResponse
from cryptography.fernet import Fernet
from fastapi import Request, status
from loguru import logger
from dcap_qvl import get_collateral_and_verify
from api.config import settings
from cryptography import x509
from cryptography.x509 import Certificate
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.backends import default_backend
from api.server.exceptions import (
    InvalidQuoteError,
    AttestationError,
    InvalidSignatureError,
    InvalidTdxConfiguration,
    MeasurementMismatchError,
    NoClientCertError,
    NoServerCertError,
)
from api.server.quote import TdxQuote, TdxVerificationResult
import hashlib

from api.server.schemas import Server, VmCacheConfig


def generate_nonce() -> str:
    """Generate a cryptographically secure nonce."""
    return secrets.token_hex(32)


def get_nonce_expiry_seconds(minutes: int = 10) -> int:
    """Get expiry time for a nonce in seconds."""
    return minutes * 60


def extract_client_cert_hash():
    async def _extract_request_client_cert(request: Request):
        try:
            cert = _get_client_certificate(request)
            cert_hash = _get_public_key_hash(cert)

            return cert_hash
        except Exception as e:
            logger.error(f"Boot attestation failed, no client cert provided:\n{e}")
            raise NoClientCertError(status_code=status.HTTP_401_UNAUTHORIZED, detail=str(e))

    return _extract_request_client_cert


def extract_server_cert_hash(response: ClientResponse):
    try:
        cert = _get_server_certificate(response)
        cert_hash = _get_public_key_hash(cert)

        return cert_hash
    except Exception as e:
        logger.error(f"Exception trying to extract cert hash from server cert:\n{e}")
        raise NoServerCertError(detail=str(e))


def _get_server_certificate(response: ClientResponse) -> bytes:
    """
    Extract client certificate from Uvicorn request.
    Simplified for FastAPI-to-FastAPI communication.
    """
    # Get the server certificate from the connection
    # The transport contains the SSL object with peer certificate info
    transport = response.connection.transport
    ssl_object = transport.get_extra_info("ssl_object")

    if ssl_object is None:
        raise ValueError("No SSL connection established")

    # Get the peer certificate in DER format
    cert_der = ssl_object.getpeercert(binary_form=True)

    if cert_der is None:
        raise ValueError("No peer certificate available")

    # Load the DER certificate
    cert = x509.load_der_x509_certificate(cert_der, default_backend())

    return cert


def _get_public_key_hash(cert: Certificate) -> str:
    """
    Compute SHA-256 hash of certificate's public key in DER format.
    This matches the bash snippet's logic:
    openssl x509 -pubkey -noout | openssl pkey -pubin -outform der | sha256sum
    """
    # Extract the public key
    public_key = cert.public_key()

    # Serialize public key to DER format (matching openssl pkey -outform der)
    public_key_der = public_key.public_bytes(
        encoding=serialization.Encoding.DER, format=serialization.PublicFormat.SubjectPublicKeyInfo
    )

    # Compute SHA-256 hash
    hash_digest = hashlib.sha256(public_key_der).hexdigest()

    return hash_digest


def _get_client_certificate(request: Request) -> bytes:
    """
    Extract client certificate from Uvicorn request.
    Simplified for FastAPI-to-FastAPI communication.
    """
    cert_header = request.headers.get("X-Client-Cert")
    if not cert_header:
        raise NoClientCertError(detail="No client certificate provided")

    # Decode the URL-encoded PEM cert from nginx
    cert_pem = unquote(cert_header).encode()

    # Parse the certificate
    cert = x509.load_pem_x509_certificate(cert_pem, default_backend())

    return cert


def extract_nonce(quote: TdxQuote):
    return quote.report_data[:64].lower()


def extract_cert_hash(quote: TdxQuote):
    return quote.report_data[64:128].lower()


def extract_report_data(quote: TdxQuote):
    # Extract nonce from report_data (first printable ASCII portion)
    nonce = extract_nonce(quote)
    cert_hash = extract_cert_hash(quote)

    return nonce, cert_hash


async def verify_quote_signature(quote: TdxQuote) -> TdxVerificationResult:
    """
    Verify the cryptographic signature of a TDX quote using dcap-qvl.

    Args:
        quote_bytes: Raw TDX quote bytes
        verify_collateral: Whether to verify against Intel's collateral (requires PCCS)

    Returns:
        True if signature is valid, False otherwise
    """

    logger.info("Verifying TDX quote signature using dcap-qvl")

    try:
    # Perform quote verification
        verified_report = await get_collateral_and_verify(quote.raw_bytes)

        result = TdxVerificationResult.from_report(verified_report)

        if result.is_valid:
            logger.success("TDX quote signature verification successful")
        else:
            error_msg = verified_report.get("error", "Unknown verification error")
            logger.error(f"TDX quote signature verification failed: {error_msg}")
            raise InvalidSignatureError("TDX quote signature verification failed")

        return result
    except Exception as e:
        logger.error(f"Unexpected error during quote verification: {e}")
        raise InvalidQuoteError(f"Unable to parse provided quote for verification.")


def verify_measurements(quote: TdxQuote) -> bool:
    """
    Verify quote measurements against expected values.

    Args:
        quote: Parsed TDX quote

    Returns:
        True if all measurements match

    Raises:
        MeasurementMismatchError: If any measurements don't match
    """
    expected_rtmrs = (
        settings.expected_boot_rmtrs
        if quote.quote_type == "boot"
        else settings.expected_runtime_rmtrs
    )
    logger.info("Verifying quote against configured MRTD and RTMRS.")
    return _verify_measurements(quote, expected_rtmrs)


def verify_result(quote: TdxQuote, result: TdxVerificationResult) -> bool:
    """
    Verify quote measurements against verification result values.

    Args:
        quote: Parsed TDX quote
        result: The verification result from DCAP

    Returns:
        True if all measurements match

    Raises:
        MeasurementMismatchError: If any measurements don't match
    """
    logger.info("Verifying quote against verification result MRTD and RTMRS.")
    return _verify_measurements(quote, result.rtmrs)


def _verify_measurements(quote: TdxQuote, expected_rtmrs: Dict[str, str]) -> bool:
    try:
        mismatches = []

        # Verify MRTD
        expected_mrtd = settings.expected_mrtd
        if quote.mrtd.upper() != expected_mrtd.upper():
            error_msg = f"MRTD mismatch: expected {expected_mrtd}, got {quote.mrtd}"
            logger.error(error_msg)
            mismatches.append(error_msg)

        # Verify RTMRs
        for rtmr_name, expected_value in expected_rtmrs.items():
            actual_value = quote.rtmrs.get(rtmr_name)
            if not actual_value:
                error_msg = f"Quote missing expected RTMR[{rtmr_name}]"
                logger.error(error_msg)
                mismatches.append(error_msg)
            elif actual_value.upper() != expected_value.upper():
                error_msg = (
                    f"RTMR {rtmr_name} mismatch: expected {expected_value}, got {actual_value}"
                )
                logger.error(error_msg)
                mismatches.append(error_msg)

        # If any mismatches found, raise with generic message
        # (detailed mismatch info is already logged above)
        if mismatches:
            logger.error(f"Measurement verification failed: {'; '.join(mismatches)}")
            raise MeasurementMismatchError()

        logger.info("Measurements verified successfully")
        return True

    except MeasurementMismatchError:
        raise
    except Exception as e:
        logger.error(f"Unexpected error during measurement verification: {e}", exc_info=True)
        # Re-raise as AttestationError for unexpected exceptions
        raise AttestationError("Measurement verification failed due to an unexpected error.")


def get_luks_passphrase() -> str:
    """
    Get the LUKS passphrase for disk decryption.

    Returns:
        LUKS passphrase string
    """

    passphrase = settings.luks_passphrase
    if not passphrase:
        logger.error("No LUKS passphrase configured")
        raise InvalidTdxConfiguration("Missing LUKS phassphrase configuration")

    return passphrase


def generate_cache_passphrase() -> str:
    """
    Generate a new cryptographically secure passphrase for cache volume encryption.

    Returns:
        128-character hex passphrase
    """
    return secrets.token_hex(64)


def _get_fernet() -> Fernet:
    """Get Fernet cipher for encrypting/decrypting cache passphrases.

    Returns:
        Fernet cipher instance

    Raises:
        InvalidTdxConfiguration: If encryption key is not configured
    """
    fernet = settings.fernet_key
    if not fernet:
        logger.error("No cache passphrase encryption key configured")
        raise InvalidTdxConfiguration(
            "CACHE_PASSPHRASE_KEY environment variable must be set. "
            "Generate a valid key with: python -c 'from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())'"
        )
    return fernet


def encrypt_passphrase(passphrase: str) -> str:
    """Encrypt a cache passphrase for storage.

    Args:
        passphrase: Plain text passphrase

    Returns:
        Encrypted passphrase (base64 encoded)
    """
    fernet = _get_fernet()
    encrypted = fernet.encrypt(passphrase.encode())
    return encrypted.decode()


def decrypt_passphrase(encrypted_passphrase: str) -> str:
    """Decrypt a stored cache passphrase.

    Args:
        encrypted_passphrase: Encrypted passphrase (base64 encoded)

    Returns:
        Plain text passphrase
    """
    fernet = _get_fernet()
    decrypted = fernet.decrypt(encrypted_passphrase.encode())
    return decrypted.decode()


async def get_cache_passphrase(db: AsyncSession, miner_hotkey: str, vm_name: str) -> str:
    """
    Get existing cache passphrase for a VM configuration.

    Args:
        db: Database session
        miner_hotkey: Miner hotkey that owns the VM
        vm_name: VM name/identifier

    Returns:
        Cache volume passphrase (decrypted)

    Raises:
        ValueError: If no passphrase exists for this VM configuration
    """
    # Look up VM cache config
    result = await db.execute(
        select(VmCacheConfig).where(
            VmCacheConfig.miner_hotkey == miner_hotkey,
            VmCacheConfig.vm_name == vm_name,
        )
    )
    vm_config = result.scalar_one_or_none()

    if not vm_config:
        raise ValueError(f"No cache passphrase found for VM {vm_name} (miner: {miner_hotkey})")

    logger.info(f"Retrieved existing cache passphrase for VM {vm_name} (miner: {miner_hotkey})")
    return decrypt_passphrase(vm_config.encrypted_passphrase)


async def create_or_update_cache_passphrase(
    db: AsyncSession, miner_hotkey: str, vm_name: str
) -> str:
    """
    Create a new cache passphrase or override existing one for a VM configuration.

    Args:
        db: Database session
        miner_hotkey: Miner hotkey that owns the VM
        vm_name: VM name/identifier

    Returns:
        Cache volume passphrase (decrypted)
    """
    # Generate new passphrase
    passphrase = generate_cache_passphrase()
    encrypted = encrypt_passphrase(passphrase)

    # Check if config already exists
    result = await db.execute(
        select(VmCacheConfig).where(
            VmCacheConfig.miner_hotkey == miner_hotkey,
            VmCacheConfig.vm_name == vm_name,
        )
    )
    vm_config = result.scalar_one_or_none()

    if vm_config:
        # Update existing config
        logger.info(f"Overriding cache passphrase for VM {vm_name} (miner: {miner_hotkey})")
        vm_config.encrypted_passphrase = encrypted
        vm_config.last_boot_at = func.now()
    else:
        # Create new config
        logger.info(f"Creating new cache passphrase for VM {vm_name} (miner: {miner_hotkey})")
        vm_config = VmCacheConfig(
            miner_hotkey=miner_hotkey,
            vm_name=vm_name,
            encrypted_passphrase=encrypted,
            last_boot_at=func.now(),
        )
        db.add(vm_config)

    await db.commit()
    await db.refresh(vm_config)

    return passphrase


async def _track_server(
    db: AsyncSession, id: str, host: str, miner_hotkey: str, is_tee: bool = False
):
    # Add server and nodes to DB
    server = Server(server_id=id, ip=host, miner_hotkey=miner_hotkey, is_tee=is_tee)

    db.add(server)
    await db.commit()
    await db.refresh(server)

    return server
