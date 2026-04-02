"""
ECIES encryption over Ristretto255 for encrypted log capture.

Uses ephemeral ECDH with the chute owner's Sr25519 public key so that
only the owner can decrypt the stored logs.

Scheme:
  1. Generate ephemeral Ristretto255 scalar + public point
  2. ECDH: shared_secret = scalarmult(ephemeral_scalar, user_pubkey)
  3. Derive AES key via HKDF-SHA256 with domain separation
  4. Encrypt with AES-256-GCM
  5. Store (ephemeral_pubkey, nonce, ciphertext, tag)
  6. Discard ephemeral scalar
"""

import os
import rbcl
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.kdf.hkdf import HKDF
from cryptography.hazmat.primitives import hashes

# Domain separation context for HKDF
HKDF_INFO = b"chutes-encrypted-logs-v1"


def generate_ephemeral_keypair() -> tuple[bytes, bytes]:
    """Generate an ephemeral Ristretto255 scalar and public point."""
    scalar = rbcl.crypto_core_ristretto255_scalar_random()
    # Public point = scalar * basepoint
    # rbcl doesn't expose a direct basepoint mult, so use the hash-to-point
    # approach: generate from the scalar via the standard basepoint.
    # Actually, crypto_scalarmult_ristretto255_base exists in newer libsodium.
    # Fall back to multiplying the standard basepoint.
    try:
        pubkey = rbcl.crypto_scalarmult_ristretto255_base(scalar)
    except AttributeError:
        # Older rbcl: use generator point (hash of empty string is NOT correct;
        # need to use the actual Ristretto basepoint)
        # The Ristretto255 basepoint encoding (compressed):
        RISTRETTO_BASEPOINT = bytes.fromhex(
            "e2f2ae0a6abc4e71a884a961c500515f58e30b6aa582dd8db6a65945e08d2d76"
        )
        pubkey = rbcl.crypto_scalarmult_ristretto255(scalar, RISTRETTO_BASEPOINT)
    return scalar, pubkey


def derive_symmetric_key(
    shared_secret: bytes,
    ephemeral_pubkey: bytes,
    user_pubkey: bytes,
) -> bytes:
    """Derive a 32-byte AES key from the ECDH shared secret with domain separation."""
    # Bind ephemeral and user pubkeys into the KDF for context
    salt = ephemeral_pubkey + user_pubkey
    hkdf = HKDF(
        algorithm=hashes.SHA256(),
        length=32,
        salt=salt,
        info=HKDF_INFO,
    )
    return hkdf.derive(shared_secret)


def encrypt_log_chunk(
    plaintext: bytes,
    user_pubkey: bytes,
    ephemeral_scalar: bytes,
    ephemeral_pubkey: bytes,
) -> bytes:
    """
    Encrypt a log chunk for the given user.

    Returns: nonce (12 bytes) + ciphertext+tag (variable length)
    """
    # ECDH
    shared_secret = rbcl.crypto_scalarmult_ristretto255(ephemeral_scalar, user_pubkey)
    if shared_secret is None:
        raise ValueError("ECDH failed: invalid point or identity result")

    # Derive AES key
    aes_key = derive_symmetric_key(shared_secret, ephemeral_pubkey, user_pubkey)

    # Encrypt with AES-256-GCM
    nonce = os.urandom(12)
    aesgcm = AESGCM(aes_key)
    ciphertext = aesgcm.encrypt(nonce, plaintext, None)

    return nonce + ciphertext


def decrypt_log_chunk(
    encrypted: bytes,
    user_private_key_scalar: bytes,
    user_pubkey: bytes,
    ephemeral_pubkey: bytes,
) -> bytes:
    """
    Decrypt a log chunk using the user's private key scalar.

    Args:
        encrypted: nonce (12 bytes) + ciphertext+tag
        user_private_key_scalar: first 32 bytes of Sr25519 secret key
        user_pubkey: user's 32-byte Sr25519 public key
        ephemeral_pubkey: the ephemeral public key stored with the log session
    """
    # ECDH: same shared secret as encryption
    shared_secret = rbcl.crypto_scalarmult_ristretto255(user_private_key_scalar, ephemeral_pubkey)
    if shared_secret is None:
        raise ValueError("ECDH failed: invalid point or identity result")

    # Derive same AES key
    aes_key = derive_symmetric_key(shared_secret, ephemeral_pubkey, user_pubkey)

    # Decrypt
    nonce = encrypted[:12]
    ciphertext = encrypted[12:]
    aesgcm = AESGCM(aes_key)
    return aesgcm.decrypt(nonce, ciphertext, None)


def validate_ristretto_point(point: bytes) -> bool:
    """Validate that a 32-byte value is a valid Ristretto255 point."""
    try:
        return rbcl.crypto_core_ristretto255_is_valid_point(point) == True  # noqa: E712
    except Exception:
        return False
