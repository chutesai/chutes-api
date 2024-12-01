"""
Utils for payments/payouts.
"""

import hvac
import secrets
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import padding
from api.config import settings


def get_wallet_key():
    """
    Get the AES 256 key for payment wallets.
    """
    secret = settings.vault_client.secrets.kv.v2.read_secret_version(
        path="payment-wallet-key",
        mount_point="secret",
    )
    return secret["data"]["data"]["key"]


def ensure_wallet_key_exists():
    """
    Ensure the vault has an AES encryption key for payment wallet secrets.
    """
    existing_wallet_key = False
    try:
        _ = get_wallet_key()
        existing_wallet_key = True
    except hvac.exceptions.InvalidPath:
        pass
    if not existing_wallet_key:
        new_key = secrets.token_bytes(32)
        settings.vault_client.secrets.kv.v2.create_or_update_secret(
            path="payment-wallet-key",
            secret={
                "key": new_key.hex(),
            },
            mount_point="secret",
        )


def encrypt_wallet_secret(secret: str):
    """
    Encrypt a payment wallet secret.
    """
    key = get_wallet_key()
    padder = padding.PKCS7(128).padder()
    cipher = Cipher(
        algorithms.AES(bytes.fromhex(key)),
        modes.CBC(bytes.fromhex(settings.wallet_iv)),
        backend=default_backend(),
    )
    padded_data = padder.update(secret.encode()) + padder.finalize()
    encryptor = cipher.encryptor()
    encrypted_data = encryptor.update(padded_data) + encryptor.finalize()
    return encrypted_data.hex()


def decrypt_wallet_secret(encrypted_secret: str) -> str:
    """
    Decrypt a payment wallet secret.
    """
    key = get_wallet_key()
    cipher = Cipher(
        algorithms.AES(bytes.fromhex(key)),
        modes.CBC(bytes.fromhex(settings.wallet_iv)),
        backend=default_backend(),
    )
    unpadder = padding.PKCS7(128).unpadder()
    decryptor = cipher.decryptor()
    decrypted_data = decryptor.update(bytes.fromhex(encrypted_secret)) + decryptor.finalize()
    unpadded_data = unpadder.update(decrypted_data) + unpadder.finalize()
    return unpadded_data.decode("utf-8")


if __name__ == "__main__":
    ensure_wallet_key_exists()
    print(decrypt_wallet_secret(encrypt_wallet_secret("testing")))
