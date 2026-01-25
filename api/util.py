"""
Utility/helper functions.
"""

import os
import re
import time
import uuid
import ast
import aiodns
import pybase64 as base64
import ctypes
import random
import semver
import string
import aiohttp
import asyncio
import secrets
import hashlib
import datetime
import traceback
import ipaddress
import importlib.util
from io import BytesIO
from PIL import Image
import orjson as json
from functools import lru_cache
from typing import Set
from loguru import logger
from api.config import settings
from async_lru import alru_cache
from urllib.parse import urlparse
from sqlalchemy.future import select
from api.constants import VLM_MAX_SIZE, MIN_REG_BALANCE, INTEGRATED_SUBNETS
from api.metasync import MetagraphNode
from api.permissions import Permissioning
from fastapi import Request, status, HTTPException
from sqlalchemy import func, or_, and_, exists
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import padding, hashes
from ipaddress import ip_address, IPv4Address, IPv6Address
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from scalecodec.utils.ss58 import is_valid_ss58_address, ss58_decode
from async_substrate_interface.async_substrate import AsyncSubstrateInterface
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.hazmat.primitives.serialization import Encoding, PublicFormat

ALLOWED_HOST_RE = re.compile(r"(?!-)[a-z\d-]{1,63}(?<!-)$")
ALLOWED_CHUTE_BUILDERS = {"build_sglang_chute", "build_vllm_chute"}


@lru_cache(maxsize=2500)
def extract_hf_model_name(chute_id: str, code: str) -> str:
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return ""

    def get_model_name(node: ast.Call) -> str:
        for keyword in node.keywords:
            if keyword.arg != "model_name":
                continue
            if isinstance(keyword.value, ast.Constant) and isinstance(keyword.value.value, str):
                return keyword.value.value
            return ""
        return ""

    def is_allowed_builder(node: ast.Call) -> bool:
        if isinstance(node.func, ast.Name):
            return node.func.id in ALLOWED_CHUTE_BUILDERS
        if isinstance(node.func, ast.Attribute):
            return node.func.attr in ALLOWED_CHUTE_BUILDERS
        return False

    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            if isinstance(node.targets[0], ast.Name) and node.targets[0].id == "chute":
                if isinstance(node.value, ast.Call) and is_allowed_builder(node.value):
                    model_name = get_model_name(node.value)
                    if model_name:
                        return model_name
        if isinstance(node, ast.AnnAssign):
            if isinstance(node.target, ast.Name) and node.target.id == "chute":
                if isinstance(node.value, ast.Call) and is_allowed_builder(node.value):
                    model_name = get_model_name(node.value)
                    if model_name:
                        return model_name

    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and is_allowed_builder(node):
            model_name = get_model_name(node)
            if model_name:
                return model_name

    return ""


def is_valid_bittensor_address(address):
    """
    Check if an ss58 appears to be valid or not.
    """
    try:
        if not is_valid_ss58_address(address):
            return False
        decoded = ss58_decode(address)
        prefix = decoded[0]
        return prefix == 42
    except Exception:
        return False


def now_str():
    """
    Return current (UTC) timestamp as string.
    """
    return datetime.datetime.utcnow().isoformat()


def sse(data):
    """
    Format response object for server-side events stream.
    """
    return f"data: {json.dumps(data).decode()}\n\n"


def gen_random_token(k: int = 16) -> str:
    """
    Generate a random token, useful for fingerprints.
    """
    return "".join(random.sample(string.ascii_letters + string.digits, k=k))


def nonce_is_valid(nonce: str) -> bool:
    """Check if the nonce is valid."""
    return nonce and nonce.isdigit() and abs(time.time() - int(nonce)) < 600


def get_signing_message(
    hotkey: str,
    nonce: str,
    payload_str: str | bytes | None,
    purpose: str | None = None,
    payload_hash: str | None = None,
) -> str:
    """Get the signing message for a given hotkey, nonce, and payload."""
    if payload_str:
        if isinstance(payload_str, str):
            payload_str = payload_str.encode()
        return f"{hotkey}:{nonce}:{hashlib.sha256(payload_str).hexdigest()}"
    elif purpose:
        return f"{hotkey}:{nonce}:{purpose}"
    elif payload_hash:
        return f"{hotkey}:{nonce}:{payload_hash}"
    else:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Either payload_str or purpose must be provided",
        )


def is_invalid_ip(ip: IPv4Address | IPv6Address) -> bool:
    """
    Check if IP address is private/local network.
    """
    return (
        ip.is_private
        or ip.is_loopback
        or ip.is_link_local
        or ip.is_multicast
        or ip.is_reserved
        or ip.is_unspecified
    )


async def get_resolved_ips(host: str) -> Set[IPv4Address | IPv6Address]:
    """
    Resolve all IP addresses for a host.
    """
    resolver = aiodns.DNSResolver()
    resolved_ips = set()
    try:
        # IPv4
        try:
            result = await resolver.query(host, "A")
            for answer in result:
                resolved_ips.add(ip_address(answer.host))
        except aiodns.error.DNSError:
            pass

        # IPv6
        try:
            result = await resolver.query(host, "AAAA")
            for answer in result:
                resolved_ips.add(ip_address(answer.host))
        except aiodns.error.DNSError:
            pass
        if not resolved_ips:
            raise ValueError(f"Could not resolve any IP addresses for host: {host}")
        return resolved_ips
    except Exception as exc:
        raise ValueError(f"DNS resolution failed for host {host}: {str(exc)}")


def extract_ip(request: Request) -> str:
    x_forwarded_for = request.headers.get("X-Forwarded-For")
    actual_ip = x_forwarded_for.split(",")[0] if x_forwarded_for else request.client.host
    return actual_ip


async def is_valid_host(host: str) -> bool:
    """
    Validate host (IP or DNS name).
    """
    if not host or len(host) > 255:
        return False
    if not all(ALLOWED_HOST_RE.match(x) for x in host.lower().rstrip(".").split(".")):
        return False
    try:
        # IP address provided.
        addr = ip_address(host)
        return not is_invalid_ip(addr)
    except ValueError:
        # DNS hostname provided, look up IPs.
        try:
            resolved_ips = await asyncio.wait_for(get_resolved_ips(host), 5.0)
            return all(not is_invalid_ip(ip) for ip in resolved_ips)
        except ValueError:
            return False


async def is_registered_to_subnet(session, user, netuid):
    """
    Check if a user is registered to a given subnet.
    """
    result = await session.execute(
        select(
            exists(
                select(1).where(MetagraphNode.netuid == netuid, MetagraphNode.hotkey == user.hotkey)
            )
        )
    )
    return result.scalar()


async def is_registered_to_integrated_subnet(session, user) -> bool:
    """
    Check if a user is registered to an integrated subnet.
    """
    if not user or not getattr(user, "hotkey", None):
        return False
    integrated_netuids = [info["netuid"] for info in INTEGRATED_SUBNETS.values()]
    if not integrated_netuids:
        return False
    stmt = select(
        exists().where(
            MetagraphNode.netuid.in_(integrated_netuids),
            MetagraphNode.hotkey == user.hotkey,
        )
    )
    result = await session.execute(stmt)
    return bool(result.scalar())


async def _limit_dev_activity(session, user, maximum, clazz):
    """
    Limit how many chutes a user can create/update per day.
    """

    if (
        user.username in ("chutes", "rayonlabs")
        or user.validator_hotkey
        or user.subnet_owner_hotkey
        or user.has_role(Permissioning.unlimited_dev)
        or user.user_id
        in (
            "b167f56b-3e8d-5ffa-88bf-5cc6513bb6f4",
            "5260fc63-dbf0-5e76-ae76-811f87fe1e19",
            "7bbd5ffa-b696-5e3a-b4cc-b8aff6854c41",
            "5bf8a979-ea71-54bf-8644-26a3411a3b58",
        )
    ):
        return

    timestamp_filters = [
        clazz.created_at >= func.now() - datetime.timedelta(days=1),
        clazz.deleted_at >= func.now() - datetime.timedelta(days=1),
    ]
    if hasattr(clazz, "updated_at"):
        timestamp_filters.append(clazz.updated_at >= func.now() - datetime.timedelta(days=1))
    query = select(clazz).where(
        and_(
            or_(*timestamp_filters),
            clazz.user_id == user.user_id,
        )
    )
    items = (await session.execute(query)).unique().scalars().all()
    if len(items) >= maximum:
        object_type = str(clazz.__name__).lower().replace("History", "")
        logger.warning(
            f"CHUTERATE: {user.user_id=} has exceeded dev limit: {maximum=} for {object_type}"
        )
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=f"You many only update/create {maximum} {object_type}s per 24 hours.",
        )


async def limit_deployments(session, user, maximum: int = 24):
    from api.chute.schemas import ChuteHistory

    await _limit_dev_activity(session, user, maximum, ChuteHistory)


async def limit_images(session, user, maximum: int = 24):
    from api.image.schemas import ImageHistory

    await _limit_dev_activity(session, user, maximum, ImageHistory)


def aes_encrypt(plaintext: bytes, key: bytes, iv: bytes = None, hex_encode=False) -> str:
    """
    Encrypt with AES.
    """
    if isinstance(key, str):
        key = bytes.fromhex(key)
    if isinstance(plaintext, str):
        plaintext = plaintext.encode()
    if not iv:
        iv = secrets.token_bytes(16)
    padder = padding.PKCS7(128).padder()
    cipher = Cipher(
        algorithms.AES(key),
        modes.CBC(iv),
        backend=default_backend(),
    )
    padded_data = padder.update(plaintext) + padder.finalize()
    encryptor = cipher.encryptor()
    encrypted_data = encryptor.update(padded_data) + encryptor.finalize()
    if not hex_encode:
        return "".join([iv.hex(), base64.b64encode(encrypted_data).decode()])
    return "".join([iv.hex(), encrypted_data.hex()])


def aes_decrypt(ciphertext: bytes, key: bytes, iv: bytes) -> bytes:
    """
    Decrypt an AES encrypted ciphertext.
    """
    if isinstance(key, str):
        key = bytes.fromhex(key)
    if isinstance(ciphertext, str):
        ciphertext = ciphertext.encode()
    if isinstance(iv, str):
        iv = bytes.fromhex(iv)
    cipher = Cipher(
        algorithms.AES(key),
        modes.CBC(iv),
        backend=default_backend(),
    )
    unpadder = padding.PKCS7(128).unpadder()
    decryptor = cipher.decryptor()
    cipher_bytes = base64.b64decode(ciphertext)
    decrypted_data = decryptor.update(cipher_bytes) + decryptor.finalize()
    unpadded_data = unpadder.update(decrypted_data) + unpadder.finalize()
    return unpadded_data


def aes_gcm_decrypt(ciphertext: bytes, key: bytes) -> bytes:
    """
    Decrypt AES-256-GCM ciphertext from runint.
    Format: nonce (12 bytes) || ciphertext || tag (16 bytes)
    Ciphertext is base64 encoded (miner's _encrypt always base64 encodes).
    """
    if isinstance(key, str):
        key = bytes.fromhex(key)

    # Ciphertext is always base64 encoded from miner (string or bytes)
    if isinstance(ciphertext, bytes):
        ciphertext = ciphertext.rstrip(b"\n\r")  # Strip trailing newlines from streaming
        ciphertext = ciphertext.decode("ascii")  # Convert to string for b64decode
    ciphertext = base64.b64decode(ciphertext)

    if len(ciphertext) < 28:  # 12 nonce + 16 tag minimum
        raise ValueError("Ciphertext too short for AES-GCM")

    nonce = ciphertext[:12]
    tag = ciphertext[-16:]
    ct = ciphertext[12:-16]

    cipher = Cipher(
        algorithms.AES(key),
        modes.GCM(nonce, tag),
        backend=default_backend(),
    )
    decryptor = cipher.decryptor()
    plaintext = decryptor.update(ct) + decryptor.finalize()
    return plaintext


def derive_ecdh_session_key(miner_pubkey_hex: str, rint_nonce_hex: str) -> tuple[str, str]:
    """
    Generate an ephemeral ECDH keypair and derive session key with miner's pubkey.

    Args:
        miner_pubkey_hex: Miner's secp256k1 public key as 128 hex chars (64 bytes = x || y)
        rint_nonce_hex: The rint_nonce from instance (32 hex chars = 16 bytes hash)

    Returns:
        (validator_pubkey_hex, session_key_hex): Validator's pubkey to send back,
        and the derived 32-byte session key for AES-256-GCM.

    The session key derivation matches runint:
        SHA256("runint-session-v1" || shared_secret || miner_pubkey || validator_pubkey || rint_nonce)
    """
    # Parse miner's public key (raw x||y format, 64 bytes)
    miner_pubkey_bytes = bytes.fromhex(miner_pubkey_hex)
    if len(miner_pubkey_bytes) != 64:
        raise ValueError(f"Invalid miner pubkey length: {len(miner_pubkey_bytes)}")

    # Parse rint_nonce (16 bytes)
    rint_nonce_bytes = bytes.fromhex(rint_nonce_hex)
    if len(rint_nonce_bytes) != 16:
        raise ValueError(f"Invalid rint_nonce length: {len(rint_nonce_bytes)}")

    # Add 0x04 prefix for uncompressed point format
    miner_pubkey_uncompressed = b"\x04" + miner_pubkey_bytes
    miner_public_key = ec.EllipticCurvePublicKey.from_encoded_point(
        ec.SECP256K1(), miner_pubkey_uncompressed
    )

    # Generate ephemeral validator keypair
    validator_private_key = ec.generate_private_key(ec.SECP256K1(), default_backend())
    validator_public_key = validator_private_key.public_key()

    # Get validator pubkey bytes (remove 0x04 prefix)
    validator_pubkey_uncompressed = validator_public_key.public_bytes(
        Encoding.X962, PublicFormat.UncompressedPoint
    )
    validator_pubkey_bytes = validator_pubkey_uncompressed[1:]  # Remove 0x04
    validator_pubkey_hex = validator_pubkey_bytes.hex()

    # Perform ECDH to get shared secret
    shared_secret = validator_private_key.exchange(ec.ECDH(), miner_public_key)

    # Derive session key: SHA256("runint-session-v1" || shared_secret || miner_pub || validator_pub || rint_nonce)
    h = hashlib.sha256()
    h.update(b"runint-session-v1")
    h.update(shared_secret)
    h.update(miner_pubkey_bytes)
    h.update(validator_pubkey_bytes)
    h.update(rint_nonce_bytes)
    session_key = h.digest()

    return validator_pubkey_hex, session_key.hex()


def aes_gcm_encrypt(plaintext: bytes, key: bytes) -> bytes:
    """
    Encrypt with AES-256-GCM.
    Format: nonce (12 bytes) || ciphertext || tag (16 bytes)
    """
    if isinstance(key, str):
        key = bytes.fromhex(key)
    if isinstance(plaintext, str):
        plaintext = plaintext.encode()

    nonce = secrets.token_bytes(12)
    cipher = Cipher(
        algorithms.AES(key),
        modes.GCM(nonce),
        backend=default_backend(),
    )
    encryptor = cipher.encryptor()
    ciphertext = encryptor.update(plaintext) + encryptor.finalize()
    # nonce || ciphertext || tag
    return nonce + ciphertext + encryptor.tag


def decrypt_instance_response(
    ciphertext: bytes | str,
    instance,
    iv: str = None,
    force_legacy: bool = False,
) -> bytes:
    """
    Decrypt a response from a miner instance.

    For chutes >= 0.5.1: uses AES-256-GCM with rint_session_key (nonce embedded in ciphertext)
    For older chutes: uses legacy AES-CBC with symmetric_key and iv

    Args:
        ciphertext: The encrypted data
        instance: The Instance object with symmetric_key and rint_session_key
        iv: IV for legacy AES-CBC (ignored for new scheme)
        force_legacy: If True, always use legacy AES-CBC (for PoVW which uses symmetric_key)

    Returns:
        Decrypted plaintext bytes
    """
    # PoVW responses always use legacy AES-CBC with symmetric_key
    if force_legacy:
        if iv is None:
            raise ValueError("iv required for legacy AES-CBC decryption")
        return aes_decrypt(ciphertext, instance.symmetric_key, iv)

    # chutes >= 0.5.1 uses AES-256-GCM with ECDH-derived session key
    if semcomp(instance.chutes_version or "0.0.0", "0.5.1") >= 0:
        if not instance.rint_session_key:
            raise ValueError("chutes >= 0.5.1 requires rint_session_key")
        return aes_gcm_decrypt(ciphertext, instance.rint_session_key)

    # Legacy AES-CBC decryption
    if iv is None:
        raise ValueError("iv required for legacy AES-CBC decryption")
    return aes_decrypt(ciphertext, instance.symmetric_key, iv)


def encrypt_instance_request(
    plaintext: bytes, instance, hex_encode=False
) -> tuple[str, str | None]:
    """
    Encrypt a request to a miner instance.

    For chutes >= 0.5.1: uses AES-256-GCM with rint_session_key
    For older chutes: uses legacy AES-CBC with symmetric_key

    Args:
        plaintext: The data to encrypt
        instance: The Instance object with symmetric_key and rint_session_key
        hex_encode: Whether to hex encode (legacy only, ignored for GCM)

    Returns:
        Tuple of (encrypted_string, iv_or_none):
        - For GCM: (base64_ciphertext, None)
        - For legacy: (iv_hex + base64/hex_ciphertext, iv_hex)
    """
    if isinstance(plaintext, str):
        plaintext = plaintext.encode()

    # chutes >= 0.5.1 uses AES-256-GCM with ECDH-derived session key
    if semcomp(instance.chutes_version or "0.0.0", "0.5.1") >= 0:
        if not instance.rint_session_key:
            raise ValueError("chutes >= 0.5.1 requires rint_session_key")
        encrypted = aes_gcm_encrypt(plaintext, instance.rint_session_key)
        if hex_encode:
            return encrypted.hex(), None
        return base64.b64encode(encrypted).decode(), None

    # Legacy AES-CBC encryption
    encrypted = aes_encrypt(plaintext, instance.symmetric_key, hex_encode=hex_encode)
    iv = encrypted[:32]  # First 32 hex chars are the IV
    return encrypted, iv


def use_encryption_v2(chutes_version: str):
    """
    Check if encryption V2 (chutes >= 0.2.0) is enabled.
    """
    if not chutes_version:
        return False
    major, minor = chutes_version.split(".")[:2]
    if major == "0" and int(minor) < 2:
        return False
    return True


def use_encrypted_path(chutes_version: str):
    """
    Check if the URL paths should be encrypted as well.
    """
    if not chutes_version:
        return False
    major, minor, bug = chutes_version.split(".")[:3]
    if int(minor) >= 2 and int(bug) >= 14 or int(minor) > 2:
        return True
    return False


def should_slurp_code(chutes_version: str):
    """
    Check if we should read the code instead of using FS challenges.
    """
    if not chutes_version:
        return False
    major, minor, bug = chutes_version.split(".")[:3]
    if int(minor) >= 2 and int(bug) >= 20 or int(minor) > 2:
        return True
    return False


def derive_envdump_key(key, version):
    """
    Derive the AES key from the envdump mechanism of chutes lib.
    """
    parts = [int(s) for s in version.split(".")]
    target_key = settings.envcheck_key
    target_salt = settings.envcheck_salt
    if parts[1] == 2 and parts[2] >= 51 or parts[1] > 2:
        target_key = settings.envcheck_52_key
        target_salt = settings.envcheck_52_salt
    stored_bytes = bytes.fromhex(target_key)
    user_bytes = key
    combined_secret = stored_bytes + user_bytes
    kdf = PBKDF2HMAC(
        algorithm=hashes.SHA256(),
        length=32,
        salt=bytes.fromhex(target_salt),
        iterations=100000,
        backend=default_backend(),
    )
    key = kdf.derive(combined_secret)
    return key


def decrypt_envdump_cipher(encrypted_b64, key, version):
    """
    Decrypt data that was encrypted from the envcheck chute code.
    """
    actual_key = derive_envdump_key(key, version)
    raw_data = base64.b64decode(encrypted_b64)
    iv = raw_data[:16]
    encrypted_data = raw_data[16:]
    cipher = Cipher(
        algorithms.AES(actual_key),
        modes.CBC(iv),
        backend=default_backend(),
    )
    unpadder = padding.PKCS7(128).unpadder()
    decryptor = cipher.decryptor()
    decrypted_data = decryptor.update(encrypted_data) + decryptor.finalize()
    unpadded_data = unpadder.update(decrypted_data) + unpadder.finalize()
    return unpadded_data


def generate_ip_token(origin_ip, extra_salt: str = None):
    target_string = f"{origin_ip}:{settings.ip_check_salt}"
    if extra_salt:
        target_string = f"{target_string}:{extra_salt}"
    return str(uuid.uuid5(uuid.NAMESPACE_OID, target_string))


def use_opencl_graval(chutes_version: str):
    """
    Check if we should use the opencl/clblast version of graval.
    """
    if not chutes_version:
        return False
    major, minor, bug = chutes_version.split(".")[:3]
    if int(minor) >= 2 and int(bug) == 50 or int(minor) > 2:
        return True
    return False


def semcomp(input_version: str, target_version: str):
    """
    Semver comparison with cleanup.
    """
    if not input_version:
        input_version = "0.0.0"
    re_match = re.match(r"^([0-9]+\.[0-9]+\.[0-9]+).*", input_version)
    clean_version = re_match.group(1) if re_match else "0.0.0"
    return semver.compare(clean_version, target_version)


async def notify_created(instance, gpu_count: int = None, gpu_type: str = None):
    message = f"Instance created: {instance.miner_hotkey=} {instance.instance_id=}"
    if gpu_count:
        message += f" {gpu_count=} {gpu_type=}"
    message += ", broadcasting"
    logger.success(message)
    try:
        log_suffix = ""
        if gpu_count:
            log_suffix = f" on {gpu_count}x{gpu_type}"
        event_data = {
            "reason": "instance_created",
            "message": f"Miner {instance.miner_hotkey} has provisioned an instance of chute {instance.chute_id}{log_suffix}",
            "data": {
                "chute_id": instance.chute_id,
                "gpu_count": gpu_count,
                "gpu_model_name": gpu_type,
                "miner_hotkey": instance.miner_hotkey,
                "instance_id": instance.instance_id,
            },
        }
        await settings.redis_client.publish("events", json.dumps(event_data).decode())
        if instance.config_id:
            event_data["filter_recipients"] = [instance.miner_hotkey]
            event_data["data"]["config_id"] = instance.config_id
            await settings.redis_client.publish("miner_broadcast", json.dumps(event_data).decode())
    except Exception:
        ...


async def notify_deleted(instance, message: str = None):
    logger.warning(
        f"Instance deleted: {instance.miner_hotkey=} {instance.instance_id=}, broadcasting"
    )
    if not message:
        message = f"Miner {instance.miner_hotkey} has deleted instance an instance of chute {instance.chute_id}."
    try:
        event_data = {
            "reason": "instance_deleted",
            "message": message,
            "data": {
                "chute_id": instance.chute_id,
                "miner_hotkey": instance.miner_hotkey,
                "instance_id": instance.instance_id,
                "config_id": instance.config_id,
            },
        }
        await settings.redis_client.publish("events", json.dumps(event_data).decode())
        event_data["filter_recipients"] = [instance.miner_hotkey]
        await settings.redis_client.publish("miner_broadcast", json.dumps(event_data).decode())
    except Exception:
        ...


async def notify_verified(instance):
    logger.success(
        f"Instance verified: {instance.miner_hotkey=} {instance.instance_id=}, broadcasting"
    )
    try:
        event_data = {
            "reason": "instance_verified",
            "data": {
                "instance_id": instance.instance_id,
                "miner_hotkey": instance.miner_hotkey,
            },
            "filter_recipients": [instance.miner_hotkey],
        }
        await settings.redis_client.publish("miner_broadcast", json.dumps(event_data).decode())
        await settings.redis_client.publish(
            "events",
            json.dumps(
                {
                    "reason": "instance_hot",
                    "message": f"Miner {instance.miner_hotkey} instance {instance.instance_id} chute {instance.chute_id} has been verified, now 'hot'!",
                    "data": {
                        "chute_id": instance.chute_id,
                        "miner_hotkey": instance.miner_hotkey,
                    },
                }
            ).decode(),
        )
    except Exception:
        ...


async def notify_job_deleted(job):
    try:
        await settings.redis_client.publish(
            "miner_broadcast",
            json.dumps(
                {
                    "reason": "job_deleted",
                    "data": {
                        "instance_id": job.instance_id,
                        "job_id": job.job_id,
                    },
                }
            ).decode(),
        )
    except Exception:
        ...


async def notify_activated(instance):
    try:
        message = f"Miner {instance.miner_hotkey} has activated instance {instance.instance_id} chute {instance.chute_id}"
        logger.success(message)
        event_data = {
            "reason": "instance_activated",
            "message": message,
            "data": {
                "chute_id": instance.chute_id,
                "miner_hotkey": instance.miner_hotkey,
                "instance_id": instance.instance_id,
                "config_id": instance.config_id,
            },
        }
        await settings.redis_client.publish("events", json.dumps(event_data).decode())
        if instance.config_id:
            event_data["filter_recipients"] = [instance.miner_hotkey]
            await settings.redis_client.publish("miner_broadcast", json.dumps(event_data).decode())
    except Exception as exc:
        logger.warning(f"Error broadcasting instance event: {exc}")


def get_current_hf_commit(model_name: str):
    """
    Helper to load the current main commit for a given repo.
    """
    from huggingface_hub import HfApi

    api = HfApi()
    for ref in api.list_repo_refs(model_name).branches:
        if ref.ref == "refs/heads/main":
            return ref.target_commit
    return None


async def recreate_vlm_payload(request_body: dict):
    """
    Check if a VLM request is valid (for us), download images/videos locally and pass to miners as b64.
    """
    futures = []

    async def _inject_b64(url, obj, key, visual_type):
        obj[key] = reformat_vlm_asset(await fetch_vlm_asset(url), visual_type)

    if not request_body.get("messages"):
        return
    for message in request_body["messages"]:
        if not isinstance(message.get("content"), list):
            continue

        for content_item in message["content"]:
            if not isinstance(content_item, dict):
                continue
            for key in ("image", "image_url", "video", "video_url"):
                if key not in content_item:
                    continue
                visual_data = content_item[key]
                visual_type = "video" if "video" in key else "image"
                if isinstance(visual_data, dict) and "url" in visual_data:
                    url = visual_data["url"]
                    if url.startswith(f"data:{visual_type}") or url.startswith("data:"):
                        continue
                    parsed_url = urlparse(url)
                    if parsed_url.scheme.lower() not in ("https", "http"):
                        raise HTTPException(
                            status_code=status.HTTP_400_BAD_REQUEST,
                            detail=f"Only HTTP(s) URLs are supported for {visual_type}s: {parsed_url.scheme} is not supported",
                        )
                    if parsed_url.port is not None and parsed_url.port not in (80, 443):
                        raise HTTPException(
                            status_code=status.HTTP_400_BAD_REQUEST,
                            detail=f"Only HTTP(s) standard ports are supported for {visual_type}s, port {parsed_url.port} is not supported",
                        )
                    futures.append(_inject_b64(url, visual_data, "url", visual_type))

                elif isinstance(visual_data, str):
                    if visual_data[:5].lower() == "data:":
                        continue
                    parsed_url = urlparse(visual_data)
                    if parsed_url.scheme != "https":
                        raise HTTPException(
                            status_code=status.HTTP_400_BAD_REQUEST,
                            detail=f"Only HTTPS URLs are supported for {visual_type}s. Got scheme: {parsed_url.scheme}",
                        )
                    if parsed_url.port is not None and parsed_url.port != 443:
                        raise HTTPException(
                            status_code=status.HTTP_400_BAD_REQUEST,
                            detail=f"Only HTTPS URLs on port 443 are supported for {visual_type}s. Got port: {parsed_url.port}",
                        )
                    futures.append(_inject_b64(visual_data, content_item, key, visual_type))

    # Perform asset downloads concurrently.
    if len(futures) > 8:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Exceeded maximum image URLs per request: {len(futures)}",
        )
    if futures:
        try:
            started_at = time.time()
            await asyncio.gather(*futures)
            logger.success(
                f"finished recreate_vlm_payload(..) with {len(futures)} "
                f"remote objects  in {time.time() - started_at} seconds"
            )
        except Exception as exc:
            logger.error(
                f"Failed to update images/videos to base64: {str(exc)}\n{traceback.format_exc()}"
            )
            if isinstance(exc, HTTPException):
                raise
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Failed to load image/video data: {str(exc)}",
            )


async def fetch_vlm_asset(url: str) -> bytes:
    """
    Fetch an asset (image or video) from the specified URL (for VLMs).
    """
    logger.info(f"VLM sixtyfourer: downloading vision asset from {url=}")
    timeout = aiohttp.ClientTimeout(connect=2, total=30)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        try:
            async with session.get(url) as response:
                if response.status != 200:
                    raise HTTPException(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        detail=f"Failed to fetch {url}: {response.status=}",
                    )
                content_type = response.headers.get("Content-Type", "").lower()
                if not content_type.startswith(("image/", "video/")):
                    logger.error(f"VLM sixtyfourer: invalid image URL: {content_type=} for {url=}")
                    raise HTTPException(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        detail=f"Invalid image URL: {content_type=} for {url=}",
                    )
                content_length = response.headers.get("Content-Length")
                if content_length and int(content_length) > VLM_MAX_SIZE:
                    logger.error(
                        f"VLM sixtyfourer: max size is {VLM_MAX_SIZE} bytes, {url=} has size {content_length} bytes"
                    )
                    raise HTTPException(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        detail=f"VLM asset max size is {VLM_MAX_SIZE} bytes, {url=} has size {content_length} bytes",
                    )
                chunks = []
                total_size = 0
                async for chunk in response.content.iter_chunked(32768):
                    total_size += len(chunk)
                    if total_size > VLM_MAX_SIZE:
                        logger.error(
                            f"VLM sixtyfourer: max size is {VLM_MAX_SIZE} bytes, already read {total_size=}"
                        )
                        raise HTTPException(
                            status_code=status.HTTP_400_BAD_REQUEST,
                            detail=f"VLM asset max size is {VLM_MAX_SIZE} bytes, already read {total_size=}",
                        )
                    chunks.append(chunk)
                logger.success(f"VLM sixtyfourer: successfully downloaded {url=}")
                return b"".join(chunks)
        except asyncio.TimeoutError:
            logger.error(f"VLM sixtyfourer: timeout downloading {url=}")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Timeout fetching image for VLM processing from {url=}",
            )
        except Exception as exc:
            logger.error(f"VLM sixtyfourer: unhandled download exception: {str(exc)}")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Unexpected error attempting to fetch image for VLM processing: {str(exc)}",
            )


def reformat_vlm_asset(data_bytes: bytes, visual_type: str = "image", max_size: int = 1024) -> str:
    """
    Pre-fetch and convert to base64 images/videos for vision models.
    """
    if visual_type == "image":
        img = Image.open(BytesIO(data_bytes))
        if img.width > max_size or img.height > max_size:
            scale_factor = max_size / max(img.width, img.height)
            new_width = int(img.width * scale_factor)
            new_height = int(img.height * scale_factor)
            logger.warning(
                f"Received large VLM payload image, resizing from {img.width=} {img.height=} to {new_width=} {new_height=}"
            )
            img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
        buffer = BytesIO()
        img_format = img.format if img.format else "PNG"
        if img_format == "JPEG":
            if img.mode in ("RGBA", "P"):
                rgb_img = Image.new("RGB", img.size, (255, 255, 255))
                rgb_img.paste(img, mask=img.split()[3] if img.mode == "RGBA" else None)
                img = rgb_img
        img.save(buffer, format=img_format)
        data_bytes = buffer.getvalue()
        return f"data:image/png;base64,{base64.b64encode(data_bytes).decode()}"
    return f"data:video/mp4;base64,{base64.b64encode(data_bytes).decode()}"


def has_legacy_private_billing(chute):
    if chute.public or "/affine" in chute.name.lower():
        return False
    return chute.created_at.replace(tzinfo=None) < datetime.datetime(
        year=2025, month=9, day=10, hour=16, tzinfo=None
    )


@alru_cache(maxsize=1)
async def get_cloudflare_ips():
    ipv4_url = "https://www.cloudflare.com/ips-v4"
    ipv6_url = "https://www.cloudflare.com/ips-v6"
    cloudflare_ranges = []
    async with aiohttp.ClientSession() as session:
        response = await session.get(ipv4_url)
        for line in (await response.text()).strip().split("\n"):
            cloudflare_ranges.append(ipaddress.ip_network(line))
        response = await session.get(ipv6_url)
        for line in (await response.text()).strip().split("\n"):
            cloudflare_ranges.append(ipaddress.ip_network(line))
    return cloudflare_ranges


async def is_cloudflare_ip(ip_address):
    cloudflare_ranges = await get_cloudflare_ips()
    ip = ipaddress.ip_address(ip_address)
    for cf_range in cloudflare_ranges:
        if ip in cf_range:
            return True
    return False


def _image_supports_cllmv(image, name: str, min_version: int) -> bool:
    if image.name != name:
        return False
    tag = image.tag.lower()
    if not tag.startswith("nightly-") or len(tag) < 18:
        return False
    date_part = tag[8:][:10]
    try:
        date_num = int(date_part)
        return date_num >= min_version
    except (ValueError, Exception):
        return False


def image_supports_cllmv(
    image, min_sglang_version: int = 2025100801, min_vllm_version: int = 2026011303
) -> bool:
    if _image_supports_cllmv(image, "sglang", min_version=min_sglang_version):
        return True
    if image.name == "vllm" and _image_supports_cllmv(image, "vllm", min_version=min_vllm_version):
        return True
    return False


async def validate_tool_call_arguments(body: dict) -> None:
    if not body.get("messages"):
        return
    for message in body["messages"]:
        if message.get("role") == "assistant" and message.get("tool_calls"):
            for item in message["tool_calls"]:
                if (
                    isinstance(item.get("function"), dict)
                    and "arguments" in item["function"]
                    and isinstance(item["function"]["arguments"], str)
                ):
                    if not item["function"]["arguments"]:
                        item["function"]["arguments"] = "null"
                    else:
                        try:
                            _ = json.loads(item["function"]["arguments"])
                        except (ValueError, json.JSONDecodeError) as exc:
                            logger.warning(f"INVALIDFUNCTIONJSON: {str(exc)}")
                            raise HTTPException(
                                status_code=status.HTTP_400_BAD_REQUEST,
                                detail="Invalid tool_calls.function.arguments value, expected JSON",
                            )


async def has_minimum_balance_for_registration(
    coldkey: str, hotkey: str = None, minimum: float = MIN_REG_BALANCE
) -> bool:
    substrate = None
    try:
        substrate = AsyncSubstrateInterface(url=settings.subtensor)
        await substrate.initialize()
        chain_head = await substrate.get_chain_head()
        block = await substrate.get_block_number(chain_head)
        block_hash = await substrate.get_block_hash(block)
        result = await substrate.query(
            module="System",
            storage_function="Account",
            params=[coldkey],
            block_hash=block_hash,
        )
        rao = result["data"]["free"]
        tao = rao / (10**9)
        if tao < minimum:
            logger.warning(f"MINREGBALANCE: {coldkey=} only has {tao=}, less than {minimum=}")
            return False
        logger.success(f"MINREGBALANCE: {coldkey=} has {tao=}, above {minimum=}")
        return True
    except Exception as exc:
        logger.error(f"MINREGBALANCE: failed to check minimum registration balance: {str(exc)}")
        return True
    finally:
        if substrate:
            try:
                await substrate.close()
            except Exception:
                ...


def load_shared_object(pkg_name: str, filename: str):
    spec = importlib.util.find_spec(pkg_name)
    if not spec or not spec.submodule_search_locations:
        raise ImportError(f"Package {pkg_name} not found")
    pkg_dir = spec.submodule_search_locations[0]
    path = os.path.join(pkg_dir, filename)
    return ctypes.CDLL(path)
