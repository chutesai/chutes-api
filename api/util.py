"""
Utility/helper functions.
"""

import re
import asyncio
import aiodns
import datetime
import hashlib
import random
import string
import uuid
import time
import secrets
import orjson as json
import base64
import semver
from loguru import logger
from typing import Set
from ipaddress import ip_address, IPv4Address, IPv6Address
from fastapi import status, HTTPException
from sqlalchemy import func, or_, and_
from sqlalchemy.future import select
from api.config import settings
from api.payment.schemas import Payment
from api.fmv.fetcher import get_fetcher
from api.permissions import Permissioning
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import padding, hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from scalecodec.utils.ss58 import is_valid_ss58_address, ss58_decode

ALLOWED_HOST_RE = re.compile(r"(?!-)[a-z\d-]{1,63}(?<!-)$")


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
    return False


async def ensure_is_developer(session, user):
    """
    Ensure a user is a developer, otherwise raise exception with helpful info.
    """
    if user.has_role(Permissioning.developer):
        return
    total_query = select(func.sum(Payment.usd_amount)).where(
        Payment.user_id == user.user_id, Payment.purpose == "developer"
    )
    total_payments = (await session.execute(total_query)).scalar() or 0
    fetcher = get_fetcher()
    fmv = await fetcher.get_price("tao")
    required_tao = (settings.developer_deposit - total_payments) / fmv
    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail=(
            "You do not have developer permissions, to enable developer permissions, "
            f"deposit ${settings.developer_deposit} USD worth of tao (currently ~{required_tao} tao) "
            f"to your developer deposit address: {user.developer_payment_address}"
        ),
    )


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


async def limit_deployments(session, user, maximum: int = 12):
    from api.chute.schemas import ChuteHistory

    await _limit_dev_activity(session, user, maximum, ChuteHistory)


async def limit_images(session, user, maximum: int = 18):
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
    clean_version = re.match(r"^([0-9]+\.[0-9]+\.[0-9]+).*", input_version).group(1)
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
