"""
Utility/helper functions.
"""

import re
import ast
import time
import uuid
import aiodns
import base64
import random
import semver
import string
import asyncio
import secrets
import hashlib
import datetime
import orjson as json
from typing import Set
from loguru import logger
from api.config import settings
from urllib.parse import urlparse
from sqlalchemy.future import select
from api.metasync import MetagraphNode
from api.payment.schemas import Payment
from api.fmv.fetcher import get_fetcher
from api.permissions import Permissioning
from fastapi import status, HTTPException
from sqlalchemy import func, or_, and_, exists
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import padding, hashes
from ipaddress import ip_address, IPv4Address, IPv6Address
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from scalecodec.utils.ss58 import is_valid_ss58_address, ss58_decode
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes

ALLOWED_HOST_RE = re.compile(r"(?!-)[a-z\d-]{1,63}(?<!-)$")

DANGEROUS_BUILTINS = {
    "eval",
    "exec",
    "compile",
    "__import__",
    "open",
    "input",
    "breakpoint",
    "help",
    "dir",
    "globals",
    "locals",
    "vars",
    "setattr",
    "delattr",
    "getattr",
    "type",
    "classmethod",
    "staticmethod",
    "property",
    "super",
    "isinstance",
    "issubclass",
    "callable",
    "hasattr",
    "hash",
    "id",
    "object",
    "memoryview",
    "bytearray",
    "bytes",
    "frozenset",
    "set",
    "dict",
    "list",
    "tuple",
    "range",
    "slice",
    "filter",
    "map",
    "zip",
    "enumerate",
    "reversed",
    "sorted",
    "any",
    "all",
}


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


async def ensure_is_developer(session, user, raise_: bool = True):
    """
    Ensure a user is a developer, otherwise raise exception with helpful info.
    """
    if user.has_role(Permissioning.developer):
        return None
    total_query = select(func.sum(Payment.usd_amount)).where(
        Payment.user_id == user.user_id, Payment.purpose == "developer"
    )
    total_payments = (await session.execute(total_query)).scalar() or 0
    fetcher = get_fetcher()
    fmv = await fetcher.get_price("tao")
    required_tao = (settings.developer_deposit - total_payments) / fmv
    exc = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail=(
            "You do not have developer permissions, to enable developer permissions, "
            f"deposit ${settings.developer_deposit} USD worth of tao (currently ~{required_tao} tao) "
            f"to your developer deposit address: {user.developer_payment_address}"
        ),
    )
    if not raise_:
        return exc
    raise exc


async def is_affine_registered(session, user):
    """
    Check if a user is registered on affine (thereby allowing limited dev activity).
    """
    result = await session.execute(
        select(
            exists(
                select(1).where(MetagraphNode.netuid == 120, MetagraphNode.hotkey == user.hotkey)
            )
        )
    )
    return result.scalar()


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


def check_vlm_payload(request_body: dict):
    """
    Check if a VLM request is valid (for us).
    """
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
                elif isinstance(visual_data, str):
                    if visual_data.startswith(f"data:{visual_type}") or visual_data.startswith(
                        "data:"
                    ):
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


def check_affine_code(code: str) -> tuple[bool, str]:
    """
    Check if an affine model meets the requirements (LLM chute using SGLang or vLLM).
    """
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        return False, f"Syntax error: {e}"

    imported_names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name != "os":
                    return False, f"Invalid import: {alias.name}. Only 'os' is allowed"
        elif isinstance(node, ast.ImportFrom):
            if node.module is None or not node.module.startswith("chutes."):
                return False, f"Invalid import from: {node.module}. Only 'from chutes.*' is allowed"
            for alias in node.names:
                imported_names.add(alias.asname if alias.asname else alias.name)

    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name) and node.func.id in DANGEROUS_BUILTINS:
                return False, f"Dangerous function '{node.func.id}' is not allowed"
            elif isinstance(node.func, ast.Attribute):
                if node.func.attr in DANGEROUS_BUILTINS:
                    return False, f"Dangerous function '{node.func.attr}' is not allowed"

        if isinstance(node, ast.Attribute):
            if node.attr.startswith("__") and node.attr.endswith("__"):
                dangerous_attrs = {
                    "__builtins__",
                    "__globals__",
                    "__code__",
                    "__class__",
                    "__subclasses__",
                    "__bases__",
                    "__mro__",
                    "__dict__",
                    "__func__",
                    "__self__",
                    "__module__",
                    "__closure__",
                    "__annotations__",
                    "__kwdefaults__",
                    "__defaults__",
                }
                if node.attr in dangerous_attrs:
                    return False, f"Access to '{node.attr}' is not allowed"

            if isinstance(node.value, ast.Name) and node.value.id == "os":
                if node.attr not in ["environ", "getenv"]:
                    return (
                        False,
                        f"os.{node.attr} is not allowed. Only os.environ and os.getenv are permitted",
                    )

        if isinstance(node, ast.Subscript):
            if (
                isinstance(node.value, ast.Attribute)
                and isinstance(node.value.value, ast.Name)
                and node.value.value.id == "os"
                and node.value.attr == "environ"
            ):
                pass
            elif isinstance(node.value, ast.Attribute) and node.value.attr in [
                "__getitem__",
                "__setitem__",
                "__delitem__",
            ]:
                return False, "Direct access to special methods is not allowed"

        if isinstance(node, ast.ClassDef):
            return False, "Class definitions are not allowed"

        if isinstance(node, ast.FunctionDef) and not isinstance(node, ast.AsyncFunctionDef):
            for parent in ast.walk(tree):
                if parent != node and isinstance(parent, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    for child in ast.walk(parent):
                        if child == node:
                            return False, "Nested function definitions are not allowed"

        if isinstance(
            node,
            (
                ast.AsyncFunctionDef,
                ast.GeneratorExp,
                ast.ListComp,
                ast.SetComp,
                ast.DictComp,
                ast.Yield,
                ast.YieldFrom,
                ast.Raise,
                ast.Try,
                ast.ExceptHandler,
                ast.With,
                ast.Assert,
                ast.Global,
                ast.Nonlocal,
            ),
        ):
            return False, f"{node.__class__.__name__} is not allowed"

        if isinstance(node, ast.FunctionDef):
            for decorator in node.decorator_list:
                if isinstance(decorator, ast.Call):
                    if (
                        isinstance(decorator.func, ast.Attribute)
                        and isinstance(decorator.func.value, ast.Name)
                        and decorator.func.value.id == "chute"
                        and decorator.func.attr == "cord"
                    ):
                        return False, "@chute.cord decorators are not allowed"
                elif isinstance(decorator, ast.Attribute):
                    if (
                        isinstance(decorator.value, ast.Name)
                        and decorator.value.id == "chute"
                        and decorator.attr == "cord"
                    ):
                        return False, "@chute.cord decorators are not allowed"
                else:
                    return False, "Decorators are not allowed"

        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Attribute):
                    current = target
                    attrs = []
                    while isinstance(current, ast.Attribute):
                        attrs.append(current.attr)
                        current = current.value

                    if isinstance(current, ast.Name) and current.id == "chute":
                        attr_chain = ".".join(reversed(attrs))
                        return False, f"Assignment to chute.{attr_chain} is not allowed"

    assignments = {}
    chute_assignment = None
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    var_name = target.id
                    if isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Name):
                        func_name = node.value.func.id
                        if func_name in ["build_sglang_chute", "build_vllm_chute"]:
                            if func_name not in imported_names:
                                return False, f"Function {func_name} is used but not imported"
                            if var_name == "chute":
                                if chute_assignment is not None:
                                    return False, "Multiple assignments to 'chute' variable"
                                chute_assignment = func_name
                                for keyword in node.value.keywords:
                                    if keyword.arg == "image":
                                        if not (
                                            isinstance(keyword.value, ast.Constant)
                                            and isinstance(keyword.value.value, str)
                                        ):
                                            return (
                                                False,
                                                "image argument must be a string literal, not Image(...)",
                                            )
                                        image_str = keyword.value.value
                                        if not (
                                            image_str.startswith("chutes/sglang")
                                            or image_str.startswith("chutes/vllm")
                                        ):
                                            return (
                                                False,
                                                "image must start with 'chutes/sglang' or 'chutes/vllm'",
                                            )
                                    elif keyword.arg == "engine_args":
                                        arg_str = ast.unparse(keyword.value)
                                        if (
                                            "trust_remote_code" in arg_str
                                            or "trust-remote-code" in arg_str
                                        ):
                                            return (
                                                False,
                                                "engine_args cannot contain 'trust_remote_code' or 'trust-remote-code'",
                                            )
                            else:
                                return (
                                    False,
                                    f"Function {func_name} must be assigned to variable 'chute', not '{var_name}'",
                                )
                    assignments[var_name] = node

    if chute_assignment is None:
        return False, "No 'chute' variable found calling build_sglang_chute or build_vllm_chute"

    top_level_vars = set()
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    top_level_vars.add(target.id)

    top_level_vars.discard("chute")
    if top_level_vars:
        return (
            False,
            f"Found extra variables: {', '.join(sorted(top_level_vars))}. Only 'chute' is allowed",
        )

    return True, f"Valid chute file with {chute_assignment}"


def fix_glm_tool_arguments(request_body: dict):
    """
    Check if a request is passing string arguments to tools and parse them.
    """
    if (
        not request_body.get("messages")
        or (request_body.get("model") or "").lower() != "zai-org/glm-4.5-fp8"
    ):
        return
    for message in request_body["messages"]:
        calls = message.get("tool_calls")
        if isinstance(calls, list):
            for call in calls:
                if isinstance(call.get("function", {}).get("arguments"), str) and call["function"][
                    "arguments"
                ].strip().startswith("{"):
                    try:
                        args = json.loads(call["function"]["arguments"])
                        formatted = []
                        for key, value in args.items():
                            formatted.append(f"<arg_key>{key}</arg_key>")
                            formatted_value = value if isinstance(value, str) else json.dumps(value)
                            formatted.append(f"<arg_value>{formatted_value}</arg_value>")
                        call["function"]["arguments"] = "\n".join(formatted) + "\n"
                    except Exception as exc:
                        logger.error(f"ERROR CHECKING GLM FUNCTION ARGUMENTS: {str(exc)}")
