import gc
import re
import os
import uuid
import time
import pybase64 as base64
import asyncio
import aiohttp
import socket
import struct
import random
import hashlib
import orjson as json
import secrets
import traceback
import tempfile
from contextlib import asynccontextmanager
from loguru import logger
from api.config import settings
from api.util import (
    decrypt_instance_response,
    encrypt_instance_request,
    decrypt_envdump_cipher,
    semcomp,
    notify_deleted,
    notify_job_deleted,
)
from api.database import get_session
from api.chute.schemas import Chute
from api.image.schemas import Image
from api.job.schemas import Job
from api.exceptions import EnvdumpMissing
from sqlalchemy import text, update, func, select
from sqlalchemy.orm import joinedload, selectinload
import api.database.orms  # noqa
import api.miner_client as miner_client
from api.instance.schemas import Instance, LaunchConfig
from api.instance.util import invalidate_instance_cache
from api.chute.codecheck import is_bad_code


TCP_STATES = {
    "01": "ESTABLISHED",
    "02": "SYN_SENT",
    "03": "SYN_RECV",
    "04": "FIN_WAIT1",
    "05": "FIN_WAIT2",
    "06": "TIME_WAIT",
    "07": "CLOSE",
    "08": "CLOSE_WAIT",
    "09": "LAST_ACK",
    "0A": "LISTEN",
    "0B": "CLOSING",
    "0C": "NEW_SYN_RECV",
}

# Short lived chutes (probably just to get bounties).
SHORT_LIVED_CHUTES = """
SELECT instance_audit.chute_id AS chute_id, EXTRACT(EPOCH FROM MAX(instance_audit.deleted_at) - MIN(instance_audit.created_at)) AS lifetime
FROM instance_audit
LEFT OUTER JOIN chutes ON instance_audit.chute_id = chutes.chute_id
WHERE chutes.name IS NULL
AND deleted_at >= now() - interval '7 days'
GROUP BY instance_audit.chute_id
HAVING EXTRACT(EPOCH FROM MAX(instance_audit.deleted_at) - MIN(instance_audit.created_at)) <= 86400
"""


def use_encrypted_slurp(chutes_version: str) -> bool:
    """
    Check if the chutes version uses encrypted slurp responses or not.
    """
    if not chutes_version:
        return False
    major, minor, bug = chutes_version.split(".")[:3]
    encrypted_slurp = False
    if major == "0" and int(minor) >= 2 and (int(minor) > 2 or int(bug) >= 20):
        encrypted_slurp = True
    return encrypted_slurp


async def load_chute_instances(chute_id):
    """
    Get all instances of a chute.
    """
    async with get_session() as session:
        query = (
            select(Instance)
            .join(Instance.config)
            .where(
                Instance.chute_id == chute_id,
                Instance.active.is_(True),
                Instance.verified.is_(True),
                LaunchConfig.env_type != "tee",  # Exclude TEE
            )
            .options(joinedload(Instance.nodes))
        )
        instances = (await session.execute(query)).unique().scalars().all()
        return instances


async def purge(target, reason="miner failed watchtower probes", valid_termination=False):
    """
    Purge an instance.
    """
    async with get_session() as session:
        await session.execute(
            text("DELETE FROM instances WHERE instance_id = :instance_id"),
            {"instance_id": target.instance_id},
        )
        await session.execute(
            text(
                "UPDATE instance_audit SET deletion_reason = :reason, valid_termination = :valid_termination WHERE instance_id = :instance_id"
            ),
            {
                "instance_id": target.instance_id,
                "reason": reason,
                "valid_termination": valid_termination,
            },
        )

        # Fail associated jobs.
        job = (
            (await session.execute(select(Job).where(Job.instance_id == target.instance_id)))
            .unique()
            .scalar_one_or_none()
        )
        if job and not job.finished_at:
            job.status = "error"
            job.error_detail = f"Instance failed monitoring probes: {reason=}"
            job.miner_terminated = True
            job.finished_at = func.now()
            await notify_job_deleted(job)

        await session.commit()


async def purge_and_notify(
    target, reason="miner failed watchtower probes", valid_termination=False
):
    """
    Purge an instance and send a notification with the reason.
    """
    await purge(target, reason=reason, valid_termination=valid_termination)
    await notify_deleted(
        target,
        message=f"Instance {target.instance_id} of miner {target.miner_hotkey} deleted by watchtower {reason=}",
    )
    await invalidate_instance_cache(target.chute_id, instance_id=target.instance_id)


async def do_slurp(instance, payload, encrypted_slurp):
    """
    Slurp a remote file.
    """
    enc_payload, iv = encrypt_instance_request(json.dumps(payload), instance)
    path, _ = encrypt_instance_request("/_slurp", instance, hex_encode=True)
    async with miner_client.post(
        instance.miner_hotkey,
        f"http://{instance.host}:{instance.port}/{path}",
        enc_payload,
        timeout=15.0,
    ) as resp:
        if resp.status == 404:
            logger.warning(
                f"Failed filesystem check with 404 for {payload=}: "
                f"{instance.miner_hotkey=} {instance.instance_id=} {instance.chute_id=}"
            )
            return None
        if encrypted_slurp:
            resp_data = (await resp.json())["json"]
            decrypted = decrypt_instance_response(resp_data, instance, iv=iv)
            return base64.b64decode(json.loads(decrypted)["contents"])
        return base64.b64decode(await resp.text())


async def get_hf_content(model, revision, filename) -> tuple[str, str]:
    """
    Get the content of a specific model file from huggingface.
    """
    cache_key = f"hfdata:{model}:{revision}:{filename}"
    local_key = str(uuid.uuid5(uuid.NAMESPACE_OID, cache_key))
    cached = await settings.redis_client.get(cache_key)
    if cached and os.path.exists(f"/tmp/{local_key}"):
        with open(f"/tmp/{local_key}", "r") as infile:
            return cached.decode(), infile.read()
    url = f"https://huggingface.co/{model}/resolve/{revision}/{filename}"
    try:
        async with aiohttp.ClientSession(raise_for_status=False) as session:
            async with session.get(url) as resp:
                resp.raise_for_status()
                content = await resp.read()
                digest = hashlib.sha256(content).hexdigest()
                await settings.redis_client.set(cache_key, digest)
                with open(f"/tmp/{local_key}", "w") as outfile:
                    outfile.write(content.decode())
                return digest, content.decode()
    except Exception as exc:
        logger.error(f"Error checking HF file content: {url} exc={exc}")
    return None, None


async def check_weight_files(
    encrypted_slurp, model, revision, instances, weight_map, hard_failed, soft_failed
):
    """
    Check the individual weight files.
    """
    to_check = [
        instance
        for instance in instances
        if instance not in hard_failed and instance not in soft_failed
    ]
    if not to_check:
        return
    weight_files = set()
    for layer, path in weight_map.get("weight_map", {}).items():
        weight_files.add(path)
    if not weight_files:
        return

    # Select a single random file to check.
    path = random.choice(list(weight_files))
    file_size = 0
    size_key = f"hfsize:{model}:{revision}:{path}"
    cached = await settings.redis_client.get(size_key)
    if cached:
        file_size = int(cached.decode())
    else:
        try:
            async with aiohttp.ClientSession() as session:
                url = f"https://huggingface.co/{model}/resolve/{revision}/{path}"
                async with session.head(url) as resp:
                    content_length = resp.headers.get("x-linked-size")
                    if content_length:
                        logger.info(f"Size of {model} -> {path}: {content_length}")
                        file_size = int(content_length)
                        await settings.redis_client.set(size_key, content_length)
                    else:
                        logger.warning(f"Could not determine size of {model} -> {path}")
                        return
        except Exception as exc:
            logger.error(f"Error checking HF for {model=} {revision=} {path=}: {str(exc)}")
            return

    # Now a random offset.
    start_byte = 0
    end_byte = min(file_size, random.randint(25, 500))
    if file_size:
        check_size = min(file_size - 1, random.randint(100, 500))
        start_byte = random.randint(0, file_size - check_size)
        end_byte = start_byte + check_size
    expected_digest = None
    try:
        async with aiohttp.ClientSession() as session:
            url = f"https://huggingface.co/{model}/resolve/{revision}/{path}"
            async with session.get(
                url, headers={"Range": f"bytes={start_byte}-{end_byte - 1}"}
            ) as resp:
                content = await resp.read()
                expected_digest = hashlib.sha256(content).hexdigest()
    except Exception as exc:
        logger.error(f"Error checking HF for {model=} {revision=} and {path=}: {str(exc)}")
        return

    # Verify each instance has the same.
    logger.info(
        f"Checking {path} bytes {start_byte}:{end_byte} of model {model} revision {revision}"
    )
    digest_counts = {}
    incorrect = []
    for instance in to_check:
        nice_name = model.replace("/", "--")
        payload = {
            "path": f"/cache/hub/models--{nice_name}/snapshots/{revision}/{path}",
            "start_byte": start_byte,
            "end_byte": end_byte,
        }
        try:
            started_at = time.time()
            data = await do_slurp(instance, payload, encrypted_slurp)
            duration = time.time() - started_at
            if data is None:
                hard_failed.append(instance)
                continue
            digest = hashlib.sha256(data).hexdigest()
            if digest not in digest_counts:
                digest_counts[digest] = 0
            digest_counts[digest] += 1
            if digest != expected_digest:
                logger.warning(
                    f"Digest of {path} on {instance.instance_id=} of {model} is incorrect: {expected_digest} vs {digest}"
                )
                incorrect.append(instance)
            else:
                logger.success(
                    f"Digest of {path} on {instance.instance_id=} of {model} is correct: [{start_byte}:{end_byte}] {expected_digest} {duration=}"
                )
                if duration > 5.0:
                    logger.warning(
                        f"Duration to fetch model weight random offset exceeded expected duration: {duration=}"
                    )
                    soft_failed.append(instance)
        except Exception as exc:
            logger.warning(
                f"Unhandled exception checking {instance.instance_id}: {exc}\n{traceback.format_exc()}"
            )
            soft_failed.append(instance)
    if incorrect:
        remaining = [i for i in to_check if i not in [incorrect + soft_failed + hard_failed]]
        if not remaining:
            logger.warning("No instances would remain after purging incorrect weights!")
            return

        hotkeys = set([inst.miner_hotkey for inst in incorrect])
        if len(digest_counts) == 1 and len(hotkeys) >= 2:
            logger.warning(
                f"Huggingface digest mismatch, but all miners are in consensus: {expected_digest=} for {path} of {model}"
            )
        else:
            for inst in incorrect:
                hard_failed.append(inst)


async def check_llm_weights(chute, instances):
    """
    Check the model weights for vllm (and sglang) templated chutes.
    """
    if not instances:
        logger.warning(f"No instances to check: {chute.name}")
        return [], []
    chute_id = chute.chute_id

    # XXX disabled for now, chute name mismatch (intentional)
    if chute_id == "561e4875-254d-588f-a36f-57c9cdef8961":
        return [], []

    # Revision will need to be a requirement in the future, and at that point
    # it can be an attribute on the chute object rather than this janky regex.
    if (revision := chute.revision) is None:
        revision_match = re.search(
            r"(?:--revision |(?:^\s+|,\s*)revision=\")([a-f0-9]{40})", chute.code, re.MULTILINE
        )
        if revision_match:
            revision = revision_match.group(1)

    # Could call out to HF and list all revisions and check for any,
    # but revision for new chutes is a required parameter now...
    if not revision:
        logger.warning(f"No revision to check: {chute.name}")
        return [], []

    # Chute name exceptions, e.g. moonshot kimi k2 with 75k ctx on b200s.
    model_match = re.search(r"^\s*model_name\s*=\s*['\"]([^'\"]+)['\"]", chute.code, re.MULTILINE)
    model_name = chute.name if not model_match else model_match.group(1)

    logger.info(f"Checking {chute.chute_id=} {model_name=} for {revision=}")
    encrypted_slurp = use_encrypted_slurp(chute.chutes_version)

    # Test each instance.
    hard_failed = []
    soft_failed = []
    instances = await load_chute_instances(chute_id)
    if not instances:
        return [], []

    # First we'll check the primary config files, then we'll test the weights from the map.
    target_paths = [
        "model.safetensors.index.json",
        "config.json",
    ]
    weight_map = None
    for target_path in target_paths:
        incorrect = []
        digest_counts = {}
        expected_digest, expected_content = await get_hf_content(model_name, revision, target_path)
        if not expected_digest:
            # Could try other means later on but for now treat as "OK".
            logger.warning(
                f"Failed to check huggingface for {target_path} on {model_name} {revision=}"
            )
            continue
        if expected_content and target_path == "model.safetensors.index.json":
            weight_map = json.loads(expected_content)
        for instance in instances:
            nice_name = model_name.replace("/", "--")
            payload = {"path": f"/cache/hub/models--{nice_name}/snapshots/{revision}/{target_path}"}
            try:
                started_at = time.time()
                data = await do_slurp(instance, payload, encrypted_slurp)
                duration = time.time() - started_at
                if data is None:
                    hard_failed.append(instance)
                    continue
                digest = hashlib.sha256(data).hexdigest()
                if digest not in digest_counts:
                    digest_counts[digest] = 0
                digest_counts[digest] += 1
                if expected_digest and expected_digest != digest:
                    logger.warning(
                        f"Digest of {target_path} on {instance.instance_id=} of {model_name} "
                        f"is incorrect: {expected_digest} vs {digest}"
                    )
                    incorrect.append(instance)
                logger.info(
                    f"Digest of {target_path} on {instance.instance_id=} of {model_name}: {digest} {duration=}"
                )
                if duration > 9.0:
                    logger.warning(
                        f"Duration to fetch model weight map exceeded expected duration: {duration=}"
                    )
                    soft_failed.append(instance)
            except Exception as exc:
                logger.warning(
                    f"Unhandled exception checking {instance.instance_id}: {exc}\n{traceback.format_exc()}"
                )
                soft_failed.append(instance)
        # Just out of an abundance of caution, we don't want to deleting everything
        # if for some reason huggingface has some mismatch but all miners report
        # exactly the same thing.
        if incorrect:
            remaining = [i for i in instances if i not in [incorrect + soft_failed + hard_failed]]
            if not remaining:
                logger.warning("No instances would remain after purging incorrect weights!")
                return

            hotkeys = set([inst.miner_hotkey for inst in incorrect])
            if len(digest_counts) == 1 and len(hotkeys) >= 2:
                logger.warning(
                    f"Huggingface digest mismatch, but all miners are in consensus: {expected_digest=} for {target_path} of {model_name}"
                )
            else:
                for inst in incorrect:
                    hard_failed.append(inst)

    # Now check the actual weights.
    if weight_map:
        await check_weight_files(
            encrypted_slurp, model_name, revision, instances, weight_map, hard_failed, soft_failed
        )
    return hard_failed, soft_failed


async def check_live_code(instance, chute, encrypted_slurp) -> bool:
    """
    Check the running command.
    """
    payload = {"path": "/proc/1/cmdline"}
    data = await do_slurp(instance, payload, encrypted_slurp)
    if not data:
        logger.warning(f"Instance returned no data on proc check: {instance.instance_id}")
        return False

    # Compare to expected command.
    command_line = data.decode().replace("\x00", " ").strip()
    command_line = re.sub(r"([^ ]+/)?python3?(\.[0-9]+)?", "python", command_line.strip())
    command_line = re.sub(r"([^ ]+/)?chutes\b", "chutes", command_line)
    seed = (
        None if semcomp(chute.chutes_version or "0.0.0", "0.3.0") >= 0 else instance.nodes[0].seed
    )
    expected = get_expected_command(chute, instance.miner_hotkey, seed, tls=False)
    if command_line != expected:
        logger.error(
            f"Failed PID 1 lookup evaluation: {instance.instance_id=} {instance.miner_hotkey=}:\n\t{command_line}\n\t{expected}"
        )
        return False

    # Double check the code.
    payload = {"path": f"/app/{chute.filename}"}
    code = await do_slurp(instance, payload, encrypted_slurp)
    if code != chute.code.encode():
        logger.error(
            f"Failed code slurp evaluation: {instance.instance_id=} {instance.miner_hotkey=}:\n{code}"
        )
        return False
    logger.success(
        f"Code and proc validation success: {instance.instance_id=} {instance.miner_hotkey=}"
    )
    return True


async def check_ping(chute, instance):
    """
    Single instance ping test.
    """
    expected = str(uuid.uuid4())
    payload, iv = encrypt_instance_request(json.dumps({"foo": expected}), instance)
    path, _ = encrypt_instance_request("/_ping", instance, hex_encode=True)
    async with miner_client.post(
        instance.miner_hotkey,
        f"http://{instance.host}:{instance.port}/{path}",
        payload,
        timeout=10.0,
    ) as resp:
        raw_content = await resp.read()
        resp_data = json.loads(raw_content)
        decrypted = decrypt_instance_response(resp_data["json"], instance, iv)
        pong = json.loads(decrypted)["foo"]
        if pong != expected:
            logger.warning(f"Incorrect challenge response to ping: {pong=} vs {expected=}")
            return False
        logger.success(f"Instance {instance.instance_id=} of {chute.name} ping success: {pong=}")
        return True


async def check_pings(chute, instances) -> list:
    """
    Simple ping  test.
    """
    if not instances:
        return []
    failed = []
    for instance in instances:
        try:
            if not await check_ping(chute, instance):
                failed.append(instance)
        except Exception as exc:
            logger.warning(
                f"Unhandled ping exception on instance {instance.instance_id} of {chute.name}: {exc}"
            )
            failed.append(instance)
    return failed


async def check_commands(chute, instances) -> list:
    """
    Check the command being used to run a chute on each instance.
    """
    if not instances:
        return [], []
    encrypted = use_encrypted_slurp(chute.chutes_version)
    if not encrypted:
        logger.info(f"Unable to check command: {chute.chutes_version=} for {chute.name}")
        return [], []
    hard_failed = []
    soft_failed = []
    for instance in instances:
        try:
            if not await check_live_code(instance, chute, encrypted):
                hard_failed.append(instance)
        except Exception as exc:
            logger.warning(f"Unhandled exception checking command {instance.instance_id=}: {exc}")
            soft_failed.append(instance)
    return hard_failed, soft_failed


async def increment_soft_fail(instance, chute):
    """
    Increment soft fail counts and purge if limit is reached.
    """
    fail_key = f"watchtower:fail:{instance.instance_id}"
    fail_count = await settings.redis_client.incr(fail_key)
    await settings.redis_client.expire(fail_key, 3600)
    if fail_count and fail_count >= 3:
        logger.warning(
            f"Instance {instance.instance_id} "
            f"miner {instance.miner_hotkey} "
            f"chute {chute.name} reached max soft fails: {fail_count}"
        )
        await purge_and_notify(instance)


def get_expected_command(chute, miner_hotkey: str, seed: int = None, tls: bool = False):
    """
    Get the command line for a given instance.
    """
    # New chutes run format expects a JWT and TLS key/cert, but not graval seed.
    if semcomp(chute.chutes_version or "0.0.0", "0.3.0") >= 0:
        parts = [
            "python",
            "chutes",
            "run",
            chute.ref_str,
            "--port",
            "8000",
            "--miner-ss58",
            miner_hotkey,
            "--validator-ss58",
            settings.validator_ss58,
        ]
        if tls:
            parts += [
                "--keyfile",
                "/app/.chutetls/key.pem",
                "--certfile",
                "/app/.chutetls/cert.pem",
            ]
        return " ".join(parts).strip()

    # Legacy format.
    return " ".join(
        [
            "python",
            "chutes",
            "run",
            chute.ref_str,
            "--port",
            "8000",
            "--graval-seed",
            str(seed),
            "--miner-ss58",
            miner_hotkey,
            "--validator-ss58",
            settings.validator_ss58,
        ]
    ).strip()


async def verify_expected_command(
    dump: dict, chute: Chute, miner_hotkey: str, seed: int = None, tls: bool = False
):
    process = dump["all_processes"][0]
    assert process["pid"] == 1, "Failed to find chutes comman as PID 1"
    assert process["username"] == "chutes", "Not running as chutes user"
    command_line = re.sub(r"([^ ]+/)?python3?(\.[0-9]+)?", "python", process["cmdline"]).strip()
    command_line = re.sub(r"([^ ]+/)?chutes\b", "chutes", command_line)
    expected = get_expected_command(chute, miner_hotkey=miner_hotkey, seed=seed, tls=tls)
    assert command_line == expected, f"Unexpected command: {command_line=} vs {expected=}"
    logger.success(f"Verified command line: {miner_hotkey=} {command_line=}")


def uuid_dict(data, current_path=[], salt=settings.envcheck_52_salt):
    flat_dict = {}
    for key, value in data.items():
        new_path = current_path + [key]
        if isinstance(value, dict):
            flat_dict.update(uuid_dict(value, new_path, salt=salt))
        else:
            uuid_key = str(uuid.uuid5(uuid.NAMESPACE_OID, json.dumps(new_path).decode() + salt))
            flat_dict[uuid_key] = value
    return flat_dict


def is_kubernetes_env(
    instance: Instance, dump: dict, log_prefix: str, standard_template: str = None
):
    # Requires chutes SDK 0.2.53+
    if semcomp(instance.chutes_version or "0.0.0", "0.2.53") < 0:
        return True

    # Lib overrides.
    if standard_template:
        exclude = {
            "UV_SYSTEM_PYTHON",
            "PYTHONUNBUFFERED",
            "PYTHONIOENCODING",
            "PYTHONWARNINGS",
        }
        banned = {
            "HTTP_PROXY",
            "HTTPS_PROXY",
            "HF_HUB_DISABLE_SSL_VERIFY",
        }
        bad = [
            key
            for key in dump["env"]
            if ("python" in key.lower() and key.upper() not in exclude) or key.upper() in banned
        ]
        if bad:
            logger.warning(f"{log_prefix} Invalid environment found: PYTHON env override(s): {bad}")
            return False

    # Verify our LD_PRELOAD (netnanny and logger).
    if semcomp(instance.chutes_version or "0.0.0", "0.3.61") >= 0:
        if (
            dump["env"].get("LD_PRELOAD")
            != "/usr/local/lib/chutes-netnanny.so:/usr/local/lib/chutes-logintercept.so"
        ):
            logger.warning(f"{log_prefix} Invalid environment found: LD_PRELOAD tampering")
            return False

    if not dump.get("k8s_info", {}).get("has_service_account"):
        logger.warning(
            f"{log_prefix} Invalid environment found: k8s (supposed) pod does not have valid service account"
        )
        return False

    if isinstance(dump.get("mounts"), dict):
        for mount in dump.get("mounts", {}).get("filesystems", []):
            if (
                "chutesfs.index" in mount["target"]
                or "sglang" in mount["target"]
                or "site-packages" in mount["target"]
                or re.search(r"^/app\/.*_src", mount["target"])
                or "dist-packages" in mount["target"]
            ):
                logger.warning(
                    f"{log_prefix} Invalid environment found: contains source code or chutesfs index mount"
                )
                return False

    logger.success(f"{log_prefix} kubernetes check passed")
    return True


async def check_sglang(instance_id: str, chute: Chute, dump: dict, log_prefix: str):
    if "build_sglang_chute(" not in chute.code or chute.standard_template != "vllm":
        return True

    processes = dump["all_processes"]

    # Extract the revision, if present.
    if (revision := chute.revision) is None:
        revision_match = re.search(
            r"(?:--revision |(?:^\s+|,\s*)revision=\")([a-f0-9]{40})", chute.code, re.MULTILINE
        )
        if revision_match:
            revision = revision_match.group(1)

    # Chute name exceptions, e.g. moonshot kimi k2 with 75k ctx on b200s.
    model_match = re.search(r"^\s*model_name\s*=\s*['\"]([^'\"]+)['\"]", chute.code, re.MULTILINE)
    model_name = chute.name if not model_match else model_match.group(1)

    found_sglang = False
    sglang_process = None
    for process in processes:
        target_exe = process["exe"] if process["exe"].strip() else process["cmdline"].split(" ")[0]
        clean_exe = re.sub(r"([^ ]+/)?python3?(\.[0-9]+)?", "python", target_exe)
        cmdline = re.sub(r"([^ ]+/)?python3?(\.[0-9]+)?", "python", process["cmdline"])
        if (
            clean_exe in ["python", "python3.10", "python3.11", "python3.12"]
            and process["username"] == "chutes"
            and cmdline.startswith(
                f"python -m sglang.launch_server --host 127.0.0.1 --port 10101 --model-path {model_name}"
            )
        ):
            if semcomp(chute.chutes_version or "0.0.0", "0.3.48") >= 0:
                if "--enable-cache-report" not in cmdline:
                    logger.warning(f"Cache report not enabled: {cmdline}")
                elif "--enable-return-hidden-states" not in cmdline:
                    logger.warning(f"Hidden states return not enabled: {cmdline}")
                else:
                    found_sglang = True
            else:
                found_sglang = True
            if revision and revision not in cmdline:
                found_sglang = False
                logger.warning(f"Did not find model revision in SGLang command: {cmdline}")
            if found_sglang:
                logger.success(f"{log_prefix} found valid SGLang chute: {process=}")
                sglang_process = process
                break

    if not found_sglang:
        logger.error(f"{log_prefix} did not find SGLang process, bad: {processes=}")
        return False

    # Track the process.
    current_pid = sglang_process["pid"]
    pid_key = f"sglangpid:{instance_id}"
    cached = await settings.redis_client.get(pid_key)
    if cached:
        previous_pid = int(cached)
        if previous_pid != current_pid:
            logger.error(
                f"{log_prefix} primary SGLang PID has changed from {previous_pid=} to {current_pid=}"
            )
            return False
    else:
        await settings.redis_client.set(pid_key, f"{current_pid}")

    return True


async def check_chute(chute_id):
    """
    Check a single chute.
    """
    async with get_session() as session:
        chute = (
            (await session.execute(select(Chute).where(Chute.chute_id == chute_id)))
            .unique()
            .scalar_one_or_none()
        )
        if not chute:
            logger.warning(f"Chute not found: {chute_id=}")
            return
        if chute.rolling_update:
            logger.warning(f"Chute has a rolling update in progress: {chute_id=}")
            return

    # Updated environment/code checks.
    instances = await load_chute_instances(chute.chute_id)
    random.shuffle(instances)
    bad_env = set()
    if semcomp(chute.chutes_version or "0.0.0", "0.2.53") >= 0 and os.getenv("ENVDUMP_UNLOCK"):
        instance_map = {instance.instance_id: instance for instance in instances}

        # Load the envdump dump outputs for each.
        missing = set(instance_map)
        async with get_dumps(instances) as paths:
            for path in paths:
                failed_envdump = False
                if not path:
                    failed_envdump = True
                    continue
                instance_id = path.split("dump-")[-1].split(".")[0]
                missing.discard(instance_id)
                instance = instance_map[instance_id]
                log_prefix = f"ENVDUMP: {instance.instance_id=} {instance.miner_hotkey=} {instance.chute_id=}"
                with open(path) as infile:
                    dump = json.loads(infile.read())

                # Ensure proper k8s env.
                if not is_kubernetes_env(instance, dump, log_prefix):
                    logger.error(f"{log_prefix} is not running a valid kubernetes environment")
                    failed_envdump = True

                # Check SGLang processes.
                if not await check_sglang(instance.instance_id, chute, dump, log_prefix):
                    logger.error(f"{log_prefix} did not find SGLang process, bad...")
                    failed_envdump = True

                # Check the running command.
                try:
                    await verify_expected_command(
                        dump,
                        chute,
                        miner_hotkey=instance.miner_hotkey,
                        seed=instance.nodes[0].seed,
                        tls=False,
                    )
                except AssertionError as exc:
                    logger.error(f"{log_prefix} failed running command check: {exc=}")
                    failed_envdump = True
                except Exception as exc:
                    logger.error(
                        f"{log_prefix} unhandled exception checking env dump: {exc=}\n{traceback.format_exc()}"
                    )
                    failed_envdump = True

                # Delete failed checks.
                if failed_envdump:
                    await purge_and_notify(
                        instance, reason="Instance failed env dump signature or process checks."
                    )
                    bad_env.add(instance.instance_id)
                    failed_count = await settings.redis_client.incr(
                        f"envdumpfail:{instance.miner_hotkey}"
                    )
                    logger.warning(
                        f"ENVDUMP: Miner {instance.miner_hotkey} has now failed {failed_count} envdump checks"
                    )
                    # if failed_count >= 5:
                    #    async with get_session() as session:
                    #        await session.execute(
                    #            text("""
                    #            UPDATE metagraph_nodes
                    #            SET blacklist_reason = 'Recurring pattern of invalid processes discovered by watchtower.'
                    #            WHERE hotkey = :hotkey
                    #            """),
                    #            {"hotkey": instance.miner_hotkey}
                    #        )

    # Filter out the ones we already blacklisted.
    instances = [instance for instance in instances if instance.instance_id not in bad_env]

    # Ping test.
    soft_failed = await check_pings(chute, instances)

    # Check the running command.
    instances = [instance for instance in instances if instance not in soft_failed]
    hard_failed, _soft_failed = await check_commands(chute, instances)
    soft_failed += _soft_failed

    # Check model weights.
    if chute.standard_template == "vllm":
        instances = [
            instance
            for instance in instances
            if instance not in soft_failed and instance not in hard_failed
        ]
        _hard_failed, _soft_failed = await check_llm_weights(chute, instances)
        hard_failed += _hard_failed
        soft_failed += _soft_failed

    # Hard failures get terminated immediately.
    for instance in hard_failed:
        if not instance:
            continue
        logger.warning(
            f"Purging instance {instance.instance_id} "
            f"miner {instance.miner_hotkey} "
            f"chute {chute.name} due to hard fail"
        )
        await purge_and_notify(instance)

    # Limit "soft" fails to max consecutive failures, allowing some downtime but not much.
    for instance in soft_failed:
        if not instance:
            continue
        await increment_soft_fail(instance, chute)

    # Update verification time for the ones that succeeded.
    to_update = [
        instance
        for instance in instances
        if instance not in soft_failed and instance not in hard_failed
    ]
    if to_update:
        async with get_session() as session:
            stmt = (
                update(Instance)
                .where(Instance.instance_id.in_([i.instance_id for i in to_update]))
                .values(last_verified_at=func.now())
                .execution_options(synchronize_session=False)
            )
            await session.execute(stmt)
            await session.commit()


async def check_all_chutes():
    """
    Check all chutes and instances, one time.
    """
    started_at = int(time.time())
    async with get_session() as session:
        chute_ids = (await session.execute(select(Chute.chute_id))).unique().scalars().all()
    if chute_ids and isinstance(chute_ids[0], tuple):
        chute_ids = [chute_id[0] for chute_id in chute_ids]
    chute_ids = list(sorted(chute_ids))
    for i in range(0, len(chute_ids), 8):
        batch = chute_ids[i : i + 8]
        logger.info(f"Initializing check of chutes: {batch}")
        await asyncio.gather(*[check_chute(chute_id) for chute_id in batch])
    delta = int(time.time()) - started_at
    logger.info(f"Finished probing all instances of {len(chute_ids)} chutes in {delta} seconds.")


async def generate_confirmed_reports(chute_id, reason):
    """
    When a chute is confirmed bad, generate reports for it.
    """
    from api.user.service import chutes_user_id

    async with get_session() as session:
        report_query = text("""
        WITH inserted AS (
            INSERT INTO reports
            (invocation_id, user_id, timestamp, confirmed_at, confirmed_by, reason)
            SELECT
                parent_invocation_id,
                :user_id,
                now(),
                now(),
                :confirmed_by,
                :reason
            FROM invocations i
            WHERE chute_id = :chute_id
            AND NOT EXISTS (
                SELECT 1 FROM reports r
                WHERE r.invocation_id = i.parent_invocation_id
            )
            ON CONFLICT (invocation_id) DO NOTHING
            RETURNING invocation_id
        )
        SELECT COUNT(*) AS report_count FROM inserted;
        """)
        count = (
            await session.execute(
                report_query,
                {
                    "user_id": await chutes_user_id(),
                    "confirmed_by": await chutes_user_id(),
                    "chute_id": chute_id,
                    "reason": reason,
                },
            )
        ).scalar()
        logger.success(f"Generated {count} reports for chute {chute_id}")
        await session.commit()


async def report_short_lived_chutes():
    """
    Generate reports for chutes that only existed for a short time, likely from scummy miners to get bounties.
    """
    query = text(SHORT_LIVED_CHUTES)
    bad_chutes = []
    async with get_session() as session:
        result = await session.execute(query)
        rows = result.fetchall()
        for row in rows:
            chute_id = row.chute_id
            lifetime = row.lifetime
            bad_chutes.append(
                (chute_id, f"chute was very short lived: {lifetime=}, likely bounty scam")
            )
            logger.warning(
                f"Detected short-lived chute {chute_id} likely part of bounty scam: {lifetime=}"
            )

    # Generate the reports in separate sessions so we don't have massive transactions.
    for chute_id, reason in bad_chutes:
        await generate_confirmed_reports(chute_id, reason)


async def remove_bad_chutes():
    """
    Remove malicious/bad chutes via AI analysis of code.
    """
    from api.user.service import chutes_user_id

    async with get_session() as session:
        chutes = (
            (await session.execute(select(Chute).where(Chute.user_id != await chutes_user_id())))
            .unique()
            .scalars()
            .all()
        )
    tasks = [is_bad_code(chute.code) for chute in chutes]
    results = await asyncio.gather(*tasks)
    for idx in range(len(chutes)):
        chute = chutes[idx]
        bad, reason = results[idx]
        if bad:
            logger.error(
                "\n".join(
                    [
                        f"Chute contains problematic code: {chute.chute_id=} {chute.name=} {chute.user_id=}",
                        json.dumps(reason).decode(),
                        "Code:",
                        chute.code,
                    ]
                )
            )
            # Delete it automatically.
            async with get_session() as session:
                chute = (
                    (await session.execute(select(Chute).where(Chute.chute_id == chute.chute_id)))
                    .unique()
                    .scalar_one_or_none()
                )
                version = chute.version
                await session.delete(chute)
                await settings.redis_client.publish(
                    "miner_broadcast",
                    json.dumps(
                        {
                            "reason": "chute_deleted",
                            "data": {"chute_id": chute.chute_id, "version": version},
                        }
                    ).decode(),
                )
                await session.commit()
            reason = f"Chute contains code identified by DeepSeek-R1 as likely cheating: {json.dumps(reason).decode()}"
            await generate_confirmed_reports(chute.chute_id, reason)
        else:
            logger.success(f"Chute seems fine: {chute.chute_id=} {chute.name=}")


async def procs_check():
    """
    Check processes.
    """
    while True:
        async with get_session() as session:
            query = (
                select(Instance)
                .where(
                    Instance.verified.is_(True),
                    Instance.active.is_(True),
                )
                .options(selectinload(Instance.nodes), selectinload(Instance.chute))
            )
            batch_size = 10
            async for row in await session.stream(query.execution_options(yield_per=batch_size)):
                instance = row[0]
                if not instance.chutes_version or not re.match(
                    r"^0\.2\.[3-9][0-9]$", instance.chutes_version
                ):
                    continue
                skip_key = f"procskip:{instance.instance_id}"
                if await settings.redis_client.get(skip_key):
                    await settings.redis_client.expire(skip_key, 60 * 60 * 24 * 2)
                    continue
                path, _ = encrypt_instance_request("/_procs", instance, hex_encode=True)
                try:
                    async with miner_client.get(
                        instance.miner_hotkey,
                        f"http://{instance.host}:{instance.port}/{path}",
                        purpose="chutes",
                        timeout=15.0,
                    ) as resp:
                        data = await resp.json()
                        env = data.get("1", {}).get("environ", {})
                        cmdline = data.get("1", {}).get("cmdline", [])
                        reason = None
                        if not cmdline and (not env or "CHUTES_EXECUTION_CONTEXT" not in env):
                            reason = f"Running an invalid process [{instance.instance_id=} {instance.miner_hotkey=}]: {cmdline=} {env=}"
                        elif len(cmdline) <= 5 or cmdline[1].split("/")[-1] != "chutes":
                            reason = f"Running an invalid process [{instance.instance_id=} {instance.miner_hotkey=}]: {cmdline=} {env=}"
                        if reason:
                            logger.warning(reason)
                            await purge_and_notify(
                                instance, reason="miner failed watchtower probes"
                            )
                        else:
                            logger.success(
                                f"Passed proc check: {instance.instance_id=} {instance.chute_id=} {instance.miner_hotkey=}"
                            )
                            await settings.redis_client.set(skip_key, "1", ex=60 * 60 * 24 * 2)
                except Exception as exc:
                    logger.warning(
                        f"Couldn't check procs for {instance.miner_hotkey=} {instance.instance_id=}, must be bad? {exc}\n{traceback.format_exc()}"
                    )
        logger.info("Finished proc check loop...")
        await asyncio.sleep(10)


async def get_env_dump(instance):
    """
    Load the environment dump from remote instance.
    """
    key = secrets.token_bytes(16)
    payload = {"key": key.hex()}
    enc_payload, _ = encrypt_instance_request(json.dumps(payload), instance)
    path, _ = encrypt_instance_request("/_env_dump", instance, hex_encode=True)
    async with miner_client.post(
        instance.miner_hotkey,
        f"http://{instance.host}:{instance.port}/{path}",
        enc_payload,
        timeout=30.0,
    ) as resp:
        if resp.status != 200:
            raise EnvdumpMissing(
                f"Received invalid response code on /_env_dump: {instance.instance_id=} {resp.status=} {await resp.text()}"
            )
        return json.loads(decrypt_envdump_cipher(await resp.text(), key, instance.chutes_version))


async def get_env_sig(instance, salt):
    """
    Load the environment signature from the remote instance.
    """
    payload = {"salt": salt}
    enc_payload, _ = encrypt_instance_request(json.dumps(payload), instance)
    path, _ = encrypt_instance_request("/_env_sig", instance, hex_encode=True)
    async with miner_client.post(
        instance.miner_hotkey,
        f"http://{instance.host}:{instance.port}/{path}",
        enc_payload,
        timeout=5.0,
    ) as resp:
        if resp.status != 200:
            raise EnvdumpMissing(
                f"Received invalid response code on /_env_sig: {instance.instance_id=} {resp.status=} {await resp.text()}"
            )
        return await resp.text()


async def get_dump(instance, outdir: str = None):
    """
    Load the (new) environment dump from the remote instance.
    """
    from chutes.envdump import DUMPER

    key = secrets.token_bytes(16).hex()
    payload = {"key": key}
    enc_payload, _ = encrypt_instance_request(json.dumps(payload), instance)
    path, _ = encrypt_instance_request("/_dump", instance, hex_encode=True)
    logger.info(f"Querying {instance.instance_id=} envdump (dump)")
    try:
        async with miner_client.post(
            instance.miner_hotkey,
            f"http://{instance.host}:{instance.port}/{path}",
            enc_payload,
            timeout=400.0,
        ) as resp:
            if resp.status != 200:
                err = f"Received invalid response code on /_dump: {resp.status=}"
                if outdir:
                    logger.error(f"ENVDUMP: {err} {instance.miner_hotkey=} {instance.instance_id=}")
                    return None
                else:
                    raise EnvdumpMissing(err)
            try:
                body = await resp.json()
                result = DUMPER.decrypt(key, body["result"])
                if outdir:
                    outpath = os.path.join(outdir, f"dump-{instance.instance_id}.json")
                    with open(outpath, "w") as outfile:
                        bytes_ = outfile.write(json.dumps(result).decode())
                    logger.success(f"Saved {bytes_} byte JSON dump to {outpath}")
                    return outpath
                return result
            except Exception as exc:
                err = f"Failed to load and decrypt _dump payload: {exc=}"
                if outdir:
                    logger.error(f"ENVDUMP: {err} {instance.miner_hotkey=} {instance.instance_id=}")
                    return None
                else:
                    raise EnvdumpMissing(err)
    except Exception as exc:
        err = f"Failed to fetch _dump: {exc=}"
        if outdir:
            logger.error(f"ENVDUMP: {err} {instance.miner_hotkey=} {instance.instance_id=}")
            return None
        else:
            raise EnvdumpMissing(err)


@asynccontextmanager
async def get_dumps(instances: list[Instance], concurrency: int = 32):
    """
    Get (new) environment dumps from all instances, controlling concurrency.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        semaphore = asyncio.Semaphore(concurrency)

        async def _dump(instance):
            async with semaphore:
                return await get_dump(instance, tmpdir)

        tasks = [_dump(instance) for instance in instances]
        results = await asyncio.gather(*tasks)
        yield results


async def get_sig(instance):
    """
    Load the (new) environment signature from remote instance.
    """
    salt = secrets.token_bytes(16).hex()
    payload = {"salt": salt}
    enc_payload, _ = encrypt_instance_request(json.dumps(payload), instance)
    path, _ = encrypt_instance_request("/_sig", instance, hex_encode=True)
    logger.info(f"Querying {instance.instance_id=} envdump (sig)")
    async with miner_client.post(
        instance.miner_hotkey,
        f"http://{instance.host}:{instance.port}/{path}",
        enc_payload,
        timeout=15.0,
    ) as resp:
        if resp.status != 200:
            raise EnvdumpMissing(f"Received invalid response code on /_dump: {resp.status=}")
        try:
            body = await resp.json()
            return body["result"]
        except Exception as exc:
            raise EnvdumpMissing(f"Failed to load and decrypt _dump payload: {exc=}")


async def slurp(instance, path, offset: int = 0, length: int = 0):
    """
    Load contents of a remote file/dir via (new) envdump lib.
    """
    from chutes.envdump import DUMPER

    key = secrets.token_bytes(16).hex()
    payload = {"key": key, "path": path, "offset": offset, "length": length}
    enc_payload, _ = encrypt_instance_request(json.dumps(payload), instance)
    path, _ = encrypt_instance_request("/_eslurp", instance, hex_encode=True)
    logger.info(f"Querying {instance.instance_id=} envdump (slurp) {payload=}")
    async with miner_client.post(
        instance.miner_hotkey,
        f"http://{instance.host}:{instance.port}/{path}",
        enc_payload,
        timeout=30.0,
    ) as resp:
        if resp.status != 200:
            raise EnvdumpMissing(f"Received invalid response code on /_eslurp: {resp.status=}")
        body = None
        try:
            body = await resp.json()
        except Exception as exc:
            raise EnvdumpMissing(
                f"Failed to load and decrypt _eslurp payload, invalid response JSON: {str(exc)}"
            )
        if not body.get("result"):
            raise EnvdumpMissing(
                f"Failed to load and decrypt _eslurp payload, result object was null: {body=}"
            )
        try:
            return DUMPER.decrypt(key, body["result"])
        except Exception as exc:
            raise EnvdumpMissing(f"Failed to load and decrypt _eslurp payload: {exc=}")


def parse_proc_net_tcp(raw_content_ipv4, raw_content_ipv6):
    connections = []
    try:
        for line in raw_content_ipv4.splitlines()[1:]:
            fields = line.split()
            if len(fields) < 4:
                continue
            local_ip, local_port = fields[1].split(":")
            remote_ip, remote_port = fields[2].split(":")
            connections.append(
                {
                    "local": f"{hex_to_ipv4(local_ip)}:{int(local_port, 16)}",
                    "remote": f"{hex_to_ipv4(remote_ip)}:{int(remote_port, 16)}",
                    "state": TCP_STATES.get(fields[3], fields[3]),
                }
            )
    except FileNotFoundError:
        pass

    try:
        for line in raw_content_ipv6.splitlines()[1:]:
            fields = line.split()
            if len(fields) < 4:
                continue
            local_ip, local_port = fields[1].split(":")
            remote_ip, remote_port = fields[2].split(":")
            connections.append(
                {
                    "local": f"{hex_to_ipv6(local_ip)}:{int(local_port, 16)}",
                    "remote": f"{hex_to_ipv6(remote_ip)}:{int(remote_port, 16)}",
                    "state": TCP_STATES.get(fields[3], fields[3]),
                }
            )
    except FileNotFoundError:
        pass

    connections.sort(key=lambda x: (x["state"] != "LISTEN", x["state"], x["local"]))
    return connections


def hex_to_ipv4(hex_ip):
    return socket.inet_ntoa(struct.pack("<I", int(hex_ip, 16)))


def hex_to_ipv6(hex_ip):
    parts = []
    for i in range(0, 32, 8):
        part = hex_ip[i : i + 8]
        reversed_part = "".join([part[j : j + 2] for j in range(6, -1, -2)])
        parts.append(reversed_part)
    return socket.inet_ntop(socket.AF_INET6, bytes.fromhex("".join(parts)))


def find_suspicious_outbound(connections):
    SERVER_PORTS = {8000, 8001, 10101}

    def is_local_ip(ip):
        return (
            ip.startswith("127.")
            or ip.startswith("10.")
            or ip.startswith("172.")
            or ip.startswith("192.168.")
            or ip == "::1"
            or ip.startswith("::ffff:127.")
            or ip.startswith("::ffff:10.")
            or ip.startswith("::ffff:172.")
            or ip.startswith("::ffff:192.168.")
        )

    for conn in connections:
        # Fix 1: Check for 'ESTABLISHED' not '01'
        if conn["state"] != "ESTABLISHED":
            continue

        # Fix 2: Parse the local/remote strings to extract IP and port
        local_ip, local_port = conn["local"].rsplit(":", 1)
        remote_ip, remote_port = conn["remote"].rsplit(":", 1)

        if int(local_port) in SERVER_PORTS:
            continue

        if is_local_ip(remote_ip):
            continue

        logger.warning(f"SUSPICIOUS: {conn=}")
        return True

    logger.success("No suspicious outbound connections.")
    return False


async def check_instance_connections(instance):
    """
    Check if the instance has any outbound connections.
    """
    if not instance.active:
        return
    try:
        payload = {"path": "/proc/net/tcp"}
        raw_tcp4 = (await do_slurp(instance, payload, True)).decode()
        payload["path"] = "/proc/net/tcp6"
        raw_tcp6 = (await do_slurp(instance, payload, True)).decode()
        connections = parse_proc_net_tcp(raw_tcp4, raw_tcp6)
        logger.info(f"{'State':<15} {'Local Address':<45} {'Remote Address':<45}")
        logger.info("-" * 105)
        for conn in connections:
            logger.info(f"{conn['state']:<15} {conn['local']:<45} {conn['remote']:<45}")
        logger.info(f"\nTotal: {len(connections)} connections")
        return find_suspicious_outbound(connections)
    except Exception as exc:
        logger.error(
            f"Unexpected error checking {instance.instance_id=} {instance.miner_hotkey=}: {str(exc)}"
        )
    return False


async def get_expected_fs_hash(chute_id: str, seed: str):
    async with get_session() as session:
        image_id = (
            (await session.execute(select(Chute.image_id).where(Chute.chute_id == chute_id)))
            .unique()
            .scalar_one_or_none()
        )
        filename = (
            (await session.execute(select(Chute.filename).where(Chute.chute_id == chute_id)))
            .unique()
            .scalar_one_or_none()
        )
        patch_version = (
            (await session.execute(select(Image.patch_version).where(Image.image_id == image_id)))
            .unique()
            .scalar_one_or_none()
        )
    from api.graval_worker import generate_fs_hash

    expected_hash = await generate_fs_hash(
        image_id, patch_version, seed=seed, sparse=False, exclude_path=f"/app/{filename}"
    )
    return expected_hash


async def verify_fs_hash(instance):
    if semcomp(instance.chutes_version or "0.0.0", "0.4.0") < 0:
        logger.warning(
            f"Unable to check FS hash, legacy chutes lib version: {instance.instance_id=} {instance.chutes_version=}"
        )
        return True

    seed = secrets.token_bytes(16).hex()
    enc_payload, _ = encrypt_instance_request(json.dumps({"salt": seed, "mode": "full"}), instance)
    path, _ = encrypt_instance_request("/_fs_hash", instance, hex_encode=True)
    try:
        async with miner_client.post(
            instance.miner_hotkey,
            f"http://{instance.host}:{instance.port}/{path}",
            enc_payload,
            timeout=90.0,
        ) as resp:
            fs_hash = (await resp.json())["result"]
            expected = await get_expected_fs_hash(instance.chute_id, seed)
            if fs_hash != expected:
                logger.warning(
                    f"Instance FS hash mismatch! {instance.instance_id=} {instance.miner_hotkey=}, {expected=} {fs_hash=}"
                )
                return False
    except Exception as exc:
        logger.error(
            f"Failed to compare FS hashes for {instance.instance_id=} {instance.miner_hotkey=}: {str(exc)}\n{traceback.format_exc()}"
        )
        return False
    return True


async def check_runint(instance: Instance) -> bool:
    """Verify runtime integrity of an instance via the /_rint endpoint."""
    if semcomp(instance.chutes_version or "0.0.0", "0.5.0") < 0:
        return True

    if not instance.rint_commitment or not instance.rint_nonce:
        logger.warning(
            f"RUNINT: {instance.instance_id=} {instance.miner_hotkey=} missing commitment/nonce"
        )
        return False

    try:
        from ecdsa import VerifyingKey, SECP256k1, BadSignatureError

        challenge = secrets.token_hex(16)
        payload = {"challenge": challenge}
        enc_payload, _ = encrypt_instance_request(json.dumps(payload), instance)
        path, _ = encrypt_instance_request("/_rint", instance, hex_encode=True)

        async with miner_client.post(
            instance.miner_hotkey,
            f"http://{instance.host}:{instance.port}/{path}",
            enc_payload,
            timeout=15.0,
        ) as resp:
            if resp.status != 200:
                logger.error(
                    f"RUNINT: {instance.instance_id=} {instance.miner_hotkey=} "
                    f"returned {resp.status}"
                )
                return False

            body = await resp.json()
            if "error" in body:
                logger.error(
                    f"RUNINT: {instance.instance_id=} {instance.miner_hotkey=} "
                    f"error: {body['error']}"
                )
                return False

            signature_hex = body.get("signature")
            epoch = body.get("epoch")

            if not signature_hex or epoch is None:
                logger.error(
                    f"RUNINT: {instance.instance_id=} {instance.miner_hotkey=} "
                    f"missing signature or epoch"
                )
                return False

            commitment_bytes = bytes.fromhex(instance.rint_commitment)
            if len(commitment_bytes) != 162 or commitment_bytes[0] != 0x03:
                logger.error(
                    f"RUNINT: {instance.instance_id=} {instance.miner_hotkey=} "
                    f"invalid commitment format: len={len(commitment_bytes)} prefix={commitment_bytes[0] if commitment_bytes else None}"
                )
                return False
            if commitment_bytes[1] != 0x03:
                logger.error(
                    f"RUNINT: {instance.instance_id=} {instance.miner_hotkey=} "
                    f"invalid commitment version: {commitment_bytes[1]}"
                )
                return False
            pubkey_bytes = commitment_bytes[2:66]
            vk = VerifyingKey.from_string(pubkey_bytes, curve=SECP256k1)

            # Message format: challenge || epoch (8 bytes LE)
            epoch_bytes = epoch.to_bytes(8, byteorder="little")
            msg = challenge.encode() + epoch_bytes
            msg_hash = hashlib.sha256(msg).digest()
            sig_bytes = bytes.fromhex(signature_hex)

            try:
                vk.verify_digest(sig_bytes, msg_hash)
            except BadSignatureError:
                logger.error(
                    f"RUNINT: {instance.instance_id=} {instance.miner_hotkey=} "
                    f"signature verification failed"
                )
                return False

            # Check epoch is advancing (detect replay attacks)
            epoch_key = f"rint_epoch:{instance.instance_id}"
            last_epoch = await settings.redis_client.get(epoch_key)
            if last_epoch is not None:
                last_epoch = int(last_epoch)
                if epoch < last_epoch:
                    logger.error(
                        f"RUNINT: {instance.instance_id=} {instance.miner_hotkey=} "
                        f"epoch went backwards: {epoch} < {last_epoch}"
                    )
                    return False
            await settings.redis_client.set(epoch_key, str(epoch), ex=86400)

            logger.success(
                f"RUNINT: {instance.instance_id=} {instance.miner_hotkey=} "
                f"verification successful {epoch=}"
            )
            return True

    except Exception as e:
        logger.error(
            f"RUNINT: {instance.instance_id=} {instance.miner_hotkey=} "
            f"error: {e}\n{traceback.format_exc()}"
        )
        return False


async def main():
    """
    Main loop, continuously check all chutes and instances.
    """

    # Secondary process check.
    asyncio.create_task(procs_check())

    while True:
        await check_all_chutes()
        await asyncio.sleep(30)


if __name__ == "__main__":
    gc.set_threshold(5000, 50, 50)
    asyncio.run(main())
