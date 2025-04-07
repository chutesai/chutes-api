import re
import os
import uuid
import time
import base64
import asyncio
import aiohttp
import random
import hashlib
import json
import traceback
from loguru import logger
from datetime import timedelta, datetime
from api.config import settings
from api.util import aes_encrypt, aes_decrypt
from api.database import get_session
from api.chute.schemas import Chute
from sqlalchemy import text, update, func, select
from sqlalchemy.orm import joinedload
import api.database.orms  # noqa
import api.miner_client as miner_client
from api.util import use_encryption_v2, use_encrypted_path
from api.instance.schemas import Instance
from api.chute.codecheck import is_bad_code
from api.invocation.util import generate_invocation_history_metrics


PAST_DAY_METRICS_QUERY = """
UPDATE chutes
SET invocation_count = (
    SELECT COUNT(distinct(parent_invocation_id))
    FROM invocations
    WHERE invocations.chute_id = chutes.chute_id and started_at >= now() - interval '1 days'
);
"""

UNDEPLOYABLE_CHUTE_QUERY = """
SELECT * FROM (
    WITH chute_stats AS (
        SELECT
            chute_id,
            count(distinct(parent_invocation_id)) as invocation_count,
            count(distinct(miner_hotkey)) as successful_miner_count
        FROM
            invocations i
        WHERE
            started_at >= now() - interval '7 days'
            AND error_message IS NULL
            AND completed_at IS NOT NULL
            AND chute_user_id != 'dff3e6bb-3a6b-5a2b-9c48-da3abcd5ca5f'
            AND NOT EXISTS(
                SELECT 1 FROM reports r
                WHERE r.invocation_id = i.parent_invocation_id
            )
        GROUP BY
            chute_id
        HAVING
            COUNT(DISTINCT(miner_hotkey)) <= 2
    ),
    audit_stats AS (
        SELECT
            cs.chute_id,
            cs.invocation_count,
            cs.successful_miner_count,
            COUNT(DISTINCT ia.miner_hotkey) AS audit_miner_count,
            MIN(ia.verified_at) AS first_verified_at
        FROM
            chute_stats cs
        LEFT JOIN
            instance_audit ia ON cs.chute_id = ia.chute_id
        GROUP BY
            cs.chute_id, cs.invocation_count, cs.successful_miner_count
    )
    SELECT * FROM audit_stats
    WHERE invocation_count > 10
    AND (
        (successful_miner_count = 1 AND audit_miner_count >= 3)
        OR
        (successful_miner_count::float / audit_miner_count::float <= 0.1)
    )
    AND first_verified_at <= now() - interval '1 hour'
    ORDER BY
        audit_miner_count ASC
) t;
"""

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

# Disproportionate invocations on new chutes.
DISPROPORTIONATE_CHUTES = """
WITH new_chutes AS (
    SELECT chute_id
    FROM chutes
    WHERE created_at >= NOW() - INTERVAL '2 days'
      AND created_at <= NOW() - INTERVAL '6 hours'
),
stats AS (
    SELECT
        i.chute_id,
        i.miner_hotkey,
        COUNT(*) AS total_count
    FROM invocations i
    INNER JOIN new_chutes nc ON i.chute_id = nc.chute_id
    WHERE i.started_at >= NOW() - INTERVAL '1 day'
    AND i.completed_at IS NOT NULL
    AND i.error_message IS NULL
    GROUP BY i.chute_id, i.miner_hotkey
),
chute_totals AS (
    SELECT
        chute_id,
        SUM(total_count) AS total_invocations_per_chute,
        COUNT(DISTINCT miner_hotkey) AS unique_hotkeys
    FROM stats
    GROUP BY chute_id
),
chute_ratios AS (
    SELECT
        s.chute_id,
        s.miner_hotkey,
        s.total_count,
        c.total_invocations_per_chute,
        c.unique_hotkeys,
        CASE
            WHEN c.total_invocations_per_chute = 0 THEN 0
            ELSE s.total_count::FLOAT / c.total_invocations_per_chute
        END AS invocation_ratio
    FROM stats s
    INNER JOIN chute_totals c ON s.chute_id = c.chute_id
    WHERE s.total_count > 10
)
SELECT
    cr.chute_id,
    cr.miner_hotkey,
    cr.total_count,
    cr.total_invocations_per_chute,
    cr.unique_hotkeys,
    cr.invocation_ratio
FROM chute_ratios cr
WHERE cr.total_count >= 100
AND cr.invocation_ratio >= 0.7
ORDER BY cr.invocation_ratio DESC;
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
            .where(
                Instance.chute_id == chute_id,
                Instance.active.is_(True),
                Instance.verified.is_(True),
            )
            .options(joinedload(Instance.nodes))
        )
        instances = (await session.execute(query)).unique().scalars().all()
        return instances


async def purge_and_notify(target, reason="miner failed watchtower probes"):
    """
    Purge an instance and send a notification with the reason.
    """
    async with get_session() as session:
        await session.execute(
            text("DELETE FROM instances WHERE instance_id = :instance_id"),
            {"instance_id": target.instance_id},
        )
        await session.execute(
            text(
                "UPDATE instance_audit SET deletion_reason = :reason WHERE instance_id = :instance_id"
            ),
            {"instance_id": target.instance_id, "reason": reason},
        )
        await session.commit()
        event_data = {
            "reason": "instance_deleted",
            "message": f"Instance {target.instance_id} of miner {target.miner_hotkey} failed watchtower checks.",
            "data": {
                "chute_id": target.chute_id,
                "instance_id": target.instance_id,
                "miner_hotkey": target.miner_hotkey,
            },
        }
        await settings.redis_client.publish("events", json.dumps(event_data))
        event_data["filter_recipients"] = [target.miner_hotkey]
        await settings.redis_client.publish("miner_broadcast", json.dumps(event_data))


async def do_slurp(instance, payload, encrypted_slurp):
    """
    Slurp a remote file.
    """
    enc_payload = aes_encrypt(json.dumps(payload).encode(), instance.symmetric_key)
    iv = enc_payload[:32]
    path = aes_encrypt("/_slurp", instance.symmetric_key, hex_encode=True)
    async with miner_client.post(
        instance.miner_hotkey,
        f"http://{instance.host}:{instance.port}/{path}",
        enc_payload,
        timeout=15.0,
    ) as resp:
        if resp.status == 404:
            logger.warning(
                f"Failed filesystem check: {path}: {instance.miner_hotkey=} {instance.instance_id=} {instance.chute_id=}"
            )
            return None
        if encrypted_slurp:
            return base64.b64decode(
                json.loads(aes_decrypt((await resp.json())["json"], instance.symmetric_key, iv=iv))[
                    "contents"
                ]
            )
        return base64.b64decode(await resp.text())


async def get_hf_content(model, revision, filename) -> tuple[str, str]:
    """
    Get the content of a specific model file from huggingface.
    """
    cache_key = f"hfdata:{model}:{revision}:{filename}".encode()
    local_key = str(uuid.uuid5(uuid.NAMESPACE_OID, cache_key))
    cached = await settings.memcache.get(cache_key)
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
                await settings.memcache.set(cache_key, digest.encode())
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
    size_key = f"hfsize:{model}:{revision}:{path}".encode()
    cached = await settings.memcache.get(size_key)
    if cached:
        file_size = int(cached.decode())
    else:
        async with aiohttp.ClientSession() as session:
            url = f"https://huggingface.co/{model}/resolve/{revision}/{path}"
            async with session.head(url) as resp:
                content_length = resp.headers.get("x-linked-size")
                if content_length:
                    logger.info(f"Size of {model} -> {path}: {content_length}")
                    file_size = int(content_length)
                    await settings.memcache.set(size_key, content_length.encode())
                else:
                    logger.warning(f"Could not determine size of {model} -> {path}")
                    return

    # Now a random offset.
    start_byte = 0
    end_byte = min(file_size, random.randint(25, 500))
    if file_size:
        check_size = min(file_size - 1, random.randint(100, 500))
        start_byte = random.randint(0, file_size - check_size)
        end_byte = start_byte + check_size
    expected_digest = None
    async with aiohttp.ClientSession() as session:
        url = f"https://huggingface.co/{model}/resolve/{revision}/{path}"
        async with session.get(
            url, headers={"Range": f"bytes={start_byte}-{end_byte - 1}"}
        ) as resp:
            content = await resp.read()
            expected_digest = hashlib.sha256(content).hexdigest()

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
            data = await do_slurp(instance, payload, encrypted_slurp)
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
                    f"Digest of {path} on {instance.instance_id=} of {model} is correct: [{start_byte}:{end_byte}] {expected_digest}"
                )
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
                hard_failed.append(incorrect)


async def check_llm_weights(chute, instances):
    """
    Check the model weights for vllm (and sglang) templated chutes.
    """
    if not instances:
        logger.warning(f"No instances to check: {chute.name}")
        return [], []
    chute_id = chute.chute_id

    # Revision will need to be a requirement in the future, and at that point
    # it can be an attribute on the chute object rather than this janky regex.
    revision_match = re.search(r"(?:--revision |^\s+revision=)([a-f0-9]{40})", chute.code)
    if not revision_match:
        # Need to fetch remote revisions and allow a range of them.
        logger.warning(f"No revision to check: {chute.name}")
        return [], []

    revision = revision_match.group(1)
    logger.info(f"Checking {chute.chute_id=} {chute.name=} for {revision=}")
    encrypted_slurp = use_encrypted_slurp(chute.chutes_version)

    # Test each instance.
    hard_failed = []
    soft_failed = []
    instances = await load_chute_instances(chute_id)
    if not instances:
        return

    # First we'll check the primary config files, then we'll test the weights from the map.
    target_paths = [
        "model.safetensors.index.json",
        "config.json",
    ]
    weight_map = None
    for target_path in target_paths:
        incorrect = []
        digest_counts = {}
        expected_digest, expected_content = await get_hf_content(chute.name, revision, target_path)
        if not expected_digest:
            # Could try other means later on but for now treat as "OK".
            logger.warning(
                f"Failed to check huggingface for {target_path} on {chute.name} {revision=}"
            )
            continue
        if expected_content and target_path == "model.safetensors.index.json":
            weight_map = json.loads(expected_content)
        for instance in instances:
            nice_name = chute.name.replace("/", "--")
            payload = {"path": f"/cache/hub/models--{nice_name}/snapshots/{revision}/{target_path}"}
            try:
                data = await do_slurp(instance, payload, encrypted_slurp)
                if data is None:
                    hard_failed.append(instance)
                    continue
                digest = hashlib.sha256(data).hexdigest()
                if digest not in digest_counts:
                    digest_counts[digest] = 0
                digest_counts[digest] += 1
                if expected_digest and expected_digest != digest:
                    logger.warning(
                        f"Digest of {target_path} on {instance.instance_id=} of {chute.name} "
                        f"is incorrect: {expected_digest} vs {digest}"
                    )
                    incorrect.append(instance)
                logger.info(
                    f"Digest of {target_path} on {instance.instance_id=} of {chute.name}: {digest}"
                )
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
                    f"Huggingface digest mismatch, but all miners are in consensus: {expected_digest=} for {target_path} of {chute.name}"
                )
            else:
                for inst in incorrect:
                    hard_failed.append(incorrect)

    # Now check the actual weights.
    if weight_map:
        await check_weight_files(
            encrypted_slurp, chute.name, revision, instances, weight_map, hard_failed, soft_failed
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
    command_line = re.sub(r"([^ ]+/)?python3?(\.[0-9]+)", "python", command_line)
    expected = " ".join(
        [
            "python",
            "/home/chutes/.local/bin/chutes",
            "run",
            chute.ref_str,
            "--port",
            "8000",
            "--graval-seed",
            str(instance.nodes[0].seed),
            "--miner-ss58",
            instance.miner_hotkey,
            "--validator-ss58",
            settings.validator_ss58,
        ]
    ).strip()
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
    payload = {"foo": expected}
    iv = None
    if use_encryption_v2(chute.chutes_version):
        payload = aes_encrypt(json.dumps(payload).encode(), instance.symmetric_key)
        iv = payload[:32]
    path = "_ping"
    if use_encrypted_path(chute.chutes_version):
        path = aes_encrypt("/_ping", instance.symmetric_key, hex_encode=True)
    async with miner_client.post(
        instance.miner_hotkey,
        f"http://{instance.host}:{instance.port}/{path}",
        payload,
        timeout=10.0,
    ) as resp:
        raw_content = await resp.read()
        pong = None
        if b'{"json":' in raw_content:
            decrypted = aes_decrypt(json.loads(raw_content)["json"], instance.symmetric_key, iv)
            pong = json.loads(decrypted)["foo"]
        else:
            pong = json.loads(raw_content)["foo"]
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
    fail_key = f"watchtower:fail:{instance.instance_id}".encode()
    if not await settings.memcache.get(fail_key):
        await settings.memcache.set(fail_key, b"0", exptime=3600)
    fail_count = await settings.memcache.incr(fail_key)
    if fail_count >= 2:
        logger.warning(
            f"Instance {instance.instance_id} "
            f"miner {instance.miner_hotkey} "
            f"chute {chute.name} reached max soft fails: {fail_count}"
        )
        await purge_and_notify(instance)


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
    if chute.name == "stablediffusionapi-realistic-vision-v61":
        return

    # Ping test.
    instances = await load_chute_instances(chute.chute_id)
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
        logger.warning(
            f"Purging instance {instance.instance_id} "
            f"miner {instance.miner_hotkey} "
            f"chute {chute.name} due to hard fail"
        )
        await purge_and_notify(instance)

    # Limit "soft" fails to max consecutive failures, allowing some downtime but not much.
    for instance in soft_failed:
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


async def purge_unverified():
    """
    Purge all unverified instances that have been sitting around for a while.
    """
    async with get_session() as session:
        query = (
            select(Instance)
            .where(
                Instance.created_at <= func.now() - timedelta(hours=2, minutes=30),
                Instance.verified.is_(False),
            )
            .options(joinedload(Instance.chute))
        )
        total = 0
        for instance in (await session.execute(query)).unique().scalars().all():
            delta = int((datetime.now() - instance.created_at.replace(tzinfo=None)).total_seconds())
            logger.warning(
                f"Purging instance {instance.instance_id} of {instance.chute.name} which was created {instance.created_at} ({delta} seconds ago)..."
            )
            logger.warning(f"  {instance.verified=} {instance.active=}")
            await purge_and_notify(
                instance, reason="Instance failed to verify within 2.5 hours of creation"
            )
            total += 1
        if total:
            logger.success(f"Purged {total} total unverified+old instances.")


async def keep_cache_warm():
    """
    Keep some of the DB-heavy endpoints warm in cache so API requests are always fast.
    """
    from api.miner.router import get_scores, get_stats

    while True:
        try:
            logger.info("About to warm up cache...")
            async with get_session() as session:
                await get_stats(miner_hotkey=None, session=session, per_chute=False, request=None)
                logger.success("Warmed up stats endpoint, per_chute=False")
                await get_stats(miner_hotkey=None, session=session, per_chute=True, request=None)
                logger.success("Warmed up stats endpoint, per_chute=True")
                await get_scores(hotkey=None, request=None)
                logger.success("Warmed up scores endpoint")
        except Exception as exc:
            logger.warning(f"Error warming up cache: {exc}")
        await asyncio.sleep(60)


async def keep_miner_chute_history_warm():
    """
    Continuously update the miner unique chute count endpoint.
    """
    from api.metasync import get_unique_chute_history

    while True:
        logger.info("Attempting to warm up unique chute history...")
        history = None
        started_at = time.time()
        try:
            history = await get_unique_chute_history()
            for hotkey, values in history.items():
                cache_key = f"uqhist:{hotkey}".encode()
                await settings.memcache.set(cache_key, json.dumps(values).encode())
        except Exception as exc:
            logger.error(f"Error warming up unique chute history: {exc}")
            await asyncio.sleep(60)
            continue
        delta = time.time() - started_at
        logger.success(
            f"Successfully warmed up unique chute history for {len(history)} hotkeys in {int(delta)} seconds."
        )
        await asyncio.sleep(300)


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


async def remove_undeployable_chutes():
    """
    Remove chutes that only one miner (or tiny subnset of miners) can deploy,
    because it's almost certainly someone trying to cheat.
    """

    query = text(UNDEPLOYABLE_CHUTE_QUERY)
    bad_chutes = []
    async with get_session() as session:
        result = await session.execute(query)
        rows = result.fetchall()
        for row in rows:
            chute_id = row.chute_id
            invocation_count = row.invocation_count
            successful_miner_count = row.successful_miner_count
            audit_miner_count = row.audit_miner_count
            bad_chutes.append(
                (
                    chute_id,
                    f"chute is not broadly deployable by miners: {invocation_count=} {successful_miner_count=} {audit_miner_count=}",
                )
            )
            logger.warning(
                f"Detected undeployable chute {chute_id} with {invocation_count} invocations, "
                f"{successful_miner_count} successful miners out of {audit_miner_count} total miners"
            )
            chute = (
                (await session.execute(select(Chute).where(Chute.chute_id == chute_id)))
                .unique()
                .scalar_one_or_none()
            )
            if chute:
                version = chute.version
                await session.delete(chute)
                await settings.redis_client.publish(
                    "miner_broadcast",
                    json.dumps(
                        {
                            "reason": "chute_deleted",
                            "data": {"chute_id": chute_id, "version": version},
                        }
                    ),
                )
        await session.commit()

    # Generate the reports in separate sessions so we don't have massive transactions.
    for chute_id, reason in bad_chutes:
        await generate_confirmed_reports(chute_id, reason)


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
                        json.dumps(reason, indent=2),
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
                    ),
                )
                await session.commit()
            reason = f"Chute contains code identified by DeepSeek-R1 as likely cheating: {json.dumps(reason)}"
            await generate_confirmed_reports(chute.chute_id, reason)
        else:
            logger.success(f"Chute seems fine: {chute.chute_id=} {chute.name=}")


async def update_past_day_metrics():
    """
    Update the past day invocation counts for sorting.
    """
    while True:
        try:
            logger.info("Updating past day metrics...")
            async with get_session() as session:
                await session.execute(text(PAST_DAY_METRICS_QUERY))
            logger.success("Updated past day invocation metric on chutes.")
            await asyncio.sleep(1800)
        except Exception as exc:
            logger.error(f"Error updating past day invocation metrics on chutes: {exc}")
            await asyncio.sleep(300)


async def generate_invocation_history_metrics_loop():
    """
    Continuously update the invocation metrics summary tables.
    """
    while True:
        try:
            logger.info("Updating global historical metrics data...")
            await generate_invocation_history_metrics()
            await asyncio.sleep(3600)
        except Exception as exc:
            logger.error(f"Error updating global historical metrics tables: {exc}")
            await asyncio.sleep(300)


async def remove_disproportionate_new_chutes():
    """
    Remove chutes that are new and have disproportionate requests to one miner.
    """
    query = text(DISPROPORTIONATE_CHUTES)
    bad_chutes = []
    async with get_session() as session:
        result = await session.execute(query)
        rows = result.fetchall()
        for row in rows:
            chute_id = row.chute_id
            miner_hotkey = row.miner_hotkey
            miner_count = row.total_count
            total_count = row.total_invocations_per_chute
            unique_hotkeys = row.unique_hotkeys
            invocation_ratio = row.invocation_ratio
            bad_chutes.append(
                (
                    chute_id,
                    f"chute is new and has disproportionate requests to a single miner: {miner_hotkey=} {miner_count=} {total_count=} {unique_hotkeys=} {invocation_ratio=}",
                )
            )
            logger.warning(
                f"Detected disproportionate invocations on chute {chute_id} going to single miner: "
                f"{miner_hotkey=} {miner_count=} {total_count=} {unique_hotkeys=} {invocation_ratio=}"
            )
            chute = (
                (await session.execute(select(Chute).where(Chute.chute_id == chute_id)))
                .unique()
                .scalar_one_or_none()
            )
            if chute:
                version = chute.version
                await session.delete(chute)
                await settings.redis_client.publish(
                    "miner_broadcast",
                    json.dumps(
                        {
                            "reason": "chute_deleted",
                            "data": {"chute_id": chute_id, "version": version},
                        }
                    ),
                )
        await session.commit()

    # Generate the reports in separate sessions so we don't have massive transactions.
    for chute_id, reason in bad_chutes:
        await generate_confirmed_reports(chute_id, reason)


async def main():
    """
    Main loop, continuously check all chutes and instances.
    """
    # Cache warmup in the background for miner stats and scores, since those are DB heavy.
    asyncio.create_task(keep_cache_warm())
    asyncio.create_task(keep_miner_chute_history_warm())

    # Metrics.
    asyncio.create_task(update_past_day_metrics())
    asyncio.create_task(generate_invocation_history_metrics_loop())

    index = 0
    while True:
        ### Only enabled in clean-up mode.
        # await remove_bad_chutes()
        if index % 10 == 0:
            ### Only enabled in clean-up mode.
            # await remove_undeployable_chutes()
            await report_short_lived_chutes()
        await purge_unverified()
        await check_all_chutes()
        await asyncio.sleep(90)
        index += 1


asyncio.run(main())
