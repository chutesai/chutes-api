import uuid
import orjson as json
import asyncio
import aiohttp
import random
import traceback
import ctypes
from typing import Any, Dict, List, Tuple
from loguru import logger
from sqlalchemy import select, text, exists
import api.database.orms  # noqa
import api.miner_client as miner_client
from api.config import settings
from api.chute.schemas import RollingUpdate, Chute
from api.database import get_session
from api.instance.schemas import Instance
from api.instance.util import invalidate_instance_cache
from api.util import aes_encrypt, notify_deleted, semcomp, load_shared_object
from api.chute.util import get_one

ENETUNREACH_TOKEN = "ENETUNREACH"
REDIS_PREFIX = "conntestfail:"
PROXY_URL = "https://proxy.chutes.ai/misc/proxy?url=ping"
LBPING_URL = "https://api.chutes.ai/_lbping"

NETNANNY = load_shared_object("chutes", "chutes-netnanny.so")
NETNANNY.verify.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_uint8]
NETNANNY.verify.restype = ctypes.c_int


async def _post_connectivity(instance: Instance, endpoint: str) -> Dict[str, Any]:
    enc_path = aes_encrypt("/_connectivity", instance.symmetric_key, hex_encode=True)
    url = f"http://{instance.host}:{instance.port}/{enc_path}"
    payload = {"endpoint": endpoint}
    payload = aes_encrypt(json.dumps(payload), instance.symmetric_key)
    async with miner_client.post(
        instance.miner_hotkey,
        url,
        payload,
        timeout=15.0,
    ) as resp:
        resp.raise_for_status()
        return await resp.json()


async def _post_netnanny_challenge(instance: Instance, challenge: str) -> Dict[str, Any]:
    enc_path = aes_encrypt("/_netnanny_challenge", instance.symmetric_key, hex_encode=True)
    url = f"http://{instance.host}:{instance.port}/{enc_path}"
    payload = {"challenge": challenge}
    payload = aes_encrypt(json.dumps(payload), instance.symmetric_key)
    async with miner_client.post(
        instance.miner_hotkey,
        url,
        payload,
        timeout=15.0,
    ) as resp:
        resp.raise_for_status()
        return await resp.json()


def _pick_random_connect_test() -> str:
    candidates = list(getattr(settings, "conntest_urls", []) or [])
    if not candidates:
        candidates = ["https://icanhazip.com", "https://www.google.com/"]
    return random.choice(candidates)


async def _hard_delete_instance(session, instance: Instance, reason: str) -> None:
    chute = (
        (await session.execute(select(Chute).where(Chute.chute_id == instance.chute_id)))
        .unique()
        .scalar_one_or_none()
    )
    if not chute:
        return
    if "too many requests" in reason.lower():
        if semcomp(instance.chutes_version or "0.0.0", "0.3.58") < 0:
            logger.warning("Ignoring 429 from older chutes version...")
        else:
            logger.error(
                f"Hmmm, why 429?: {instance.instance_id=} {instance.miner_hotkey=} {instance.chute_id=} {chute.name=} {chute.chute_id=}"
            )
        return
    if chute.public or "affine" in chute.name.lower():
        return

    logger.error(
        f"🛑 HARD FAIL (egress policy violation): deleting {instance.instance_id=} "
        f"{instance.miner_hotkey=} {instance.chute_id=} {chute.name=} {chute.chute_id=}. Reason: {reason}"
    )
    return
    await session.delete(instance)
    await session.execute(
        text(
            "UPDATE instance_audit SET deletion_reason = :reason WHERE instance_id = :instance_id"
        ),
        {"instance_id": instance.instance_id, "reason": reason},
    )
    await notify_deleted(instance, message=reason)
    await invalidate_instance_cache(instance.chute_id, instance_id=instance.instance_id)
    await session.commit()


async def _record_failure_or_delete(session, instance: Instance, hard_reason: str | None) -> None:
    if hard_reason:
        await _hard_delete_instance(session, instance, hard_reason)
        return
    rkey = f"{REDIS_PREFIX}{instance.instance_id}"
    try:
        failure_count = await settings.redis_client.incr(rkey)
        await settings.redis_client.expire(rkey, 600)
    except Exception as redis_exc:
        logger.warning(f"Redis error tracking failures for {instance.instance_id}: {redis_exc}")
        failure_count = 3
    if failure_count >= 3:
        await _hard_delete_instance(
            session,
            instance,
            "Failed 3 or more consecutive connectivity probes.",
        )


async def _verify_netnanny(instance: Instance, allow_external_egress: bool) -> None:
    """
    Raises RuntimeError with a descriptive reason if verification fails.
    """
    challenge = str(uuid.uuid4())
    res = await _post_netnanny_challenge(instance, challenge)
    miner_hash = res.get("hash")
    miner_egress = res.get("allow_external_egress")
    if miner_hash is None or miner_egress is None:
        raise RuntimeError(
            "Netnanny challenge missing required fields (hash/allow_external_egress)."
        )
    if bool(miner_egress) != bool(allow_external_egress):
        raise RuntimeError(
            f"Netnanny reported allow_external_egress={miner_egress} "
            f"but chute requires {allow_external_egress}."
        )
    if not NETNANNY.verify(
        challenge.encode(), miner_hash.encode(), ctypes.c_uint8(allow_external_egress)
    ):
        raise RuntimeError("Netnanny verify() returned failure.")


async def check_instance_connectivity(
    instance: Instance, delete_on_failure: bool = True
) -> Tuple[str, bool]:
    logger.info(
        f"Connectivity check: {instance.instance_id=} {instance.miner_hotkey=} {instance.chute_id=}"
    )
    chute = await get_one(instance.chute_id)
    if not chute or semcomp(chute.chutes_version or "0.0.0", "0.3.50") < 0:
        logger.warning(f"Unable to perform connectivity tests for {instance.chute_id=}")
        return instance.instance_id, True

    allow_egress = chute.allow_external_egress
    try:
        await _verify_netnanny(instance, allow_egress)
        logger.success(f"🔒 netnanny challenge verified for {instance.instance_id=}")
    except Exception as exc:
        logger.error(
            f"❌ netnanny verification failed for {instance.instance_id=}: {str(exc)}\n{traceback.format_exc()}"
        )
        if delete_on_failure:
            async with get_session() as session:
                # Treat as hard violation: miner is misreporting or hash invalid.
                await _hard_delete_instance(
                    session,
                    instance,
                    f"Netnanny verification failed: {exc}",
                )
        return instance.instance_id, False

    random_test = _pick_random_connect_test()
    if allow_egress:
        required_successes = [LBPING_URL, random_test]
        try:
            for target in required_successes:
                result = await _post_connectivity(instance, target)
                if not result.get("connection_established"):
                    raise RuntimeError(
                        f"Egress allowed, but connection_established is False for {target}; "
                        f"error={result.get('error')}"
                    )
            logger.success(
                f"✅ egress allowed & verified for {instance.instance_id=} (lbping + random ok)"
            )
            try:
                await settings.redis_client.delete(f"{REDIS_PREFIX}{instance.instance_id}")
            except Exception:
                pass
            return instance.instance_id, True
        except Exception as exc:
            logger.error(
                f"❌ egress-allowed probe failed for {instance.instance_id=}: {exc}\n"
                f"{traceback.format_exc()}"
            )
            if delete_on_failure:
                async with get_session() as session:
                    await _record_failure_or_delete(session, instance, hard_reason=None)
            return instance.instance_id, False
    else:
        try:
            proxy_res = await _post_connectivity(instance, PROXY_URL)
            if not proxy_res.get("connection_established") or proxy_res.get("status_code") != 200:
                raise RuntimeError(
                    f"Proxy must succeed but got connection_established="
                    f"{proxy_res.get('connection_established')} status_code={proxy_res.get('status_code')} "
                    f"error={proxy_res.get('error')}"
                )
            disallowed = [LBPING_URL, random_test]
            bad_violation_reason = None
            for target in disallowed:
                res = await _post_connectivity(instance, target)
                conn = bool(res.get("connection_established"))
                if conn:
                    bad_violation_reason = (
                        f"Egress disabled but connection_established is True for {target}"
                    )
                    break
            if bad_violation_reason:
                if delete_on_failure:
                    async with get_session() as session:
                        await _hard_delete_instance(session, instance, bad_violation_reason)
                return instance.instance_id, False
            logger.success(
                f"✅ egress blocked & verified for {instance.instance_id=} "
                f"(proxy ok, direct outbound blocked with ENETUNREACH)"
            )
            try:
                await settings.redis_client.delete(f"{REDIS_PREFIX}{instance.instance_id}")
            except Exception:
                pass
            return instance.instance_id, True

        except Exception as exc:
            logger.error(
                f"❌ egress-blocked probe failed for {instance.instance_id=}: {exc}\n"
                f"{traceback.format_exc()}"
            )
            async with get_session() as session:
                await _record_failure_or_delete(session, instance, hard_reason=None)
            return instance.instance_id, False


async def check_connectivity_all(max_concurrent: int = 32) -> None:
    # Make sure we can do the tests...
    smoke_tests = [
        PROXY_URL,
        LBPING_URL,
        "https://icanhazip.com",
        "https://www.google.com",
    ]
    for url in smoke_tests:
        async with aiohttp.ClientSession(raise_for_status=True) as session:
            async with session.get(url) as resp:
                assert resp.ok
                logger.success(f"Successfully pinged {url=} in smoke test.")

    semaphore = asyncio.Semaphore(max_concurrent)

    async def guarded(instance: Instance):
        async with semaphore:
            return await check_instance_connectivity(instance)

    async with get_session() as session:
        q = select(Instance).where(
            Instance.active.is_(True),
            ~exists(
                select(1)
                .select_from(RollingUpdate)
                .where(RollingUpdate.chute_id == Instance.chute_id)
            ),
        )
        stream = await session.stream(q)
        instances: List[Instance] = []
        async for row in stream.unique():
            instances.append(row[0])

    logger.info(f"Connectivity: checking {len(instances)} active instances")
    results = await asyncio.gather(*(guarded(i) for i in instances))
    ok = sum(1 for _, passed in results if passed)
    bad = len(results) - ok
    logger.success("=" * 80)
    logger.success(f"Connectivity check complete: {ok} passed, {bad} failed")
    if bad:
        failed_ids = [iid for iid, passed in results if not passed]
        logger.warning(f"Failed instances: {failed_ids}")


if __name__ == "__main__":
    asyncio.run(check_connectivity_all())
