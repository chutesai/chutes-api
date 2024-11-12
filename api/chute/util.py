"""
Application logic and utilities for chutes.
"""

import aiohttp
import re
import uuid
import random
import datetime
import io
import traceback
import orjson as json
from loguru import logger
from typing import List
from sqlalchemy import or_, text, update, func, String
from sqlalchemy.future import select
from api.config import settings
from api.database import SessionLocal
from api.util import sse, now_str
from api.chute.schemas import Chute, NodeSelector
from api.user.schemas import User
from api.instance.schemas import Instance


REQUEST_SAMPLE_RATIO = 0.05
TRACK_INVOCATION = text(
    """
INSERT INTO invocations (
    invocation_id,
    chute_id,
    chute_user_id,
    function_name,
    user_id,
    image_id,
    image_user_id,
    instance_id,
    miner_uid,
    miner_hotkey,
    started_at,
    completed_at,
    error_message,
    request_path,
    response_path,
    reported_at,
    report_reason,
    compute_multiplier,
    bounty
) VALUES (
    :invocation_id,
    :chute_id,
    :chute_user_id,
    :function_name,
    :user_id,
    :image_id,
    :image_user_id,
    :instance_id,
    :miner_uid,
    :miner_hotkey,
    CURRENT_TIMESTAMP,
    NULL,
    NULL,
    :request_path,
    NULL,
    NULL,
    NULL,
    :compute_multiplier,
    0
) RETURNING to_char(date_trunc('week', started_at), 'IYYY_IW') AS suffix
"""
).columns(suffix=String)
UPDATE_INVOCATION = """
WITH removed_bounty AS (
    DELETE FROM bounties
    WHERE chute_id = :chute_id
    RETURNING bounty
)
UPDATE partitioned_invocations_{suffix} SET
    completed_at = CURRENT_TIMESTAMP,
    error_message = CAST(:error_message AS TEXT),
    response_path = CAST(:response_path AS TEXT),
    bounty = CASE
        WHEN :error_message IS NULL THEN COALESCE((SELECT bounty FROM removed_bounty), bounty)
        ELSE bounty
    END
WHERE invocation_id = :invocation_id
"""


async def get_chute_by_id_or_name(chute_id_or_name, db, current_user):
    """
    Helper to load a chute by ID or full chute name (optional username/chute name)
    """
    if not chute_id_or_name:
        return None
    name_match = re.match(
        r"(?:([a-z0-9][a-z0-9_-]*)/)?([a-z0-9][a-z0-9_-]*)$",
        chute_id_or_name.lstrip("/"),
        re.I,
    )
    if not name_match:
        return None
    query = (
        select(Chute)
        .join(User, Chute.user_id == User.user_id)
        .where(or_(Chute.public.is_(True), Chute.user_id == current_user.user_id))
    )
    username = name_match.group(1) or current_user.username
    chute_name = name_match.group(2)
    query = query.where(User.username == username).where(
        or_(Chute.name == chute_name, Chute.chute_id == chute_id_or_name)
    )
    result = await db.execute(query)
    return result.scalar_one_or_none()


async def chute_id_by_slug(slug: str):
    """
    Check if a chute exists with the specified slug (which is a subdomain for standard apps).
    """
    async with SessionLocal() as session:
        if chute_id := (
            await session.execute(select(Chute.chute_id).where(Chute.slug == slug))
        ).scalar_one_or_none():
            return chute_id
    return None


async def _invoke_one(
    chute: Chute, path: str, stream: bool, args: str, kwargs: str, target: Instance
):
    """
    Try invoking a chute/cord with a single instance.
    """
    # Update last query time for this target.
    async with SessionLocal() as session:
        await session.execute(
            update(Instance)
            .where(Instance.instance_id == target.instance_id)
            .values({"last_queried_at": func.now()})
        )
        await session.commit()

    # Call the miner's endpoint.
    session = aiohttp.ClientSession(raise_for_status=True)
    response = await session.post(
        f"http://{target.host}:{target.port}/{chute.chute_id}{path}",
        json={"args": args, "kwargs": kwargs},
    )
    if stream:
        try:
            async for chunk in response.content:
                yield chunk.decode()
        finally:
            await response.release()
            await session.close()
    else:
        data = await response.json()
        await response.release()
        await session.close()
        yield data


async def invoke(
    chute: Chute,
    user_id: str,
    path: str,
    function: str,
    stream: bool,
    args: str,
    kwargs: str,
    targets: List[Instance],
):
    """
    Helper to actual perform function invocations, retrying when a target fails.
    """
    invocation_id = str(uuid.uuid4())
    chute_id = chute.chute_id
    yield sse(
        {
            "trace": {
                "timestamp": now_str(),
                "invocation_id": invocation_id,
                "chute_id": chute_id,
                "function": function,
                "message": f"identified {len(targets)} available targets",
            },
        }
    )
    logger.info(f"trying function invocation with up to {len(targets)} targets")
    request_path = None
    today = str(datetime.date.today())
    if random.random() <= REQUEST_SAMPLE_RATIO:
        request_path = f"invocations/{today}/{chute_id}/request.json"
        try:
            await settings.storage_client.put_object(
                settings.storage_bucket,
                request_path,
                io.BytesIO(json.dumps({"args": args, "kwargs": kwargs})),
                length=-1,
                part_size=10 * 1024 * 1024,
            )
        except Exception as exc:
            logger.error(f"failed to sample request: {exc}")
            request_path = None
    partition_suffix = None
    for target in targets:
        async with SessionLocal() as session:
            result = await session.execute(
                TRACK_INVOCATION,
                {
                    "invocation_id": invocation_id,
                    "function_name": function,
                    "chute_id": chute.chute_id,
                    "chute_user_id": chute.user_id,
                    "user_id": user_id,
                    "image_id": chute.image_id,
                    "image_user_id": chute.image.user_id,
                    "instance_id": target.instance_id,
                    "miner_uid": target.miner_uid,
                    "miner_hotkey": target.miner_hotkey,
                    "request_path": request_path,
                    "compute_multiplier": NodeSelector(**chute.node_selector).compute_multiplier,
                },
            )
            partition_suffix = result.scalar()
            await session.commit()

        try:
            yield sse(
                {
                    "trace": {
                        "timestamp": now_str(),
                        "invocation_id": invocation_id,
                        "chute_id": chute_id,
                        "function": function,
                        "message": f"attempting to query target={target.instance_id} uid={target.miner_uid} hotkey={target.miner_hotkey} coldkey={target.miner_coldkey}",
                    },
                }
            )
            response_data = []
            async for data in _invoke_one(chute, path, stream, args, kwargs, target):
                yield sse({"result": data})
                if request_path:
                    response_data.append(data)

            async with SessionLocal() as session:
                response_path = None
                if request_path:
                    response_path = request_path.replace("request.json", "response.json")
                    try:
                        await settings.storage_client.put_object(
                            settings.storage_bucket,
                            response_path,
                            io.BytesIO(json.dumps({"args": args, "kwargs": kwargs})),
                            length=-1,
                            part_size=10 * 1024 * 1024,
                        )
                    except Exception as exc:
                        logger.error(f"failed to sample request: {exc}")
                        response_path = None
                await session.execute(
                    text(UPDATE_INVOCATION.format(suffix=partition_suffix)),
                    {
                        "chute_id": chute_id,
                        "invocation_id": invocation_id,
                        "error_message": None,
                        "response_path": response_path,
                    },
                )
                await session.commit()

            yield sse(
                {
                    "trace": {
                        "timestamp": now_str(),
                        "invocation_id": invocation_id,
                        "chute_id": chute_id,
                        "function": function,
                        "message": f"successfully called {function=} on target={target.instance_id} uid={target.miner_uid} hotkey={target.miner_hotkey} coldkey={target.miner_coldkey}",
                    }
                }
            )
            return
        except Exception as exc:
            async with SessionLocal() as session:
                await session.execute(
                    text(UPDATE_INVOCATION.format(suffix=partition_suffix)),
                    {
                        "chute_id": chute_id,
                        "invocation_id": invocation_id,
                        "error_message": f"{exc}\n{traceback.format_exc()}",
                        "response_path": None,
                    },
                )
                await session.commit()

            yield sse(
                {
                    "trace": {
                        "timestamp": now_str(),
                        "invocation_id": invocation_id,
                        "chute_id": chute_id,
                        "function": function,
                        "message": f"error encountered while querying target={target.instance_id} uid={target.miner_uid} hotkey={target.miner_hotkey} coldkey={target.miner_coldkey}: {exc=}",
                    },
                }
            )
            logger.error(
                f"Error trying to call instance_id={target.instance_id} [chute_id={target.chute_id}]: {exc}"
            )
    yield sse({"error": "exhausted all available targets to no avail"})
