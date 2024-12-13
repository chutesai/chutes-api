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
import pybase64 as base64
from loguru import logger
from typing import List
from sqlalchemy import and_, or_, text, update, func, String
from sqlalchemy.future import select
from sqlalchemy.orm import selectinload
from api.config import settings
from api.constants import ENCRYPTED_HEADER
from api.database import get_session
from api.exceptions import InstanceRateLimit
from api.util import sse, now_str
from api.chute.schemas import Chute, NodeSelector
from api.user.schemas import User
from api.miner_client import sign_request
from api.instance.schemas import Instance
from api.payment.constants import COMPUTE_UNIT_PRICE_BASIS
from api.permissions import Permissioning

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
WHERE invocation_id = :invocation_id AND miner_uid = :miner_uid
RETURNING CEIL(EXTRACT(EPOCH FROM (completed_at - started_at))) * compute_multiplier AS total_compute_units
"""


async def get_chute_by_id_or_name(chute_id_or_name, db, current_user, load_instances: bool = False):
    """
    Helper to load a chute by ID or full chute name (optional username/chute name)
    """
    if not chute_id_or_name:
        return None
    name_match = re.match(
        r"/?(?:([a-zA-Z0-9_\.-]{3,15})/)?([a-z0-9][a-z0-9_\.\/-]*)$",
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
    if load_instances:
        query = query.options(selectinload(Chute.instances))
    username = name_match.group(1) or current_user.username
    chute_name = name_match.group(2)
    chute_id_or_name = chute_id_or_name.lstrip("/")
    query = query.where(
        or_(
            and_(
                User.username == current_user.username,
                Chute.name == chute_name,
            ),
            and_(
                User.username == current_user.username,
                Chute.name == chute_id_or_name,
            ),
            and_(
                User.username == username,
                Chute.name == chute_name,
            ),
            Chute.chute_id == chute_id_or_name,
        )
    )
    result = await db.execute(query)
    return result.scalar_one_or_none()


async def chute_id_by_slug(slug: str):
    """
    Check if a chute exists with the specified slug (which is a subdomain for standard apps).
    """
    async with get_session() as session:
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
    async with get_session() as session:
        await session.execute(
            update(Instance)
            .where(Instance.instance_id == target.instance_id)
            .values({"last_queried_at": func.now()})
        )
        await session.commit()

    # Call the miner's endpoint.
    path = path.lstrip("/")
    response = None
    payload = {"args": args, "kwargs": kwargs}
    encrypted = False
    if settings.graval_url:
        device_dicts = [node.graval_dict() for node in target.nodes]
        target_index = random.randint(0, len(device_dicts) - 1)
        target_device = device_dicts[target_index]
        seed = target.nodes[0].seed
        async with aiohttp.ClientSession(raise_for_status=True) as graval_session:
            try:
                async with graval_session.post(
                    f"{settings.graval_url}/encrypt",
                    json={
                        "payload": {
                            "args": args,
                            "kwargs": kwargs,
                        },
                        "device_info": target_device,
                        "device_id": target_index,
                        "seed": seed,
                    },
                    timeout=3.0,
                ) as resp:
                    payload = await resp.json()
                    encrypted = True
            except Exception as exc:
                logger.error(
                    f"Error encrypting payload: {str(exc)}, sending plain text\n{traceback.format_exc()}"
                )

    session = aiohttp.ClientSession()
    headers, payload_string = sign_request(miner_ss58=target.miner_hotkey, payload=payload)
    if encrypted:
        headers.update({ENCRYPTED_HEADER: "true"})
    logger.debug(
        f"Attempting invocation of {chute.chute_id=} on {target.instance_id=} {encrypted=}"
    )
    response = await session.post(
        f"http://{target.host}:{target.port}/{path}",
        data=payload_string,
        headers=headers,
    )
    try:
        if response.status == 429:
            raise InstanceRateLimit(
                "Instance {target.instance_id=} has returned a rate limit error!"
            )
        response.raise_for_status()
        if stream:
            async for chunk in response.content:
                yield chunk.decode()
        else:
            content_type = response.headers.get("content-type")
            if content_type in (None, "application/json"):
                json_data = await response.json()
                data = {"content_type": content_type, "json": json_data}
            elif content_type.startswith("text/"):
                text_data = await response.text()
                data = {"content_type": content_type, "text": text_data}
            else:
                raw_data = await response.read()
                data = {"content_type": content_type, "bytes": base64.b64encode(raw_data).decode()}
            yield data
    finally:
        await response.release()
        await session.close()


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
            async with settings.s3_client() as s3:
                await s3.upload_fileobj(
                    io.BytesIO(json.dumps({"args": args, "kwargs": kwargs})),
                    settings.storage_bucket,
                    request_path,
                )
        except Exception as exc:
            logger.error(f"failed to sample request: {exc}")
            request_path = None
    partition_suffix = None
    for target in targets:
        async with get_session() as session:
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

            async with get_session() as session:
                # Save the response if we're randomly sampling this one.
                response_path = None
                if request_path:
                    response_path = request_path.replace("request.json", "response.json")
                    try:
                        async with settings.s3_client() as s3:
                            await s3.upload_fileobj(
                                io.BytesIO(json.dumps(response_data)),
                                settings.storage_bucket,
                                response_path,
                            )
                    except Exception as exc:
                        logger.error(f"failed to sample request: {exc}")
                        response_path = None

                # Mark the invocation as complete.
                result = await session.execute(
                    text(UPDATE_INVOCATION.format(suffix=partition_suffix)),
                    {
                        "chute_id": chute_id,
                        "invocation_id": invocation_id,
                        "miner_uid": target.miner_uid,
                        "error_message": None,
                        "response_path": response_path,
                    },
                )

                # Calculate the credits used and deduct from user's balance.
                compute_units = result.scalar_one_or_none()
                if compute_units:
                    balance_used = compute_units * COMPUTE_UNIT_PRICE_BASIS / 3600
                    result = await session.execute(
                        update(User)
                        .where(User.user_id == user_id)
                        .where(
                            User.permissions_bitmask.op("&")(Permissioning.free_account.bitmask)
                            != 0
                        )
                        .values(balance=User.balance - balance_used)
                        .returning(User.balance, User.permissions_bitmask)
                    )
                    row = result.first_one_or_none()
                    if row is not None:
                        new_balance, permissions_bitmask = row
                        if permissions_bitmask & Permissioning.free_account.bitmask == 0:
                            logger.debug(
                                f"Deducted ${balance_used:.12f} from {user_id=}, new balance = ${new_balance:.12f}"
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
            error_message = f"{exc}\n{traceback.format_exc()}"
            if isinstance(exc, InstanceRateLimit):
                error_message = "RATE_LIMIT"
            async with get_session() as session:
                await session.execute(
                    text(UPDATE_INVOCATION.format(suffix=partition_suffix)),
                    {
                        "chute_id": chute_id,
                        "invocation_id": invocation_id,
                        "miner_uid": target.miner_uid,
                        "error_message": error_message,
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
                f"Error trying to call instance_id={target.instance_id} [chute_id={target.chute_id}]: {exc} -- {traceback.format_exc()}"
            )
    yield sse({"error": "exhausted all available targets to no avail"})
