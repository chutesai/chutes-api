"""
Application logic and utilities for chutes.
"""

import aiohttp
import re
from loguru import logger
from typing import List
from sqlalchemy import or_
from sqlalchemy.future import select
from run_api.chute.schemas import Chute
from run_api.user.schemas import User
from run_api.instance.schemas import Instance
from run_api.utils import sse, now_str


async def get_chute_by_id_or_name(chute_id_or_name, db, current_user):
    """
    Helper to load a chute by ID or full chute name (optional username/chute name)
    """
    name_match = re.match(
        r"(?:([a-z0-9][a-z0-9_-]*)/)?([a-z0-9][a-z0-9_-]*)$",
        chute_id_or_name.lstrip("/"),
        re.I,
    )
    query = (
        select(Chute)
        .join(User, Chute.user_id == User.user_id)
        .where(or_(Chute.public.is_(True), Chute.user_id == current_user.user_id))
    )
    if name_match:
        username = name_match.group(1) or current_user.username
        chute_name = name_match.group(2)
        query = query.where(User.username == username).where(Chute.name == chute_name)
    else:
        query = query.where(Chute.chute_id == chute_id_or_name)
    result = await db.execute(query)
    return result.scalar_one_or_none()


async def _invoke_one(
    chute: Chute, path: str, stream: bool, args: str, kwargs: str, target: Instance
):
    """
    Try invoking a chute/cord with a single instance.
    """
    session = aiohttp.ClientSession(raise_for_status=True)
    response = await session.post(
        f"http://{target.ip}:{target.port}/{chute.chute_id}{path}",
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
    chute_id = chute.chute_id
    yield sse(
        {
            "trace": {
                "timestamp": now_str(),
                "chute_id": chute_id,
                "function": function,
                "message": f"identified {len(targets)} available targets",
            },
        }
    )
    logger.info(f"trying function invocation with up to {len(targets)} targets")
    for target in targets:
        try:
            yield sse(
                {
                    "trace": {
                        "timestamp": now_str(),
                        "chute_id": chute_id,
                        "function": function,
                        "message": f"attempting to query target={target.instance_id} uid={target.miner_uid} hotkey={target.miner_hotkey} coldkey={target.miner_coldkey}",
                    },
                }
            )
            async for data in _invoke_one(chute, path, stream, args, kwargs, target):
                yield sse({"result": data})
            yield sse(
                {
                    "trace": {
                        "timestamp": now_str(),
                        "chute_id": chute_id,
                        "function": function,
                        "message": f"successfully called {function=} on target={target.instance_id} uid={target.miner_uid} hotkey={target.miner_hotkey} coldkey={target.miner_coldkey}",
                    }
                }
            )
            return
        except Exception as exc:
            yield sse(
                {
                    "trace": {
                        "timestamp": now_str(),
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
