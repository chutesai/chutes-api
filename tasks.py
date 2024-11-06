import os
os.environ["VAULT_URL"] = "http://127.0.0.1:777"
import asyncio
import functools
from sqlalchemy.exc import IntegrityError

from fastapi import HTTPException
import invoke
from loguru import logger
from run_api.user.schemas import User
from run_api.api_key.schemas import APIKey, APIKeyArgs
from run_api.database import Base, get_db_session, engine
from sqlalchemy.ext.asyncio import AsyncSession

# The below have to be here to prevent sqlalchemey initialisation errors
from run_api.chute.schemas import Chute  # noqa: F401
from run_api.image.schemas import Image  # noqa: F401
from run_api.instance.schemas import Instance  # noqa: F401
from run_api.user import events  # noqa: F401  # NOTE:  I find this one especially annoying / dangerous


def run_async(fn):
    @functools.wraps(fn)
    def inner(*args, **kwargs):
        return asyncio.run(fn(*args, **kwargs))

    return inner


@invoke.task
@run_async
async def run_migrations(context: invoke.Context):
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


@invoke.task(pre=[run_migrations])
@run_async
async def add_user(
    context: invoke.Context,
    username: str,
    coldkey: str | None = None,
    hotkey: str | None = None,
):
    async with await anext(get_db_session()) as db:
        user = User(username=username)

        if coldkey:
            user.coldkey = coldkey

        if hotkey:
            user.hotkey = hotkey

        db.add(user)
        await db.commit()
        await db.refresh(user)
        return user


@invoke.task(pre=[run_migrations])
@run_async
async def add_api_key(user: User, name: str = "test-key", admin: bool = False):

    db: AsyncSession = get_db_session()
    key_args = APIKeyArgs(
        admin=admin,
        name=name,
    )
    api_key, _ = APIKey.create(user.user_id, key_args)
    try:
        db.add(api_key)
        await db.commit()
        await db.refresh(api_key)
    except IntegrityError as exc:
        if "unique constraint" in str(exc):
            raise Exception("An API key already exists with this name")
        raise

    return api_key


@invoke.task(pre=[run_migrations])
@run_async
async def dev_setup(context: invoke.Context):

    users = [
        User(
            username="test",
            coldkey=None,
            hotkey=None,
        ),
        User(
            username="test2",
            coldkey="5C4hrfjw9DjXZTzV3MwzrrAr9P1MJhSrvWGWqi1eSuyUpnhM",
            hotkey=None,
        ),
    ]

    # setup accounts, give each of them an API key
    for user in users:
        # create or find the account
        account: User = await add_user(**user)
        if not account:
            logger.info(
                "Created Account: {} ({} - {})".format(
                    account.username, account.fingerprint, account.coldkey
                )
            )

        # attach an API key
        api_key = await add_api_key(account)
        logger.info("Created ApiKey: {}".format(api_key))

    # that's it!


# @invoke.task(post=[dev_setup])
# @run_async
# async def dev_start(context: invoke.Context):
#     # setup the servers
#     context.run(
#         "docker compose --env-file .prod.env up -d --scale=api=0 --scale=caddy=0 --scale=postgres-proxy=1 --scale=redis-proxy=1"
#     )


# @invoke.task
# def dev_stop(context: invoke.Context, volumes: bool = False):
#     command = "docker compose --env-file .prod.env down --remove-orphans"
#     if volumes:
#         command += " --volumes"

#     context.run(command)
