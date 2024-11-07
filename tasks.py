import os

os.environ["VAULT_URL"] = "http://127.0.0.1:777"
import asyncio
import functools
from sqlalchemy.exc import IntegrityError

from fastapi import HTTPException
import asyncclick as click
from loguru import logger
from run_api.user.schemas import User
from run_api.api_key.schemas import APIKey, APIKeyArgs
from run_api.database import Base, SessionLocal, get_db, engine
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import delete, select, func

# The below have to be here to prevent sqlalchemey initialisation errors
from run_api.chute.schemas import Chute  # noqa: F401
from run_api.image.schemas import Image  # noqa: F401
from run_api.instance.schemas import Instance  # noqa: F401
from run_api.user import events  # noqa: F401


@click.command()
async def run_migrations():
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


async def add_user(
    username: str,
    coldkey: str | None = None,
    hotkey: str | None = None,
):
    async with get_db() as db:
        user = User(username=username)

        _query = select(User).where(User.username == username)
        existing_user = await db.execute(_query)
        existing_user = existing_user.scalars().first()

        if existing_user:
            return existing_user

        if coldkey:
            user.coldkey = coldkey

        if hotkey:
            user.hotkey = hotkey

        db.add(user)
        await db.commit()
        await db.refresh(user)
        await db.close()

        logger.info(f"Added user: {user}")
        return User(
            user_id=user.user_id,
            username=user.username,
            coldkey=user.coldkey,
            hotkey=user.hotkey,
        )


async def add_api_key(user: User, name: str = "test-key", admin: bool = False):
    async with get_db() as db:
        key_args = APIKeyArgs(
            admin=admin,
            name=name,
        )

        _query = select(APIKey).where(APIKey.name == name)
        existing_key = await db.execute(_query)
        existing_key = existing_key.scalars().first()

        if existing_key:
            await db.execute(
                delete(APIKey).where(APIKey.api_key_id == existing_key.api_key_id)
            )
            await db.commit()

        api_key, actual_api_key_lol = APIKey.create(user.user_id, key_args)
        try:
            db.add(api_key)
            await db.commit()
            await db.refresh(api_key)
        except IntegrityError as exc:
            if "unique constraint" in str(exc):
                raise Exception("An API key already exists with this name")
            raise

        logger.info(f"Added API key: {api_key} for user {user.username}")
        logger.info(f"The key is: {actual_api_key_lol}")
    return actual_api_key_lol


@click.command()
async def dev_setup():
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
        User(
            username="test3",
            coldkey="5C4hrfjw9DjXZTzV3MwzrrAr9P1MJhSrvWGWqi1eSuyUpnhM",
            hotkey="5EhZMRNFXZoh3BmFDsroRAHU9a9takaq3bYTLVEad9xZJMWv",
        ),
    ]

    # setup accounts, give each of them an API key

    accounts = []
    api_keys = []
    for user in users:
        # create or find the account
        account: User = await add_user(
            username=user.username, coldkey=user.coldkey, hotkey=user.hotkey
        )
        if not account:
            logger.info(
                "Created Account: {} ({} - {})".format(
                    account.username, account.fingerprint, account.coldkey
                )
            )
        print("sleeping!")

        # attach an API key
        api_key = await add_api_key(account)

        accounts.append(account)
        api_keys.append(api_key)

    logger.info(f"Accounts: {accounts}")
    logger.info(f"API keys: {api_keys}")


if __name__ == "__main__":
    dev_setup(_anyio_backend="asyncio")
