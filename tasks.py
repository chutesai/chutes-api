import os
import time

os.environ["VAULT_URL"] = "http://127.0.0.1:777"
import asyncio
from sqlalchemy.exc import IntegrityError

import typer
from loguru import logger
from run_api.api_key.schemas import APIKey, APIKeyArgs
from run_api.database import Base, get_db, engine
from sqlalchemy import delete, select

# The below have to be here to prevent SQLAlchemy initialization errors
from run_api.user.schemas import User
from run_api.chute.schemas import Chute  # noqa: F401
from run_api.image.schemas import Image  # noqa: F401
from run_api.instance.schemas import Instance  # noqa: F401
from run_api.user import events  # noqa: F401
from rich.table import Table
from rich.console import Console

app = typer.Typer(no_args_is_help=True)


async def _run_migrations():
    """Run database migrations."""

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    logger.info("Migrations run successfully.")


@app.command()
def run_migrations():
    """Run database migrations."""

    asyncio.run(_run_migrations())


async def _add_user(
    username: str,
    coldkey: str | None = None,
    hotkey: str | None = None,
):
    async with get_db() as db:
        user, fingerpint = User.create(
            username=username, coldkey=coldkey, hotkey=hotkey
        )

        _query = select(User).where(User.username == username)
        existing_user = await db.execute(_query)
        existing_user = existing_user.scalars().first()

        if existing_user:
            await db.execute(delete(User).where(User.username == username))
            await db.commit()

        if coldkey:
            user.coldkey = coldkey

        if hotkey:
            user.hotkey = hotkey

        db.add(user)
        await db.commit()
        await db.refresh(user)
        await db.close()

        logger.info(f"Added user: {user}")
        return user, fingerpint


@app.command()
def add_user(username: str, coldkey: str | None = None, hotkey: str | None = None):
    """Add a user to the database."""

    asyncio.run(_add_user(username, coldkey, hotkey))


async def _add_api_key(user: User, name: str = "test-key", admin: bool = False):
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


@app.command()
def add_api_key(user_id: int, name: str = "test-key", admin: bool = False):
    """Add an API key to the database."""
    asyncio.run(_add_api_key(user_id=user_id, name=name, admin=admin))


async def _dev_setup():

    os.system("docker compose  up -d")
    time.sleep(2)
    await _run_migrations()
    users = [
        User(
            username="test_key",
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

    accounts = []
    api_keys = []
    for user in users:
        account, fingerprint = await _add_user(
            username=user.username, coldkey=user.coldkey, hotkey=user.hotkey
        )
        if not account:
            logger.info(
                "Created Account: {} ({} - {})".format(
                    account.username, account.fingerprint_hash, account.coldkey
                )
            )

        api_key = await _add_api_key(account)

        accounts.append((account, fingerprint))
        api_keys.append(api_key)

    logger.info(f"Accounts: {accounts}")
    logger.info(f"API keys: {api_keys}")


@app.command()
def dev_setup():
    """Setup development environment with test users and API keys."""

    asyncio.run(_dev_setup())


async def _remove_all_users():
    async with get_db() as db:
        await db.execute(delete(User))
        await db.execute(delete(APIKey))
        await db.commit()


@app.command()
def remove_all_users():
    """Remove all users from the database."""

    asyncio.run(_remove_all_users())


async def _remove_user(username: str):
    async with get_db() as db:
        await db.execute(delete(User).where(User.username == username))
        await db.commit()


@app.command()
def remove_user(username: str):
    """Remove a user from the database."""

    asyncio.run(_remove_user(username))


async def _list_users():
    async with get_db() as db:
        users = await db.execute(select(User))

        # Create table
        table = Table(show_header=True, header_style="bold magenta")

        # Add columns
        table.add_column("Username")
        table.add_column("Coldkey")
        table.add_column("Hotkey")
        table.add_column("Created At")

        # Add rows
        for user in users.scalars().all():
            table.add_row(
                user.username, user.coldkey, user.hotkey, str(user.created_at)
            )

        # Display table
        console = Console()
        console.print(table)


@app.command()
def list_users():
    """List all users in the database."""

    asyncio.run(_list_users())


async def _destroy_database():
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)


@app.command()
def destroy_database():
    """Destroy the database."""

    asyncio.run(_destroy_database())


@app.command()
def reset():
    """Reset the database and run dev setup"""

    async def run_reset():
        await _destroy_database()
        await _dev_setup()

    asyncio.run(run_reset())


@app.command()
def start_miner(
    chutes_dir: str = typer.Option(
        default="~/chutes", help="The directory containing the chutes source code."
    )
):
    """Start the miner."""
    # First copy chutes source dir to the data/ container
    chutes_dir = os.path.expanduser(chutes_dir)
    logger.info(f"Copying chutes source dir to data/chutes: {chutes_dir}")

    os.system(f"git -C {chutes_dir} pull")

    os.system(
        f"rsync -av --exclude-from='.gitignore' --exclude='.git' --exclude='venv' --exclude='.venv' --exclude='__pycache__' '{chutes_dir}/' data/chutes/"
    )
    logger.info(
        f"Copied chutes source dir to data/chutes: {chutes_dir}. Now building and starting the miner."
    )
    os.system("docker compose -f docker-compose-miner.yml build vllm")
    logger.info("Built the miner. Now starting the miner.")
    os.system("docker compose -f docker-compose-miner.yml up -d vllm")
    logger.info("Miner started. You can now use the miner.")


if __name__ == "__main__":
    app()
