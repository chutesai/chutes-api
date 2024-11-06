import asyncio
import functools
import json
import pathlib

import asyncpg
import dotenv
import invoke
import redis
from loguru import logger
from run_api.user.schemas import User
from run_api.database import get_db_session

dotenv.load_dotenv(".prod.env")  # ew


def run_async(fn):
    @functools.wraps(fn)
    def inner(*args, **kwargs):
        return asyncio.run(fn(*args, **kwargs))

    return inner


@invoke.task
def dev_migrations(context: invoke.Context):
    context.run("dbmate --no-dump-schema --wait up")


@invoke.task
@run_async
async def dev_add_user(
    context: invoke.Context,
    hotkey: str,
    coldkey: str,
    username: str | None,
    subnet_id: int = 19,
):
    db = await get_db_session()

    user = User(
        username=username,
        coldkey=coldkey,
    )
    db.add(user)
    await db.commit()



@invoke.task
@run_async
async def dev_setup(context: invoke.Context):
    data_path = pathlib.Path("tmp_data.json")
    if not data_path.is_file():
        return

    # load and parse the temp data
    data = json.loads(data_path.read_text())

    # these credentials should work with postgres-proxy enabled
    await asyncio.sleep(1)
    db: asyncpg.Connection = await asyncpg.connect(
        "postgres://postgres:postgres@localhost/postgres?sslmode=disable"
    )

    # these credentials should work with redis-proxy enabled
    cache = redis.asyncio.Redis.from_url("redis://localhost")
    await cache.get("")  # forces redis to connect and will pop if it's unreachable

    # setup accounts, give each of them an API key
    for account_raw in data.get("accounts", []):
        # create or find the account
        account = await sql.account_get_by_coldkey(db, account_raw["coldkey"])
        if not account:
            fingerprint = account_raw.get("fingerprint", utils.gen_random(16))
            account = await sql.account_create(db, fingerprint, account_raw["coldkey"])
            logger.info(
                "Created Account: {} ({} - {})".format(
                    account.id, account.fingerprint, account.coldkey
                )
            )
        else:
            logger.info(
                "Account {} ({} - {}) was already there.".format(
                    account.id,
                    account.fingerprint,
                    account.coldkey,
                )
            )

        # attach an API key
        token = "19_{}".format(utils.gen_random(32))
        api_key = await sql.api_key_create(db, account, token)
        logger.info("Created ApiKey: {}".format(api_key.token))

        # ensure the rate limiter doesn't get in the way
        for validator in data.get("validators", []):
            await utils.rate_limit_set(cache, validator["hotkey"], account.coldkey, 999)

    # that's it!
    await db.close()


@invoke.task(post=[dev_setup])
@run_async
async def dev_start(context: invoke.Context):
    # setup the servers
    context.run("docker compose --env-file .prod.env up -d --scale=api=0 --scale=caddy=0 --scale=postgres-proxy=1 --scale=redis-proxy=1")  # fmt:skip # noqa: E501


@invoke.task
def dev_stop(context: invoke.Context, volumes: bool = False):
    command = "docker compose --env-file .prod.env down --remove-orphans"
    if volumes:
        command += " --volumes"

    context.run(command)
