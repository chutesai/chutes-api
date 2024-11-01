"""
Main API entrypoint.
"""

import asyncio
import fickling
from loguru import logger
from fastapi import FastAPI
from fastapi.responses import ORJSONResponse
from run_api.chute.router import router as chute_router
from run_api.image.router import router as image_router
from run_api.instance.schemas import Instance  # noqa: F401
from run_api.database import Base, engine
from run_api.config import settings
import run_api.chute.events  # noqa: F401
import run_api.image.events  # noqa: F401

app = FastAPI(default_response_class=ORJSONResponse)

app.include_router(chute_router, prefix="/chutes", tags=["Chutes"])
app.include_router(image_router, prefix="/images", tags=["Images"])

fickling.always_check_safety()


@app.on_event("startup")
async def startup():
    """
    Execute all initialization/startup code, e.g. ensuring tables exist and such.
    """
    # Normal table creation stuff.
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    # Manual DB migrations.
    process = await asyncio.create_subprocess_exec(
        "dbmate",
        "--url",
        settings.sqlalchemy.replace("+asyncpg", "") + "?sslmode=disable",
        "--migrations-dir",
        "run_api/migrations",
        "migrate",
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )

    async def log_migrations(stream, name):
        log_method = logger.info if name == "stdout" else logger.warning
        while True:
            line = await stream.readline()
            if line:
                decoded_line = line.decode().strip()
                log_method(decoded_line)
            else:
                break

    asyncio.gather(
        log_migrations(process.stdout, "stdout"),
        log_migrations(process.stderr, "stderr"),
        process.wait(),
    )
    if process.returncode == 0:
        logger.success("successfull applied all DB migrations")
    else:
        logger.error(f"failed to run db migrations returncode={process.returncode}")

    # Buckets.
    if not await settings.storage_client.bucket_exists(settings.storage_bucket):
        await settings.storage_client.make_bucket(settings.storage_bucket)
