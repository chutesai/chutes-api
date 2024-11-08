"""
Main API entrypoint.
"""

import re
import asyncio
import fickling
import hashlib
from loguru import logger
from fastapi import FastAPI, Request, APIRouter
from fastapi.responses import ORJSONResponse
from api.api_key.schemas import APIKey  # noqa: F401
from api.node.schemas import Node  # noqa: F401
from api.challenge.schemas import Challenge  # noqa: F401
from api.api_key.router import router as api_key_router
from api.chute.router import router as chute_router
from api.bounty.router import router as bounty_router
from api.image.router import router as image_router
from api.invocation.router import router as invocation_router
from api.invocation.router import host_invocation_router
from api.registry.router import router as registry_router
from api.user.router import router as user_router
from api.node.router import router as node_router
from api.instance.schemas import Instance  # noqa: F401
from api.chute.util import chute_id_by_slug
from api.database import Base, engine
from api.config import settings
import api.chute.events  # noqa: F401
import api.image.events  # noqa: F401
import api.user.events  # noqa: F401
import api.node.events  # noqa: F401

app = FastAPI(default_response_class=ORJSONResponse)

default_router = APIRouter()
default_router.include_router(user_router, prefix="/users", tags=["Users"])
default_router.include_router(chute_router, prefix="/chutes", tags=["Chutes"])
default_router.include_router(bounty_router, prefix="/bounties", tags=["Chutes"])
default_router.include_router(image_router, prefix="/images", tags=["Images"])
default_router.include_router(node_router, prefix="/nodes", tags=["Nodes"])
default_router.include_router(
    invocation_router, prefix="/invocations", tags=["Invocations"]
)
default_router.include_router(
    registry_router, prefix="/registry", tags=["Authentication"]
)
default_router.include_router(
    api_key_router, prefix="/api_keys", tags=["Authentication"]
)
app.include_router(default_router)
app.include_router(host_invocation_router)

# Pickle safety checks.
fickling.always_check_safety()


@app.middleware("http")
async def host_router_middleware(request: Request, call_next):
    """
    Route differentiation for hostname-based simple invocations.
    """
    request.state.chute_id = None
    host = request.headers.get("host", "")
    host_parts = re.search(r"^([a-z0-9-]+)\.[a-z0-9-]+", host)
    if host_parts and (chute_id := await chute_id_by_slug(host_parts.group(1).lower())):
        request.state.chute_id = chute_id
        request.state.auth_method = "invoke"
        request.state.auth_object_type = "chutes"
        request.state.auth_object_id = chute_id
        app.router = host_invocation_router
    else:
        request.state.auth_method = "read"
        if request.method.lower() in ("post", "put", "patch"):
            request.state.auth_method = "write"
        elif request.method.lower() == "delete":
            request.state.auth_method = "delete"
        request.state.auth_object_type = request.url.path.split("/")[1]
        # XXX at some point, perhaps we can support objects by name too, but for
        # now, for auth to work (easily) we just need to only support UUIDs when
        # using API keys.
        path_match = re.match(r"^/[^/]+/([^/]+)$", request.url.path)
        if path_match:
            request.state.auth_object_id = path_match.group(1)
        else:
            request.state.auth_object_id = "__list_or_invalid__"
        app.router = default_router
    return await call_next(request)


@app.middleware("http")
async def request_body_checksum(request: Request, call_next):
    if request.method in ["POST", "PUT", "PATCH"]:
        body = await request.body()
        sha256_hash = hashlib.sha256(body).hexdigest()
        request.state.body_sha256 = sha256_hash
    else:
        request.state.body_sha256 = None
    return await call_next(request)


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
        "api/migrations",
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

    await asyncio.gather(
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
