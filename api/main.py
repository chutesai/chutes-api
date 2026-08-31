"""
Main API entrypoint.
"""

import api.logging_bootstrap  # noqa: F401  # configures JSON logging before anything logs
import os
import re
import gc
import asyncio
import uuid

# import fickling
import hashlib
from loguru import logger
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request, APIRouter, HTTPException, status, Response
from fastapi.responses import ORJSONResponse
from sqlalchemy import select, text
import api.database.orms  # noqa: F401
from prometheus_client import (
    generate_latest,
    CollectorRegistry,
    multiprocess,
    CONTENT_TYPE_LATEST,
)
from prometheus_fastapi_instrumentator import Instrumentator
from concurrent.futures import ThreadPoolExecutor
from api.api_key.router import router as api_key_router
from api.chute.router import router as chute_router
from api.bounty.router import router as bounty_router
from api.image.router import router as image_router
from api.invocation.router import router as invocation_router
from api.invocation.router import conditional_invocation_router
from api.registry.router import router as registry_router
from api.user.router import router as user_router
from api.node.router import router as node_router
from api.instance.router import router as instance_router
from api.payment.router import router as payment_router
from api.miner.router import router as miner_router
from api.logo.router import router as logo_router
from api.job.router import router as jobs_router
from api.secret.router import router as secrets_router
from api.guesser import router as guess_router
from api.audit.router import router as audit_router
from api.server.router import router as servers_router
from api.misc.router import router as misc_router
from api.idp.router import router as idp_router
from api.e2e.router import router as e2e_router
from api.encrypted_logs.router import router as encrypted_logs_router
from api.chute_logs.router import router as chute_logs_router
from api.chute_logs.loki import LokiClient
from api.model_alias.router import router as model_alias_router
from api.external_backend.admin_router import router as external_admin_router
from api.external_backend.auth import external_auth_scope
from api.external_backend.polling import (
    start_external_operation_poller,
    stop_external_operation_poller,
)
from api.external_backend.realtime import (
    router as external_realtime_router,
    shutdown_realtime,
)
from api.external_backend.router import router as external_operation_router
from api.external_backend.schemas import ExternalOperation
from api.external_backend.service import shutdown_external_invocations
from api.chute.path_policy import is_reserved_canonical_chute_path
from api.chute.util import chute_id_by_slug, get_one
from api.database import Base, engine, get_session
from api.config import settings
from api.metrics.util import keep_gauges_fresh
from api.instance.util import start_instance_invalidation_listener
from api.log import install_asyncio_exception_handler


def _has_invocation_credentials(request: Request) -> bool:
    if (request.headers.get("authorization") or "").strip():
        return True
    return all(
        (request.headers.get(name) or "").strip()
        for name in ("x-chutes-hotkey", "x-chutes-signature", "x-chutes-nonce")
    )


async def loop_lag_monitor(interval: float = 0.1, warn_threshold: float = 0.2):
    """
    Very lightweight event-loop lag monitor.
    Produces *summary only* — no full stack traces.
    """
    loop = asyncio.get_running_loop()
    last = loop.time()

    ignored_task_str = (
        "aiohttp",
        "ClientSession",
        "ClientResponse",
        "TCPConnector",
    )

    def _should_ignore(task: asyncio.Task) -> bool:
        r = repr(task)
        return any(s in r for s in ignored_task_str)

    while True:
        await asyncio.sleep(interval)
        now = loop.time()
        lag = now - last - interval
        last = now

        if lag <= warn_threshold:
            continue

        ms = lag * 1000.0
        tasks = [
            t
            for t in asyncio.all_tasks(loop)
            if t is not asyncio.current_task(loop=loop) and not _should_ignore(t)
        ]

        # Group tasks by coroutine/function name (high-level signal)
        summary = {}
        for t in tasks:
            coro = t.get_coro()
            name = getattr(coro, "__qualname__", coro.__class__.__name__)
            summary.setdefault(name, 0)
            summary[name] += 1
        logger.warning(
            f"Event loop lag: {ms:.1f}ms, task summary during lag: {summary}"
        )


@asynccontextmanager
async def lifespan(_: FastAPI):
    """
    Execute all initialization/startup code, e.g. ensuring tables exist and such.
    """
    gc.set_threshold(5000, 50, 50)

    install_asyncio_exception_handler()

    loop = asyncio.get_event_loop()
    executor = ThreadPoolExecutor(max_workers=64)
    loop.set_default_executor(executor)

    asyncio.create_task(loop_lag_monitor())
    asyncio.create_task(keep_gauges_fresh())
    asyncio.create_task(start_instance_invalidation_listener())

    # Normal table creation stuff.
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    # dbmate migrations, make sure we only run them in a single process since we use workers > 1
    worker_pid_file = "/tmp/api.pid"
    is_migration_process = False
    try:
        if not os.path.exists(worker_pid_file):
            with open(worker_pid_file, "x") as outfile:
                outfile.write(str(os.getpid()))
            is_migration_process = True
        else:
            with open(worker_pid_file, "r") as infile:
                designated_pid = int(infile.read().strip())
            is_migration_process = os.getpid() == designated_pid
    except FileExistsError:
        with open(worker_pid_file, "r") as infile:
            designated_pid = int(infile.read().strip())
        is_migration_process = os.getpid() == designated_pid
    start_external_operation_poller()
    try:
        if not is_migration_process:
            yield
            return
        yield
    finally:
        try:
            await stop_external_operation_poller()
        finally:
            try:
                await shutdown_external_invocations()
            finally:
                try:
                    await shutdown_realtime()
                finally:
                    # Close the shared Loki connection pool on shutdown (no-op if never used).
                    await LokiClient.aclose()


app = FastAPI(default_response_class=ORJSONResponse, lifespan=lifespan)
os.makedirs("/tmp/prometheus_multiproc", exist_ok=True)
Instrumentator(
    should_instrument_requests_inprogress=True,
    inprogress_name="http_requests_inprogress",
    inprogress_labels=False,
).instrument(app)

default_router = APIRouter()
default_router.include_router(conditional_invocation_router)
default_router.include_router(user_router, prefix="/users", tags=["Users"])
default_router.include_router(chute_router, prefix="/chutes", tags=["Chutes"])
default_router.include_router(bounty_router, prefix="/bounties", tags=["Chutes"])
default_router.include_router(image_router, prefix="/images", tags=["Images"])
default_router.include_router(node_router, prefix="/nodes", tags=["Nodes"])
default_router.include_router(payment_router, tags=["Pricing", "Payments"])
default_router.include_router(instance_router, prefix="/instances", tags=["Instances"])
default_router.include_router(
    invocation_router, prefix="/invocations", tags=["Invocations"]
)
default_router.include_router(
    registry_router, prefix="/registry", tags=["Authentication"]
)
default_router.include_router(
    api_key_router, prefix="/api_keys", tags=["Authentication"]
)
default_router.include_router(miner_router, prefix="/miner", tags=["Miner"])
default_router.include_router(logo_router, prefix="/logos", tags=["Logo"])
default_router.include_router(guess_router, prefix="/guess", tags=["ConfigGuesser"])
default_router.include_router(audit_router, prefix="/audit", tags=["Audit"])
default_router.include_router(jobs_router, prefix="/jobs", tags=["Job"])
default_router.include_router(secrets_router, prefix="/secrets", tags=["Secret"])
default_router.include_router(misc_router, prefix="/misc", tags=["Miscellaneous"])
default_router.include_router(servers_router, prefix="/servers", tags=["Servers"])
default_router.include_router(idp_router, prefix="/idp", tags=["Identity Provider"])
default_router.include_router(e2e_router, prefix="/e2e", tags=["E2E Encryption"])
default_router.include_router(
    encrypted_logs_router, prefix="/encrypted_logs", tags=["Encrypted Logs"]
)
default_router.include_router(chute_logs_router, prefix="/logs", tags=["Logs"])
default_router.include_router(
    model_alias_router, prefix="/model_aliases", tags=["Model Aliases"]
)
default_router.include_router(
    external_admin_router, prefix="/external", tags=["External Backends"]
)
default_router.include_router(
    external_operation_router, prefix="/external", tags=["External Operations"]
)
default_router.include_router(external_realtime_router)


# Do not use app for this, else middleware picks it up
async def ping():
    try:
        async with get_session() as session:
            await session.execute(text("SELECT 1"))
            async with get_session(readonly=True) as ro_session:
                await ro_session.execute(text("SELECT 1"))
            return {"message": "pong"}
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Database connectivity problems: {str(e)}",
        )


# Prometheus metrics endpoint.
async def get_latest_metrics(request: Request):
    if request.state.has_resolved_ip:
        raise HTTPException(status_code=403, detail="Forbidden")
    registry = CollectorRegistry()
    multiprocess.MultiProcessCollector(registry)
    data = generate_latest(registry)
    return Response(data, media_type=CONTENT_TYPE_LATEST)


default_router.get("/ping")(ping)
default_router.get("/_metrics")(get_latest_metrics)


# OpenID Connect discovery endpoint at root level (standard location)
@default_router.get("/.well-known/openid-configuration")
async def openid_configuration_root(request: Request):
    """
    OpenID Connect Discovery endpoint.
    """
    from api.idp.schemas import get_available_scopes

    idp_base = f"https://api.{settings.base_domain}/idp"

    return {
        "issuer": f"https://api.{settings.base_domain}",
        "authorization_endpoint": f"{idp_base}/authorize",
        "token_endpoint": f"{idp_base}/token",
        "userinfo_endpoint": f"{idp_base}/userinfo",
        "revocation_endpoint": f"{idp_base}/token/revoke",
        "introspection_endpoint": f"{idp_base}/token/introspect",
        "scopes_supported": list(get_available_scopes().keys()),
        "response_types_supported": ["code"],
        "response_modes_supported": ["query"],
        "grant_types_supported": ["authorization_code", "refresh_token"],
        "token_endpoint_auth_methods_supported": [
            "client_secret_post",
            "client_secret_basic",
            "none",
        ],
        "code_challenge_methods_supported": ["plain", "S256"],
        "service_documentation": "https://docs.chutes.ai/oauth",
        "subject_types_supported": ["public"],
        "claims_supported": [
            "sub",
            "username",
            "created_at",
        ],
    }


app.include_router(default_router)

# Pickle safety checks.
# fickling.always_check_safety()


class _RequestBodyTooLarge(Exception):
    pass


def _declared_request_body_size(request: Request) -> int | None:
    """Parse one unambiguous Content-Length value before reading the body."""

    values = [
        value
        for name, value in request.scope.get("headers", ())
        if name.lower() == b"content-length"
    ]
    if not values:
        return None
    if len(values) != 1:
        raise ValueError("Multiple Content-Length headers are not accepted.")
    try:
        decoded = values[0].decode("ascii")
    except UnicodeDecodeError as exc:
        raise ValueError("Content-Length must be an ASCII integer.") from exc
    if not decoded.isdigit() or len(decoded) > 20:
        raise ValueError("Content-Length must be a non-negative integer.")
    return int(decoded)


async def _read_bounded_request_body(request: Request, limit: int) -> bytes:
    """Read at most ``limit`` bytes and preserve the body for downstream handlers."""

    body = bytearray()
    async for chunk in request.stream():
        if len(chunk) > limit - len(body):
            raise _RequestBodyTooLarge
        body.extend(chunk)
    buffered = bytes(body)
    # Starlette's BaseHTTPMiddleware passes this cached body through call_next.
    request._body = buffered
    return buffered


def _body_limit_response() -> ORJSONResponse:
    return ORJSONResponse(
        status_code=status.HTTP_413_CONTENT_TOO_LARGE,
        content={"detail": "Request body is too large."},
        headers={"Connection": "close"},
    )


@app.middleware("http")
async def host_router_middleware(request: Request, call_next):
    """
    Route differentiation for hostname-based simple invocations.
    """
    try:
        declared_size = _declared_request_body_size(request)
    except ValueError as exc:
        return ORJSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={"detail": str(exc)},
            headers={"Connection": "close"},
        )
    if declared_size is not None and declared_size > settings.max_request_body_bytes:
        return _body_limit_response()

    # Buffer through one bounded reader for every HTTP method. This covers
    # chunked bodies without Content-Length, including data-driven GET/HEAD
    # endpoints, before any route-specific parser can allocate without a cap.
    try:
        body = await _read_bounded_request_body(
            request, settings.max_request_body_bytes
        )
    except _RequestBodyTooLarge:
        return _body_limit_response()

    # Calculate request body integrity hashes for miner/signed requests.
    if request.method in ["POST", "PUT", "PATCH", "DELETE"]:
        sha256_hash = hashlib.sha256(body).hexdigest()
        request.state.body_sha256 = sha256_hash
    else:
        request.state.body_sha256 = None
    resolved_ip = (request.headers.get("X-Resolved-IP") or "").strip()
    request.state.client_ip = resolved_ip or request.client.host
    request.state.has_resolved_ip = bool(resolved_ip)
    request.state.chute_id = None
    request.state.free_invocation = False
    request.state.invocation_dispatch = False
    host = request.headers.get("host", "")
    host_parts = re.search(r"^([a-z0-9-]+)\.[a-z0-9-]+", host.lower())

    # Preserve the API health endpoint while allowing an actual Chute hostname
    # to declare the same Cord path.
    if request.url.path == "/ping" and (not host_parts or host_parts.group(1) == "api"):
        return await call_next(request)

    # Slug overrides, if any.
    slug = host_parts.group(1).lower() if host_parts else None
    if slug == "chutes-qwen-qwen3-embedding-8b":
        slug = "chutes-qwen-qwen3-embedding-8b-tee"

    # MEGALLM
    if (
        host_parts
        and host_parts.group(1) == "llm"
        and (request.method.lower() == "post" or request.url.path == "/v1/models")
    ):
        request.state.chute_id = "__megallm__"
        request.state.auth_method = "invoke"
        request.state.auth_object_type = "chutes"
        request.state.auth_object_id = "__megallm__"
        request.state.invocation_dispatch = True

    # MEGAEMBED
    elif (
        host_parts
        and host_parts.group(1) == "embed"
        and request.method.lower() == "post"
    ):
        request.state.chute_id = "__megaembed__"
        request.state.auth_method = "invoke"
        request.state.auth_object_type = "chutes"
        request.state.auth_object_id = "__megaembed__"
        request.state.invocation_dispatch = True

    # MEGADIFFUSER
    elif (
        host_parts
        and host_parts.group(1) == "image"
        and request.method.lower() == "post"
    ):
        request.state.chute_id = "__megadiffuser__"
        request.state.auth_method = "invoke"
        request.state.auth_object_type = "chutes"
        request.state.auth_object_id = "__megadiffuser__"
        request.state.invocation_dispatch = True

    # Hostname based router.
    elif (
        host_parts
        and host_parts.group(1) != "api"
        and (chute_id := await chute_id_by_slug(slug))
    ):
        if request.url.path == "/ping":
            candidate = await get_one(chute_id)
            declares_ping = bool(
                candidate
                and candidate.execution_backend == "external"
                and any(
                    cord.get("public_api_path") == "/ping"
                    and str(cord.get("public_api_method", "POST")).upper()
                    == request.method.upper()
                    for cord in candidate.cords or []
                )
            )
            if not declares_ping:
                request.state.auth_method = "read"
                return await call_next(request)
        request.state.chute_id = chute_id
        request.state.auth_method = "invoke"
        request.state.auth_object_type = "chutes"
        request.state.auth_object_id = chute_id
        request.state.invocation_dispatch = True

    # Normal router.
    else:
        request.state.auth_method = "read"
        if request.method.lower() in ("post", "put", "patch"):
            request.state.auth_method = "write"
        elif request.method.lower() == "delete":
            request.state.auth_method = "delete"

        # External Cords may expose any supported HTTP method through either the
        # Chute hostname or the canonical API-domain Chute URL. Canonical evidence
        # and model-info paths remain reserved management endpoints; otherwise only
        # swap routers after confirming the suffix is an actual Cord.
        inv_match = re.match(r"^/chutes/([^/]+)/(.+)$", request.url.path, re.I)
        if inv_match and request.method.upper() in {
            "GET",
            "POST",
            "PUT",
            "PATCH",
            "DELETE",
            "HEAD",
        }:
            raw_chute_id = inv_match.group(1)
            try:
                # Canonical Cord URLs are ID-addressed. Restricting this lookup
                # prevents a Chute name such as ``code`` from shadowing the
                # management route with the same first path segment.
                chute_id = str(uuid.UUID(raw_chute_id))
            except ValueError:
                chute_id = None
            public_path = f"/{inv_match.group(2)}"
            if (
                chute_id
                and not is_reserved_canonical_chute_path(public_path)
                and not _has_invocation_credentials(request)
            ):
                # Avoid resolving a private Chute or Cord before authentication.
                # Unknown and declared canonical invocation paths intentionally
                # share one generic response, including method and body shape.
                return ORJSONResponse(
                    status_code=status.HTTP_404_NOT_FOUND,
                    content={"detail": "Not Found"},
                )
            candidate = await get_one(chute_id) if chute_id else None
            declares_cord = bool(
                candidate
                and not is_reserved_canonical_chute_path(public_path)
                and any(
                    cord.get("public_api_path") == public_path
                    and str(cord.get("public_api_method", "POST")).upper()
                    == request.method.upper()
                    for cord in candidate.cords or []
                )
            )
            if (
                chute_id
                and not is_reserved_canonical_chute_path(public_path)
                and _has_invocation_credentials(request)
            ):
                # Classify authenticated-form canonical requests uniformly. Auth
                # runs before the handler checks this private declaration bit, so
                # an invalid credential cannot distinguish a known Cord from an
                # unknown one by 401-vs-404 behavior.
                request.state.auth_method = "invoke"
                request.state.chute_id = chute_id
                request.state.auth_object_id = chute_id
                request.state.auth_object_type = "chutes"
                request.state.invocation_public_path = public_path
                request.state.canonical_cord_declared = declares_cord
                request.state.invocation_dispatch = True

        # E2E endpoints are chute invocations for OAuth scope purposes.
        if request.state.auth_method != "invoke":
            if request.url.path.startswith("/e2e/instances/"):
                chute_id = request.url.path.split("/")[3]
                request.state.auth_method = "invoke"
                request.state.chute_id = chute_id
                request.state.auth_object_id = chute_id
                request.state.auth_object_type = "chutes"
            elif request.method.lower() == "post" and request.url.path == "/e2e/invoke":
                chute_id = request.headers.get("x-chute-id") or "__list_or_invalid__"
                request.state.auth_method = "invoke"
                request.state.chute_id = chute_id
                request.state.auth_object_id = chute_id
                request.state.auth_object_type = "chutes"

        if request.state.auth_method != "invoke":
            external_scope = external_auth_scope(request.url.path)
            if external_scope is not None:
                request.state.auth_object_type = external_scope.object_type
                request.state.auth_object_id = external_scope.object_id
                if (
                    external_scope.object_type == "invocations"
                    and external_scope.object_id != "__list_or_invalid__"
                ):
                    async with get_session(readonly=True) as session:
                        operation_chute_id = (
                            await session.execute(
                                select(ExternalOperation.chute_id).where(
                                    ExternalOperation.operation_id
                                    == external_scope.object_id
                                )
                            )
                        ).scalar_one_or_none()
                    if operation_chute_id:
                        # A key that submitted an async invocation must also be able
                        # to follow its status/artifact URL. Endpoint ownership checks
                        # still bind the operation to the authenticated user.
                        request.state.auth_alternative_scopes = (
                            ("chutes", operation_chute_id, "invoke"),
                        )
            # Handle /users/me/* paths specially for OAuth scope checking
            elif request.url.path.startswith("/users/me"):
                if "/balance" in request.url.path:
                    request.state.auth_object_type = "billing"
                elif "/quota" in request.url.path:
                    request.state.auth_object_type = "account"
                else:
                    request.state.auth_object_type = "account"
                request.state.auth_object_id = "__self__"
            else:
                request.state.auth_object_type = request.url.path.split("/")[-1]
                # XXX at some point, perhaps we can support objects by name too, but for
                # now, for auth to work (easily) we just need to only support UUIDs when
                # using API keys.
                path_match = re.match(r"^/[^/]+/([^/]+)$", request.url.path)
                if path_match:
                    request.state.auth_object_id = path_match.group(1)
                else:
                    request.state.auth_object_id = "__list_or_invalid__"
    return await call_next(request)
