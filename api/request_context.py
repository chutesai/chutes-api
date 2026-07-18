"""
Per-router request-identity binding for structured logging.

`bind_request_context` is a FastAPI dependency: attach it at the router level

    router = APIRouter(dependencies=[Depends(bind_request_context)])

and every log line emitted while handling a request to that router -- including bare
`logger.*` calls at deep raise sites -- carries the request's identity (ip, miner hotkey,
and any identity-like path params), with nothing threaded through call arguments.

Why a dependency and not app middleware: a dependency runs in the endpoint's own context,
so the contextvar bind reliably reaches the endpoint and everything it calls. It is also
scoped per-router (opt-in) rather than global. Identity known only after a lookup
(e.g. server_id resolved from a vm_name, instance_id from a config) is added deeper via
`api.log.update_log_context()`.
"""

from fastapi import Request

from api.constants import HOTKEY_HEADER
from api.log import update_log_context

# Route path params that identify the subject of a request. request.path_params only holds
# the matched route's real params at runtime, so this simply binds whichever are present.
_IDENTITY_PATH_PARAMS = (
    "server_id",
    "server_name_or_id",
    "vm_name",
    "instance_id",
    "config_id",
    "chute_id",
    "node_id",
)


async def bind_request_context(request: Request) -> None:
    """Bind this request's ambient identity into the logging context (see module docstring).

    Purely observability -- never raises or rejects, so it is safe on anonymous/pre-registration
    routes. Missing values are dropped by update_log_context.
    """
    fields = {
        "ip": getattr(request.state, "client_ip", None),
        "miner_hotkey": request.headers.get(HOTKEY_HEADER),
    }
    for key in _IDENTITY_PATH_PARAMS:
        value = request.path_params.get(key)
        if value is not None:
            fields[key] = value
    update_log_context(**fields)
