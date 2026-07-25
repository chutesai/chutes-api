"""
Registry authentication.
"""

from typing import Optional
from fastapi import Request, APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from cryptography.x509 import Certificate
from loguru import logger

from api.config import settings
from api.constants import HOTKEY_HEADER, SIGNATURE_HEADER, NONCE_HEADER
from api.database import get_db_session
from api.user.service import get_current_user
from api.server.service import lookup_server_by_ip
from api.server.util import (
    verify_server_cert,
    require_registry_proxy_secret,
    extract_optional_client_cert,
)
from api.server.exceptions import AttestationError
from api.log import update_log_context
from api.util import semcomp


router = APIRouter()


@router.get("/auth")
async def registry_auth(
    request: Request,
    db: AsyncSession = Depends(get_db_session),
    _proxy=Depends(require_registry_proxy_secret()),
    client_cert: Optional[Certificate] = Depends(extract_optional_client_cert()),
):
    """
    Authenticate registry/docker pull requests.

    In the Depot proxy flow this endpoint is called by nginx auth_request to
    verify the miner before nginx proxies to Depot's token endpoint; it only
    needs to return 200 on success (or 401 on failure).

    The VM's attested measurement version decides which auth is required:

    - version >= registry_mtls_min_version: mTLS.  The nginx frontend forwards the
      client cert as X-Client-Cert; the leaf is verified against the CA registered
      for the VM.  No legacy fallback — a VM this new must use mTLS.
    - otherwise (older VM, or unknown IP): legacy Bittensor hotkey/signature/nonce.

    Setting registry_mtls_min_version to "0.0.0" forces every attested VM onto
    mTLS — the kill switch that closes the legacy "any registered miner can pull
    any private chute" hole once the fleet is fully migrated.

    The require_registry_proxy_secret dependency, when REGISTRY_PROXY_SECRET is configured,
    rejects requests that do not carry X-Registry-Proxy-Auth matching the secret,
    preventing X-Client-Cert / X-Real-IP spoofing on connections that bypass the
    registry proxy.  The mTLS client cert is extracted (typed) by extract_optional_client_cert.
    """
    # Resolve the VM at the source IP; its attested version dictates the auth path.
    client_ip = request.headers.get("X-Real-IP") or (
        request.client.host if request.client else None
    )
    server = await lookup_server_by_ip(db, client_ip) if client_ip else None

    # Bind request identity into the ambient log context (release/next's mechanism), so every
    # log line for this pull -- across the auth path and deep raise sites -- is filterable by
    # ip / server_id / server_name / miner_hotkey. This router has no vm_name path param, so
    # the server identity comes from the IP lookup above.
    update_log_context(
        ip=client_ip,
        server_id=getattr(server, "server_id", None),
        server_name=getattr(server, "name", None),
        miner_hotkey=request.headers.get(HOTKEY_HEADER),
    )

    if (
        server is not None
        and server.version
        and semcomp(server.version, settings.registry_mtls_min_version) >= 0
    ):
        _verify_mtls_client_cert(client_cert, server, client_ip)
    else:
        await _legacy_registry_auth(request)

    # nginx auth_request only needs the status code; a 200 authorizes the pull.
    return {"authenticated": True}


def _verify_mtls_client_cert(
    client_cert: Optional[Certificate], server, client_ip: str | None
) -> None:
    """Require a valid mTLS client cert for VMs at/above registry_mtls_min_version.

    Delegates to the shared verify_server_cert: the client leaf (extracted from X-Client-Cert
    by extract_optional_client_cert) is verified against the CA the VM recorded via
    POST /servers/{vm_name}/provision, adding registry-context logging.

    Raises NoClientCertError (403) if no client cert is presented or the VM has no CA on file,
    or InvalidClientCertError (403) if the leaf fails verification (both are AttestationError).
    """
    try:
        verify_server_cert(client_cert, server)
    except AttestationError as e:
        logger.warning(
            f"registry: mTLS rejected for VM at {client_ip} "
            f"(server={server.name}, version={server.version})"
        )
        raise HTTPException(status_code=e.http_status, detail=e.message)
    logger.info(f"registry: mTLS auth accepted for VM at {client_ip} (server={server.name})")


async def _legacy_registry_auth(request: Request) -> None:
    """
    Perform legacy Bittensor SS58 hotkey/signature/nonce auth for registry pulls.

    Manually extracts auth headers from the request and calls the authenticator so
    that we can invoke it outside of FastAPI's dependency injection (it runs only on
    the legacy branch, not unconditionally as a route dependency).

    Raises HTTPException(401) if the credentials are absent or invalid.
    """
    authenticator = get_current_user(purpose="registry", registered_to=settings.netuid)
    hotkey = request.headers.get(HOTKEY_HEADER)
    signature = request.headers.get(SIGNATURE_HEADER)
    nonce = request.headers.get(NONCE_HEADER)
    authorization = request.headers.get("Authorization")
    await authenticator(
        request=request,
        api_key=None,
        hotkey=hotkey,
        signature=signature,
        nonce=nonce,
        authorization=authorization,
    )
