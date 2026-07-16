"""
Registry authentication.
"""

from fastapi import Request, Response, APIRouter, Depends
from api.user.schemas import User
from api.user.service import get_current_user
from api.config import settings


router = APIRouter()


@router.get("/auth")
async def registry_auth(
    request: Request,
    response: Response,
    current_user: User = Depends(
        get_current_user(purpose="registry", registered_to=settings.netuid)
    ),
):
    """
    Authenticate registry/docker pull requests.

    In the Depot proxy flow, this endpoint is called by nginx auth_request
    to verify the miner's hotkey. It only needs to return 200 on success
    (or 401 on failure via the dependency). The actual registry auth token
    is obtained by nginx proxying to Depot's token endpoint.
    """
    return {"authenticated": True}
