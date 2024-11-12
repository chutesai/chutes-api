"""
Registry authentication.
"""

import pybase64 as base64
from fastapi import Request, Response, APIRouter, Depends
from api.user.schemas import User
from api.user.service import get_current_user
from api.config import settings


router = APIRouter()


@router.get("/auth")
async def registry_auth(
    request: Request,
    response: Response,
    current_user: User = Depends(get_current_user(purpose="registry", registered_to=settings.netuid)),
):
    """
    Authentication registry/docker pull requests.
    """
    auth_string = base64.b64encode(f":{settings.registry_password}".encode())
    response.headers["Authorization"] = f"Basic {auth_string}"
    return {"authenticated": True}
