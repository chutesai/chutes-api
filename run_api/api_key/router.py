"""
API keys router.
"""

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
from run_api.api_key.schemas import APIKeyArgs, APIKey
from run_api.user.schemas import User
from run_api.user.service import get_current_user
from run_api.database import get_db_session
from run_api.config import settings

router = APIRouter()


@router.post("/")
async def create_api_key(
    key_args: APIKeyArgs,
    db: AsyncSession = Depends(get_db_session),
    current_user: User = Depends(get_current_user()),
):
    """
    Create a new API key.
    """
    api_key, one_time_secret = APIKey.create(key_args)
    db.add(api_key)
    await db.commit()
    return api_key
