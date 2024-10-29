"""
User logic/code.
"""
from fastapi import Depends, Header, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from database import get_db_session
from user.schemas import User


async def get_current_user(
    user_id: str = Header(..., alias="X-Parachutes-UserID"),
    authorization: str = Header(..., alias="Authorization"),
    db: AsyncSession = Depends(get_db_session),
) -> User:
    """
    Load the current user from the database.
    """
    token = authorization.split(" ")[1] if " " in authorization else None
    if not token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing authorization token",
        )
    query = select(User).where(User.user_id == user_id)
    result = await db.execute(query)
    user = result.scalar_one_or_none()
    if not user or not user.verify_api_key(token):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token or user not found",
        )
    return user
