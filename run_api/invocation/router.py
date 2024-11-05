"""
Invocations router.
"""

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import text, String
from sqlalchemy.ext.asyncio import AsyncSession
from run_api.user.schemas import User
from run_api.user.service import get_current_user
from run_api.database import get_db_session
from run_api.invocation.schemas import Report

router = APIRouter()

CHECK_EXISTS = text(
    "SELECT user_id, report_reason, to_char(date_trunc('week', started_at), 'IYYY_IW') AS table_suffix FROM invocations WHERE invocation_id = :invocation_id"
).columns(user_id=String, table_suffix=String, report_reason=String)
SAVE_REPORT = "UPDATE partitioned_invocations_{table_suffix} SET report_reason = :report_reason, reported_at = CURRENT_TIMESTAMP WHERE invocation_id = :invocation_id RETURNING reported_at"


@router.post("/{invocation_id}/report")
async def report_invocation(
    invocation_id: str,
    report: Report,
    db: AsyncSession = Depends(get_db_session),
    current_user: User = Depends(get_current_user()),
):
    result = await db.execute(CHECK_EXISTS, {"invocation_id": invocation_id})
    item = result.fetchone()
    user_id, existing_reason, table_suffix = None, None, None
    if item:
        user_id, existing_reason, table_suffix = item
    if not item or user_id != current_user.user_id:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Invocation not found, or does not belong to you",
        )
    if existing_reason is not None:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="A report has already been filed for this invocation",
        )
    result = await db.execute(
        text(SAVE_REPORT.format(table_suffix=table_suffix)),
        {
            "report_reason": report.reason,
            "invocation_id": invocation_id,
        },
    )
    await db.commit()
    reported_at = result.scalar()
    return {
        "status": f"report received for {invocation_id=} @ {reported_at}",
    }
