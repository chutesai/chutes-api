"""
Audit router.
"""

import io
import uuid
from datetime import datetime
from loguru import logger
from fastapi import APIRouter, Depends, Request, Header
from sqlalchemy.ext.asyncio import AsyncSession
from api.constants import HOTKEY_HEADER
from api.audit.schemas import AuditEntry
from api.user.schemas import User
from api.user.service import get_current_user
from api.database import get_db_session
from api.config import settings

router = APIRouter()


@router.post("/miner_data")
async def add_miner_audit_data(
    request: Request,
    hotkey: str | None = Header(None, alias=HOTKEY_HEADER),
    block: str | None = Header(None, alias="X-Chutes-Block"),
    _: User = Depends(get_current_user(registered_to=settings.netuid)),
    db: AsyncSession = Depends(get_db_session),
):
    destination = f"audit/miner/{hotkey}/{block}.json"
    entry_id = str(uuid.uuid5(uuid.NAMESPACE_OID, f"{hotkey}.{block}"))
    async with settings.s3_client() as s3:
        await s3.upload_fileobj(
            io.BytesIO(await request.body()),
            settings.storage_bucket,
            destination,
            ExtraArgs={"ContentType": "application/json"},
        )
    logger.success(
        f"Uploaded {hotkey} audit data commited on block {block} to {settings.storage_bucket}/{destination}"
    )

    # Add to DB.
    payload = await request.json()
    audit_entry = AuditEntry(
        entry_id=entry_id,
        hotkey=hotkey,
        block=int(block),
        path=destination,
        start_time=datetime.fromisoformat(payload["start_time"]).replace(tzinfo=None),
        end_time=datetime.fromisoformat(payload["end_time"]).replace(tzinfo=None),
    )
    db.add(audit_entry)
    await db.commit()
    await db.refresh(audit_entry)
    return audit_entry
