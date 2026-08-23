"""
Host profile lifecycle: notify on new submissions, reconcile against the measurement config.

Runs on a schedule. Deliberately kept out of the generation CLI so that side stays storage-agnostic
-- the API owns the profile store:

  * notify    -- a new host class was submitted; alert so someone can generate for it. Generation
                 is NEVER triggered automatically; a human decides.
  * reconcile -- the measurement config is the source of truth. Once a fingerprint is published
                 there, its profile is marked measured.

Nothing here deletes: every profile is retained, measured or not. A submission nobody generated for
is still the record that someone asked, and a fingerprint cannot be inverted back to the topology
inputs, so a row is the only copy of them. Both are idempotent.
"""

import asyncio

from loguru import logger
from sqlalchemy import update
from sqlalchemy.sql import func

import api.database.orms  # noqa
from api.database import get_session
from api.notify import send_discord_alert
from api.server.schemas import HostProfileRecord
from api.server.util import list_pending_profiles, reconcile_host_profiles


async def notify_new_submissions(db) -> list[str]:
    """Alert once per newly submitted host class."""
    notified = []
    for record in await list_pending_profiles(db):
        if record.notified_at:
            continue
        gpu = (record.profile or {}).get("gpu") or {}
        sent = await send_discord_alert(
            f"**New host profile submitted**\n"
            f"Fingerprint: `{record.fingerprint}`\n"
            f"GPUs: {gpu.get('count')}x {', '.join(gpu.get('pci_device_ids') or []) or 'unknown'}\n"
            f"Awaiting measurement generation."
        )
        # Stamp only once it actually went out, so a webhook outage re-alerts next run.
        if sent:
            await db.execute(
                update(HostProfileRecord)
                .where(HostProfileRecord.fingerprint == record.fingerprint)
                .values(notified_at=func.now())
            )
            await db.commit()
            notified.append(record.fingerprint)
    return notified


async def main() -> None:
    async with get_session() as db:
        notified = await notify_new_submissions(db)
        logger.info(f"Host profile notify: alerted on {len(notified)} new submission(s)")

        promoted = await reconcile_host_profiles(db)
        logger.info(f"Host profile reconcile: marked {len(promoted)} profile(s) measured")


if __name__ == "__main__":
    asyncio.run(main())
