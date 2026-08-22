"""
Host profile bucket lifecycle: notify on new submissions, reconcile against the measurement config.

Runs on a schedule. Two jobs, deliberately kept out of the generation CLI so that side stays
storage-agnostic -- the API owns the bucket:

  * notify    -- a new host class landed under pending/; alert so someone can generate for it.
                 Generation is NEVER triggered automatically; a human decides.
  * reconcile -- the measurement config is the source of truth. Once a fingerprint is published
                 there, its profile belongs in the retained set, so move pending/ -> measured/.

Both are idempotent: notifications are deduplicated in redis, and promotion no-ops on a
fingerprint that has already moved.
"""

import asyncio

from loguru import logger

import api.database.orms  # noqa
from api.config import settings
from api.notify import send_discord_alert
from api.server.util import list_pending_fingerprints, reconcile_host_profiles

# Fingerprints already alerted on. Regenerable state -- losing it re-alerts, it never re-generates.
NOTIFIED_KEY = "host_profile:notified"


async def notify_new_submissions() -> list[str]:
    """Alert once per newly submitted host class."""
    pending = await list_pending_fingerprints()
    if not pending:
        return []

    new = [fp for fp in pending if not await settings.redis_client.sismember(NOTIFIED_KEY, fp)]
    for fingerprint in new:
        sent = await send_discord_alert(
            f"**New host profile submitted**\n"
            f"Fingerprint: `{fingerprint}`\n"
            f"Awaiting measurement generation. Fetch it from "
            f"`{settings.host_profile_prefix}/pending/{fingerprint}.json`, generate, then add the "
            f"hardware entry with this fingerprint to the measurement config."
        )
        # Only mark once it actually went out, so a webhook outage re-alerts next run.
        if sent:
            await settings.redis_client.sadd(NOTIFIED_KEY, fingerprint)
    return new


async def main() -> None:
    new = await notify_new_submissions()
    logger.info(f"Host profile notify: {len(new)} new submission(s)")

    promoted = await reconcile_host_profiles()
    logger.info(f"Host profile reconcile: promoted {len(promoted)} profile(s) to measured/")


if __name__ == "__main__":
    asyncio.run(main())
