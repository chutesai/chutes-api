"""
Discord webhook alerts.

Best-effort and never fatal: a notification failure must not fail the work that triggered it.
"""

import aiohttp
from loguru import logger

from api.config import settings

# Discord rejects a message body over 2000 characters.
MAX_CONTENT = 1900


async def send_discord_alert(content: str) -> bool:
    """Post to the configured Discord webhook. Returns False when unconfigured or failed."""
    webhook = settings.discord_webhook_url
    if not webhook:
        logger.info(f"No DISCORD_WEBHOOK_URL configured, skipping alert: {content}")
        return False
    try:
        async with (
            aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=10)) as session,
            session.post(webhook, json={"content": content[:MAX_CONTENT]}) as response,
        ):
            if response.status >= 400:
                logger.warning(
                    f"Discord webhook returned {response.status}: {await response.text()}"
                )
                return False
        return True
    except Exception as exc:
        logger.warning(f"Failed to send Discord alert: {exc}")
        return False
