"""
Sync the metagraph to the database, broadcast any updated nodes.
"""

import hashlib
import json
import asyncio
import redis
from sqlalchemy import text
from sqlalchemy.dialects.postgresql import insert
from fiber.chain.interface import get_substrate
from fiber.chain.fetch_nodes import get_nodes_for_netuid
from fiber.logging_utils import get_logger
from metasync.database import engine, Base, SessionLocal
from metasync.shared import create_metagraph_node_class
from metasync.config import settings

MetagraphNode = create_metagraph_node_class(Base)
logger = get_logger(__name__)


async def sync_and_save_metagraph(substrate, redis_client):
    """
    Load the metagraph for our subnet and persist it to the database.
    """
    nodes = get_nodes_for_netuid(substrate, settings.netuid)
    if not nodes:
        raise Exception("Failed to load metagraph nodes!")
    updated = 0
    async with SessionLocal() as session:
        hotkeys = ", ".join([f"'{node.hotkey}'" for node in nodes])
        result = await session.execute(
            text(
                f"DELETE FROM metagraph_nodes WHERE netuid = :netuid AND hotkey NOT IN ({hotkeys}) AND node_id >= 0"
            ),
            {
                "netuid": settings.netuid,
            },
        )
        for node in nodes:
            node_dict = node.dict()
            node_dict.pop("last_updated", None)
            node_dict["checksum"] = hashlib.sha256(json.dumps(node_dict).encode()).hexdigest()
            statement = insert(MetagraphNode).values(node_dict)
            statement = statement.on_conflict_do_update(
                index_elements=["hotkey"],
                set_={key: getattr(statement.excluded, key) for key, value in node_dict.items()},
                where=MetagraphNode.checksum != node_dict["checksum"],
            )
            result = await session.execute(statement)
            if result.rowcount > 0:
                logger.info(f"Detected metagraph update for {node.hotkey=}")
                redis_client.publish(f"metagraph_change:{settings.netuid}", json.dumps(node_dict))
                updated += 1
        if updated:
            logger.info(f"Updated {updated} nodes for netuid={settings.netuid}")
        else:
            logger.info(f"No metagraph changes detected for netuid={settings.netuid}")
        await session.commit()


async def main():
    """
    Main.
    """
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    redis_client = redis.Redis.from_url(settings.redis_url)
    substrate = get_substrate(subtensor_address=settings.subtensor)
    logger.info("Attempting to resync metagraph...")
    await asyncio.wait_for(sync_and_save_metagraph(substrate, redis_client), 60)


if __name__ == "__main__":
    asyncio.run(main())
