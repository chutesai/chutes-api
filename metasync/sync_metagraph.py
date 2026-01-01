"""
Sync the metagraph to the database, broadcast any updated nodes.
"""

import os
import hashlib
import json
import asyncio
from sqlalchemy import text
from sqlalchemy.dialects.postgresql import insert
from async_substrate_interface import AsyncSubstrateInterface
from scalecodec.utils.ss58 import ss58_encode
from loguru import logger
from metasync.database import engine, Base, SessionLocal
from metasync.shared import create_metagraph_node_class
from metasync.config import settings

MetagraphNode = create_metagraph_node_class(Base)
SS58_FORMAT = 42


def _ss58_encode(address, ss58_format: int = SS58_FORMAT) -> str:
    """Convert address bytes to SS58 string."""
    if isinstance(address, str):
        return address
    if isinstance(address, (list, tuple)) and len(address) > 0:
        if not isinstance(address[0], int):
            address = address[0]
    return ss58_encode(bytes(address).hex(), ss58_format)


async def get_nodes_for_netuid(substrate: AsyncSubstrateInterface, netuid: int) -> list[dict]:
    """
    Fetch all nodes for a given netuid using the SubnetInfoRuntimeApi.
    """
    response = await substrate.runtime_call(
        api="SubnetInfoRuntimeApi",
        method="get_metagraph",
        params=[netuid],
    )
    metagraph = response if isinstance(response, dict) else response.value

    nodes = []
    for uid in range(len(metagraph["hotkeys"])):
        axon = metagraph["axons"][uid]
        nodes.append(
            {
                "hotkey": _ss58_encode(metagraph["hotkeys"][uid]),
                "coldkey": _ss58_encode(metagraph["coldkeys"][uid]),
                "node_id": uid,
                "netuid": metagraph["netuid"],
                "incentive": metagraph["incentives"][uid],
                "alpha_stake": metagraph["alpha_stake"][uid] * 10**-9,
                "tao_stake": metagraph["tao_stake"][uid] * 10**-9,
                "stake": metagraph["total_stake"][uid] * 10**-9,
                "trust": 0,  # XXX metagraph["trust"][uid], removed https://github.com/opentensor/subtensor/pull/2158
                "vtrust": metagraph["consensus"][uid],
                "last_updated": float(metagraph["last_update"][uid]),
                "ip": str(axon["ip"]),
                "ip_type": axon["ip_type"],
                "port": axon["port"],
                "protocol": axon["protocol"],
            }
        )
    return nodes


async def sync_and_save_metagraph(netuid: int):
    """
    Load the metagraph for our subnet and persist it to the database.
    """
    async with AsyncSubstrateInterface(url=settings.subtensor) as substrate:
        nodes = await get_nodes_for_netuid(substrate, netuid)
        if not nodes:
            raise Exception("Failed to load metagraph nodes!")
        updated = 0
        async with SessionLocal() as session:
            hotkeys = ", ".join([f"'{node['hotkey']}'" for node in nodes])
            await session.execute(
                text(
                    f"DELETE FROM metagraph_nodes WHERE netuid = :netuid AND hotkey NOT IN ({hotkeys}) AND node_id >= 0"
                ),
                {
                    "netuid": netuid,
                },
            )
            for node in nodes:
                node_dict = dict(node)
                node_dict.pop("last_updated", None)
                node_dict["checksum"] = hashlib.sha256(
                    json.dumps(node_dict, sort_keys=True).encode()
                ).hexdigest()
                statement = insert(MetagraphNode).values(node_dict)
                statement = statement.on_conflict_do_update(
                    index_elements=["netuid", "hotkey"],
                    set_={key: getattr(statement.excluded, key) for key in node_dict.keys()},
                    where=MetagraphNode.checksum != node_dict["checksum"],
                )
                result = await session.execute(statement)
                if result.rowcount > 0:
                    logger.info(f"Detected metagraph update for hotkey={node['hotkey']}")
                    updated += 1
            if updated:
                logger.info(f"Updated {updated} nodes for {netuid=}")
            else:
                logger.info(f"No metagraph changes detected for {netuid=}")
            await session.commit()


async def main():
    """
    Main.
    """
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    try:
        logger.info(f"Attempting to resync metagraph for {settings.netuid=}")
        await asyncio.wait_for(sync_and_save_metagraph(netuid=settings.netuid), 60)
        logger.info(f"Successfully synced metagraph for {settings.netuid=}")

        # Other subnets (affine, babelbit, score, etc.).
        from api.constants import INTEGRATED_SUBNETS

        for subnet_info in INTEGRATED_SUBNETS.values():
            await asyncio.wait_for(sync_and_save_metagraph(netuid=subnet_info["netuid"]), 60)
    finally:
        await engine.dispose()

    os._exit(0)


if __name__ == "__main__":
    asyncio.run(main())
