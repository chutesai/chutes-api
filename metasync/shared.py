"""
ORM definitions for metagraph nodes.
"""

from api.config import settings
from api.database import get_session
from loguru import logger
from sqlalchemy.sql import func
from sqlalchemy.orm import relationship
from sqlalchemy import Column, String, DateTime, Integer, Float, text
from metasync.constants import (
    SCORING_INTERVAL,
    INSTANCES_QUERY,
)


def create_metagraph_node_class(base):
    """
    Instantiate our metagraph node class from a dynamic declarative base.
    """

    class MetagraphNode(base):
        __tablename__ = "metagraph_nodes"
        hotkey = Column(String, primary_key=True)
        netuid = Column(Integer, primary_key=True)
        checksum = Column(String, nullable=False)
        coldkey = Column(String, nullable=False)
        node_id = Column(Integer)
        incentive = Column(Float)
        stake = Column(Float)
        tao_stake = Column(Float)
        alpha_stake = Column(Float)
        trust = Column(Float)
        vtrust = Column(Float)
        last_updated = Column(Integer)
        ip = Column(String)
        ip_type = Column(Integer)
        port = Column(Integer)
        protocol = Column(Integer)
        real_host = Column(String)
        real_port = Column(Integer)
        synced_at = Column(DateTime, server_default=func.now())
        blacklist_reason = Column(String)

        servers = relationship("Server", back_populates="miner")

    return MetagraphNode


async def get_scoring_data(interval: str = SCORING_INTERVAL):
    """
    Compute miner scores based purely on compute_units (instance lifetime * compute_multiplier).
    All bonuses (bounty age, urgency, TEE, private) are baked into compute_multiplier at activation.
    """
    instances_query = text(INSTANCES_QUERY.format(interval=interval))

    # Load active miners from metagraph (and map coldkey pairings to de-dupe multi-hotkey miners).
    raw_values = {}
    blacklisted_hotkeys = set()
    logger.info(f"Loading metagraph for netuid={settings.netuid}...")
    async with get_session() as session:
        metagraph_nodes = await session.execute(
            text(
                f"SELECT coldkey, hotkey, blacklist_reason FROM metagraph_nodes WHERE netuid = {settings.netuid} AND node_id >= 0"
            )
        )
        hot_cold_map = {}
        for coldkey, hotkey, blacklist_reason in metagraph_nodes:
            hot_cold_map[hotkey] = coldkey
            if blacklist_reason:
                blacklisted_hotkeys.add(hotkey)
        coldkey_counts = {
            coldkey: sum([1 for _, ck in hot_cold_map.items() if ck == coldkey])
            for coldkey in hot_cold_map.values()
        }
    if blacklisted_hotkeys:
        logger.info(f"Found {len(blacklisted_hotkeys)} blacklisted miners to exclude from scoring")

    # Base score - instances active during the scoring period.
    logger.info("Fetching scores based on active instances during scoring interval...")
    async with get_session() as session:
        instances_result = await session.execute(instances_query)
        for (
            hotkey,
            total_instances,
            bounty_score,
            instance_seconds,
            instance_compute_units,
        ) in instances_result:
            if not hotkey or hotkey not in hot_cold_map or hotkey in blacklisted_hotkeys:
                continue
            raw_values[hotkey] = {
                "total_instances": float(total_instances or 0.0),
                "bounty_score": float(bounty_score or 0.0),
                "instance_seconds": float(instance_seconds or 0.0),
                "instance_compute_units": float(instance_compute_units or 0.0),
            }

    # Build scores from instance compute units.
    scores = {hk: data["instance_compute_units"] for hk, data in raw_values.items()}

    # Purge multi-hotkey miners - keep only the highest scoring hotkey per coldkey
    hotkeys_to_remove = set()
    for coldkey in set(hot_cold_map.values()):
        if coldkey_counts.get(coldkey, 0) > 1:
            coldkey_hotkeys = [
                hk for hk, ck in hot_cold_map.items() if ck == coldkey and hk in scores
            ]
            if len(coldkey_hotkeys) > 1:
                coldkey_hotkeys.sort(key=lambda hk: scores.get(hk, 0.0), reverse=True)
                hotkeys_to_remove.update(coldkey_hotkeys[1:])

    for hotkey in hotkeys_to_remove:
        scores.pop(hotkey, None)
        raw_values.pop(hotkey, None)
        logger.warning(f"Purging hotkey from multi-uid miner: {hotkey=}")

    # Normalize to distribution.
    score_sum = sum(max(0.0, v) for v in scores.values())
    if score_sum > 0:
        final_scores = {hk: max(0.0, v) / score_sum for hk, v in scores.items()}
    else:
        n = max(len(scores), 1)
        final_scores = {hk: 1.0 / n for hk in scores.keys()}

    sorted_hotkeys = sorted(final_scores.keys(), key=lambda k: final_scores[k], reverse=True)
    logger.info(
        f"{'#':<3} {'Hotkey':<48} {'Score':<10} {'Instances':<10} {'Seconds':<12} {'Compute':<12}"
    )
    logger.info("-" * 100)
    for rank, hotkey in enumerate(sorted_hotkeys, 1):
        data = raw_values.get(hotkey, {})
        logger.info(
            f"{rank:<3} "
            f"{hotkey:<48} "
            f"{final_scores[hotkey]:<10.6f} "
            f"{int(data.get('total_instances', 0)):<10} "
            f"{int(data.get('instance_seconds', 0)):<12} "
            f"{int(data.get('instance_compute_units', 0)):<12}"
        )

    return {"raw_values": raw_values, "final_scores": final_scores}


if __name__ == "__main__":
    import asyncio

    asyncio.run(get_scoring_data(interval="7 days"))
