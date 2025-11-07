"""
ORM definitions for metagraph nodes.
"""

from api.config import settings
from api.database import get_session
from loguru import logger
from sqlalchemy.sql import func
from sqlalchemy import Column, String, DateTime, Integer, Float, text
from metasync.constants2 import (
    BONUS,
    SCORING_INTERVAL,
    INSTANCES_QUERY,
    NORMALIZED_COMPUTE_QUERY,
    INVENTORY_QUERY,
    DEMAND_COMPUTE_WEIGHT,
    DEMAND_COUNT_WEIGHT,
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

    return MetagraphNode


async def get_scoring_data(interval: str = SCORING_INTERVAL):
    compute_query = text(NORMALIZED_COMPUTE_QUERY.format(interval=interval))
    inventory_query = text(INVENTORY_QUERY.format(interval=interval))
    instances_query = text(INSTANCES_QUERY.format(interval=interval))

    # Load active miners from metagraph (and map coldkey pairings to de-dupe multi-hotkey miners).
    raw_values = {}
    boosts = {}
    logger.info(f"Loading metagraph for netuid={settings.netuid}...")
    async with get_session() as session:
        metagraph_nodes = await session.execute(
            text(f"SELECT coldkey, hotkey FROM metagraph_nodes WHERE netuid = {settings.netuid} AND node_id >= 0")
        )
        hot_cold_map = {hotkey: coldkey for coldkey, hotkey in metagraph_nodes}
        coldkey_counts = {
            coldkey: sum([1 for _, ck in hot_cold_map.items() if ck == coldkey])
            for coldkey in hot_cold_map.values()
        }

    # Base score - instances active during the scoring period.
    logger.info("Fetching base score values based on active instances during scoring interval...")
    async with get_session() as session:
        instances_result = await session.execute(instances_query)
        for (
            hotkey,
            total_instances,
            bounties,
            instance_seconds,
            instance_compute_units,
        ) in instances_result:
            if not hotkey or hotkey not in hot_cold_map:
                continue
            raw_values[hotkey] = {
                "total_instances": total_instances,
                "bounties": bounties,
                "instance_seconds": instance_seconds,
                "instance_compute_units": instance_compute_units,
                "success_rate": 0.0,
                "invocation_compute_units": 0.0,
                "invocation_count": 0.0,
                "unique_chute_gpus": 0.0,
            }

    # Get the invocation metrics to calculate boosts for "demand" and "success_ratio"
    logger.info("Fetching invocation metrics to calculate demand and success ratio boosts...")
    async with get_session() as session:
        compute_result = await session.execute(compute_query)
        for hotkey, successful_count, error_count, compute_units in compute_result:
            if hotkey not in raw_values:
                continue
            raw_values[hotkey]["success_rate"] = successful_count / (
                (successful_count + error_count) or 1.0
            )
            raw_values[hotkey]["invocation_compute_units"] = compute_units
            raw_values[hotkey]["invocation_count"] = successful_count

    # Get the unique chute ("breadth" bonus) data.
    logger.info(f"Fetching unique chute GPU score to calculate breadth bonus...")
    async with get_session() as session:
        unique_result = await session.execute(inventory_query)
        for hotkey, unique_chute_gpus, total_active_gpus in unique_result:
            if hotkey not in raw_values:
                continue
            raw_values[hotkey]["unique_chute_gpus"] = unique_chute_gpus

    # First, we'll calculate the scores as [0,1] range based on compute units.
    logger.info("Normalizing scores and adding boosts...")
    base_scores = {}
    for hotkey, data in raw_values.items():
        base_scores[hotkey] = data["instance_compute_units"]

    # Purge multi-hotkey miners - keep only the highest scoring hotkey per coldkey
    hotkeys_to_remove = set()
    for coldkey in set(hot_cold_map.values()):
        if coldkey_counts[coldkey] > 1:
            coldkey_hotkeys = [
                hk for hk, ck in hot_cold_map.items() if ck == coldkey and hk in base_scores
            ]
            if len(coldkey_hotkeys) > 1:
                coldkey_hotkeys.sort(key=lambda hk: base_scores[hk], reverse=True)
                hotkeys_to_remove.update(coldkey_hotkeys[1:])


    # Remove the lower-scoring hotkeys
    for hotkey in hotkeys_to_remove:
        base_scores.pop(hotkey, None)
        raw_values.pop(hotkey, None)
        logger.warning(f"Purging hotkey from multi-uid miner: {hotkey=}")

    # Helper function to normalize and apply exponential
    def normalize_and_exp(values_dict, key, exp=1.4):
        values = [data.get(key, 0) for data in values_dict.values()]
        max_val = max(values) if values else 1.0
        min_val = min(values) if values else 0.0
        range_val = max_val - min_val if max_val != min_val else 1.0
        normalized = {}
        for hotkey, data in values_dict.items():
            norm_val = (data.get(key, 0) - min_val) / range_val if range_val > 0 else 0
            normalized[hotkey] = norm_val**exp
        exp_max = max(normalized.values()) if normalized else 1.0
        if exp_max > 0:
            for hotkey in normalized:
                normalized[hotkey] /= exp_max
        return normalized

    # Calculate all of the bonuses.
    bonuses = {}

    # Breadth bonus (unique_chute_gpus, non-selectiveness in deploying chutes).
    breadth_scores = normalize_and_exp(raw_values, "unique_chute_gpus", 2.0)

    # Demand bonus (miner deploys chutes that get a lot of real-world invocation usage).
    invoc_compute_scores = normalize_and_exp(raw_values, "invocation_compute_units", 2.0)
    invoc_count_scores = normalize_and_exp(raw_values, "invocation_count", 2.0)
    demand_scores = {}
    for hotkey in raw_values:
        demand_scores[hotkey] = DEMAND_COMPUTE_WEIGHT * invoc_compute_scores.get(
            hotkey, 0
        ) + DEMAND_COUNT_WEIGHT * invoc_count_scores.get(hotkey, 0)

    # Bounties (miner was first to activate an instance of a chute that had a bounty).
    bounty_scores = normalize_and_exp(raw_values, "bounties", 2.0)

    # Success rate (miner generally has a higher success rate in invocations).
    success_scores = normalize_and_exp(raw_values, "success_rate", 2.0)

    # Normalize the base scores to sum to 1.0
    total_base = sum(base_scores.values()) if base_scores else 1.0
    if total_base > 0:
        for hotkey in base_scores:
            base_scores[hotkey] /= total_base

    # Apply bonuses.
    final_scores = {}
    for hotkey in base_scores:
        score = base_scores[hotkey]
        # Add each bonus
        score += BONUS["breadth"] * breadth_scores.get(hotkey, 0)
        score += BONUS["demand"] * demand_scores.get(hotkey, 0)
        score += BONUS["bounty"] * bounty_scores.get(hotkey, 0)
        score += BONUS["success_rate"] * success_scores.get(hotkey, 0)
        final_scores[hotkey] = score

    # Normalize to ensure sum equals 1.0
    total_final = sum(final_scores.values()) if final_scores else 1.0
    if total_final > 0:
        for hotkey in final_scores:
            final_scores[hotkey] /= total_final

    # Logging.
    sorted_hotkeys = sorted(final_scores.keys(), key=lambda k: final_scores[k], reverse=True)
    logger.info(
        f"{'#':<3} "
        f"{'Hotkey':<48} "
        f"{'Score':<10} "
        f"{'Base':<10} "
        f"{'Breadth':<10} "
        f"{'Demand':<10} "
        f"{'Bounty':<10} "
        f"{'Success':<10}"
    )
    logger.info("-" * 120)

    for rank, hotkey in enumerate(sorted_hotkeys, 1):  # Start from 1
        logger.info(
            f"{rank:<3} "
            f"{hotkey:<48} "
            f"{final_scores[hotkey]:<10.6f} "
            f"{base_scores.get(hotkey, 0):<10.6f} "
            f"{BONUS['breadth'] * breadth_scores.get(hotkey, 0):<10.6f} "
            f"{BONUS['demand'] * demand_scores.get(hotkey, 0):<10.6f} "
            f"{BONUS['bounty'] * bounty_scores.get(hotkey, 0):<10.6f} "
            f"{BONUS['success_rate'] * success_scores.get(hotkey, 0):<10.6f}"
        )

    return raw_values, final_scores


if __name__ == "__main__":
    import asyncio
    asyncio.run(get_scoring_data(interval = '7 days'))
