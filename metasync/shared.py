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
    BONUS,
    BONUS_EXP,
    BOUNTY_DECAY,
    BOUNTY_RHO,
    SCORING_INTERVAL,
    INSTANCES_QUERY,
    NORMALIZED_COMPUTE_QUERY,
    INVENTORY_QUERY,
    DEMAND_COMPUTE_WEIGHT,
    DEMAND_COUNT_WEIGHT,
    BONUS_WEIGHT,
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
    compute_query = text(NORMALIZED_COMPUTE_QUERY.format(interval=interval))
    inventory_query = text(INVENTORY_QUERY.format(interval=interval))
    instances_query = text(
        INSTANCES_QUERY.format(interval=interval, bounty_decay=BOUNTY_DECAY, bounty_rho=BOUNTY_RHO)
    )

    # Load active miners from metagraph (and map coldkey pairings to de-dupe multi-hotkey miners).
    raw_values = {}
    logger.info(f"Loading metagraph for netuid={settings.netuid}...")
    async with get_session() as session:
        metagraph_nodes = await session.execute(
            text(
                f"SELECT coldkey, hotkey FROM metagraph_nodes WHERE netuid = {settings.netuid} AND node_id >= 0"
            )
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
            bounty_score,
            instance_seconds,
            instance_compute_units,
        ) in instances_result:
            if not hotkey or hotkey not in hot_cold_map:
                continue
            raw_values[hotkey] = {
                "total_instances": float(total_instances or 0.0),
                "bounty_score": float(bounty_score or 0.0),
                "instance_seconds": float(instance_seconds or 0.0),
                "instance_compute_units": float(instance_compute_units or 0.0),
                "invocation_compute_units": 0.0,
                "invocation_count": 0.0,
                "unique_chute_gpus": 0.0,
            }

    # Get the invocation metrics to calculate boosts for "demand"
    logger.info("Fetching invocation metrics to calculate demand boost...")
    async with get_session() as session:
        compute_result = await session.execute(compute_query)
        for hotkey, count, compute_units in compute_result:
            if hotkey not in raw_values:
                continue
            raw_values[hotkey]["invocation_compute_units"] = float(compute_units or 0.0)
            raw_values[hotkey]["invocation_count"] = count

    # Get the unique chute ("breadth" bonus) data.
    logger.info("Fetching unique chute GPU score to calculate breadth bonus...")
    async with get_session() as session:
        unique_result = await session.execute(inventory_query)
        for hotkey, unique_chute_gpus, total_active_gpus in unique_result:
            if hotkey not in raw_values:
                continue
            raw_values[hotkey]["unique_chute_gpus"] = float(unique_chute_gpus or 0.0)

    # Build base scores from instance compute units.
    base_scores = {hk: data["instance_compute_units"] for hk, data in raw_values.items()}

    # Purge multi-hotkey miners - keep only the highest scoring hotkey per coldkey
    hotkeys_to_remove = set()
    for coldkey in set(hot_cold_map.values()):
        if coldkey_counts.get(coldkey, 0) > 1:
            coldkey_hotkeys = [
                hk for hk, ck in hot_cold_map.items() if ck == coldkey and hk in base_scores
            ]
            if len(coldkey_hotkeys) > 1:
                coldkey_hotkeys.sort(key=lambda hk: base_scores.get(hk, 0.0), reverse=True)
                hotkeys_to_remove.update(coldkey_hotkeys[1:])

    for hotkey in hotkeys_to_remove:
        base_scores.pop(hotkey, None)
        raw_values.pop(hotkey, None)
        logger.warning(f"Purging hotkey from multi-uid miner: {hotkey=}")

    # Helpers
    def minmax_then_exp_to_dist(values_map: dict[str, float], exp: float) -> dict[str, float]:
        """
        Min-max to [0,1], raise to 'exp', then sum-normalize to a distribution.
        Returns a dict that sums to 1 across keys (unless empty).
        """
        if not values_map:
            return {}
        vals = list(values_map.values())
        vmin, vmax = min(vals), max(vals)
        rng = max(vmax - vmin, 1e-12)

        powered = {k: ((v - vmin) / rng) ** exp for k, v in values_map.items()}
        S = sum(powered.values())
        if S <= 0:
            # uniform fallback
            n = len(powered)
            return {k: 1.0 / n for k in powered}
        return {k: powered[k] / S for k in powered}

    def category_from_raw(raw_key: str) -> dict[str, float]:
        return {hk: raw_values[hk].get(raw_key, 0.0) for hk in raw_values.keys()}

    base_sum = sum(max(0.0, v) for v in base_scores.values())
    base_dist = {}
    if base_sum > 0:
        base_dist = {hk: max(0.0, v) / base_sum for hk, v in base_scores.items()}
    else:
        n = max(len(base_scores), 1)
        base_dist = {hk: 1.0 / n for hk in base_scores.keys()}

    base_weight = 1.0 - BONUS_WEIGHT
    base_contrib = {hk: base_weight * base_dist.get(hk, 0.0) for hk in raw_values.keys()}

    logger.info("Computing bonus distributions...")

    # Category raw maps
    breadth_raw = category_from_raw("unique_chute_gpus")
    invoc_compute_raw = category_from_raw("invocation_compute_units")
    invoc_count_raw = category_from_raw("invocation_count")
    bounty_raw = category_from_raw("bounty_score")

    # Demand raw is a weighted mix of invoc compute & count (still per-miner raw before dist)
    demand_raw = {}
    for hk in raw_values.keys():
        dcw = float(DEMAND_COMPUTE_WEIGHT)
        dnw = float(DEMAND_COUNT_WEIGHT)
        ws = max(dcw + dnw, 1e-12)
        demand_raw[hk] = (
            dcw * invoc_compute_raw.get(hk, 0.0) + dnw * invoc_count_raw.get(hk, 0.0)
        ) / ws

    # Turn each category into a distribution via min-max → exp → sum-normalize
    breadth_dist = minmax_then_exp_to_dist(breadth_raw, BONUS_EXP)
    demand_dist = minmax_then_exp_to_dist(demand_raw, BONUS_EXP)
    bounty_dist = minmax_then_exp_to_dist(bounty_raw, BONUS_EXP)

    # Normalize BONUS weights across the categories that we’re actually using.
    w_breadth = float(BONUS.get("breadth", 0.0))
    w_demand = float(BONUS.get("demand", 0.0))
    w_bounty = float(BONUS.get("bounty", 0.0))
    W = max(w_breadth + w_demand + w_bounty, 1e-12)
    wb, wd, wbo = w_breadth / W, w_demand / W, w_bounty / W

    # Blend category distributions into a single bonus distribution and scale by bonus weight.
    blended_bonus_dist = {}
    for hk in raw_values.keys():
        blended_bonus_dist[hk] = (
            wb * breadth_dist.get(hk, 0.0)
            + wd * demand_dist.get(hk, 0.0)
            + wbo * bounty_dist.get(hk, 0.0)
        )
    bonus_weight = BONUS_WEIGHT
    bonus_contrib = {hk: bonus_weight * blended_bonus_dist.get(hk, 0.0) for hk in raw_values.keys()}

    final_scores = {
        hk: base_contrib.get(hk, 0.0) + bonus_contrib.get(hk, 0.0) for hk in raw_values.keys()
    }
    sorted_hotkeys = sorted(final_scores.keys(), key=lambda k: final_scores[k], reverse=True)
    logger.info(
        f"{'#':<3} "
        f"{'Hotkey':<48} "
        f"{'Score':<10} "
        f"{'Base':<10} "
        f"{'Breadth':<10} "
        f"{'Demand':<10} "
        f"{'Bounty':<10}"
    )
    logger.info("-" * 120)
    for rank, hotkey in enumerate(sorted_hotkeys, 1):
        b_total = bonus_contrib.get(hotkey, 0.0)
        eps = 1e-12
        cat_share = (
            wb * breadth_dist.get(hotkey, 0.0)
            + wd * demand_dist.get(hotkey, 0.0)
            + wbo * bounty_dist.get(hotkey, 0.0)
        ) + eps
        breadth_c = b_total * (wb * breadth_dist.get(hotkey, 0.0)) / cat_share
        demand_c = b_total * (wd * demand_dist.get(hotkey, 0.0)) / cat_share
        bounty_c = b_total * (wbo * bounty_dist.get(hotkey, 0.0)) / cat_share
        logger.info(
            f"{rank:<3} "
            f"{hotkey:<48} "
            f"{final_scores[hotkey]:<10.6f} "
            f"{base_contrib.get(hotkey, 0.0):<10.6f} "
            f"{breadth_c:<10.6f} "
            f"{demand_c:<10.6f} "
            f"{bounty_c:<10.6f} "
        )

    return {"raw_values": raw_values, "final_scores": final_scores}


if __name__ == "__main__":
    import asyncio

    asyncio.run(get_scoring_data(interval="7 days"))
