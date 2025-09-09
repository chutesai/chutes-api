"""
ORM definitions for metagraph nodes.
"""

from api.database import get_session
from loguru import logger
from sqlalchemy.sql import func
from sqlalchemy.orm import relationship
from sqlalchemy import Column, String, DateTime, Integer, Float, text
from metasync.constants import (
    FEATURE_WEIGHTS,
    SCORING_INTERVAL,
    PRIVATE_INSTANCES_QUERY,
    NORMALIZED_COMPUTE_QUERY,
    UNIQUE_CHUTE_AVERAGE_QUERY,
)


def create_metagraph_node_class(base):
    """
    Instantiate our metagraph node class from a dynamic declarative base.
    """

    class MetagraphNode(base):
        __tablename__ = "metagraph_nodes"
        hotkey = Column(String, primary_key=True)
        checksum = Column(String, nullable=False)
        coldkey = Column(String, nullable=False)
        node_id = Column(Integer)
        incentive = Column(Float)
        netuid = Column(Integer)
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

        nodes = relationship(
            "Node",
            back_populates="miner",
            cascade="all, delete-orphan",
        )
        servers = relationship("Server", back_populates="miner")
        challenge_results = relationship(
            "ChallengeResult",
            back_populates="miner",
            cascade="all, delete-orphan",
        )

    return MetagraphNode


async def get_scoring_data():
    compute_query = text(NORMALIZED_COMPUTE_QUERY.format(interval=SCORING_INTERVAL))
    unique_query = text(UNIQUE_CHUTE_AVERAGE_QUERY.format(interval=SCORING_INTERVAL))
    private_query = text(PRIVATE_INSTANCES_QUERY.format(interval=SCORING_INTERVAL))

    raw_compute_values = {}
    highest_unique = 0
    async with get_session() as session:
        metagraph_nodes = await session.execute(
            text("SELECT coldkey, hotkey FROM metagraph_nodes WHERE netuid = 64 AND node_id >= 0")
        )
        hot_cold_map = {hotkey: coldkey for coldkey, hotkey in metagraph_nodes}
        coldkey_counts = {
            coldkey: sum([1 for _, ck in hot_cold_map.items() if ck == coldkey])
            for coldkey in hot_cold_map.values()
        }
        compute_result = await session.execute(compute_query)
        unique_result = await session.execute(unique_query)
        private_result = await session.execute(private_query)
        for hotkey, invocation_count, bounty_count, compute_units in compute_result:
            if not hotkey:
                continue
            raw_compute_values[hotkey] = {
                "invocation_count": invocation_count,
                "bounty_count": bounty_count,
                "compute_units": compute_units,
                "unique_chute_count": 0,
            }
        for miner_hotkey, average_active_chutes in unique_result:
            if not miner_hotkey:
                continue
            if miner_hotkey not in raw_compute_values:
                continue
            raw_compute_values[miner_hotkey]["unique_chute_count"] = average_active_chutes
            if average_active_chutes > highest_unique:
                highest_unique = average_active_chutes

        # Add in private instance compute units (jobs, private chutes).
        for miner_hotkey, total_instances, seconds, compute_units in private_result:
            if miner_hotkey not in raw_compute_values:
                continue
            logger.info(
                f"{miner_hotkey=} had {total_instances} private instances, {seconds=} {compute_units=}"
            )
            raw_compute_values[miner_hotkey]["bounty_count"] += total_instances
            raw_compute_values[miner_hotkey]["compute_units"] += float(compute_units)

            # XXX Subject to change, but for now give one invocation per second for private instances.
            raw_compute_values[miner_hotkey]["invocation_count"] += int(seconds)
            raw_compute_values[miner_hotkey].update(
                {
                    "private_instance_count": total_instances,
                    "private_instance_seconds": float(seconds),
                    "private_instance_compute_units": float(compute_units),
                }
            )

    # Log the raw values.
    for hotkey, values in raw_compute_values.items():
        logger.info(f"{hotkey}: {values}")

    totals = {
        key: sum(row[key] for row in raw_compute_values.values()) or 1.0 for key in FEATURE_WEIGHTS
    }

    normalized_values = {}
    unique_scores = [
        row["unique_chute_count"]
        for row in raw_compute_values.values()
        if row["unique_chute_count"]
    ]
    unique_scores.sort()
    n = len(unique_scores)
    if n > 0:
        if n % 2 == 0:
            median_unique_score = (unique_scores[n // 2 - 1] + unique_scores[n // 2]) / 2
        else:
            median_unique_score = unique_scores[n // 2]
    else:
        median_unique_score = 0
    for key in FEATURE_WEIGHTS:
        for hotkey, row in raw_compute_values.items():
            if hotkey not in normalized_values:
                normalized_values[hotkey] = {}
            if key == "unique_chute_count":
                if row[key] >= median_unique_score:
                    normalized_values[hotkey][key] = (row[key] / highest_unique) ** 1.3
                else:
                    normalized_values[hotkey][key] = (row[key] / highest_unique) ** 2.2
            else:
                normalized_values[hotkey][key] = row[key] / totals[key]

    # Re-normalize unique to [0, 1]
    unique_sum = sum([val["unique_chute_count"] for val in normalized_values.values()])
    old_unique_sum = sum([val["unique_chute_count"] for val in raw_compute_values.values()])
    for hotkey in normalized_values:
        normalized_values[hotkey]["unique_chute_count"] /= unique_sum
        old_value = raw_compute_values[hotkey]["unique_chute_count"] / old_unique_sum
        logger.info(
            f"Normalized, exponential unique score {hotkey} = {normalized_values[hotkey]['unique_chute_count']}, vs default: {old_value}"
        )

    pre_final_scores = {
        hotkey: sum(norm_value * FEATURE_WEIGHTS[key] for key, norm_value in metrics.items())
        for hotkey, metrics in normalized_values.items()
    }

    # Punish multi-uid miners.
    sorted_hotkeys = sorted(
        pre_final_scores.keys(), key=lambda h: pre_final_scores[h], reverse=True
    )
    penalized_scores = {}
    coldkey_used = set()
    for hotkey in sorted_hotkeys:
        coldkey = hot_cold_map[hotkey]
        if coldkey in coldkey_used:
            penalized_scores[hotkey] = 0.0
            logger.warning(
                f"Zeroing multi-uid miner {hotkey=} {coldkey=} count={coldkey_counts[coldkey]}"
            )
        else:
            penalized_scores[hotkey] = pre_final_scores[hotkey]
        coldkey_used.add(coldkey)

    # Normalize final scores by sum of penalized scores, just to make the incentive value match nicely.
    total = sum([val for hk, val in penalized_scores.items()])
    final_scores = {key: score / total for key, score in penalized_scores.items() if score > 0}

    # Log the final score.
    sorted_hotkeys = sorted(final_scores.keys(), key=lambda h: final_scores[h], reverse=True)
    for hotkey in sorted_hotkeys:
        coldkey_count = coldkey_counts[hot_cold_map[hotkey]]
        logger.info(f"{hotkey} ({coldkey_count=}): {final_scores[hotkey]}")

    return {
        "raw_values": raw_compute_values,
        "totals": totals,
        "normalized": normalized_values,
        "final_scores": final_scores,
    }
