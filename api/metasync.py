from api.database import Base, get_session
from api.config import settings
from metasync.shared import create_metagraph_node_class
from metasync.constants import (
    FEATURE_WEIGHTS,
    SCORING_INTERVAL,
    NORMALIZED_COMPUTE_QUERY,
    UNIQUE_CHUTE_AVERAGE_QUERY,
    UNIQUE_CHUTE_HISTORY_QUERY,
)
from sqlalchemy import select, text

MetagraphNode = create_metagraph_node_class(Base)


async def get_miner_by_hotkey(hotkey, db):
    """
    Helper to load a node by ID.
    """
    if not hotkey:
        return None
    query = (
        select(MetagraphNode)
        .where(MetagraphNode.hotkey == hotkey)
        .where(MetagraphNode.netuid == settings.netuid)
    )
    result = await db.execute(query)
    return result.scalar_one_or_none()


async def get_scoring_data():
    compute_query = text(NORMALIZED_COMPUTE_QUERY.format(interval=SCORING_INTERVAL))
    unique_query = text(UNIQUE_CHUTE_AVERAGE_QUERY.format(interval=SCORING_INTERVAL))
    raw_compute_values = {}
    highest_unique = 0
    async with get_session() as session:
        metagraph_nodes = await session.execute(
            text("SELECT coldkey, hotkey FROM metagraph_nodes WHERE netuid = 64 AND node_id >= 0")
        )
        hot_cold_map = {hotkey: coldkey for coldkey, hotkey in metagraph_nodes}
        compute_result = await session.execute(compute_query)
        unique_result = await session.execute(unique_query)
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
    for hotkey in normalized_values:
        normalized_values[hotkey]["unique_chute_count"] /= unique_sum

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
        else:
            penalized_scores[hotkey] = pre_final_scores[hotkey]
        coldkey_used.add(coldkey)

    # Normalize final scores by sum of penalized scores, just to make the incentive value match nicely.
    total = sum([val for hk, val in penalized_scores.items()])
    final_scores = {key: score / total for key, score in penalized_scores.items() if score > 0}

    return {
        "raw_values": raw_compute_values,
        "totals": totals,
        "normalized": normalized_values,
        "final_scores": final_scores,
    }


async def get_unique_chute_history():
    query = text(UNIQUE_CHUTE_HISTORY_QUERY.format(interval=SCORING_INTERVAL))
    values = {}
    async with get_session() as session:
        result = await session.execute(query)
        for hotkey, timepoint, count in result:
            if hotkey not in values:
                values[hotkey] = []
            values[hotkey].append({"time": timepoint, "count": count})
    return values
