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
    async with get_session() as session:
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
                raw_compute_values[miner_hotkey] = {
                    "invocation_count": 0,
                    "bounty_count": 0,
                    "compute_units": 0,
                    "unique_chute_count": 0,
                }
            raw_compute_values[miner_hotkey]["unique_chute_count"] = average_active_chutes
    totals = {
        key: sum(row[key] for row in raw_compute_values.values()) or 1.0 for key in FEATURE_WEIGHTS
    }
    normalized_values = {
        hotkey: {key: row[key] / totals[key] for key in FEATURE_WEIGHTS}
        for hotkey, row in raw_compute_values.items()
    }
    final_scores = {
        hotkey: sum(norm_value * FEATURE_WEIGHTS[key] for key, norm_value in metrics.items())
        for hotkey, metrics in normalized_values.items()
    }
    return {
        "raw_values": raw_compute_values,
        "totals": totals,
        "normalized": normalized_values,
        "final_scores": final_scores,
    }


async def get_unique_chute_history():
    query = text(UNIQUE_CHUTE_HISTORY_QUERY.format(interval=SCORING_INTERVAL))
    values = {}
    async with get_session(readonly=True) as session:
        result = await session.execute(query)
        for hotkey, timepoint, count in result:
            if hotkey not in values:
                values[hotkey] = []
            values[hotkey].append({"time": timepoint, "count": count})
    return values
