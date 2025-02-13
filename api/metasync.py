from api.database import Base, get_session
from api.config import settings
from metasync.shared import create_metagraph_node_class
from metasync.constants import FEATURE_WEIGHTS, SCORING_INTERVAL, NORMALIZED_COMPUTE_QUERY
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
    query = text(NORMALIZED_COMPUTE_QUERY.format(interval=SCORING_INTERVAL))
    raw_compute_values = {}
    header = ["hotkey", "invocation_count", "unique_chute_count", "bounty_count", "compute_units"]
    async with get_session() as session:
        result = await session.execute(query)
        for row in result:
            obj = dict(zip(header, row))
            raw_compute_values[obj["hotkey"]] = obj
    totals = {
        key: sum(row[key] for row in raw_compute_values.values()) or 1.0 for key in header[1:]
    }
    normalized_values = {
        hotkey: {key: row[key] / totals[key] for key in header[1:]}
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
