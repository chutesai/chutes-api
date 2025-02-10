"""
Calculates and schedules weights every SCORING_PERIOD
"""

from api.database import get_session
from sqlalchemy import text

import asyncio
from fiber import SubstrateInterface
from fiber.chain import weights
from fiber.logging_utils import get_logger
from fiber.chain import fetch_nodes
from fiber.networking.models import NodeWithFernet as Node
from fiber.chain.interface import get_substrate

from metasync.database import engine, Base
from metasync.config import settings
from fiber.chain.chain_utils import query_substrate

VERSION_KEY = 69420  # Doesn't matter too much in chutes' case
logger = get_logger(__name__)

# Proportion of weights to assign to each metric.
FEATURE_WEIGHTS = {
    "compute_units": 0.35,  # Total amount of compute time (compute muliplier * total time).
    "invocation_count": 0.3,  # Total number of invocations.
    "unique_chute_count": 0.25,  # Number of unique chutes over the scoring period.
    "bounty_count": 0.1,  # Number of bounties received (not bounty values, just counts).
}
SCORING_INTERVAL = "7 days"
NORMALIZED_COMPUTE_QUERY = """
WITH computation_rates AS (
    SELECT
        chute_id,
        percentile_cont(0.5) WITHIN GROUP (ORDER BY extract(epoch from completed_at - started_at) / (metrics->>'steps')::float) as median_step_time,
        percentile_cont(0.5) WITHIN GROUP (ORDER BY extract(epoch from completed_at - started_at) / (metrics->>'tokens')::float) as median_token_time
    FROM invocations
    WHERE ((metrics->>'steps' IS NOT NULL and (metrics->>'steps')::float > 0) OR (metrics->>'tokens' IS NOT NULL and (metrics->>'tokens')::float > 0))
      AND started_at >= NOW() - INTERVAL '2 days'
    GROUP BY chute_id
)
SELECT
    i.miner_hotkey,
    COUNT(*) as invocation_count,
    COUNT(DISTINCT(i.chute_id)) AS unique_chute_count,
    COUNT(CASE WHEN i.bounty > 0 THEN 1 END) AS bounty_count,
    sum(
        i.bounty +
        i.compute_multiplier *
        CASE
            WHEN i.metrics->>'steps' IS NOT NULL
                AND r.median_step_time IS NOT NULL
                AND EXTRACT(EPOCH FROM (i.completed_at - i.started_at)) > ((i.metrics->>'steps')::float * r.median_step_time)
            THEN (i.metrics->>'steps')::float * r.median_step_time
            WHEN i.metrics->>'tokens' IS NOT NULL
                AND r.median_token_time IS NOT NULL
                AND EXTRACT(EPOCH FROM (i.completed_at - i.started_at)) > ((i.metrics->>'tokens')::float * r.median_token_time)
            THEN (i.metrics->>'tokens')::float * r.median_token_time
            ELSE EXTRACT(EPOCH FROM (i.completed_at - i.started_at))
        END
    ) AS compute_units
FROM invocations i
LEFT JOIN computation_rates r ON i.chute_id = r.chute_id
WHERE i.started_at > NOW() - INTERVAL '{interval}'
AND i.error_message IS NULL
AND i.miner_uid > 0
AND i.completed_at IS NOT NULL
GROUP BY i.miner_hotkey
ORDER BY compute_units DESC;
"""


async def _get_validator_node_id(
    substrate: SubstrateInterface, netuid: int, ss58_address: str
) -> str | None:
    substrate, uid = query_substrate(
        substrate, "SubtensorModule", "Uids", [netuid, ss58_address], return_value=True
    )
    return substrate, uid


async def _get_weights_to_set(
    hotkeys_to_node_ids: dict[str, int],
) -> tuple[list[int], list[float]] | None:
    """
    Query the invocations for the past {SCORING INTERVAL} to calculate weights.

    Factors included in scoring are:
    - total compute time provided (as a factor of compute multiplier PLUS bounties awarded)
    - total number of invocations processed
    - number of unique chutes executed
    - number of bounties claimed

    Future improvements:
    - Punish errors more than just ignoring them
    - Have a decaying, normalised reward, rather than a fixed window
    """

    query = text(NORMALIZED_COMPUTE_QUERY.format(interval=SCORING_INTERVAL))
    raw_compute_values = {}
    header = ["hotkey", "invocation_count", "unique_chute_count", "bounty_count", "compute_units"]
    async with get_session() as session:
        result = await session.execute(query)
        for row in result:
            obj = dict(zip(header, row))
            raw_compute_values[obj["hotkey"]] = obj
            logger.info(obj)

    # Normalize the values based on totals so they are all in the range [0.0, 1.0]
    totals = {
        key: sum(row[key] for row in raw_compute_values.values()) or 1.0 for key in header[1:]
    }
    normalized_values = {
        hotkey: {key: row[key] / totals[key] for key in header[1:]}
        for hotkey, row in raw_compute_values.items()
    }
    # Adjust the values by the feature weights, e.g. compute_time gets more weight than bounty count.
    final_scores = {
        hotkey: sum(norm_value * FEATURE_WEIGHTS[key] for key, norm_value in metrics.items())
        for hotkey, metrics in normalized_values.items()
    }

    # Final weights per node.
    node_ids = []
    node_weights = []
    for hotkey, compute_score in final_scores.items():
        if hotkey not in hotkeys_to_node_ids:
            logger.debug(f"Miner {hotkey} not found on metagraph. Ignoring.")
            continue

        node_weights.append(compute_score)
        node_ids.append(hotkeys_to_node_ids[hotkey])
        logger.info(f"Normalized score for {hotkey}: {compute_score}")

    return node_ids, node_weights


async def _get_and_set_weights(substrate: SubstrateInterface) -> None:
    substrate, validator_node_id = await _get_validator_node_id(
        substrate, settings.netuid, settings.validator_ss58
    )

    if validator_node_id is None:
        raise ValueError(
            "Validator node id not found on the metagraph"
            f", are you sure hotkey {settings.validator_ss58} is registered on subnet {settings.netuid}?"
        )

    all_nodes: list[Node] = fetch_nodes.get_nodes_for_netuid(substrate, settings.netuid)
    hotkeys_to_node_ids = {node.hotkey: node.node_id for node in all_nodes}

    result = await _get_weights_to_set(hotkeys_to_node_ids)
    if result is None:
        logger.warning("No weights to set. Skipping weight setting.")
        return

    node_ids, node_weights = result
    if len(node_ids) == 0:
        logger.warning("No nodes to set weights for. Skipping weight setting.")
        return

    logger.info("Weights calculated, about to set...")

    all_node_ids = [node.node_id for node in all_nodes]
    all_node_weights = [0.0 for _ in all_nodes]
    for node_id, node_weight in zip(node_ids, node_weights):
        all_node_weights[node_id] = node_weight

    logger.info(f"Node ids: {all_node_ids}")
    logger.info(f"Node weights: {all_node_weights}")
    logger.info(
        f"Number of non zero node weights: {sum(1 for weight in all_node_weights if weight != 0)}"
    )

    try:
        success = weights.set_node_weights(
            substrate=substrate,
            keypair=settings.validator_keypair,
            node_ids=all_node_ids,
            node_weights=all_node_weights,
            netuid=settings.netuid,
            version_key=VERSION_KEY,
            validator_node_id=int(validator_node_id),
            wait_for_inclusion=False,
            wait_for_finalization=False,
            max_attempts=3,
        )
    except Exception as e:
        logger.error(f"Failed to set weights: {e}")
        return False

    if success:
        logger.info("Weights set successfully.")
        return True
    else:
        logger.error("Failed to set weights :(")
        return False


async def set_weights_periodically() -> None:
    substrate = get_substrate(
        subtensor_network=settings.subtensor_network,
        subtensor_address=settings.subtensor_address,
    )
    substrate, uid = query_substrate(
        substrate,
        "SubtensorModule",
        "Uids",
        [settings.netuid, settings.validator_ss58],
        return_value=True,
    )

    consecutive_failures = 0
    set_weights_interval_blocks = 150
    while True:
        substrate, current_block = query_substrate(
            substrate, "System", "Number", [], return_value=True
        )
        substrate, last_updated_value = query_substrate(
            substrate,
            "SubtensorModule",
            "LastUpdate",
            [settings.netuid],
            return_value=False,
        )
        updated: float = current_block - last_updated_value[uid].value
        logger.info(f"Last updated: {updated} for my uid: {uid}")
        if updated < set_weights_interval_blocks:
            blocks_to_sleep = set_weights_interval_blocks - updated + 1
            logger.info(
                f"Last updated: {updated} - sleeping for {blocks_to_sleep} blocks as we set recently..."
            )
            await asyncio.sleep(12 * blocks_to_sleep)  # sleep until we can set weights
            continue

        try:
            success = await _get_and_set_weights(substrate)
        except Exception as e:
            logger.error(f"Failed to set weights with error: {e}")
            success = False

        if success:
            consecutive_failures = 0
            logger.info("Successfully set weights!")
            continue

        consecutive_failures += 1

        logger.info(
            f"Failed to set weights {consecutive_failures} times in a row"
            " - sleeping for 10 blocks before trying again..."
        )
        await asyncio.sleep(12 * 10)  # Try again in 10 blocks


async def main():
    """
    Main.
    """
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    await set_weights_periodically()


if __name__ == "__main__":
    asyncio.run(main())
