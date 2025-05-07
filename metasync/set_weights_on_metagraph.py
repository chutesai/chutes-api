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
from metasync.constants import (
    UNIQUE_CHUTE_AVERAGE_QUERY,
    NORMALIZED_COMPUTE_QUERY,
    UTILIZATION_RATIO_QUERY,
    SCORING_INTERVAL,
    FEATURE_WEIGHTS,
    UTILIZATION_THRESHOLD,
)

VERSION_KEY = 69420  # Doesn't matter too much in chutes' case
logger = get_logger(__name__)


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

    compute_query = text(NORMALIZED_COMPUTE_QUERY.format(interval=SCORING_INTERVAL))
    unique_query = text(UNIQUE_CHUTE_AVERAGE_QUERY.format(interval=SCORING_INTERVAL))
    utilization_query = text(UTILIZATION_RATIO_QUERY.format(interval="8 hours"))
    raw_compute_values = {}
    highest_unique = 0.0
    async with get_session() as session:
        # Metagraph query if we enable multi-uid punishments.
        metagraph_nodes = await session.execute(
            text("SELECT coldkey, hotkey FROM metagraph_nodes WHERE netuid = 64 AND node_id >= 0")
        )
        hot_cold_map = {hotkey: coldkey for coldkey, hotkey in metagraph_nodes}

        compute_result = await session.execute(compute_query)
        unique_result = await session.execute(unique_query)
        utilization_result = await session.execute(utilization_query)

        # Get the set of miners with less than useless utilization.
        utilization = {hotkey: float(utilization) for hotkey, utilization in utilization_result}

        # Compute units, invocation counts, and bounties.
        for hotkey, invocation_count, bounty_count, compute_units in compute_result:
            if (ut := utilization.get(hotkey, 0.0)) < UTILIZATION_THRESHOLD:
                logger.warning(f"Miner {hotkey} has utilization ratio {ut}, zero score...")
                continue
            raw_compute_values[hotkey] = {
                "invocation_count": invocation_count,
                "bounty_count": bounty_count,
                "compute_units": compute_units,
                "unique_chute_count": 0,
            }

        # Average active unique chute counts.
        for miner_hotkey, average_active_chutes in unique_result:
            if miner_hotkey not in raw_compute_values:
                continue
            raw_compute_values[miner_hotkey]["unique_chute_count"] = average_active_chutes
            if average_active_chutes > highest_unique:
                highest_unique = average_active_chutes

    # Logging.
    for hotkey, values in raw_compute_values.items():
        logger.info(f"{hotkey}: {values}")

    # Normalize the values based on totals so they are all in the range [0.0, 1.0]
    totals = {
        key: sum(row[key] for row in raw_compute_values.values()) or 1.0 for key in FEATURE_WEIGHTS
    }
    normalized_values = {}
    mean_unique_score = totals["unique_chute_count"] / (len(raw_compute_values) or 1)
    for key in FEATURE_WEIGHTS:
        for hotkey, row in raw_compute_values.items():
            if hotkey not in normalized_values:
                normalized_values[hotkey] = {}
            if key == "unique_chute_count":
                if row[key] >= mean_unique_score:
                    normalized_values[hotkey][key] = (row[key] / highest_unique) ** 1.2
                else:
                    normalized_values[hotkey][key] = (row[key] / highest_unique) ** 2.0
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

    # Adjust the values by the feature weights, e.g. compute_time gets more weight than bounty count.
    pre_final_scores = {
        hotkey: sum(norm_value * FEATURE_WEIGHTS[key] for key, norm_value in metrics.items())
        for hotkey, metrics in normalized_values.items()
    }

    # Punish multi-uid miners.
    sorted_hotkeys = sorted(
        pre_final_scores.keys(), key=lambda h: pre_final_scores[h], reverse=True
    )
    coldkey_counts = {
        coldkey: sum([1 for _, ck in hot_cold_map.items() if ck == coldkey])
        for coldkey in hot_cold_map.values()
    }
    penalized_scores = {}
    coldkey_used = set()
    for hotkey in sorted_hotkeys:
        coldkey = hot_cold_map[hotkey]
        if coldkey in coldkey_used:
            logger.warning(
                f"Zeroing multi-uid miner {hotkey=} {coldkey=} count={coldkey_counts[coldkey]}"
            )
            penalized_scores[hotkey] = 0.0
        else:
            penalized_scores[hotkey] = pre_final_scores[hotkey]
        coldkey_used.add(coldkey)

    # Normalize final scores by sum of penalized scores, just to make the incentive value match nicely.
    total = sum([val for hk, val in penalized_scores.items()])
    final_scores = {key: score / total for key, score in penalized_scores.items() if score > 0}
    sorted_hotkeys = sorted(final_scores.keys(), key=lambda h: final_scores[h], reverse=True)
    for hotkey in sorted_hotkeys:
        coldkey_count = coldkey_counts[hot_cold_map[hotkey]]
        logger.info(f"{hotkey} ({coldkey_count=}): {final_scores[hotkey]}")

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
        updated: float = current_block - last_updated_value[uid]
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
