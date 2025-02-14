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
from metasync.constants import FEATURE_WEIGHTS, SCORING_INTERVAL, NORMALIZED_COMPUTE_QUERY
from fiber.chain.chain_utils import query_substrate

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
