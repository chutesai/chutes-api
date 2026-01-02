"""
Calculates and schedules weights every SCORING_PERIOD
"""

import asyncio
from async_substrate_interface import AsyncSubstrateInterface
from loguru import logger
from metasync.database import engine, Base
from metasync.config import settings
from metasync.shared import get_scoring_data
from metasync.sync_metagraph import get_nodes_for_netuid

VERSION_KEY = 69420  # Doesn't matter too much in chutes' case
U16_MAX = 65535


def _normalize_and_quantize_weights(
    node_ids: list[int], node_weights: list[float]
) -> tuple[list[int], list[int]]:
    """
    Normalize weights to sum to 1, then quantize to U16 values.
    """
    if not node_weights:
        return [], []

    total = sum(node_weights)
    if total <= 0:
        return node_ids, [0] * len(node_weights)

    # Normalize and quantize to U16
    normalized = [w / total for w in node_weights]
    quantized = [min(int(w * U16_MAX), U16_MAX) for w in normalized]

    # Filter out zero weights
    filtered_ids = []
    filtered_weights = []
    for nid, w in zip(node_ids, quantized):
        if w > 0:
            filtered_ids.append(nid)
            filtered_weights.append(w)

    return filtered_ids, filtered_weights


async def _get_validator_uid(
    substrate: AsyncSubstrateInterface, netuid: int, ss58_address: str
) -> int | None:
    """Get the UID for a validator on a given netuid."""
    result = await substrate.query(
        module="SubtensorModule",
        storage_function="Uids",
        params=[netuid, ss58_address],
    )
    if result is None:
        return None
    return int(result.value) if hasattr(result, "value") else int(result)


async def _get_last_update(substrate: AsyncSubstrateInterface, netuid: int) -> dict[int, int]:
    """Get the last update block for all UIDs on a netuid."""
    result = await substrate.query(
        module="SubtensorModule",
        storage_function="LastUpdate",
        params=[netuid],
    )
    if result is None:
        return {}
    value = result.value if hasattr(result, "value") else result
    if isinstance(value, list):
        return {i: int(v) for i, v in enumerate(value)}
    return {}


async def _get_current_block(substrate: AsyncSubstrateInterface) -> int:
    """Get the current block number."""
    result = await substrate.query(
        module="System",
        storage_function="Number",
        params=[],
    )
    return int(result.value) if hasattr(result, "value") else int(result)


async def _set_weights(
    substrate: AsyncSubstrateInterface,
    node_ids: list[int],
    node_weights: list[int],
    netuid: int,
    version_key: int,
) -> bool:
    """
    Submit the set_weights extrinsic to the chain.
    """
    call = await substrate.compose_call(
        call_module="SubtensorModule",
        call_function="set_weights",
        call_params={
            "dests": node_ids,
            "weights": node_weights,
            "netuid": netuid,
            "version_key": version_key,
        },
    )
    extrinsic = await substrate.create_signed_extrinsic(
        call=call,
        keypair=settings.validator_keypair,
        era={"period": 5},
    )
    receipt = await substrate.submit_extrinsic(
        extrinsic,
        wait_for_inclusion=True,
        wait_for_finalization=False,
    )
    return await receipt.is_success


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
    scoring_data = await get_scoring_data()
    final_scores = scoring_data["final_scores"]
    node_ids = []
    node_weights = []
    for hotkey, compute_score in final_scores.items():
        if hotkey not in hotkeys_to_node_ids:
            logger.debug(f"Miner {hotkey} not found on metagraph. Ignoring.")
            continue
        node_weights.append(compute_score)
        node_ids.append(hotkeys_to_node_ids[hotkey])
        logger.info(f"Setting score for {hotkey=} to {compute_score=}")
    return node_ids, node_weights


async def _get_and_set_weights(substrate: AsyncSubstrateInterface) -> bool:
    """Get weights from scoring data and set them on chain."""
    validator_uid = await _get_validator_uid(substrate, settings.netuid, settings.validator_ss58)

    if validator_uid is None:
        raise ValueError(
            "Validator node id not found on the metagraph"
            f", are you sure hotkey {settings.validator_ss58} is registered on subnet {settings.netuid}?"
        )

    nodes = await get_nodes_for_netuid(substrate, settings.netuid)
    hotkeys_to_node_ids = {node["hotkey"]: node["node_id"] for node in nodes}

    result = await _get_weights_to_set(hotkeys_to_node_ids)
    if result is None:
        logger.warning("No weights to set. Skipping weight setting.")
        return False

    node_ids, node_weights = result
    if len(node_ids) == 0:
        logger.warning("No nodes to set weights for. Skipping weight setting.")
        return False

    logger.info("Weights calculated, about to set...")

    # Build weights array for all nodes (including zeros for inactive ones)
    all_node_ids = [node["node_id"] for node in nodes]
    all_node_weights = [0.0 for _ in nodes]
    for node_id, node_weight in zip(node_ids, node_weights):
        all_node_weights[node_id] = node_weight

    logger.info(f"Node ids: {all_node_ids}")
    logger.info(f"Node weights: {all_node_weights}")
    logger.info(
        f"Number of non zero node weights: {sum(1 for weight in all_node_weights if weight != 0)}"
    )

    # Normalize and quantize weights
    quantized_ids, quantized_weights = _normalize_and_quantize_weights(
        all_node_ids, all_node_weights
    )

    try:
        success = await _set_weights(
            substrate=substrate,
            node_ids=quantized_ids,
            node_weights=quantized_weights,
            netuid=settings.netuid,
            version_key=VERSION_KEY,
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
    """
    Main loop to periodically set weights on the metagraph.
    """
    set_weights_interval_blocks = 150

    async with AsyncSubstrateInterface(url=settings.subtensor) as substrate:
        validator_uid = await _get_validator_uid(
            substrate, settings.netuid, settings.validator_ss58
        )
        if validator_uid is None:
            raise ValueError(
                f"Validator {settings.validator_ss58} not registered on subnet {settings.netuid}"
            )

        consecutive_failures = 0
        while True:
            current_block = await _get_current_block(substrate)
            last_update_map = await _get_last_update(substrate, settings.netuid)
            last_updated = last_update_map.get(validator_uid, 0)
            updated = current_block - last_updated

            logger.info(f"Last updated: {updated} blocks ago for uid: {validator_uid}")

            if updated < set_weights_interval_blocks:
                blocks_to_sleep = set_weights_interval_blocks - updated + 1
                logger.info(
                    f"Last updated: {updated} - sleeping for {blocks_to_sleep} blocks as we set recently..."
                )
                await asyncio.sleep(12 * blocks_to_sleep)
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
            await asyncio.sleep(12 * 10)


async def main():
    """
    Main.
    """
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    await set_weights_periodically()


if __name__ == "__main__":
    asyncio.run(main())
