"""
Scale down underutilized chutes.
"""

import asyncio
import random
import math
from loguru import logger
from collections import defaultdict
from datetime import timedelta, datetime, timezone
from api.constants import EXPANSION_UTILIZATION_THRESHOLD, UNDERUTILIZED_CAP
from api.database import get_session
from api.chute.schemas import Chute
from sqlalchemy import text, select
from sqlalchemy.orm import selectinload
import api.database.orms  # noqa
from api.instance.schemas import Instance
from watchtower import purge_and_notify


async def scale_down():
    """
    Scale down chutes that are underutilized and kick a random miner if/when capped.
    """

    # Identify the chutes that need to be scaled down.
    instances_removed = 0
    to_downsize = []
    async with get_session() as session:
        query = text("""
          WITH instance_counts AS (
            SELECT chute_id, COUNT(*) AS count
            FROM instances
            GROUP BY chute_id
          )
          SELECT
            cu.chute_id,
            avg_busy_ratio,
            total_invocations,
            total_rate_limit_errors,
            count AS instance_count
          FROM chute_utilization cu
          JOIN instance_counts ic ON cu.chute_id = ic.chute_id
          WHERE count >= :cap
          AND avg_busy_ratio < :ratio
          AND total_rate_limit_errors = 0
          AND NOT EXISTS (
            SELECT FROM rolling_updates ru WHERE ru.chute_id = cu.chute_id
          )
        """)
        results = await session.execute(
            query, {"cap": UNDERUTILIZED_CAP, "ratio": EXPANSION_UTILIZATION_THRESHOLD}
        )
        rows = results.mappings().all()
        utilization_data = [dict(row) for row in rows]
        for item in utilization_data:
            to_downsize.append(item["chute_id"])

    # Perform the actual kicks.
    for chute_id in to_downsize:
        # Use separate sessions to avoid really long-lived potentially blocking sessions.
        async with get_session() as session:
            chute = (
                (
                    await session.execute(
                        select(Chute)
                        .where(Chute.chute_id == chute_id)
                        .options(selectinload(Chute.instances).selectinload(Instance.nodes))
                    )
                )
                .unique()
                .scalar_one_or_none()
            )
            if not chute:
                logger.warning(f"Chute not found: {chute_id=}")
                return
            if chute.rolling_update:
                logger.warning(f"Chute has a rolling update in progress: {chute_id=}")

            # Instead of a loop here while the count is too high, we'll only kick one
            # per interval, since the utilization ratio will inevitably change after.
            active = [inst for inst in chute.instances if inst.verified and inst.active]
            instances = []
            for instance in active:
                if len(instance.nodes) != chute.node_selector.get("gpu_count"):
                    logger.warning(f"Bad instance? {instance.instance_id=} {instance.verified=}")
                    await purge_and_notify(
                        instance, reason="instance node count does not match node selector"
                    )
                else:
                    instances.append(instance)

            # Instead of a loop here while the count is too high, we'll only kick (a few)
            # per interval, since the utilization ratio will inevitably change after.
            if len(instances) < UNDERUTILIZED_CAP:
                continue
            number_to_kick = math.ceil(len(instances) / UNDERUTILIZED_CAP)
            logger.warning(
                f"Downsizing chute {chute_id}, current count = {len(instances)}, removing {number_to_kick} unlucky instances"
            )

            kicked = set()
            for idx in range(number_to_kick):
                instances = [i for i in instances if i.instance_id not in kicked]
                # Kick a miner with highest instance counts, when > 1.
                unlucky_instance = None
                unlucky_reason = None
                counts = defaultdict(int)
                for instance in instances:
                    counts[instance.miner_hotkey] += 1
                max_count = max(counts.values())
                if max_count > 1:
                    max_miners = [hotkey for hotkey, count in counts.items() if count == max_count]
                    unlucky = random.choice(max_miners)
                    unlucky_instance = random.choice(
                        [instance for instance in instances if instance.miner_hotkey == unlucky]
                    )
                    unlucky_reason = (
                        "Selected an unlucky instance via miner duplicates: "
                        f"{chute.chute_id=} {unlucky_instance.instance_id=} "
                        f"{unlucky_instance.miner_hotkey=} {unlucky_instance.nodes[0].gpu_identifier=} "
                        f"{idx + 1} of {number_to_kick}"
                    )
                    logger.info(unlucky_reason)

                ###################################################################################
                #  XXX: This could be enabled - select the most expensive GPU to kick             #
                #  each interval... The reason it is not enabled, currently, is because           #
                #  the most expensive GPUs have the most diversity of chutes they are             #
                #  capable of running, so kicking out the more expensive GPUs could actually      #
                #  be counterproductive and incentivize people adding cheaper GPUs, which we      #
                #  want to avoid. Can revisit over time if it's worth doing so, but unlikely      #
                #  as we are moving towards confidential compute which requires hopper/blackwell. #
                ###################################################################################

                ## If each miner only has one, go by the most expensive GPU.
                # if not unlucky_instance:
                #    instance_multipliers = {
                #        instance.instance_id: {
                #            "mult": COMPUTE_MULTIPLIER[instance.nodes[0].gpu_identifier],
                #            "inst": instance,
                #        }
                #        for instance in chute.instances
                #    }
                #    max_multiplier = max([val["mult"] for val in instance_multipliers.values()])
                #    min_multiplier = min([val["mult"] for val in instance_multipliers.values()])
                #    if (
                #        min([val["mult"] for val in instance_multipliers.values()])
                #        != max_multiplier
                #    ):
                #        most_expensive = [
                #            val["inst"]
                #            for _, val in instance_multipliers.items()
                #            if val["mult"] == max_multiplier
                #        ]
                #        unlucky_instance = random.choice(most_expensive)
                #        unlucky_reason = (
                #            "Selected an unlucky instance via most expensive GPU: "
                #            f"{chute.chute_id=} {unlucky_instance.instance_id=} "
                #            f"{unlucky_instance.miner_hotkey=} {unlucky_instance.nodes[0].gpu_identifier=} "
                #            f"{min_multiplier=} vs {max_multiplier=}"
                #            f"{idx+1} of {number_to_kick}"
                #        )
                #        logger.info(unlucky_reason)

                # Random for now, but will be maxing geographical distribution once mechanism is in place.
                if not unlucky_instance:
                    established = [
                        instance
                        for instance in instances
                        if datetime.now(timezone.utc) - instance.created_at >= timedelta(hours=1)
                    ]
                    if established:
                        unlucky_instance = random.choice(established)
                        unlucky_reason = (
                            f"Selected an unlucky instance at random: {chute.chute_id=} "
                            f"{unlucky_instance.instance_id=} {unlucky_instance.miner_hotkey=} "
                            f"{idx + 1} of {number_to_kick}"
                        )
                        logger.info(unlucky_reason)

                # Purge the unlucky one.
                if unlucky_instance:
                    kicked.add(unlucky_instance.instance_id)
                    await purge_and_notify(unlucky_instance, reason=unlucky_reason)
                    instances_removed += 1
            else:
                logger.info(f"No need to downsize {chute_id}, count={len(instances)}")
    if instances_removed:
        logger.success(f"Scaled down {instances_removed} underutilized instances")
    return instances_removed


if __name__ == "__main__":
    asyncio.run(scale_down())
