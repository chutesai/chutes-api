"""
Migrate an integrated subnet chute to TEE with pro_6000 node selector.

Updates:
- tee=True
- node_selector.include = ["pro_6000"] (preserves gpu_count)
- Code AST: node_selector include, tee=True
- For affine chutes: also forces TP=1 / DP=gpu_count engine_args

Publishes chute_updated event over websocket after committing.

Usage:
  python scripts/migrate_to_tee.py <chute_id>              # dry-run
  python scripts/migrate_to_tee.py <chute_id> --apply      # persist changes
"""

import sys
import uuid
import asyncio
import orjson as json
from loguru import logger
from api.config import settings
from api.database import get_session
from api.chute.schemas import Chute
from api.affine import transform_code_for_tee
from api.constants import INTEGRATED_SUBNETS
from sqlalchemy import select, func


def _is_affine_chute(name: str) -> bool:
    return "affine" in name.lower()


async def main():
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <chute_id> [--apply]")
        sys.exit(1)

    chute_id = sys.argv[1]
    dry_run = "--apply" not in sys.argv

    if dry_run:
        logger.info("DRY RUN mode (pass --apply to persist changes)")
    else:
        logger.warning("APPLY mode: changes will be persisted")

    async with get_session() as session:
        chute = (
            await session.execute(select(Chute).where(Chute.chute_id == chute_id))
        ).scalar_one_or_none()

        if chute is None:
            logger.error(f"Chute {chute_id} not found")
            sys.exit(1)

        # Verify it's an integrated subnet chute.
        is_subnet = any(
            info["model_substring"] in chute.name.lower() for info in INTEGRATED_SUBNETS.values()
        )
        if not is_subnet:
            logger.error(f"Chute {chute.name} ({chute_id}) is not an integrated subnet chute")
            sys.exit(1)

        is_affine = _is_affine_chute(chute.name)
        gpu_count = chute.node_selector.get("gpu_count", 1)

        logger.info(f"Chute: {chute.name} ({chute_id})")
        logger.info(f"  tee: {chute.tee} -> True")
        logger.info(f"  node_selector: {chute.node_selector}")
        logger.info(f"  -> include=['pro_6000'], gpu_count={gpu_count}")
        if is_affine:
            logger.info(f"  Affine chute — will also force TP=1, DP={gpu_count}")

        # Transform code.
        new_code = transform_code_for_tee(chute.code, gpu_count, is_affine)
        if new_code is None:
            logger.error("Failed to transform code AST")
            sys.exit(1)

        if new_code != chute.code:
            logger.info("Code transformed successfully")
        else:
            logger.info("Code unchanged")

        if dry_run:
            print(f"\n--- New code ---\n{new_code}\n")
            print("Run with --apply to persist changes.")
            return

        # Apply changes.
        chute.tee = True
        new_ns = dict(chute.node_selector)
        new_ns["include"] = ["pro_6000"]
        chute.node_selector = new_ns
        chute.code = new_code
        chute.version = str(uuid.uuid5(uuid.NAMESPACE_OID, f"{chute.image_id}:{new_code}"))
        chute.updated_at = func.now()

        await session.commit()
        await session.refresh(chute)

        # Publish event to miners.
        await settings.redis_client.publish(
            "miner_broadcast",
            json.dumps(
                {
                    "reason": "chute_updated",
                    "data": {
                        "chute_id": chute.chute_id,
                        "version": chute.version,
                        "job_only": not chute.cords,
                    },
                }
            ).decode(),
        )

        logger.success(f"Migrated {chute.name} ({chute_id}) to TEE with pro_6000")


asyncio.run(main())
