"""
Re-sign existing registry images under the new static hostname.

Before the static-hostname change, images were signed as:
    {VALIDATOR_SS58}.localregistry.chutes.ai:5000/{repo}@{digest}

After the change, new images are signed as:
    localregistry.chutes.ai:5000/{repo}@{digest}

This script re-signs existing images under the new hostname so that VMs booting
with the new firmware (which looks for signatures under the static hostname) can
verify images that were originally built before the migration.

Usage (from inside the forge pod, where cosign + registry access are available):

    # Dry run (preview what would be signed):
    python scripts/resign_images.py --dry-run

    # Re-sign images built in the last 365 days (default):
    python scripts/resign_images.py

    # Re-sign images built in the last 90 days:
    python scripts/resign_images.py --days 90

    # Re-sign a specific image by image_id:
    python scripts/resign_images.py --image-id <image_id>

Requirements:
    - Run from the forge pod (cosign binary at PATH, key at /etc/cosign/cosign.key,
      /etc/hosts entry resolving localregistry.chutes.ai to the internal registry IP)
    - COSIGN_KEY and COSIGN_PASSWORD env vars must be set (already present in forge pod)
    - POSTGRESQL env var must point to the database
"""

import argparse
import asyncio
import os
import sys
from datetime import datetime, timezone, timedelta

from loguru import logger
from sqlalchemy import select
from sqlalchemy.orm import selectinload

from api.config import settings
from api.database import get_session
from api.image.schemas import Image
from api.image.forge import get_image_digest


INTERNAL_REGISTRY = "registry"
NEW_EXTERNAL_REGISTRY = "localregistry.chutes.ai"


async def resign_image(image: Image, dry_run: bool) -> bool:
    """Re-sign a single image under the new static hostname.

    Returns True on success (including dry-run), False on failure.
    """
    base_tag = f"{image.user.username}/{image.name}:{image.tag}".lower()
    if image.patch_version and image.patch_version != "initial":
        short_tag = f"{base_tag}-{image.patch_version}"
    else:
        short_tag = base_tag

    full_image_tag = f"{settings.registry_host.rstrip('/')}/{short_tag}"

    try:
        image_digest = await get_image_digest(full_image_tag)
    except Exception as exc:
        logger.error(f"  [{image.image_id}] Failed to get digest for {full_image_tag}: {exc}")
        return False

    # Build the digest reference and rewrite to the new external hostname
    repo = full_image_tag.rsplit(":", 1)[0]
    digest_ref = f"{repo}@{image_digest}"
    new_ref = digest_ref.replace(f"{INTERNAL_REGISTRY}:5000", f"{NEW_EXTERNAL_REGISTRY}:5000")

    if dry_run:
        logger.info(f"  [{image.image_id}] [dry-run] would sign: {new_ref}")
        return True

    cosign_key = settings.cosign_key
    cosign_password = settings.cosign_password
    if not cosign_key or not cosign_password:
        logger.error("COSIGN_KEY and COSIGN_PASSWORD must be set")
        return False

    process = await asyncio.create_subprocess_exec(
        "cosign",
        "sign",
        "--allow-http-registry",
        "--key",
        str(cosign_key),
        new_ref,
        "--yes",
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout, stderr = await process.communicate(input=f"{cosign_password}\n".encode())

    if process.returncode == 0:
        logger.success(f"  [{image.image_id}] Signed: {new_ref}")
        return True
    else:
        logger.error(
            f"  [{image.image_id}] Sign failed for {new_ref}: {stderr.decode().strip()}"
        )
        return False


async def main(dry_run: bool, days: int, image_id: str | None) -> None:
    async with get_session() as session:
        query = (
            select(Image)
            .options(selectinload(Image.user))
            .where(Image.status.like("built%"))
        )

        if image_id:
            query = query.where(Image.image_id == image_id)
        else:
            cutoff = datetime.now(timezone.utc) - timedelta(days=days)
            query = query.where(Image.build_completed_at >= cutoff)

        query = query.order_by(Image.build_completed_at.asc())

        result = await session.execute(query)
        images = result.scalars().all()

    if not images:
        logger.info("No images found matching criteria.")
        return

    logger.info(
        f"Found {len(images)} image(s) to re-sign under {NEW_EXTERNAL_REGISTRY} "
        f"{'(dry-run)' if dry_run else ''}"
    )

    succeeded = 0
    failed = 0
    for image in images:
        logger.info(
            f"Processing [{image.image_id}] {image.user.username}/{image.name}:{image.tag} "
            f"patch={image.patch_version} built={image.build_completed_at}"
        )
        ok = await resign_image(image, dry_run=dry_run)
        if ok:
            succeeded += 1
        else:
            failed += 1

    logger.info(
        f"\nDone. total={len(images)} succeeded={succeeded} failed={failed}"
        + (" (dry-run, no images were actually signed)" if dry_run else "")
    )

    if failed:
        sys.exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview which images would be re-signed without actually signing",
    )
    parser.add_argument(
        "--days",
        type=int,
        default=365,
        help="Re-sign images built within the last N days (default: 365)",
    )
    parser.add_argument(
        "--image-id",
        default=None,
        help="Re-sign a specific image by image_id (ignores --days)",
    )
    args = parser.parse_args()

    asyncio.run(main(dry_run=args.dry_run, days=args.days, image_id=args.image_id))
