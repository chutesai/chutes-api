"""
Image forge -- build images and push to local registry with buildah.
"""

import asyncio
import zipfile
import os
import tempfile
import traceback
from loguru import logger
from run_api.config import settings
from run_api.database import SessionLocal
from run_api.exceptions import (
    UnsafeExtraction,
    BuildFailure,
    PushFailure,
    BuildTimeout,
    PushTimeout,
)
from run_api.image.schemas import Image
from sqlalchemy import func
from sqlalchemy.future import select
from taskiq import TaskiqEvents
from taskiq_redis import ListQueueBroker, RedisAsyncResultBackend


broker = ListQueueBroker(
    url=settings.redis_url, queue_name="forge"
).with_result_backend(
    RedisAsyncResultBackend(redis_url=settings.redis_url, result_ex_time=3600)
)


@broker.on_event(TaskiqEvents.WORKER_STARTUP)
async def initialize_mappings(*_, **__):
    """
    Ensure ORM modules are all loaded.
    """
    from run_api.chute.schemas import Chute
    from run_api.user.schemas import User


def safe_extract(zip_path):
    """
    Safer way to extract zip archives, preventing creation of files out of current directory.
    """
    base_dir = os.path.dirname(os.path.abspath(zip_path))
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        for member in zip_ref.namelist():
            target_path = os.path.normpath(os.path.join(base_dir, member))
            if not target_path.startswith(base_dir):
                raise UnsafeExtraction(f"Unsafe path detected: {member}")
            zip_ref.extract(member, base_dir)


async def build_and_push_image(image):
    """
    Perform the actual image build via buildah.
    """
    short_tag = f"{image.user.user_id}/{image.name}:{image.tag}"
    full_image_tag = f"{settings.registry_host.rstrip('/')}/{short_tag}"

    try:
        process = await asyncio.create_subprocess_exec(
            "buildah",
            "build",
            "--tag",
            full_image_tag,
            "--tag",
            short_tag,
            ".",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await asyncio.wait_for(
            process.communicate(), timeout=settings.build_timeout
        )
        logger.info(f"[build stdout]\n{stdout.decode()}")
        logger.error(f"[build stderr]\n{stderr.decode()}")
        if process.returncode == 0:
            logger.success(f"Successfull built {full_image_tag}, pushing...")
        else:
            logger.error("Image build failed, check logs for more details!")
            raise BuildFailure(f"Build of {full_image_tag} failed!")
    except asyncio.TimeoutError:
        logger.error(f"Error waiting for build of {full_image_tag} to finish!")
        process.kill()
        await process.communicate()
        raise BuildTimeout(
            f"Build of {full_image_tag} timed out after {settings.build_timeout} seconds."
        )

    # Push the image.
    try:
        process = await asyncio.create_subprocess_exec(
            "buildah",
            "push",
            full_image_tag,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await asyncio.wait_for(
            process.communicate(), timeout=settings.push_timeout
        )
        logger.info(f"[push stdout]\n{stdout.decode()}")
        logger.error(f"[push stderr]\n{stderr.decode()}")
        if process.returncode == 0:
            logger.success(f"Successfull pushed {full_image_tag}, done!")
        else:
            logger.error("Image push failed, check logs for more details!")
            raise PushFailure(f"Push of {full_image_tag} failed!")
    except asyncio.TimeoutError:
        logger.error("Error waiting for image push to finish!")
        process.kill()
        await process.communicate()
        raise PushTimeout(
            f"Push of {full_image_tag} timed out after {settings.push_timeout} seconds."
        )
    return short_tag


@broker.task
async def forge(image_id: str):
    """
    Build an image and push it to the registry.
    """
    async with SessionLocal() as session:
        result = await session.execute(
            select(Image).where(Image.image_id == image_id).limit(1)
        )
        image = result.scalars().first()
        if not image:
            logger.error(f"Image does not exist: {image_id=}")
            return
        if image.status != "pending build":
            logger.error(
                f"Image status is not pending: {image_id=} status={image.status}"
            )
            return
        image.status = "building"
        image.build_started_at = func.now()
        await session.commit()
        await session.refresh(image)

    # Download the build context.
    short_tag = None
    error_message = None
    with tempfile.TemporaryDirectory() as build_dir:
        context_path = os.path.join(build_dir, "chute.zip")
        await settings.storage_client.fget_object(
            settings.storage_bucket,
            f"build-contexts/{image.user_id}/{image_id}.zip",
            context_path,
        )
        await settings.storage_client.fget_object(
            settings.storage_bucket,
            f"build-contexts/{image.user_id}/{image_id}.Dockerfile",
            os.path.join(build_dir, "Dockerfile"),
        )
        try:
            starting_dir = os.getcwd()
            os.chdir(build_dir)
            safe_extract(context_path)
            short_tag = await build_and_push_image(image)
        except Exception as exc:
            logger.error(f"Error building {image_id=}: {exc}\n{traceback.format_exc()}")
            error_message = str(exc)
        finally:
            os.chdir(starting_dir)

    # Update status.
    async with SessionLocal() as session:
        result = await session.execute(
            select(Image).where(Image.image_id == image_id).limit(1)
        )
        image = result.scalars().first()
        if not image:
            logger.warning(f"Image vanished while building! {image_id}")
            return
        image.status = "built and pushed" if short_tag else "error"
        if short_tag:
            image.status = "built and pushed"
            image.short_tag = short_tag
            image.build_completed_at = func.now()
        else:
            image.status = f"error: {error_message}"
        await session.commit()
        await session.refresh(image)
