"""
Image forge -- build images and push to local registry with buildah.
"""

import asyncio
import zipfile
import os
import glob
import tempfile
import traceback
import time
import orjson as json
from loguru import logger
from api.config import settings
from api.database import SessionLocal
from api.exceptions import (
    UnsafeExtraction,
    BuildFailure,
    PushFailure,
    BuildTimeout,
    PushTimeout,
)
from api.image.schemas import Image
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
    import api.database.orms  # noqa: F401


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

    # Helper to capture and stream logs.
    started_at = time.time()

    async def _capture_logs(stream, name):
        log_method = logger.info if name == "stdout" else logger.warning
        with open(f"{name}.log", "w") as outfile:
            while True:
                line = await stream.readline()
                if line:
                    decoded_line = line.decode().strip()
                    log_method(f"[build {short_tag}]: {decoded_line}")
                    outfile.write(decoded_line.strip() + "\n")
                    await settings.redis_client.xadd(
                        f"forge:{image.image_id}:stream",
                        {
                            "data": json.dumps(
                                {"log_type": name, "log": decoded_line}
                            ).decode()
                        },
                    )
                else:
                    break

    # Build.
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

        await asyncio.wait_for(
            asyncio.gather(
                _capture_logs(process.stdout, "stdout"),
                _capture_logs(process.stderr, "stderr"),
                process.wait(),
            ),
            timeout=settings.build_timeout,
        )
        if process.returncode == 0:
            delta = time.time() - started_at
            message = f"Successfull built {full_image_tag} in {round(delta, 5)} seconds, pushing..."
            logger.success(message)
            await settings.redis_client.xadd(
                f"forge:{image.image_id}:stream",
                {"data": json.dumps({"log_type": "stdout", "log": message}).decode()},
            )
        else:
            message = "Image build failed, check logs for more details!"
            logger.error(message)
            await settings.redis_client.xadd(
                f"forge:{image.image_id}:stream",
                {"data": json.dumps({"log_type": "stderr", "log": message}).decode()},
            )
            await settings.redis_client.xadd(
                f"forge:{image.image_id}:stream", {"data": "DONE"}
            )
            raise BuildFailure(f"Build of {full_image_tag} failed!")
    except asyncio.TimeoutError:
        message = f"Build of {full_image_tag} timed out after {settings.build_timeout} seconds."
        logger.error(message)
        await settings.redis_client.xadd(
            f"forge:{image.image_id}:stream",
            {"data": json.dumps({"log_type": "stderr", "log": message}).decode()},
        )
        await settings.redis_client.xadd(
            f"forge:{image.image_id}:stream", {"data": "DONE"}
        )
        process.kill()
        await process.communicate()
        raise BuildTimeout(message)

    # Push.
    await settings.redis_client.xadd(
        f"forge:{image.image_id}:stream",
        {
            "data": json.dumps(
                {"log_type": "stdout", "log": "pushing image to registry..."}
            ).decode()
        },
    )
    try:
        verify = str(not settings.registry_insecure).lower()
        process = await asyncio.create_subprocess_exec(
            "buildah",
            f"--tls-verify={verify}",
            "push",
            full_image_tag,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        await asyncio.wait_for(
            asyncio.gather(
                _capture_logs(process.stdout, "stdout"),
                _capture_logs(process.stderr, "stderr"),
                process.wait(),
            ),
            timeout=settings.build_timeout,
        )
        if process.returncode == 0:
            logger.success(f"Successfull pushed {full_image_tag}, done!")
            delta = time.time() - started_at
            message = (
                "\N{hammer and wrench} "
                + f" finished pushing image {image.image_id} in {round(delta, 5)} seconds"
            )
            await settings.redis_client.xadd(
                f"forge:{image.image_id}:stream",
                {"data": json.dumps({"log_type": "stdout", "log": message}).decode()},
            )
            logger.success(message)
        else:
            message = "Image push failed, check logs for more details!"
            logger.error(message)
            await settings.redis_client.xadd(
                f"forge:{image.image_id}:stream",
                {"data": json.dumps({"log_type": "stderr", "log": message}).decode()},
            )
            await settings.redis_client.xadd(
                f"forge:{image.image_id}:stream", {"data": "DONE"}
            )
            raise PushFailure(f"Push of {full_image_tag} failed!")
    except asyncio.TimeoutError:
        message = (
            f"Push of {full_image_tag} timed out after {settings.push_timeout} seconds."
        )
        logger.error(message)
        await settings.redis_client.xadd(
            f"forge:{image.image_id}:stream",
            {"data": json.dumps({"log_type": "stderr", "log": message}).decode()},
        )
        await settings.redis_client.xadd(
            f"forge:{image.image_id}:stream", {"data": "DONE"}
        )
        process.kill()
        await process.communicate()
        raise PushTimeout(
            f"Push of {full_image_tag} timed out after {settings.push_timeout} seconds."
        )

    # Generate filesystem challenge data.
    message = "Generating filesystem challenge data..."
    await settings.redis_client.xadd(
        f"forge:{image.image_id}:stream",
        {"data": json.dumps({"log_type": "stdout", "log": message}).decode()},
    )
    logger.info(message)
    try:
        process = await asyncio.create_subprocess_exec(
            "bash",
            "/usr/local/bin/generate_fs_challenge.sh",
            short_tag,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        await asyncio.wait_for(
            asyncio.gather(
                _capture_logs(process.stdout, "stdout"),
                _capture_logs(process.stderr, "stderr"),
                process.wait(),
            ),
            timeout=300,
        )
        if process.returncode == 0:
            for path in glob.glob("/tmp/fschallenge_*.data"):
                destination = f"fschallenge/{image.user_id}/{image.image_id}/{os.path.basename(path)}"
                await settings.storage_client.put_object(
                    settings.storage_bucket,
                    destination,
                    open(path, "rb"),
                    length=-1,
                    part_size=10 * 1024 * 1024,
                )
                message = f"Successfully generated filesystem challenge data: {os.path.basename(path)}"
                await settings.redis_client.xadd(
                    f"forge:{image.image_id}:stream",
                    {
                        "data": json.dumps(
                            {"log_type": "stdout", "log": message}
                        ).decode()
                    },
                )
                logger.success(message)
        else:
            message = "Error generating filesystem challenge data."
            await settings.redis_client.xadd(
                f"forge:{image.image_id}:stream",
                {"data": json.dumps({"log_type": "stderr", "log": message}).decode()},
            )
            logger.error(message)
    except Exception as exc:
        try:
            message = f"Error generating filesystem challenge data: {exc}"
            logger.error(message)
            await settings.redis_client.xadd(
                f"forge:{image.image_id}:stream",
                {"data": json.dumps({"log_type": "stderr", "log": message}).decode()},
            )
            process.kill()
            await process.communicate()
        except Exception:
            ...

    # DONE!
    delta = time.time() - started_at
    message = (
        "\N{hammer and wrench} "
        + f" completed forging image {image.image_id} in {round(delta, 5)} seconds"
    )
    await settings.redis_client.xadd(
        f"forge:{image.image_id}:stream",
        {"data": json.dumps({"log_type": "stdout", "log": message}).decode()},
    )
    logger.success(message)
    await settings.redis_client.xadd(f"forge:{image.image_id}:stream", {"data": "DONE"})
    return short_tag


@broker.task
async def forge(image_id: str):
    """
    Build an image and push it to the registry.
    """
    os.system("bash /usr/local/bin/buildah_cleanup.sh")
    async with SessionLocal() as session:
        result = await session.execute(
            select(Image).where(Image.image_id == image_id).limit(1)
        )
        image = result.scalar_one_or_none()
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
            f"forge/{image.user_id}/{image_id}.zip",
            context_path,
        )
        await settings.storage_client.fget_object(
            settings.storage_bucket,
            f"forge/{image.user_id}/{image_id}.Dockerfile",
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

        # Upload logs.
        log_paths = []
        if os.path.exists(log_path := os.path.join(build_dir, "stdout.log")):
            log_paths.append(log_path)
        if os.path.exists(log_path := os.path.join(build_dir, "sterror.log")):
            log_paths.append(log_path)
        for path in log_paths:
            destination = (
                f"forge/{image.user_id}/{image.image_id}.{os.path.basename(path)}"
            )
            await settings.storage_client.put_object(
                settings.storage_bucket,
                destination,
                open(path, "rb"),
                length=-1,
                part_size=10 * 1024 * 1024,
            )

    # Update status.
    async with SessionLocal() as session:
        result = await session.execute(
            select(Image).where(Image.image_id == image_id).limit(1)
        )
        image = result.scalar_one_or_none()
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

    # Cleanup.
    os.system("bash /usr/local/bin/buildah_cleanup.sh")
