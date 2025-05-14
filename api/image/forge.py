"""
Image forge -- build images and push to local registry with buildah.
"""

import asyncio
import zipfile
import uuid
import hashlib
import os
import re
import glob
import tempfile
import traceback
import time
import random
import base64
import orjson as json
from loguru import logger
from api.config import settings
from api.database import get_session
from api.exceptions import (
    UnsafeExtraction,
    BuildFailure,
    PushFailure,
    BuildTimeout,
    PushTimeout,
)
from api.image.schemas import Image
from api.fs_challenge.schemas import FSChallenge
from sqlalchemy import func
from sqlalchemy.future import select
from sqlalchemy.orm import joinedload
from taskiq import TaskiqEvents
from taskiq_redis import ListQueueBroker, RedisAsyncResultBackend
from api.database import orms  # noqa

broker = ListQueueBroker(url=settings.redis_url, queue_name="forge").with_result_backend(
    RedisAsyncResultBackend(redis_url=settings.redis_url, result_ex_time=3600)
)


@broker.on_event(TaskiqEvents.WORKER_STARTUP)
async def initialize(*_, **__):
    """
    Ensure ORM modules are all loaded, and login to docker hub to avoid rate-limiting.
    """
    import api.database.orms  # noqa: F401

    username = os.getenv("DOCKER_PULL_USERNAME")
    password = os.getenv("DOCKER_PULL_PASSWORD")
    if username and password:
        process = await asyncio.create_subprocess_exec(
            "buildah", "login", "-u", username, "-p", password, "docker.io"
        )
        await process.wait()
        if process.returncode == 0:
            logger.success(f"Authenticated to docker hub with {username=}")
        else:
            logger.warning(f"Failed authentication: {username=}")

    for base_image in ("parachutes/base-python:3.12.7", "parachutes/base-python:3.12.9"):
        process = await asyncio.create_subprocess_exec(
            "buildah",
            "pull",
            base_image,
        )
        await process.wait()
        if process.returncode == 0:
            logger.success("Succesfully warmed base image cache.")
        else:
            logger.warning("Failed to warm up base image.")


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
    short_tag = f"{image.user.username}/{image.name}:{image.tag}"
    full_image_tag = f"{settings.registry_host.rstrip('/')}/{short_tag}"

    # Helper to capture and stream logs.
    started_at = time.time()

    async def _capture_logs(stream, name):
        log_method = logger.info if name == "stdout" else logger.warning
        while True:
            line = await stream.readline()
            if line:
                decoded_line = line.decode().strip()
                log_method(f"[build {short_tag}]: {decoded_line}")
                with open("build.log", "a+") as outfile:
                    outfile.write(decoded_line.strip() + "\n")
                await settings.redis_client.xadd(
                    f"forge:{image.image_id}:stream",
                    {"data": json.dumps({"log_type": name, "log": decoded_line}).decode()},
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
            await settings.redis_client.xadd(f"forge:{image.image_id}:stream", {"data": "DONE"})
            raise BuildFailure(f"Build of {full_image_tag} failed!")
    except asyncio.TimeoutError:
        message = f"Build of {full_image_tag} timed out after {settings.build_timeout} seconds."
        logger.error(message)
        await settings.redis_client.xadd(
            f"forge:{image.image_id}:stream",
            {"data": json.dumps({"log_type": "stderr", "log": message}).decode()},
        )
        await settings.redis_client.xadd(f"forge:{image.image_id}:stream", {"data": "DONE"})
        process.kill()
        await process.communicate()
        raise BuildTimeout(message)

    # Scan with trivy.
    await settings.redis_client.xadd(
        f"forge:{image.image_id}:stream",
        {
            "data": json.dumps(
                {"log_type": "stdout", "log": "scanning image with trivy..."}
            ).decode()
        },
    )
    logger.info("Scanning image with trivy...")
    try:
        process = await asyncio.create_subprocess_exec(
            "bash",
            "/usr/local/bin/trivy_scan.sh",
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
            timeout=settings.scan_timeout,
        )
        if process.returncode == 0:
            message = f"No HIGH|CRITICAL vulnerabilities detected in {short_tag}"
            await settings.redis_client.xadd(
                f"forge:{image.image_id}:stream",
                {"data": json.dumps({"log_type": "stdout", "log": message}).decode()},
            )
            logger.success(message)
        else:
            message = f"Issues scanning {short_tag} with trivy!"
            await settings.redis_client.xadd(
                f"forge:{image.image_id}:stream",
                {"data": json.dumps({"log_type": "stderr", "log": message}).decode()},
            )
            logger.error(message)
            raise BuildFailure(f"Failed trivy image scan: {short_tag}")
    except asyncio.TimeoutError:
        message = f"Trivy scan of {short_tag} timed out after."
        logger.error(message)
        await settings.redis_client.xadd(
            f"forge:{image.image_id}:stream",
            {"data": json.dumps({"log_type": "stderr", "log": message}).decode()},
        )
        await settings.redis_client.xadd(f"forge:{image.image_id}:stream", {"data": "DONE"})
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
                "\N{HAMMER AND WRENCH} "
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
            await settings.redis_client.xadd(f"forge:{image.image_id}:stream", {"data": "DONE"})
            raise PushFailure(f"Push of {full_image_tag} failed!")
    except asyncio.TimeoutError:
        message = f"Push of {full_image_tag} timed out after {settings.push_timeout} seconds."
        logger.error(message)
        await settings.redis_client.xadd(
            f"forge:{image.image_id}:stream",
            {"data": json.dumps({"log_type": "stderr", "log": message}).decode()},
        )
        await settings.redis_client.xadd(f"forge:{image.image_id}:stream", {"data": "DONE"})
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
                destination = (
                    f"fschallenge/{image.user_id}/{image.image_id}/{os.path.basename(path)}"
                )
                async with settings.s3_client() as s3:
                    await s3.upload_file(path, settings.storage_bucket, destination)
                message = (
                    f"Successfully generated filesystem challenge data: {os.path.basename(path)}"
                )
                await settings.redis_client.xadd(
                    f"forge:{image.image_id}:stream",
                    {"data": json.dumps({"log_type": "stdout", "log": message}).decode()},
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
        "\N{HAMMER AND WRENCH} "
        + f" completed forging image {image.image_id} in {round(delta, 5)} seconds"
    )
    await settings.redis_client.xadd(
        f"forge:{image.image_id}:stream",
        {"data": json.dumps({"log_type": "stdout", "log": message}).decode()},
    )
    logger.success(message)
    await settings.redis_client.xadd(f"forge:{image.image_id}:stream", {"data": "DONE"})
    return short_tag


async def generate_fs_challenges(image_id: str):
    """
    Generate filesystem challenges after an image is created.
    """
    async with get_session() as session:
        image = (
            (
                await session.execute(
                    select(Image)
                    .options(joinedload(Image.fs_challenges))
                    .where(Image.image_id == image_id)
                )
            )
            .unique()
            .scalar_one_or_none()
        )
        if not image:
            logger.warning(f"Unable to locate {image_id=}")
            return
        if image.fs_challenges:
            logger.warning(f"Already have filesystem challenges for {image_id=}")

    challenges = []
    async with settings.s3_client() as s3:
        for suffix in ["newsample", "workdir", "corelibs"]:
            key = f"fschallenge/{image.user_id}/{image.image_id}/fschallenge_{suffix}.data"
            entries = []
            try:
                response = await s3.get_object(Bucket=settings.storage_bucket, Key=key)
                data = (await response["Body"].read()).decode("utf-8")
                entries = [line for line in data.splitlines() if line.strip()]
            except Exception as exc:
                logger.error(f"Error loading challenge data @ {key}: {exc}")
                continue
            for line in entries:
                file_data_match = re.match(
                    r"^(.*)?:__size__:(.*):__checksum__:.*?:__head__:(.*)?:__tail__:(.*)", line
                )
                if not file_data_match:
                    logger.warning(f"Bad challenge data found in {key}: {line}")
                    continue
                filename, size, head, tail = file_data_match.groups()
                head = None if head == "NONE" else base64.b64decode(head.encode())
                tail = None if tail == "NONE" else base64.b64decode(tail.encode())
                size = int(size)
                if not size or not head or len(head) <= 12:
                    continue
                current_count = len(challenges)
                target_count = {"newsample": 20, "workdir": 100, "corelibs": 50}[suffix]
                logger.info(f"Generating {target_count} challenges for {image_id=} {filename=}")
                while len(challenges) <= current_count + target_count:
                    length = random.randint(10, len(head))
                    offset = 0 if length == len(head) else random.randint(0, len(head) - length - 1)
                    challenges.append(
                        {
                            "filename": filename,
                            "length": length,
                            "offset": offset,
                            "type": suffix,
                            "expected": hashlib.sha256(head[offset : offset + length]).hexdigest(),
                        }
                    )
                    # if tail:
                    #    length = random.randint(10, len(tail))
                    #    offset = (
                    #        0
                    #        if length == len(tail)
                    #        else random.randint(size - len(tail) - length - 1, size - len(tail))
                    #    )
                    #    challenges.append(
                    #        {
                    #            "filename": filename,
                    #            "length": length,
                    #            "offset": offset,
                    #            "type": suffix,
                    #            "expected": hashlib.sha256(
                    #                tail[offset - size : offset - size + length]
                    #            ).hexdigest(),
                    #        }
                    #    )

    # Persist all of the challenges to DB.
    logger.info(f"About to save {len(challenges)} challenges for {image_id=}")
    async with get_session() as session:
        for challenge in challenges:
            session.add(
                FSChallenge(
                    challenge_id=str(uuid.uuid4()),
                    image_id=image_id,
                    filename=challenge["filename"],
                    offset=challenge["offset"],
                    length=challenge["length"],
                    expected=challenge["expected"],
                    challenge_type=challenge["type"],
                )
            )
        await session.commit()
    logger.success(
        f"Successfully persisted {len(challenges)} filesystem challenges for {image_id=}"
    )


@broker.task
async def forge(image_id: str):
    """
    Build an image and push it to the registry.
    """
    os.system("bash /usr/local/bin/buildah_cleanup.sh")
    async with get_session() as session:
        result = await session.execute(select(Image).where(Image.image_id == image_id).limit(1))
        image = result.scalar_one_or_none()
        if not image:
            logger.error(f"Image does not exist: {image_id=}")
            return
        if image.status != "pending build":
            logger.error(f"Image status is not pending: {image_id=} status={image.status}")
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
        dockerfile_path = os.path.join(build_dir, "Dockerfile")
        async with settings.s3_client() as s3:
            await s3.download_file(
                settings.storage_bucket, f"forge/{image.user_id}/{image_id}.zip", context_path
            )
        async with settings.s3_client() as s3:
            await s3.download_file(
                settings.storage_bucket,
                f"forge/{image.user_id}/{image_id}.Dockerfile",
                dockerfile_path,
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
        if os.path.exists(log_path := os.path.join(build_dir, "build.log")):
            destination = f"forge/{image.user_id}/{image.image_id}.log"
            async with settings.s3_client() as s3:
                await s3.upload_file(log_path, settings.storage_bucket, destination)

    # Cache a sampling of filesystem challenges.
    await generate_fs_challenges(image_id)

    # Update status.
    async with get_session() as session:
        result = await session.execute(select(Image).where(Image.image_id == image_id).limit(1))
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

    await settings.redis_client.publish(
        "miner_broadcast",
        json.dumps(
            {
                "reason": "image_created",
                "data": {
                    "image_id": image_id,
                },
            }
        ).decode(),
    )

    # Cleanup.
    os.system("bash /usr/local/bin/buildah_cleanup.sh")
