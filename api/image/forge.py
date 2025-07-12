"""
Image forge -- build images and push to local registry with buildah.
"""

import asyncio
import zipfile
import uuid
import os
import hashlib
import tempfile
import traceback
import time
import shutil
import chutes
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
from api.chute.schemas import Chute
from sqlalchemy import func
from sqlalchemy.future import select
from taskiq import TaskiqEvents
from taskiq_redis import ListQueueBroker, RedisAsyncResultBackend
from api.database import orms  # noqa
from api.graval_worker import handle_rolling_update

broker = ListQueueBroker(url=settings.redis_url, queue_name="forge").with_result_backend(
    RedisAsyncResultBackend(redis_url=settings.redis_url, result_ex_time=3600)
)
CFSV_PATH = os.path.join(os.path.dirname(chutes.__file__), "cvfs")


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

    for base_image in ("parachutes/python:3.12.9",):
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


async def build_and_push_image(image, build_dir):
    """
    Perform the actual image build via buildah with filesystem verification.
    """
    base_tag = f"{image.user.username}/{image.name}:{image.tag}"
    if image.patch_version and image.patch_version != "initial":
        short_tag = f"{base_tag}-{image.patch_version}"
    else:
        short_tag = base_tag
    full_image_tag = f"{settings.registry_host.rstrip('/')}/{short_tag}"

    # Copy cfsv binary to build directory
    build_cfsv_path = os.path.join(build_dir, "cvfs")
    shutil.copy2(CFSV_PATH, build_cfsv_path)

    # Modify the Dockerfile to include filesystem verification stages
    original_dockerfile = os.path.join(build_dir, "Dockerfile")
    modified_dockerfile = await inject_cfsv_stages(original_dockerfile, build_cfsv_path)

    # Helper to capture and stream logs (same as before)
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

    # Build with buildah.
    try:
        storage_driver = os.getenv("STORAGE_DRIVER", "overlay")
        storage_opts = os.getenv("STORAGE_OPTS", "overlay.mount_program=/usr/bin/fuse-overlayfs")

        # Build all stages including the intermediate one
        build_cmd = [
            "buildah",
            "build",
            "--isolation",
            "chroot",
            "--storage-driver",
            storage_driver,
            "--layers",
            "--tag",
            f"{short_tag}:filesystemverificationmanager",
            "--target",
            "filesystemverificationmanager",
            "-f",
            modified_dockerfile,
        ]
        if storage_driver == "overlay" and storage_opts:
            for opt in storage_opts.split(","):
                build_cmd.extend(["--storage-opt", opt.strip()])
        build_cmd.append(".")

        # First build the intermediate stage
        process = await asyncio.create_subprocess_exec(
            *build_cmd,
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

        if process.returncode != 0:
            raise BuildFailure("Build of intermediate stage failed!")

        # Extract the filesystem data
        data_file_path = await extract_cfsv_data(short_tag, build_dir)
        await upload_filesystem_verification_data(image, data_file_path)

        # Now build the final image
        build_cmd = [
            "buildah",
            "build",
            "--isolation",
            "chroot",
            "--storage-driver",
            storage_driver,
            "--layers",
            "--tag",
            full_image_tag,
            "--tag",
            short_tag,
            "--target",
            "final_layer",
            "-f",
            modified_dockerfile,
        ]
        if storage_driver == "overlay" and storage_opts:
            for opt in storage_opts.split(","):
                build_cmd.extend(["--storage-opt", opt.strip()])
        build_cmd.append(".")

        process = await asyncio.create_subprocess_exec(
            *build_cmd,
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
            message = (
                f"Successfully built {full_image_tag} in {round(delta, 5)} seconds, pushing..."
            )
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


async def inject_cfsv_stages(dockerfile_path: str, cfsv_binary_path: str) -> str:
    """
    Update the dockerfile to use the three-stage build to generate filesystem challenge data/index.
    """
    with open(dockerfile_path, "r") as f:
        original_dockerfile = f.read()
    lines = original_dockerfile.strip().split("\n")
    last_from_idx = -1
    for i, line in enumerate(lines):
        if line.strip().upper().startswith("FROM"):
            last_from_idx = i
    if last_from_idx == -1:
        raise ValueError("No FROM statement found in Dockerfile")
    from_line = lines[last_from_idx]
    if " AS " in from_line.upper():
        base_alias = from_line.split()[-1]
    else:
        base_alias = "base_layer"
        lines[last_from_idx] = f"{from_line} AS {base_alias}"
    new_dockerfile_lines = []
    new_dockerfile_lines.extend(lines[: last_from_idx + 1])
    new_dockerfile_lines.extend(lines[last_from_idx + 1 :])
    new_dockerfile_lines.extend(
        [
            "",
            f"FROM {base_alias} AS filesystemverificationmanager",
            f"COPY {os.path.basename(cfsv_binary_path)} /tmp/cfsv",
            "RUN chmod +x /tmp/cfsv",
            "RUN /tmp/cfsv index / /tmp/chutesfs.index",
            "RUN /tmp/cfsv collect / /tmp/chutesfs.index /tmp/chutesfs.data",
            "",
            f"FROM {base_alias} AS final_layer",
            "COPY --from=filesystemverificationmanager /tmp/chutesfs.index /etc/chutesfs.index",
        ]
    )
    modified_dockerfile_path = dockerfile_path + ".modified"
    with open(modified_dockerfile_path, "w") as f:
        f.write("\n".join(new_dockerfile_lines))
    return modified_dockerfile_path


async def extract_cfsv_data(image_tag: str, build_dir: str) -> str:
    """
    Extract the filesystem verification data from the filesystemverificationmanager stage.
    Returns the path to the extracted data file.
    """
    container_name = f"cfsv-extract-{uuid.uuid4().hex[:8]}"
    try:
        process = await asyncio.create_subprocess_exec(
            "buildah",
            "from",
            "--name",
            container_name,
            f"{image_tag}:filesystemverificationmanager",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await process.communicate()
        if process.returncode != 0:
            raise Exception(
                f"Failed to create container from intermediate stage: {stderr.decode()}"
            )
        data_file_path = os.path.join(build_dir, "chutesfs.data")
        process = await asyncio.create_subprocess_exec(
            "buildah",
            "copy",
            "--from",
            container_name,
            "/tmp/chutesfs.data",
            data_file_path,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await process.communicate()
        if process.returncode != 0:
            raise Exception(f"Failed to copy data file: {stderr.decode()}")
        return data_file_path
    finally:
        process = await asyncio.create_subprocess_exec(
            "buildah",
            "rm",
            container_name,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        await process.communicate()


async def upload_filesystem_verification_data(image, data_file_path: str):
    """
    Upload the filesystem verification data to S3 with the correct path structure.
    """
    s3_key = f"image_hash_blobs/{image.image_id}/{image.patch_version}.data"
    async with settings.s3_client() as s3:
        await s3.upload_file(data_file_path, settings.storage_bucket, s3_key)
    logger.success(f"Uploaded filesystem verification data to {s3_key}")


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
            short_tag = await build_and_push_image(image, build_dir)
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


@broker.task
async def update_chutes_lib(image_id: str, chutes_version: str):
    """
    Update the chutes library in an existing image without rebuilding from scratch.
    """
    async with get_session() as session:
        result = await session.execute(select(Image).where(Image.image_id == image_id).limit(1))
        image = result.scalar_one_or_none()
        if not image:
            logger.error(f"Image does not exist: {image_id=}")
            return
        if image.chutes_version == chutes_version:
            logger.info(f"Image {image_id} already has chutes version {chutes_version}")
            return
        await session.refresh(image, ["user"])

    patch_version = hashlib.sha256(f"{image_id}:{chutes_version}".encode()).hexdigest()[:12]

    # Determine source and target tags
    base_tag = f"{image.user.username}/{image.name}:{image.tag}"
    if image.patch_version and image.patch_version != "initial":
        source_tag = f"{base_tag}-{image.patch_version}"
    else:
        source_tag = base_tag
    target_tag = f"{base_tag}-{patch_version}"
    full_source_tag = f"{settings.registry_host.rstrip('/')}/{source_tag}"
    full_target_tag = f"{settings.registry_host.rstrip('/')}/{target_tag}"

    # Rebuild the image with the updated chutes lib.
    error_message = None
    success = False
    with tempfile.TemporaryDirectory() as build_dir:
        try:
            build_cfsv_path = os.path.join(build_dir, "cvfs")
            shutil.copy2(CFSV_PATH, build_cfsv_path)
            dockerfile_content = f"""FROM {full_source_tag} AS base_layer
RUN pip install --upgrade chutes=={chutes_version}

FROM base_layer AS filesystemverificationmanager
COPY cvfs /tmp/cvfs
RUN chmod +x /tmp/cvfs
RUN /tmp/cvfs index / /tmp/chutesfs.index
RUN /tmp/cvfs collect / /tmp/chutesfs.index /tmp/chutesfs.data

FROM base_layer AS final_layer
COPY --from=filesystemverificationmanager /tmp/chutesfs.index /etc/chutesfs.index
RUN pip install --upgrade chutes=={chutes_version}
"""

            dockerfile_path = os.path.join(build_dir, "Dockerfile")
            with open(dockerfile_path, "w") as f:
                f.write(dockerfile_content)

            logger.info(f"Updating chutes library in {source_tag} to version {chutes_version}")

            storage_driver = os.getenv("STORAGE_DRIVER", "overlay")
            storage_opts = os.getenv(
                "STORAGE_OPTS", "overlay.mount_program=/usr/bin/fuse-overlayfs"
            )
            build_cmd = [
                "buildah",
                "build",
                "--isolation",
                "chroot",
                "--storage-driver",
                storage_driver,
                "--layers",
                "--tag",
                f"{target_tag}:filesystemverificationmanager",
                "--target",
                "filesystemverificationmanager",
                "-f",
                dockerfile_path,
            ]
            if storage_driver == "overlay" and storage_opts:
                for opt in storage_opts.split(","):
                    build_cmd.extend(["--storage-opt", opt.strip()])
            build_cmd.append(build_dir)
            process = await asyncio.create_subprocess_exec(
                *build_cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await process.communicate()
            if process.returncode != 0:
                raise BuildFailure(f"Failed to build intermediate stage: {stderr.decode()}")

            data_file_path = await extract_cfsv_data(target_tag, build_dir)

            # Upload to S3
            s3_key = f"image_hash_blobs/{image_id}/{patch_version}.data"
            async with settings.s3_client() as s3:
                await s3.upload_file(data_file_path, settings.storage_bucket, s3_key)
            logger.success(f"Uploaded filesystem verification data to {s3_key}")

            # Build final image
            build_cmd = [
                "buildah",
                "build",
                "--isolation",
                "chroot",
                "--storage-driver",
                storage_driver,
                "--layers",
                "--tag",
                full_target_tag,
                "--tag",
                target_tag,
                "--target",
                "final_layer",
                "-f",
                dockerfile_path,
            ]
            if storage_driver == "overlay" and storage_opts:
                for opt in storage_opts.split(","):
                    build_cmd.extend(["--storage-opt", opt.strip()])
            build_cmd.append(build_dir)
            process = await asyncio.create_subprocess_exec(
                *build_cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await process.communicate()
            if process.returncode != 0:
                raise BuildFailure(f"Failed to build final image: {stderr.decode()}")

            # Push to registry
            verify = str(not settings.registry_insecure).lower()
            process = await asyncio.create_subprocess_exec(
                "buildah",
                f"--tls-verify={verify}",
                "push",
                full_target_tag,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await process.communicate()
            if process.returncode != 0:
                raise PushFailure(f"Failed to push image: {stderr.decode()}")
            logger.success(f"Successfully pushed updated image {full_target_tag}")
            success = True
        except Exception as exc:
            logger.error(
                f"Error updating chutes lib for {image_id}: {exc}\n{traceback.format_exc()}"
            )
            error_message = str(exc)

    # Update the image with the new patch version, tag, etc.
    if success:
        affected_chute_ids = []
        async with get_session() as session:
            result = await session.execute(select(Image).where(Image.image_id == image_id).limit(1))
            image = result.scalar_one_or_none()
            if image:
                image.patch_version = patch_version
                image.chutes_version = chutes_version
                image.short_tag = target_tag
                await session.commit()
                logger.success(
                    f"Updated image {image_id} to chutes version {chutes_version}, patch version {patch_version}"
                )
                chutes_result = await session.execute(
                    select(Chute.chute_id, Chute.chutes_version).where(Chute.image_id == image_id)
                )
                affected_chute_ids = []
                for row in chutes_result.fetchall():
                    await handle_rolling_update.kiq(
                        row[0], row[1], reason="image updated due to chutes lib upgrade"
                    )
                logger.info(f"Found {len(affected_chute_ids)} chutes affected by image update")

                # Notify miners of the update
                image_path = f"{image.user.username}/{image.name}:{image.tag}-{patch_version}"
                await settings.redis_client.publish(
                    "miner_broadcast",
                    json.dumps(
                        {
                            "reason": "image_updated",
                            "data": {
                                "image_id": image_id,
                                "short_tag": image.short_tag,
                                "patch_version": patch_version,
                                "chutes_version": chutes_version,
                                "chute_ids": affected_chute_ids,
                                "image": image_path,
                            },
                        }
                    ).decode(),
                )
    else:
        logger.error(f"Failed to update chutes lib for image {image_id}: {error_message}")

    # Cleanup
    os.system("bash /usr/local/bin/buildah_cleanup.sh")
