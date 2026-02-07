"""
Image forge -- build images and push to local registry with buildah.
"""

import asyncio
from typing import Any, Callable
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
    SignFailure,
    SignTimeout,
    UnsafeExtraction,
    BuildFailure,
    PushFailure,
    BuildTimeout,
    PushTimeout,
)
from api.util import semcomp
from api.image.schemas import Image
from api.chute.schemas import Chute, RollingUpdate
from api.graval_worker import handle_rolling_update
from sqlalchemy import func, text, update
from sqlalchemy.orm import selectinload
from sqlalchemy.future import select
from api.database import orms  # noqa

CFSV_PATH = os.path.join(os.path.dirname(chutes.__file__), "cfsv")
CFSV_V2_PATH = f"{CFSV_PATH}_v2"
CFSV_V3_PATH = f"{CFSV_PATH}_v3"


async def initialize():
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

    for base_image in ("parachutes/python:3.12",):
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
    Perform the actual image build via buildah.
    """
    base_tag = f"{image.user.username}/{image.name}:{image.tag}".lower()
    if image.patch_version and image.patch_version != "initial":
        short_tag = f"{base_tag}-{image.patch_version}"
    else:
        short_tag = base_tag
    full_image_tag = f"{settings.registry_host.rstrip('/')}/{short_tag}"

    # Copy cfsv binary to build directory
    build_cfsv_path = os.path.join(build_dir, "cfsv")
    if semcomp(image.chutes_version or "0.0.0", "0.5.2") >= 0:
        shutil.copy2(CFSV_V3_PATH, build_cfsv_path)
    elif semcomp(image.chutes_version or "0.0.0", "0.4.6") >= 0:
        shutil.copy2(CFSV_V2_PATH, build_cfsv_path)
    else:
        shutil.copy2(CFSV_PATH, build_cfsv_path)
    os.chmod(build_cfsv_path, 0o755)

    # Helper to capture and stream logs
    started_at = time.time()

    async def _capture_logs(stream, name, capture=True):
        """Helper to capture logs. Set capture=False to suppress logs."""
        if not capture:
            # Just consume the stream without logging
            while True:
                line = await stream.readline()
                if not line:
                    break
            return

        log_method = logger.info if name == "stdout" else logger.warning
        while True:
            line = await stream.readline()
            if line:
                decoded_line = line.decode().strip()
                log_method(f"[build {short_tag}]: {decoded_line}")
                with open("build.log", "a+") as outfile:
                    outfile.write(decoded_line.strip() + "\n")
                await settings.redis_client.client.xadd(
                    f"forge:{image.image_id}:stream",
                    {"data": json.dumps({"log_type": name, "log": decoded_line}).decode()},
                )
            else:
                break

    try:
        storage_driver = os.getenv("STORAGE_DRIVER", "overlay")
        storage_opts = os.getenv("STORAGE_OPTS", "overlay.mount_program=/usr/bin/fuse-overlayfs")

        # Stage 1: Build the original image with a precise tag
        original_tag = f"{short_tag}-original-{uuid.uuid4().hex[:8]}"
        logger.info(f"Stage 1: Building original image as {original_tag}")

        build_cmd = [
            "buildah",
            "build",
            "--isolation",
            "chroot",
            "--storage-driver",
            storage_driver,
            "--layers",
            "--tag",
            original_tag,
            "-f",
            os.path.join(build_dir, "Dockerfile"),
        ]
        if storage_driver == "overlay" and storage_opts:
            for opt in storage_opts.split(","):
                build_cmd.extend(["--storage-opt", opt.strip()])
        if settings.registry_insecure:
            build_cmd.extend(["--tls-verify=false"])
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

        if process.returncode != 0:
            raise BuildFailure("Build of original image failed!")

        # Inject chutes.
        chutes_tag = f"{original_tag}-chutes"
        chutes_dockerfile_content = f"""FROM {original_tag}
USER root
ENV LD_PRELOAD=""
RUN rm -f /etc/chutesfs.index
RUN usermod -aG root chutes || true
RUN chmod g+rwx /usr/local/lib /usr/local/bin /usr/local/share /usr/local/share/man
RUN chmod g+rwx /usr/local/lib/python3.12/dist-packages || true
USER chutes
RUN pip install chutes=={image.chutes_version}
RUN cp -f $(python -c 'import chutes; import os; print(os.path.join(os.path.dirname(chutes.__file__), "chutes-netnanny.so"))') /usr/local/lib/chutes-netnanny.so
RUN cp -f $(python -c 'import chutes; import os; print(os.path.join(os.path.dirname(chutes.__file__), "chutes-logintercept.so"))') /usr/local/lib/chutes-logintercept.so
RUN cp -f $(python -c 'import chutes; import os; print(os.path.join(os.path.dirname(chutes.__file__), "chutes-cfsv.so"))') /usr/local/lib/chutes-cfsv.so
ENV LD_PRELOAD=/usr/local/lib/chutes-netnanny.so:/usr/local/lib/chutes-logintercept.so
WORKDIR /app
"""
        chutes_dockerfile_path = os.path.join(build_dir, "Dockerfile.chutes")
        with open(chutes_dockerfile_path, "w") as f:
            f.write(chutes_dockerfile_content)
        build_cmd = [
            "buildah",
            "build",
            "--isolation",
            "chroot",
            "--layers=false",
            "--storage-opt",
            "overlay.mountopt=metacopy=on",
            "--tag",
            chutes_tag,
            "-f",
            chutes_dockerfile_path,
        ]
        if settings.registry_insecure:
            build_cmd.extend(["--tls-verify=false"])
        build_cmd.append(build_dir)

        process = await asyncio.create_subprocess_exec(
            *build_cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            await asyncio.wait_for(
                asyncio.gather(
                    _capture_logs(process.stdout, "stdout"),
                    _capture_logs(process.stderr, "stderr"),
                    process.wait(),
                ),
                timeout=settings.build_timeout,
            )
        except Exception as exc:
            raise BuildFailure(f"Failed waiting for chutes image image: {str(exc)}")
        if process.returncode != 0:
            raise BuildFailure("Failed to install chutes library into image!")

        # Build filesystem verification image.
        verification_tag = f"{short_tag}-fsv-{uuid.uuid4().hex[:8]}"
        logger.info(f"Building filesystem verification image as {verification_tag}")
        fsv_dockerfile_content = f"""FROM {chutes_tag}
ARG CFSV_OP
ARG PS_OP
ENV LD_PRELOAD=""
ENV PYTHONDONTWRITEBYTECODE=1
RUN rm -rf does_not_exist.py does_not_exist
RUN PS_OP="${{PS_OP}}" chutes run does_not_exist:chute --generate-inspecto-hash > /tmp/inspecto.hash
COPY cfsv /cfsv
RUN CFSV_OP="${{CFSV_OP}}" /cfsv index / /tmp/chutesfs.index
USER root
RUN cp -f /tmp/chutesfs.index /etc/chutesfs.index && chmod a+r /etc/chutesfs.index
USER chutes
RUN CFSV_OP="${{CFSV_OP}}" /cfsv collect / /etc/chutesfs.index /tmp/chutesfs.data
"""
        if semcomp(image.chutes_version, "0.5.3") >= 0 and image.name in ("sglang", "vllm"):
            from api.user.service import chutes_user_id

            if image.user_id == await chutes_user_id():
                fsv_dockerfile_content += """
RUN python -m cllmv.pkg_hash > /tmp/package_hashes.json
"""
        fsv_dockerfile_path = os.path.join(build_dir, "Dockerfile.fsv")
        with open(fsv_dockerfile_path, "w") as f:
            f.write(fsv_dockerfile_content)

        build_cmd = [
            "buildah",
            "build",
            "--isolation",
            "chroot",
            "--build-arg",
            f"CFSV_OP={os.getenv('CFSV_OP', str(uuid.uuid4()))}",
            "--build-arg",
            f"PS_OP={os.getenv('PS_OP', str(uuid.uuid4()))}",
            "--layers=false",
            "--storage-opt",
            "overlay.mountopt=metacopy=on",
            "--tag",
            verification_tag,
            "-f",
            fsv_dockerfile_path,
        ]
        if settings.registry_insecure:
            build_cmd.extend(["--tls-verify=false"])
        build_cmd.append(build_dir)

        process = await asyncio.create_subprocess_exec(
            *build_cmd,
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.DEVNULL,
        )
        try:
            await asyncio.wait_for(process.wait(), timeout=settings.build_timeout)
        except Exception as exc:
            raise BuildFailure(f"Failed waiting for filesystem verification image: {str(exc)}")
        if process.returncode != 0:
            raise BuildFailure("Build of filesystem verification image failed!")

        # Extract the data file from the verification image
        (
            data_file_path,
            package_hashes,
            inspecto_hash,
        ) = await extract_cfsv_data_from_verification_image(verification_tag, build_dir)
        image.inspecto = inspecto_hash
        image.package_hashes = package_hashes
        await upload_filesystem_verification_data(image, data_file_path)

        # Build final image that combines original + index file
        logger.info(f"Building final image as {short_tag}")

        final_dockerfile_content = f"""FROM {verification_tag} as fsv
FROM {chutes_tag}
COPY --from=fsv /etc/chutesfs.index /etc/chutesfs.index
ENV PYTHONDONTWRITEBYTECODE=1
ENTRYPOINT []
"""
        final_dockerfile_path = os.path.join(build_dir, "Dockerfile.final")
        with open(final_dockerfile_path, "w") as f:
            f.write(final_dockerfile_content)

        build_cmd = [
            "buildah",
            "build",
            "--isolation",
            "chroot",
            "--layers=false",
            "--storage-opt",
            "overlay.mountopt=metacopy=on",
            "--tag",
            full_image_tag,
            "--tag",
            short_tag,
            "-f",
            final_dockerfile_path,
        ]
        if settings.registry_insecure:
            build_cmd.extend(["--tls-verify=false"])
        build_cmd.append(build_dir)

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
            await settings.redis_client.client.xadd(
                f"forge:{image.image_id}:stream",
                {"data": json.dumps({"log_type": "stdout", "log": message}).decode()},
            )
        else:
            raise BuildFailure(f"Final build of {full_image_tag} failed!")

    except asyncio.TimeoutError:
        message = f"Build of {full_image_tag} timed out after {settings.build_timeout} seconds."
        logger.error(message)
        await settings.redis_client.client.xadd(
            f"forge:{image.image_id}:stream",
            {"data": json.dumps({"log_type": "stderr", "log": message}).decode()},
        )
        await settings.redis_client.client.xadd(f"forge:{image.image_id}:stream", {"data": "DONE"})
        process.kill()
        await process.communicate()

        raise BuildTimeout(message)

    # Scan with trivy
    await trivy_image_scan(image, short_tag, _capture_logs)

    # Push
    await settings.redis_client.client.xadd(
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
            "--compression-format",
            "gzip",
            "--compression-level",
            "1",
            "--format",
            "v2s2",
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
            logger.success(f"Successfully pushed {full_image_tag}.")
            delta = time.time() - started_at
            message = (
                "\N{HAMMER AND WRENCH} "
                + f" finished pushing image {image.image_id} in {round(delta, 5)} seconds"
            )
            await settings.redis_client.client.xadd(
                f"forge:{image.image_id}:stream",
                {"data": json.dumps({"log_type": "stdout", "log": message}).decode()},
            )
            logger.success(message)
        else:
            message = "Image push failed, check logs for more details!"
            logger.error(message)
            await settings.redis_client.client.xadd(
                f"forge:{image.image_id}:stream",
                {"data": json.dumps({"log_type": "stderr", "log": message}).decode()},
            )
            await settings.redis_client.client.xadd(
                f"forge:{image.image_id}:stream", {"data": "DONE"}
            )
            raise PushFailure(f"Push of {full_image_tag} failed!")
    except asyncio.TimeoutError:
        message = f"Push of {full_image_tag} timed out after {settings.push_timeout} seconds."
        logger.error(message)
        await settings.redis_client.client.xadd(
            f"forge:{image.image_id}:stream",
            {"data": json.dumps({"log_type": "stderr", "log": message}).decode()},
        )
        await settings.redis_client.client.xadd(f"forge:{image.image_id}:stream", {"data": "DONE"})
        process.kill()
        await process.communicate()
        raise PushTimeout(
            f"Push of {full_image_tag} timed out after {settings.push_timeout} seconds."
        )

    # SIGN
    await sign_image(image, full_image_tag, _capture_logs)

    # DONE!
    delta = time.time() - started_at
    message = (
        "\N{HAMMER AND WRENCH} "
        + f" completed forging image {image.image_id} in {round(delta, 5)} seconds"
    )
    await settings.redis_client.client.xadd(
        f"forge:{image.image_id}:stream",
        {"data": json.dumps({"log_type": "stdout", "log": message}).decode()},
    )
    logger.success(message)
    await settings.redis_client.client.xadd(f"forge:{image.image_id}:stream", {"data": "DONE"})
    return short_tag


async def trivy_image_scan(image, short_tag, _capture_logs: Callable[[Any, Any, bool], None]):
    await settings.redis_client.client.xadd(
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
            await settings.redis_client.client.xadd(
                f"forge:{image.image_id}:stream",
                {"data": json.dumps({"log_type": "stdout", "log": message}).decode()},
            )
            logger.success(message)
        else:
            message = f"Issues scanning {short_tag} with trivy!"
            await settings.redis_client.client.xadd(
                f"forge:{image.image_id}:stream",
                {"data": json.dumps({"log_type": "stderr", "log": message}).decode()},
            )
            logger.error(message)
            raise BuildFailure(f"Failed trivy image scan: {short_tag}")
    except asyncio.TimeoutError:
        message = f"Trivy scan of {short_tag} timed out after."
        logger.error(message)
        await settings.redis_client.client.xadd(
            f"forge:{image.image_id}:stream",
            {"data": json.dumps({"log_type": "stderr", "log": message}).decode()},
        )
        await settings.redis_client.client.xadd(f"forge:{image.image_id}:stream", {"data": "DONE"})
        process.kill()
        await process.communicate()
        raise BuildTimeout(message)


async def get_image_digest(image_tag: str) -> str:
    """Get digest using cosign triangulate"""
    process = await asyncio.create_subprocess_exec(
        "cosign",
        "triangulate",
        "--allow-http-registry",
        image_tag,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout, stderr = await process.communicate()

    if process.returncode != 0:
        raise SignFailure(f"Failed to get digest for {image_tag}: {stderr.decode()}")

    # cosign triangulate returns the signature reference like:
    # localhost:5000/test-sign:sha256-f043a1e0f30aba1263f163794779c6916f13c18871217c0910525818d752c636.sig
    triangulate_output = stdout.decode().strip()

    # Extract the digest from the signature reference
    # Format: registry/repo:sha256-<digest>.sig
    if ":sha256-" in triangulate_output and triangulate_output.endswith(".sig"):
        # Extract the part between 'sha256-' and '.sig'
        digest_part = triangulate_output.split(":sha256-")[1].replace(".sig", "")
        digest = f"sha256:{digest_part}"
    else:
        raise SignFailure(f"Unexpected triangulate output format: {triangulate_output}")

    return digest


async def sign_image(
    image,
    image_tag: str,
    _capture_logs: Callable[[Any, Any, bool], None],
    stream: bool = True,
):
    """Sign the image using cosign"""
    started_at = time.time()
    try:
        image_digest = await get_image_digest(image_tag)
        image_digest_tag = f"{image_tag.rsplit(':', 1)[0]}@{image_digest}"

        external_registry = f"{settings.validator_ss58}.localregistry.chutes.ai"
        internal_registry = "registry"

        # Rebuild the full reference
        image_digest_tag = image_digest_tag.replace(
            f"{internal_registry}:5000", f"{external_registry}:5000"
        )

        process = await asyncio.create_subprocess_exec(
            "cosign",
            "sign",
            "--allow-http-registry",
            "--key",
            f"{settings.cosign_key}",
            image_digest_tag,
            "--yes",
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await process.communicate(input=f"{settings.cosign_password}\n".encode())

        await asyncio.wait_for(
            asyncio.gather(
                _capture_logs(process.stdout, "stdout"),
                _capture_logs(process.stderr, "stderr"),
                process.wait(),
            ),
            timeout=settings.build_timeout,
        )
        if process.returncode == 0:
            logger.success(f"Successfully signed {image_digest_tag}, done!")
            if stream:
                delta = time.time() - started_at
                message = (
                    "\N{HAMMER AND WRENCH} "
                    + f" finished signing image {image.image_id} in {round(delta, 1)} seconds"
                )
                await settings.redis_client.client.xadd(
                    f"forge:{image.image_id}:stream",
                    {"data": json.dumps({"log_type": "stdout", "log": message}).decode()},
                )
                logger.success(message)
        else:
            message = "Image sign failed, check logs for more details!"
            logger.error(message)
            if stream:
                await settings.redis_client.client.xadd(
                    f"forge:{image.image_id}:stream",
                    {"data": json.dumps({"log_type": "stderr", "log": message}).decode()},
                )
                await settings.redis_client.client.xadd(
                    f"forge:{image.image_id}:stream", {"data": "DONE"}
                )
            raise SignFailure(f"Sign of {image_tag} failed!")
    except asyncio.TimeoutError:
        message = f"Sign of {image_digest_tag} timed out after {settings.push_timeout} seconds."
        logger.error(message)
        if stream:
            await settings.redis_client.client.xadd(
                f"forge:{image.image_id}:stream",
                {"data": json.dumps({"log_type": "stderr", "log": message}).decode()},
            )
            await settings.redis_client.client.xadd(
                f"forge:{image.image_id}:stream", {"data": "DONE"}
            )
        process.kill()
        await process.communicate()
        raise SignTimeout(f"Sign of {image_tag} timed out after {settings.push_timeout} seconds.")


async def extract_cfsv_data_from_verification_image(verification_tag: str, build_dir: str):
    """
    Extract the data file from the filesystem verification image.
    Uses mount to directly access the filesystem.
    """
    container_id = None
    mount_path = None
    inspecto_hash = None
    package_hashes = None
    try:
        # Create container from the verification image
        process = await asyncio.create_subprocess_exec(
            "buildah",
            "from",
            verification_tag,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await process.communicate()
        if process.returncode != 0:
            raise Exception(f"Failed to create container: {stderr.decode()}")

        container_id = stdout.decode().strip()
        logger.info(f"Created container {container_id} from {verification_tag}")

        # Mount the container filesystem to access files directly
        process = await asyncio.create_subprocess_exec(
            "buildah",
            "mount",
            container_id,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await process.communicate()
        if process.returncode != 0:
            raise Exception(f"Failed to mount container: {stderr.decode()}")

        mount_path = stdout.decode().strip()
        logger.info(f"Mounted container at {mount_path}")

        # Copy the file directly from the mounted filesystem
        source_path = os.path.join(mount_path, "tmp", "chutesfs.data")
        data_file_path = os.path.join(build_dir, "chutesfs.data")

        # Check if source file exists
        if not os.path.exists(source_path):
            tmp_path = os.path.join(mount_path, "tmp")
            if os.path.exists(tmp_path):
                files = os.listdir(tmp_path)
                logger.warning(f"Files in /tmp: {files}")
            raise Exception(f"Data file not found at {source_path}")

        # Load inspecto hash.
        with open(os.path.join(mount_path, "tmp", "inspecto.hash")) as infile:
            inspecto_hash = infile.readlines()[-1].strip()
            assert inspecto_hash

        # Package hashes.
        hashes_json_path = os.path.join(mount_path, "tmp", "package_hashes.json")
        if os.path.exists(hashes_json_path):
            with open(hashes_json_path, "r") as infile:
                package_hashes = json.loads(infile.read())

        # Use shutil to copy the file
        shutil.copy2(source_path, data_file_path)
        logger.info(f"Successfully copied data file from {source_path} to {data_file_path}")

        return data_file_path, package_hashes, inspecto_hash
    finally:
        # Unmount if we mounted
        if mount_path and container_id:
            process = await asyncio.create_subprocess_exec(
                "buildah",
                "umount",
                container_id,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            await process.communicate()
            logger.info(f"Unmounted container {container_id}")

        # Clean up the container
        if container_id:
            process = await asyncio.create_subprocess_exec(
                "buildah",
                "rm",
                container_id,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            await process.communicate()
            logger.info(f"Removed container {container_id}")


async def upload_filesystem_verification_data(image, data_file_path: str):
    """
    Upload the filesystem verification data to S3 with the correct path structure.
    """
    # Handle None patch_version by defaulting to "initial"
    patch_version = image.patch_version if image.patch_version is not None else "initial"
    s3_key = f"image_hash_blobs/{image.image_id}/{patch_version}.data"
    async with settings.s3_client() as s3:
        await s3.upload_file(data_file_path, settings.storage_bucket, s3_key)
    logger.success(f"Uploaded filesystem verification data to {s3_key}")


async def get_target_image_id() -> str | None:
    """
    Get the image_id to build, ensuring no other processes can lock this same image_id.
    """
    async with get_session() as session:
        subquery = (
            select(Image.image_id)
            .where(Image.status == "pending build")
            .order_by(Image.created_at.asc())
            .limit(1)
            .scalar_subquery()
        )
        stmt = (
            update(Image)
            .where(
                Image.image_id == subquery,
                Image.status == "pending build",
            )
            .values(
                status="building",
            )
            .returning(Image.image_id)
            .execution_options(synchronize_session=False)
        )
        result = await session.execute(stmt)
        image_id = result.scalar_one_or_none()
        await session.commit()
        return image_id


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
        image.status = "building"
        image.build_started_at = func.now()
        await session.commit()
        await session.refresh(image)

    logger.info(f"Picked up forge task for {image_id=}: {image.name=} {image.tag=}")

    # Download the build context
    short_tag = None
    error_message = None
    inspecto_hash = None
    package_hashes = None
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
            inspecto_hash = image.inspecto
            package_hashes = image.package_hashes
        except Exception as exc:
            logger.error(f"Error building {image_id=}: {exc}\n{traceback.format_exc()}")
            error_message = str(exc)
        finally:
            os.chdir(starting_dir)

        # Upload logs
        if os.path.exists(log_path := os.path.join(build_dir, "build.log")):
            destination = f"forge/{image.user_id}/{image.image_id}.log"
            async with settings.s3_client() as s3:
                await s3.upload_file(log_path, settings.storage_bucket, destination)

    # Update status
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
            image.inspecto = inspecto_hash
            image.package_hashes = package_hashes
            image.build_completed_at = func.now()
        else:
            image.status = f"error: {error_message}"
        await session.commit()
        await session.refresh(image)

    await settings.redis_client.client.publish(
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

    # Cleanup
    os.system("bash /usr/local/bin/buildah_cleanup.sh")


async def update_chutes_lib(image_id: str, chutes_version: str, force: bool = False):
    """
    Update the chutes library in an existing image without rebuilding from scratch.
    """
    patch_version = hashlib.sha256(f"{image_id}:{chutes_version}".encode()).hexdigest()[:12]
    async with get_session() as session:
        result = await session.execute(select(Image).where(Image.image_id == image_id).limit(1))
        image = result.scalar_one_or_none()
        if not image:
            logger.error(f"Image does not exist: {image_id=}")
            return
        if image.chutes_version == chutes_version:
            logger.info(f"Image {image_id} already has chutes version {chutes_version}")
            if not force:
                return
            patch_version = hashlib.sha256(f"{image_id}:{time.time()}".encode()).hexdigest()[:12]
        await session.refresh(image, ["user"])

    # Determine source and target tags
    base_tag = f"{image.user.username}/{image.name}:{image.tag}".lower()
    if image.patch_version and image.patch_version != "initial":
        source_tag = f"{base_tag}-{image.patch_version}"
    else:
        source_tag = base_tag
    target_tag = f"{base_tag}-{patch_version}"
    full_source_tag = f"{settings.registry_host.rstrip('/')}/{source_tag}"
    full_target_tag = f"{settings.registry_host.rstrip('/')}/{target_tag}"

    # Helper to capture and stream logs
    async def _capture_logs(stream, name, capture=True):
        """Helper to capture logs. Set capture=False to suppress logs."""
        if not capture:
            # Just consume the stream without logging
            while True:
                line = await stream.readline()
                if not line:
                    break
            return

        log_method = logger.info if name == "stdout" else logger.warning
        while True:
            line = await stream.readline()
            if line:
                decoded_line = line.decode().strip()
                log_method(f"[update {target_tag}]: {decoded_line}")
            else:
                break

    # Rebuild the image with the updated chutes lib
    error_message = None
    success = False
    with tempfile.TemporaryDirectory() as build_dir:
        try:
            build_cfsv_path = os.path.join(build_dir, "cfsv")
            if semcomp(chutes_version or "0.0.0", "0.5.2") >= 0:
                shutil.copy2(CFSV_V3_PATH, build_cfsv_path)
            elif semcomp(chutes_version or "0.0.0", "0.4.6") >= 0:
                shutil.copy2(CFSV_V2_PATH, build_cfsv_path)
            else:
                shutil.copy2(CFSV_PATH, build_cfsv_path)
            os.chmod(build_cfsv_path, 0o755)

            storage_driver = os.getenv("STORAGE_DRIVER", "overlay")
            storage_opts = os.getenv(
                "STORAGE_OPTS", "overlay.mount_program=/usr/bin/fuse-overlayfs"
            )

            # Stage 1: Build updated base image with a precise tag
            updated_tag = f"{target_tag}-updated-{uuid.uuid4().hex[:8]}"
            logger.info(f"Stage 1: Building updated image as {updated_tag}")

            dockerfile_content = f"""FROM {full_source_tag}
USER root
ENV LD_PRELOAD=""
RUN rm -f /etc/chutesfs.index
RUN usermod -aG root chutes || true
RUN chmod g+rwx /usr/local/lib /usr/local/bin /usr/local/share /usr/local/share/man
RUN chmod g+rwx /usr/local/lib/python3.12/dist-packages || true
USER chutes
RUN pip install chutes=={chutes_version}
RUN cp -f $(python -c 'import chutes; import os; print(os.path.join(os.path.dirname(chutes.__file__), "chutes-netnanny.so"))') /usr/local/lib/chutes-netnanny.so
RUN cp -f $(python -c 'import chutes; import os; print(os.path.join(os.path.dirname(chutes.__file__), "chutes-logintercept.so"))') /usr/local/lib/chutes-logintercept.so
RUN cp -f $(python -c 'import chutes; import os; print(os.path.join(os.path.dirname(chutes.__file__), "chutes-cfsv.so"))') /usr/local/lib/chutes-cfsv.so
ENV LD_PRELOAD=/usr/local/lib/chutes-netnanny.so:/usr/local/lib/chutes-logintercept.so
"""
            dockerfile_path = os.path.join(build_dir, "Dockerfile.update")
            with open(dockerfile_path, "w") as f:
                f.write(dockerfile_content)

            build_cmd = [
                "buildah",
                "build",
                "--isolation",
                "chroot",
                "--layers=false",
                "--storage-opt",
                "overlay.mountopt=metacopy=on",
                "--tag",
                updated_tag,
                "-f",
                dockerfile_path,
            ]
            if settings.registry_insecure:
                build_cmd.extend(["--tls-verify=false"])
            build_cmd.append(build_dir)

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
                raise BuildFailure("Failed to build updated image!")

            # Stage 2: Build filesystem verification image from the updated image
            verification_tag = f"{target_tag}-fsv-{uuid.uuid4().hex[:8]}"
            logger.info(f"Stage 2: Building filesystem verification image as {verification_tag}")

            fsv_dockerfile_content = f"""FROM {updated_tag}
ARG CFSV_OP
ARG PS_OP
ENV LD_PRELOAD=""
ENV PYTHONDONTWRITEBYTECODE=1
RUN rm -rf does_not_exist.py does_not_exist
RUN PS_OP="${{PS_OP}}" chutes run does_not_exist:chute --generate-inspecto-hash > /tmp/inspecto.hash
COPY cfsv /cfsv
RUN CFSV_OP="${{CFSV_OP}}" /cfsv index / /tmp/chutesfs.index
USER root
RUN cp -f /tmp/chutesfs.index /etc/chutesfs.index && chmod a+r /etc/chutesfs.index
USER chutes
RUN CFSV_OP="${{CFSV_OP}}" /cfsv collect / /etc/chutesfs.index /tmp/chutesfs.data
"""
            if semcomp(chutes_version, "0.5.3") >= 0 and image.name in ("sglang", "vllm"):
                from api.user.service import chutes_user_id

                if image.user_id == await chutes_user_id():
                    fsv_dockerfile_content += """
RUN python -m cllmv.pkg_hash > /tmp/package_hashes.json
"""
            fsv_dockerfile_path = os.path.join(build_dir, "Dockerfile.fsv")
            with open(fsv_dockerfile_path, "w") as f:
                f.write(fsv_dockerfile_content)

            build_cmd = [
                "buildah",
                "build",
                "--isolation",
                "chroot",
                "--build-arg",
                f"CFSV_OP={os.getenv('CFSV_OP', str(uuid.uuid4()))}",
                "--build-arg",
                f"PS_OP={os.getenv('PS_OP', str(uuid.uuid4()))}",
                "--layers=false",
                "--storage-opt",
                "overlay.mountopt=metacopy=on",
                "--tag",
                verification_tag,
                "-f",
                fsv_dockerfile_path,
            ]
            if settings.registry_insecure:
                build_cmd.extend(["--tls-verify=false"])
            build_cmd.append(build_dir)

            process = await asyncio.create_subprocess_exec(
                *build_cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )

            # Run without capturing logs to avoid noise
            await asyncio.wait_for(
                asyncio.gather(
                    _capture_logs(process.stdout, "stdout", capture=False),
                    _capture_logs(process.stderr, "stderr", capture=False),
                    process.wait(),
                ),
                timeout=settings.build_timeout,
            )

            if process.returncode != 0:
                raise BuildFailure("Failed to build filesystem verification image!")

            # Extract and upload data file
            (
                data_file_path,
                package_hashes,
                inspecto_hash,
            ) = await extract_cfsv_data_from_verification_image(verification_tag, build_dir)
            s3_key = f"image_hash_blobs/{image_id}/{patch_version}.data"
            async with settings.s3_client() as s3:
                await s3.upload_file(data_file_path, settings.storage_bucket, s3_key)
            logger.success(f"Uploaded filesystem verification data to {s3_key}")

            # Stage 3: Build final image that combines updated + index file
            logger.info(f"Stage 3: Building final image as {target_tag}")

            final_dockerfile_content = f"""FROM {verification_tag} as fsv
FROM {updated_tag} as base
COPY --from=fsv /tmp/chutesfs.index /etc/chutesfs.index
ENV PYTHONDONTWRITEBYTECODE=1
ENTRYPOINT []
"""
            final_dockerfile_path = os.path.join(build_dir, "Dockerfile.final")
            with open(final_dockerfile_path, "w") as f:
                f.write(final_dockerfile_content)

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
                "-f",
                final_dockerfile_path,
            ]
            if storage_driver == "overlay" and storage_opts:
                for opt in storage_opts.split(","):
                    build_cmd.extend(["--storage-opt", opt.strip()])
            if settings.registry_insecure:
                build_cmd.extend(["--tls-verify=false"])
            build_cmd.append(build_dir)

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
                raise BuildFailure("Failed to build final image!")

            # Push to registry
            logger.info("Pushing updated image to registry...")
            verify = str(not settings.registry_insecure).lower()
            process = await asyncio.create_subprocess_exec(
                "buildah",
                f"--tls-verify={verify}",
                "push",
                full_target_tag,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )

            await asyncio.wait_for(
                asyncio.gather(
                    _capture_logs(process.stdout, "stdout"),
                    _capture_logs(process.stderr, "stderr"),
                    process.wait(),
                ),
                timeout=settings.push_timeout,
            )

            if process.returncode != 0:
                raise PushFailure("Failed to push image!")

            logger.success(f"Successfully pushed updated image {full_target_tag}")
            success = True

        except asyncio.TimeoutError:
            message = f"Update of {full_target_tag} timed out"
            logger.error(message)
            process.kill()
            await process.communicate()
            raise BuildTimeout(message)
        except Exception as exc:
            logger.error(
                f"Error updating chutes lib for {image_id}: {exc}\n{traceback.format_exc()}"
            )
            error_message = str(exc)

        # SIGN
        if success:
            await sign_image(image, full_target_tag, _capture_logs, stream=True)

    # Update the image with the new patch version, tag, etc.
    affected_chute_ids = []
    if success:
        async with get_session() as session:
            image = (
                (await session.execute(select(Image).where(Image.image_id == image_id)))
                .unique()
                .scalar_one_or_none()
            )
            chutes = []
            if image:
                image.patch_version = patch_version
                image.chutes_version = chutes_version
                image.short_tag = target_tag
                image.inspecto = inspecto_hash
                image.package_hashes = package_hashes
                await session.commit()
                await session.refresh(image)
                logger.success(
                    f"Updated image {image_id} to chutes version {chutes_version}, patch version {patch_version}"
                )

                # Update the associated chutes with the new version
                chutes_result = await session.execute(
                    select(Chute)
                    .where(Chute.image_id == image_id)
                    .options(selectinload(Chute.instances))
                )
                chutes = chutes_result.scalars().all()
                permitted = {}
                for chute in chutes:
                    logger.warning(
                        f"Need to trigger rolling update for {chute.chute_id=} to use new image",
                    )
                    old_version = chute.version
                    for instance in chute.instances:
                        logger.warning(
                            f"Need to update {instance.instance_id=} {instance.miner_hotkey=} for {instance.chute_id=} to use new image"
                        )
                        if instance.miner_hotkey not in permitted:
                            permitted[instance.miner_hotkey] = 0
                        permitted[instance.miner_hotkey] += 1
                    chute.chutes_version = chutes_version
                    chute.version = str(
                        uuid.uuid5(
                            uuid.NAMESPACE_OID,
                            f"{image.image_id}:{image.patch_version}:{chute.code}",
                        )
                    )
                    affected_chute_ids.append(chute.chute_id)

                    # Create the rolling update record
                    await session.execute(
                        text(
                            "DELETE FROM rolling_updates WHERE chute_id = :chute_id",
                        ),
                        {"chute_id": chute.chute_id},
                    )
                    if permitted:
                        session.add(
                            RollingUpdate(
                                chute_id=chute.chute_id,
                                old_version=old_version,
                                new_version=chute.version,
                                permitted=permitted,
                            )
                        )

                # Commit the chute updates
                await session.commit()

            # Trigger the rolling update tasks.
            for chute in chutes:
                logger.warning(f"Triggering rolling update task: {chute.chute_id=}")
                await handle_rolling_update.kiq(
                    chute.chute_id, chute.version, reason="image updated due to chutes lib upgrade"
                )

            # Notify miners of the update
            image_path = f"{image.user.username}/{image.name}:{image.tag}-{patch_version}"
            await settings.redis_client.client.publish(
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


async def main():
    await initialize()

    while True:
        image_id = None
        try:
            image_id = await asyncio.wait_for(get_target_image_id(), 10)
        except Exception as exc:
            logger.error(f"Failed to fetch image: {str(exc)}\n{traceback.format_exc()}")
        if not image_id:
            await asyncio.sleep(10)
            continue
        await forge(image_id)


if __name__ == "__main__":
    asyncio.run(main())
