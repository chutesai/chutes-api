"""
Remote (depot) image forge.
"""

import asyncio
from asyncio.subprocess import PIPE
from typing import Callable
import re
import uuid
import os
import hashlib
import tempfile
import traceback
import time
import shutil
import orjson as json
from loguru import logger
from api.config import settings
from api.database import get_session
from api.exceptions import (
    SignFailure,
    SignTimeout,
    BuildFailure,
    BuildTimeout,
)
from api.util import semcomp
from api.image.schemas import Image
from api.chute.schemas import Chute, RollingUpdate
from api.graval_worker import handle_rolling_update
from api.image.forge import (
    safe_extract,
    get_target_image_id,
    upload_filesystem_verification_data,
    upload_bytecode_manifest,
    upload_bytecode_manifest_json,
    CFSV_PATH,
    CFSV_V2_PATH,
    CFSV_V3_PATH,
    CFSV_V4_PATH,
    BCM_SO_PATH,
    MANIFEST_DRIVER_PATH,
)
from sqlalchemy import func, text
from sqlalchemy.orm import selectinload
from sqlalchemy.future import select
from api.database import orms  # noqa

# Minimal environment for depot build subprocesses.
_DEPOT_ENV = {
    "PATH": "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/home/chutes/.depot/bin",
    "HOME": os.environ.get("HOME", "/home/chutes"),
    "DEPOT_TOKEN": settings.depot_token,
}


def _depot_registry_ref(repo: str, tag: str) -> str:
    """Return a fully qualified Depot registry image reference.

    Format: {org_registry}/{repo}:{tag}
    Example: foo.registry.depot.dev/alice/model:v1
    """
    return f"{settings.depot_registry}/{repo}:{tag}"


def _safe_ref(ref: str) -> str:
    """Strip the registry hostname from an image ref for logging.

    'foo.registry.depot.dev/alice/model:v1' -> 'alice/model:v1'
    """
    if settings.depot_registry and ref.startswith(settings.depot_registry + "/"):
        return ref[len(settings.depot_registry) + 1 :]
    return ref


async def _depot_build(
    args: list[str],
    cwd: str,
    capture_logs: Callable,
    timeout: int,
) -> asyncio.subprocess.Process:
    """Run `depot build` with standard project/env settings.

    When --save is in args, a --metadata-file is automatically added so the
    build ID can be retrieved afterwards via _get_build_id().
    """
    # Inject --metadata-file when saving so we can depot push afterwards.
    metadata_path = None
    if "--save" in args:
        metadata_path = os.path.join(cwd, f".depot-meta-{uuid.uuid4().hex[:8]}.json")
        args = args + ["--metadata-file", metadata_path]

    cmd = ["depot", "build", "--project", settings.depot_project_id] + args
    process = await asyncio.create_subprocess_exec(
        *cmd, stdout=PIPE, stderr=PIPE, env=_DEPOT_ENV, cwd=cwd
    )
    await asyncio.wait_for(
        asyncio.gather(
            capture_logs(process.stdout, "stdout"),
            capture_logs(process.stderr, "stderr"),
            process.wait(),
        ),
        timeout=timeout,
    )
    # Stash metadata path on the process object for _depot_push to find.
    process._depot_metadata_path = metadata_path
    return process


async def _depot_push(process: asyncio.subprocess.Process, tag: str, timeout: int = 900) -> None:
    """Push a --save'd depot build to the registry.

    Reads the build ID from the metadata file written by _depot_build,
    then runs `depot push` which pushes from Depot's cache to the registry.
    """
    metadata_path = getattr(process, "_depot_metadata_path", None)
    if not metadata_path or not os.path.exists(metadata_path):
        raise BuildFailure("No depot metadata file found — was --save used?")

    with open(metadata_path) as f:
        metadata = json.loads(f.read())

    # Metadata may be {"depot.build": {"buildID": "xxx", ...}} or flat {"buildID": "xxx"}.
    build_id = metadata.get("buildID", "")
    if not build_id:
        depot_build = metadata.get("depot.build", {})
        if isinstance(depot_build, dict):
            build_id = depot_build.get("buildID", "")
        elif isinstance(depot_build, str):
            build_id = depot_build
    if not build_id:
        raise BuildFailure(
            f"Could not find build ID in depot metadata. Keys: {list(metadata.keys())}"
        )

    logger.info(f"Pushing build {build_id} as {_safe_ref(tag)}")
    cmd = [
        "depot",
        "push",
        "--project",
        settings.depot_project_id,
        "--tag",
        tag,
        build_id,
    ]
    push_proc = await asyncio.create_subprocess_exec(*cmd, stdout=PIPE, stderr=PIPE, env=_DEPOT_ENV)
    stdout, stderr = await asyncio.wait_for(push_proc.communicate(), timeout=timeout)
    if push_proc.returncode != 0:
        raise BuildFailure(f"depot push failed for {_safe_ref(tag)}: {stderr.decode()}")
    logger.info(f"Pushed {_safe_ref(tag)}")


async def _is_real_image_tag(repo: str, oci_tag: str) -> bool:
    """Check the database to see if any image actually uses this tag.

    Returns True if an image exists with a matching user/name/tag/patch_version
    that would produce this repo:oci_tag combination in the registry.

    This prevents us from accidentally clobbering or deleting a real image
    when creating or cleaning up intermediate build tags.
    """
    # repo = "username/imagename", oci_tag could be "v1" or "v1-patchhash"
    parts = repo.split("/", 1)
    if len(parts) != 2:
        return False
    username, image_name = parts

    from api.user.schemas import User

    async with get_session() as session:
        result = await session.execute(
            select(Image)
            .join(User, Image.user_id == User.user_id)
            .where(
                User.username == username,
                Image.name == image_name,
            )
        )
        images = result.scalars().all()
        for image in images:
            # Reconstruct every possible OCI tag this image could produce.
            base = image.tag.lower()
            candidates = {base}
            if image.patch_version and image.patch_version != "initial":
                candidates.add(f"{base}-{image.patch_version}")
            if oci_tag in candidates:
                return True
    return False


async def _safe_intermediate_tag(repo: str, oci_tag: str, suffix: str) -> str:
    """Generate an intermediate OCI tag that is guaranteed not to collide
    with any real image tag in the database.

    Tries up to 5 times with different UUIDs before giving up.
    """
    for _ in range(5):
        candidate = f"{oci_tag}-{suffix}-{uuid.uuid4().hex[:8]}"
        if not await _is_real_image_tag(repo, candidate):
            return candidate
        logger.warning(f"Intermediate tag {candidate} collides with a real image, retrying")
    raise BuildFailure(
        f"Could not generate a safe intermediate tag for {repo}:{oci_tag} "
        f"after 5 attempts — this should never happen"
    )


async def _cleanup_intermediate_tags(tags: list[str], final_tag: str) -> None:
    """Delete intermediate image tags from the Depot registry.

    Safety checks:
      - Skips any tag matching the final image tag.
      - Queries the DB to verify the tag doesn't belong to a real image.
    """
    for tag in tags:
        if tag == final_tag:
            logger.warning(f"Skipping cleanup of {_safe_ref(tag)} — matches final image tag")
            continue
        # Parse repo:oci_tag from the full registry ref.
        # tag = "registry/repo:oci_tag"
        without_registry = tag.split("/", 1)[-1] if "/" in tag else tag
        if ":" in without_registry:
            repo_part, oci_tag_part = without_registry.rsplit(":", 1)
        else:
            logger.warning(f"Skipping cleanup of {_safe_ref(tag)} — could not parse tag")
            continue
        if await _is_real_image_tag(repo_part, oci_tag_part):
            logger.error(
                f"REFUSING to delete {_safe_ref(tag)} — it matches a real image in the database!"
            )
            continue
        try:
            process = await asyncio.create_subprocess_exec(
                "crane",
                "delete",
                tag,
                stdout=PIPE,
                stderr=PIPE,
            )
            stdout, stderr = await process.communicate()
            if process.returncode == 0:
                logger.info(f"Cleaned up intermediate tag {_safe_ref(tag)}")
            else:
                logger.warning(f"Failed to clean up {_safe_ref(tag)}: {stderr.decode().strip()}")
        except Exception as exc:
            logger.warning(f"Error cleaning up {_safe_ref(tag)}: {exc}")


async def initialize():
    """
    Ensure ORM modules are loaded, login to Docker Hub and Depot registry for cosign.
    """
    import api.database.orms  # noqa: F401

    # Login cosign + crane to Depot registry (needed for sign_image and get_image_digest).
    if settings.depot_registry and settings.depot_registry_token:
        # cosign login -u ... -p ... <registry>
        process = await asyncio.create_subprocess_exec(
            "cosign",
            "login",
            "-u",
            "x-token",
            "-p",
            settings.depot_registry_token,
            settings.depot_registry,
        )
        await process.wait()
        if process.returncode == 0:
            logger.success("cosign authenticated to depot registry")
        else:
            logger.warning("cosign failed to authenticate to depot registry")

        # crane auth login -u ... -p ... <registry>
        process = await asyncio.create_subprocess_exec(
            "crane",
            "auth",
            "login",
            "-u",
            "x-token",
            "-p",
            settings.depot_registry_token,
            settings.depot_registry,
        )
        await process.wait()
        if process.returncode == 0:
            logger.success("crane authenticated to depot registry")
        else:
            logger.warning("crane failed to authenticate to depot registry")


async def get_image_digest(image_tag: str) -> str:
    """Get the image digest from the registry using crane."""
    process = await asyncio.create_subprocess_exec(
        "crane",
        "digest",
        image_tag,
        stdout=PIPE,
        stderr=PIPE,
    )
    stdout, stderr = await process.communicate()

    if process.returncode != 0:
        raise SignFailure(f"Failed to get digest for {_safe_ref(image_tag)}: {stderr.decode()}")

    digest = stdout.decode().strip()
    if not digest.startswith("sha256:"):
        raise SignFailure(f"Unexpected digest format: {digest}")

    return digest


async def sign_image(
    image,
    image_tag: str,
    stream: bool = True,
):
    """Sign the image using cosign against Depot's registry.

    Only streams brief status messages to redis -- no raw cosign output.
    """
    started_at = time.time()
    if stream:
        await _stream_status(image.image_id, "signing image...")
    try:
        image_digest = await get_image_digest(image_tag)
        image_digest_tag = f"{image_tag.rsplit(':', 1)[0]}@{image_digest}"

        process = await asyncio.create_subprocess_exec(
            "cosign",
            "sign",
            "--key",
            f"{settings.cosign_key}",
            image_digest_tag,
            "--yes",
            stdin=PIPE,
            stdout=PIPE,
            stderr=PIPE,
        )
        stdout, stderr = await process.communicate(input=f"{settings.cosign_password}\n".encode())

        if process.returncode == 0:
            logger.success(f"Successfully signed {_safe_ref(image_digest_tag)}")
            if stream:
                delta = time.time() - started_at
                await _stream_status(
                    image.image_id,
                    f"image signed in {round(delta, 1)} seconds",
                )
        else:
            # Log full stderr server-side only; send sanitised message to redis.
            logger.error(f"Image sign failed: {stderr.decode()}")
            if stream:
                await _stream_status(image.image_id, "image signing failed", log_type="stderr")
                await settings.redis_client.client.xadd(
                    f"forge:{image.image_id}:stream", {"data": "DONE"}
                )
            raise SignFailure(f"Sign of {_safe_ref(image_tag)} failed!")
    except SignFailure:
        raise
    except asyncio.TimeoutError:
        logger.error(
            f"Sign of {_safe_ref(image_tag)} timed out after {settings.push_timeout} seconds."
        )
        if stream:
            await _stream_status(image.image_id, "image signing timed out", log_type="stderr")
            await settings.redis_client.client.xadd(
                f"forge:{image.image_id}:stream", {"data": "DONE"}
            )
        raise SignTimeout(
            f"Sign of {_safe_ref(image_tag)} timed out after {settings.push_timeout} seconds."
        )


async def _stream_status(image_id: str, message: str, log_type: str = "stdout"):
    """Send a brief status message to the forge redis stream (no raw build output)."""
    await settings.redis_client.client.xadd(
        f"forge:{image_id}:stream",
        {"data": json.dumps({"log_type": log_type, "log": message}).decode()},
    )


async def _drain_logs(stream, name, label: str = ""):
    """Read subprocess output, log server-side only. Nothing goes to redis."""
    log_method = logger.info if name == "stdout" else logger.warning
    while True:
        line = await stream.readline()
        if not line:
            break
        if label:
            log_method(f"[{label}]: {line.decode().strip()}")
        else:
            log_method(line.decode().strip())


_FROM_RE = re.compile(r"^\s*FROM\s+(?:--platform=\S+\s+)?(\S+)", re.IGNORECASE | re.MULTILINE)


def _validate_dockerfile_bases(dockerfile_path: str) -> None:
    """Reject Dockerfiles that pull from anything other than parachutes/* images.

    Allows:
      - parachutes/<image>:<tag>
      - scratch (Docker built-in)
      - References to earlier build stages (e.g. FROM builder)
    """
    with open(dockerfile_path) as f:
        content = f.read()

    stage_names: set[str] = set()
    for match in _FROM_RE.finditer(content):
        image_ref = match.group(1)

        # Collect AS aliases so later FROM <alias> is allowed.
        as_match = re.search(r"\bAS\s+(\S+)", match.group(0), re.IGNORECASE)
        if as_match:
            stage_names.add(as_match.group(1).lower())

        ref_lower = image_ref.lower()

        # Allow scratch and references to earlier stages.
        if ref_lower == "scratch" or ref_lower in stage_names:
            continue

        # Strip tag/digest for the prefix check.
        image_name = ref_lower.split(":")[0].split("@")[0]
        if not image_name.startswith("parachutes/"):
            raise BuildFailure(
                f"Dockerfile FROM targets must use parachutes/* base images, got: {image_ref}"
            )


def _copy_cfsv_binary(image, build_dir: str) -> str:
    """Copy the appropriate cfsv binary version into the build directory."""
    build_cfsv_path = os.path.join(build_dir, "cfsv")
    version = image.chutes_version or "0.0.0"
    if semcomp(version, "0.5.5") >= 0:
        shutil.copy2(CFSV_V4_PATH, build_cfsv_path)
    elif semcomp(version, "0.5.2") >= 0:
        shutil.copy2(CFSV_V3_PATH, build_cfsv_path)
    elif semcomp(version, "0.4.6") >= 0:
        shutil.copy2(CFSV_V2_PATH, build_cfsv_path)
    else:
        shutil.copy2(CFSV_PATH, build_cfsv_path)
    os.chmod(build_cfsv_path, 0o755)
    return build_cfsv_path


async def build_and_push_image(image, build_dir):
    """
    Build and push an image via Depot remote builders.

    Stages:
      1. Build user's Dockerfile and push to Depot registry
      2. Install chutes lib on top and push
      3. Filesystem verification + extract files via --output
      4. Build final image (chutes + index) and push
      5. Trivy scan via remote multi-stage build
      6. Cosign sign
    """
    # Depot OCI refs use: registry/project/repo:tag
    # repo = "username/imagename", tag = OCI-compliant (no / or :)
    repo = f"{image.user.username}/{image.name}".lower()
    oci_tag = image.tag.lower()
    if image.patch_version and image.patch_version != "initial":
        oci_tag = f"{oci_tag}-{image.patch_version}"
    short_tag = f"{repo}:{oci_tag}"

    _validate_dockerfile_bases(os.path.join(build_dir, "Dockerfile"))
    _copy_cfsv_binary(image, build_dir)

    intermediate_tags: list[str] = []
    started_at = time.time()

    async def _capture_logs(stream, name, capture=True):
        if not capture:
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
        # Stage 1: Build user's Dockerfile and push to Depot registry.
        original_oci_tag = await _safe_intermediate_tag(repo, oci_tag, "original")
        original_depot_ref = _depot_registry_ref(repo, original_oci_tag)
        logger.info(f"Stage 1: Building original image as {_safe_ref(original_depot_ref)}")

        process = await _depot_build(
            [
                "--save",
                "--tag",
                original_depot_ref,
                "-f",
                os.path.join(build_dir, "Dockerfile"),
                ".",
            ],
            cwd=build_dir,
            capture_logs=_capture_logs,
            timeout=settings.build_timeout,
        )
        if process.returncode != 0:
            raise BuildFailure("Build of original image failed!")
        await _depot_push(process, original_depot_ref)
        intermediate_tags.append(original_depot_ref)

        # Stage 2: Install chutes lib (status only, no raw log streaming).
        chutes_oci_tag = await _safe_intermediate_tag(repo, oci_tag, "chutes")
        chutes_depot_tag = _depot_registry_ref(repo, chutes_oci_tag)
        logger.info(f"Stage 2: Installing chutes lib as {_safe_ref(chutes_depot_tag)}")
        await _stream_status(image.image_id, "installing chutes library...")

        chutes_dockerfile_content = f"""FROM {original_depot_ref}
USER root
ENV LD_PRELOAD=""
RUN rm -f /etc/chutesfs.index
RUN usermod -aG root chutes || true
RUN chmod g+rwx /usr/local/lib /usr/local/bin /usr/local/share /usr/local/share/man
RUN chmod g+rwx /usr/local/lib/python3.12/dist-packages || true
RUN find / -xdev -type f -name '*.pyc' -exec rm -f {{}} \\; || true
RUN find / -xdev -type d -name __pycache__ -exec rm -rf {{}} \\; || true
USER chutes
ENV PYTHONDONTWRITEBYTECODE=1
RUN pip install chutes=={image.chutes_version}
RUN uv cache clean --force
"""
        if semcomp(image.chutes_version or "0.0.0", "0.5.5") >= 0:
            chutes_dockerfile_content += """RUN cp -f $(python -c 'import chutes; import os; print(os.path.join(os.path.dirname(chutes.__file__), "chutes-aegis.so"))') /usr/local/lib/chutes-aegis.so
ENV LD_PRELOAD=/usr/local/lib/chutes-aegis.so
"""
        else:
            chutes_dockerfile_content += """RUN cp -f $(python -c 'import chutes; import os; print(os.path.join(os.path.dirname(chutes.__file__), "chutes-netnanny.so"))') /usr/local/lib/chutes-netnanny.so
RUN cp -f $(python -c 'import chutes; import os; print(os.path.join(os.path.dirname(chutes.__file__), "chutes-logintercept.so"))') /usr/local/lib/chutes-logintercept.so
RUN cp -f $(python -c 'import chutes; import os; print(os.path.join(os.path.dirname(chutes.__file__), "chutes-cfsv.so"))') /usr/local/lib/chutes-cfsv.so
ENV LD_PRELOAD=/usr/local/lib/chutes-netnanny.so:/usr/local/lib/chutes-logintercept.so
"""
        chutes_dockerfile_content += "WORKDIR /app\n"
        chutes_dockerfile_path = os.path.join(build_dir, "Dockerfile.chutes")
        with open(chutes_dockerfile_path, "w") as f:
            f.write(chutes_dockerfile_content)

        process = await _depot_build(
            [
                "--save",
                "--tag",
                chutes_depot_tag,
                "-f",
                chutes_dockerfile_path,
                ".",
            ],
            cwd=build_dir,
            capture_logs=lambda s, n: _drain_logs(s, n, label=f"chutes {short_tag}"),
            timeout=settings.build_timeout,
        )
        if process.returncode != 0:
            raise BuildFailure("Failed to install chutes library into image!")
        await _depot_push(process, chutes_depot_tag)
        intermediate_tags.append(chutes_depot_tag)
        await _stream_status(image.image_id, "chutes library installed")

        # Stage 3: Filesystem verification + extract (status only).
        logger.info("Stage 3: Building filesystem verification image")
        await _stream_status(image.image_id, "generating filesystem verification data...")

        fsv_dockerfile_content = f"""FROM {chutes_depot_tag}
USER chutes
ARG CFSV_OP
ARG PS_OP
ENV LD_PRELOAD=""
ENV PYTHONDONTWRITEBYTECODE=1
RUN rm -rf does_not_exist.py does_not_exist
RUN PS_OP="${{PS_OP}}" chutes run does_not_exist:chute --generate-inspecto-hash > /tmp/inspecto.hash
USER root
RUN rm -f /etc/ld.so.preload /etc/bytecode.manifest /tmp/chutesfs.index /etc/chutesfs.index /tmp/chutesfs.data
USER chutes
COPY cfsv /cfsv
RUN --network=none CFSV_OP="${{CFSV_OP}}" /cfsv index / /tmp/chutesfs.index
USER root
RUN cp -f /tmp/chutesfs.index /etc/chutesfs.index && chmod a+r /etc/chutesfs.index
USER chutes
RUN --network=none CFSV_OP="${{CFSV_OP}}" /cfsv collect / /etc/chutesfs.index /tmp/chutesfs.data
"""

        # Generate bytecode manifest (V2) for chutes >= 0.5.5.
        has_bcm = False
        build_bcm_path = os.path.join(build_dir, "chutes-bcm.so")
        build_driver_path = os.path.join(build_dir, "generate_manifest_driver.py")
        if (
            semcomp(image.chutes_version or "0.0.0", "0.5.5") >= 0
            and os.path.exists(BCM_SO_PATH)
            and os.path.exists(MANIFEST_DRIVER_PATH)
        ):
            shutil.copy2(BCM_SO_PATH, build_bcm_path)
            shutil.copy2(MANIFEST_DRIVER_PATH, build_driver_path)
            has_bcm = True
            fsv_dockerfile_content += """COPY chutes-bcm.so /tmp/chutes-bcm.so
COPY generate_manifest_driver.py /tmp/generate_manifest_driver.py
RUN CFSV_OP="${CFSV_OP}" python /tmp/generate_manifest_driver.py \
    --output /tmp/bytecode.manifest \
    --json-output /tmp/bytecode.manifest.json \
    --lib /tmp/chutes-bcm.so \
    --extra-dirs /usr/local/lib/python3.12/site-packages
"""

        has_package_hashes = False
        if semcomp(image.chutes_version, "0.5.3") >= 0 and image.name in ("sglang", "vllm"):
            from api.user.service import chutes_user_id

            if image.user_id == await chutes_user_id():
                has_package_hashes = True
                fsv_dockerfile_content += """
USER root
RUN cp -f /tmp/bytecode.manifest /etc/bytecode.manifest || true
USER chutes
RUN CFSV_OP="${CFSV_OP}" python -m cllmv.pkg_hash > /tmp/package_hashes.json
"""

        # Append extract stage for --output type=local.
        fsv_dockerfile_content += "\nFROM scratch AS extract\n"
        fsv_dockerfile_content += "COPY --from=0 /tmp/chutesfs.data /chutesfs.data\n"
        fsv_dockerfile_content += "COPY --from=0 /tmp/inspecto.hash /inspecto.hash\n"
        fsv_dockerfile_content += "COPY --from=0 /tmp/chutesfs.index /chutesfs.index\n"
        if has_bcm:
            fsv_dockerfile_content += "COPY --from=0 /tmp/bytecode.manifest /bytecode.manifest\n"
            fsv_dockerfile_content += (
                "COPY --from=0 /tmp/bytecode.manifest.json /bytecode.manifest.json\n"
            )
        if has_package_hashes:
            fsv_dockerfile_content += (
                "COPY --from=0 /tmp/package_hashes.json /package_hashes.json\n"
            )

        fsv_dockerfile_path = os.path.join(build_dir, "Dockerfile.fsv")
        with open(fsv_dockerfile_path, "w") as f:
            f.write(fsv_dockerfile_content)

        extracted_dir = os.path.join(build_dir, "extracted")
        os.makedirs(extracted_dir, exist_ok=True)

        process = await _depot_build(
            [
                "--build-arg",
                f"CFSV_OP={os.getenv('CFSV_OP', str(uuid.uuid4()))}",
                "--build-arg",
                f"PS_OP={os.getenv('PS_OP', str(uuid.uuid4()))}",
                "--output",
                f"type=local,dest={extracted_dir}",
                "--target",
                "extract",
                "-f",
                fsv_dockerfile_path,
                ".",
            ],
            cwd=build_dir,
            capture_logs=lambda s, n: _drain_logs(s, n, label=f"fsv {short_tag}"),
            timeout=settings.build_timeout,
        )
        if process.returncode != 0:
            raise BuildFailure("Build of filesystem verification image failed!")
        await _stream_status(image.image_id, "filesystem verification complete")

        # Read extracted files.
        data_file_path = os.path.join(extracted_dir, "chutesfs.data")
        if not os.path.exists(data_file_path):
            files = os.listdir(extracted_dir) if os.path.exists(extracted_dir) else []
            raise BuildFailure(f"chutesfs.data not found in extracted output. Files: {files}")

        with open(os.path.join(extracted_dir, "inspecto.hash")) as infile:
            inspecto_hash = infile.readlines()[-1].strip()
            assert inspecto_hash

        package_hashes = None
        hashes_json_path = os.path.join(extracted_dir, "package_hashes.json")
        if os.path.exists(hashes_json_path):
            with open(hashes_json_path, "r") as infile:
                package_hashes = json.loads(infile.read())

        image.inspecto = inspecto_hash
        image.package_hashes = package_hashes
        await upload_filesystem_verification_data(image, data_file_path)

        bytecode_manifest_path = os.path.join(extracted_dir, "bytecode.manifest")
        if not os.path.exists(bytecode_manifest_path):
            bytecode_manifest_path = None
        bytecode_manifest_json_path = os.path.join(extracted_dir, "bytecode.manifest.json")
        if not os.path.exists(bytecode_manifest_json_path):
            bytecode_manifest_json_path = None

        if bytecode_manifest_path:
            await upload_bytecode_manifest(image, bytecode_manifest_path)
        if bytecode_manifest_json_path:
            await upload_bytecode_manifest_json(image, bytecode_manifest_json_path)

        # Stage 4: Build final image and push (status only).
        final_depot_tag = _depot_registry_ref(repo, oci_tag)
        logger.info(f"Stage 4: Building final image as {_safe_ref(final_depot_tag)}")
        await _stream_status(image.image_id, "building final image...")

        index_path = os.path.join(extracted_dir, "chutesfs.index")
        final_index_path = os.path.join(build_dir, "chutesfs.index")
        shutil.copy2(index_path, final_index_path)

        final_dockerfile_content = f"""FROM {chutes_depot_tag}
COPY chutesfs.index /etc/chutesfs.index
ENV PYTHONDONTWRITEBYTECODE=1
"""
        if bytecode_manifest_path:
            final_manifest_path = os.path.join(build_dir, "bytecode.manifest")
            shutil.copy2(bytecode_manifest_path, final_manifest_path)
            final_dockerfile_content += "COPY bytecode.manifest /etc/bytecode.manifest\n"

        if semcomp(image.chutes_version or "0.0.0", "0.5.5") >= 0:
            final_dockerfile_content += (
                "USER root\n"
                "RUN echo '/usr/local/lib/chutes-aegis.so' > /etc/ld.so.preload && chmod 0644 /etc/ld.so.preload\n"
                "USER chutes\n"
                "ENV LD_PRELOAD=/usr/local/lib/chutes-aegis.so\n"
            )
        final_dockerfile_content += "ENTRYPOINT []\n"

        final_dockerfile_path = os.path.join(build_dir, "Dockerfile.final")
        with open(final_dockerfile_path, "w") as f:
            f.write(final_dockerfile_content)

        process = await _depot_build(
            [
                "--save",
                "--tag",
                final_depot_tag,
                "-f",
                final_dockerfile_path,
                ".",
            ],
            cwd=build_dir,
            capture_logs=lambda s, n: _drain_logs(s, n, label=f"final {short_tag}"),
            timeout=settings.build_timeout,
        )
        if process.returncode != 0:
            raise BuildFailure(f"Final build of {_safe_ref(final_depot_tag)} failed!")
        await _depot_push(process, final_depot_tag)

        delta = time.time() - started_at
        message = f"image built and pushed in {round(delta, 1)} seconds"
        logger.success(message)
        await _stream_status(image.image_id, message)

    except asyncio.TimeoutError:
        message = f"Build timed out after {settings.build_timeout} seconds."
        logger.error(message)
        await settings.redis_client.client.xadd(
            f"forge:{image.image_id}:stream",
            {"data": json.dumps({"log_type": "stderr", "log": message}).decode()},
        )
        await settings.redis_client.client.xadd(f"forge:{image.image_id}:stream", {"data": "DONE"})
        process.kill()
        await process.communicate()
        raise BuildTimeout(message)

    # Stage 5: Trivy scan via remote multi-stage build.
    await trivy_image_scan(image, final_depot_tag, build_dir, _capture_logs)

    # Stage 6: Cosign sign.
    await sign_image(image, final_depot_tag)

    # Clean up intermediate images from the registry.
    await _cleanup_intermediate_tags(intermediate_tags, final_depot_tag)

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


async def trivy_image_scan(
    image,
    final_image_tag: str,
    build_dir: str,
    _capture_logs: Callable,
):
    """Run trivy scan remotely on Depot using a trusted scanner image."""
    await settings.redis_client.client.xadd(
        f"forge:{image.image_id}:stream",
        {
            "data": json.dumps(
                {"log_type": "stdout", "log": "scanning image with trivy..."}
            ).decode()
        },
    )
    logger.info("Scanning image with trivy (remote)...")

    scan_dockerfile = f"""FROM {final_image_tag} AS target
FROM aquasec/trivy:0.70.0 AS scanner
RUN --mount=type=bind,from=target,source=/,target=/scan-target \
    trivy rootfs --severity HIGH,CRITICAL --scanners vuln --ignore-unfixed /scan-target
"""
    scan_dockerfile_path = os.path.join(build_dir, "Dockerfile.scan")
    with open(scan_dockerfile_path, "w") as f:
        f.write(scan_dockerfile)

    try:
        process = await _depot_build(
            [
                "-f",
                scan_dockerfile_path,
                "--no-cache",
                ".",
            ],
            cwd=build_dir,
            capture_logs=_capture_logs,
            timeout=settings.scan_timeout,
        )
        if process.returncode == 0:
            short_tag = (
                final_image_tag.split(":")[-1] if ":" in final_image_tag else final_image_tag
            )
            message = f"No HIGH|CRITICAL vulnerabilities detected in {short_tag}"
            await settings.redis_client.client.xadd(
                f"forge:{image.image_id}:stream",
                {"data": json.dumps({"log_type": "stdout", "log": message}).decode()},
            )
            logger.success(message)
        else:
            message = "Issues scanning image with trivy!"
            await settings.redis_client.client.xadd(
                f"forge:{image.image_id}:stream",
                {"data": json.dumps({"log_type": "stderr", "log": message}).decode()},
            )
            logger.error(message)
            raise BuildFailure(f"Failed trivy image scan: {_safe_ref(final_image_tag)}")
    except asyncio.TimeoutError:
        message = "Trivy scan timed out."
        logger.error(message)
        await settings.redis_client.client.xadd(
            f"forge:{image.image_id}:stream",
            {"data": json.dumps({"log_type": "stderr", "log": message}).decode()},
        )
        await settings.redis_client.client.xadd(f"forge:{image.image_id}:stream", {"data": "DONE"})
        process.kill()
        await process.communicate()
        raise BuildTimeout(message)


async def forge(image_id: str):
    """
    Build an image and push it to Depot's registry.
    """
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

        if os.path.exists(log_path := os.path.join(build_dir, "build.log")):
            destination = f"forge/{image.user_id}/{image.image_id}.log"
            async with settings.s3_client() as s3:
                await s3.upload_file(log_path, settings.storage_bucket, destination)

    async with get_session() as session:
        result = await session.execute(select(Image).where(Image.image_id == image_id).limit(1))
        image = result.scalar_one_or_none()
        if not image:
            logger.warning(f"Image vanished while building! {image_id}")
            return
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


async def update_chutes_lib(image_id: str, chutes_version: str, force: bool = False):
    """
    Update the chutes library in an existing image without rebuilding from scratch.
    Uses Depot remote builders instead of local buildah.
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

    repo = f"{image.user.username}/{image.name}".lower()
    base_oci_tag = image.tag.lower()
    if image.patch_version and image.patch_version != "initial":
        source_oci_tag = f"{base_oci_tag}-{image.patch_version}"
    else:
        source_oci_tag = base_oci_tag
    target_oci_tag = f"{base_oci_tag}-{patch_version}"
    target_tag = f"{repo}:{target_oci_tag}"

    source_depot_tag = _depot_registry_ref(repo, source_oci_tag)
    target_depot_tag = _depot_registry_ref(repo, target_oci_tag)

    error_message = None
    success = False
    inspecto_hash = None
    package_hashes = None
    intermediate_tags: list[str] = []
    with tempfile.TemporaryDirectory() as build_dir:
        try:
            _copy_cfsv_binary(image, build_dir)

            # Stage 1: Build updated base image with new chutes lib.
            updated_oci_tag = await _safe_intermediate_tag(repo, target_oci_tag, "updated")
            updated_depot_tag = _depot_registry_ref(repo, updated_oci_tag)
            logger.info(f"Stage 1: Building updated image as {_safe_ref(updated_depot_tag)}")

            dockerfile_content = f"""FROM {source_depot_tag}
USER root
ENV LD_PRELOAD=""
RUN rm -f /etc/chutesfs.index /usr/bin/cautious-launcher /etc/ld.so.preload
RUN usermod -aG root chutes || true
RUN chmod g+rwx /usr/local/lib /usr/local/bin /usr/local/share /usr/local/share/man
RUN chmod g+rwx /usr/local/lib/python3.12/dist-packages || true
RUN find / -xdev -type f -name '*.pyc' -exec rm -f {{}} \\; || true
RUN find / -xdev -type d -name __pycache__ -exec rm -rf {{}} \\; || true
USER chutes
ENV PYTHONDONTWRITEBYTECODE=1
RUN pip install chutes=={chutes_version}
RUN uv cache clean --force
"""
            if semcomp(chutes_version or "0.0.0", "0.5.5") >= 0:
                dockerfile_content += """RUN cp -f $(python -c 'import chutes; import os; print(os.path.join(os.path.dirname(chutes.__file__), "chutes-aegis.so"))') /usr/local/lib/chutes-aegis.so
ENV LD_PRELOAD=/usr/local/lib/chutes-aegis.so
"""
            else:
                dockerfile_content += """RUN cp -f $(python -c 'import chutes; import os; print(os.path.join(os.path.dirname(chutes.__file__), "chutes-netnanny.so"))') /usr/local/lib/chutes-netnanny.so
RUN cp -f $(python -c 'import chutes; import os; print(os.path.join(os.path.dirname(chutes.__file__), "chutes-logintercept.so"))') /usr/local/lib/chutes-logintercept.so
RUN cp -f $(python -c 'import chutes; import os; print(os.path.join(os.path.dirname(chutes.__file__), "chutes-cfsv.so"))') /usr/local/lib/chutes-cfsv.so
ENV LD_PRELOAD=/usr/local/lib/chutes-netnanny.so:/usr/local/lib/chutes-logintercept.so
"""
            dockerfile_path = os.path.join(build_dir, "Dockerfile.update")
            with open(dockerfile_path, "w") as f:
                f.write(dockerfile_content)

            process = await _depot_build(
                [
                    "--save",
                    "--tag",
                    updated_depot_tag,
                    "-f",
                    dockerfile_path,
                    ".",
                ],
                cwd=build_dir,
                capture_logs=lambda s, n: _drain_logs(s, n, label=f"update {target_tag}"),
                timeout=settings.build_timeout,
            )
            if process.returncode != 0:
                raise BuildFailure("Failed to build updated image!")
            await _depot_push(process, updated_depot_tag)
            intermediate_tags.append(updated_depot_tag)

            # Stage 2: Filesystem verification + extract.
            logger.info("Stage 2: Building filesystem verification image")

            fsv_dockerfile_content = f"""FROM {updated_depot_tag}
ARG CFSV_OP
ARG PS_OP
USER chutes
ENV LD_PRELOAD=""
ENV PYTHONDONTWRITEBYTECODE=1
RUN rm -rf does_not_exist.py does_not_exist
RUN PS_OP="${{PS_OP}}" chutes run does_not_exist:chute --generate-inspecto-hash > /tmp/inspecto.hash
USER root
RUN rm -f /etc/ld.so.preload /etc/bytecode.manifest /tmp/chutesfs.index /etc/chutesfs.index /tmp/chutesfs.data
USER chutes
COPY cfsv /cfsv
RUN --network=none CFSV_OP="${{CFSV_OP}}" /cfsv index / /tmp/chutesfs.index
USER root
RUN cp -f /tmp/chutesfs.index /etc/chutesfs.index && chmod a+r /etc/chutesfs.index
USER chutes
RUN --network=none CFSV_OP="${{CFSV_OP}}" /cfsv collect / /etc/chutesfs.index /tmp/chutesfs.data
"""

            has_bcm = False
            build_bcm_path = os.path.join(build_dir, "chutes-bcm.so")
            build_driver_path = os.path.join(build_dir, "generate_manifest_driver.py")
            if (
                semcomp(chutes_version or "0.0.0", "0.5.5") >= 0
                and os.path.exists(BCM_SO_PATH)
                and os.path.exists(MANIFEST_DRIVER_PATH)
            ):
                shutil.copy2(BCM_SO_PATH, build_bcm_path)
                shutil.copy2(MANIFEST_DRIVER_PATH, build_driver_path)
                has_bcm = True
                fsv_dockerfile_content += """COPY chutes-bcm.so /tmp/chutes-bcm.so
COPY generate_manifest_driver.py /tmp/generate_manifest_driver.py
RUN CFSV_OP="${CFSV_OP}" python /tmp/generate_manifest_driver.py \
    --output /tmp/bytecode.manifest \
    --json-output /tmp/bytecode.manifest.json \
    --lib /tmp/chutes-bcm.so \
    --extra-dirs /usr/local/lib/python3.12/site-packages
"""

            has_package_hashes = False
            if semcomp(chutes_version, "0.5.3") >= 0 and image.name in ("sglang", "vllm"):
                from api.user.service import chutes_user_id

                if image.user_id == await chutes_user_id():
                    has_package_hashes = True
                    fsv_dockerfile_content += """
USER root
RUN cp -f /tmp/bytecode.manifest /etc/bytecode.manifest || true
USER chutes
RUN CFSV_OP="${CFSV_OP}" python -m cllmv.pkg_hash > /tmp/package_hashes.json
"""

            # Extract stage.
            fsv_dockerfile_content += "\nFROM scratch AS extract\n"
            fsv_dockerfile_content += "COPY --from=0 /tmp/chutesfs.data /chutesfs.data\n"
            fsv_dockerfile_content += "COPY --from=0 /tmp/inspecto.hash /inspecto.hash\n"
            fsv_dockerfile_content += "COPY --from=0 /tmp/chutesfs.index /chutesfs.index\n"
            if has_bcm:
                fsv_dockerfile_content += (
                    "COPY --from=0 /tmp/bytecode.manifest /bytecode.manifest\n"
                )
                fsv_dockerfile_content += (
                    "COPY --from=0 /tmp/bytecode.manifest.json /bytecode.manifest.json\n"
                )
            if has_package_hashes:
                fsv_dockerfile_content += (
                    "COPY --from=0 /tmp/package_hashes.json /package_hashes.json\n"
                )

            fsv_dockerfile_path = os.path.join(build_dir, "Dockerfile.fsv")
            with open(fsv_dockerfile_path, "w") as f:
                f.write(fsv_dockerfile_content)

            extracted_dir = os.path.join(build_dir, "extracted")
            os.makedirs(extracted_dir, exist_ok=True)

            process = await _depot_build(
                [
                    "--build-arg",
                    f"CFSV_OP={os.getenv('CFSV_OP', str(uuid.uuid4()))}",
                    "--build-arg",
                    f"PS_OP={os.getenv('PS_OP', str(uuid.uuid4()))}",
                    "--output",
                    f"type=local,dest={extracted_dir}",
                    "--target",
                    "extract",
                    "-f",
                    fsv_dockerfile_path,
                    ".",
                ],
                cwd=build_dir,
                capture_logs=lambda s, n: _drain_logs(s, n, label=f"fsv {target_tag}"),
                timeout=settings.build_timeout,
            )
            if process.returncode != 0:
                raise BuildFailure("Failed to build filesystem verification image!")

            # Read extracted files.
            data_file_path = os.path.join(extracted_dir, "chutesfs.data")
            if not os.path.exists(data_file_path):
                raise BuildFailure("chutesfs.data not found in extracted output")

            with open(os.path.join(extracted_dir, "inspecto.hash")) as infile:
                inspecto_hash = infile.readlines()[-1].strip()
                assert inspecto_hash

            hashes_json_path = os.path.join(extracted_dir, "package_hashes.json")
            if os.path.exists(hashes_json_path):
                with open(hashes_json_path, "r") as infile:
                    package_hashes = json.loads(infile.read())

            # Upload cfsv data.
            s3_key = f"image_hash_blobs/{image_id}/{patch_version}.data"
            async with settings.s3_client() as s3:
                await s3.upload_file(data_file_path, settings.storage_bucket, s3_key)
            logger.success(f"Uploaded filesystem verification data to {s3_key}")

            bytecode_manifest_path = os.path.join(extracted_dir, "bytecode.manifest")
            if not os.path.exists(bytecode_manifest_path):
                bytecode_manifest_path = None
            bytecode_manifest_json_path = os.path.join(extracted_dir, "bytecode.manifest.json")
            if not os.path.exists(bytecode_manifest_json_path):
                bytecode_manifest_json_path = None

            if bytecode_manifest_path:
                manifest_s3_key = f"image_hash_blobs/{image_id}/{patch_version}.manifest"
                async with settings.s3_client() as s3:
                    await s3.upload_file(
                        bytecode_manifest_path, settings.storage_bucket, manifest_s3_key
                    )
                logger.success(f"Uploaded bytecode manifest to {manifest_s3_key}")

            if bytecode_manifest_json_path:
                manifest_json_s3_key = f"image_hash_blobs/{image_id}/{patch_version}.manifest.json"
                async with settings.s3_client() as s3:
                    await s3.upload_file(
                        bytecode_manifest_json_path, settings.storage_bucket, manifest_json_s3_key
                    )
                logger.success(f"Uploaded bytecode manifest JSON to {manifest_json_s3_key}")

            # Stage 3: Build final image and push.
            logger.info(f"Stage 3: Building final image as {_safe_ref(target_depot_tag)}")

            index_path = os.path.join(extracted_dir, "chutesfs.index")
            final_index_path = os.path.join(build_dir, "chutesfs.index")
            shutil.copy2(index_path, final_index_path)

            final_dockerfile_content = f"""FROM {updated_depot_tag}
COPY chutesfs.index /etc/chutesfs.index
ENV PYTHONDONTWRITEBYTECODE=1
"""
            if bytecode_manifest_path:
                final_manifest_path = os.path.join(build_dir, "bytecode.manifest")
                shutil.copy2(bytecode_manifest_path, final_manifest_path)
                final_dockerfile_content += "COPY bytecode.manifest /etc/bytecode.manifest\n"
            if semcomp(chutes_version or "0.0.0", "0.5.5") >= 0:
                final_dockerfile_content += (
                    "USER root\n"
                    "RUN echo '/usr/local/lib/chutes-aegis.so' > /etc/ld.so.preload && chmod 0644 /etc/ld.so.preload\n"
                    "USER chutes\n"
                    "ENV LD_PRELOAD=/usr/local/lib/chutes-aegis.so\n"
                )
            final_dockerfile_content += "ENTRYPOINT []\n"

            final_dockerfile_path = os.path.join(build_dir, "Dockerfile.final")
            with open(final_dockerfile_path, "w") as f:
                f.write(final_dockerfile_content)

            process = await _depot_build(
                [
                    "--save",
                    "--tag",
                    target_depot_tag,
                    "-f",
                    final_dockerfile_path,
                    ".",
                ],
                cwd=build_dir,
                capture_logs=lambda s, n: _drain_logs(s, n, label=f"final {target_tag}"),
                timeout=settings.build_timeout,
            )
            if process.returncode != 0:
                raise BuildFailure("Failed to build final image!")
            await _depot_push(process, target_depot_tag)

            logger.success(f"Successfully pushed updated image {_safe_ref(target_depot_tag)}")
            success = True

        except asyncio.TimeoutError:
            message = f"Update of {_safe_ref(target_depot_tag)} timed out"
            logger.error(message)
            process.kill()
            await process.communicate()
            raise BuildTimeout(message)
        except Exception as exc:
            logger.error(
                f"Error updating chutes lib for {image_id}: {exc}\n{traceback.format_exc()}"
            )
            error_message = str(exc)

        # Clean up intermediate images from the registry.
        await _cleanup_intermediate_tags(intermediate_tags, target_depot_tag)

        # Sign if successful.
        if success:
            await sign_image(image, target_depot_tag, stream=True)

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

                await session.commit()

            for chute in chutes:
                logger.warning(f"Triggering rolling update task: {chute.chute_id=}")
                await handle_rolling_update.kiq(
                    chute.chute_id, chute.version, reason="image updated due to chutes lib upgrade"
                )

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
