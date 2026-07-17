#!/usr/bin/env bash
#
# reproduce_graval_fsv.sh
#
# Reproduce the filesystem-verification hash the way GRAVAL produces it at launch
# (the `expected_hash`): download the stored S3 datamap
# (image_hash_blobs/<image_id>/<patch_version>.data) and run
# `cfsv validate <seed> <mode> <datamap> <exclude>`.
#
# This is the counterpart to reproduce_miner_fsv.sh. Run both with the same
# image / seed / exclude and compare:
#   - graval hash (this script)  == expected_hash  (from the stored S3 datamap)
#   - miner hash (other script)  == received        (from the live image FS)
# If they differ, the stored datamap does not correspond to the shipped image
# (stale / cache-skewed artifact).
#
# NOTE: there is no separate "forge hash" -- forge only uploads the datamap; the
# hash derived from that datamap IS this graval validate. So this script fully
# covers "the hash based on the S3 datamap".
#
# Requirements (satisfied inside a graval-worker pod):
#   - `uv` + the api package (provides api.graval_worker.generate_fs_hash)
#   - S3 credentials + CFSV_OP env (present in graval/forge pods)
#   - the cfsv binaries bundled with the chutes lib
#
# Usage:
#   ./reproduce_graval_fsv.sh --image-id <id> --seed <config_id> --exclude </app/<chute.filename>>
#   ./reproduce_graval_fsv.sh --image-id <id> --patch-version <pv> --seed <config_id> --exclude </app/<file>>
#
# Example:
#   ./reproduce_graval_fsv.sh \
#       --image-id 300c6c17-5e11-500e-89fc-2b0930b86c52 \
#       --seed 8a71b2d3-ea9c-44c2-a62a-61358afaff8a \
#       --exclude /app/sv_chutes.py
#
set -euo pipefail

IMAGE_ID=""
PATCH_VERSION=""
SEED=""
EXCLUDE=""
MODE="full"      # full | sparse
RUNS=2           # repeat for a determinism check

die() { echo "ERROR: $*" >&2; exit 1; }

usage() {
  sed -n '2,35p' "$0" | sed 's/^# \{0,1\}//'
  exit "${1:-0}"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --image-id)      IMAGE_ID="$2"; shift 2;;
    --patch-version) PATCH_VERSION="$2"; shift 2;;
    --seed)          SEED="$2"; shift 2;;
    --exclude)       EXCLUDE="$2"; shift 2;;
    --mode)          MODE="$2"; shift 2;;
    --runs)          RUNS="$2"; shift 2;;
    -h|--help)       usage 0;;
    *) die "unknown argument: $1 (use --help)";;
  esac
done

[[ -n "$IMAGE_ID" ]] || die "--image-id is required"
[[ -n "$SEED" ]]     || die "--seed <config_id> is required"
[[ -n "$EXCLUDE" ]]  || die "--exclude </app/<chute.filename>> is required"
[[ "$MODE" == "full" || "$MODE" == "sparse" ]] || die "--mode must be 'full' or 'sparse'"
command -v uv >/dev/null 2>&1 || die "uv not found on PATH"

# Resolve patch_version from the DB if not supplied (matches what the API passes
# to generate_fs_hash: chute.image.patch_version).
if [[ -z "$PATCH_VERSION" ]]; then
  echo ">> resolving patch_version for image_id=$IMAGE_ID ..." >&2
  PATCH_VERSION="$(uv run python - "$IMAGE_ID" <<'PY'
import asyncio, sys
import api.database.orms  # noqa: F401
from sqlalchemy import select
from api.database import get_session
from api.image.schemas import Image

async def main(image_id):
    async with get_session() as s:
        img = (await s.execute(select(Image).where(Image.image_id == image_id))).scalar_one()
        print(img.patch_version or "initial")

asyncio.run(main(sys.argv[1]))
PY
)"
  [[ -n "$PATCH_VERSION" ]] || die "failed to resolve patch_version for image_id=$IMAGE_ID"
  echo ">> resolved patch_version: $PATCH_VERSION" >&2
fi

echo ">> validating stored S3 datamap image_hash_blobs/$IMAGE_ID/$PATCH_VERSION.data ..." >&2

uv run python - "$IMAGE_ID" "$PATCH_VERSION" "$SEED" "$EXCLUDE" "$MODE" "$RUNS" <<'PY'
import asyncio, sys
import api.database.orms  # noqa: F401
from api.graval_worker import generate_fs_hash

image_id, patch_version, seed, exclude, mode, runs = (
    sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5], int(sys.argv[6])
)
sparse = mode == "sparse"

# Unwrap the taskiq task to its underlying coroutine function.
fn = getattr(generate_fs_hash, "original_func", generate_fs_hash)

async def main():
    last, ok = None, True
    for i in range(runs):
        h = await fn(image_id, patch_version, seed, sparse, exclude)
        print(f"graval[{i + 1}/{runs}]: {h}")
        if last is not None and h != last:
            ok = False
        last = h
    if not ok:
        print("WARNING: non-deterministic result across runs!", file=sys.stderr)
    print("GRAVALHASH=" + (last or "NONE"))

asyncio.run(main())
PY

echo
echo "=================================================================="
echo " image_id      : $IMAGE_ID"
echo " patch_version : $PATCH_VERSION"
echo " config_id     : $SEED   (seed)"
echo " exclude_path  : $EXCLUDE"
echo " mode          : $MODE"
echo "=================================================================="
echo
echo "Compare GRAVALHASH (expected, from stored S3 datamap) against the miner"
echo "hash from reproduce_miner_fsv.sh (received, from the live image FS)."
