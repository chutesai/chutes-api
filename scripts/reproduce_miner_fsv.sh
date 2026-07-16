#!/usr/bin/env bash
#
# reproduce_miner_fsv.sh
#
# Reproduce the filesystem-verification hash EXACTLY the way a miner instance
# produces it at launch (the `received`/`args.fsv` value), so it can be compared
# against graval's `validate` output (the `expected_hash` derived from the stored
# S3 datamap).
#
# Why a depot build and not a plain `docker run`?
#   - The miner computes the hash via chutes' `generate_filesystem_hash()`, which
#     loads `cfsv_challenge` out of chutes-aegis.so and reads the baked
#     /etc/chutesfs.index + the live rootfs. It needs the CFSV_OP secret and must
#     run INSIDE the image. Aegis blocks entrypoint overrides, so we bake a tiny
#     RUN layer into a throwaway build (FROM the image) and read the printed hash
#     from the build log. This is the same path the running instance takes.
#
# Requirements (all satisfied inside a forge pod):
#   - `depot` CLI, authenticated to the depot project (forge already is)
#   - CFSV_OP env var set (present in the forge/graval pods)
#   - `uv` (only needed for --image-id ref resolution)
#
# Usage:
#   ./reproduce_miner_fsv.sh --ref <depot_image_ref> --seed <config_id> --exclude </app/<chute.filename>>
#   ./reproduce_miner_fsv.sh --image-id <image_id>    --seed <config_id> --exclude </app/<chute.filename>>
#
# Examples:
#   ./reproduce_miner_fsv.sh \
#       --ref xkdl0v4hq8.registry.depot.dev/sv_3/turbovision-crime-v4:latest-abc123def456 \
#       --seed 8a71b2d3-ea9c-44c2-a62a-61358afaff8a \
#       --exclude /app/sv_chutes.py
#
#   # Resolve the depot ref automatically from the DB (uses the image's current
#   # tag + patch_version), so you don't have to remember it:
#   ./reproduce_miner_fsv.sh \
#       --image-id 300c6c17-5e11-500e-89fc-2b0930b86c52 \
#       --seed 8a71b2d3-ea9c-44c2-a62a-61358afaff8a \
#       --exclude /app/sv_chutes.py
#
set -euo pipefail

REF=""
IMAGE_ID=""
SEED=""
EXCLUDE=""
MODE="full"
KEEP=0

die() { echo "ERROR: $*" >&2; exit 1; }

usage() {
  sed -n '2,40p' "$0" | sed 's/^# \{0,1\}//'
  exit "${1:-0}"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --ref)      REF="$2"; shift 2;;
    --image-id) IMAGE_ID="$2"; shift 2;;
    --seed)     SEED="$2"; shift 2;;
    --exclude)  EXCLUDE="$2"; shift 2;;
    --mode)     MODE="$2"; shift 2;;
    --keep)     KEEP=1; shift;;
    -h|--help)  usage 0;;
    *) die "unknown argument: $1 (use --help)";;
  esac
done

[[ -n "$SEED" ]]    || die "--seed <config_id> is required"
[[ -n "$EXCLUDE" ]] || die "--exclude </app/<chute.filename>> is required"
command -v depot >/dev/null 2>&1 || die "depot CLI not found on PATH"
: "${CFSV_OP:?CFSV_OP must be set in the environment (it is inside the forge/graval pods)}"

# ---------------------------------------------------------------------------
# Resolve the depot image ref from the DB if only --image-id was provided.
# Uses the repo's own helpers so it stays in sync with how forge tags images.
# ---------------------------------------------------------------------------
if [[ -z "$REF" ]]; then
  [[ -n "$IMAGE_ID" ]] || die "provide either --ref or --image-id"
  echo ">> resolving depot ref for image_id=$IMAGE_ID ..." >&2
  REF="$(uv run python - "$IMAGE_ID" <<'PY'
import asyncio, sys
import api.database.orms  # noqa: F401  (register ORM mappers)
from sqlalchemy import select
from api.database import get_session
from api.image.schemas import Image
from api.image.remote_forge import _depot_repo_ref

async def main(image_id):
    async with get_session() as s:
        img = (await s.execute(select(Image).where(Image.image_id == image_id))).scalar_one()
        await s.refresh(img, ["user"])
        repo = f"{img.user.username}/{img.name}".lower()
        oci_tag = img.tag.lower()
        if img.patch_version and img.patch_version != "initial":
            oci_tag = f"{oci_tag}-{img.patch_version}"
        print(_depot_repo_ref(repo, oci_tag))

asyncio.run(main(sys.argv[1]))
PY
)"
  [[ -n "$REF" ]] || die "failed to resolve depot ref for image_id=$IMAGE_ID"
  echo ">> resolved ref: $REF" >&2
fi

# ---------------------------------------------------------------------------
# Build context: the miner-hash python + a throwaway Dockerfile.
# ---------------------------------------------------------------------------
workdir="$(mktemp -d)"
if [[ "$KEEP" -eq 1 ]]; then
  echo ">> keeping build context at $workdir" >&2
else
  trap 'rm -rf "$workdir"' EXIT
fi

printf '%s' "$CFSV_OP" > "$workdir/cfsv_op.secret"

# The exact code path the miner runs at launch (chutes/entrypoint/run.py:
# generate_filesystem_hash -> cfsv.challenge(salt, mode, "/", "/etc/chutesfs.index", exclude)).
# Falls back to the wrapper directly if importing the entrypoint module fails.
cat > "$workdir/_miner_fsv.py" <<'PY'
import asyncio, sys

seed = sys.argv[1]
exclude = sys.argv[2]
mode = sys.argv[3] if len(sys.argv) > 3 else "full"

h = None
try:
    from chutes.entrypoint.run import generate_filesystem_hash as g
    h = asyncio.run(g(seed, exclude, mode))
except Exception as e:  # heavy entrypoint import unavailable under aegis, etc.
    sys.stderr.write(f"[fallback] entrypoint path failed ({e}); using cfsv_wrapper directly\n")
    from chutes.cfsv_wrapper import get_cfsv
    h = get_cfsv().challenge(seed, mode, "/", "/etc/chutesfs.index", exclude)

print("MINERHASH=" + (h or "NONE"), flush=True)
PY

cat > "$workdir/Dockerfile" <<EOF
FROM $REF
USER chutes
WORKDIR /app
ARG CACHEBUST=0
ARG FSV_SEED
ARG FSV_EXCLUDE
ARG FSV_MODE=full
COPY _miner_fsv.py /tmp/_miner_fsv.py
RUN --mount=type=secret,id=cfsv_op,mode=0444 \\
    CB="\$CACHEBUST" CFSV_OP="\$(cat /run/secrets/cfsv_op)" \\
    python /tmp/_miner_fsv.py "\$FSV_SEED" "\$FSV_EXCLUDE" "\$FSV_MODE"
EOF

log="$workdir/build.log"
echo ">> running miner challenge inside $REF ..." >&2

# No output/--save: we only need the RUN layer to execute and print. CACHEBUST
# guarantees the RUN re-executes (its stdout is what we scrape) instead of being
# served from depot's layer cache.
depot build \
  --secret "id=cfsv_op,src=$workdir/cfsv_op.secret" \
  --build-arg "CACHEBUST=$(date +%s)-$RANDOM" \
  --build-arg "FSV_SEED=$SEED" \
  --build-arg "FSV_EXCLUDE=$EXCLUDE" \
  --build-arg "FSV_MODE=$MODE" \
  --progress plain \
  -f "$workdir/Dockerfile" \
  "$workdir" 2>&1 | tee "$log"

hash="$(grep -oE 'MINERHASH=[0-9a-fA-F]+|MINERHASH=NONE' "$log" | tail -1 | cut -d= -f2 || true)"
[[ -n "$hash" && "$hash" != "NONE" ]] || die "could not extract miner hash from build log (see above)"

echo
echo "=================================================================="
echo " image ref     : $REF"
echo " config_id     : $SEED   (salt)"
echo " exclude_path  : $EXCLUDE"
echo " mode          : $MODE"
echo " MINER FSV HASH: $hash"
echo "=================================================================="
echo
echo "Compare against graval validate (expected_hash) for the same image/patch/seed."
