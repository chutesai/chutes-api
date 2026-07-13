#!/usr/bin/env bash
#
# reproduce_fresh_collect_fsv.sh
#
# Collect a FRESH datamap directly from a shipped image (using the image's baked
# /etc/chutesfs.index), then validate it and print its sha256. This is the "#3
# oracle": what the datamap *should* be for the image as it actually ships.
#
# Use it to answer "does the stored S3 datamap match the image?":
#   - FRESH_VALIDATE  should equal the MINER hash (reproduce_miner_fsv.sh) if the
#     image is internally consistent.
#   - FRESH_DATA_SHA  compared to the sha of the stored S3 <patch>.data tells you
#     whether the S3 datamap was collected from a DIFFERENT filesystem state.
#   - index_verbose.txt (extracted) lists the current image's indexed files, to
#     localize which files drifted vs the build-time datamap.
#
# Runs the collect with aegis disabled (LD_PRELOAD="") exactly like the forge
# Stage-3 collect, inside a depot build FROM the image.
#
# Run in a forge pod (needs depot + CFSV_OP).
#
# Usage:
#   ./reproduce_fresh_collect_fsv.sh --ref <image_ref> --seed <config_id> --exclude </app/<file>> [--out DIR]
#
set -euo pipefail

REF=""; SEED=""; EXCLUDE=""; MODE="full"; OUT="/tmp/fresh_collect_out"
die() { echo "ERROR: $*" >&2; exit 1; }

while [[ $# -gt 0 ]]; do
  case "$1" in
    --ref)     REF="$2"; shift 2;;
    --seed)    SEED="$2"; shift 2;;
    --exclude) EXCLUDE="$2"; shift 2;;
    --mode)    MODE="$2"; shift 2;;
    --out)     OUT="$2"; shift 2;;
    -h|--help) sed -n '2,26p' "$0" | sed 's/^# \{0,1\}//'; exit 0;;
    *) die "unknown argument: $1";;
  esac
done

[[ -n "$REF" ]]     || die "--ref <image_ref> is required"
[[ -n "$SEED" ]]    || die "--seed <config_id> is required"
[[ -n "$EXCLUDE" ]] || die "--exclude </app/<file>> is required"
command -v depot >/dev/null 2>&1 || die "depot CLI not found"
: "${CFSV_OP:?CFSV_OP must be set}"

work="$(mktemp -d)"; trap 'rm -rf "$work"' EXIT
printf '%s' "$CFSV_OP" > "$work/cfsv_op.secret"
rm -rf "$OUT"; mkdir -p "$OUT"

cat > "$work/run.sh" <<'SH'
set -e
export CFSV_OP="$(cat /run/secrets/cfsv_op)"
BIN=/usr/local/lib/python3.12/dist-packages/chutes/cfsv_v4
"$BIN" collect / /etc/chutesfs.index /tmp/fresh.data
sha256sum /tmp/fresh.data | awk '{print $1}' > /tmp/fresh.data.sha
"$BIN" validate "$FSV_SEED" "$FSV_MODE" /tmp/fresh.data "$FSV_EXCLUDE" > /tmp/fresh.validate 2>&1 || true
"$BIN" index / /tmp/fresh.index --verbose > /tmp/index_verbose.txt 2>&1 || true
SH

cat > "$work/Dockerfile" <<EOF
FROM $REF
USER root
ENV LD_PRELOAD=""
ARG CACHEBUST=0
ARG FSV_SEED
ARG FSV_EXCLUDE
ARG FSV_MODE=full
ENV FSV_SEED=\$FSV_SEED FSV_EXCLUDE=\$FSV_EXCLUDE FSV_MODE=\$FSV_MODE CACHEBUST=\$CACHEBUST
COPY run.sh /tmp/run.sh
RUN --network=none --mount=type=secret,id=cfsv_op,mode=0444 sh /tmp/run.sh
FROM scratch AS out
COPY --from=0 /tmp/fresh.data /fresh.data
COPY --from=0 /tmp/fresh.data.sha /fresh.data.sha
COPY --from=0 /tmp/fresh.validate /fresh.validate
COPY --from=0 /tmp/index_verbose.txt /index_verbose.txt
EOF

depot build \
  --secret "id=cfsv_op,src=$work/cfsv_op.secret" \
  --build-arg "CACHEBUST=$(date +%s)-$RANDOM" \
  --build-arg "FSV_SEED=$SEED" \
  --build-arg "FSV_EXCLUDE=$EXCLUDE" \
  --build-arg "FSV_MODE=$MODE" \
  --output "type=local,dest=$OUT" \
  --target out \
  -f "$work/Dockerfile" \
  "$work"

fresh_sha="$(cat "$OUT/fresh.data.sha" 2>/dev/null || echo '?')"
fresh_validate="$(grep -oE '[0-9a-f]{64}' "$OUT/fresh.validate" 2>/dev/null | tail -1 || echo '?')"

echo
echo "=================================================================="
echo " image ref        : $REF"
echo " config_id (seed) : $SEED"
echo " exclude_path     : $EXCLUDE"
echo " FRESH_DATA_SHA   : $fresh_sha       (sha256 of datamap collected from THIS image)"
echo " FRESH_VALIDATE   : $fresh_validate  (== miner hash if image is self-consistent)"
echo " index_verbose    : $OUT/index_verbose.txt"
echo "=================================================================="
echo
echo "Compare FRESH_DATA_SHA to the sha256 of the stored S3 <image_id>/<patch>.data."
echo "  differ  -> S3 datamap was collected from a DIFFERENT filesystem than the image."
echo "Compare FRESH_VALIDATE to the miner hash (reproduce_miner_fsv.sh)."
echo "  equal   -> the image is self-consistent; only the STORED datamap is wrong."
