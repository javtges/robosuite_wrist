#!/usr/bin/env bash
# Generates the demo datasets, every source recorded (wrist camera, doorview scene camera, wrist wrench, robot
# state, full action), one dataset per wrist mode, each in SHARDS parallel processes:
#
#   bash visuomotor/sh/gen_datasets.sh                                  # all three: static znoise schedule
#   MODES="schedule" EPISODES=200 SHARDS=32 bash visuomotor/sh/gen_datasets.sh
#   MODES=static STATIC_Z=-1 bash visuomotor/sh/gen_datasets.sh --camera-size 96 128   # extra flags -> gen_demos
#
#   MODES       wrist modes to generate (default "static znoise schedule")
#   EPISODES    episodes per dataset (default 100)
#   SHARDS      processes per dataset (default 16); each writes shard<i>_lock<f>.hdf5, the trainer reads them all
#   STATIC_Z    static mode's z (default 0 = home)
#   SEED        plan seed (default 0): same seed = same door poses and aiming errors in every mode
# Output: $DATA_ROOT/<mode>/shard<i>_lock<f>.hdf5, logs in $DATA_ROOT/<mode>/logs/. Refuses to write into a mode
# directory that already holds shards (delete it, or set DATA_ROOT elsewhere).
set -euo pipefail
source "$(dirname "$0")/env.sh"
MODES="${MODES:-static znoise schedule}"
EPISODES="${EPISODES:-100}"
SHARDS="${SHARDS:-16}"
STATIC_Z="${STATIC_Z:-0}"
SEED="${SEED:-0}"

for mode in $MODES; do
  out="$DATA_ROOT/$mode"
  if compgen -G "$out/shard*.hdf5" > /dev/null; then
    echo "$out already holds shards; not overwriting" >&2
    exit 1
  fi
  mkdir -p "$out/logs"
  echo "[$mode] $EPISODES episodes in $SHARDS shards -> $out"
  for ((i = 0; i < SHARDS; i++)); do
    "$PY" -m visuomotor.gen_demos --wrist-mode "$mode" --static-z "$STATIC_Z" --episodes "$EPISODES" \
      --shard "$i/$SHARDS" --seed "$SEED" --out "$out/shard$i.hdf5" "$@" > "$out/logs/shard$i.log" 2>&1 &
  done
done
wait
for mode in $MODES; do
  grep -h "^done:" "$DATA_ROOT/$mode/logs/"*.log | awk -v m="$mode" '{split($2, a, "/"); o += a[1]; n += a[2]} END {printf "[%s] %d/%d opened\n", m, o, n}'
done
