#!/usr/bin/env bash
# Evaluates diffusion-policy checkpoints in the sim (visuomotor/dp/eval_dp.py), both locks, logged to wandb:
#
#   bash visuomotor/sh/eval_dp.sh dp_schedule_z_wrist_cam-scene_cam-wrench-robot_s0          # a run name: its latest.pt
#   bash visuomotor/sh/eval_dp.sh runs/visuomotor/dp/<run>/epoch_0300.pt --video 4           # a checkpoint path
#   bash visuomotor/sh/eval_dp.sh 'dp_schedule_*'                                           # every matching run, in turn
#
#   EPISODES   episodes per lock friction (default 50)
#   WORKERS    parallel envs (default 16)
# Extra flags go to eval_dp (e.g. --cylinder-frictions 0.4 --max-seconds 60 --video 4). Results:
# <run>/eval_<ckpt>.json; wandb run eval_<run> in group eval_<dataset> (the training run's dataset).
set -euo pipefail
source "$(dirname "$0")/env.sh"
[ $# -ge 1 ] || { echo "usage: eval_dp.sh <run name | glob | checkpoint path> [eval_dp flags]" >&2; exit 1; }
TARGET="$1"; shift
EPISODES="${EPISODES:-50}"
WORKERS="${WORKERS:-16}"

if [ -f "$TARGET" ]; then CKPTS=("$TARGET"); else
  CKPTS=(); for d in "$RUN_ROOT"/dp/$TARGET; do [ -f "$d/latest.pt" ] && CKPTS+=("$d/latest.pt"); done
fi
[ ${#CKPTS[@]} -gt 0 ] || { echo "no checkpoint for $TARGET under $RUN_ROOT/dp" >&2; exit 1; }
for ckpt in "${CKPTS[@]}"; do
  run="$(basename "$(dirname "$ckpt")")"
  tag="$(echo "$run" | cut -d_ -f2)"
  echo "[eval] $ckpt"
  "$PY" -m visuomotor.dp.eval_dp --ckpt "$ckpt" --episodes "$EPISODES" --workers "$WORKERS" \
    --wandb-group "eval_${tag}" --wandb-name "eval_${run}" --tags "eval" "run:$run" "$@"
done
