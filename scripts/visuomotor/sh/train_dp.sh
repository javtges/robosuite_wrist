#!/usr/bin/env bash
# Trains one diffusion policy (visuomotor/dp/train_dp.py), logged to wandb:
#
#   DATASET=schedule WRIST=z OBS=wrist_cam,scene_cam,wrench,robot bash visuomotor/sh/train_dp.sh
#   DATASET=static WRIST=stiff OBS=wrist_cam,wrench SEED=1 bash visuomotor/sh/train_dp.sh --epochs 500   # extra flags -> train_dp
#
#   DATASET   static | znoise | schedule (under DATA_ROOT) or a path (file or directory); default schedule
#   WRIST     what the wrist does when the policy runs: stiff (static, z = -1) | medium (static, z = 0) | z (the policy
#             outputs the wrist's z as a 10th action); default z
#   OBS       comma-separated subset of wrist_cam,scene_cam,wrench,robot; default all four
#   SEED      default 0
#   NAME      run name; default dp_<dataset>_<wrist>_<obs>_s<seed>
#   EVAL_EVERY, EVAL_EPISODES   in-process sim eval every k epochs (default 50) on n episodes per lock (default 10)
# Output: $RUN_ROOT/dp/$NAME/ (cfg.json, latest.pt, epoch_*.pt, train.log). wandb: project $WANDB_PROJECT,
# group dp_<dataset>, tags dataset:<d> wrist:<w> obs:<o>.
set -euo pipefail
source "$(dirname "$0")/env.sh"
DATASET="${DATASET:-schedule}"
WRIST="${WRIST:-z}"
OBS="${OBS:-wrist_cam,scene_cam,wrench,robot}"
SEED="${SEED:-0}"
EVAL_EVERY="${EVAL_EVERY:-50}"
EVAL_EPISODES="${EVAL_EPISODES:-10}"
DATA="$(dataset_dir "$DATASET")"
TAG="$(dataset_tag "$DATA")"
NAME="${NAME:-dp_${TAG}_${WRIST}_${OBS//,/-}_s${SEED}}"
OUT="$RUN_ROOT/dp/$NAME"
mkdir -p "$OUT"
echo "[$NAME] $DATA -> $OUT"
"$PY" -m visuomotor.dp.train_dp --data "$DATA" --wrist "$WRIST" --obs "$OBS" --seed "$SEED" --out "$OUT" \
  --wandb-group "dp_${TAG}" --wandb-name "$NAME" --tags "dataset:$TAG" "wrist:$WRIST" "obs:$OBS" \
  --eval-every "$EVAL_EVERY" --eval-episodes "$EVAL_EPISODES" "$@" 2>&1 | tee -a "$OUT/train.log"
