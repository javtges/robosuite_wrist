#!/usr/bin/env bash
# Three diffusion policies, one per wrist configuration, all three at once (one GPU each):
#
#   stiff   trained on static demos at z = -1   ($STIFF_DATA)
#   medium  trained on static demos at z =  0   ($MEDIUM_DATA)
#   z       trained on schedule demos           ($Z_DATA): the expert moves the wrist's z by task phase
#
# Every policy sees the wrist camera, the wrist wrench and the robot's state (TCP pose + the wrist command in force),
# no scene camera, and outputs the wrist's z as its 10th action (train_dp --wrist z). For stiff and medium the
# demos hold z fixed, so the policy should learn to output a constant z.
#
#   bash visuomotor/sh/train_three_dp.sh                          # from scripts/
#   GPUS="0 0 1" bash visuomotor/sh/train_three_dp.sh --epochs 500  # extra flags -> every train_dp run
#   DRY=1 bash visuomotor/sh/train_three_dp.sh                    # print the runs, start nothing
#
# Datasets (gen_datasets.sh). Medium and z come from the default run; stiff needs its own static run at z = -1:
#   MODES="static schedule" bash visuomotor/sh/gen_datasets.sh
#   DATA_ROOT=$PWD/../data/visuomotor/stiff MODES=static STATIC_Z=-1 bash visuomotor/sh/gen_datasets.sh
#
#   STIFF_DATA, MEDIUM_DATA, Z_DATA   defaults $DATA_ROOT/stiff/static, $DATA_ROOT/static, $DATA_ROOT/schedule
#   GPUS        three CUDA device ids, stiff / medium / z (default "0 1 2")
#   SEED        default 0
#   EVAL_EVERY, EVAL_EPISODES   in-process sim eval every k epochs (default 50) on n episodes per lock (default 10)
# Output: $RUN_ROOT/dp/dp_<stiff|medium|z>_wrist_cam-wrench-robot_s<seed>/ (train.log, cfg.json, checkpoints).
# wandb: group dp_three, tags wrist:<stiff|medium|z>. A run whose latest.pt exists is skipped unless FORCE=1.
set -euo pipefail
source "$(dirname "$0")/env.sh"
STIFF_DATA="${STIFF_DATA:-$DATA_ROOT/stiff/static}"
MEDIUM_DATA="${MEDIUM_DATA:-$DATA_ROOT/static}"
Z_DATA="${Z_DATA:-$DATA_ROOT/schedule}"
read -r -a GPU_LIST <<< "${GPUS:-0 1 2}"
SEED="${SEED:-0}"
EVAL_EVERY="${EVAL_EVERY:-50}"
EVAL_EPISODES="${EVAL_EPISODES:-10}"
OBS=wrist_cam,wrench,robot

LABELS=(stiff medium z)
DATAS=("$STIFF_DATA" "$MEDIUM_DATA" "$Z_DATA")
[ ${#GPU_LIST[@]} -eq 3 ] || { echo "GPUS needs three device ids (stiff medium z), got '${GPU_LIST[*]}'" >&2; exit 1; }
for d in "${DATAS[@]}"; do
  compgen -G "$d/*.hdf5" > /dev/null || { echo "no dataset shards in $d (see the gen_datasets.sh lines at the top)" >&2; exit 1; }
done

for i in 0 1 2; do
  label="${LABELS[$i]}" data="${DATAS[$i]}" gpu="${GPU_LIST[$i]}"
  name="dp_${label}_${OBS//,/-}_s${SEED}"
  out="$RUN_ROOT/dp/$name"
  if [ -f "$out/latest.pt" ] && [ -z "${FORCE:-}" ]; then echo "skip $name (exists; resume with --resume)"; continue; fi
  echo "run $name: $data -> $out (gpu $gpu)"
  [ -n "${DRY:-}" ] && continue
  mkdir -p "$out"
  CUDA_VISIBLE_DEVICES="$gpu" "$PY" -m visuomotor.dp.train_dp --data "$data" --wrist z --obs "$OBS" --seed "$SEED" \
    --out "$out" --device cuda --wandb-group dp_three --wandb-name "$name" \
    --tags "wrist:$label" "dataset:$(basename "$data")" "obs:$OBS" \
    --eval-every "$EVAL_EVERY" --eval-episodes "$EVAL_EPISODES" "$@" >> "$out/train.log" 2>&1 &
done
wait
echo "done; logs in $RUN_ROOT/dp/dp_*_${OBS//,/-}_s${SEED}/train.log"
