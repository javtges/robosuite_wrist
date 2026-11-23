#!/usr/bin/env bash
# The diffusion-policy ablation: every DATASET x WRIST x OBS set x SEED, through train_dp.sh, JOBS at a time:
#
#   bash visuomotor/sh/sweep_dp.sh                                        # the full grid (3 x 3 x 4 = 36 runs)
#   DATASETS=schedule WRISTS="z" OBS_SETS="wrist_cam,robot wrist_cam,scene_cam,robot" bash visuomotor/sh/sweep_dp.sh
#   GPUS="0 1 2 3" JOBS=8 bash visuomotor/sh/sweep_dp.sh --epochs 300    # extra flags -> every train_dp run
#   DRY=1 bash visuomotor/sh/sweep_dp.sh                                  # print the runs, start nothing
#
#   DATASETS  default "static znoise schedule"
#   WRISTS    default "stiff medium z"
#   OBS_SETS  space-separated observation sets, each a comma-separated subset of wrist_cam,scene_cam,wrench,robot.
#             Default: the camera ablation, each with wrench + robot state, and both cameras without the wrench:
#             "wrist_cam,wrench,robot scene_cam,wrench,robot wrist_cam,scene_cam,wrench,robot wrist_cam,scene_cam,robot"
#   SEEDS     default "0"
#   GPUS      CUDA device ids handed out round robin (default "0"; GPUS="" = CPU, --device cpu, cores split over JOBS)
#   JOBS      runs at a time (default: one per GPU, or 1 on CPU)
# A run whose $RUN_ROOT/dp/<name>/latest.pt exists is skipped unless FORCE=1 (resume it with train_dp.sh --resume).
set -euo pipefail
source "$(dirname "$0")/env.sh"
DATASETS="${DATASETS:-static znoise schedule}"
WRISTS="${WRISTS:-stiff medium z}"
OBS_SETS="${OBS_SETS:-wrist_cam,wrench,robot scene_cam,wrench,robot wrist_cam,scene_cam,wrench,robot wrist_cam,scene_cam,robot}"
SEEDS="${SEEDS:-0}"
GPUS="${GPUS-0}"
read -r -a GPU_LIST <<< "$GPUS"
JOBS="${JOBS:-$(( ${#GPU_LIST[@]} > 0 ? ${#GPU_LIST[@]} : 1 ))}"
HERE="$(dirname "$0")"

i=0
for d in $DATASETS; do for w in $WRISTS; do for o in $OBS_SETS; do for s in $SEEDS; do
  tag="$(dataset_tag "$(dataset_dir "$d")")"
  name="dp_${tag}_${w}_${o//,/-}_s${s}"
  if [ -f "$RUN_ROOT/dp/$name/latest.pt" ] && [ -z "${FORCE:-}" ]; then echo "skip $name (exists)"; continue; fi
  if [ ${#GPU_LIST[@]} -gt 0 ]; then dev=(--device cuda); gpu="${GPU_LIST[$(( i % ${#GPU_LIST[@]} ))]}"; else dev=(--device cpu); gpu=""; fi
  echo "run $name ${gpu:+(gpu $gpu)}"
  if [ -z "${DRY:-}" ]; then
    while [ "$(jobs -rp | wc -l)" -ge "$JOBS" ]; do sleep 5; done
    mkdir -p "$RUN_ROOT/dp/$name"
    # CPU runs share the cores: OMP_NUM_THREADS = cores / JOBS
    [ -z "$gpu" ] && export OMP_NUM_THREADS="${OMP_NUM_THREADS:-$(( $(getconf _NPROCESSORS_ONLN) / JOBS > 0 ? $(getconf _NPROCESSORS_ONLN) / JOBS : 1 ))}"
    CUDA_VISIBLE_DEVICES="$gpu" DATASET="$d" WRIST="$w" OBS="$o" SEED="$s" NAME="$name" \
      bash "$HERE/train_dp.sh" "${dev[@]}" "$@" > "$RUN_ROOT/dp/$name/sweep.log" 2>&1 &
  fi
  i=$(( i + 1 ))
done; done; done; done
wait
echo "sweep done: $i runs"
