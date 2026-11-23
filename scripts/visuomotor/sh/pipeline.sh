#!/usr/bin/env bash
# The whole study, stage by stage, each stage one of the composable scripts next to this file (their variables pass
# through: MODES, EPISODES, SHARDS, DATASETS, WRISTS, OBS_SETS, SEEDS, GPUS, JOBS, BASES, STRUCTURES, NOISES, ...):
#
#   STAGES="gen" bash visuomotor/sh/pipeline.sh                  # 1. the three datasets
#   STAGES="dp eval" bash visuomotor/sh/pipeline.sh               # 2. the diffusion-policy grid, 3. sim eval of every run
#   STAGES="finetune" BASES="dp_schedule_z_wrist_cam-scene_cam-wrench-robot_s0" bash visuomotor/sh/pipeline.sh
#   STAGES="gen dp eval finetune" bash visuomotor/sh/pipeline.sh  # everything (finetune on FINETUNE_BASES, below)
#
#   STAGES          any of gen dp eval finetune, in order (default "gen dp eval")
#   EVAL_RUNS       glob of dp runs to evaluate (default 'dp_*')
#   FINETUNE_BASES  bases for the finetune stage when BASES is unset (default oracle)
set -euo pipefail
HERE="$(dirname "$0")"
STAGES="${STAGES:-gen dp eval}"
for stage in $STAGES; do
  echo "==== $stage ===="
  case "$stage" in
    gen) bash "$HERE/gen_datasets.sh" ;;
    dp) bash "$HERE/sweep_dp.sh" ;;
    eval) bash "$HERE/eval_dp.sh" "${EVAL_RUNS:-dp_*}" ;;
    finetune) BASES="${BASES:-${FINETUNE_BASES:-oracle}}" bash "$HERE/sweep_finetune.sh" ;;
    *) echo "unknown stage $stage (gen dp eval finetune)" >&2; exit 1 ;;
  esac
done
