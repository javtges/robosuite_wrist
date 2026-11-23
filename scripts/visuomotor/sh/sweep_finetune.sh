#!/usr/bin/env bash
# The fine-tuning ablation: every BASE x STRUCTURE x NOISE x SEED through finetune.sh, JOBS at a time, each on its
# own port pair:
#
#   BASES="dp_schedule_z_wrist_cam-scene_cam-wrench-robot_s0" bash visuomotor/sh/sweep_finetune.sh       # 4 x 4 runs
#   BASES=oracle STRUCTURES="wrist_z wrist_6d" NOISES="none gaussian:noise_cov_source=force_budget" bash visuomotor/sh/sweep_finetune.sh
#   JOBS=12 bash visuomotor/sh/sweep_finetune.sh --max_steps 200000      # extra flags -> every run
#   DRY=1 bash visuomotor/sh/sweep_finetune.sh                           # print the runs, start nothing
#
#   BASES       space-separated: oracle and/or diffusion-policy run names / checkpoint paths (default oracle)
#   STRUCTURES  default "wrist_z wrist_6d arm_wrist_z arm_wrist_6d"
#   NOISES      default "none random gaussian:noise_cov_source=force_budget ergodic" (finetune.sh's NOISE syntax)
#   SEEDS       default "0"
#   JOBS        runs at a time (default 4); each run is a learner and an actor process
#   PORT_BASE   first port (default 5600); run i uses PORT_BASE + 2i and the next
set -euo pipefail
source "$(dirname "$0")/env.sh"
BASES="${BASES:-oracle}"
STRUCTURES="${STRUCTURES:-wrist_z wrist_6d arm_wrist_z arm_wrist_6d}"
NOISES="${NOISES:-none random gaussian:noise_cov_source=force_budget ergodic}"
SEEDS="${SEEDS:-0}"
JOBS="${JOBS:-4}"
PORT_BASE="${PORT_BASE:-5600}"
HERE="$(dirname "$0")"

i=0
for b in $BASES; do for st in $STRUCTURES; do for nz in $NOISES; do for s in $SEEDS; do
  port=$(( PORT_BASE + 2 * i ))
  echo "run base $b, structure $st, noise $nz, seed $s (port $port)"
  if [ -z "${DRY:-}" ]; then
    while [ "$(jobs -rp | wc -l)" -ge "$JOBS" ]; do sleep 10; done
    BASE="$b" STRUCTURE="$st" NOISE="$nz" SEED="$s" PORT="$port" bash "$HERE/finetune.sh" "$@" &
  fi
  i=$(( i + 1 ))
done; done; done; done
wait
echo "sweep done: $i runs"
