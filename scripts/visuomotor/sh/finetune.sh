#!/usr/bin/env bash
# One SERL residual fine-tuning run (visuomotor/residual/train_residual.py: learner + actor) on top of a frozen base,
# logged to wandb:
#
#   BASE=dp_schedule_z_wrist_cam-scene_cam-wrench-robot_s0 STRUCTURE=wrist_z NOISE=random bash visuomotor/sh/finetune.sh
#   BASE=oracle STRUCTURE=arm_wrist_6d NOISE="gaussian:noise_cov_source=force_budget,noise_tau=0.2" bash visuomotor/sh/finetune.sh
#   BASE=oracle NOISE="ergodic:ergodic_axes=z,ergodic_umax=0.05" bash visuomotor/sh/finetune.sh --max_steps 200000
#
#   BASE        oracle (the scripted expert; BASE_SCHEDULE / BASE_STATIC_Z set its wrist) | a diffusion-policy run name
#               under $RUN_ROOT/dp (its latest.pt) | a checkpoint path. Default oracle
#   STRUCTURE   wrist_z | wrist_6d | arm_wrist_z | arm_wrist_6d (what the residual moves; default wrist_z)
#   NOISE       wrist-side sampling during training: none | random | gaussian | ergodic, optionally with sampler flags
#               as name:flag=value,flag=value (flags of train_residual.py without the leading --). Default none
#   SEED        default 0
#   PORT        agentlace port, uses PORT and PORT+1 (default 5588); concurrent runs need different ports
#   FRICTIONS   lock frictions cycled per episode, comma-separated (default 0.15,0.4)
#   BASE_DEVICE torch device of a diffusion-policy base (default cpu)
# Extra flags go to both learner and actor (e.g. --wrist_scale 0.3 --demo_rollouts 20 --max_steps 300000).
# Output: $RUN_ROOT/residual/<run>/ (eval.csv, checkpoints, final/), logs in $RUN_ROOT/residual/_logs/.
# wandb (unless WANDB_MODE=disabled): run <structure>-<noise>-<base>-s<seed>, group the same without the seed.
set -euo pipefail
source "$(dirname "$0")/env.sh"
BASE="${BASE:-oracle}"
STRUCTURE="${STRUCTURE:-wrist_z}"
NOISE="${NOISE:-none}"
SEED="${SEED:-0}"
PORT="${PORT:-5588}"
FRICTIONS="${FRICTIONS:-0.15,0.4}"
BASE_DEVICE="${BASE_DEVICE:-cpu}"

# NOISE "name:k=v,k=v" -> (--wrist_noise name --k v --k v)
noise_flags() {
  local spec="$1" name="${1%%:*}" kv
  NOISE_ARGS=(--wrist_noise "$name")
  if [[ "$spec" == *:* ]]; then
    IFS=',' read -r -a kv <<< "${spec#*:}"
    for p in "${kv[@]}"; do NOISE_ARGS+=("--${p%%=*}=${p#*=}"); done
  fi
}
noise_flags "$NOISE"

if [ "$BASE" = "oracle" ]; then
  BASE_ARGS=(--base oracle --base_schedule "${BASE_SCHEDULE:-soft_insert_home_turn}")
  [ -n "${BASE_STATIC_Z:-}" ] && BASE_ARGS+=(--base_static_z "$BASE_STATIC_Z")
  BASE_TAG="oracle"
else
  if [ -f "$BASE" ]; then CKPT="$BASE"; else CKPT="$RUN_ROOT/dp/$BASE/latest.pt"; fi
  [ -f "$CKPT" ] || { echo "no base checkpoint: $CKPT" >&2; exit 1; }
  BASE_ARGS=(--base dp --base_ckpt "$CKPT" --base_device "$BASE_DEVICE")
  BASE_TAG="$(basename "$(dirname "$CKPT")")"
fi
WANDB_FLAG=--nodebug
[ "$WANDB_MODE" = "disabled" ] && WANDB_FLAG=--debug

LOGS="$RUN_ROOT/residual/_logs"
mkdir -p "$LOGS"
LOG="$LOGS/${STRUCTURE}_${NOISE//[:,=]/_}_${BASE_TAG}_s${SEED}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=.1
COMMON=(--structure "$STRUCTURE" "${NOISE_ARGS[@]}" "${BASE_ARGS[@]}" --cylinder_friction "$FRICTIONS" --seed "$SEED"
        --port "$PORT" --run_root "$RUN_ROOT/residual" "$WANDB_FLAG" "$@")
echo "[finetune] structure $STRUCTURE, noise $NOISE, base $BASE_TAG, seed $SEED, port $PORT -> $LOG.{learner,actor}.log"
"$PY_SERL" -m visuomotor.residual.train_residual --learner "${COMMON[@]}" > "$LOG.learner.log" 2>&1 &
LEARNER=$!
trap 'kill $LEARNER 2>/dev/null || true' EXIT
sleep 10
"$PY_SERL" -m visuomotor.residual.train_residual --actor "${COMMON[@]}" > "$LOG.actor.log" 2>&1
