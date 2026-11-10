#!/usr/bin/env bash
# Runs one SERL training (learner + actor) on KeyInsertionWrist-v0 with the oracle demos, from scripts/:
#   bash key_insertion/launch_serl.sh <stiff|soft|variable> <seed> [port] [extra flags for both...]
# Logs go to key_insertion/runs/<mode>_seed<seed>/ (eval.csv, learner.log, actor.log, checkpoints/).
# Concurrent runs need different ports (each uses port and port + 1).
set -euo pipefail
MODE=$1
SEED=$2
PORT=${3:-5488}
shift $(( $# < 3 ? $# : 3 ))
cd "$(dirname "$0")/.."
DIR=key_insertion/runs/${MODE}_seed${SEED}
mkdir -p "$DIR"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=.1
export PYTHONUNBUFFERED=1  # logs show progress as it happens
COMMON=(--mode "$MODE" --seed "$SEED" --port "$PORT" "$@")
python -m key_insertion.train_serl --learner --demo_path "key_insertion/demos/${MODE}_20.pkl" "${COMMON[@]}" > "$DIR/learner.log" 2>&1 &
LEARNER=$!
trap 'kill $LEARNER 2>/dev/null || true' EXIT
sleep 5
python -m key_insertion.train_serl --actor "${COMMON[@]}" > "$DIR/actor.log" 2>&1
