#!/usr/bin/env bash
# Shared settings for the visuomotor .sh files. Sourced, not run. Every variable can be overridden from the
# environment, e.g.  DATA_ROOT=/scratch/vm WANDB_ENTITY=mylab bash visuomotor/sh/gen_datasets.sh
#
#   REPO          repo root (default: derived from this file)
#   PY            robosuite_env python (demos, diffusion-policy training and eval)
#   PY_SERL       SERL venv python (residual fine-tuning; needs torch for a diffusion-policy base)
#   DATA_ROOT     datasets: $DATA_ROOT/<static|znoise|schedule>/*.hdf5
#   RUN_ROOT      runs: $RUN_ROOT/dp/<name>/ and $RUN_ROOT/residual/<name>/
#   WANDB_PROJECT wandb project for every run (default hsa-door-visuomotor)
#   WANDB_ENTITY  wandb entity (team/user); unset = your default
#   WANDB_MODE    online | offline | disabled (default online)

VM_SH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="${REPO:-$(cd "$VM_SH/../../.." && pwd)}"
SCRIPTS="$REPO/scripts"
PY="${PY:-$REPO/robosuite_env/bin/python}"
PY_SERL="${PY_SERL:-$REPO/serl_env/bin/python}"
[ -x "$PY_SERL" ] || PY_SERL="$HOME/serl_env/bin/python"
DATA_ROOT="${DATA_ROOT:-$REPO/data/visuomotor}"
RUN_ROOT="${RUN_ROOT:-$REPO/runs/visuomotor}"
export WANDB_PROJECT="${WANDB_PROJECT:-hsa-door-visuomotor}"
export WANDB_MODE="${WANDB_MODE:-online}"
[ -n "${WANDB_ENTITY:-}" ] && export WANDB_ENTITY
export PYTHONUNBUFFERED=1
# Offscreen rendering for the cameras: EGL on a headless Linux server (override with MUJOCO_GL=osmesa if no GPU
# driver); macOS uses its own default
if [ "$(uname)" = "Linux" ]; then export MUJOCO_GL="${MUJOCO_GL:-egl}"; fi

cd "$SCRIPTS"

# dataset_dir <static|znoise|schedule|path>: a dataset name under DATA_ROOT, or a path as is
dataset_dir() { case "$1" in static | znoise | schedule) echo "$DATA_ROOT/$1" ;; *) echo "$1" ;; esac; }

# run_name: a short tag from the dataset, e.g. dataset_tag /x/y/schedule -> schedule
dataset_tag() { basename "$1"; }
