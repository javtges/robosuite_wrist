#!/usr/bin/env bash
# One-time setup of a machine for the visuomotor pipeline (scripts/visuomotor/sh/gen_datasets.sh and friends).
# Run from anywhere:   bash SETUP.sh
# Idempotent: re-running reuses an existing robosuite_env and only reinstalls what changed.
#
#   SKIP_APT=1      don't check / install the apt packages (no sudo on this machine)
#   RECREATE=1      delete robosuite_env and build it from scratch
#   MUJOCO_GL=...   force the render backend to test (default: egl on Linux, falling back to osmesa)
#
# The runtime variables (MUJOCO_GL, DATA_ROOT, RUN_ROOT, WANDB_*, PYTHONUNBUFFERED, ...) are set by
# scripts/visuomotor/sh/env.sh every time a visuomotor .sh file runs, so nothing has to go in your shell profile.
# If this script reports that only osmesa works, run the pipeline with MUJOCO_GL=osmesa in front.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO"
VENV="$REPO/robosuite_env"
PY="$VENV/bin/python"
OS="$(uname)"
say() { printf '\n==> %s\n' "$*"; }

# ---- 1. System packages (Linux): compiler + headers for evdev's source build, EGL / OSMesa for offscreen rendering
if [ "$OS" = "Linux" ] && [ -z "${SKIP_APT:-}" ] && command -v apt-get > /dev/null; then
  say "Checking system packages"
  pkgs=(build-essential python3-dev linux-libc-dev libegl1 libgl1 libosmesa6 libglfw3)
  missing=()
  for p in "${pkgs[@]}"; do dpkg -s "$p" > /dev/null 2>&1 || missing+=("$p"); done
  if [ ${#missing[@]} -gt 0 ]; then
    echo "installing: ${missing[*]}"
    sudo apt-get update && sudo apt-get install -y "${missing[@]}"
  else
    echo "all present"
  fi
fi

# ---- 2. uv
if ! command -v uv > /dev/null; then
  say "Installing uv"
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi
echo "uv: $(uv --version)"

# ---- 3. robosuite_env (Python 3.14; uv downloads the interpreter if the system lacks it)
if [ -n "${RECREATE:-}" ] && [ -d "$VENV" ]; then
  say "Removing $VENV"
  rm -rf "$VENV"
fi
if [ ! -x "$PY" ]; then
  say "Creating robosuite_env (Python 3.14)"
  uv venv "$VENV" --python 3.14
fi
say "Installing packages"
if [ "$OS" = "Darwin" ]; then req=requirements_macos.txt; else req=requirements.txt; fi
# Run from the repo root so "-e ./robosuite" resolves to the vendored copy
uv pip install --python "$PY" -r "$req"

# ---- 4. Checks. Run from scripts/, never the repo root (there, "import robosuite" picks up the ./robosuite folder)
cd "$REPO/scripts"
say "Checking imports"
"$PY" - << 'EOF'
import h5py, mujoco, numpy, robosuite, scipy, torch
print(f"robosuite {robosuite.__version__} ({robosuite.__file__})")
print(f"mujoco {mujoco.__version__}, numpy {numpy.__version__}, torch {torch.__version__}, "
      f"cuda available: {torch.cuda.is_available()}")
EOF

render_ok() {
  MUJOCO_GL="$1" PYOPENGL_PLATFORM="$1" "$PY" - << 'EOF' > /dev/null 2>&1
import mujoco
m = mujoco.MjModel.from_xml_string("<mujoco><worldbody><light pos='0 0 1'/><geom size='.1'/></worldbody></mujoco>")
d = mujoco.MjData(m)
mujoco.mj_forward(m, d)
with mujoco.Renderer(m, 64, 64) as r:
    r.update_scene(d)
    assert r.render().any()
EOF
}
GL=""
if [ "$OS" = "Linux" ]; then
  say "Checking offscreen rendering"
  for backend in ${MUJOCO_GL:-egl osmesa}; do
    if render_ok "$backend"; then GL="$backend"; echo "MUJOCO_GL=$backend works"; break; fi
    echo "MUJOCO_GL=$backend failed"
  done
  if [ -z "$GL" ]; then
    echo "No offscreen backend works; the cameras in gen_demos can't render. Install a GPU driver with EGL, or libosmesa6." >&2
    exit 1
  fi
fi
export MUJOCO_GL="${GL:-${MUJOCO_GL:-}}"

say "Checking the KeyDoor environment"
"$PY" - << 'EOF'
import os, sys
sys.path.insert(0, os.getcwd())
import robosuite  # noqa: F401
from visuomotor import gen_demos  # noqa: F401  (imports the whole demo stack)
print("visuomotor.gen_demos imports OK")
EOF

# ---- 5. Output directories (git-ignored)
mkdir -p "$REPO/data/visuomotor" "$REPO/runs/visuomotor"

say "Done"
cat << EOF
robosuite_env: $VENV
Generate the datasets (from scripts/):
  cd $REPO/scripts && bash visuomotor/sh/gen_datasets.sh
EOF
[ "$OS" = "Linux" ] && [ "$GL" != "egl" ] && echo "EGL failed here, so prefix it with MUJOCO_GL=$GL"
echo "Training runs (train_dp.sh etc.) log to wandb: run 'robosuite_env/bin/wandb login' once, or set WANDB_MODE=offline."
