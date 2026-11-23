# Visuomotor door policies: demos, diffusion policies, wrist fine-tuning

Generate the demo datasets once, with every data source recorded. Then reuse them for every ablation: which observations
the diffusion policy gets, what the wrist does when the policy runs, and how SERL fine-tunes the wrist on top of it.
Every training run logs to wandb.

```
gen_datasets.sh ──> data/visuomotor/{static,znoise,schedule}/shard*_lock*.hdf5
                         │
sweep_dp.sh / train_dp.sh ──> runs/visuomotor/dp/<run>/          diffusion policy (robosuite_env, torch)
                         │            │
                  eval_dp.sh          └──> sweep_finetune.sh / finetune.sh ──> runs/visuomotor/residual/<run>/
                                                SERL residual on the frozen policy (SERL venv, jax + torch)
pipeline.sh chains the stages.
```

Task: KeyDoor (`robosuite/robosuite/environments/manipulation/key_door.py`). Insert the key, turn it, push the door
open. FR3 arm, compliant 6-DoF HSA wrist, 20 Hz control. Run every script from `scripts/`.

## Files

| Path | What it does |
|---|---|
| `common.py` | What the parts share: dataset schema, observation groups, action layout, wrist modes, env factory |
| `gen_demos.py` | One dataset shard from the scripted expert, all sources recorded |
| `dp/` | Diffusion policy: `dataset.py`, `model.py` (ResNet-18 + 1-D conditional UNet, own DDPM/DDIM), `normalizer.py`, `train_dp.py`, `eval_dp.py`, `policy.py` (`DiffusionPolicyRunner`, used by eval and as the residual's base) |
| `residual/` | SERL fine-tuning: `door_residual_env.py` (gym `DoorResidual-v0`), `exploration.py` (wrist samplers), `train_residual.py` (async learner/actor), `compat.py` (SciPy 1.11 shim for the SERL venv) |
| `sh/` | The composable run scripts (below). `env.sh` holds the shared settings |

## Setup on the server

```
uv venv robosuite_env --python 3.14 && uv pip install --python robosuite_env/bin/python -r requirements.txt
git clone https://github.com/rail-berkeley/serl.git ../serl && git -C ../serl checkout 1fa2af7
uv venv serl_env --python 3.10 && uv pip install --python serl_env/bin/python -r scripts/key_insertion/requirements_serl.txt
wandb login
```

`requirements.txt` now has torch, torchvision and wandb. `requirements_serl.txt` has torch (CPU is enough for the base
policy). On a CUDA server, install the CUDA builds of torch, and of jax for SERL (see the comment at the top of
`requirements_serl.txt`). On Linux, `env.sh` sets `MUJOCO_GL=egl` for headless rendering; use `MUJOCO_GL=osmesa` if
EGL is not available.

Settings in `sh/env.sh`, all of which can be overridden from the environment: `DATA_ROOT` (default `data/visuomotor`),
`RUN_ROOT` (default `runs/visuomotor`), `PY`, `PY_SERL`, `WANDB_PROJECT` (default `hsa-door-visuomotor`),
`WANDB_ENTITY`, `WANDB_MODE` (`online` / `offline` / `disabled`).

## 1. Datasets

```
bash visuomotor/sh/gen_datasets.sh                                   # static, znoise, schedule: 100 episodes each
MODES=schedule EPISODES=300 SHARDS=48 bash visuomotor/sh/gen_datasets.sh
MODES=static STATIC_Z=-1 bash visuomotor/sh/gen_datasets.sh          # a static dataset at the stiff end
```

The expert is a pose-waypoint state machine (approach → align → insert → hold → turn → push). It is force-blind and
aims with an injected perception error (±1.5 mm, ±3°). Episodes alternate between the normal lock (0.15 N·m) and the
stiff lock (0.4 N·m). The door pose is randomized (±2 cm, ±5°). With the same `SEED`, all three datasets see the same
door poses and aiming errors.

| `--wrist-mode` | What the wrist does during the demos |
|---|---|
| `static` | Holds one z all episode (`STATIC_Z`, default 0 = home) |
| `znoise` | z-axis exploration. It switches to a random z in [−1, 1], holds it 0.3–1.5 s, then switches again, with no task logic. Expect more failures (stiff z while inserting) and longer episodes. |
| `schedule` | The expert's phase schedule (`soft_insert_home_turn`): soft (z = +1) to insert and seat, home (z = 0) to turn and push |

In every mode the wrist moves only along z, which is its stiffness. The other five axes deflect passively. Every z
switch runs at 1 z/s, and the arm compensates for the plate's travel.

**Recorded every step** (`common.py` has the full schema):
- the wrist camera (`gripper0_right_ee_cam_image`) and the scene camera (`doorview_image`: on the door frame, looking
  across the door face at the keyhole), 120×160 uint8, upright, gzip;
- the wrist's leg-sensed wrench (`wrist_wrench`);
- the wrist command in force (`wrist_cmd`) and the expert's phase;
- every KeyDoor observation, and the full 12-dim action: TCP pose plus the wrist's 6 normalized axes.

`--successes-only` drops failed episodes; by default they are kept with `success=False`.

**Cost:** about 6 s of CPU per 300-step episode, one core per shard. On 100 cores, 3 × 100 episodes take a few minutes.
Disk is about 9 MB per episode for both cameras. znoise episodes are about 2× longer.

## 2. Diffusion policies

```
DATASET=schedule WRIST=z OBS=wrist_cam,scene_cam,wrench,robot bash visuomotor/sh/train_dp.sh
bash visuomotor/sh/sweep_dp.sh                                       # the full grid, 36 runs
GPUS="0 1 2 3" JOBS=8 DATASETS=schedule bash visuomotor/sh/sweep_dp.sh --epochs 500
GPUS="" JOBS=4 bash visuomotor/sh/sweep_dp.sh                         # CPU-only server: 4 runs, cores split between them
DRY=1 bash visuomotor/sh/sweep_dp.sh                                 # list the runs only
```

| Variable | Choices |
|---|---|
| `DATASET` / `DATASETS` | `static`, `znoise`, `schedule`, or a path |
| `WRIST` / `WRISTS` | `stiff`: the wrist static at z = −1. `medium`: static at z = 0. `z`: the policy outputs the wrist's z as a 10th action, rate-limited to 1 z/s when it runs |
| `OBS` / `OBS_SETS` | Any comma-separated subset of `wrist_cam`, `scene_cam`, `wrench` (leg-sensed wrist F/T), `robot` (TCP pose + the wrist command in force: the robot's state and last action) |

With `WRIST=stiff` or `medium`, the `robot` group leaves out `wrist_cmd`. At deployment it is constant, so a static
policy trained on znoise or schedule data would otherwise see z values it never sees at test time.

**Model:** Chi et al.'s CNN diffusion policy.
- One ResNet-18 per camera (GroupNorm, spatial softmax, random 90% crop in training).
- The low-dim observations are concatenated, over 2 observation steps.
- A 1-D conditional UNet predicts 16 actions, of which 8 are executed (0.4 s), using DDIM with 16 steps.
- The action is TCP position plus 6-D rotation, plus wrist z for `WRIST=z`.

The model has about 92 M parameters with both cameras (about 370 MB per checkpoint).

`train_dp.py --help` lists every flag. Useful ones:
- `--epochs`, `--batch-size`, `--lr`
- `--successes-only`, `--max-demos`
- `--no-cache-images`: about 300 episodes with both cameras need 15–20 GB of RAM if cached
- `--pretrained-vision`: needs internet or cached ImageNet weights
- `--resume`, `--dry-run`

In-process sim eval runs every `EVAL_EVERY` epochs (default 50).

**wandb:** run `dp_<dataset>_<wrist>_<obs>_s<seed>`, group `dp_<dataset>`, tags `dataset:`, `wrist:`, `obs:`.

**Eval:**
```
bash visuomotor/sh/eval_dp.sh dp_schedule_z_wrist_cam-scene_cam-wrench-robot_s0 --video 4
EPISODES=100 WORKERS=32 bash visuomotor/sh/eval_dp.sh 'dp_*'
```
Each eval reports, overall and per lock: success rate, time to success, peak flange force, seated and unlocked rates,
force aborts, and mean wrist z. Results go to wandb run `eval_<run>` (group `eval_<dataset>`) and to
`<run>/eval_latest.json`.

## 3. SERL residual fine-tuning

The diffusion policy stays frozen and acts as the base. SAC learns a bounded residual on top of it, at 20 Hz, from
low-dim observations. The task reward is success, minus a force penalty, with an over-force abort. `BASE=oracle` uses
the scripted expert as the base instead, which is useful before any policy is trained.

```
BASE=dp_schedule_z_wrist_cam-scene_cam-wrench-robot_s0 STRUCTURE=wrist_z NOISE=random bash visuomotor/sh/finetune.sh
BASES="dp_schedule_z_wrist_cam-scene_cam-wrench-robot_s0" bash visuomotor/sh/sweep_finetune.sh     # 4 structures x 4 noises
```

| `STRUCTURE` | Agent action | What it moves |
|---|---|---|
| `wrist_z` | 1 | The wrist's z on top of the base: `z_base + s·a` |
| `wrist_6d` | 6 | All six wrist axes, clipped to the stroke |
| `arm_wrist_z` | 7 | Arm (±3 mm, ±2° on the base's TCP target) plus wrist z |
| `arm_wrist_6d` | 12 | Arm plus all six wrist axes |

Scales: `--wrist_scale` (1 or 6 values), `--pos_scale`, `--rot_scale`. `--wrist_rate` (default 1 z/s, the expert's rate)
limits how fast the executed wrist command changes. Without it, an untrained agent jumps the plate and the force limit
ends the episode at step 1.

**Observation (37 dims):** TCP pose relative to the base's target, leg-sensed wrench, wrist deflection, wrist command,
and the base's action. `--privileged_obs` adds the key-in-lock pose and the lock and door angles. `--phase_obs` adds a
task-phase one-hot.

**Wrist-side sampling during training** (`NOISE`; off at evaluation). Sampler flags go after a colon:
`NOISE="name:flag=value,flag=value"`.

| `NOISE` | What it samples | Main flags |
|---|---|---|
| `none` | Only SAC's own policy noise | |
| `random` | i.i.d. noise each step on the active wrist axes | `noise_std` (1 or 6 values), `noise_dist=gaussian\|uniform` |
| `gaussian` | A structured 6-D Gaussian N(0, Σ). Σ comes from the flags (`noise_std` plus `noise_corr` / `noise_cov` .npy files), or from `noise_cov_source=force_budget`: CLAUDE.md's force-budget water-filling, Σ ∝ R⁻¹ with R = KᵀW_fK and K the wrist's stiffness at its current command | `noise_f_max`, `noise_tau_max`, `noise_budget`, `noise_caps`, `noise_cov_every`; `noise_tau` > 0 gives Ornstein–Uhlenbeck noise with the same covariance |
| `ergodic` | Ergodic coverage (spectral multiscale coverage) of the active wrist axes, toward a target density built online from the observed wrist forces. A first version, meant to be reworked | `ergodic_axes`, `ergodic_modes`, `ergodic_umax`, `ergodic_target=low_force\|force_variance\|uniform`, `ergodic_f_scale` |

The replay buffer stores the executed residual (the agent's action plus the sampled perturbation, clipped), so the
off-policy update sees what actually ran. `python -m visuomotor.residual.exploration` runs the samplers' self-checks.

Other flags (`train_residual.py --help`):
- `--demo_rollouts N`: zero-residual base episodes used as a 50/50 RLPD demo buffer;
- `--max_steps`, `--actor_hz`;
- `--force_penalty`, `--force_limit`.

`FRICTIONS` (default `0.15,0.4`) sets the locks, cycled per episode.

**wandb:** run `<structure>-<noise tag>-<base>-s<seed>`, group the same without the seed.

**Output:** `runs/visuomotor/residual/<run>/` (`eval.csv`, checkpoints, `final/`) and logs in `_logs/`.

## Whole pipeline

```
STAGES="gen dp eval" bash visuomotor/sh/pipeline.sh
STAGES=finetune BASES="dp_schedule_z_wrist_cam-scene_cam-wrench-robot_s0 oracle" bash visuomotor/sh/pipeline.sh
```

## Known limits

- **The ergodic sampler is a first version.** Its target density (low force by default) and its control law are
  placeholders to rework.
- **Wrist commands that change abruptly give large physics-step force spikes:** about 28 N per mm of plate travel at
  the home wrist. White `random` noise with a large `noise_std` hits the force limit often. Prefer force-budget or OU
  noise, small scales, or `--force_measure step`.
- **A `wrist_6d` residual on a diffusion-policy base** feeds the policy wrist commands (x, y, rotations) that it never
  saw in its data. The `robot` observation group includes them.
- **The residual env's `skip_approach`** runs the base with zero residual until the key is within 25 mm of the lock.
  A poor base can spend up to 400 steps there.
- **The SERL venv's SciPy 1.11** lacks `Rotation ** int`. `residual/compat.py` patches the one function that uses it;
  robosuite is unchanged. Upgrading SciPy to 1.12 or later makes the shim unnecessary.
- **The wrist camera and the measured key mount** are fitted by `ee_camera.enable()` before any env is built. Every
  script does this from the dataset or checkpoint's `ee_camera` record. A plain `suite.make("KeyDoor")` has no wrist
  camera.
