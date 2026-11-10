# Running the Panda + Stewart wrist + ALOHA gripper embodiment

## Setup

```bash
cd /Users/jamesavtges/robosuite_wrist
source robosuite_env/bin/activate      # robosuite (editable, ./robosuite) + robosuite_models are installed here
```

From a fresh clone (robosuite is vendored in `./robosuite`; the venv isn't in git):

```bash
git clone <this repo> robosuite_wrist
cd robosuite_wrist
uv venv robosuite_env --python 3.14
uv pip install --python robosuite_env/bin/python -r requirements_macos.txt   # macOS
uv pip install --python robosuite_env/bin/python -r requirements.txt         # Linux
```

Both files install `./robosuite` in editable mode, and the versions are exact. `requirements.txt` holds the packages
for every platform. `requirements_macos.txt` includes it and adds the `pyobjc-*` packages, which exist only on macOS.
On Linux, `requirements.txt` also installs `evdev` and `python-xlib` (pynput's Linux backends) as minimum versions,
not pins. `evdev` may compile from source, which needs a C compiler and the Python and kernel headers (on Debian or
Ubuntu: `sudo apt install build-essential python3-dev linux-libc-dev`). The venv is platform-specific: build a new one
on each machine, and don't copy `robosuite_env/` between macOS and Linux.

`./robosuite` is a copy of ARISE-Initiative/robosuite at commit `5ce6643f` (master, "Merge pull request #816"), with
its git history removed. The robot / wrist changes are committed directly in this repo, under `robosuite/robosuite/`
(see "Where the code lives" below). To pull in a later upstream robosuite, diff that commit against the new one and
apply the result here. Demos (`demos/`, `*.hdf5`) and `screenshots/` are git-ignored.

Run Python from a script file, or from any directory other than the project root. If you run `python -c` / `python -`
from `/Users/jamesavtges/robosuite_wrist`, Python imports the `robosuite/` repo folder as a namespace package, and you get
`module 'robosuite' has no attribute 'make'`.

## Use it in any task

The robot is named `PandaStewartAloha`. Every standard task stays the same; only `robots=` changes:

```python
import numpy as np
import robosuite as suite

env = suite.make(
    "Lift",                         # also Stack, PickPlaceCan, NutAssemblySquare, Door, ...
    robots="PandaStewartAloha",
    has_renderer=True,              # set False for headless use
    use_camera_obs=False,
)
obs = env.reset()
for _ in range(200):
    obs, reward, done, info = env.step(np.zeros(env.action_dim))
    env.render()
```

### Action space (13-dim, default OSC_POSE arm controller)

| index | meaning |
|---|---|
| 0:6  | arm: OSC pose of the flange-fixed tcp (see below), delta by default. `scripts/teleop_wrist.py` switches it to absolute targets in the robot base frame. |
| 6:12 | wrist: absolute platform pose target relative to home, normalized to [-1, 1]: x, y (±10 mm), z (±20 mm), roll, pitch (±10°), yaw (±10°). 0 means home. Targets the legs can't reach are pulled back toward home along the same direction (only possible if you set `leg_stroke` smaller than the default). |
| 12   | gripper: -1 opens, +1 closes (ALOHA gripper, incremental) |

The gripper is last, as it is for the stock Panda.

### The arm tracks the flange, not the fingertips

The arm's OSC controller tracks `gripper0_right_wrist_tcp`, a site fixed to the wrist base, which is bolted to the arm
flange. It sits where the wrist's top plate is at home (like HSA-UMI's `hsa_tcp`), with the grip site's orientation at
home. Wrist motion therefore adds to the arm's motion instead of being cancelled by it. In a test, commanding the wrist
+20 mm in z while the arm held its target moved the tool 20.0 mm, with the tcp moving under 0.1 mm. When the controller
tracked the grip site, the tool moved only 12.6 mm, because the arm pushed back against the wrist.

Arm rotations pivot about the tcp, so the fingertips (14 cm further out) swing when you rotate. To pivot about the
fingertips instead, set `StewartAlohaGripper.tcp = "grip"` before creating the env. The tcp then sits at the grip
site's home position, still fixed to the flange. Robots that aren't built by `WristFixedBaseRobot` keep robosuite's
default and track the grip site.

### Leg-length control variant (13-dim)

```python
env = suite.make("Lift", robots="PandaStewartAloha", gripper_types="StewartAlohaGripperLegControl")
```

Here the 6 wrist actions are leg length offsets from home, normalized by the leg stroke (±1 = ±39 mm). Each
leg can't exceed its travel, but that alone doesn't keep the platform valid: large mixed leg commands drive it
through singular configurations, and in testing it flipped (pitch past 360°). So leg commands are also pulled
back toward home until the resulting platform pose, found by forward kinematics, is inside the same workspace box
as pose control. Within the box, legs and pose are just two coordinate systems for the same motions. This costs
about 4 ms per step.

### Locked-wrist variant (7-dim, same action space as the stock Panda)

```python
env = suite.make("Lift", robots="PandaStewartAloha", gripper_types="StewartAlohaGripperLockedWrist")
```

The wrist stays physically present and is held at home. Use this variant with robosuite's own teleop devices
(`robosuite/scripts/collect_human_demonstrations.py`). They send `[grasp] * gripper_dof`, so with the 13-dim version a
grasp press would also drive the wrist to a corner of its workspace. `scripts/teleop_wrist.py` (below) doesn't have
that problem and drives the arm, the wrist and the gripper. Use this variant as well for any pipeline that assumes
`[pos3, rot3, grip1]` actions, such as MimicGen's default robosuite interface.

### No-gripper variant (12-dim: the wrist with nothing on it)

```python
env = suite.make("Lift", robots="PandaStewartAloha", gripper_types="StewartWristNoGripper")
```

The wrist's top plate carries nothing, so nothing loads or deflects the soft wrist. A massless `NullGripper` keeps the
`grip_site` frame (at the top plate) that robosuite and the scripts expect. The action is the arm's 6 dims plus the
wrist's 6, with no grip dimension. The switch is the `attach_gripper` class attribute on `StewartAlohaGripper`, which
this variant sets to `False`. In the teleop and the nullspace evaluation, pass `--no-gripper`; there, `g` does nothing.

### Observations

- `robot0_eef_pos` / `robot0_eef_quat`: the true `grip_site`, between the ALOHA fingerpads. It includes wrist motion.
- `robot0_tcp_pos` / `robot0_tcp_quat` (world frame, quat xyzw): the flange-fixed tcp that the arm controller tracks.
  It doesn't include wrist motion.
- `robot0_gripper_qpos` / `qvel` (14-dim): `[left_finger, right_finger, wrist x, y, z, roll, pitch, yaw, leg0 ... leg5]`.
  The pose entries are the platform's measured pose relative to home (m / rad), in exactly the units and rotation
  convention of the pose command. The leg entries are leg length offsets from home (m).
- Cameras: `robot0_eye_in_hand` (moved onto the wrist top plate, so it moves with the gripper) and
  `gripper0_right_wrist_cam_right` (the ALOHA's built-in D405 camera).

### RL / IL

- Gym: `robosuite.wrappers.GymWrapper(env)` works as-is (13-dim action space).
- robomimic / MimicGen: the robot and grippers are registered inside robosuite itself (no extra import needed). Any
  `robosuite.make(env_name, robots="PandaStewartAloha", ...)` recreated from dataset env metadata therefore works, and
  so does state replay through `sim.set_state_from_flattened`.

## Teleoperation and demo recording

`scripts/teleop_wrist.py` follows HSA-UMI's SpaceMouse teleop (`ros2_ws/src/hsa_ros2/hsa_ros2/spacemouse_teleop.py`
and `teleop_math.py` in that repo), entirely in simulation:

1. The input is a velocity. Each tick it is integrated into an absolute target pose of the arm's tcp, in the robot
   base frame.
2. The target is clipped to a workspace box.
3. If the arm ends up more than 15 cm from the target, the target snaps back onto the arm (re-seeding).
4. The target is sent to OSC_POSE as an absolute command. The commanded target is also what gets recorded as the action.

The same input can drive the wrist instead. The arm then holds its target, and the wrist adds its motion on top. A
third mode, **nullspace**, also drives the wrist, but the arm moves the opposite way so the gripper stays still in
space. "Nullspace mode" below says how to run it. Its math, implementation and workspace evaluation are in
[`scripts/nullspace/README.md`](scripts/nullspace/README.md).

Run it from `scripts/`. On macOS the viewer needs `mjpython`, which is in the venv. On Linux, use `python` in every
command below (see "Running on Linux").

```bash
cd /Users/jamesavtges/robosuite_wrist/scripts
source ../robosuite_env/bin/activate

mjpython teleop_wrist.py --env Lift --input spacemouse         # the puck
mjpython teleop_wrist.py --env Lift --input keyboard           # 6D keyboard
mjpython teleop_wrist.py --env Lift --input fake_sine          # no device: HSA-UMI's slow circle in x-y
python teleop_wrist.py --input fake_sine --no-render --max-seconds 10 --key-script "0.5:c,8:s,9:q"   # headless
```

Demos go to `../demos/<env>_<time>.hdf5`. Pass `--out FILE` to choose the file; if it already exists, new demos are
appended. `--env` takes any robosuite single-arm task (Lift, Stack, PickPlaceCan, NutAssemblySquare, Door, ...).
`python teleop_wrist.py -h` lists every option: speeds, workspace box, axis map, plan, condition, control rate.

### Controls

Keep the terminal focused. Command keys are read from it, as in HSA-UMI, and the viewer window has its own key
bindings.

| SpaceMouse | |
|---|---|
| push / twist | moves the target at up to 0.20 m/s and 0.15 rad/s (`--max-pos-speed`, `--max-rot-speed`). In wrist mode the limits are 0.04 m/s and 0.35 rad/s (`--wrist-pos-speed`, `--wrist-rot-speed`). |
| left button held | rotate only (translation locked) |
| right button held | z allowed. Without it, z is locked so an accidental push can't drive the tool into the table. |
| both buttons together | next mode: arm → wrist → nullspace → arm |

| Keyboard motion (hold; robot base frame: +x away from the base, +y to its left, +z up) | |
|---|---|
| `i` / `k`, `j` / `l`, `u` / `o` | ±x, ±y, ±z |
| `[` / `]`, ↑ / ↓, ← / → | ± rotation about x, y, z |

The keyboard has a key for every axis, so it doesn't use the button modes. It listens with `pynput`, which needs
macOS permission: System Settings > Privacy & Security > Accessibility and Input Monitoring > enable your terminal app,
then restart the terminal. Without it, the script prints a warning and keyboard motion does nothing.

| Command key | |
|---|---|
| `c` | start recording an episode |
| `s` | stop: save the episode, then reset the scene |
| Backspace | drop the episode in progress (then reset the scene), or the last saved demo if not recording |
| Space | stage + 1 (recorded per step) |
| `n` | skip to the next plan row (`--plan rows.csv`: one row per episode, advances on save, stored with the demo) |
| `p` | print the tcp and wrist poses, their targets and the teleop state, and the flange force |
| `f` | print the flange force every 0.5 s (on / off; `--print-forces` starts it on). The viewer always shows it. |
| `r` | re-seed the arm target on the arm |
| `x` | toggle the wrist stiff (1.0×) / soft (`--soft-scale`, 0.3×) leg stiffness |
| `m` | next mode: arm → wrist → nullspace → arm (same as both puck buttons). Nullspace mode uses `--nullspace-comp plate` (default), `feedback` or `feedforward`. |
| `g` | toggle the gripper open / closed |
| `h` | send the wrist target home |
| `z` | reset the scene (not while recording) |
| `q` | quit (drops an episode in progress) |

The wrist keeps its target while you drive the arm, and the reverse. In wrist mode the input is converted into the
wrist's frame, so a given push moves the tool the same way in both modes, just slower and within the wrist's
±10 mm / ±20 mm / ±10° range.

The workspace box defaults to x 0.25–0.80 m, y ±0.40 m and z up to 0.60 m in the robot base frame. The floor (z-min)
is computed from the task's table or bin height, so the gripper's lowest point just touches it with the wrist at home.
The script prints the box at startup; override any edge with `--x-min` ... `--z-max`.

### Nullspace mode

To try nullspace mode without a SpaceMouse, start with the keyboard or with fake_sine, then press `m` twice in the
terminal (arm → wrist → nullspace):

```bash
cd /Users/jamesavtges/robosuite_wrist/scripts
source ../robosuite_env/bin/activate

mjpython teleop_wrist.py --env Lift --input keyboard    # m twice, then hold i/k, j/l, u/o, [ ], arrow keys
mjpython teleop_wrist.py --env Lift --input fake_sine   # m twice; the wrist traces a slow x-y circle
mjpython teleop_wrist.py --env Lift --input keyboard --nullspace-comp feedback      # compare: no plate feedback
```

`--input` defaults to `spacemouse`, so pass it when no puck is connected. When the mode engages, the script prints
`nullspace: input drives the wrist, the arm cancels it (plate)`.

The default `plate` compensation also closes the loop on the top plate's measured pose (in the sim, the wrist's
passive joints). The soft wrist sags under the gripper's weight by an amount that depends on its pose, so without
this the gripper drifts by several millimetres as the wrist moves. `feedback` and `feedforward` leave that sag out;
they're kept for comparison, and because a real wrist needs a plate pose sensor or model for `plate`. See
[`scripts/nullspace/README.md`](scripts/nullspace/README.md).

- **What moves.** The input drives the wrist, with wrist mode's speeds (`--wrist-pos-speed`, `--wrist-rot-speed`) and
  limits (±10 mm in x and y, ±20 mm in z, ±10°). The arm moves the opposite way, so the gripper stays still in space.
- **`r`** re-seeds the arm target on the arm, as in arm mode. It also re-anchors the held gripper pose at the place
  where the gripper is now.
- **`h`** sends the wrist home. The arm moves to keep the gripper where it is. This is the quickest way to see the
  effect.
- **`p`** prints the tcp and wrist poses. In nullspace mode the wrist pose changes and the arm's tcp moves the
  opposite way, while the gripper stays where it is.
- **fake_sine** pushes the wrist into its ±10 mm x/y limits and holds it there, so the motion stops at the corners
  of the circle. This is expected.
- **Watching it.** The motions are small (10–20 mm, 10°). Zoom the viewer in on the wrist and gripper, or it can look
  as if nothing happens.

The math, the implementation and the workspace evaluation are in
[`scripts/nullspace/README.md`](scripts/nullspace/README.md).

### Testing and evaluating nullspace mode

There are two levels: unit tests (seconds), and a workspace evaluation (minutes) that measures where the mode works.
Run both from `scripts/` with the venv active:

```bash
cd /Users/jamesavtges/robosuite_wrist/scripts && source ../robosuite_env/bin/activate

python -m pytest nullspace/test_nullspace.py -q   # 9 tests, ~10 s

python -m nullspace.evaluate --quick              # smoke test: 80 runs, ~2 min
python -m nullspace.evaluate                      # full study: 544 runs, ~10 min on 10 cores -> nullspace/results/
python -m nullspace.make_figures                  # -> nullspace/figures/*.png, nullspace/results/summary.md
python -m nullspace.make_demo_gif                 # -> nullspace/figures/nullspace_demo.gif

python -m nullspace.evaluate --no-gripper         # the same study with no gripper -> nullspace/results_nogripper/
python -m nullspace.make_figures --no-gripper     # -> nullspace/figures_nogripper/, results_nogripper/summary.md
```

- **The unit tests** check:
  - the frame algebra: the plate stays on the anchor, and feedback inverts feedforward;
  - plate feedback learning a deflection, and its anti-windup;
  - the plate model against MuJoCo;
  - the mode cycle;
  - a closed-loop run of each strategy through the real teleop.
- **The evaluation** drives the real teleop loop with a simulated SpaceMouse.
  - **Cases:** 120 grid points across the arm's workspace, 14 tool orientations at the workspace center, and the
    center point at 20 Hz and 50 Hz.
  - **Strategies:** each case runs with every one: none (wrist mode), feedforward, feedback, plate.
  - **Each run:** pushes every wrist axis to its limits and back, and measures how far the top plate moves.

  `results/summary.md` has the tables. `results/cases.csv` has every metric per case. The README explains the metrics
  and the "usable" criterion. `--workers N` sets the number of parallel processes (default: CPU cores − 1).

### Flange force readout

The bottom-left corner of the viewer shows the wrench that the wrist and gripper apply to the Franka at its flange.
`f` prints the same readout in the terminal every 0.5 s, and `p` prints it once.

| Row | What it is |
|---|---|
| external force / moment | The total below, minus the static weight of the wrist and gripper: contact and inertial loads only. It reads 0 in free space and at rest. Pressing the gripper into the table gives a negative z force, a push into the flange. |
| total force, with weight | Everything the wrist transmits to the arm, including its weight (about 10 N in free space) |
| wrist leg stiffness | The six legs' current spring constants: the gains of MuJoCo's leg actuators. They fall from 100 kN/m (wrist contracted) to 300 N/m (extended) and scale with `x`. |
| wrist deflection | Measured top-plate pose minus its command: how far the load pushes the wrist off its target |

The readout uses the frame of the wrist's base body (the bottom plate), at the flange, with +z pointing from the
flange toward the gripper. Values come from MuJoCo:
- The total is MuJoCo's own interaction force between the wrist base body and the arm's hand body it's mounted on
  (`cfrc_int` after `mj_rnePostConstraint`). It includes the legs' spring and damper forces, the contacts, gravity and
  inertia.
- The weight that is subtracted uses MuJoCo's subtree mass and center of mass.
- `modern_robotics` moves both wrenches to the flange frame (`RpToTrans`, `Adjoint`; F_b = [Ad_T_ab]ᵀ F_a).

`test_teleop.py::test_flange_force_matches_contact` checks the readout against MuJoCo's own numbers:
- In free space, the total equals the subtree's weight and the external wrench is 0.
- With the fingers pushed into the table, the external force equals the summed contact forces on the gripper, to
  within 1 N.

**Stiffness and force.** Springs in series all carry the same force. The wrist's stiffness changes the flange force
only when the wrist takes a real share of an imposed motion, such as the arm pushing the gripper into the table. The
force is then set by the series stiffness, 1 / (1/k_arm + 1/k_wrist). In a test pushing the arm's target 20 mm into
the table:

| Wrist, 15 mm from home | Leg k | Flange force | Wrist deflection | Arm tcp error |
|---|---|---|---|---|
| contracted | 50 kN/m | 101 N | 0.5 mm | 22 mm |
| extended | 0.6 kN/m | 35 N | 9.9 mm | 11 mm |
| contracted, `x` soft (0.3×) | 15 kN/m | 96 N | 1.5 mm | 20 mm |
| extended, `x` soft (0.3×) | 0.2 kN/m | 10 N | 8.4 mm | 2.7 mm |

The arm's operational-space controller acts like a spring of about 4–5 kN/m.
- **Contracted**, the wrist is about 50× stiffer than the arm. The arm absorbs almost the whole push, and the force is
  set by the arm.
- **Extended**, the wrist is softer than the arm. It takes most of the push, and the flange sees much less force.
  With `x` as well, it sees about 1/10 of the contracted force.

(The extended rows' deflections include the gravity sag, which reverses as the table takes the gripper's weight.)

To see it in the viewer:

1. Start with `--floor-margin -0.03`. The negative floor margin lets the arm push the gripper up to 30 mm into the
   table; by default the workspace box stops the arm just as the gripper touches it.
2. In wrist mode, hold `o` to extend the wrist; `u` contracts it.
3. In arm mode, move down to push the gripper into the table (press the puck down with its right button held, or
   hold `o`), and read the external force. Repeat with the wrist contracted, and with `x`.

The stiffness range is set by `leg_stiffness_contracted` and `leg_stiffness_extended` in `StewartPlatformConfig`
(see "Changing the wrist").

### What's recorded

One hdf5 file that robomimic and robosuite's `playback_demonstrations_from_hdf5.py` both read. The layout is
documented in `scripts/demo_writer.py`:
- `data.attrs`: `env_args` (robomimic: env name, robosuite version, every `suite.make` kwarg including the
  absolute-OSC controller config), plus robosuite's `env` / `env_info` and the `StewartPlatformConfig` used.
- `demo_<i>`:
  - `states`
  - `actions` (13-dim: tcp target pos + rotvec in the base frame, wrist pose normalized, gripper)
  - `rewards`, `dones`
  - `obs/*`, `next_obs/*`: every low-dimensional observation, including `robot0_tcp_pos`
  - `teleop/stage`, `teleop/mode`, `teleop/wrist_stiffness_scale`
  - attrs: `model_file`, `success`, `plan_row`, `condition`, and the controller state that a replay must restore

Images aren't stored. robomimic's `dataset_states_to_obs.py` renders them from the states.

The recorded actions are absolute, so a policy trained on them needs the same absolute controller. Rebuilding the env
from `env_args` gives you that. The wrist stiffness toggle (`x`) changes model parameters, not the sim state. The
per-step `teleop/wrist_stiffness_scale` records it, but an env rebuilt for policy rollouts starts stiff (1.0×).

### Check a file

```bash
python validate_demos.py ../demos/Lift_<time>.hdf5                         # structure + replay of every demo
mjpython validate_demos.py ../demos/Lift_<time>.hdf5 --demo demo_0 --render  # watch one
```

The validator checks every key and length, then replays each demo. It rebuilds the env from `env_args`, loads the
demo's model and first state, and restores the gripper's internal command, the arm controller's null-space posture
and the per-step wrist stiffness. It then steps the recorded actions and compares the result against the recorded
states. A demo passes if the gripper stays within 1 mm and the task success matches. In testing the error was
0.000–0.007 mm, including a grasp-and-lift.

Before the null-space posture was restored, the replayed arm drifted 0.01 rad in a joint the tool doesn't depend
on. OSC pulls the redundant 7th DoF toward the posture at scene reset, and that is controller state, not sim state.

### Testing without a SpaceMouse

```bash
cd scripts && python -m pytest test_teleop.py -q
```

14 tests cover:
- HSA's axis mapping
- the button modes
- base-frame rotation
- wrist mode moving the tool like arm mode
- SpaceMouse HID decoding of synthetic reports
- terminal key parsing and keyboard holds
- the hdf5 writer (save / drop / append)
- a headless record → validate → replay run

`python -m pytest nullspace/test_nullspace.py -q` runs the nullspace mode's 9 tests. "Testing and evaluating
nullspace mode" above covers them and the workspace evaluation.

For an interactive check without the puck, run `--input fake_sine` or `--input keyboard` with the viewer and try every
command key.

`python make_teleop_gifs.py` renders the teleop behavior to `screenshots/teleop_*.gif` (~1 min). It drives the real
`Teleop` loop with a simulated puck: raw readings go through the same HSA axis map and button logic as the real
device, the mode switch is a both-buttons press, and the gripper uses the `g` key. Each frame has a wide view, a
close-up that follows the wrist, the mode and puck input, and the measured arm tcp motion and wrist offset (robot
base frame).
1. `teleop_1_arm_6d.gif`: the arm in 6D: ±x, ±y, ±z (right button), then rotations (left button).
2. `teleop_2_wrist.gif`: toggle to the wrist and move it in 6D to its limits. The arm's tcp stays within 0.3 mm.
3. `teleop_3_gripper.gif`: `g` closes, then opens.
4. `teleop_4_arm_wrist_offset.gif`: the wrist is left raised 20 mm, rolled and yawed. Toggle back to the arm and move
   it in 6D; the wrist keeps its offset.

### When the SpaceMouse is plugged in

- The reader uses `hid` (already installed; robosuite's own driver uses it) and decodes reports exactly as pyspacemouse
  2.1.0 does, which is what HSA-UMI reads with. It covers the SpaceNavigator, SpaceMouse Compact / Wireless / Pro /
  Enterprise, and the Universal Receiver. Button 0/1 are the left/right buttons.
- If no device is found: quit 3Dconnexion's driver (3DxWare / 3DconnexionHelper), which can hold the device on macOS.
- **Check the axis mapping first.** `--axis-map hsa` (the default) is HSA-UMI's matrix exactly, robot = [−z, x, y] of
  the device. That matrix is UMI's `tx_zup_spnav`, written for libspacenav's y-up axes, but HSA-UMI feeds it
  pyspacemouse's readings, which I believe are z-up (x right, y forward, z up). If so, lifting the puck moves the arm
  along −x. I couldn't test this without a device. Run `mjpython teleop_wrist.py --input spacemouse --print-axes` and
  push each axis. If the motion feels scrambled, use `--axis-map zup`: puck forward → +x, left → +y, up → +z for an
  operator behind the robot, which is how the default viewer camera looks.

### Differences from HSA-UMI

- One absolute target per control tick goes to OSC. HSA-UMI streams 6-waypoint chunks extrapolated along the puck's
  velocity to its impedance controller.
- Added:
  - the keyboard and fake inputs
  - wrist and nullspace modes (both buttons, `m`)
  - the gripper (`g`), wrist home (`h`) and scene reset (`z`)
  - an automatic scene reset after `s` or a drop
  - `x` scaling the Stewart legs' stiffness, instead of switching HSA servo setpoints
- The recording rate is the control rate (`--control-freq`, default 20 Hz, robosuite's usual), not a camera frame rate.

### Running on Linux

This was written on macOS and hasn't been run on Linux. The notes below come from the code and the packages'
documented behavior.

- **Use `python`, not `mjpython`.** Only macOS needs `mjpython`, because macOS requires windows to run on the main
  thread. On Linux the viewer opens from plain `python`. The script checks for `mjpython` only on macOS.
- **Install with `requirements.txt`.** It leaves out the macOS-only `pyobjc-*` packages and adds pynput's Linux
  backends (see Setup).
- **Display.** The viewer needs a desktop session with OpenGL. The scripts that only render offscreen can also run with
  no display if you set `MUJOCO_GL=egl` (GPU) or `MUJOCO_GL=osmesa` (CPU). These are `make_teleop_gifs.py`,
  `nullspace/make_demo_gif.py`, `nullspace/evaluate.py` and `nullspace/make_figures.py`.
- **Keyboard motion.** `pynput` reads keys through X11. Under Wayland it usually doesn't see key presses sent to other
  windows, so the motion keys may do nothing. Use an X11 session ("Ubuntu on Xorg" at login, for example). The command
  keys (`m`, `c`, `s`, ...) are read from the terminal, so they work under either.
- **SpaceMouse.** Reading the device needs access to `/dev/hidraw*`. Add a udev rule for 3Dconnexion devices (vendor
  `256f`; older Logitech-era units use `046d`, so add a second line with that), then unplug and replug the puck:

  ```bash
  echo 'KERNEL=="hidraw*", ATTRS{idVendor}=="256f", MODE="0666"' | sudo tee /etc/udev/rules.d/99-spacemouse.rules
  sudo udevadm control --reload-rules && sudo udevadm trigger
  ```

  If `spacenavd` is running, stop it. It can hold the device, as 3DxWare does on macOS.

## Key insertion task (`KeyInsertion`)

A top-down key-in-lock / RAM-stick-in-slot task. A rigid 10 g key sits on the wrist's top plate in place of the ALOHA
gripper (gripper `StewartWristKey`), and the task is to push it down into a lock on the table. The keyhole fits the
blade with 0.25 mm clearance per side and has a 2 mm chamfered entrance. A spring-loaded plunger at the bottom pushes
back with 4 N at first touch, rising to 10 N when the key is seated, so seating takes a real push that the wrist carries.
The lock's position and yaw are randomized on every reset. Full description, results and figures:
[scripts/key_insertion/README.md](scripts/key_insertion/README.md).

```python
env = suite.make("KeyInsertion", robots="PandaStewartAloha")   # picks gripper_types="StewartWristKey" itself
```

The action is 12-dim: the arm's tcp pose (6) and the wrist's pose (6), with no grip. Success means the key is seated,
i.e. the plunger is pushed down at least 90% of its 4 mm travel. Extra observations: `lock_pos` / `lock_quat`,
`key_tip_to_lock_pos` / `_quat` (the key's tip in the keyhole frame), `plunger_depression`, and `flange_wrench` (the
external wrench on the arm's flange, as in the teleop readout). The env's keyword arguments set the fit and the push:
`clearance`, `chamfer`, `slot_depth`, `plunger_travel`, `seat_preload`, `seat_force`, `lock_friction`, the lock's
randomization ranges, and `lock_alpha` (see-through lock walls for rendering).

Watch the scripted expert (oracle), then change the wrist's stiffness or give it a perception error:

```bash
cd scripts
mjpython -m key_insertion.oracle                              # home wrist, lock where the oracle thinks it is
mjpython -m key_insertion.oracle --wrist stiff --dx 1 --dyaw 3   # contracted (stiff) wrist, keyhole 1 mm / 3 deg off
mjpython -m key_insertion.oracle --wrist soft --dx 1 --dyaw 3    # extended (soft) wrist, same error
mjpython -m key_insertion.oracle --wrist 0.5 --stiffness-scale 2 # any wrist z in [-1, 1], legs 2x stiffer
mjpython -m key_insertion.oracle --k-extended 1000 --k-contracted 50000   # change the stiffness law itself
python -m key_insertion.oracle --no-render --episodes 10      # headless, one summary line per episode
```

Use plain `python` instead of `mjpython` on Linux. Other flags: `--speed` (mm/s), `--max-force` (the oracle's force
stop), `--retract`, `--clearance` / `--chamfer` (mm) / `--seat-force` (N), `--arm-kp`, `--fixed-lock`, `--slowdown`.
The viewer shows the phase, the depth, the flange and key contact forces, the wrist's deflection and the leg stiffness.

The oracle holds the wrist at one pose throughout and lowers the arm's target straight down. It doesn't use the force
to steer, so whatever the chamfer has to correct is taken up by the wrist's compliance (and the arm's impedance). That
makes it a direct probe of how stiffness affects the task.

Teleop works too: `mjpython teleop_wrist.py --env KeyInsertion --input spacemouse`. Recorded demos keep the
randomized lock pose in their `model_file`, so `validate_demos.py` replays them exactly.

Study and figures:

```bash
cd scripts
python -m pytest key_insertion/test_key_insertion.py -q   # env, key, oracle (7 tests, ~30 s)
python -m key_insertion.evaluate         # ~800 oracle episodes over stiffness x perception error (~8 min; --quick ~3)
python -m key_insertion.make_figures     # figures/force_profiles.png, success_maps.png, compliance_sweep.png
python -m key_insertion.render_scene     # figures/scene.png, insertion.gif
```

### SERL version (`KeyInsertionWrist-v0`)

A gym env of the same task for SERL: the lock fixed, a 40 N RAM-style snap latch, the arm on a diagonal 20 kN/m
Cartesian impedance (`CARTESIAN_IMPEDANCE`), the key starting near the keyhole, and three wrist modes: stiff, soft,
and variable (the wrist's z as an action). A scripted oracle supplies the demos. Design, numbers and commands:
[scripts/key_insertion/README.md](scripts/key_insertion/README.md#serl-does-a-variable-stiffness-wrist-help). SERL
needs its own Python 3.10 venv (`scripts/key_insertion/requirements_serl.txt`).

```bash
cd scripts
python -m pytest key_insertion/test_serl_env.py -q           # controller, latch, env, oracle, timestep (11 tests, ~1 min)
python -m key_insertion.collect_demos --mode all --n 20    # with the SERL venv's python
bash key_insertion/launch_serl.sh variable 0               # SERL venv: learner + actor, logs in key_insertion/runs/
```

## Validation and screenshots

```bash
python scripts/render_stewart_wrist.py      # writes ./screenshots/
cd scripts && python -m pytest test_teleop.py -q   # teleop stack (see above)
```

The script produces:
- `all_tasks.png`, plus `<Task>_<camera>.png`: Lift, Stack, PickPlaceCan, NutAssemblySquare and Door from the
  frontview, agentview and eye-in-hand cameras.
- `wrist_workspace_limits.png`: one row per axis (x, y, z extension, roll and pitch tilt, yaw twist) showing
  -limit / home / +limit, with commanded and measured values.
- `scripted_lift.png`: an open-loop pick in Lift (prints `Scripted Lift success: True`).

## Where the code lives (all inside `./robosuite/robosuite`)

| file | role |
|---|---|
| `models/wrists/wrist_model.py` | `WristModel` interface: MJCF string, mount body, actuators, observed joints, `format_action` |
| `models/wrists/stewart_platform.py` | `StewartPlatformConfig` (geometry, workspace, dynamics) + `StewartPlatformWrist` (MJCF generator, closed-form IK) |
| `models/grippers/wrist_gripper.py` | `WristGripperModel`: generic composition of any wrist with any robosuite gripper, and the flange wrench (`flange_wrench`); `StewartAlohaGripper`, `StewartAlohaGripperLegControl`, `StewartAlohaGripperLockedWrist`, `StewartWristNoGripper`, `StewartWristKey` |
| `models/grippers/key_tool.py` | `KeyConfig` + `KeyTool`: the rigid key (blade, bow, 10 g) as a 0-dof gripper |
| `environments/manipulation/key_insertion.py` | `KeyInsertion`: the lock (chamfered slot, spring plunger or `latch_profile`), randomization, reward, observations, contact forces |
| `controllers/parts/arm/cartesian_impedance.py` | `CartesianImpedanceController` (`CARTESIAN_IMPEDANCE`): diagonal Cartesian stiffness, damping critical given the arm's inertia; shares OSC's goal handling (`OperationalSpaceController.compute_errors`) |
| `models/robots/compositional.py` | `PandaStewartAloha`: Panda arm, default gripper, init pose, eye-in-hand camera re-mount |
| `robots/wrist_robot.py` | `WristFixedBaseRobot`: arm controller on the flange tcp, `tcp_pos` / `tcp_quat` observations, per-step leg gains, mocap sleeves |
| `models/grippers/__init__.py`, `robots/__init__.py` | registration |

Teleop (in `./scripts`): `teleop_wrist.py` (main loop, keys, viewer), `teleop_math.py` (target integration, axis
maps), `teleop_inputs.py` (SpaceMouse / keyboard / fake inputs, terminal keys), `demo_writer.py` (hdf5),
`validate_demos.py`, `test_teleop.py`.

### Changing the wrist (geometry, workspace, leg stiffness)

All three StewartAloha variants build their wrist from `StewartAlohaGripper.wrist_config`. Replace it before
creating the env:

```python
from robosuite.models.grippers import StewartAlohaGripper
from robosuite.models.wrists import StewartPlatformConfig

StewartAlohaGripper.wrist_config = StewartPlatformConfig(
    leg_stiffness_contracted=1e5,  # N/m per leg, wrist fully contracted (default 1e5)
    leg_stiffness_extended=300.0,  # N/m per leg, wrist fully extended (default 300)
    # leg_stroke=0.025,   # fixed leg travel (default: just enough for the whole workspace box)
    # z_range=0.015, tilt_range=8.0, ...
)
env = suite.make("Lift", robots="PandaStewartAloha")
```

You can also edit the defaults in `models/wrists/stewart_platform.py`.

- **Length-dependent leg stiffness.** The defaults follow the real wrist: very soft when extended, about as stiff as
  a rigid mount when contracted. Each leg's stiffness depends on its *commanded* (unloaded) length L:
  - `leg_stiffness_contracted` (100 kN/m) at the wrist's fully contracted pose (z = −`z_range`, every leg at its
    shortest);
  - `leg_stiffness_extended` (300 N/m) at the fully extended pose (z = +`z_range`);
  - log-linear in L in between (about 6 kN/m at home), and clamped to that range beyond it, e.g. when tilted.

  Set both equal for constant stiffness. The gains are set on the live model before every sim step
  (`WristFixedBaseRobot` → `WristGripperModel.update_actuator_gains` → `StewartPlatformWrist.actuator_gains`).

  | Wrist pose | Leg k | Wrist, axial (K_zz) | Wrist, lateral (K_xx) | Wrist, tilt (K_roll) |
  |---|---|---|---|---|
  | fully contracted (z −20 mm) | 100 kN/m | 400 kN/m | 100 kN/m | 417 N·m/rad |
  | home | 6.0 kN/m | 28 kN/m | 3.9 kN/m | 29 N·m/rad |
  | fully extended (z +20 mm) | 300 N/m | 1.5 kN/m | 140 N/m | 1.6 N·m/rad |

- **Damping and armature follow the stiffness.** `leg_damping` and `leg_armature` are the values for the stiffest
  leg:
  - Leg damping scales with √k.
  - The leg armature (an artificial reflected inertia that keeps the stiff legs stable at the 2 ms timestep) scales
    with k. Every leg's natural frequency therefore stays at its stiff-end value, and the soft wrist isn't slowed by
    inertia it doesn't have.
  - The top plate's passive-joint damping (`top_lin_damping`, `top_rot_damping`) scales with √(mean leg k).

  With the defaults, a wrist step doesn't overshoot and settles within 2% in 0.03–0.7 s, slowest when the wrist is
  extended. Above about 2e5 N/m for `leg_stiffness_contracted`, raise `leg_armature` proportionally, because the leg
  springs are integrated explicitly.
- **Gravity sag.** The soft wrist sags visibly under the gripper's 0.59 kg. As on the real wrist, the top plate
  (`top_mass`, 10 g) adds almost nothing; the strut rods add 60 g, partly carried by the plate. Pointing down, the
  wrist sags about 0.4 mm at home and 2–3 mm when 15 mm extended. With `x` (0.3×), that becomes 1.0 mm and 7 mm. The commanded wrist pose is where the
  plate would be with no load. `p` and the flange-force readout show the measured pose and the deflection.
- **6D stiffness.** `wrist.platform_stiffness(pose)` returns the 6×6 K = Jᵀ diag(k) J in the pose coordinates
  (x, y, z, roll, pitch, yaw), with the legs commanded to `pose`. Get the wrist from
  `env.robots[0].gripper["right"].wrist`; the last commanded pose is `wrist.last_commanded_pose`.
  - **Check without gravity:** in the full robot with gravity off, applying ± forces and torques to the top plate
    matches K⁻¹ to within 0.2% on all 6 axes. That holds at the z limits, at home and at a tilted corner, and with
    `x` on or off. This needs `eq_solimp` = (0.9999, 0.99999, 0.001). The old (0.99, 0.999, 0.001) let the ball
    joints add compliance in series, so the wrist was about 33% softer than its legs.
  - **With gravity:** the gripper's weight preloads the struts, which adds a geometric stiffness term that K leaves
    out. At home the measured stiffness differs from K by up to about 20%; extended, by 10–70%, the most with `x`. That
    is real mechanics of a loaded soft wrist, simulated by MuJoCo, not a numerical error.
- **Stability.** A random stress test showed no NaNs, no MuJoCo warnings, no leg reaching its stroke limit, and at
  most 1.1 mm of loop-closure stretch. It ran 3 seeds × 4 episodes of random arm jumps, full-range wrist steps, `x`,
  gripper and mode toggles.
- **Runtime stiffness.** `wrist.stiffness_scale` (default 1.0) multiplies every leg's stiffness, and damping and
  armature follow it. The teleop's `x` key toggles it. It isn't part of the sim state.
- To write a completely new wrist, make a `WristModel` subclass and pass it to
  `WristGripperModel(wrist=..., gripper=...)`. Nothing outside the wrist file needs to know how it works inside.

### Notes / design decisions

- Geometry: the legs start on a 120 mm diameter circle and end on a 90 mm one, and the two anchor planes are 80 mm
  apart. The plates are 8 mm thick and 132 / 102 mm across. A visual-only 30 mm spacer, 132 mm across like the base
  plate, sits between the arm flange and the base plate (`spacer_height`, `spacer_radius`). The wrist is 126 mm from
  flange to top face. Home leg length is 90.6 mm, at 28° from vertical.
- Workspace (unchanged): x/y ±10 mm, z ±20 mm, roll/pitch ±10°, yaw ±10°. Every corner of that box is reachable,
  which takes a ±39 mm leg stroke. Each leg is drawn as a three-stage telescope: housing, middle sleeve, rod. Nothing
  extends below the pivot.
- The middle sleeves are mocap bodies with no physics: no joints, constraints, mass or collision. The robot class
  `WristFixedBaseRobot` (`robots/wrist_robot.py`) moves them to half the rod's travel before every sim step and after
  reset. They lag the physics by one 2 ms step (≤0.3 mm while moving), and this costs about 1 ms per policy step.
  If you restore a saved state without stepping (e.g. robomimic `reset_to`), the sleeves show stale positions until the
  next step. MuJoCo requires mocap bodies to sit directly under the world, so they are added to the robot model's
  worldbody rather than inside the wrist.
- The wrist is a real closed chain. It has 6 legs, each with a universal joint, a position-actuated prismatic joint
  and a ball-joint constraint to the top plate. The 6 platform DoF are passive joints, and all joints are 0 at home.
- The loop-closure constraints are harder than MuJoCo's defaults (`eq_solref`, `eq_solimp`). With the defaults, the
  chain stretched by up to 9 mm during fast moves. The passive platform joints are damped (`top_lin_damping`,
  `top_rot_damping`) so that the arm's OSC controller can't excite the chain's fast internal modes.
- Measured performance:
  - Each axis reaches its limits to within 0.1 mm / 0.1°.
  - The platform sags about 0.26 mm under the gripper's weight.
  - With zero action, the hand drifts 0.03 mm.
  - A full-range step settles in about 100 ms with about 10% overshoot.
  - Random full-range commands every 50 ms, with the arm moving too, overshoot the box by up to about 1.5× in
    transients. That's real dynamics, not clipping, since the commanded pose is always inside the box.
- The ALOHA `grip_site` is moved from the palm (z = 0.04) to between the fingerpads (z = 0.10). That matches
  robosuite's convention, which the reward shaping and `robot0_eef_pos` use. The arm controller tracks the flange tcp
  instead (see above). The robosuite_models files are not modified.
- `PandaStewartAloha.init_qpos` puts the grip site at the stock Panda's starting pose (with no initialization
  noise). The wrist and gripper together are about 17 cm longer than the Panda hand.
- `Wipe` forces its own `WipingGripper`, so in that task the wrist is replaced, as it would be for any robot.
