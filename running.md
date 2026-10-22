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
uv pip install --python robosuite_env/bin/python -r requirements.txt   # exact versions; installs ./robosuite editable
```

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
space. Its math, implementation and workspace evaluation are in
[`scripts/nullspace/README.md`](scripts/nullspace/README.md).

Run it from `scripts/`. On macOS the viewer needs `mjpython`, which is in the venv:

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
| `p` | print the tcp and wrist poses, their targets and the teleop state |
| `r` | re-seed the arm target on the arm |
| `x` | toggle the wrist stiff (1.0×) / soft (`--soft-scale`, 0.3×) leg stiffness |
| `m` | next mode: arm → wrist → nullspace → arm (same as both puck buttons). Nullspace mode uses `--nullspace-comp feedback` (default) or `feedforward`. |
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

`python -m pytest nullspace/test_nullspace.py -q` runs the nullspace mode's 7 tests (see its README).

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
| `models/grippers/wrist_gripper.py` | `WristGripperModel`: generic composition of any wrist with any robosuite gripper; `StewartAlohaGripper`, `StewartAlohaGripperLegControl`, `StewartAlohaGripperLockedWrist` |
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
    actuator_kp=3e4,      # leg stiffness at home length, N/m (default 1e5)
    # leg_stiffness_exponent=0,  # constant stiffness (default 1: k = EA/L)
    # leg_stroke=0.025,   # fixed leg travel (default: just enough for the whole workspace box)
    # z_range=0.015, tilt_range=8.0, ...
)
env = suite.make("Lift", robots="PandaStewartAloha")
```

You can also edit the defaults in `models/wrists/stewart_platform.py`.

- Length-dependent leg stiffness (on by default): each leg's stiffness is
  `k = actuator_kp * (L_home / L) ** leg_stiffness_exponent`, where L is the leg's *commanded* (unloaded) length.
  The default exponent 1 is an elastic member, k = EA/L: shorter legs are stiffer. 0 gives the old constant
  stiffness. Leg damping scales with sqrt(k), so each leg's damping ratio stays the same. The gains are set on the
  live model before every sim step (`WristFixedBaseRobot` → `WristGripperModel.update_actuator_gains`
  → `StewartPlatformWrist.actuator_gains`). Over the default workspace box, k ranges from 0.70× to 1.67× of
  `actuator_kp` (exponent 1), which stays under the ~2e5 N/m stability limit for the default armature. With a larger
  exponent or `actuator_kp`, keep `actuator_kp * max ratio` under that limit, or raise `leg_armature`.
- 6D stiffness: `wrist.platform_stiffness(pose)` returns the 6×6 K = Jᵀ diag(k) J in the pose coordinates
  (x, y, z, roll, pitch, yaw), with the legs commanded to `pose`. Get the wrist from
  `env.robots[0].gripper["right"].wrist`; the last commanded pose is `wrist.last_commanded_pose`. A fixed-base test
  that applies forces and torques to the top plate matches K⁻¹ to within 1% on all 6 axes, at home, at the z limits
  and at a workspace corner, for exponents 1 and 2. For that match, `eq_solimp` is now (0.9999, 0.99999, 0.001). The
  old (0.99, 0.999, 0.001) let the ball joints add compliance in series, so the wrist was about 33% softer than its
  legs.
- Leg stiffness: I tested `actuator_kp` from 1e4 to 5e5 N/m with random arm and wrist actions, and it was stable
  throughout. Lower values make the wrist sag more under the gripper's weight. Above about 2e5, raise
  `leg_armature` proportionally, because the leg actuators are integrated explicitly at the 2 ms timestep.
- Runtime stiffness: `wrist.stiffness_scale` (default 1.0) multiplies every leg's stiffness, with damping following
  √k. The teleop's `x` key toggles it. It isn't part of the sim state.
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
