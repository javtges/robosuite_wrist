"""
Teleoperate PandaStewartAloha (Panda arm + Stewart wrist + ALOHA gripper) in any robosuite task with a SpaceMouse,
a 6D keyboard, or a fake input, and record demonstrations to a robomimic / robosuite hdf5 file.

Follows HSA-UMI's spacemouse_teleop: the input is a velocity that is integrated into an absolute target pose of the
arm's tcp (a frame fixed to the flange at the wrist's mount plate, like hsa_tcp), clipped to a workspace box, and
re-seeded onto the arm if the two drift more than --reseed-distance apart. The target is sent as an absolute OSC_POSE
command in the robot base frame, and it is what gets recorded as the action. m (or both puck buttons) cycles what the
input drives:
    arm        the arm's tcp target; the wrist holds its pose
    wrist      the wrist's pose target; the arm holds its target, and wrist motion adds on top of it
    nullspace  the wrist's pose target, as in wrist mode, while the arm moves the opposite way so the wrist's top plate
               and the gripper stay still (nullspace/README.md)

Run from this directory. On macOS, use mjpython when the viewer is on:
    mjpython teleop_wrist.py --env Lift --input spacemouse
    mjpython teleop_wrist.py --env Lift --input keyboard
    python teleop_wrist.py --env Lift --input fake_sine --no-render --max-seconds 10 --key-script "0.5:c,8:s,9:q"

SpaceMouse (HSA-UMI's button modes): left button held = rotate (translation locked); right button held = z allowed
(otherwise z is locked); both buttons pressed together = next mode (arm -> wrist -> nullspace -> arm).

Keyboard motion (hold; robot base frame: +x away from the base, +y to its left, +z up):
    i / k  +x / -x      j / l  +y / -y      u / o  +z / -z
    [ / ]  +rot x / -   up / down  +rot y / -   left / right  +rot z / -
Command keys are read from the terminal, so keep the terminal focused (the viewer window has its own key bindings).

The viewer shows the wrench the wrist and gripper apply to the Franka's flange (MuJoCo's own interaction force between
the wrist base and the arm), the wrist's leg stiffness and its deflection under load; f also prints it in the terminal.
"""

import argparse
import csv
import dataclasses
import datetime
import os
import sys
import time

import mujoco
import mujoco.viewer
import numpy as np
from scipy.spatial.transform import Rotation

import robosuite as suite
from robosuite.controllers import load_composite_controller_config

from arm_controller import ARM_CONTROLLERS, apply_arm_controller, describe as describe_arm_controller
from demo_writer import DemoWriter
from teleop_inputs import FakeSineSource, KeyboardSource, NullSource, SpaceMouseSource, TerminalKeys
from nullspace import STRATEGIES, NullspaceCompensator
from teleop_math import clamp_workspace, integrate_target, integrate_wrist_target

ROBOT = "PandaStewartAloha"
ARM, WRIST, NULLSPACE = 0, 1, 2
MODE_NAMES = ("arm", "wrist", "nullspace")

COMMANDS = [
    ("c", "start recording an episode"),
    ("s", "stop: save the episode, then reset the scene"),
    ("Backspace", "drop the episode in progress (then reset the scene), or the last saved one if idle"),
    ("Space", "stage + 1"),
    ("n", "skip to the next plan row (--plan)"),
    ("p", "print the poses and the teleop state"),
    ("f", "print the flange force every 0.5 s (on / off); the viewer always shows it"),
    ("r", "re-seed the arm target on the arm"),
    ("x", "toggle the wrist stiff / soft (--soft-scale)"),
    ("m", "next mode: arm -> wrist -> nullspace -> arm (same as both puck buttons)"),
    ("g", "toggle the gripper open / closed"),
    ("h", "send the wrist target home"),
    ("z", "reset the scene (not while recording)"),
    ("q", "quit (drops an episode in progress)"),
]


class PassiveViewer:
    """
    One MuJoCo passive viewer for the whole session. robosuite's own "mjviewer" renderer closes and reopens its window
    on every env.reset(), and under mjpython (macOS) the reopen fails because the close is asynchronous. With
    hard_reset=False the model and data survive resets, so one window can follow them. A new model (a demo loaded
    from its model_file) gets a new window.
    """

    # Over the robot's shoulder, looking along +x (away from the base), so screen directions match the robot frame
    CAMERA = dict(lookat=[0.0, 0.0, 0.88], distance=1.5, azimuth=35.0, elevation=-35.0)

    def __init__(self):
        if sys.platform == "darwin" and getattr(mujoco.viewer, "_MJPYTHON", None) is None:
            sys.exit("On macOS the viewer needs mjpython: run `mjpython <script> ...` (or pass --no-render).")
        self.handle, self.model = None, None

    def sync(self, sim, texts=None):
        """Returns False once the user has closed the window. texts: overlay, as mujoco.viewer's set_texts takes it."""
        model, data = sim.model._model, sim.data._data
        if self.handle is not None and self.model is not model:
            self.close()
        if self.handle is None:
            t0 = time.monotonic()
            while True:
                try:
                    self.handle = mujoco.viewer.launch_passive(model, data, show_left_ui=False, show_right_ui=False)
                    break
                except RuntimeError as e:
                    # mjpython: the previous window is still shutting down
                    if "already open" not in str(e) or time.monotonic() - t0 > 5.0:
                        raise
                    time.sleep(0.05)
            self.model = model
            self.handle.opt.geomgroup[0] = 0  # hide collision geoms, as robosuite does
            for k, v in self.CAMERA.items():
                setattr(self.handle.cam, k, v)
        if not self.handle.is_running():
            return False
        if texts is not None:
            self.handle.set_texts(texts)
        self.handle.sync()
        return True

    def close(self):
        if self.handle is None:
            return
        self.handle.close()
        t0 = time.monotonic()
        while self.handle.is_running() and time.monotonic() - t0 < 2.0:
            time.sleep(0.01)
        self.handle = None


def make_env_args(args):
    """robomimic's env_args: everything needed to rebuild the env (with rendering off)"""
    cfg = load_composite_controller_config(robot=ROBOT)
    cfg["body_parts"]["right"].update(input_type="absolute", input_ref_frame="base")
    apply_arm_controller(cfg, getattr(args, "arm_controller", "osc"), getattr(args, "arm_gains_yaml", None))
    env_kwargs = dict(
        robots=[ROBOT],
        **({"gripper_types": "StewartWristNoGripper"} if getattr(args, "no_gripper", False) else {}),
        controller_configs=cfg,
        control_freq=args.control_freq,
        use_camera_obs=False,
        has_renderer=False,
        has_offscreen_renderer=False,
        ignore_done=True,
        hard_reset=False,
    )
    return {"env_name": args.env, "type": 1, "env_version": suite.__version__, "env_kwargs": env_kwargs}


def make_input(args):
    if args.input == "spacemouse":
        return SpaceMouseSource(deadzone=args.deadzone, mapping=args.axis_map)
    if args.input == "keyboard":
        return KeyboardSource()
    if args.input == "fake_sine":
        return FakeSineSource()
    return NullSource()


def load_plan(path):
    if path is None:
        return []
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def parse_key_script(text):
    """ "0.5:c,8:s" -> [(0.5, "c"), (8.0, "s")], teleop seconds since start"""
    if not text:
        return []
    out = []
    for item in text.split(","):
        t, key = item.split(":", 1)
        out.append((float(t), key.strip().lower()))
    return sorted(out)


class Teleop:
    def __init__(self, args):
        self.args = args
        self.hz = args.control_freq
        self.env_args = make_env_args(args)
        self.viewer = None if args.no_render else PassiveViewer()
        self.env = suite.make(args.env, **self.env_args["env_kwargs"])
        self.robot = self.env.robots[0]
        self.gripper = self.robot.gripper["right"]
        self.wrist = self.gripper.wrist
        self.wrist_limits = self.wrist.action_scale
        self.writer = DemoWriter(args.out, self.env_args, dataclasses.asdict(self.wrist.config))
        self.input = make_input(args)
        if isinstance(self.input, FakeSineSource):
            self.input.clock = lambda: self.env.sim.data.time  # deterministic in sim time
        self.keys = TerminalKeys() if not args.no_terminal_keys else None
        self.key_script = parse_key_script(args.key_script)
        self.plan, self.plan_idx = load_plan(args.plan), 0

        self.mode = ARM
        self.stiff = True
        self.stage = 0
        self.gripper_closed = False
        self._both_prev = False
        self._last_reseed_warn = -1e9
        self.ticks = 0  # since start; sim time restarts at every scene reset
        self.print_forces = getattr(args, "print_forces", False)
        self.reset_scene()
        self.box = self.workspace_box()
        self.nullspace = NullspaceCompensator.from_sim(self.env.sim, self.gripper, self.wrist)

    # ---- geometry ----------------------------------------------------------------------------------------------------

    def base_pose(self):
        return self.robot.composite_controller.get_controller_base_pose(controller_name="right")

    def measured_tcp(self):
        """Measured tcp pose [pos, rotvec], robot base frame"""
        p0, R0 = self.base_pose()
        site = self.gripper.important_sites["tcp"]
        pos = R0.T @ (self.env.sim.data.get_site_xpos(site) - p0)
        R = R0.T @ self.env.sim.data.get_site_xmat(site)
        return np.concatenate([pos, Rotation.from_matrix(R).as_rotvec()])

    def wrist_rotation(self):
        """Orientation of the wrist base frame in the robot base frame"""
        _, R0 = self.base_pose()
        return R0.T @ self.env.sim.data.get_body_xmat(self.gripper.correct_naming(self.wrist_base_body))

    @property
    def has_gripper(self):
        """False with --no-gripper: the wrist carries nothing, and the action has no grip dimension"""
        return self.gripper.gripper.dof > 0

    @property
    def wrist_base_body(self):
        return self.gripper.worldbody[0].get("name")[len(self.gripper.naming_prefix) :]

    def measured_wrist_pose(self):
        names = [f"wrist_top_{a}" for a in ("x", "y", "z", "rx", "ry", "rz")]
        return np.array([self.env.sim.data.get_joint_qpos(self.gripper.correct_naming(n)) for n in names])

    def flange_wrench(self):
        """(total, external) wrench the wrist applies to the Franka's flange: WristGripperModel.flange_wrench"""
        return self.gripper.flange_wrench(self.env.sim)

    def leg_stiffness(self):
        """The wrist legs' current spring constants (N/m), read from the MuJoCo position actuators' gains"""
        m = self.env.sim.model._model
        ids = [m.actuator(self.gripper.correct_naming(a)).id for a in self.wrist.actuators]
        return m.actuator_gainprm[ids, 0].copy()

    def wrist_deflection(self):
        """Measured top-plate pose minus its command: how far the load pushes the wrist off its target (m, rad)"""
        return self.measured_wrist_pose() - self.wrist_command

    def force_lines(self):
        """(label, value) rows of the flange-force readout"""
        total, ext = self.flange_wrench()
        k = self.leg_stiffness() / 1e3
        defl = self.wrist_deflection()
        v = lambda x: "[" + " ".join(f"{c:+7.2f}" for c in x) + "]"
        return [
            ("flange force on the Franka", "(flange frame, +z toward the gripper)"),
            ("external force (N)", f"{v(ext[3:])}  |F| {np.linalg.norm(ext[3:]):6.2f}"),
            ("external moment (N m)", f"{v(ext[:3])}  |M| {np.linalg.norm(ext[:3]):6.2f}"),
            ("total force, with weight (N)", f"{v(total[3:])}  |F| {np.linalg.norm(total[3:]):6.2f}"),
            ("wrist leg stiffness (kN/m)", f"{k.min():.1f} - {k.max():.1f}  (x{self.wrist.stiffness_scale:g})"),
            ("wrist deflection", f"{np.linalg.norm(defl[:3]) * 1e3:.3f} mm  {np.degrees(np.linalg.norm(defl[3:])):.3f} deg"),
        ]

    def workspace_box(self):
        a = self.args
        box = dict(x_min=a.x_min, x_max=a.x_max, y_min=a.y_min, y_max=a.y_max, z_min=a.z_min, z_max=a.z_max)
        if box["z_min"] is None:
            # Floor: the gripper's lowest point (wrist at home) just touches the table / bin floor
            floor = next((getattr(self.env, k)[2] for k in ("table_offset", "bin1_pos") if hasattr(self.env, k)), None)
            if floor is None:
                box["z_min"] = 0.0
            else:
                m, d = self.env.sim.model._model, self.env.sim.data._data
                tcp_z = d.site_xpos[m.site(self.gripper.important_sites["tcp"]).id][2]
                lowest = min(
                    (
                        d.geom_xpos[i][2] - m.geom_rbound[i]
                        for i in range(m.ngeom)
                        if m.geom(i).name.startswith(self.gripper.naming_prefix)
                        and "wrist" not in m.geom(i).name
                        and m.geom_contype[i]
                        and m.geom_rbound[i] < 0.05
                    ),
                    default=None,
                )
                if lowest is None:  # no gripper: the top plate's face (pointing down) is the lowest point
                    plate = m.geom(self.gripper.correct_naming("wrist_top_col")).id
                    lowest = d.geom_xpos[plate][2] - m.geom_size[plate][1]
                box["z_min"] = floor - self.base_pose()[0][2] + (tcp_z - lowest) + a.floor_margin
        return box

    # ---- scene / modes -----------------------------------------------------------------------------------------------

    def reset_scene(self):
        self.obs = self.env.reset()
        self.target = self.measured_tcp()
        if hasattr(self, "box"):
            self.target = clamp_workspace(self.target, self.box)
        self.wrist_target = np.zeros(6)
        self.wrist_command = np.zeros(6)  # what the wrist is sent: the target, or the nullspace feedback command
        self.gripper_closed = False
        self.stage = 0
        if getattr(self, "mode", ARM) == NULLSPACE:
            self.anchor_nullspace()
        self._announced_success = False
        self.success_any = False

    def anchor_nullspace(self):
        """
        Freezes the plate pose that nullspace mode holds. Feedforward: where the arm's target and the wrist's command
        put it. Feedback: where the arm actually is (measured tcp) with the wrist's command, so the wrist also absorbs
        the arm's steady-state tracking error and the plate stays exactly where it was.
        """
        comp = self.args.nullspace_comp
        if comp == "plate":
            # The plate where it actually is; the operator's wrist target becomes that measured pose
            measured = self.measured_wrist_pose()
            self.nullspace.anchor(self.measured_tcp(), measured, self.wrist_command)
            self.wrist_target = measured.copy()
            return
        tcp = self.measured_tcp() if comp == "feedback" else self.target
        self.nullspace.anchor(tcp, self.wrist_command)

    def toggle_mode(self):
        if self.mode == NULLSPACE:
            self.wrist_target = self.wrist_command.copy()  # the wrist keeps exactly what it was sent: no jump
        self.mode = (self.mode + 1) % 3
        if self.mode == NULLSPACE:
            self.anchor_nullspace()
            print(f"nullspace: input drives the wrist, the arm cancels it ({self.args.nullspace_comp})")
        else:
            print(f"input drives the {MODE_NAMES[self.mode]}")

    def print_state(self):
        t = self.measured_tcp()
        np.set_printoptions(precision=4, suppress=True)
        print(
            f"tcp (base frame) pos {t[:3]} rotvec {t[3:]} | target pos {self.target[:3]} rotvec {self.target[3:]}\n"
            f"wrist pose (m, rad) {self.measured_wrist_pose()} | target {self.wrist_target}\n"
            f"mode {MODE_NAMES[self.mode]} | wrist {'stiff' if self.stiff else 'soft'} "
            f"(x{self.wrist.stiffness_scale:g}) | gripper {'closed' if self.gripper_closed else 'open'} | stage {self.stage} | "
            f"{'RECORDING ' + str(self.writer.num_steps) + ' steps' if self.writer.recording else 'not recording'} | "
            f"{len(self.writer.demo_names)} demos in {self.args.out}"
        )
        self.print_force_lines()

    def print_force_lines(self):
        print("\n".join(f"  {label:30s} {value}" for label, value in self.force_lines()))

    def plan_row(self):
        return self.plan[self.plan_idx] if self.plan_idx < len(self.plan) else None

    def announce_plan(self):
        if self.plan:
            row = self.plan_row()
            print(f"plan row {self.plan_idx}: {row}" if row is not None else "plan finished")

    # ---- keys --------------------------------------------------------------------------------------------------------

    def handle_key(self, key):
        """Returns False to quit"""
        w = self.writer
        if key == "c":
            if w.recording:
                print("already recording")
            else:
                w.start(
                    dict(
                        plan_row=self.plan_row() or {},
                        condition=self.args.condition,
                        input=self.input.name,
                        gripper_current_action=np.asarray(self.gripper.gripper.current_action).tolist(),
                        model_file=self.env.sim.model.get_xml(),
                        # OSC pulls the redundant arm DoF toward this posture (taken at the scene reset); it is controller
                        # state, not sim state, so a replay has to restore it
                        controller_initial_joint=np.asarray(self.robot.part_controllers["right"].initial_joint).tolist(),
                    )
                )
                self.success_any = False
                print("recording started")
        elif key == "s":
            if not w.recording:
                print("not recording (c starts an episode)")
                return True
            n = w.num_steps
            name = w.save(success=self.env._check_success(), success_any=self.success_any)
            if name is None:
                print("episode was empty: nothing saved")
            else:
                print(f"saved {name}: {n} steps ({n / self.hz:.1f} s), task success: {self.success_any}")
                if n < self.args.min_steps:
                    print(f"WARNING: {name} is shorter than {self.args.min_steps} steps (Backspace drops it)")
                if self.plan:
                    self.plan_idx += 1
                    self.announce_plan()
            self.reset_scene()
        elif key == "backspace":
            if w.recording:
                w.discard()
                print("episode dropped")
                self.reset_scene()
            else:
                name = w.delete_last()
                print(f"deleted {name}" if name else "no saved demo to delete")
        elif key == "space":
            self.stage += 1
            print(f"stage {self.stage}")
        elif key == "n":
            if self.plan:
                self.plan_idx += 1
                self.announce_plan()
        elif key == "p":
            self.print_state()
        elif key == "f":
            self.print_forces = not self.print_forces
            print(f"flange force printout {'on' if self.print_forces else 'off'}")
        elif key == "r":
            self.target = clamp_workspace(self.measured_tcp(), self.box)
            if self.mode == NULLSPACE:
                self.anchor_nullspace()
            print("arm target re-seeded on the arm")
        elif key == "x":
            self.stiff = not self.stiff
            self.wrist.stiffness_scale = 1.0 if self.stiff else self.args.soft_scale
            print(f"wrist {'stiff' if self.stiff else 'soft'} (leg stiffness x{self.wrist.stiffness_scale:g})")
        elif key in ("m", "tab"):
            self.toggle_mode()
        elif key == "g":
            if not self.has_gripper:
                print("no gripper (--no-gripper)")
                return True
            self.gripper_closed = not self.gripper_closed
            print(f"gripper {'closing' if self.gripper_closed else 'opening'}")
        elif key == "h":
            self.wrist_target = np.zeros(6)
            print("wrist target home")
        elif key == "z":
            if w.recording:
                print("recording: s saves or Backspace drops the episode first")
            else:
                self.reset_scene()
                print("scene reset")
        elif key == "q":
            if w.recording:
                w.discard()
                print("episode in progress dropped")
            return False
        return True

    # ---- loop --------------------------------------------------------------------------------------------------------

    def tick(self):
        """One control step. Returns False when the viewer was closed."""
        a = self.args
        axes, buttons = self.input.read()
        both = buttons[0] and buttons[1]
        if both and not self._both_prev:
            self.toggle_mode()
        self._both_prev = both
        if both:
            axes = np.zeros(6)
        if a.print_axes and hasattr(self.input, "read_raw"):
            raw, b = self.input.read_raw()
            print(f"raw [x y z roll pitch yaw] {np.round(raw, 2)} buttons {b} -> robot {np.round(axes, 2)}", end="\r")

        exclusive = self.input.exclusive
        kw = dict(hz=self.hz, rotation_mode=buttons[0], z_enabled=buttons[1] or not exclusive, exclusive=exclusive)
        if self.mode == ARM:
            self.target = clamp_workspace(
                integrate_target(self.target, axes, max_pos_speed=a.max_pos_speed, max_rot_speed=a.max_rot_speed, **kw),
                self.box,
            )
        else:  # WRIST and NULLSPACE: the input moves the wrist
            self.wrist_target = integrate_wrist_target(
                self.wrist_target,
                axes,
                self.wrist_rotation(),
                self.wrist_limits,
                max_pos_speed=a.wrist_pos_speed,
                max_rot_speed=a.wrist_rot_speed,
                **kw,
            )

        if self.mode == NULLSPACE:
            # The arm target that keeps the plate on its anchor at this wrist pose (feedforward)
            self.target = clamp_workspace(self.nullspace.arm_target(self.wrist_target), self.box)

        # An arm far from its target is not following it: snap the target onto the arm and start there
        measured = self.measured_tcp()
        dist = float(np.linalg.norm(measured[:3] - self.target[:3]))
        if dist > a.reseed_distance:
            if self.elapsed - self._last_reseed_warn > 2.0:
                print(f"arm is {dist:.2f} m from the target: re-seeding the target on the arm")
                self._last_reseed_warn = self.elapsed
            self.target = clamp_workspace(measured, self.box)
            if self.mode == NULLSPACE:
                self.anchor_nullspace()

        if self.mode == NULLSPACE and a.nullspace_comp in ("feedback", "plate"):
            # The wrist also cancels whatever the arm hasn't caught up with yet; equals wrist_target once it has
            self.wrist_command = self.nullspace.wrist_command(measured)
            if a.nullspace_comp == "plate":
                # ...and its own deflection under load, from the measured plate pose
                self.wrist_command = self.nullspace.plate_command(self.wrist_command, self.measured_wrist_pose(), 1.0 / self.hz)
        else:
            self.wrist_command = self.wrist_target

        grip = [1.0 if self.gripper_closed else -1.0] if self.has_gripper else []
        action = self.robot.create_action_vector(
            {"right": self.target, "right_gripper": np.concatenate([self.wrist_command / self.wrist_limits, grip])}
        )

        recording = self.writer.recording
        if recording:
            state = self.env.sim.get_state().flatten()
            obs_before = self.obs
        self.obs, reward, _, _ = self.env.step(action)
        if recording:
            self.writer.add(
                state, action, reward, obs_before, self.obs, self.stage, self.mode, self.wrist.stiffness_scale
            )
        success = bool(self.env._check_success())
        if success and not self._announced_success:
            print("task success")
            self._announced_success = True
        self.success_any |= success and recording
        self.ticks += 1
        if self.print_forces and self.ticks % max(1, round(0.5 * self.hz)) == 0:
            self.print_force_lines()
        texts = None
        if self.viewer is not None:
            rows = self.force_lines()
            font, pos = mujoco.mjtFontScale.mjFONTSCALE_150, mujoco.mjtGridPos.mjGRID_BOTTOMLEFT
            texts = [(font, pos, "\n".join(r[0] for r in rows), "\n".join(r[1] for r in rows))]
        if self.viewer is not None and not self.viewer.sync(self.env.sim, texts):
            print("viewer closed")
            return False
        return True

    @property
    def elapsed(self):
        """Teleop time since start (s), at the control rate"""
        return self.ticks / self.hz

    def run(self):
        print(f"\n{self.args.env} | input: {self.input.name} | writing to {self.args.out} ({len(self.writer.demo_names)} demos)")
        print(describe_arm_controller(self.env_args["env_kwargs"]["controller_configs"]))
        print("workspace box (base frame, m): " + ", ".join(f"{k} {v:.3f}" for k, v in self.box.items()))
        for k, desc in COMMANDS:
            print(f"  {k:10s} {desc}")
        self.announce_plan()
        period = 1.0 / self.hz
        next_t = time.monotonic()
        try:
            while True:
                keys = self.keys.pop_all() if self.keys is not None else []
                while self.key_script and self.elapsed >= self.key_script[0][0]:
                    keys.append(self.key_script.pop(0)[1])
                if not all(self.handle_key(k) for k in keys):
                    break
                if not self.tick():
                    break
                if self.args.max_seconds is not None and self.elapsed >= self.args.max_seconds:
                    break
                if not self.args.no_realtime:
                    next_t += period
                    delay = next_t - time.monotonic()
                    if delay > 0:
                        time.sleep(delay)
                    elif delay < -0.25:
                        next_t = time.monotonic()  # fell behind (slow sim step): don't try to catch up
        finally:
            self.close()

    def close(self):
        if self.keys is not None:
            self.keys.close()
        self.input.close()
        print(f"\n{len(self.writer.demo_names)} demos in {self.args.out}")
        self.writer.close()
        if self.viewer is not None:
            self.viewer.close()
        self.env.close()


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--env", default="Lift", help="robosuite task (Lift, Stack, PickPlaceCan, NutAssemblySquare, Door, ...)")
    p.add_argument("--input", default="spacemouse", choices=["spacemouse", "keyboard", "fake_sine", "none"])
    p.add_argument("--out", default=None, help="hdf5 file; appended to if it exists (default: ../demos/<env>_<time>.hdf5)")
    p.add_argument("--control-freq", type=float, default=20.0, help="policy / teleop rate (Hz); also the recorded rate")
    p.add_argument("--arm-controller", default="osc", choices=list(ARM_CONTROLLERS),
                   help="arm: robosuite's OSC_POSE, or HSA-UMI's Cartesian impedance law with its gains (arm_controller.py)")
    p.add_argument("--arm-gains-yaml", default=None,
                   help="with --arm-controller impedance: read the gains from HSA-UMI's nuc/config/hsa_controllers.yaml")
    p.add_argument("--max-pos-speed", type=float, default=0.20, help="arm, m/s at full deflection (HSA-UMI: 0.20)")
    p.add_argument("--max-rot-speed", type=float, default=0.15, help="arm, rad/s at full deflection (HSA-UMI: 0.15)")
    p.add_argument("--wrist-pos-speed", type=float, default=0.04, help="wrist, m/s at full deflection")
    p.add_argument("--wrist-rot-speed", type=float, default=0.35, help="wrist, rad/s at full deflection")
    p.add_argument("--axis-map", default="hsa", choices=["hsa", "zup"], help="SpaceMouse axes -> robot (see teleop_math)")
    p.add_argument("--deadzone", type=float, default=0.0)
    p.add_argument("--print-axes", action="store_true", help="print the raw SpaceMouse axes and buttons every tick")
    p.add_argument("--print-forces", action="store_true", help="start with the flange force printout on (f toggles it)")
    p.add_argument("--no-gripper", action="store_true", help="the wrist without a gripper (gripper_types StewartWristNoGripper)")
    p.add_argument("--reseed-distance", type=float, default=0.15, help="m; beyond this the target snaps onto the arm")
    p.add_argument(
        "--nullspace-comp",
        default="plate",
        choices=STRATEGIES,
        help="nullspace mode: feedforward = arm target only; feedback = the wrist also cancels the arm's lag; "
        "plate = feedback, plus the wrist cancels its own deflection under load, from the measured plate pose",
    )
    p.add_argument("--soft-scale", type=float, default=0.3, help="leg stiffness multiplier of the soft wrist (x)")
    p.add_argument("--x-min", type=float, default=0.25)
    p.add_argument("--x-max", type=float, default=0.80)
    p.add_argument("--y-min", type=float, default=-0.40)
    p.add_argument("--y-max", type=float, default=0.40)
    p.add_argument("--z-min", type=float, default=None, help="default: the gripper just touches the table (wrist home)")
    p.add_argument("--z-max", type=float, default=0.60)
    p.add_argument("--floor-margin", type=float, default=0.0, help="m added to the automatic z-min")
    p.add_argument("--plan", default=None, help="CSV, one row per episode; rows advance on save, n skips")
    p.add_argument("--condition", default="", help="free-form tag stored with every episode")
    p.add_argument("--min-steps", type=int, default=10, help="warn when a saved episode is shorter")
    p.add_argument("--no-render", action="store_true")
    p.add_argument("--no-realtime", action="store_true", help="step as fast as possible")
    p.add_argument("--no-terminal-keys", action="store_true", help="ignore the terminal (e.g. when run from a script)")
    p.add_argument("--max-seconds", type=float, default=None, help="quit after this much teleop time")
    p.add_argument("--key-script", default="", help='command keys at teleop times (s since start), e.g. "0.5:c,4:m,8:s,9:q"')
    args = p.parse_args()
    if args.out is None:
        os.makedirs(os.path.join(here, "..", "demos"), exist_ok=True)
        stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        args.out = os.path.normpath(os.path.join(here, "..", "demos", f"{args.env}_{stamp}.hdf5"))
    Teleop(args).run()


if __name__ == "__main__":
    main()
