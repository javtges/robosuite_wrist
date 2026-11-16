"""
Scripted expert (oracle) for KeyInsertion: the Panda + Stewart wrist + rigid key pushes the key into the lock.

    cd scripts
    mjpython -m key_insertion.oracle                         # watch it (macOS; plain python on Linux)
    mjpython -m key_insertion.oracle --wrist soft --dx 2     # soft wrist, keyhole believed 2 mm off in x
    python -m key_insertion.oracle --no-render --episodes 5  # headless, prints a summary per episode

The oracle knows the lock's pose (plus whatever perception error --dx / --dy / --dyaw add) and nothing else: it does
not use force feedback to steer. It moves the key's tip above the believed keyhole, aligns the blade with the slot,
then lowers the arm's target straight down at --speed until the key is seated (success), the flange force passes
--max-force (jammed: it stops pushing), or the target is --overrun below the seated depth (it gives up). The wrist is
held at a fixed pose throughout (--wrist), so it acts as a passive spring between the arm and the key: whatever lateral
or angular error the chamfer has to correct is taken up by the wrist (and by the arm's own impedance). How stiff that
spring is, is what --wrist / --stiffness-scale / --k-contracted / --k-extended change.

Everything here runs in MuJoCo: forces are MuJoCo's contact and interaction forces (env.key_contact_wrench,
WristGripperModel.flange_wrench), not modelled outside the simulator.
"""

import argparse
import dataclasses
import os
import sys
import time

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

import robosuite as suite
from robosuite.controllers import load_composite_controller_config
from robosuite.models.grippers import StewartWristKey

from arm_controller import apply_arm_controller
from hsa_robots import DEFAULT_ROBOT, ROBOTS, robot_key, robot_name

ROBOT = None  # see make_env(robot=...): hsa_robots.DEFAULT_ROBOT (fr3) unless told otherwise
# Wrist poses (normalized z: -1 fully contracted = stiffest legs, +1 fully extended = softest legs)
WRIST_PRESETS = {"stiff": -1.0, "home": 0.0, "soft": 1.0}


# Diagonal Cartesian impedance for the arm (arm="impedance"): N/m for position, N*m/rad for orientation
IMPEDANCE_KP = [20000.0] * 3 + [300.0] * 3


def make_env(
    control_freq=20.0,
    stiffness=None,
    offscreen=False,
    arm="osc",
    arm_kp=None,
    seed=None,
    robot=None,
    arm_overrides=None,
    **env_kwargs,
):
    """
    KeyInsertion with an absolute-pose arm controller. @arm: "osc" (the robosuite OSC the teleop uses, stiffness
    kp * Lambda), "impedance" (CARTESIAN_IMPEDANCE, diagonal stiffness IMPEDANCE_KP, Lambda-shaped damping) or "hsa"
    (the robot's law and gains: arm_controller.apply_arm_controller, with the streaming chunk reference).
    @stiffness: dict of StewartPlatformConfig overrides (e.g. leg_stiffness_extended=3000.0), applied to the wrist
    before the env is built. @arm_kp: overrides the controller's kp (OSC: robosuite default 150; impedance: the
    stiffness itself). @arm_overrides: dict of part-controller keys set last (e.g. translational_clip=0.02).
    """
    robot = robot_name(robot or DEFAULT_ROBOT)
    cfg = load_composite_controller_config(robot=robot)
    cfg["body_parts"]["right"].update(input_type="absolute", input_ref_frame="base")
    if arm == "impedance":
        cfg["body_parts"]["right"].update(type="CARTESIAN_IMPEDANCE", kp=IMPEDANCE_KP, damping_ratio=1.0)
    elif arm == "hsa":
        apply_arm_controller(cfg, "impedance", robot=robot_key(robot))
    else:
        assert arm == "osc", f"Unknown arm controller {arm}"
    if arm_kp is not None:
        cfg["body_parts"]["right"]["kp"] = arm_kp
    if arm_overrides:
        cfg["body_parts"]["right"].update(arm_overrides)
    default_wrist = StewartWristKey.wrist_config
    if stiffness:
        StewartWristKey.wrist_config = dataclasses.replace(default_wrist, **stiffness)
    try:
        env = suite.make(
            "KeyInsertion",
            robots=robot,
            controller_configs=cfg,
            control_freq=control_freq,
            has_renderer=False,
            has_offscreen_renderer=offscreen,
            use_camera_obs=False,
            ignore_done=True,
            hard_reset=False,
            seed=seed,
            **env_kwargs,
        )
    finally:
        StewartWristKey.wrist_config = default_wrist
    return env


def make_T(R, p):
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = R, p
    return T


class KeyOracle:
    """
    Args:
        env (KeyInsertion): the env (absolute-pose OSC, see make_env)
        wrist_z (float): normalized wrist z command held throughout, -1 (contracted, stiff) to 1 (extended, soft)
        stiffness_scale (float): multiplier on every leg's stiffness (StewartPlatformWrist.stiffness_scale)
        perception_error (3-array): (dx, dy, dyaw) added to the true keyhole pose (m, m, rad), lock frame
        speed (float): insertion speed of the arm's target (m/s)
        hover (float): height of the key's tip above the keyhole before inserting (m)
        overrun (float): how far below the seated depth the target may go before giving up (m)
        max_force (float): flange force (N) at which it stops pushing (jammed)
        hold (float): how long to keep pushing once seated (s)
        retract (bool): pull the key back out afterwards
    """

    APPROACH, ALIGN, INSERT, HOLD, RETRACT, DONE = range(6)
    PHASES = ("approach", "align", "insert", "hold", "retract", "done")

    def __init__(
        self,
        env,
        wrist_z=0.0,
        stiffness_scale=1.0,
        perception_error=(0.0, 0.0, 0.0),
        speed=0.01,
        hover=0.02,
        overrun=0.015,
        max_force=40.0,
        hold=0.5,
        retract=False,
    ):
        self.env = env
        self.robot = env.robots[0]
        self.gripper = self.robot.gripper["right"]
        self.wrist = self.gripper.wrist
        self.wrist_z = wrist_z
        self.stiffness_scale = stiffness_scale
        self.perception_error = np.asarray(perception_error, dtype=float)
        self.speed, self.hover, self.overrun = speed, hover, overrun
        self.max_force, self.hold, self.retract = max_force, hold, retract
        self.dt = 1.0 / env.control_freq

    # ---- frames ------------------------------------------------------------------------------------------------------

    def site_T(self, name):
        d = self.env.sim.data
        return make_T(d.get_site_xmat(name), d.get_site_xpos(name))

    def believed_lock_T(self):
        """The keyhole frame the oracle believes in: the true one, offset by the perception error (lock frame)"""
        p, R = self.env.lock_pose()
        dx, dy, dyaw = self.perception_error
        return make_T(R @ Rotation.from_euler("z", dyaw).as_matrix(), p + R @ [dx, dy, 0.0])

    @property
    def wrist_action(self):
        return np.array([0, 0, self.wrist_z, 0, 0, 0], dtype=float)

    def tcp_target_for_tip(self, T_tip):
        """Base-frame [pos, rotvec] for the arm's tcp that puts the key's tip at world pose @T_tip"""
        p0, R0 = self.robot.composite_controller.get_controller_base_pose(controller_name="right")
        T_tcp = T_tip @ self.T_tip_tcp
        return np.concatenate([R0.T @ (T_tcp[:3, 3] - p0), Rotation.from_matrix(R0.T @ T_tcp[:3, :3]).as_rotvec()])

    def tip_goal(self, height):
        """World pose of the key's tip @height above the believed keyhole, blade aligned with the slot"""
        L = self.believed_lock_T()
        return L @ make_T(self.R_tip_in_lock, [0, 0, height])

    # ---- episode -----------------------------------------------------------------------------------------------------

    def reset(self):
        """Call after env.reset(): settles the wrist at its pose, and plans the approach"""
        self.wrist.stiffness_scale = self.stiffness_scale
        self.phase, self.t_phase = self.APPROACH, 0.0
        # Let the wrist settle at its commanded pose, with the arm holding where it is
        tcp_site = self.gripper.important_sites["tcp"]
        p0, R0 = self.robot.composite_controller.get_controller_base_pose(controller_name="right")
        T = self.site_T(tcp_site)
        hold = np.concatenate([R0.T @ (T[:3, 3] - p0), Rotation.from_matrix(R0.T @ T[:3, :3]).as_rotvec()])
        for _ in range(int(0.5 / self.dt)):
            self.env.step(self.action(hold))
        # The key's tip relative to the tcp, as the wrist (loaded only by the key's weight) holds it now. Frozen from
        # here on: contact deflects the wrist away from this, which is the compliance the task is about.
        self.measure_tip_offset()
        # Blade orientation in the lock frame: tip z down the keyhole, blade width along the slot (either way round,
        # whichever is closer to how the key is held now)
        _, R_now = self.env.key_tip_in_lock()
        candidates = [np.diag([-1.0, 1.0, -1.0]), np.diag([1.0, -1.0, -1.0])]
        self.R_tip_in_lock = max(candidates, key=lambda R: np.trace(R.T @ R_now))
        # Approach: min-jerk from here to above the keyhole
        self.start = self.site_T(tcp_site) @ np.eye(4)
        self.start_target = hold
        goal = self.tcp_target_for_tip(self.tip_goal(self.hover))
        dist = np.linalg.norm(goal[:3] - hold[:3])
        self.approach_time = max(1.5, dist / 0.08)
        self.approach_goal = goal
        self.insert_height = self.hover
        self.seated_at = None
        self.result = None

    def action(self, tcp_target):
        return self.robot.create_action_vector(
            {"right": tcp_target, "right_gripper": self.wrist_action}
        )

    def flange_force(self):
        return self.gripper.flange_wrench(self.env.sim)[1][3:]

    def next_target(self):
        """The arm's tcp target for this step, advancing the phase machine"""
        self.t_phase += self.dt
        seated_depth = self.env.slot_depth
        if self.phase == self.APPROACH:
            s = min(self.t_phase / self.approach_time, 1.0)
            s = 10 * s**3 - 15 * s**4 + 6 * s**5  # min-jerk
            a, b = self.start_target, self.approach_goal
            rot = Slerp([0, 1], Rotation.from_rotvec([a[3:], b[3:]]))(s).as_rotvec()
            target = np.concatenate([a[:3] + s * (b[:3] - a[:3]), rot])
            if self.t_phase >= self.approach_time:
                self._enter(self.ALIGN)
            return target
        if self.phase == self.ALIGN:
            # Hold above the keyhole until the arm has caught up with its target
            if self.t_phase >= 0.5:
                self._enter(self.INSERT)
            return self.tcp_target_for_tip(self.tip_goal(self.hover))
        if self.phase == self.INSERT:
            self.insert_height -= self.speed * self.dt
            force = np.linalg.norm(self.flange_force())
            if self.seated():
                self.seated_at = self.env.sim.data.time
                self._enter(self.HOLD)
            elif force > self.max_force:
                self.result = "jammed"
                self._enter(self.DONE)
            elif self.insert_height < -seated_depth - self.overrun:
                self.result = "stuck"
                self._enter(self.DONE)
            return self.tcp_target_for_tip(self.tip_goal(self.insert_height))
        if self.phase == self.HOLD:
            if self.t_phase >= self.hold:
                self.result = "seated"
                self._enter(self.RETRACT if self.retract else self.DONE)
            return self.tcp_target_for_tip(self.tip_goal(self.insert_height))
        if self.phase == self.RETRACT:
            self.insert_height += self.speed * self.dt
            if self.insert_height >= self.hover:
                self._enter(self.DONE)
            return self.tcp_target_for_tip(self.tip_goal(self.insert_height))
        return self.tcp_target_for_tip(self.tip_goal(self.insert_height))

    def seated(self):
        """Is the key seated? The env's success for KeyInsertion; subclasses with a longer task override this"""
        return bool(self.env._check_success())

    def measure_tip_offset(self):
        """The key's tip relative to the tcp as the wrist holds it now (free, in the current orientation and sag)"""
        tcp_site = self.gripper.important_sites["tcp"]
        self.T_tip_tcp = np.linalg.inv(self.site_T(self.gripper.important_sites["grip_site"])) @ self.site_T(tcp_site)

    def _enter(self, phase):
        self.phase, self.t_phase = phase, 0.0

    @property
    def done(self):
        return self.phase == self.DONE

    def log_row(self):
        """One time step of everything the figures plot"""
        env = self.env
        total, ext = self.gripper.flange_wrench(env.sim)
        f_key, _ = env.key_contact_wrench()
        p, R = env.key_tip_in_lock()
        names = [f"wrist_top_{a}" for a in ("x", "y", "z", "rx", "ry", "rz")]
        wrist = np.array([env.sim.data.get_joint_qpos(self.gripper.correct_naming(n)) for n in names])
        return dict(
            t=env.sim.data.time,
            phase=self.phase,
            depth=env.insertion_depth(),
            target_depth=-self.insert_height,
            lateral=np.linalg.norm(p[:2]),
            tip_x=p[0],
            tip_y=p[1],
            yaw=np.arctan2(R[0, 1], R[1, 1]),  # blade width axis vs the slot's long side, about the lock's z
            tilt=np.arccos(np.clip(-R[2, 2], -1.0, 1.0)),  # key axis vs the keyhole's axis
            plunger=env.plunger_depression(),
            flange_force=ext[3:].copy(),
            flange_moment=ext[:3].copy(),
            key_force=f_key,
            wrist_pose=wrist,
            wrist_deflection=wrist - self.wrist.last_commanded_pose,
            leg_k=self.wrist.leg_stiffness(self.wrist.leg_offsets(self.wrist.last_commanded_pose)),
            success=bool(env._check_success()),
        )


def run_episode(env, oracle, max_seconds=15.0, on_step=None, reset=True):
    """
    Runs the oracle for one episode. Returns (result, log): result is "seated", "jammed", "stuck" or "timeout"; log is a
    dict of stacked arrays, one row per control step (KeyOracle.log_row). @on_step(env, oracle, row) is called after
    every step (e.g. to render); returning False stops the episode.
    """
    if reset:
        env.reset()
    oracle.reset()
    rows = []
    while not oracle.done and env.sim.data.time < max_seconds:
        env.step(oracle.action(oracle.next_target()))
        rows.append(oracle.log_row())
        if on_step is not None and on_step(env, oracle, rows[-1]) is False:
            break
    result = oracle.result or "timeout"
    log = {k: np.array([r[k] for r in rows]) for k in rows[0]} if rows else {}
    return result, log


def summarize(result, log):
    if not log:
        return result
    ins = log["phase"] >= KeyOracle.INSERT
    f = np.linalg.norm(log["flange_force"][ins], axis=1) if ins.any() else np.zeros(1)
    fk = np.linalg.norm(log["key_force"][ins], axis=1) if ins.any() else np.zeros(1)
    t_ins = log["t"][ins]
    return (
        f"{result:8s} depth {log['depth'][-1] * 1e3:5.1f} mm, plunger {log['plunger'].max() * 1e3:4.2f} mm | "
        f"peak force: flange {f.max():5.1f} N, key {fk.max():5.1f} N | max wrist deflection "
        f"{np.abs(log['wrist_deflection'][ins, :3]).max() * 1e3 if ins.any() else 0:4.2f} mm | "
        f"insert {t_ins[-1] - t_ins[0] if len(t_ins) else 0:4.2f} s"
    )


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--wrist", default="home", help="wrist pose held: stiff / home / soft, or a normalized z in [-1, 1]")
    p.add_argument("--stiffness-scale", type=float, default=1.0, help="multiplier on every leg's stiffness")
    p.add_argument("--k-contracted", type=float, default=None, help="leg stiffness (N/m) fully contracted (default 1e5)")
    p.add_argument("--k-extended", type=float, default=None, help="leg stiffness (N/m) fully extended (default 300)")
    p.add_argument("--dx", type=float, default=0.0, help="perception error: keyhole believed this far off in x (mm)")
    p.add_argument("--dy", type=float, default=0.0, help="... in y, along the slot (mm)")
    p.add_argument("--dyaw", type=float, default=0.0, help="... in yaw (deg)")
    p.add_argument("--speed", type=float, default=10.0, help="insertion speed of the arm's target (mm/s)")
    p.add_argument("--max-force", type=float, default=40.0, help="stop pushing above this flange force (N)")
    p.add_argument("--retract", action="store_true", help="pull the key back out after seating")
    p.add_argument("--clearance", type=float, default=None, help="keyhole minus blade, per dimension (mm; env default 0.5)")
    p.add_argument("--chamfer", type=float, default=None, help="keyhole entrance chamfer (mm; env default 2)")
    p.add_argument("--seat-force", type=float, default=None, help="plunger force when seated (N; env default 10)")
    p.add_argument("--arm-kp", type=float, default=None, help="arm OSC stiffness gain (robosuite default 150)")
    p.add_argument("--fixed-lock", action="store_true", help="lock at the table center, not randomized")
    p.add_argument("--episodes", type=int, default=1)
    p.add_argument("--robot", default=DEFAULT_ROBOT, choices=sorted(ROBOTS), help="fr3 or fer (default $HSA_ROBOT or fr3)")
    p.add_argument("--arm", default="osc", choices=["osc", "impedance", "hsa"],
                   help="arm controller: robosuite OSC, the sim-only diagonal impedance, or the robot's HSA law and gains")
    p.add_argument("--clip", type=float, default=None, help="with --arm hsa: translational_clip (mm), i.e. the force ceiling / K")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--control-freq", type=float, default=20.0)
    p.add_argument("--no-render", action="store_true", help="headless")
    p.add_argument("--slowdown", type=float, default=1.0, help="viewer playback: >1 is slower than real time")
    args = p.parse_args()

    wrist_z = WRIST_PRESETS[args.wrist] if args.wrist in WRIST_PRESETS else float(args.wrist)
    stiffness = {}
    if args.k_contracted is not None:
        stiffness["leg_stiffness_contracted"] = args.k_contracted
    if args.k_extended is not None:
        stiffness["leg_stiffness_extended"] = args.k_extended
    env_kwargs = {}
    if args.clearance is not None:
        env_kwargs["clearance"] = args.clearance * 1e-3
    if args.chamfer is not None:
        env_kwargs["chamfer"] = args.chamfer * 1e-3
    if args.seat_force is not None:
        env_kwargs["seat_force"] = args.seat_force
    if args.fixed_lock:
        env_kwargs.update(lock_x_range=(0, 0), lock_y_range=(0, 0), lock_yaw_range=(0, 0))
    overrides = {"translational_clip": args.clip * 1e-3} if args.clip is not None else None
    env = make_env(args.control_freq, stiffness=stiffness, arm=args.arm, arm_kp=args.arm_kp, seed=args.seed, robot=args.robot,
                   arm_overrides=overrides, **env_kwargs)  # fmt: skip
    oracle = KeyOracle(
        env,
        wrist_z=wrist_z,
        stiffness_scale=args.stiffness_scale,
        perception_error=(args.dx * 1e-3, args.dy * 1e-3, np.radians(args.dyaw)),
        speed=args.speed * 1e-3,
        max_force=args.max_force,
        retract=args.retract,
    )
    k = oracle.wrist.leg_stiffness(oracle.wrist.leg_offsets(np.array([0, 0, wrist_z * oracle.wrist.config.z_range, 0, 0, 0])))
    print(f"wrist z {wrist_z:+.2f} (normalized), leg stiffness {k.mean() * args.stiffness_scale:.0f} N/m"
          + (f" ({k.mean():.0f} x{args.stiffness_scale:g})" if args.stiffness_scale != 1 else ""))

    on_step = None
    viewer = None
    if not args.no_render:
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from teleop_wrist import PassiveViewer
        import mujoco

        viewer = PassiveViewer()
        viewer.CAMERA = dict(lookat=[0.0, 0.0, 0.86], distance=0.75, azimuth=125.0, elevation=-20.0)
        clock = {"next": None}

        def on_step(env, oracle, row):
            f = row["flange_force"]
            rows = [
                ("phase", KeyOracle.PHASES[row["phase"]]),
                ("depth / plunger (mm)", f"{row['depth'] * 1e3:6.1f} / {row['plunger'] * 1e3:4.2f}"),
                ("flange force (N)", f"[{f[0]:+6.1f} {f[1]:+6.1f} {f[2]:+6.1f}] |F| {np.linalg.norm(f):5.1f}"),
                ("key contact force (N)", f"{np.linalg.norm(row['key_force']):5.1f}"),
                ("wrist deflection (mm)", f"{np.linalg.norm(row['wrist_deflection'][:3]) * 1e3:5.2f}"),
                ("leg stiffness (N/m)", f"{row['leg_k'].mean():.0f}"),
            ]
            texts = [(mujoco.mjtFontScale.mjFONTSCALE_150, mujoco.mjtGridPos.mjGRID_BOTTOMLEFT,
                      "\n".join(r[0] for r in rows), "\n".join(r[1] for r in rows))]
            if not viewer.sync(env.sim, texts):
                return False
            now = time.monotonic()
            clock["next"] = now if clock["next"] is None else clock["next"] + oracle.dt * args.slowdown
            if clock["next"] > now:
                time.sleep(clock["next"] - now)
            return True

    try:
        for ep in range(args.episodes):
            result, log = run_episode(env, oracle, on_step=on_step)
            print(f"episode {ep}: {summarize(result, log)}")
            if viewer is not None and not viewer.handle.is_running():
                break
    finally:
        if viewer is not None:
            viewer.close()
        env.close()


if __name__ == "__main__":
    main()
