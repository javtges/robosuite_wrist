"""
Scripted expert for KeyDoor: insert the key (KeyOracle), turn it to unlock, push the door open.

    cd scripts
    python -m key_insertion.door_oracle                          # viewer, home wrist, no perception error
    python -m key_insertion.door_oracle --wrist soft --dyaw 3     # soft wrist, keyhole believed 3 deg off
    python -m key_insertion.door_oracle --no-render --episodes 5  # headless, one summary line per episode
    python -m key_insertion.door_oracle --arm hsa --wrist stiff   # the robot's arm law and gains

Phases, after KeyOracle's approach / align / insert / hold:
    turn    the key's tip frame is rotated about the keyhole axis at `turn_speed` (deg/s) until the cylinder passes
            the env's unlock angle, or `turn_overrun` past the cylinder's range ("cannot turn": the key is not seated,
            or the pins hold). The tip's commanded position stays at the seated depth.
    push    the arm's target is the measured flange moved `push_lead` into the door along the keyhole axis (a
            force-style push: 12 mm through the robot law's clip is its 20 N ceiling), orientation following the lock,
            until the door passes the env's open angle ("opened") or `push_timeout` s pass with the door still shut
            ("door stuck").

The keyhole frame follows the door (lock_site is in the cylinder), so every target is recomputed against the lock's
current pose: once the door swings, the target swings with it.
"""

import argparse
import os
import sys
import time

import numpy as np
from scipy.spatial.transform import Rotation

import robosuite as suite
from robosuite.controllers import load_composite_controller_config

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from arm_controller import apply_arm_controller  # noqa: E402
from hsa_robots import DEFAULT_ROBOT, ROBOTS, robot_key, robot_name  # noqa: E402
from key_insertion.oracle import IMPEDANCE_KP, WRIST_PRESETS, KeyOracle, make_T  # noqa: E402


def make_door_env(control_freq=20.0, arm="osc", robot=None, arm_overrides=None, offscreen=False, seed=None, **env_kwargs):
    """KeyDoor with an absolute-pose arm controller, as key_insertion.oracle.make_env builds KeyInsertion"""
    robot = robot_name(robot or DEFAULT_ROBOT)
    cfg = load_composite_controller_config(robot=robot)
    cfg["body_parts"]["right"].update(input_type="absolute", input_ref_frame="base")
    if arm == "impedance":
        cfg["body_parts"]["right"].update(type="CARTESIAN_IMPEDANCE", kp=IMPEDANCE_KP, damping_ratio=1.0)
    elif arm == "hsa":
        apply_arm_controller(cfg, "impedance", robot=robot_key(robot))
    else:
        assert arm == "osc", f"Unknown arm controller {arm}"
    if arm_overrides:
        cfg["body_parts"]["right"].update(arm_overrides)
    return suite.make(
        "KeyDoor",
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


class DoorOracle(KeyOracle):
    """KeyOracle plus turning the key and pushing the door"""

    APPROACH, ALIGN, INSERT, HOLD, TURN, PUSH, DONE = range(7)
    PHASES = ("approach", "align", "insert", "hold", "turn", "push", "done")
    RETRACT = -1  # KeyOracle's retract phase is not used here

    def __init__(self, env, turn_speed=45.0, turn_overrun=10.0, push_lead=0.012, push_timeout=6.0, **kw):
        super().__init__(env, retract=False, **kw)
        self.turn_speed = np.radians(turn_speed)
        self.turn_overrun = np.radians(turn_overrun)
        self.push_lead, self.push_timeout = push_lead, push_timeout
        self.turn_angle = 0.0

    def reset(self):
        super().reset()
        self.turn_angle = 0.0
        # The approach also turns the tool from pointing down to pointing at the door: give the rotation its time
        rot = Rotation.from_rotvec(self.approach_goal[3:]) * Rotation.from_rotvec(self.start_target[3:]).inv()
        self.approach_time = max(self.approach_time, rot.magnitude() / 0.3, 3.0)

    def seated(self):
        return bool(self.env.key_seated())

    def _enter(self, phase):
        super()._enter(phase)
        if phase == self.INSERT:
            # The approach turned the tool from pointing down to pointing at the door, which changes how the wrist sags
            # under the key and plate: re-measure where the tip is before aiming it, as a camera on the tool would
            self.measure_tip_offset()

    def tip_goal(self, height):
        """World pose of the key's tip @height above the believed keyhole, turned by the current turn angle"""
        L = self.believed_lock_T()
        Rz = Rotation.from_euler("z", self.turn_angle).as_matrix()
        return L @ make_T(Rz @ self.R_tip_in_lock, [0, 0, height])

    def next_target(self):
        env = self.env
        if self.phase < self.TURN:
            target = super().next_target()
            if self.phase == self.DONE and self.result == "seated":  # KeyOracle's hold finished: carry on
                self.result = None
                self._enter(self.TURN)
            return target
        self.t_phase += self.dt
        if self.phase == self.TURN:
            self.turn_angle = min(self.turn_angle + self.turn_speed * self.dt, env.cylinder_range + self.turn_overrun)
            # Turn to the cylinder's stop, not just past the unlock angle: a cylinder that coasts on to its stop while
            # the arm holds the key short of it is a twist the arm's rotational spring applies through the blade
            if env.unlocked() and env.cylinder_angle() >= env.cylinder_range - np.radians(3.0):
                self._enter(self.PUSH)
            elif self.turn_angle >= env.cylinder_range + self.turn_overrun - 1e-9 and self.t_phase > 1.0:
                self.result = "cannot turn"
                self._enter(self.DONE)
            elif np.linalg.norm(self.flange_force()) > self.max_force:
                self.result = "jammed"
                self._enter(self.DONE)
            return self.tcp_target_for_tip(self.tip_goal(self.insert_height))
        if self.phase == self.PUSH:
            self.turn_angle = env.cylinder_angle()  # follow the cylinder: push with no twist on the blade
            if env._check_success():
                self.result = "opened"
                self._enter(self.DONE)
            elif self.t_phase > self.push_timeout and env.door_angle() < np.radians(2.0):
                self.result = "door stuck"
                self._enter(self.DONE)
            elif np.linalg.norm(self.flange_force()) > self.max_force:
                self.result = "jammed"
                self._enter(self.DONE)
            return self.push_target()
        return self.push_target()

    def push_target(self):
        """
        A force-style push: the arm's target is the measured tcp moved `push_lead` into the door along the keyhole
        axis, with the orientation of the (turning, swinging) lock frame. The lead through the controller's clip sets
        the push force (12 mm -> the 20 N ceiling on the robot's law) wherever the door has swung to, and nothing
        accumulates sideways while the arm's yaw lags the door, which a target computed from the tip's goal would.
        """
        p, R = self.env.lock_pose()
        tcp_T = self.site_T(self.gripper.important_sites["tcp"])
        p0, R0 = self.robot.composite_controller.get_controller_base_pose(controller_name="right")
        goal = self.tip_goal(self.insert_height) @ self.T_tip_tcp  # orientation from the lock frame and the turn
        pos = tcp_T[:3, 3] - self.push_lead * R[:, 2]
        return np.concatenate([R0.T @ (pos - p0), Rotation.from_matrix(R0.T @ goal[:3, :3]).as_rotvec()])

    def log_row(self):
        row = super().log_row()
        row.update(
            door_angle=self.env.door_angle(),
            cylinder_angle=self.env.cylinder_angle(),
            bolt=self.env.bolt_retraction(),
            unlocked=bool(self.env.unlocked()),
            turn_angle=self.turn_angle,
        )
        return row


def run_door_episode(env, oracle, max_seconds=30.0, on_step=None, reset=True):
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
    ph = log["phase"]
    f = np.linalg.norm(log["flange_force"], axis=1)
    def peak(p):
        m = ph == p
        return f[m].max() if m.any() else 0.0
    return (
        f"{result:12s} depth {log['depth'][-1] * 1e3:5.1f} mm, cylinder {np.degrees(log['cylinder_angle'][-1]):5.1f} deg, "
        f"door {np.degrees(log['door_angle'][-1]):5.1f} deg | peak flange: insert {peak(DoorOracle.INSERT):5.1f} N, "
        f"turn {peak(DoorOracle.TURN):5.1f} N, push {peak(DoorOracle.PUSH):5.1f} N | {log['t'][-1]:.1f} s"
    )


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--wrist", default="home", help="stiff / home / soft, or a normalized z in [-1, 1]")
    p.add_argument("--stiffness-scale", type=float, default=1.0)
    p.add_argument("--dx", type=float, default=0.0, help="perception error across the slot (mm)")
    p.add_argument("--dy", type=float, default=0.0, help="... along the slot (mm)")
    p.add_argument("--dyaw", type=float, default=0.0, help="... in yaw (deg)")
    p.add_argument("--arm", default="osc", choices=["osc", "impedance", "hsa"])
    p.add_argument("--robot", default=DEFAULT_ROBOT, choices=sorted(ROBOTS))
    p.add_argument("--speed", type=float, default=10.0, help="insertion speed (mm/s)")
    p.add_argument("--turn-speed", type=float, default=45.0, help="deg/s")
    p.add_argument("--push-lead", type=float, default=12.0, help="mm the target leads the flange into the door (force through the clip)")
    p.add_argument("--max-force", type=float, default=40.0, help="flange force (N) at which the oracle gives up (inf with --arm hsa)")
    p.add_argument("--door-mass-lb", type=float, default=3.0)
    p.add_argument("--fixed-door", action="store_true", help="no door pose randomization")
    p.add_argument("--episodes", type=int, default=1)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--control-freq", type=float, default=20.0)
    p.add_argument("--no-render", action="store_true")
    p.add_argument("--slowdown", type=float, default=1.0, help="viewer: play this many times slower than real time")
    args = p.parse_args()

    wrist_z = WRIST_PRESETS[args.wrist] if args.wrist in WRIST_PRESETS else float(args.wrist)
    kw = dict(door_mass=args.door_mass_lb * 0.45359237)
    if args.fixed_door:
        kw.update(lock_x_range=(0, 0), lock_y_range=(0, 0), lock_yaw_range=(0, 0))
    env = make_door_env(args.control_freq, arm=args.arm, robot=args.robot, seed=args.seed, **kw)
    max_force = np.inf if args.arm == "hsa" and args.max_force == 40.0 else args.max_force
    oracle = DoorOracle(
        env,
        wrist_z=wrist_z,
        stiffness_scale=args.stiffness_scale,
        perception_error=(args.dx * 1e-3, args.dy * 1e-3, np.radians(args.dyaw)),
        speed=args.speed * 1e-3,
        turn_speed=args.turn_speed,
        push_lead=args.push_lead * 1e-3,
        max_force=max_force,
    )
    viewer = None
    if not args.no_render:
        from teleop_wrist import PassiveViewer

        viewer = PassiveViewer()
        import mujoco

        def on_step(env_, o, row):
            lines = [
                f"phase {o.PHASES[o.phase]}   depth {row['depth'] * 1e3:5.1f} mm   cylinder {np.degrees(row['cylinder_angle']):5.1f} deg   door {np.degrees(row['door_angle']):5.1f} deg",
                f"flange force {np.linalg.norm(row['flange_force']):5.1f} N   unlocked {row['unlocked']}",
            ]
            texts = [(mujoco.mjtFontScale.mjFONTSCALE_150, mujoco.mjtGridPos.mjGRID_BOTTOMLEFT, "\n".join(lines), "")]
            time.sleep(args.slowdown / env_.control_freq)
            return viewer.sync(env_.sim, texts)
    else:
        on_step = None
    try:
        for ep in range(args.episodes):
            result, log = run_door_episode(env, oracle, on_step=on_step)
            print(f"episode {ep}: {summarize(result, log)}")
    finally:
        if viewer is not None:
            viewer.close()


if __name__ == "__main__":
    main()
