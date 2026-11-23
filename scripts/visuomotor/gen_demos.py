"""
Visuomotor door demonstrations from the scripted expert (key_insertion/door_oracle.py: a pose-waypoint state machine,
approach -> align -> insert -> hold -> turn -> push, force-blind, aiming with an injected perception error). Every
episode records everything a policy might be given (see common.py): the wrist camera, the scene camera (doorview),
the wrist's leg-sensed wrench, the robot's state and the full 12-dim robosuite action. Policies then pick their
inputs at training time, so one dataset serves every observation ablation.

Three wrist behaviours during the demos (--wrist-mode), one dataset each:
    static     the wrist holds one z all episode (--static-z, default 0 = home)
    znoise     z-axis exploration: the wrist switches to a random z in --znoise-range, holds it for a random
               --znoise-hold seconds, switches again; no task logic. Each switch runs at 1 z/s with the arm
               compensating the plate's travel (WristScheduler), as the scheduled switches do
    schedule   the expert's phase schedule (--schedule, default soft_insert_home_turn: soft to insert and seat,
               home to turn and push)
In all three the wrist moves only along z (its stiffness); the other five wrist axes deflect passively.

Run with robosuite_env from scripts/ (one shard per process; see visuomotor/sh/gen_datasets.sh):
    python -m visuomotor.gen_demos --wrist-mode schedule --episodes 100 --shard 0/8 --out ../data/vm/schedule/s0.hdf5

Writes <out stem>_lock<friction>.hdf5 per lock friction (the friction is part of env_args). Failed episodes are kept
with success=False unless --successes-only.
"""

import argparse
import dataclasses
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import robosuite as suite  # noqa: E402

from arm_impedance.door_study import PHASES, SCHEDULES, door_phase  # noqa: E402
from arm_impedance.wrist_study import WristScheduler  # noqa: E402
from demo_writer import DemoWriter  # noqa: E402
from hsa_robots import DEFAULT_ROBOT, ROBOTS  # noqa: E402
from key_insertion import ee_camera  # noqa: E402  (after robosuite: its MUJOCO_GL default then has no effect)
from key_insertion.door_demos import make_env_args  # noqa: E402
from key_insertion.door_oracle import DoorOracle  # noqa: E402
from visuomotor.common import DEMO_WRIST_MODES, obs_extras  # noqa: E402


class ZNoiseScheduler(WristScheduler):
    """WristScheduler that ignores the phase and switches to a random z after a random hold, forever"""

    def __init__(self, rng, z_range=(-1.0, 1.0), hold=(0.3, 1.5), **kw):
        super().__init__("fixed_home", **kw)
        self.rng, self.z_range, self.hold = rng, z_range, hold
        self.next_t = None

    def __call__(self, env, oracle, row):
        if self.next_t is None:
            self.next_t = row["t"] + self.rng.uniform(*self.hold)
        if not self.switching and row["t"] >= self.next_t:
            z = float(self.rng.uniform(*self.z_range))
            if abs(z - oracle.wrist_z) > 0.05:
                self._begin_switch(oracle, row, z)
            self.next_t = row["t"] + self.rng.uniform(*self.hold)
        self._advance(oracle, row)
        return True


def make_hook(args, rng):
    """(hook, initial z) for --wrist-mode"""
    kw = dict(phase_fn=door_phase, order=PHASES)
    if args.wrist_mode == "static":
        sched = {"static": {p: (args.static_z, 1.0) for p in PHASES}}
        return WristScheduler("static", schedules=sched, **kw), args.static_z
    if args.wrist_mode == "znoise":
        z0 = float(rng.uniform(*args.znoise_range))
        return ZNoiseScheduler(rng, tuple(args.znoise_range), tuple(args.znoise_hold), schedules=SCHEDULES, **kw), z0
    return WristScheduler(args.schedule, schedules=SCHEDULES, **kw), SCHEDULES[args.schedule]["free"][0]


def record_episode(env, writer, hook, z0, dx, dyaw, attrs, max_seconds=40.0):
    """One expert episode into the writer (key_insertion/door_demos.record_episode plus the extra observations)"""
    env.reset()
    robot = env.robots[0]
    gripper = robot.gripper["right"]
    ctrl = robot.part_controllers["right"]
    oracle = DoorOracle(env, wrist_z=z0, perception_error=(dx * 1e-3, 0.0, np.radians(dyaw)), max_force=np.inf)
    oracle.reset()  # settles the wrist for 0.5 s before recording starts
    wrist_cmd = np.array([0, 0, oracle.wrist_z, 0, 0, 0], dtype=float)
    obs = obs_extras(env, env._get_observations(force_update=True), wrist_cmd, oracle.phase)
    writer.start(
        dict(
            attrs,
            dx_mm=float(dx),
            dyaw_deg=float(dyaw),
            input="door_oracle",
            plan_row={},
            gripper_current_action=np.asarray(gripper.gripper.current_action).tolist(),
            model_file=env.sim.model.get_xml(),
            controller_initial_joint=np.asarray(ctrl.initial_joint).tolist(),
            **({"controller_state": ctrl.controller_state()} if hasattr(ctrl, "controller_state") else {}),
        )
    )
    steps = 0
    while not oracle.done and env.sim.data.time < max_seconds:
        state = env.sim.get_state().flatten()
        action = oracle.action(oracle.next_target())
        obs_before = obs
        raw, reward, _, _ = env.step(action)
        row = oracle.log_row()
        hook(env, oracle, row)
        obs = obs_extras(env, raw, action[6:12], oracle.phase)
        writer.add(state, action, reward, obs_before, obs, oracle.phase, 0, gripper.wrist.stiffness_scale)
        steps += 1
    result = oracle.result or "timeout"
    writer._attrs.update(result=result, n_switches=hook.n_switches, switch_tool_mm=round(hook.switch_tool_mm, 2))
    writer.save(success=result == "opened", success_any=result == "opened")
    return result, steps


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", required=True, help="output .hdf5 (one file per lock friction: _lock<f> is appended)")
    p.add_argument("--wrist-mode", required=True, choices=DEMO_WRIST_MODES)
    p.add_argument("--static-z", type=float, default=0.0, help="static: the wrist's normalized z (-1 stiff .. +1 soft)")
    p.add_argument("--znoise-range", type=float, nargs=2, default=[-1.0, 1.0], help="znoise: z targets drawn uniformly here")
    p.add_argument("--znoise-hold", type=float, nargs=2, default=[0.3, 1.5], help="znoise: seconds between switches, uniform")
    p.add_argument("--schedule", default="soft_insert_home_turn", choices=list(SCHEDULES), help="schedule: the phase schedule")
    p.add_argument("--episodes", type=int, default=100, help="total episodes of the plan (this shard records its share)")
    p.add_argument("--shard", default="0/1", help="i/n: record episodes i, i+n, i+2n, ... of the plan")
    p.add_argument("--lock-frictions", type=float, nargs="+", default=[0.15, 0.4], help="cylinder friction (N m), cycled over episodes")
    p.add_argument("--max-dx", type=float, default=1.5, help="aiming error across the slot, uniform +- (mm)")
    p.add_argument("--max-dyaw", type=float, default=3.0, help="aiming error in yaw, uniform +- (deg)")
    p.add_argument("--cameras", nargs="+", default=["wrist", "doorview"], help="'wrist' and/or scene camera names")
    p.add_argument("--camera-size", type=int, nargs=2, default=[120, 160], metavar=("H", "W"))
    p.add_argument("--robot", default=DEFAULT_ROBOT, choices=sorted(ROBOTS))
    p.add_argument("--control-freq", type=float, default=20.0)
    p.add_argument("--max-seconds", type=float, default=40.0)
    p.add_argument("--seed", type=int, default=0, help="the plan's seed: shards of one dataset must share it")
    p.add_argument("--successes-only", action="store_true")
    args = p.parse_args()

    i, n = (int(x) for x in args.shard.split("/"))
    rng = np.random.default_rng(args.seed)
    plan = [
        dict(k=k, friction=args.lock_frictions[k % len(args.lock_frictions)], dx=float(rng.uniform(-args.max_dx, args.max_dx)),
             dyaw=float(rng.uniform(-args.max_dyaw, args.max_dyaw)), seed=int(rng.integers(1 << 31)))
        for k in range(args.episodes)
    ]  # fmt: skip
    mine = [c for c in plan if c["k"] % n == i]
    if "wrist" in args.cameras:
        ee_camera.enable()  # class-level: every env built below has the camera and the measured key mount
    cam_attrs = json.dumps(dict(camera=dataclasses.asdict(ee_camera.StewartWristKey.camera_config),
                                key=dataclasses.asdict(ee_camera.StewartWristKey.key_config))) if "wrist" in args.cameras else None  # fmt: skip
    demo_config = json.dumps({k: v for k, v in vars(args).items() if k not in ("out", "shard")})
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    envs, writers = {}, {}
    print(f"[{args.wrist_mode}] shard {i}/{n}: {len(mine)} of {args.episodes} episodes -> {args.out}", flush=True)
    opened, t0 = 0, time.time()
    try:
        for c in mine:
            fr = c["friction"]
            if fr not in envs:
                env_args = make_env_args(args.robot, args.control_freq, fr, args.cameras, args.camera_size)
                envs[fr] = suite.make(env_args["env_name"], **env_args["env_kwargs"])
                path = args.out if len(args.lock_frictions) == 1 else args.out.replace(".hdf5", f"_lock{fr:g}.hdf5")
                writers[fr] = DemoWriter(path, env_args, dataclasses.asdict(envs[fr].robots[0].gripper["right"].wrist.config))
                writers[fr].data.attrs["demo_config"] = demo_config
                if cam_attrs:
                    writers[fr].data.attrs["ee_camera"] = cam_attrs
            env = envs[fr]
            env.rng = np.random.default_rng(c["seed"])
            ep_rng = np.random.default_rng(c["seed"] + 1)
            hook, z0 = make_hook(args, ep_rng)
            attrs = dict(cylinder_friction=fr, plan_index=c["k"], wrist_mode=args.wrist_mode, z0=z0,
                         schedule=args.schedule if args.wrist_mode == "schedule" else args.wrist_mode,
                         condition=f"{args.wrist_mode}, lock {fr:g} N m")  # fmt: skip
            result, steps = record_episode(env, writers[fr], hook, z0, c["dx"], c["dyaw"], attrs, args.max_seconds)
            if args.successes_only and result != "opened":
                writers[fr].delete_last()
            opened += result == "opened"
            print(f"  episode {c['k']:4d}: lock {fr:g} N m, dx {c['dx']:+.2f} mm, dyaw {c['dyaw']:+.1f} deg -> {result:12s} "
                  f"{steps} steps, {hook.n_switches} switches", flush=True)  # fmt: skip
    finally:
        for w in writers.values():
            w.close()
    print(f"done: {opened}/{len(mine)} opened in {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
