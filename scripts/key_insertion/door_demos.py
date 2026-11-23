"""
Door demonstrations from the scheduled expert, in the robomimic hdf5 that the teleop writes (demo_writer.py), so a
policy trains on them exactly as on teleop demos. The 12-dim action's wrist slot carries the stiffness schedule at
every step (wrist z: +1 extended / soft, -1 contracted / stiff): that trace is the supervision for a stiffness head.

    cd scripts
    python -m key_insertion.door_demos --episodes 4 --out ../demos/door_smoke.hdf5          # try it
    python validate_demos.py ../demos/door_smoke.hdf5                                       # structure + replay
    python -m key_insertion.door_demos --episodes 200 --shard 0/4 --out ../demos/door_s0.hdf5   # one of four shards
    python -m key_insertion.door_demos --episodes 200 --shard 1/4 --out ../demos/door_s1.hdf5   # (run in parallel)
    python -m key_insertion.door_demos --episodes 4 --cameras wrist --out ../demos/door_cam_smoke.hdf5        # visuomotor
    python -m key_insertion.door_demos --episodes 4 --cameras wrist doorview --out ../demos/door_cam2_smoke.hdf5

Each episode: a random door pose (+-2 cm, +-5 deg), an aiming error drawn inside the soft wrist's tolerance
(dx +-1.5 mm, dyaw +-3 deg), a lock friction from --lock-frictions (normal and stiff locks, so the turn needs the
switch), and the schedule (default soft_insert_home_turn: extended to insert and seat, home to turn and push). The
expert inserts, turns to the stop, pushes with a 12 mm lead. Episodes that do not open the door are kept with
success=False unless --successes-only.

Per-episode attrs beyond the teleop's: schedule, dx_mm, dyaw_deg, cylinder_friction, result, n_switches. Observation
keys are KeyDoor's (lock pose, key tip in the lock, plunger, flange wrench, door / cylinder / bolt / unlocked, the
robot's tcp and eef poses, gripper qpos = fingers + wrist pose + leg offsets).

--cameras adds image observations: 'wrist' is HSA-UMI's end-effector camera (key_insertion/ee_camera.py; it also puts
the measured key mount on the wrist, as that camera needs), any other name is one of KeyDoor's scene cameras
(doorview, aimed at the door and keyhole from the latch side, is the one made for this; also frontview, agentview,
sideview, birdview, lockview, robot0_robotview). Images are stored in obs/ and next_obs/ as
<camera>_image (N, H, W, 3) uint8, upright (row 0 at the top, as robomimic's env wrapper returns them), gzip-compressed,
at --camera-size. The file's env_args then carry the camera settings, and data.attrs["ee_camera"] holds the camera and
key configs that ee_camera.enable() must reproduce before building the env (validate_demos.py does this).
"""

import argparse
import dataclasses
import json
import os
import sys

import numpy as np

import robosuite as suite
from robosuite.controllers import load_composite_controller_config

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from arm_controller import apply_arm_controller  # noqa: E402
from arm_impedance.door_study import PHASES, SCHEDULES, door_phase  # noqa: E402
from arm_impedance.wrist_study import WristScheduler  # noqa: E402
from demo_writer import DemoWriter  # noqa: E402
from hsa_robots import DEFAULT_ROBOT, ROBOTS, robot_key, robot_name  # noqa: E402
from key_insertion import ee_camera  # noqa: E402  (after robosuite: its MUJOCO_GL default then has no effect)
from key_insertion.door_oracle import DoorOracle  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))


def camera_names(cameras):
    """--cameras -> the model's camera names ('wrist' -> the end-effector camera)"""
    return [ee_camera.CAMERA if c == "wrist" else c for c in cameras]


def make_env_args(robot, control_freq, cylinder_friction, cameras=(), camera_size=(120, 160)):
    """robomimic env_args: everything validate_demos.py / a rollout needs to rebuild the env (rendering off unless
    @cameras: then offscreen, with those cameras' images in the observations)"""
    robot = robot_name(robot)
    cfg = load_composite_controller_config(robot=robot)
    cfg["body_parts"]["right"].update(input_type="absolute", input_ref_frame="base")
    apply_arm_controller(cfg, "impedance", robot=robot_key(robot))
    env_kwargs = dict(
        robots=[robot],
        controller_configs=cfg,
        control_freq=control_freq,
        cylinder_friction=float(cylinder_friction),
        use_camera_obs=False,
        has_renderer=False,
        has_offscreen_renderer=False,
        ignore_done=True,
        hard_reset=False,
    )
    if cameras:
        names = camera_names(cameras)
        env_kwargs.update(use_camera_obs=True, has_offscreen_renderer=True, camera_names=names,
                          camera_heights=[camera_size[0]] * len(names), camera_widths=[camera_size[1]] * len(names))  # fmt: skip
    return {"env_name": "KeyDoor", "type": 1, "env_version": suite.__version__, "env_kwargs": env_kwargs}


def upright(obs):
    """robosuite renders row 0 at the bottom; store images the way robomimic's env wrapper returns them"""
    return {k: (v[::-1].copy() if k.endswith("_image") else v) for k, v in obs.items()}


def record_episode(env, writer, schedule, dx, dyaw, attrs, max_seconds=40.0, hook_cls=WristScheduler):
    """One expert episode into the writer. Returns (result, steps)."""
    obs = env.reset()
    # after the reset: robosuite rebuilds the controllers on every reset, so take the refs now, not before
    robot = env.robots[0]
    gripper = robot.gripper["right"]
    ctrl = robot.part_controllers["right"]
    oracle = DoorOracle(env, wrist_z=SCHEDULES[schedule]["free"][0], perception_error=(dx * 1e-3, 0.0, np.radians(dyaw)), max_force=np.inf)
    hook = hook_cls(schedule, schedules=SCHEDULES, phase_fn=door_phase, order=PHASES)
    # the oracle's reset() steps the env for 0.5 s to settle the wrist; do it before recording starts
    oracle.reset()
    obs = env._get_observations(force_update=True)
    writer.start(
        dict(
            attrs,
            schedule=schedule,
            dx_mm=float(dx),
            dyaw_deg=float(dyaw),
            input="door_oracle",
            condition=f"{schedule}, lock {attrs['cylinder_friction']:g} N m",
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
        obs, reward, _, _ = env.step(action)
        row = oracle.log_row()
        hook(env, oracle, row)
        writer.add(state, action, reward, upright(obs_before), upright(obs), oracle.phase, 0, gripper.wrist.stiffness_scale)
        steps += 1
    result = oracle.result or "timeout"
    writer._attrs.update(result=result, n_switches=hook.n_switches, switch_tool_mm=round(hook.switch_tool_mm, 2))
    writer.save(success=result == "opened", success_any=result == "opened")
    return result, steps


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", required=True)
    p.add_argument("--episodes", type=int, default=50, help="total episodes of the plan (this shard records its share)")
    p.add_argument("--shard", default="0/1", help="i/n: record episodes i, i+n, i+2n, ... of the plan")
    p.add_argument("--schedule", default="soft_insert_home_turn", choices=list(SCHEDULES))
    p.add_argument("--lock-frictions", type=float, nargs="+", default=[0.15, 0.4], help="cylinder friction (N m), cycled over episodes")
    p.add_argument("--max-dx", type=float, default=1.5, help="aiming error across the slot, uniform +- (mm)")
    p.add_argument("--max-dyaw", type=float, default=3.0, help="aiming error in yaw, uniform +- (deg)")
    p.add_argument("--robot", default=DEFAULT_ROBOT, choices=sorted(ROBOTS))
    p.add_argument("--control-freq", type=float, default=20.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--successes-only", action="store_true")
    p.add_argument("--cameras", nargs="+", default=[], help="image observations: 'wrist' (end-effector camera) and/or scene camera names")
    p.add_argument("--camera-size", type=int, nargs=2, default=[120, 160], metavar=("H", "W"), help="image size (default 120 160, the camera's 4:3)")
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
                                    key=dataclasses.asdict(ee_camera.StewartWristKey.key_config)))  # fmt: skip
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    envs, writers = {}, {}
    print(f"shard {i}/{n}: {len(mine)} of {args.episodes} episodes -> {args.out}", flush=True)
    opened = 0
    try:
        for c in mine:
            fr = c["friction"]
            if fr not in envs:
                env_args = make_env_args(args.robot, args.control_freq, fr, args.cameras, args.camera_size)
                envs[fr] = suite.make(env_args["env_name"], **env_args["env_kwargs"])
                # one hdf5 per lock friction: the file's env_args must match every demo in it
                path = args.out if len(args.lock_frictions) == 1 else args.out.replace(".hdf5", f"_lock{fr:g}.hdf5")
                writers[fr] = DemoWriter(path, env_args, dataclasses.asdict(envs[fr].robots[0].gripper["right"].wrist.config))
                if "wrist" in args.cameras:
                    writers[fr].data.attrs["ee_camera"] = cam_attrs
            env = envs[fr]
            env.rng = np.random.default_rng(c["seed"])
            result, steps = record_episode(env, writers[fr], args.schedule, c["dx"], c["dyaw"], dict(cylinder_friction=fr, plan_index=c["k"]))
            if args.successes_only and result != "opened":
                writers[fr].delete_last()
            opened += result == "opened"
            print(f"  episode {c['k']:4d}: lock {fr:g} N m, dx {c['dx']:+.2f} mm, dyaw {c['dyaw']:+.1f} deg -> {result:12s} {steps} steps", flush=True)
    finally:
        for w in writers.values():
            w.close()
    print(f"done: {opened}/{len(mine)} opened")


if __name__ == "__main__":
    main()
