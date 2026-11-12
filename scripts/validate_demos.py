"""
Checks a teleop hdf5 file (teleop_wrist.py) and replays its demos.

    python validate_demos.py ../demos/Lift_<time>.hdf5              # structure + open-loop action replay of every demo
    python validate_demos.py FILE --demo demo_0 --no-replay          # structure only, one demo
    mjpython validate_demos.py FILE --demo demo_0 --render           # watch the replay (macOS needs mjpython)

Replay: rebuilds the env from the file's env_args, loads each demo's model_file and first state, restores the
gripper's internal action, the arm controller's null-space posture and the per-step wrist stiffness, then steps the recorded actions and compares the
resulting MuJoCo states with the recorded ones. It prints the largest gripper (eef) position error. That error is small
but not exactly zero: MuJoCo's solver warm start is not part of the saved state, so the first step differs slightly.
"""

import argparse
import json
import time

import h5py
import numpy as np

import robosuite as suite
from teleop_wrist import PassiveViewer

REQUIRED = ["states", "actions", "rewards", "dones", "teleop/stage", "teleop/mode", "teleop/wrist_stiffness_scale"]


def check_structure(f, names):
    data = f["data"]
    env_args = json.loads(data.attrs["env_args"])
    problems = []
    for k in ("env", "env_info", "total"):
        if k not in data.attrs:
            problems.append(f"data.attrs[{k!r}] missing")
    total = 0
    for name in names:
        g = data[name]
        n = int(g.attrs["num_samples"])
        total += n
        for k in REQUIRED:
            if k not in g or len(g[k]) != n:
                problems.append(f"{name}/{k} missing or not {n} long")
        for grp in ("obs", "next_obs"):
            if grp not in g or not len(g[grp]):
                problems.append(f"{name}/{grp} missing")
            else:
                problems += [f"{name}/{grp}/{k} is not {n} long" for k in g[grp] if len(g[grp][k]) != n]
        if "model_file" not in g.attrs:
            problems.append(f"{name}: model_file missing")
        if "dones" in g and g["dones"][-1] != 1:
            problems.append(f"{name}: dones[-1] != 1")
    if len(names) == len(data.keys()) and total != int(data.attrs["total"]):
        problems.append(f"data.attrs['total'] = {data.attrs['total']}, but the demos hold {total} steps")
    return env_args, problems


def summarize(f, names):
    for name in names:
        g = f["data"][name]
        a = g.attrs
        modes = sorted(set(g["teleop/mode"][()].tolist()))
        scales = sorted(set(np.round(g["teleop/wrist_stiffness_scale"][()], 3).tolist()))
        print(
            f"  {name}: {a['num_samples']} steps, {a['duration_s']:.1f} s, success {bool(a['success'])} "
            f"(any {bool(a['success_any'])}), input {a['input']}, stages 0..{g['teleop/stage'][()].max()}, "
            f"modes {[('arm', 'wrist', 'nullspace')[m] for m in modes]}, "
            f"wrist stiffness x{scales}, plan {a['plan_row']}, condition {a['condition']!r}"
        )


def replay(env, g, viewer):
    robot = env.robots[0]
    gripper = robot.gripper["right"]
    env.reset()
    # Demos recorded in one session share one model: reload only when it changes (a new model needs a new viewer)
    if getattr(env, "_loaded_model_file", None) != g.attrs["model_file"]:
        env.reset_from_xml_string(env.edit_model_xml(g.attrs["model_file"]))
        env._loaded_model_file = g.attrs["model_file"]
    env.sim.reset()
    states = g["states"][()]
    env.sim.set_state_from_flattened(states[0])
    env.sim.forward()
    gripper.gripper.current_action = np.array(json.loads(g.attrs["gripper_current_action"]))
    robot.part_controllers["right"].update_initial_joints(json.loads(g.attrs["controller_initial_joint"]))
    actions, scales = g["actions"][()], g["teleop/wrist_stiffness_scale"][()]
    eef = g["next_obs/robot0_eef_pos"][()]
    max_state_err = max_eef_err = 0.0
    period = 1.0 / env.control_freq
    for t, action in enumerate(actions):
        gripper.wrist.stiffness_scale = scales[t]
        obs, _, _, _ = env.step(action)
        max_eef_err = max(max_eef_err, float(np.linalg.norm(obs["robot0_eef_pos"] - eef[t])))
        if t + 1 < len(states):
            max_state_err = max(max_state_err, float(np.abs(env.sim.get_state().flatten() - states[t + 1]).max()))
        if viewer is not None:
            if not viewer.sync(env.sim):
                break
            time.sleep(period)
    gripper.wrist.stiffness_scale = 1.0
    return max_state_err, max_eef_err, bool(env._check_success())


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("file")
    p.add_argument("--demo", default=None, help="one demo (e.g. demo_0); default: all")
    p.add_argument("--no-replay", action="store_true")
    p.add_argument("--render", action="store_true")
    p.add_argument("--tolerance-mm", type=float, default=1.0, help="largest eef replay error that passes")
    args = p.parse_args()

    with h5py.File(args.file, "r") as f:
        names = sorted(f["data"].keys(), key=lambda k: int(k.split("_")[1]))
        if args.demo is not None:
            names = [args.demo]
        env_args, problems = check_structure(f, names)
        kw = env_args["env_kwargs"]
        print(
            f"{args.file}: {env_args['env_name']}, robots {kw['robots']}, control_freq {kw['control_freq']} Hz, "
            f"arm {kw['controller_configs']['body_parts']['right']['type']} "
            f"input_type {kw['controller_configs']['body_parts']['right']['input_type']}, "
            f"{len(f['data'].keys())} demos, {f['data'].attrs['total']} steps"
        )
        summarize(f, names)
        print("structure: OK" if not problems else "structure: PROBLEMS\n  " + "\n  ".join(problems))
        ok = not problems
        if not args.no_replay and names:
            env = suite.make(env_args["env_name"], **kw)
            viewer = PassiveViewer() if args.render else None
            for name in names:
                g = f["data"][name]
                if any(pr.startswith(name + ":") or pr.startswith(name + "/") for pr in problems):
                    print(f"  replay {name}: skipped (structure problems)")
                    continue
                state_err, eef_err, success = replay(env, g, viewer)
                passed = eef_err * 1e3 <= args.tolerance_mm and success == bool(g.attrs["success"])
                ok &= passed
                print(
                    f"  replay {name}: max eef error {eef_err * 1e3:.3f} mm, max state error {state_err:.2e}, "
                    f"success {success} (recorded {bool(g.attrs['success'])}): {'OK' if passed else 'FAIL'}"
                )
            if viewer is not None:
                viewer.close()
            env.close()
    print("VALIDATION OK" if ok else "VALIDATION FAILED")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
