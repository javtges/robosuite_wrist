"""
Measures how well the nullspace teleop mode keeps the wrist's top plate (and the gripper) still, across the arm's
workspace. Drives the real Teleop loop (teleop_wrist.py) with a simulated SpaceMouse (make_teleop_gifs.SimulatedPuck),
so the input path, the button modes and the arm controller are exactly what an operator gets.

    cd scripts && python -m nullspace.evaluate              # full study, ~5 min on 10 cores -> nullspace/results/
    python -m nullspace.evaluate --quick                     # 8 points, for a smoke test

For every test case (a tool pose + a strategy + a control rate):
    1. Reset the scene, move the cube out of the way, drive the arm (arm mode) to the tool pose, let it settle.
    2. Switch to the strategy's mode: "none" = wrist mode (the arm holds still: the baseline), "feedforward" /
       "feedback" = nullspace mode with that compensation.
    3. Run the excursion sequence on the puck: for each of the 6 wrist axes (robot-frame directions, left button for
       rotations, right button for z, as an operator would): push past the +limit (1.5x the time to reach it, so the
       wrist saturates at its limit, or at a corner of its range when the tool is rotated), hold; push past the -limit,
       hold; push back toward 0, then press h (wrist home), hold. Holds are HOLD_S = 1.5 s.
    4. Track the top plate's pose against its pose at step 2, plus the arm's tcp, the wrist and the joint limits.

Outputs (results/):
    cases.csv               one row per test case, every metric below
    timeseries_<name>.npz   per-tick traces for the center point (all strategies, both rates)
"""

import argparse
import csv
import itertools
import multiprocessing as mp
import os
import sys
import tempfile
import time

import numpy as np
from scipy.spatial.transform import Rotation

HERE = os.path.dirname(os.path.abspath(__file__))
SCRIPTS = os.path.dirname(HERE)
RESULTS = os.path.join(HERE, "results")

# Usability thresholds (see README.md, "What counts as usable")
SETTLED_POS_MM, SETTLED_ROT_DEG = 1.0, 0.5
FINAL_POS_MM, FINAL_ROT_DEG = 0.5, 0.25
WRIST_RANGE_FRAC = 0.95
REACH_POS_MM, REACH_ROT_DEG = 2.0, 1.0
HOLD_S = 1.5  # hold after each push; "settled" errors are taken at the end of the holds

GRID_X = [0.30, 0.39, 0.48, 0.57, 0.66, 0.75]
GRID_Y = [-0.35, -0.175, 0.0, 0.175, 0.35]
GRID_Z = [0.08, 0.22, 0.36, 0.50]
CENTER = (0.48, 0.0, 0.22)
# Tool orientations for the orientation study (at CENTER): rotations about the robot base axes, applied to the
# start orientation (gripper pointing down)
ORIENTATIONS = [("x", a) for a in (-40, -20, 20, 40)] + [("y", a) for a in (-40, -20, 20, 40)] + [
    ("z", a) for a in (-135, -90, -45, 45, 90, 135)
]
STRATEGY_MODES = {"none": "WRIST", "feedforward": "NULLSPACE", "feedback": "NULLSPACE"}


def _make_teleop(control_freq):
    sys.path.insert(0, SCRIPTS)
    import make_teleop_gifs as mg

    return mg.make_teleop(os.path.join(tempfile.mkdtemp(), "unused.hdf5"), control_freq=control_freq)


_TELEOPS = {}


def get_teleop(control_freq):
    if control_freq not in _TELEOPS:
        _TELEOPS[control_freq] = _make_teleop(control_freq)
    return _TELEOPS[control_freq]


def run_case(case):
    """Runs one test case. Returns (metrics dict, timeseries dict or None)."""
    import teleop_wrist as tw
    from nullspace.controller import make_T, pose_error

    t = get_teleop(case["control_freq"])
    t.args.nullspace_comp = case["strategy"] if case["strategy"] != "none" else "feedback"
    hz = t.hz
    env, d, puck = t.env, t.env.sim.data, t.input
    m = env.sim.model._model

    # 1. Fresh scene, cube out of the way, arm mode, wrist home
    t.mode = tw.ARM
    t.reset_scene()
    cube_jnt = m.body(env.cube_body_id).jntadr[0]
    qadr = m.jnt_qposadr[cube_jnt]
    d.qpos[qadr : qadr + 3] = [-1.5, -1.5, 0.03]
    env.sim.forward()
    puck.release()

    start = t.measured_tcp()
    goal_R = Rotation.from_rotvec(start[3:])
    if case["rot_axis"] is not None:
        goal_R = Rotation.from_euler(case["rot_axis"], case["rot_deg"], degrees=True) * goal_R
    goal = np.concatenate([case["pos"], goal_R.as_rotvec()])
    # Move the target there at 0.25 m/s (orientation slerped alongside), then settle
    n = max(int(np.ceil(np.linalg.norm(goal[:3] - start[:3]) / 0.25 * hz)), int(np.ceil(np.radians(abs(case["rot_deg"] or 0)) / 0.5 * hz)), 1)
    R0 = Rotation.from_rotvec(start[3:])
    rel = (goal_R * R0.inv()).as_rotvec()
    for i in range(1, n + 1):
        f = i / n
        t.target = np.concatenate([start[:3] + f * (goal[:3] - start[:3]), (Rotation.from_rotvec(f * rel) * R0).as_rotvec()])
        t.tick()
    # Settle: the arm needs ~3 s after a long move (it ends ~0.4 mm off, held by joint friction)
    arm_dofs = t.robot._ref_joint_vel_indexes
    for k in range(int(6.0 * hz)):
        t.target = goal.copy()
        t.tick()
        if k >= hz and np.abs(d.qvel[arm_dofs]).max() < 2e-3:
            break
    meas = t.measured_tcp()
    reach_pos = np.linalg.norm(meas[:3] - goal[:3]) * 1e3
    reach_rot = np.degrees((Rotation.from_rotvec(meas[3:]) * goal_R.inv()).magnitude())
    metrics = dict(case_id=case["case_id"], study=case["study"], strategy=case["strategy"], control_freq=hz,
                   x=case["pos"][0], y=case["pos"][1], z=case["pos"][2], rot_axis=case["rot_axis"] or "",
                   rot_deg=case["rot_deg"] or 0, reach_pos_mm=reach_pos, reach_rot_deg=reach_rot)  # fmt: skip
    reachable = reach_pos <= REACH_POS_MM and reach_rot <= REACH_ROT_DEG
    metrics["reachable"] = reachable

    # 2. Switch mode
    while t.mode != getattr(tw, STRATEGY_MODES[case["strategy"]]):
        t.toggle_mode()

    top = t.gripper.correct_naming("wrist_top")
    grip = t.gripper.important_sites["grip_site"]
    plate = lambda: make_T(d.get_body_xmat(top), d.get_body_xpos(top))
    tool = lambda: make_T(d.get_site_xmat(grip), d.get_site_xpos(grip))
    P0, G0, tcp0 = plate(), tool(), t.measured_tcp()
    lo, hi = m.jnt_range[[m.joint(j).id for j in t.robot.robot_joints]].T

    rec = dict(t=[], plate_dp=[], plate_drot=[], tool_dp=[], wrist_target=[], wrist_cmd=[], wrist_meas=[],
               tcp_target=[], tcp_meas=[], phase=[])  # fmt: skip
    peak = np.zeros(4)  # plate pos, plate rot, tool pos, tool rot
    min_margin = np.inf
    settled = []  # (plate pos err, rot err) at the end of each hold
    wrist_at_hold = []  # (axis, sign, measured wrist pose) at the end of +/- holds

    def step(n_ticks, phase):
        nonlocal min_margin
        for _ in range(n_ticks):
            t.tick()
            P, G = plate(), tool()
            ep, eg = pose_error(P0, P), pose_error(G0, G)
            peak[:] = np.maximum(peak, [ep[0], ep[1], eg[0], eg[1]])
            q = t.robot._joint_positions
            min_margin = min(min_margin, float(np.min(np.minimum(q - lo, hi - q))))
            rec["t"].append(len(rec["t"]) / hz)
            rec["plate_dp"].append(P0[:3, :3].T @ (P[:3, 3] - P0[:3, 3]))  # in the plate's initial frame
            rec["plate_drot"].append(Rotation.from_matrix(P0[:3, :3].T @ P[:3, :3]).as_rotvec())
            rec["tool_dp"].append(G[:3, 3] - G0[:3, 3])
            rec["wrist_target"].append(t.wrist_target.copy())
            rec["wrist_cmd"].append(t.wrist_command.copy())
            rec["wrist_meas"].append(t.measured_wrist_pose())
            rec["tcp_target"].append(t.target.copy())
            rec["tcp_meas"].append(t.measured_tcp())
            rec["phase"].append(phase)

    limits = t.wrist_limits
    speeds = np.array([t.args.wrist_pos_speed] * 3 + [t.args.wrist_rot_speed] * 3)
    step(int(0.5 * hz), -1)
    returns = []  # plate error after each homing
    for axis in range(6):
        T_lim = limits[axis] / speeds[axis]
        v = np.zeros(6)
        for sign, dur, tag in ((1, 1.5 * T_lim, "+"), (-1, 3.0 * T_lim, "-"), (1, T_lim, "0")):
            v[axis] = sign
            puck.push(v, left=axis >= 3, right=axis == 2)
            step(int(round(dur * hz)), axis)
            puck.release()
            if tag == "0":
                t.handle_key("h")
            step(int(HOLD_S * hz), axis)
            e = pose_error(P0, plate())
            settled.append(e)
            if tag == "0":
                returns.append(e)
            else:
                wrist_at_hold.append((axis, t.measured_wrist_pose()))
    returns = np.array(returns)
    final = (returns[:, 0].max(), returns[:, 1].max())

    settled = np.array(settled)
    # How much of its range the wrist reached on each pushed axis: the largest |coordinate| / limit at the two holds
    reached = []
    for axis in range(6):
        held = np.array([w for a, w in wrist_at_hold if a == axis])
        reached.append(float(np.min(np.max(np.abs(held) / limits, axis=1))))
    metrics.update(
        peak_plate_pos_mm=peak[0] * 1e3, peak_plate_rot_deg=np.degrees(peak[1]),
        peak_tool_pos_mm=peak[2] * 1e3, peak_tool_rot_deg=np.degrees(peak[3]),
        settled_plate_pos_mm=settled[:, 0].max() * 1e3, settled_plate_rot_deg=np.degrees(settled[:, 1].max()),
        final_plate_pos_mm=final[0] * 1e3, final_plate_rot_deg=np.degrees(final[1]),
        tcp_travel_mm=float(np.max(np.linalg.norm(np.array(rec["tcp_meas"])[:, :3] - tcp0[:3], axis=1))) * 1e3,
        wrist_range_frac=min(reached), min_joint_margin_deg=np.degrees(min_margin),
    )  # fmt: skip
    metrics["usable"] = bool(
        reachable
        and metrics["settled_plate_pos_mm"] <= SETTLED_POS_MM
        and metrics["settled_plate_rot_deg"] <= SETTLED_ROT_DEG
        and metrics["final_plate_pos_mm"] <= FINAL_POS_MM
        and metrics["final_plate_rot_deg"] <= FINAL_ROT_DEG
        and metrics["wrist_range_frac"] >= WRIST_RANGE_FRAC
    )
    ts = {k: np.array(v) for k, v in rec.items()} if case.get("save_timeseries") else None
    return metrics, ts


def build_cases(quick=False):
    cases = []
    xs, ys, zs = (GRID_X[::2], GRID_Y[::2], GRID_Z[::3]) if quick else (GRID_X, GRID_Y, GRID_Z)
    for (x, y, z), s in itertools.product(itertools.product(xs, ys, zs), STRATEGY_MODES):
        cases.append(dict(study="grid", pos=np.array([x, y, z]), rot_axis=None, rot_deg=None, strategy=s, control_freq=20.0))
    if not quick:
        for (ax, deg), s in itertools.product(ORIENTATIONS, STRATEGY_MODES):
            cases.append(dict(study="orientation", pos=np.array(CENTER), rot_axis=ax, rot_deg=deg, strategy=s, control_freq=20.0))
    for s, hz in itertools.product(STRATEGY_MODES, (20.0, 50.0)):
        cases.append(dict(study="center", pos=np.array(CENTER), rot_axis=None, rot_deg=None, strategy=s, control_freq=hz,
                          save_timeseries=True))  # fmt: skip
    for i, c in enumerate(cases):
        c["case_id"] = i
    return cases


def _worker(case):
    import warnings

    warnings.filterwarnings("ignore")
    try:
        return run_case(case), case
    except Exception as e:  # keep the sweep going; report the failure
        return (dict(case_id=case["case_id"], error=repr(e)), None), case


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--quick", action="store_true")
    p.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    args = p.parse_args()
    os.makedirs(RESULTS, exist_ok=True)
    cases = build_cases(args.quick)
    print(f"{len(cases)} test cases on {args.workers} workers")
    t0 = time.time()
    rows = []
    with mp.get_context("spawn").Pool(args.workers) as pool:
        for i, ((metrics, ts), case) in enumerate(pool.imap_unordered(_worker, cases)):
            rows.append(metrics)
            if "error" in metrics:
                print(f"  case {case['case_id']} FAILED: {metrics['error']}")
            if ts is not None:
                np.savez(os.path.join(RESULTS, f"timeseries_{case['strategy']}_{int(case['control_freq'])}hz.npz"), **ts)
            if (i + 1) % 20 == 0 or i + 1 == len(cases):
                print(f"  {i + 1}/{len(cases)} done ({time.time() - t0:.0f} s)")
    rows.sort(key=lambda r: r["case_id"])
    keys = list(next(r for r in rows if "error" not in r).keys())
    with open(os.path.join(RESULTS, "cases.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys + ["error"], extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: (f"{v:.6g}" if isinstance(v, float) else v) for k, v in r.items()})
    print(f"wrote {os.path.relpath(os.path.join(RESULTS, 'cases.csv'), SCRIPTS)} ({time.time() - t0:.0f} s)")


if __name__ == "__main__":
    sys.path.insert(0, SCRIPTS)
    main()
