"""
Wrist-less (variable) Cartesian impedance study on KeyInsertion: what the arm's own impedance can and cannot do for a
rigidly held key, with the robot's controller law and gains (arm_controller.HSA_GAINS, FR3 by default).

    cd scripts
    python -m arm_impedance.evaluate            # ~900 episodes, a few minutes on 20 cores -> arm_impedance/results/
    python -m arm_impedance.evaluate --quick    # coarser grids
    python -m arm_impedance.make_figures        # -> arm_impedance/figures/*.png

"Wrist-less": the Stewart wrist is held fully contracted (wrist_z = -1, 100 kN/m legs, 100 kN/m lateral and
417 N m/rad in tilt), 50x stiffer than the arm's spring, so the arm's impedance is the only compliance that matters.

Studies (the lock at the table centre, the oracle believes it is offset by the perception error):
    grid        OSC (robosuite's, what the sim teleop used) vs the HSA law: success over (dx, dyaw)
    rot_k       HSA law with the rotational stiffness swept (the damping ratio kept), over the error set
    trans_k     HSA law with the translational stiffness swept, the force ceiling kept at 20 N (clip = 20 N / K)
    ceiling     HSA law with the force ceiling swept through translational_clip (K = 2000 N/m)
    reference   HSA law with the streaming chunk reference vs the goal held per step
    schedule    variable impedance: gains switched by task phase (SCHEDULES), over the error set and random lock poses
    profile     full time series of a few cases for the force-vs-depth figure

results/episodes.csv    one row per episode: study, gains, perception error, result, force and time metrics
results/profiles.npz    time series of the profile episodes
results/summary.md      tables per study
"""

import argparse
import csv
import multiprocessing as mp
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from arm_controller import HSA_GAINS  # noqa: E402
from hsa_robots import DEFAULT_ROBOT  # noqa: E402
from key_insertion.evaluate import FIXED_LOCK, metrics  # noqa: E402
from key_insertion.oracle import KeyOracle, make_env, run_episode  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")
WRIST_Z = -1.0  # fully contracted: the rigid stand-in

G = HSA_GAINS[DEFAULT_ROBOT]
K_T, K_R, D_T, D_R, CLIP_T, CLIP_R = (
    G["translational_stiffness"], G["rotational_stiffness"], G["translational_damping"], G["rotational_damping"],
    G["translational_clip"], G["rotational_clip"],
)  # fmt: skip
ZETA_T, ZETA_R = D_T / (2 * np.sqrt(K_T)), D_R / (2 * np.sqrt(K_R))  # the robot's damping ratios (0.995, 0.286)

# The error set shared by the sweeps: across the slot (x) and yaw, the two the stiff wrist fails on
ERRORS = [(dx, 0.0, dyaw) for dx in (-1.5, -0.75, 0.0, 0.75, 1.5) for dyaw in (-4.0, -2.0, 0.0, 2.0, 4.0)]
PROFILE_ERRORS = {"none": (0.0, 0.0, 0.0), "offset": (1.0, 0.0, 3.0)}


def gains(k_t=K_T, k_r=K_R, clip_t=CLIP_T, clip_r=CLIP_R, zeta_t=ZETA_T, zeta_r=ZETA_R):
    """Controller settings for a translational / rotational stiffness, the damping critical-scaled like the robot's"""
    return dict(
        kp=[k_t] * 3 + [k_r] * 3,
        damping=[2 * zeta_t * np.sqrt(k_t), 2 * zeta_r * np.sqrt(k_r)],
        translational_clip=clip_t,
        rotational_clip=clip_r,
    )


def apply_gains(ctrl, g):
    """Live gain change on a CartesianImpedanceController, as the robot's set_parameters would"""
    ctrl.kp = np.asarray(g["kp"], dtype=float)
    ctrl.damping = ctrl._six(g["damping"])
    ctrl.error_clip = np.array([g["translational_clip"]] * 3 + [g["rotational_clip"]] * 3)


# ---- variable impedance schedules ----------------------------------------------------------------------------------
# A schedule maps the task phase to gains. Phases, from the oracle and the key's depth in the slot:
#   free     approach / align, no contact
#   search   inserting, not yet past the chamfer (depth < 3 mm)
#   slide    inserting, blade in the slot, before the plunger (3 mm <= depth < 24 mm)
#   seat     pushing the plunger (depth >= 24 mm) and holding
SOFT_ROT = 15.0  # N m/rad: 10x softer than the robot's rotation
WRIST_ROT = 1.5  # N m/rad: the soft wrist's own tilt stiffness, 100x softer than the robot's rotation
SCHEDULES = {
    "fixed": {"free": gains(), "search": gains(), "slide": gains(), "seat": gains()},
    "soft_rot_insert": {"free": gains(), "search": gains(k_r=SOFT_ROT), "slide": gains(k_r=SOFT_ROT), "seat": gains()},
    "soft_insert": {"free": gains(), "search": gains(k_t=500.0, k_r=SOFT_ROT, clip_t=0.04), "slide": gains(k_t=500.0, k_r=SOFT_ROT, clip_t=0.04), "seat": gains()},
    "soft_insert_firm_seat": {"free": gains(), "search": gains(k_t=500.0, k_r=SOFT_ROT, clip_t=0.04), "slide": gains(k_t=500.0, k_r=SOFT_ROT, clip_t=0.04), "seat": gains(clip_t=0.02)},
    "gentle_search": {"free": gains(), "search": gains(k_r=SOFT_ROT, clip_t=0.005), "slide": gains(k_r=SOFT_ROT), "seat": gains()},
    "soft_rot_always": {"free": gains(k_r=SOFT_ROT), "search": gains(k_r=SOFT_ROT), "slide": gains(k_r=SOFT_ROT), "seat": gains(k_r=SOFT_ROT)},
    "wrist_rot_insert": {"free": gains(), "search": gains(k_r=WRIST_ROT), "slide": gains(k_r=WRIST_ROT), "seat": gains()},
}  # fmt: skip


def task_phase(oracle, depth):
    if oracle.phase < KeyOracle.INSERT:
        return "free"
    if oracle.phase >= KeyOracle.HOLD or depth >= 0.024:
        return "seat"
    return "search" if depth < 0.003 else "slide"


def controller(env):
    """The arm's part controller. robosuite rebuilds the controllers on every env.reset(): look it up, never cache it"""
    return env.robots[0].part_controllers["right"]


class Scheduler:
    """on_step hook: applies the schedule's gains for the phase the task is in, and records the switches"""

    def __init__(self, schedule):
        self.schedule = schedule
        self.current = None
        self.trace = []

    def __call__(self, env, oracle, row):
        phase = task_phase(oracle, row["depth"])
        if phase != self.current:
            apply_gains(controller(env), self.schedule[phase])
            self.current = phase
        self.trace.append((row["t"], phase))
        return True


# ---- cases -----------------------------------------------------------------------------------------------------------

_ENVS = {}
WRIST_OVERRIDES = {}  # StewartPlatformConfig fields for every env built here (wrist_calibration.WRIST_PRESETS)


def _env(arm, reference, fixed_lock, seed):
    """
    One env per worker process, arm controller, reference mode and lock setting; the gains are set live per episode
    (apply_gains), as the robot's set_parameters would. The random generator is reseeded so each episode depends only
    on its seed.
    """
    key = (arm, reference, fixed_lock, tuple(sorted(WRIST_OVERRIDES.items())))
    if key not in _ENVS:
        overrides = {"reference": reference} if arm == "hsa" else None
        _ENVS[key] = make_env(arm=arm, robot=DEFAULT_ROBOT, arm_overrides=overrides, stiffness=WRIST_OVERRIDES or None, **(FIXED_LOCK if fixed_lock else {}))
    env = _ENVS[key]
    env.rng = np.random.default_rng(seed)
    return env


def run_case(case, on_step=None):
    """One episode. @on_step(env, oracle, row) is called after every step, e.g. to record video."""
    overrides = case.get("overrides") or {}
    env = _env(case["arm"], overrides.get("reference", "chunks"), case["fixed_lock"], case["seed"])
    env.reset()  # rebuilds the controllers, so the gains go on afterwards
    if case["arm"] == "hsa":
        apply_gains(controller(env), gains(**_gains_kwargs(overrides)))
    oracle = KeyOracle(
        env,
        wrist_z=case.get("wrist_z", WRIST_Z),
        perception_error=(case["dx_mm"] * 1e-3, case["dy_mm"] * 1e-3, np.radians(case["dyaw_deg"])),
        # OSC has no force ceiling: the oracle's 40 N stop stands in for the robot's reflex. The HSA law is bounded by
        # its clip, so it runs until the key seats, stalls (target 15 mm past the seat) or times out
        max_force=40.0 if case["arm"] == "osc" else np.inf,
    )
    hooks = [Scheduler(SCHEDULES[case["schedule"]])] if case.get("schedule") else []
    if case.get("wrist_schedule"):
        from arm_impedance.wrist_study import WristScheduler

        hooks.append(WristScheduler(case["wrist_schedule"]))
    hook = hooks[0] if hooks else None
    if on_step is not None:
        hooks.append(on_step)
    combined = (lambda env_, o, row: all(h(env_, o, row) is not False for h in hooks)) if hooks else None
    t0 = time.time()
    try:
        result, log = run_episode(env, oracle, on_step=combined, reset=False, max_seconds=case.get("max_seconds", 15.0))
    except Exception as e:  # noqa: BLE001
        return dict(case, result="error", error=repr(e))
    row = dict(case, **metrics(result, log), wall_s=time.time() - t0)
    row["max_yaw_deg"] = float(np.degrees(np.abs(log["yaw"][log["phase"] >= KeyOracle.INSERT]).max())) if (log["phase"] >= KeyOracle.INSERT).any() else 0.0
    if case.get("keep_log"):
        row["log"] = log
        if hook is not None and hasattr(hook, "trace"):
            row["log"]["schedule_phase"] = np.array([p for _, p in hook.trace])
    for h in hooks:
        if hasattr(h, "summary"):
            row.update(h.summary())
    return row


def _gains_kwargs(overrides):
    return dict(
        k_t=overrides["kp"][0] if "kp" in overrides else K_T,
        k_r=overrides["kp"][3] if "kp" in overrides else K_R,
        clip_t=overrides.get("translational_clip", CLIP_T),
        clip_r=overrides.get("rotational_clip", CLIP_R),
    )


def _case(study, label, arm="hsa", overrides=None, dx=0.0, dy=0.0, dyaw=0.0, **kw):
    g = _gains_kwargs(overrides or {}) if arm == "hsa" else dict(k_t=np.nan, k_r=np.nan, clip_t=np.nan, clip_r=np.nan)
    d = dict(
        study=study, label=label, arm=arm, overrides=overrides, dx_mm=dx, dy_mm=dy, dyaw_deg=dyaw, fixed_lock=True,
        seed=0, k_t=g["k_t"], k_r=g["k_r"], clip_t_mm=g["clip_t"] * 1e3, ceiling_N=g["k_t"] * g["clip_t"],
        schedule="",
    )  # fmt: skip
    d.update(kw)
    return d


def cases(quick=False):
    out = []
    errors = ERRORS if not quick else [e for e in ERRORS if e[0] in (-1.5, 0.0, 1.5) and e[2] in (-4.0, 0.0, 4.0)]

    for name, err in PROFILE_ERRORS.items():
        out.append(_case("profile", "osc", arm="osc", dx=err[0], dy=err[1], dyaw=err[2], keep_log=True, error_case=name))
        out.append(_case("profile", "fixed", overrides=gains(), dx=err[0], dy=err[1], dyaw=err[2], keep_log=True, error_case=name, schedule="fixed"))
        out.append(_case("profile", "soft_rot_insert", overrides=gains(), dx=err[0], dy=err[1], dyaw=err[2], keep_log=True, error_case=name, schedule="soft_rot_insert"))

    n = 5 if quick else 9
    for arm in ("osc", "hsa"):
        for dx in np.linspace(-2.0, 2.0, n):
            for dyaw in np.linspace(-6.0, 6.0, n):
                out.append(_case("grid", arm, arm=arm, overrides=gains() if arm == "hsa" else None, dx=dx, dyaw=dyaw))

    for k_r in (1.5, 5.0, 15.0, 50.0, 150.0, 450.0):
        for dx, dy, dyaw in errors:
            out.append(_case("rot_k", f"Kr {k_r:g}", overrides=gains(k_r=k_r), dx=dx, dy=dy, dyaw=dyaw))

    for k_t in (500.0, 1000.0, 2000.0, 4000.0, 8000.0):
        for dx, dy, dyaw in errors:
            out.append(_case("trans_k", f"Kt {k_t:g}", overrides=gains(k_t=k_t, clip_t=20.0 / k_t), dx=dx, dy=dy, dyaw=dyaw))

    for clip in (0.0025, 0.005, 0.0075, 0.01, 0.015, 0.02, 0.03):
        for dx, dy, dyaw in errors:
            out.append(_case("ceiling", f"{K_T * clip:g} N", overrides=gains(clip_t=clip), dx=dx, dy=dy, dyaw=dyaw))

    for ref in ("chunks", "hold"):
        for dx, dy, dyaw in errors:
            out.append(_case("reference", ref, overrides=dict(gains(), reference=ref), dx=dx, dy=dy, dyaw=dyaw))

    for name in SCHEDULES:
        for dx, dy, dyaw in errors:
            out.append(_case("schedule", name, overrides=gains(), dx=dx, dy=dy, dyaw=dyaw, schedule=name))
        for seed in range(5 if quick else 15):
            out.append(_case("schedule", name, overrides=gains(), schedule=name, fixed_lock=False, seed=seed))
    return out


# ---- output ----------------------------------------------------------------------------------------------------------

COLUMNS = "| label | episodes | seated | stuck | jammed | timeout | peak flange (N, med) | peak lateral (N, med) | push seated (N, med) | contact→seat (s, med) |"


def _table(rows, key="label"):
    lines = [COLUMNS, "|---|---|---|---|---|---|---|---|---|---|"]
    for label in dict.fromkeys(r[key] for r in rows):
        w = [r for r in rows if r[key] == label]
        n = lambda res: sum(r["result"] == res for r in w)  # noqa: E731
        push = [r["hold_push_N"] for r in w if r["success"]]
        seat = [r["contact_to_seat_s"] for r in w if r["success"] and np.isfinite(r["contact_to_seat_s"])]
        lines.append(
            f"| {label} | {len(w)} | {n('seated')} ({100 * n('seated') / len(w):.0f}%) | {n('stuck')} | {n('jammed')} | {n('timeout')} | "
            f"{np.median([r['peak_flange_N'] for r in w]):.1f} | {np.median([r['peak_lateral_N'] for r in w]):.1f} | "
            f"{np.median(push) if push else float('nan'):.1f} | {np.median(seat) if seat else float('nan'):.2f} |"
        )
    return lines


def write_summary(rows, path):
    lines = ["# Wrist-less arm impedance: KeyInsertion oracle results", "", "Generated by `python -m arm_impedance.evaluate`.",
             f"Wrist fully contracted (z = {WRIST_Z}). Robot gains: K ({K_T:g} N/m, {K_R:g} N m/rad), D ({D_T:g}, {D_R:g}), "
             f"clip ({CLIP_T * 1e3:g} mm, {CLIP_R:g} rad) -> {K_T * CLIP_T:g} N ceiling.", ""]  # fmt: skip
    titles = {
        "grid": "grid: OSC vs the HSA law over (dx, dyaw)",
        "rot_k": "rot_k: rotational stiffness (N m/rad), damping ratio kept",
        "trans_k": "trans_k: translational stiffness (N/m), ceiling kept at 20 N",
        "ceiling": "ceiling: force ceiling through translational_clip, K = 2000 N/m",
        "reference": "reference: streaming chunks vs goal held per step",
        "schedule": "schedule: variable impedance by task phase (25 error cases + random lock poses)",
    }
    for study, title in titles.items():
        sel = [r for r in rows if r["study"] == study and r["result"] != "error"]
        if sel:
            lines += [f"## {title}", ""] + _table(sel) + [""]
    sel = [r for r in rows if r["study"] == "schedule" and r["result"] != "error" and r["fixed_lock"]]
    if sel:
        lines += ["### schedule, error cases only", ""] + _table(sel) + [""]
    sel = [r for r in rows if r["study"] == "schedule" and r["result"] != "error" and not r["fixed_lock"]]
    if sel:
        lines += ["### schedule, random lock poses only (no perception error)", ""] + _table(sel) + [""]
    errors = [r for r in rows if r["result"] == "error"]
    if errors:
        lines += [f"{len(errors)} episodes raised an error (see episodes.csv)", ""]
    with open(path, "w") as f:
        f.write("\n".join(lines))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--quick", action="store_true")
    p.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    p.add_argument("--study", default=None, help="run one study only")
    args = p.parse_args()

    todo = cases(args.quick)
    if args.study:
        todo = [c for c in todo if c["study"] == args.study]
    print(f"{len(todo)} episodes on {args.workers} workers")
    t0 = time.time()
    rows = []
    with mp.get_context("spawn").Pool(args.workers) as pool:
        for i, row in enumerate(pool.imap_unordered(run_case, todo, chunksize=2)):
            rows.append(row)
            if (i + 1) % 50 == 0 or i + 1 == len(todo):
                print(f"  {i + 1}/{len(todo)} ({time.time() - t0:.0f} s)")
    order = {s: i for i, s in enumerate(("profile", "grid", "rot_k", "trans_k", "ceiling", "reference", "schedule"))}
    rows.sort(key=lambda r: (order[r["study"]], r["label"], r["dx_mm"], r["dyaw_deg"], r["seed"]))

    os.makedirs(RESULTS, exist_ok=True)
    profiles = {}
    for r in rows:
        log = r.pop("log", None)
        if log is not None:
            for k, v in log.items():
                profiles[f"{r['label']}/{r['error_case']}/{k}"] = v
    np.savez_compressed(os.path.join(RESULTS, "profiles.npz"), **profiles)
    fields = [k for k in dict.fromkeys(k for r in rows for k in r) if k not in ("keep_log", "overrides")]
    with open(os.path.join(RESULTS, "episodes.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: (f"{v:.6g}" if isinstance(v, float) else v) for k, v in r.items()})
    write_summary(rows, os.path.join(RESULTS, "summary.md"))
    print(f"wrote {len(rows)} episodes to {os.path.relpath(RESULTS, os.path.dirname(HERE))}/ in {time.time() - t0:.0f} s")
    print(open(os.path.join(RESULTS, "summary.md")).read())


if __name__ == "__main__":
    main()
