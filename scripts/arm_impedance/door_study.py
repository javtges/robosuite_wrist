"""
The wrist-schedule study on KeyDoor: insert, turn, push, with the wrist's stiffness fixed or switched by task phase,
over aiming errors and the things that make a real door hard. Robot's arm law and FR3 gains throughout.

    cd scripts
    python -m arm_impedance.door_study             # 490 episodes, 4 workers at low priority (~40 min on a laptop)
    python -m arm_impedance.door_study --workers 2  # gentler still; Ctrl-C stops, rerunning resumes from the checkpoint
    python -m arm_impedance.door_study --study friction
    python -m arm_impedance.door_figures           # -> arm_impedance/figures/door_*.png

Studies:
    schedule   8 wrist settings x 15 aiming errors (dx +-1.5 mm, dyaw +-4 deg), mentor's wrist model
    schedule_hard  the same on a stiff lock (cylinder friction 0.4 N m), where the soft wrist alone cannot turn the key
    calib      the 3 fixed settings and the 2 main schedules with the interim calibration (measured_v0)
    friction   cylinder friction 0.05 / 0.15 / 0.4 / 0.8 N m (what the key feels turning), 3 fixed wrists, 3 errors
    slot       slot wall friction 0.2 / 0.4 / 0.8 (key-on-brass), same
    mass       door 1.5 / 3 / 6 / 10 lb, same
    hinge      hinge friction 0.1 / 0.3 / 1 / 2 N m (a stiff or sticky hinge), same
    random     random door pose (+-2 cm, +-5 deg), 8 settings x 5 seeds

Phases for the schedules: free -> search (< 3 mm) -> slide -> seat -> turn -> push. A schedule gives (wrist z,
stiffness scale) per phase; switching uses the extension mechanism with the plate-travel compensation of wrist_study.

Metrics per episode: result (opened / stuck / cannot turn / door stuck / jammed / timeout), the phase reached, peak
flange force in insert, turn and push, peak lateral force on the key, time from contact to open.
"""

import argparse
import csv
import multiprocessing as mp
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from arm_impedance.wrist_calibration import WRIST_PRESETS  # noqa: E402
from arm_impedance.wrist_study import HOME, SOFT, SOFT_SCALE, STIFF, WristScheduler  # noqa: E402
from key_insertion.door_oracle import DoorOracle, make_door_env, run_door_episode  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results_door")
FIXED = dict(lock_x_range=(0, 0), lock_y_range=(0, 0), lock_yaw_range=(0, 0))

PHASES = ("free", "search", "slide", "seat", "turn", "push")


def door_phase(oracle, row):
    if oracle.phase < DoorOracle.INSERT:
        return "free"
    if oracle.phase == DoorOracle.TURN:
        return "turn"
    if oracle.phase >= DoorOracle.PUSH:
        return "push"
    if oracle.phase == DoorOracle.HOLD or row["depth"] >= 0.024:
        return "seat"
    return "search" if row["depth"] < 0.003 else "slide"


def _sched(**per_phase):
    d = {p: (HOME, 1.0) for p in PHASES}
    d.update(per_phase)
    return d


SCHEDULES = {
    "fixed_stiff": {p: (STIFF, 1.0) for p in PHASES},
    "fixed_home": {p: (HOME, 1.0) for p in PHASES},
    "fixed_soft": {p: (SOFT, 1.0) for p in PHASES},
    # soft to find the slot and seat, stiff to turn and push (the switch happens with the key seated and the pins free)
    "soft_insert_stiff_turn": _sched(free=(STIFF, 1.0), search=(SOFT, 1.0), slide=(SOFT, 1.0), seat=(SOFT, 1.0), turn=(STIFF, 1.0), push=(STIFF, 1.0)),
    "soft_insert_home_turn": _sched(free=(HOME, 1.0), search=(SOFT, 1.0), slide=(SOFT, 1.0), seat=(SOFT, 1.0), turn=(HOME, 1.0), push=(HOME, 1.0)),
    # soft through the turn, stiff only to push
    "soft_until_push": _sched(free=(SOFT, 1.0), search=(SOFT, 1.0), slide=(SOFT, 1.0), seat=(SOFT, 1.0), turn=(SOFT, 1.0), push=(STIFF, 1.0)),
    # the idealised switch (stiffness only, no plate motion): soft to insert, home to turn and push
    "scale_soft_insert": _sched(search=(HOME, SOFT_SCALE), slide=(HOME, SOFT_SCALE), seat=(HOME, SOFT_SCALE)),
    "scale_soft_until_push": _sched(search=(HOME, SOFT_SCALE), slide=(HOME, SOFT_SCALE), seat=(HOME, SOFT_SCALE), turn=(HOME, SOFT_SCALE)),
}  # fmt: skip
MECHANISM = {n: ("scale" if n.startswith("scale") else "fixed" if n.startswith("fixed") else "extension") for n in SCHEDULES}
FIXED_THREE = ("fixed_stiff", "fixed_home", "fixed_soft")

ERRORS = [(dx, 0.0, dyaw) for dx in (-1.5, 0.0, 1.5) for dyaw in (-4.0, -2.0, 0.0, 2.0, 4.0)]
SMALL_ERRORS = [(0.0, 0.0, dyaw) for dyaw in (0.0, 2.0, 4.0)]

_ENVS = {}


def _env(env_kwargs, stiffness, fixed, seed):
    key = (tuple(sorted(env_kwargs.items())), tuple(sorted((stiffness or {}).items())), fixed)
    if key not in _ENVS:
        _ENVS[key] = make_door_env(arm="hsa", stiffness=stiffness, **(FIXED if fixed else {}), **env_kwargs)
    env = _ENVS[key]
    env.rng = np.random.default_rng(seed)
    return env


def run_case(case):
    env = _env(case["env_kwargs"], case.get("stiffness"), case["fixed_lock"], case["seed"])
    env.reset()
    sched = SCHEDULES[case["schedule"]]
    oracle = DoorOracle(
        env,
        wrist_z=sched["free"][0],
        perception_error=(case["dx_mm"] * 1e-3, 0.0, np.radians(case["dyaw_deg"])),
        max_force=np.inf,
    )
    hook = WristScheduler(case["schedule"], schedules=SCHEDULES, phase_fn=door_phase, order=PHASES)
    t0 = time.time()
    try:
        result, log = run_door_episode(env, oracle, max_seconds=40.0, on_step=hook, reset=False)
    except Exception as e:  # noqa: BLE001
        return dict(case, result="error", error=repr(e))
    ph = log["phase"]
    f = np.linalg.norm(log["flange_force"], axis=1)
    mom = np.linalg.norm(log["flange_moment"], axis=1)
    peak = lambda p: float(f[ph == p].max()) if (ph == p).any() else 0.0  # noqa: E731
    peak_m = lambda p: float(mom[ph == p].max()) if (ph == p).any() else 0.0  # noqa: E731
    reached = DoorOracle.PHASES[int(ph[ph != DoorOracle.DONE].max())] if (ph != DoorOracle.DONE).any() else "none"
    contact = np.flatnonzero((ph >= DoorOracle.INSERT) & (np.linalg.norm(log["key_force"], axis=1) > 0.5))
    opened = np.flatnonzero(log["door_angle"] >= env.open_angle)
    row = dict(
        case,
        result=result,
        success=result == "opened",
        reached=reached,
        depth_mm=float(log["depth"][-1] * 1e3),
        cylinder_deg=float(np.degrees(log["cylinder_angle"][-1])),
        door_deg=float(np.degrees(log["door_angle"][-1])),
        peak_insert_N=peak(DoorOracle.INSERT),
        peak_turn_N=peak(DoorOracle.TURN),
        peak_push_N=peak(DoorOracle.PUSH),
        # turning a key loads the flange with a moment, not a force: the lock's friction shows up here
        peak_turn_Nm=peak_m(DoorOracle.TURN),
        peak_insert_Nm=peak_m(DoorOracle.INSERT),
        peak_push_Nm=peak_m(DoorOracle.PUSH),
        peak_flange_N=float(f[ph >= DoorOracle.INSERT].max()) if (ph >= DoorOracle.INSERT).any() else 0.0,
        peak_lateral_N=float(np.linalg.norm(log["key_force"][:, :2], axis=1).max()),
        contact_to_open_s=float(log["t"][opened[0]] - log["t"][contact[0]]) if len(opened) and len(contact) else float("nan"),
        n_switches=hook.n_switches,
        switch_tool_mm=round(hook.switch_tool_mm, 2),
        wall_s=time.time() - t0,
    )
    return row


def _case(study, label, schedule, dx=0.0, dyaw=0.0, env_kwargs=None, stiffness=None, fixed=True, seed=0, factor=""):
    return dict(
        study=study, label=label, schedule=schedule, mechanism=MECHANISM[schedule], dx_mm=dx, dyaw_deg=dyaw,
        env_kwargs=dict(env_kwargs or {}), stiffness=stiffness, wrist_calib="measured_v0" if stiffness else "default",
        fixed_lock=fixed, seed=seed, factor=factor,
    )  # fmt: skip


def cases(quick=False):
    out = []
    errors = ERRORS if not quick else SMALL_ERRORS
    for name in SCHEDULES:
        for dx, _, dyaw in errors:
            out.append(_case("schedule", name, name, dx, dyaw))
    for name in SCHEDULES:  # a stiff lock (0.4 N m to turn): where the soft wrist cannot turn the key on its own
        for dx, _, dyaw in errors:
            out.append(_case("schedule_hard", name, name, dx, dyaw, env_kwargs={"cylinder_friction": 0.4}, factor="cylinder_friction=0.4"))
    calib = WRIST_PRESETS["measured_v0"]
    for name in FIXED_THREE + ("soft_insert_stiff_turn", "soft_until_push"):
        for dx, _, dyaw in errors:
            out.append(_case("calib", name, name, dx, dyaw, stiffness=calib))
    sweeps = {
        "friction": ("cylinder_friction", (0.05, 0.15, 0.4, 0.8)),
        "slot": ("lock_friction", (0.2, 0.4, 0.8)),
        "mass": ("door_mass", tuple(lb * 0.45359237 for lb in (1.5, 3.0, 6.0, 10.0))),
        "hinge": ("hinge_friction", (0.1, 0.3, 1.0, 2.0)),
    }
    for study, (key, values) in sweeps.items():
        for v in values:
            for name in FIXED_THREE:
                for dx, _, dyaw in SMALL_ERRORS:
                    out.append(_case(study, name, name, dx, dyaw, env_kwargs={key: v}, factor=f"{key}={v:g}"))
    for name in SCHEDULES:
        for seed in range(2 if quick else 5):
            out.append(_case("random", name, name, fixed=False, seed=seed))
    return out


COLS = "| setting | episodes | opened | stuck at insert | cannot turn | door stuck | timeout | peak insert (N, med) | peak turn (N) | peak turn (N m) | peak push (N) | contact→open (s, med) |"


def _num(r, k):
    v = r.get(k)
    try:
        return float(v) if v not in ("", None) else float("nan")
    except (TypeError, ValueError):
        return float("nan")


def _table(rows, key="label"):
    lines = [COLS, "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for label in dict.fromkeys(r[key] for r in rows):
        w = [r for r in rows if r[key] == label]
        n = lambda res: sum(r["result"] == res for r in w)  # noqa: E731
        t = [_num(r, "contact_to_open_s") for r in w if r["success"] in (True, "True") and np.isfinite(_num(r, "contact_to_open_s"))]
        med = lambda k: np.nanmedian([_num(r, k) for r in w]) if any(np.isfinite(_num(r, k)) for r in w) else float("nan")  # noqa: E731
        lines.append(
            f"| {label} | {len(w)} | {n('opened')} ({100 * n('opened') / len(w):.0f}%) | {n('stuck')} | {n('cannot turn')} | {n('door stuck')} | {n('timeout') + n('jammed')} | "
            f"{med('peak_insert_N'):.1f} | {med('peak_turn_N'):.1f} | {med('peak_turn_Nm'):.2f} | {med('peak_push_N'):.1f} | "
            f"{np.median(t) if t else float('nan'):.1f} |"
        )
    return lines


def write_summary(rows, path):
    rows = [r for r in rows if r["result"] != "error"]
    lines = ["# KeyDoor: wrist settings, schedules and door factors", "", "Generated by `python -m arm_impedance.door_study`. Robot's arm law and FR3 gains; the expert inserts, turns to the stop, pushes with a 12 mm lead.", ""]
    for study, title in (
        ("schedule", "schedule: 15 aiming errors (dx ±1.5 mm × dyaw ±4°), mentor's wrist model, cylinder friction 0.15 N m"),
        ("schedule_hard", "schedule_hard: the same on a stiff lock (cylinder friction 0.4 N m)"),
        ("calib", "calib: the schedule study with the interim calibration (measured_v0)"),
        ("random", "random door poses, no aiming error"),
    ):
        sel = [r for r in rows if r["study"] == study]
        if sel:
            lines += [f"## {title}", ""] + _table(sel) + [""]
    for study, title in (("friction", "cylinder friction (N m)"), ("slot", "slot wall friction"), ("mass", "door mass (kg)"), ("hinge", "hinge friction (N m)")):
        sel = [r for r in rows if r["study"] == study]
        if not sel:
            continue
        lines += [f"## {title}: 3 fixed wrists × yaw 0 / 2 / 4°", ""]
        for factor in dict.fromkeys(r["factor"] for r in sel):
            lines += [f"**{factor}**", ""] + _table([r for r in sel if r["factor"] == factor]) + [""]
    with open(path, "w") as f:
        f.write("\n".join(lines))


def case_id(c):
    """A stable name for a case, for the checkpoint"""
    return f"{c['study']}|{c['factor']}|{c['schedule']}|{c['wrist_calib']}|{c['dx_mm']:g}|{c['dyaw_deg']:g}|{int(c['fixed_lock'])}|{c['seed']}"


FIELDS_SKIP = ("env_kwargs", "stiffness")


def _write(rows, path):
    rows = sorted(rows, key=lambda r: (ORDER[r["study"]], r["factor"], SORDER[r["schedule"]], r["dx_mm"], r["dyaw_deg"], r["seed"]))
    fields = [k for k in dict.fromkeys(k for r in rows for k in r) if k not in FIELDS_SKIP]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: (f"{v:.6g}" if isinstance(v, float) else v) for k, v in r.items()})


ORDER = {s: i for i, s in enumerate(("schedule", "schedule_hard", "calib", "friction", "slot", "mass", "hinge", "random"))}
SORDER = {n: i for i, n in enumerate(SCHEDULES)}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--quick", action="store_true")
    p.add_argument("--study", default=None)
    p.add_argument("--workers", type=int, default=4, help="parallel episodes; 4 leaves a laptop usable (20 does not)")
    p.add_argument("--nice", type=int, default=10, help="process priority of the workers (0 = normal, 19 = lowest)")
    p.add_argument("--restart", action="store_true", help="ignore the checkpoint and rerun everything")
    p.add_argument("--restart-study", default=None, help="drop one study from the checkpoint and rerun it")
    args = p.parse_args()
    os.makedirs(RESULTS, exist_ok=True)
    ckpt = os.path.join(RESULTS, "episodes.csv")
    todo = cases(args.quick)
    if args.study:
        todo = [c for c in todo if c["study"] == args.study]
    done = {}
    if os.path.exists(ckpt) and not args.restart:
        for r in csv.DictReader(open(ckpt)):
            for k, v in r.items():  # back to numbers / bools, so resumed rows sort and summarize like fresh ones
                if v in ("True", "False"):
                    r[k] = v == "True"
                elif k not in ("study", "label", "schedule", "mechanism", "result", "reached", "factor", "wrist_calib", "error", "case_id"):
                    try:
                        r[k] = float(v)
                    except (TypeError, ValueError):
                        pass
            done[r["case_id"]] = r
        if args.restart_study:  # rerun one study, keep the rest
            done = {k: r for k, r in done.items() if r["study"] != args.restart_study}
        todo = [c for c in todo if case_id(c) not in done]
        print(f"checkpoint: {len(done)} episodes already done, {len(todo)} to go", flush=True)
    print(f"{len(todo)} episodes on {args.workers} workers (nice {args.nice}); Ctrl-C stops, rerun resumes", flush=True)
    os.nice(args.nice)  # the pool's workers inherit it
    t0 = time.time()
    rows = list(done.values())
    try:
        with mp.get_context("spawn").Pool(args.workers) as pool:
            for i, row in enumerate(pool.imap_unordered(run_case, todo, chunksize=1)):
                row["case_id"] = case_id(row)
                rows.append(row)
                if (i + 1) % 5 == 0 or i + 1 == len(todo):
                    _write(rows, ckpt)  # checkpoint every few episodes
                    print(f"  {i + 1}/{len(todo)} ({time.time() - t0:.0f} s)", flush=True)
    except KeyboardInterrupt:
        _write(rows, ckpt)
        print(f"\nstopped; {len(rows)} episodes saved to {ckpt}, rerun to resume")
        return
    _write(rows, ckpt)
    rows = [r for r in rows if r.get("result") != "error"]
    write_summary(rows, os.path.join(RESULTS, "summary.md"))
    print(open(os.path.join(RESULTS, "summary.md")).read())


if __name__ == "__main__":
    main()
