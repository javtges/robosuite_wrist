"""
Runs the KeyInsertion oracle (oracle.py) over wrist stiffnesses and perception errors, and writes the results that
make_figures.py plots:

    cd scripts
    python -m key_insertion.evaluate            # every study, ~800 episodes, a few minutes on a multi-core machine
    python -m key_insertion.evaluate --quick    # coarser grids, ~1 minute
    python -m key_insertion.make_figures

results/episodes.csv    one row per episode: study, wrist z, leg stiffness, perception error, result, force metrics
results/profiles.npz    full time series of the profile episodes (force vs depth plots)
results/summary.md      success rates and forces per study and wrist

Studies (the lock is at the table center with yaw 0 unless noted; the oracle believes it is offset by the error):
    profile     stiff / home / soft wrist, no error and one moderate error (1 mm in x, 3 deg in yaw); time series saved
    grid_yaw    error in x (across the slot) vs yaw, per wrist
    grid_xy     error in x (across the slot) vs y (along it), per wrist
    sweep       wrist z from fully contracted to fully extended, each over the same set of errors
    random      randomized lock poses (the env's own reset randomization), no error, per wrist
"""

import argparse
import csv
import multiprocessing as mp
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from key_insertion.oracle import WRIST_PRESETS, KeyOracle, make_env, run_episode  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")
PROFILE_ERROR = (0.001, 0.0, np.radians(3.0))
FIXED_LOCK = dict(lock_x_range=(0, 0), lock_y_range=(0, 0), lock_yaw_range=(0, 0))

_ENVS = {}


def _env(fixed_lock, seed):
    """
    One env per worker process and lock setting (the oracle's settings don't need a new env), with its random
    generator reseeded, so each episode (initialization noise, lock pose) depends only on its seed
    """
    if fixed_lock not in _ENVS:
        _ENVS[fixed_lock] = make_env(**(FIXED_LOCK if fixed_lock else {}))
    env = _ENVS[fixed_lock]
    env.rng = np.random.default_rng(seed)
    return env


def metrics(result, log):
    """Scalar summary of one episode"""
    ins = log["phase"] >= KeyOracle.INSERT
    f_flange = np.linalg.norm(log["flange_force"], axis=1)
    f_key = np.linalg.norm(log["key_force"], axis=1)
    contact = np.flatnonzero(ins & (f_key > 0.5))
    t_contact = log["t"][contact[0]] if len(contact) else np.nan
    seated = np.flatnonzero(log["success"])
    hold = log["phase"] == KeyOracle.HOLD
    return dict(
        result=result,
        success=result == "seated",
        depth_mm=log["depth"][-1] * 1e3,
        plunger_mm=log["plunger"].max() * 1e3,
        peak_flange_N=f_flange[ins].max() if ins.any() else 0.0,
        peak_key_N=f_key[ins].max() if ins.any() else 0.0,
        peak_lateral_N=np.linalg.norm(log["key_force"][ins, :2], axis=1).max() if ins.any() else 0.0,
        # flange frame +z points toward the key: pushing reads negative, so report the push as positive
        hold_push_N=-log["flange_force"][hold, 2].mean() if hold.any() else np.nan,
        max_wrist_lateral_mm=np.linalg.norm(log["wrist_deflection"][ins, :2], axis=1).max() * 1e3 if ins.any() else 0.0,
        max_wrist_axial_mm=np.abs(log["wrist_deflection"][ins, 2]).max() * 1e3 if ins.any() else 0.0,
        max_tilt_deg=np.degrees(log["tilt"][ins].max()) if ins.any() else 0.0,
        contact_to_seat_s=(log["t"][seated[0]] - t_contact) if len(seated) and len(contact) else np.nan,
        leg_k=float(log["leg_k"][0].mean()),
    )


def run_case(case):
    env = _env(case["fixed_lock"], case["seed"])
    oracle = KeyOracle(
        env,
        wrist_z=case["wrist_z"],
        stiffness_scale=case.get("stiffness_scale", 1.0),
        perception_error=(case["dx_mm"] * 1e-3, case["dy_mm"] * 1e-3, np.radians(case["dyaw_deg"])),
    )
    t0 = time.time()
    try:
        result, log = run_episode(env, oracle)
    except Exception as e:  # noqa: BLE001 - one bad case shouldn't stop the study
        return dict(case, result="error", error=repr(e))
    row = dict(case, **metrics(result, log), wall_s=time.time() - t0)
    if case.get("keep_log"):
        row["log"] = log
    return row


def cases(quick=False):
    out = []
    wrists = list(WRIST_PRESETS.items())
    base = dict(dx_mm=0.0, dy_mm=0.0, dyaw_deg=0.0, fixed_lock=True, seed=0)

    for name, z in wrists:
        out.append(dict(base, study="profile", wrist=name, wrist_z=z, keep_log=True, error_case="none"))
        dx, dy, dyaw = PROFILE_ERROR
        out.append(dict(base, study="profile", wrist=name, wrist_z=z, keep_log=True, error_case="offset",
                        dx_mm=dx * 1e3, dy_mm=dy * 1e3, dyaw_deg=np.degrees(dyaw)))  # fmt: skip

    n = 5 if quick else 9
    dxs = np.linspace(-2.0, 2.0, n)
    for name, z in wrists:
        for dx in dxs:
            for dyaw in np.linspace(-6.0, 6.0, n):
                out.append(dict(base, study="grid_yaw", wrist=name, wrist_z=z, dx_mm=dx, dyaw_deg=dyaw))
            for dy in np.linspace(-2.0, 2.0, n):
                out.append(dict(base, study="grid_xy", wrist=name, wrist_z=z, dx_mm=dx, dy_mm=dy))

    errors = [(dx, 0.0, dyaw) for dx in (-1.5, -0.75, 0.0, 0.75, 1.5) for dyaw in (-4.0, 0.0, 4.0)]
    for z in np.linspace(-1.0, 1.0, 5 if quick else 9):
        for dx, dy, dyaw in errors:
            out.append(dict(base, study="sweep", wrist=f"z{z:+.2f}", wrist_z=z, dx_mm=dx, dy_mm=dy, dyaw_deg=dyaw))

    for name, z in wrists:
        for seed in range(10 if quick else 30):
            out.append(dict(base, study="random", wrist=name, wrist_z=z, fixed_lock=False, seed=seed))
    return out


def write_summary(rows, path):
    lines = ["# KeyInsertion oracle results", "", "Generated by `python -m key_insertion.evaluate`.", ""]
    for study in ("profile", "grid_yaw", "grid_xy", "sweep", "random"):
        sel = [r for r in rows if r["study"] == study and r["result"] != "error"]
        if not sel:
            continue
        lines += [f"## {study}", "", "| wrist | leg k (N/m) | episodes | seated | jammed | stuck | peak flange force (N, median) | push while seated (N, median) |",
                  "|---|---|---|---|---|---|---|---|"]  # fmt: skip
        for wrist in dict.fromkeys(r["wrist"] for r in sel):
            w = [r for r in sel if r["wrist"] == wrist]
            count = lambda res: sum(r["result"] == res for r in w)  # noqa: E731
            push = [r["hold_push_N"] for r in w if r["success"]]
            lines.append(
                f"| {wrist} | {w[0]['leg_k']:.0f} | {len(w)} | {count('seated')} ({100 * count('seated') / len(w):.0f}%) | "
                f"{count('jammed')} | {count('stuck')} | {np.median([r['peak_flange_N'] for r in w]):.1f} | "
                f"{np.median(push) if push else float('nan'):.1f} |"
            )
        lines.append("")
    errors = [r for r in rows if r["result"] == "error"]
    if errors:
        lines += [f"{len(errors)} episodes raised an error (see episodes.csv)", ""]
    with open(path, "w") as f:
        f.write("\n".join(lines))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--quick", action="store_true", help="coarser grids and fewer random poses")
    p.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    args = p.parse_args()

    todo = cases(args.quick)
    print(f"{len(todo)} episodes on {args.workers} workers")
    t0 = time.time()
    rows = []
    with mp.get_context("spawn").Pool(args.workers) as pool:
        for i, row in enumerate(pool.imap_unordered(run_case, todo, chunksize=2)):
            rows.append(row)
            if (i + 1) % 50 == 0 or i + 1 == len(todo):
                print(f"  {i + 1}/{len(todo)} ({time.time() - t0:.0f} s)")
    key = lambda r: (r["study"], r["wrist_z"], r["dx_mm"], r["dy_mm"], r["dyaw_deg"], r["seed"])  # noqa: E731
    rows.sort(key=key)

    os.makedirs(RESULTS, exist_ok=True)
    profiles = {}
    for r in rows:
        log = r.pop("log", None)
        if log is not None:
            for k, v in log.items():
                profiles[f"{r['wrist']}/{r['error_case']}/{k}"] = v
    np.savez_compressed(os.path.join(RESULTS, "profiles.npz"), **profiles)
    fields = list(dict.fromkeys(k for r in rows for k in r if k != "keep_log"))
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
