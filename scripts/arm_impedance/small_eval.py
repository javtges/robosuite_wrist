"""
A small, readable evaluation: a few perception errors x a few controllers, one line per episode, a ledger CSV and
two charts. The same ledger columns as the big study, so the charts come from `ledger_figures` and work on any
ledger, including one filled by hand from hardware trials.

    cd scripts
    python -m arm_impedance.small_eval                         # 30 episodes: robot law and OSC, 3 dx x 5 dyaw
    python -m arm_impedance.small_eval --controllers hsa --wrist soft     # the robot law with the soft wrist
    python -m arm_impedance.small_eval --schedule gentle_search --dx 0 --dyaw -4 0 4
    python -m arm_impedance.small_eval --out results_small_fer --robot fer   (set HSA_ROBOT=fer first)

Writes <out>/episodes.csv and <out>/summary.md, and <out>/success_map.png + <out>/forces.png via ledger_figures.
"""

import argparse
import csv
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import arm_impedance.evaluate as ae  # noqa: E402
from arm_impedance.ledger_figures import LEDGER_COLUMNS, make_ledger_figures, summarize  # noqa: E402
from key_insertion.oracle import WRIST_PRESETS  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
NAMES = {"hsa": "robot law", "osc": "OSC"}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--controllers", default="hsa,osc", help="comma list of hsa (the robot's law and gains), osc")
    p.add_argument("--dx", type=float, nargs="+", default=[-1.5, 0.0, 1.5], help="perception error across the slot (mm)")
    p.add_argument("--dyaw", type=float, nargs="+", default=[-4, -2, 0, 2, 4], help="perception error in yaw (deg)")
    p.add_argument("--wrist", default="stiff", help="stiff (contracted: the wrist-less stand-in), home, soft, or a z in [-1, 1]")
    p.add_argument("--schedule", default="", choices=[""] + list(ae.SCHEDULES), help="hsa only: a variable-impedance schedule")
    p.add_argument("--clip", type=float, default=None, help="hsa only: translational_clip (mm), the force ceiling / 2000")
    p.add_argument("--seeds", type=int, default=1, help="repeat each case with this many random lock poses (seed 0 = fixed lock)")
    p.add_argument("--out", default=os.path.join(HERE, "results_small"))
    args = p.parse_args()

    wrist_z = WRIST_PRESETS[args.wrist] if args.wrist in WRIST_PRESETS else float(args.wrist)
    ae.WRIST_Z = wrist_z
    controllers = [c.strip() for c in args.controllers.split(",") if c.strip()]
    overrides = ae.gains(clip_t=args.clip * 1e-3) if args.clip is not None else ae.gains()
    rows = []
    print(f"{'#':>3} {'controller':11s} {'believed error':>17s} | {'result':7s} {'depth':>8s} {'peak push':>10s} {'in contact':>10s}")
    t0 = time.time()
    n = 0
    for arm in controllers:
        for seed in range(args.seeds):
            for dx in args.dx:
                for dyaw in args.dyaw:
                    n += 1
                    label = NAMES[arm] + (f" {args.schedule}" if args.schedule and arm == "hsa" else "")
                    case = ae._case(
                        "small", label, arm=arm, overrides=overrides if arm == "hsa" else None, dx=dx, dyaw=dyaw,
                        schedule=args.schedule if arm == "hsa" else "", fixed_lock=seed == 0, seed=seed,
                    )  # fmt: skip
                    r = ae.run_case(case)
                    r["wrist"] = args.wrist
                    r["source"] = "sim"
                    rows.append(r)
                    depth = r.get("depth_mm", float("nan"))
                    print(
                        f"{n:3d} {label:11s} {dx:+5.1f} mm, {dyaw:+3.0f} deg | {r['result']:7s} {depth:5.1f} mm "
                        f"{r.get('peak_flange_N', float('nan')):7.1f} N {r.get('contact_to_seat_s', float('nan')):7.1f} s"
                    )
    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "episodes.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=LEDGER_COLUMNS, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: (f"{v:.6g}" if isinstance(v, float) else v) for k, v in r.items()})
    with open(os.path.join(args.out, "summary.md"), "w") as f:
        f.write(summarize(rows, title=f"small eval: wrist {args.wrist}, {len(rows)} episodes in {time.time() - t0:.0f} s"))
    print(open(os.path.join(args.out, "summary.md")).read())
    make_ledger_figures(os.path.join(args.out, "episodes.csv"), args.out)


if __name__ == "__main__":
    main()
