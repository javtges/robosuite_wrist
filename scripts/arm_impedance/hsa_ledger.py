"""
HSA-UMI's real ledgers (the demo recorder's and rollout_ledger's CSVs under data/key_in_hole) as the common ledger
that ledger_figures.py charts, so the robot's results sit next to the sim's in the same pictures.

    cd scripts
    python -m arm_impedance.hsa_ledger ../../HSA-UMI/data/key_in_hole --out arm_impedance/hardware_ledgers

Two kinds of file are recognised by their columns:
    demo ledger      one row per recorded teleop episode: outcome (success / failure, the operator's key), condition
                     (rigid, wrist_extended, ...), insert_s / turn_s / open_s. No placement offset, no force.
    rollout ledger   one row per policy trial: placement dx_mm / dy_mm / dtheta_deg, outcome, failure_code
                     (miss, jam, timeout), insertion_depth_mm, time_s, checkpoint, condition.

Mapping: label = "<condition> <dataset or checkpoint> (demos | rollouts)"; result = seated for success, else jammed for
a jam code, timeout for a timeout, stuck otherwise (a miss); dyaw_deg = dtheta_deg; time_s = the trial time (rollouts)
or insert_s (demos). peak_flange_N stays blank: the force is in the bags, not the ledgers.
"""

import argparse
import csv
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from arm_impedance.ledger_figures import LEDGER_COLUMNS, make_ledger_figures, summarize  # noqa: E402


def convert(path, root):
    rel = os.path.relpath(path, root).split(os.sep)  # e.g. rigid/v2/eval/v2f_best/ledger.csv
    with open(path, newline="") as f:
        rows = [r for r in csv.DictReader(f) if any((v or "").strip() for v in r.values())]
    if not rows:
        return []
    kind = "rollouts" if "placement_id" in rows[0] else "demos"
    tag = rel[-2] if kind == "rollouts" else "/".join(rel[:2])
    out = []
    for r in rows:
        cond = r.get("condition", "") or rel[0]
        outcome = (r.get("outcome") or "").strip().lower()
        code = (r.get("failure_code") or "").strip().lower()
        if outcome == "success":
            result = "seated"
        elif "jam" in code:
            result = "jammed"
        elif "timeout" in code or (r.get("stopped_by") == "timeout" and kind == "demos"):
            result = "timeout"
        else:
            result = "stuck"
        out.append(dict(
            source="hardware", label=f"{cond} {tag} ({kind})", wrist=cond, arm="robot law", schedule="",
            dx_mm=float(r.get("dx_mm") or 0.0), dyaw_deg=float(r.get("dtheta_deg") or 0.0),
            seed=r.get("rollout") or r.get("episode", ""), result=result,
            depth_mm=float(r["insertion_depth_mm"]) if (r.get("insertion_depth_mm") or "").strip() else float("nan"),
            time_s=float(r["time_s"]) if (r.get("time_s") or "").strip() else (float(r["insert_s"]) if (r.get("insert_s") or "").strip() else float("nan")),
            notes=(r.get("notes") or "") + (f" [{code}]" if code else ""),
        ))  # fmt: skip
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("root", help="HSA-UMI/data/key_in_hole, or one ledger.csv")
    p.add_argument("--out", required=True)
    p.add_argument("--kind", default="all", choices=["all", "demos", "rollouts"])
    args = p.parse_args()
    paths = [args.root] if args.root.endswith(".csv") else sorted(
        os.path.join(d, "ledger.csv") for d, _, fs in os.walk(args.root) if "ledger.csv" in fs
    )
    root = os.path.dirname(args.root) if args.root.endswith(".csv") else args.root
    rows = []
    for path in paths:
        rows += convert(path, root)
    if args.kind != "all":
        rows = [r for r in rows if r["label"].endswith(f"({args.kind})")]
    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "episodes.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=LEDGER_COLUMNS, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: (f"{v:.6g}" if isinstance(v, float) else v) for k, v in r.items()})
    with open(os.path.join(args.out, "summary.md"), "w") as f:
        f.write(summarize(rows, title=f"HSA-UMI hardware ledgers under {args.root} ({len(paths)} files, {len(rows)} episodes)"))
    print(open(os.path.join(args.out, "summary.md")).read())
    make_ledger_figures(os.path.join(args.out, "episodes.csv"), args.out)


if __name__ == "__main__":
    main()
