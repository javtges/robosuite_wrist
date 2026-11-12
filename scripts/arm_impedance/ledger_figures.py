"""
Charts from an episode ledger: one row per insertion attempt, whoever ran it (the sim's oracle, a policy rollout, or
a person logging hardware trials by hand). Only these columns are needed; the rest are optional.

    label         what was being tested ("robot law", "OSC", "soft wrist", "gentle_search", ...)
    dx_mm         perception error across the slot (mm); 0 if none
    dyaw_deg      perception error in yaw (deg); 0 if none
    result        seated | stuck | jammed | timeout
    peak_flange_N peak external force on the flange during insertion (N); blank if not measured

Optional: contact_to_seat_s, hold_push_N, peak_lateral_N, depth_mm, wrist, seed, source (sim | hardware), notes.

    cd scripts
    python -m arm_impedance.ledger_figures results_small/episodes.csv            # -> charts next to the csv
    python -m arm_impedance.ledger_figures ~/hardware_ledger.csv --out ~/figs    # a hand-filled hardware ledger

success_map.png   one grid per label: outcome at every (dx, dyaw); the fraction seated in the title
forces.png        per label: success rate, median peak flange force, median time from contact to seated
"""

import argparse
import csv
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import ListedColormap  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

LEDGER_COLUMNS = [
    "source", "label", "wrist", "arm", "schedule", "dx_mm", "dyaw_deg", "seed", "result", "depth_mm", "peak_flange_N",
    "peak_lateral_N", "hold_push_N", "contact_to_seat_s", "max_yaw_deg", "k_t", "k_r", "clip_t_mm", "ceiling_N", "notes",
]  # fmt: skip
REQUIRED = ("label", "dx_mm", "dyaw_deg", "result")

SURFACE, INK, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#898781", "#e1e0d9"
OUTCOMES = {"seated": "#0ca30c", "stuck": "#e6a100", "jammed": "#d03b3b", "timeout": MUTED}
plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE, "font.size": 10,
    "axes.titlesize": 11, "axes.titleweight": "bold", "axes.titlelocation": "left", "axes.grid": True,
    "grid.color": GRID, "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
})  # fmt: skip


def load_ledger(path):
    rows = []
    with open(path) as f:
        for r in csv.DictReader(f):
            missing = [k for k in REQUIRED if not r.get(k, "").strip()]
            if missing:
                raise ValueError(f"{path}: row {r} lacks {missing}")
            for k in ("dx_mm", "dyaw_deg", "peak_flange_N", "contact_to_seat_s", "hold_push_N", "peak_lateral_N", "depth_mm"):
                v = r.get(k, "")
                r[k] = float(v) if v not in ("", None, "nan") else np.nan
            r["result"] = r["result"].strip().lower()
            rows.append(r)
    return rows


def _rate(rows):
    return 100.0 * np.mean([r["result"] == "seated" for r in rows]) if rows else np.nan


def _med(rows, key, seated_only=False):
    v = [r[key] for r in rows if np.isfinite(r[key]) and (r["result"] == "seated" or not seated_only)]
    return float(np.median(v)) if v else np.nan


def summarize(rows, title="ledger"):
    lines = [f"# {title}", "", "| label | episodes | seated | stuck | jammed | timeout | peak flange (N, median) | contact→seat (s, median) |", "|---|---|---|---|---|---|---|---|"]
    for label in dict.fromkeys(r["label"] for r in rows):
        w = [r for r in rows if r["label"] == label]
        n = lambda res: sum(r["result"] == res for r in w)  # noqa: E731
        lines.append(
            f"| {label} | {len(w)} | {n('seated')} ({_rate(w):.0f}%) | {n('stuck')} | {n('jammed')} | {n('timeout')} | "
            f"{_med(w, 'peak_flange_N'):.1f} | {_med(w, 'contact_to_seat_s', True):.2f} |"
        )
    return "\n".join(lines) + "\n"


def fig_success_map(rows, path):
    labels = list(dict.fromkeys(r["label"] for r in rows))
    dxs = sorted(set(r["dx_mm"] for r in rows))
    dyaws = sorted(set(r["dyaw_deg"] for r in rows))
    fig, axes = plt.subplots(1, len(labels), figsize=(4.6 * len(labels) + 1, 4.6), constrained_layout=True, squeeze=False)
    codes = {k: i for i, k in enumerate(OUTCOMES)}
    cmap = ListedColormap(list(OUTCOMES.values()))
    step_x = (dxs[-1] - dxs[0]) / max(len(dxs) - 1, 1) if len(dxs) > 1 else 1.0
    step_y = (dyaws[-1] - dyaws[0]) / max(len(dyaws) - 1, 1) if len(dyaws) > 1 else 1.0
    for ax, label in zip(axes[0], labels):
        sel = [r for r in rows if r["label"] == label]
        grid = np.full((len(dyaws), len(dxs)), np.nan)
        for r in sel:
            i, j = dyaws.index(r["dyaw_deg"]), dxs.index(r["dx_mm"])
            # several episodes on one cell (random lock poses): show the worst outcome
            code = codes.get(r["result"], 3)
            grid[i, j] = code if np.isnan(grid[i, j]) else max(grid[i, j], code)
        ext = [dxs[0] - step_x / 2, dxs[-1] + step_x / 2, dyaws[0] - step_y / 2, dyaws[-1] + step_y / 2]
        ax.imshow(grid, cmap=cmap, vmin=-0.5, vmax=len(OUTCOMES) - 0.5, origin="lower", extent=ext, aspect="auto")
        ax.set_xticks(dxs)
        ax.set_yticks(dyaws)
        ax.set_xlabel("perception error across the slot, dx (mm)")
        ax.set_ylabel("perception error in yaw (deg)")
        ax.set_title(f"{label}: {_rate(sel):.0f}% seated ({len(sel)} episodes)")
        ax.grid(False)
    axes[0][0].legend(handles=[Patch(color=c, label=k) for k, c in OUTCOMES.items()], loc="upper left", ncol=4, bbox_to_anchor=(0, -0.16))
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def fig_forces(rows, path):
    labels = list(dict.fromkeys(r["label"] for r in rows))
    x = np.arange(len(labels))
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2), constrained_layout=True)
    groups = {label: [r for r in rows if r["label"] == label] for label in labels}
    axes[0].bar(x, [_rate(groups[l]) for l in labels], color="#2a78d6")
    axes[0].set_ylim(0, 105)
    axes[0].set_ylabel("seated (%)")
    axes[0].set_title("Success")
    axes[1].bar(x, [_med(groups[l], "peak_flange_N") for l in labels], color="#eb6834")
    axes[1].set_ylabel("peak flange force (N, median)")
    axes[1].set_title("Peak force on the arm")
    axes[2].bar(x, [_med(groups[l], "contact_to_seat_s", True) for l in labels], color="#0ca30c")
    axes[2].set_ylabel("contact → seated (s, median of successes)")
    axes[2].set_title("Time to seat")
    for ax in axes:
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=20, ha="right")
        ax.grid(axis="x", visible=False)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def make_ledger_figures(csv_path, out_dir):
    rows = load_ledger(csv_path)
    os.makedirs(out_dir, exist_ok=True)
    fig_success_map(rows, os.path.join(out_dir, "success_map.png"))
    fig_forces(rows, os.path.join(out_dir, "forces.png"))
    print(f"wrote {out_dir}/success_map.png and forces.png from {len(rows)} episodes")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("csv")
    p.add_argument("--out", default=None, help="directory for the charts (default: next to the csv)")
    args = p.parse_args()
    make_ledger_figures(args.csv, args.out or os.path.dirname(os.path.abspath(args.csv)))


if __name__ == "__main__":
    main()
