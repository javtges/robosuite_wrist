"""
Plots of the wrist-schedule study (run wrist_study.py first):

    cd scripts && python -m arm_impedance.wrist_figures

figures/wrist_schedules.png   per schedule: success over the 27 error cases and over random poses, peak flange force,
                              peak lateral force on the key, time to seat
figures/wrist_yaw_tolerance.png  success vs yaw error for each schedule: how much aiming error each forgives
"""

import csv
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from arm_impedance.wrist_study import MECHANISM, RESULTS, SCHEDULES  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
FIGS = os.path.join(HERE, "figures")
SURFACE, INK, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#898781", "#e1e0d9"
MECH_COLOR = {"fixed": "#898781", "extension": "#2a78d6", "scale": "#7a4fc9"}
plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE, "font.size": 10,
    "axes.titlesize": 11, "axes.titleweight": "bold", "axes.titlelocation": "left", "axes.grid": True,
    "grid.color": GRID, "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
})  # fmt: skip
TEXT = ("study", "label", "arm", "result", "schedule", "wrist_schedule", "mechanism", "error")


def load():
    rows = []
    for r in csv.DictReader(open(os.path.join(RESULTS, "episodes.csv"))):
        if r["result"] == "error":
            continue
        for k, v in r.items():
            if v in ("True", "False"):
                r[k] = v == "True"
            elif k not in TEXT and v not in ("", None):
                try:
                    r[k] = float(v)
                except ValueError:
                    pass
        rows.append(r)
    return rows


def rate(rows):
    return 100.0 * np.mean([r["success"] for r in rows]) if rows else np.nan


def med(rows, key, seated_only=False):
    v = [r[key] for r in rows if (r["success"] or not seated_only) and isinstance(r[key], float) and np.isfinite(r[key])]
    return float(np.median(v)) if v else np.nan


def save(fig, name):
    os.makedirs(FIGS, exist_ok=True)
    path = os.path.join(FIGS, name)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("wrote", os.path.relpath(path, os.path.dirname(HERE)))


def fig_bars(rows):
    names = [n for n in SCHEDULES if any(r["label"] == n for r in rows)]
    err = {n: [r for r in rows if r["label"] == n and r["fixed_lock"]] for n in names}
    rnd = {n: [r for r in rows if r["label"] == n and not r["fixed_lock"]] for n in names}
    colors = [MECH_COLOR[MECHANISM[n]] for n in names]
    x = np.arange(len(names))
    fig, axes = plt.subplots(1, 4, figsize=(18, 5), constrained_layout=True)
    axes[0].bar(x - 0.2, [rate(err[n]) for n in names], 0.4, color=colors)
    axes[0].bar(x + 0.2, [rate(rnd[n]) for n in names], 0.4, color=colors, alpha=0.45)
    axes[0].set_ylim(0, 105)
    axes[0].set_ylabel("seated (%)")
    axes[0].set_title("Success: 27 error cases (solid), random poses (faded)")
    axes[1].bar(x, [med(err[n], "peak_flange_N") for n in names], color="#eb6834")
    axes[1].set_ylabel("peak flange force (N, median)")
    axes[1].set_title("Peak force on the arm, error cases")
    axes[2].bar(x, [med(err[n], "peak_lateral_N") for n in names], color="#7a4fc9")
    axes[2].set_ylabel("peak lateral key force (N, median)")
    axes[2].set_title("Peak lateral force on the key, error cases")
    axes[3].bar(x, [med(err[n] + rnd[n], "contact_to_seat_s", True) for n in names], color="#0ca30c")
    axes[3].set_ylabel("contact → seated (s, median of successes)")
    axes[3].set_title("Time to seat")
    for ax in axes:
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=35, ha="right", fontsize=8)
        ax.grid(axis="x", visible=False)
    from matplotlib.patches import Patch

    axes[0].legend(handles=[Patch(color=c, label=m) for m, c in MECH_COLOR.items()], loc="lower left", fontsize=8, title="mechanism")
    fig.suptitle("Wrist stiffness by task phase (free → search → slide → seat), robot's arm controller", fontweight="bold", x=0, ha="left")
    save(fig, "wrist_schedules.png")


def fig_yaw(rows):
    names = [n for n in SCHEDULES if any(r["label"] == n for r in rows)]
    yaws = sorted(set(abs(r["dyaw_deg"]) for r in rows if r["fixed_lock"]))
    fig, ax = plt.subplots(figsize=(9, 5), constrained_layout=True)
    styles = {"fixed": "-", "extension": "--", "scale": ":"}
    cmap = plt.get_cmap("tab10")
    for i, n in enumerate(names):
        ys = [rate([r for r in rows if r["label"] == n and r["fixed_lock"] and abs(r["dyaw_deg"]) == y]) for y in yaws]
        ax.plot(yaws, ys, linestyle=styles[MECHANISM[n]], marker="o", color=cmap(i % 10), label=n, lw=1.8)
    ax.set_xlabel("|yaw aiming error| (deg), averaged over ±, over dx ∈ {−1.5, 0, +1.5} mm")
    ax.set_ylabel("seated (%)")
    ax.set_ylim(-3, 103)
    ax.set_title("How much yaw error each wrist setting forgives")
    ax.legend(fontsize=8, ncol=2)
    save(fig, "wrist_yaw_tolerance.png")


def main():
    rows = load()
    fig_bars(rows)
    fig_yaw(rows)


if __name__ == "__main__":
    main()
