"""
Plots of the door study (run door_study.py first):

    cd scripts && python -m arm_impedance.door_figures

figures/door_schedules.png   per wrist setting: success over the 15 aiming errors (mentor's model and measured_v0),
                             peak flange force per phase, time from contact to open
figures/door_factors.png     the three fixed wrists against cylinder friction, slot friction, door mass, hinge friction:
                             success and peak force of the phase the factor loads
"""

import csv
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from arm_impedance.door_study import FIXED_THREE, MECHANISM, RESULTS, SCHEDULES  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
FIGS = os.path.join(HERE, "figures")
SURFACE, INK, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#898781", "#e1e0d9"
MECH_COLOR = {"fixed": "#898781", "extension": "#2a78d6", "scale": "#7a4fc9"}
WRIST_COLOR = {"fixed_stiff": "#eb6834", "fixed_home": "#898781", "fixed_soft": "#2a78d6"}
plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE, "font.size": 10,
    "axes.titlesize": 11, "axes.titleweight": "bold", "axes.titlelocation": "left", "axes.grid": True,
    "grid.color": GRID, "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
})  # fmt: skip
TEXT = ("study", "label", "schedule", "mechanism", "result", "reached", "factor", "wrist_calib", "error")


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


def med(rows, key, ok_only=False):
    v = [r[key] for r in rows if (r["success"] or not ok_only) and isinstance(r[key], float) and np.isfinite(r[key])]
    return float(np.median(v)) if v else np.nan


def save(fig, name):
    os.makedirs(FIGS, exist_ok=True)
    path = os.path.join(FIGS, name)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("wrote", os.path.relpath(path, os.path.dirname(HERE)))


def fig_schedules(rows):
    names = list(SCHEDULES)
    sched = {n: [r for r in rows if r["study"] == "schedule" and r["label"] == n] for n in names}
    hard = {n: [r for r in rows if r["study"] == "schedule_hard" and r["label"] == n] for n in names}
    calib = {n: [r for r in rows if r["study"] == "calib" and r["label"] == n] for n in names}
    x = np.arange(len(names))
    fig, axes = plt.subplots(1, 3, figsize=(19, 5.4), constrained_layout=True)
    colors = [MECH_COLOR[MECHANISM[n]] for n in names]
    w = 0.27
    axes[0].bar(x - w, [rate(sched[n]) for n in names], w, color=colors, label="normal lock (0.15 N m)")
    axes[0].bar(x, [rate(hard[n]) if hard[n] else np.nan for n in names], w, color=colors, alpha=0.6, hatch="xx", label="stiff lock (0.4 N m)")
    axes[0].bar(x + w, [rate(calib[n]) if calib[n] else np.nan for n in names], w, color=colors, alpha=0.4, hatch="//", label="normal lock, measured_v0 wrist")
    axes[0].set_ylim(0, 105)
    axes[0].set_ylabel("door opened (%)")
    axes[0].set_title("Success over 15 aiming errors")
    h = [Patch(color=c, label=m) for m, c in MECH_COLOR.items()] + [Patch(facecolor="w", edgecolor="k", hatch="xx", label="stiff lock"), Patch(facecolor="w", edgecolor="k", hatch="//", label="measured_v0 wrist")]
    axes[0].legend(handles=h, loc="lower left", fontsize=8, ncol=2)
    for i, (key, c, lab) in enumerate((("peak_insert_N", "#0ca30c", "insert"), ("peak_turn_N", "#e6a100", "turn"), ("peak_push_N", "#d03b3b", "push"))):
        axes[1].bar(x + (i - 1) * w, [med(sched[n], key) for n in names], w, color=c, label=lab)
    axes[1].set_ylabel("peak flange force (N, median), normal lock")
    axes[1].set_title("Peak force per phase")
    axes[1].legend(fontsize=8)
    axes[2].bar(x - 0.2, [med(sched[n], "peak_turn_Nm") for n in names], 0.4, color="#e6a100", label="normal lock")
    axes[2].bar(x + 0.2, [med(hard[n], "peak_turn_Nm") if hard[n] else np.nan for n in names], 0.4, color="#e6a100", alpha=0.6, hatch="xx", label="stiff lock")
    axes[2].set_ylabel("peak flange moment while turning (N m, median)")
    axes[2].set_title("Turning torque the arm supplies")
    axes[2].legend(fontsize=8)
    for ax in axes:
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=35, ha="right", fontsize=8)
        ax.grid(axis="x", visible=False)
    fig.suptitle("KeyDoor: insert, turn, push with the wrist fixed or switched by phase (robot's arm law)", fontweight="bold", x=0, ha="left")
    save(fig, "door_schedules.png")


def fig_factors(rows):
    panels = (
        ("friction", "cylinder friction (N m)", "peak_turn_Nm", "peak turn moment at the flange (N m)"),
        ("slot", "slot wall friction", "peak_insert_N", "peak insert force (N)"),
        ("mass", "door mass (kg)", "peak_push_N", "peak push force (N)"),
        ("hinge", "hinge friction (N m)", "peak_push_N", "peak push force (N)"),
    )
    fig, axes = plt.subplots(2, 4, figsize=(18, 8), constrained_layout=True)
    for col, (study, xlabel, fkey, flabel) in enumerate(panels):
        sel = [r for r in rows if r["study"] == study]
        values = sorted(set(float(r["factor"].split("=")[1]) for r in sel))
        for n in FIXED_THREE:
            w = [r for r in sel if r["label"] == n]
            ys = [rate([r for r in w if float(r["factor"].split("=")[1]) == v]) for v in values]
            fs = [med([r for r in w if float(r["factor"].split("=")[1]) == v], fkey) for v in values]
            axes[0, col].plot(values, ys, marker="o", color=WRIST_COLOR[n], label=n.replace("fixed_", ""))
            axes[1, col].plot(values, fs, marker="o", color=WRIST_COLOR[n], label=n.replace("fixed_", ""))
        axes[0, col].set_ylim(-3, 103)
        axes[0, col].set_ylabel("door opened (%), yaw 0 / 2 / 4°")
        axes[0, col].set_title(xlabel)
        axes[1, col].set_ylabel(flabel)
        axes[1, col].set_xlabel(xlabel)
        for ax in axes[:, col]:
            ax.set_xscale("log" if study in ("friction", "hinge") else "linear")
    axes[0, 0].legend(title="wrist", fontsize=8)
    fig.suptitle("What makes the door hard: three fixed wrist settings against the lock and door parameters", fontweight="bold", x=0, ha="left")
    save(fig, "door_factors.png")


def main():
    rows = load()
    fig_schedules(rows)
    fig_factors(rows)


if __name__ == "__main__":
    main()
