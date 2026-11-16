"""
Plot of the 1-D z sweep (run z_sweep.py first): success and forces against the wrist's z, with the HSA twist it
corresponds to on the top axis, one line per yaw error, one panel column per wrist model.

    cd scripts && python -m arm_impedance.z_figures     # -> figures/z_sweep.png
"""

import csv
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from arm_impedance.wrist_calibration import normalized_z_to_twist  # noqa: E402
from arm_impedance.z_sweep import MODELS, RESULTS  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
FIGS = os.path.join(HERE, "figures")
SURFACE, GRID = "#fcfcfb", "#e1e0d9"
plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE, "font.size": 10,
    "axes.titlesize": 11, "axes.titleweight": "bold", "axes.titlelocation": "left", "axes.grid": True,
    "grid.color": GRID, "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
})  # fmt: skip


def main():
    rows = []
    for r in csv.DictReader(open(os.path.join(RESULTS, "episodes.csv"))):
        if r["result"] == "error":
            continue
        for k in ("z", "dx_mm", "dyaw_deg", "peak_flange_N", "peak_lateral_N", "max_wrist_lateral_mm", "contact_to_seat_s"):
            r[k] = float(r[k]) if r[k] not in ("", "nan") else np.nan
        r["success"] = r["success"] == "True"
        rows.append(r)
    yaws = sorted(set(r["dyaw_deg"] for r in rows))
    cmap = plt.get_cmap("viridis")
    fig, axes = plt.subplots(3, len(MODELS), figsize=(7 * len(MODELS), 11), constrained_layout=True, squeeze=False)
    for col, model in enumerate(MODELS):
        sel = [r for r in rows if r["model"] == model]
        zs = sorted(set(r["z"] for r in sel))
        for i, y in enumerate(yaws):
            w = [r for r in sel if r["dyaw_deg"] == y]
            succ = [100 * np.mean([r["success"] for r in w if r["z"] == z]) for z in zs]
            force = [np.median([r["peak_flange_N"] for r in w if r["z"] == z]) for z in zs]
            lat = [np.median([r["peak_lateral_N"] for r in w if r["z"] == z]) for z in zs]
            c = cmap(i / max(len(yaws) - 1, 1))
            axes[0, col].plot(zs, succ, marker="o", color=c, label=f"yaw {y:.0f}°")
            axes[1, col].plot(zs, force, marker="o", color=c)
            axes[2, col].plot(zs, lat, marker="o", color=c)
        axes[0, col].set_ylim(-3, 103)
        axes[0, col].set_ylabel("seated (%, over dx ∈ {−1.5, 0, +1.5} mm)")
        axes[0, col].set_title(f"wrist model: {model}")
        axes[1, col].set_ylabel("peak flange force (N, median)")
        axes[2, col].set_ylabel("peak lateral force on the key (N, median)")
        axes[2, col].set_xlabel("wrist z  (−1 contracted / stiff  →  +1 extended / soft)")
        for ax in axes[:, col]:
            top = ax.secondary_xaxis("top", functions=(lambda z: normalized_z_to_twist(z), lambda d: 2 * np.interp(d, [0, 360], [0, 18.93]) / 18.93 - 1))
            top.set_xlabel("HSA twist (deg, from the 11-08 sweep)", fontsize=8)
    axes[0, 0].legend(fontsize=8, title="aiming error")
    fig.suptitle("1-D stiffness: insertion with the wrist's z held at one level (robot's arm law)", fontweight="bold", x=0, ha="left")
    os.makedirs(FIGS, exist_ok=True)
    path = os.path.join(FIGS, "z_sweep.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print("wrote", os.path.relpath(path, os.path.dirname(HERE)))


if __name__ == "__main__":
    main()
