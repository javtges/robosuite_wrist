"""
Plots of the wrist-less arm impedance results (run evaluate.py first):

    cd scripts && python -m arm_impedance.make_figures

figures/success_maps.png      outcome and peak flange force over (dx, dyaw): robosuite's OSC vs the robot's HSA law
figures/stiffness_sweeps.png  success rate and forces vs rotational stiffness, translational stiffness and force ceiling
figures/schedules.png         the variable-impedance schedules: success, peak force, time to seat
figures/force_profiles.png    push force vs key depth for the profile episodes
"""

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
from arm_impedance.evaluate import CLIP_T, K_R, K_T, PROFILE_ERRORS, RESULTS, SCHEDULES  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
FIGS = os.path.join(HERE, "figures")

SURFACE, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
ARM_COLORS = {"osc": "#eb6834", "hsa": "#2a78d6", "fixed": "#2a78d6", "soft_rot_insert": "#7a4fc9"}
GOOD, WARNING, CRITICAL = "#0ca30c", "#e6a100", "#d03b3b"
OUTCOMES = {"seated": GOOD, "stuck": WARNING, "jammed": CRITICAL, "timeout": MUTED}

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans"],
    "font.size": 10, "text.color": INK, "axes.labelcolor": INK2, "axes.edgecolor": AXIS,
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False, "lines.linewidth": 2.0, "legend.frameon": False,
    "axes.titlesize": 11, "axes.titleweight": "bold", "axes.titlelocation": "left",
})  # fmt: skip

TEXT = ("study", "label", "arm", "result", "error_case", "error", "schedule")


def load_rows():
    rows = []
    for r in csv.DictReader(open(os.path.join(RESULTS, "episodes.csv"))):
        if r["result"] == "error":
            continue
        for k, v in r.items():
            if v in ("True", "False"):
                r[k] = v == "True"
            elif k not in TEXT and v != "":
                r[k] = float(v)
        rows.append(r)
    return rows


def save(fig, name):
    os.makedirs(FIGS, exist_ok=True)
    path = os.path.join(FIGS, name)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("wrote", os.path.relpath(path, os.path.dirname(HERE)))


def rate(rows):
    return 100.0 * np.mean([r["success"] for r in rows]) if rows else np.nan


def med(rows, key, only_success=False):
    v = [r[key] for r in rows if (r["success"] or not only_success) and np.isfinite(r[key])]
    return np.median(v) if v else np.nan


# ---- success maps ------------------------------------------------------------------------------------------------------


def fig_maps(rows):
    grid = [r for r in rows if r["study"] == "grid"]
    dxs = sorted(set(r["dx_mm"] for r in grid))
    dyaws = sorted(set(r["dyaw_deg"] for r in grid))
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 9.0), constrained_layout=True)
    codes = {k: i for i, k in enumerate(OUTCOMES)}
    cmap = ListedColormap(list(OUTCOMES.values()))
    titles = {"osc": "robosuite OSC: kp 150, stiffness kp·Λ, no ceiling", "hsa": f"HSA law: K {K_T:g} N/m, {K_R:g} N·m/rad, {K_T * CLIP_T:g} N ceiling"}
    for col, arm in enumerate(("osc", "hsa")):
        sel = [r for r in grid if r["arm"] == arm]
        out = np.full((len(dyaws), len(dxs)), np.nan)
        force = np.full_like(out, np.nan)
        for r in sel:
            i, j = dyaws.index(r["dyaw_deg"]), dxs.index(r["dx_mm"])
            out[i, j] = codes.get(r["result"], 3)
            force[i, j] = r["peak_flange_N"]
        ext = [dxs[0] - 0.25, dxs[-1] + 0.25, dyaws[0] - 0.75, dyaws[-1] + 0.75]
        ax = axes[0, col]
        ax.imshow(out, cmap=cmap, vmin=-0.5, vmax=len(OUTCOMES) - 0.5, origin="lower", extent=ext, aspect="auto")
        ax.set_title(f"{titles[arm]}\noutcome, {rate(sel):.0f}% seated")
        ax = axes[1, col]
        im = ax.imshow(force, cmap="magma_r", vmin=0, vmax=50, origin="lower", extent=ext, aspect="auto")
        ax.set_title("peak flange force (N)")
        fig.colorbar(im, ax=ax, shrink=0.8)
    for ax in axes.flat:
        ax.set_xlabel("perception error across the slot, dx (mm)")
        ax.set_ylabel("perception error in yaw (deg)")
        ax.grid(False)
    axes[0, 0].legend(handles=[Patch(color=c, label=k) for k, c in OUTCOMES.items()], loc="upper left", ncol=4, bbox_to_anchor=(0, -0.14))
    fig.suptitle("Wrist-less key insertion (wrist fully contracted): the arm controller is the only compliance", fontweight="bold", x=0.0, ha="left")
    save(fig, "success_maps.png")


# ---- sweeps ----------------------------------------------------------------------------------------------------------


def _sweep_panel(ax, rows, xkey, xlabel, log=True, mark=None):
    xs = sorted(set(r[xkey] for r in rows))
    groups = {
        "all 25 error cases": lambda r: True,
        "yaw error only (dx = 0)": lambda r: r["dx_mm"] == 0 and r["dyaw_deg"] != 0,
        "lateral error only (yaw = 0)": lambda r: r["dyaw_deg"] == 0 and r["dx_mm"] != 0,
    }
    styles = {"all 25 error cases": (INK, "-", "o"), "yaw error only (dx = 0)": ("#7a4fc9", "--", "^"), "lateral error only (yaw = 0)": ("#2a78d6", "--", "s")}
    for name, pred in groups.items():
        ys = [rate([r for r in rows if r[xkey] == x and pred(r)]) for x in xs]
        c, ls, m = styles[name]
        ax.plot(xs, ys, color=c, linestyle=ls, marker=m, label=name)
    ax.set_ylim(-3, 103)
    ax.set_ylabel("seated (%)")
    ax.set_xlabel(xlabel)
    if log:
        ax.set_xscale("log")
    if mark is not None:
        ax.axvline(mark, color=MUTED, linewidth=1, linestyle=":")
        ax.text(mark, 100, " robot today", color=MUTED, fontsize=8, va="top")
    ax2 = ax.twinx()
    ax2.plot(xs, [med([r for r in rows if r[xkey] == x], "peak_flange_N") for x in xs], color="#eb6834", linestyle="-", marker=".", linewidth=1.2, label="median peak flange force (N)")
    ax2.set_ylabel("peak flange force (N, median)", color="#eb6834")
    ax2.tick_params(axis="y", colors="#eb6834")
    ax2.grid(False)
    ax2.spines["right"].set_visible(True)
    ax2.spines["right"].set_color("#eb6834")
    return ax, ax2


def fig_sweeps(rows):
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8), constrained_layout=True)
    _sweep_panel(axes[0], [r for r in rows if r["study"] == "rot_k"], "k_r", "rotational stiffness K_rot (N·m/rad)", mark=K_R)
    axes[0].set_title("Rotational stiffness (damping ratio kept)")
    _sweep_panel(axes[1], [r for r in rows if r["study"] == "trans_k"], "k_t", "translational stiffness K (N/m), ceiling kept at 20 N", mark=K_T)
    axes[1].set_title("Translational stiffness, 20 N ceiling")
    ax, ax2 = _sweep_panel(axes[2], [r for r in rows if r["study"] == "ceiling"], "ceiling_N", "force ceiling K·clip (N), K = 2000 N/m", log=False, mark=K_T * CLIP_T)
    axes[2].set_title("Force ceiling (translational_clip)")
    h1, l1 = axes[0].get_legend_handles_labels()
    fig.legend(h1 + [plt.Line2D([], [], color="#eb6834", marker=".", linewidth=1.2)], l1 + ["median peak flange force (right axis)"], loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.08))
    save(fig, "stiffness_sweeps.png")


# ---- schedules -------------------------------------------------------------------------------------------------------


def fig_schedules(rows):
    sched = [r for r in rows if r["study"] == "schedule"]
    names = [n for n in SCHEDULES if any(r["label"] == n for r in sched)]
    fig, axes = plt.subplots(1, 4, figsize=(17, 4.6), constrained_layout=True)
    x = np.arange(len(names))
    err = {n: [r for r in sched if r["label"] == n and r["fixed_lock"]] for n in names}
    rnd = {n: [r for r in sched if r["label"] == n and not r["fixed_lock"]] for n in names}
    axes[0].bar(x - 0.2, [rate(err[n]) for n in names], 0.4, color="#2a78d6", label="25 perception-error cases")
    axes[0].bar(x + 0.2, [rate(rnd[n]) for n in names], 0.4, color="#898781", label="random lock poses, no error")
    axes[0].set_ylabel("seated (%)")
    axes[0].set_ylim(0, 105)
    axes[0].set_title("Success")
    axes[0].legend(loc="lower left", fontsize=8)
    axes[1].bar(x, [med(err[n], "peak_flange_N") for n in names], color="#eb6834")
    axes[1].set_ylabel("peak flange force (N, median)")
    axes[1].set_title("Peak flange force, error cases")
    axes[2].bar(x, [med(err[n], "peak_lateral_N") for n in names], color="#7a4fc9")
    axes[2].set_ylabel("peak lateral key force (N, median)")
    axes[2].set_title("Peak lateral force on the key, error cases")
    axes[3].bar(x, [med(err[n] + rnd[n], "contact_to_seat_s", only_success=True) for n in names], color="#0ca30c")
    axes[3].set_ylabel("contact → seated (s, median of successes)")
    axes[3].set_title("Time to seat")
    for ax in axes:
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=35, ha="right", fontsize=8)
        ax.grid(axis="x", visible=False)
    fig.suptitle("Variable impedance by task phase (free → search → slide → seat); 'fixed' is the robot's gains throughout", fontweight="bold", x=0.0, ha="left")
    save(fig, "schedules.png")


# ---- profiles --------------------------------------------------------------------------------------------------------


def fig_profiles():
    prof = np.load(os.path.join(RESULTS, "profiles.npz"))
    labels = ("osc", "fixed", "soft_rot_insert")
    names = {"osc": "robosuite OSC", "fixed": "HSA law, robot gains", "soft_rot_insert": "HSA law, K_rot 15 during insertion"}
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6), constrained_layout=True)
    for ax, (case, err) in zip(axes, PROFILE_ERRORS.items()):
        for label in labels:
            key = f"{label}/{case}/"
            if key + "t" not in prof:
                continue
            depth, f = prof[key + "depth"] * 1e3, -prof[key + "flange_force"][:, 2]
            ins = prof[key + "phase"] >= 2
            ax.plot(depth[ins], f[ins], color=ARM_COLORS[label], label=names[label], alpha=0.9)
        ax.axvline(26, color=MUTED, linewidth=1, linestyle=":")
        ax.text(26, ax.get_ylim()[1] * 0.95, " plunger", color=MUTED, fontsize=8, va="top")
        ax.axhline(20, color="#eb6834", linewidth=1, linestyle=":")
        ax.text(0.5, 20, "20 N ceiling", color="#eb6834", fontsize=8, va="bottom")
        ax.set_xlabel("key depth in the slot (mm)")
        ax.set_ylabel("push on the flange (N)")
        ax.set_title(f"perception error: {'none' if case == 'none' else f'{err[0]:g} mm across the slot, {err[2]:g}° yaw'}")
    axes[0].legend(loc="upper left")
    save(fig, "force_profiles.png")


def main():
    rows = load_rows()
    fig_maps(rows)
    fig_sweeps(rows)
    fig_schedules(rows)
    fig_profiles()


if __name__ == "__main__":
    main()
