"""
Figures and the summary table for the nullspace evaluation (reads results/ from evaluate.py, writes figures/ and
results/summary.md).

    cd scripts && python -m nullspace.make_figures
"""

import csv
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, LogNorm  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

from nullspace.evaluate import (  # noqa: E402
    FINAL_POS_MM,
    FINAL_ROT_DEG,
    GRID_X,
    GRID_Y,
    GRID_Z,
    RESULTS,
    SETTLED_POS_MM,
    SETTLED_ROT_DEG,
    WRIST_RANGE_FRAC,
)

HERE = os.path.dirname(os.path.abspath(__file__))
FIGS = os.path.join(HERE, "figures")

# Reference palette (dataviz skill): light surface, ink, hairlines; strategies = neutral baseline + slots 2 and 1
SURFACE, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
COLORS = {"none": "#898781", "feedforward": "#eb6834", "feedback": "#2a78d6"}
LABELS = {"none": "no compensation (wrist mode)", "feedforward": "nullspace, feedforward", "feedback": "nullspace, feedback"}
GOOD, CRITICAL = "#0ca30c", "#d03b3b"
BLUES = ["#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7", "#3987e5", "#2a78d6", "#256abf",
         "#1c5cab", "#184f95", "#104281", "#0d366b"]  # fmt: skip
SEQ = LinearSegmentedColormap.from_list("blue_seq", BLUES)
AXIS_NAMES = ["x", "y", "z", "roll", "pitch", "yaw"]

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans"],  # has the ✓ ✗ glyphs
    "font.size": 10, "text.color": INK, "axes.labelcolor": INK2, "axes.edgecolor": AXIS,
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False, "lines.linewidth": 2.0, "legend.frameon": False,
    "axes.titlesize": 11, "axes.titleweight": "bold", "axes.titlelocation": "left",
})  # fmt: skip


def load_cases():
    rows = list(csv.DictReader(open(os.path.join(RESULTS, "cases.csv"))))
    out = []
    for r in rows:
        if r.get("error"):
            continue
        for k, v in r.items():
            if k in ("study", "strategy", "rot_axis", "error"):
                continue
            if v in ("True", "False"):
                r[k] = v == "True"
            elif v != "":
                r[k] = float(v)
        out.append(r)
    return out


def save(fig, name):
    os.makedirs(FIGS, exist_ok=True)
    path = os.path.join(FIGS, name)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote nullspace/figures/{name}")


# ---------------------------------------------------------------------------------------------------------------------


def phase_bands(ax, ts):
    """Shade the excursion of each wrist axis and label it at the top"""
    phase = ts["phase"]
    for a in range(6):
        idx = np.where(phase == a)[0]
        if not len(idx):
            continue
        t0, t1 = ts["t"][idx[0]], ts["t"][idx[-1]]
        if a % 2 == 0:
            ax.axvspan(t0, t1, color="#f0efec", zorder=0, lw=0)
        ax.text((t0 + t1) / 2, 0.97, AXIS_NAMES[a], transform=ax.get_xaxis_transform(), ha="center", va="top",
                color=INK2, fontsize=8.5)  # fmt: skip


def fig_timeseries(hz=20):
    ts = {s: dict(np.load(os.path.join(RESULTS, f"timeseries_{s}_{hz}hz.npz"))) for s in COLORS}
    fig, axes = plt.subplots(4, 1, figsize=(10, 10), sharex=True, gridspec_kw=dict(hspace=0.42))
    ref = ts["feedback"]
    # (a) wrist motion (the same for all strategies: it's the operator's input) - feedback run, normalized
    ax = axes[0]
    w = ref["wrist_meas"]
    lim = np.array([0.010, 0.010, 0.020] + [np.radians(10)] * 3)
    ax.plot(ref["t"], np.max(np.abs(w / lim), axis=1) * 100, color=INK2, lw=1.5)
    ax.set_ylabel("wrist excursion\n(% of range, largest axis)")
    ax.set_ylim(0, 110)
    ax.set_title("The operator drives the wrist to each axis' limits and back (20 Hz)")
    # (b) arm tcp motion: the arm moves opposite to the wrist in nullspace mode
    ax = axes[1]
    for s, d in ts.items():
        disp = np.linalg.norm(d["tcp_meas"][:, :3] - d["tcp_meas"][0, :3], axis=1) * 1e3
        ax.plot(d["t"], disp, color=COLORS[s], label=LABELS[s])
    ax.set_ylabel("arm tcp displacement\n(mm)")
    ax.set_title("The arm moves the opposite way in nullspace mode (and holds still in wrist mode)")
    # (c, d) plate error
    for ax, key, unit, title in (
        (axes[2], "plate_dp", "mm", "Top plate position error"),
        (axes[3], "plate_drot", "deg", "Top plate orientation error"),
    ):
        for s, d in ts.items():
            e = np.linalg.norm(d[key], axis=1) * (1e3 if unit == "mm" else 180 / np.pi)
            ax.plot(d["t"], np.maximum(e, 1e-3), color=COLORS[s], label=LABELS[s])
        ax.set_yscale("log")
        ax.set_ylabel(f"error ({unit}, log)")
        ax.set_title(f"{title}: what the gripper feels")
    axes[3].set_xlabel("time (s)")
    for ax in axes:
        phase_bands(ax, ref)
    handles = [Line2D([], [], color=COLORS[s], lw=2, label=LABELS[s]) for s in COLORS]
    fig.legend(handles=handles, loc="upper left", ncols=3, bbox_to_anchor=(0.08, 0.955), fontsize=9)
    fig.suptitle("Nullspace mode at the workspace center: one full excursion of every wrist axis", x=0.08, ha="left",
                 y=0.985, fontsize=12, fontweight="bold")  # fmt: skip
    for ax in axes:
        ax.set_title(ax.get_title(), pad=6, fontsize=10, fontweight="normal", color=INK)
        ax.set_ylim(top=ax.get_ylim()[1] * (1.25 if ax.get_yscale() == "linear" else 8))
    save(fig, "timeseries_center.png")


def fig_zoom(hz=20):
    """One excursion (+x) at full resolution: transient and settling"""
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), gridspec_kw=dict(wspace=0.3))
    for ax, key, unit in ((axes[0], "plate_dp", "mm"), (axes[1], "plate_drot", "deg")):
        for s in ("feedforward", "feedback"):
            for rate, ls in ((20, "-"), (50, "--")):
                d = dict(np.load(os.path.join(RESULTS, f"timeseries_{s}_{rate}hz.npz")))
                idx = np.where(d["phase"] == 0)[0]
                t = d["t"][idx] - d["t"][idx[0]]
                e = np.linalg.norm(d[key][idx], axis=1) * (1e3 if unit == "mm" else 180 / np.pi)
                ax.plot(t, e, color=COLORS[s], ls=ls, lw=1.8 if ls == "-" else 1.5,
                        label=f"{s}, {rate} Hz")  # fmt: skip
        ax.set_xlabel("time since the x excursion began (s)")
        ax.set_ylabel(f"plate {'position' if unit == 'mm' else 'orientation'} error ({unit})")
    axes[0].set_title("x excursion: to +10 mm, to -10 mm, then home")
    axes[1].legend(fontsize=9, loc="upper right")
    save(fig, "transient_x_excursion.png")


def grid_cases(cases, strategy):
    return [c for c in cases if c["study"] == "grid" and c["strategy"] == strategy]


def fig_maps(cases, metric, title, fname, vmin, vmax, thresh):
    strategies = ("none", "feedforward", "feedback")
    zs = sorted(GRID_Z, reverse=True)
    fig, axes = plt.subplots(len(zs), 3, figsize=(11, 2.9 * len(zs)), sharey=True,
                             gridspec_kw=dict(hspace=0.32, wspace=0.06))  # fmt: skip
    norm = LogNorm(vmin=vmin, vmax=vmax)
    for j, s in enumerate(strategies):
        by = {(c["x"], c["y"], c["z"]): c for c in grid_cases(cases, s)}
        for i, z in enumerate(zs):
            ax = axes[i, j]
            ax.grid(False)
            M = np.full((len(GRID_X), len(GRID_Y)), np.nan)
            for a, x in enumerate(GRID_X):
                for b, y in enumerate(GRID_Y):
                    c = by.get((x, y, z))
                    if c is None:
                        continue
                    M[a, b] = max(c[metric], vmin)
            ax.imshow(M, cmap=SEQ, norm=norm, origin="lower", aspect="auto",
                      extent=(-0.5, len(GRID_Y) - 0.5, -0.5, len(GRID_X) - 0.5))  # fmt: skip
            for a, x in enumerate(GRID_X):
                for b, y in enumerate(GRID_Y):
                    c = by.get((x, y, z))
                    if c is None:
                        continue
                    v = c[metric]
                    dark = norm(max(v, vmin)) > 0.55
                    txt = f"{v:.1f}" if v >= 1 else f"{v:.2f}"
                    ax.text(b - 0.12, a, txt, ha="center", va="center", fontsize=7.5, color="white" if dark else INK)
                    if not c["reachable"]:
                        sym, col = "—", INK2
                    elif s == "none":
                        continue
                    else:
                        sym, col = ("✓", GOOD) if c["usable"] else ("✗", CRITICAL)
                    ax.text(b + 0.33, a, sym, ha="center", va="center", fontsize=9, color=col, fontweight="bold",
                            bbox=dict(boxstyle="circle,pad=0.12", fc=SURFACE, ec="none", alpha=0.9))  # fmt: skip
            ax.set_xticks(range(len(GRID_Y)), [f"{y:+.2f}" for y in GRID_Y], fontsize=8)
            ax.tick_params(length=0)
            ax.set_yticks(range(len(GRID_X)), [f"{x:.2f}" for x in GRID_X], fontsize=8)
            if i == 0:
                ax.set_title(LABELS[s], fontsize=9.5)
            if j == 0:
                ax.set_ylabel(f"z = {z:.2f} m\nx (m, away from base)")
            if i == len(zs) - 1:
                ax.set_xlabel("y (m, robot's left +)")
    cbar = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=SEQ), ax=axes, shrink=0.5, pad=0.02)
    cbar.set_label(title)
    cbar.outline.set_visible(False)
    fig.suptitle(f"{title[0].upper() + title[1:]} across the arm's workspace (tcp position, robot base frame; "
                 f"gripper pointing down)\n"
                 f"✓ usable (reachable, settled ≤ {SETTLED_POS_MM} mm / {SETTLED_ROT_DEG}°, back at start ≤ "
                 f"{FINAL_POS_MM} mm / {FINAL_ROT_DEG}°, wrist reached ≥ {WRIST_RANGE_FRAC:.0%} of its range)   "
                 f"✗ reachable, not usable   — arm can't reach the pose",
                 x=0.06, ha="left", fontsize=10, y=0.955)  # fmt: skip
    save(fig, fname)


def fig_cdf(cases):
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8), gridspec_kw=dict(wspace=0.28))
    panels = (
        ("settled_plate_pos_mm", "settled position error (mm)", SETTLED_POS_MM),
        ("settled_plate_rot_deg", "settled orientation error (deg)", SETTLED_ROT_DEG),
        ("peak_plate_pos_mm", "peak (transient) position error (mm)", None),
    )
    for ax, (metric, label, th) in zip(axes, panels):
        for s in COLORS:
            g = [c for c in grid_cases(cases, s) if c["reachable"]]
            v = np.sort([c[metric] for c in g])
            ax.step(v, np.arange(1, len(v) + 1) / len(v) * 100, where="post", color=COLORS[s], label=LABELS[s])
        if th is not None:
            ax.axvline(th, color=INK2, lw=1, ls=":")
            ax.text(th, 2, f" threshold {th}", color=INK2, fontsize=8, rotation=90, va="bottom")
        ax.set_xscale("log")
        ax.set_xlabel(label)
        ax.set_ylim(0, 102)
    axes[0].set_ylabel("% of reachable grid points\nat or below")
    axes[0].set_title("Share of the workspace that meets a given error")
    axes[2].legend(fontsize=8.5, loc="lower right")
    save(fig, "error_cdf.png")


def fig_3d(cases):
    fig = plt.figure(figsize=(11, 5))
    for k, s in enumerate(("feedforward", "feedback")):
        ax = fig.add_subplot(1, 2, k + 1, projection="3d")
        g = grid_cases(cases, s)
        for status, marker, col, lab in (
            ("usable", "o", GOOD, "usable"),
            ("not usable", "X", CRITICAL, "reachable, not usable"),
            ("unreachable", "s", MUTED, "arm can't reach the pose"),
        ):
            sel = [c for c in g if (status == "unreachable" and not c["reachable"])
                   or (status == "usable" and c["usable"])
                   or (status == "not usable" and c["reachable"] and not c["usable"])]  # fmt: skip
            if sel:
                ax.scatter([c["x"] for c in sel], [c["y"] for c in sel], [c["z"] for c in sel], marker=marker, s=46,
                           color=col, edgecolor=SURFACE, linewidth=1.2, label=f"{lab} ({len(sel)})", depthshade=False)  # fmt: skip
        ax.scatter([0], [0], [0], marker="^", s=80, color=INK, label="robot base")
        ax.set_xlabel("x (m)")
        ax.set_ylabel("y (m)")
        ax.set_zlabel("z (m)")
        ax.set_title(LABELS[s], fontsize=10)
        ax.view_init(elev=22, azim=-125)
        ax.legend(fontsize=8, loc="upper left", bbox_to_anchor=(0.0, 0.0))
    save(fig, "workspace_3d.png")


def fig_orientation(cases):
    oc = [c for c in cases if c["study"] == "orientation"]
    keys = sorted({(c["rot_axis"], c["rot_deg"]) for c in oc}, key=lambda k: ("xyz".index(k[0]), k[1]))
    fig, axes = plt.subplots(2, 1, figsize=(11, 6.2), sharex=True, gridspec_kw=dict(hspace=0.3))
    width = 0.36
    for ax, metric, label in ((axes[0], "settled_plate_pos_mm", "settled position error (mm)"),
                              (axes[1], "settled_plate_rot_deg", "settled orientation error (deg)")):  # fmt: skip
        for k, s in enumerate(("feedforward", "feedback")):
            by = {(c["rot_axis"], c["rot_deg"]): c for c in oc if c["strategy"] == s}
            vals = [by[key][metric] if key in by else np.nan for key in keys]
            xs = np.arange(len(keys)) + (k - 0.5) * width
            ax.bar(xs, vals, width=width - 0.04, color=COLORS[s], label=LABELS[s])
            for x, key, v in zip(xs, keys, vals):
                c = by.get(key)
                if c is not None and not c["reachable"]:
                    ax.text(x, v * 1.15, "—", fontsize=10, color=INK2, ha="center", va="bottom", fontweight="bold")
        th = SETTLED_POS_MM if "pos" in metric else SETTLED_ROT_DEG
        ax.axhline(th, color=INK2, lw=1, ls=":")
        ax.text(len(keys) - 0.5, th, f"threshold {th}", color=INK2, fontsize=8, ha="right", va="bottom")
        ax.set_ylabel(label)
        ax.set_yscale("log")
    axes[1].set_xticks(range(len(keys)), [f"{a} {d:+.0f}°" for a, d in keys])
    axes[1].set_xlabel("tool orientation: rotation about the robot base axis from 'gripper down' (tcp at the workspace center)")
    axes[0].set_title("Tool orientation study (— above a bar: the arm can't reach that tool pose at all)")
    axes[0].legend(fontsize=9, loc="upper right", bbox_to_anchor=(1.0, 1.32), ncols=2)
    save(fig, "orientation_study.png")


def fig_rate(cases):
    cc = [c for c in cases if c["study"] == "center"]
    metrics = [("peak_plate_pos_mm", "peak pos (mm)"), ("peak_plate_rot_deg", "peak rot (deg)"),
               ("settled_plate_pos_mm", "settled pos (mm)"), ("settled_plate_rot_deg", "settled rot (deg)"),
               ("final_plate_pos_mm", "back at start (mm)")]  # fmt: skip
    fig, axes = plt.subplots(1, len(metrics), figsize=(13, 3.4), gridspec_kw=dict(wspace=0.45))
    shades = {20.0: "#86b6ef", 50.0: "#1c5cab"}
    for ax, (m, lab) in zip(axes, metrics):
        for k, s in enumerate(("none", "feedforward", "feedback")):
            for r, hz in enumerate((20.0, 50.0)):
                c = next((c for c in cc if c["strategy"] == s and c["control_freq"] == hz), None)
                if c is None:
                    continue
                ax.bar(k + (r - 0.5) * 0.38, max(c[m], 1e-3), width=0.34, color=shades[hz])
        ax.set_xticks(range(3), ["none", "feed-\nforward", "feed-\nback"], fontsize=8.5)
        ax.set_yscale("log")
        ax.set_title(lab, fontsize=9.5)
    fig.legend(handles=[Patch(color=c, label=f"{int(hz)} Hz control") for hz, c in shades.items()], ncols=2,
               loc="upper right", bbox_to_anchor=(0.98, 1.06), fontsize=9)
    fig.suptitle("Control rate: the teleop loop at 20 Hz (robosuite default) vs 50 Hz, workspace center", x=0.06,
                 ha="left", fontsize=10.5, y=1.04)  # fmt: skip
    save(fig, "control_rate.png")


def summary(cases):
    lines = ["# Nullspace evaluation summary", "", "Generated by `python -m nullspace.make_figures` from `results/cases.csv`.", ""]
    lines += ["## Workspace grid (gripper pointing down, 20 Hz)", ""]
    lines += ["| strategy | reachable | usable (of reachable) | settled pos median / 95th (mm) | settled rot median / 95th (deg) | peak pos median / 95th (mm) | peak rot median / 95th (deg) | back at start, worst (mm) |",
              "|---|---|---|---|---|---|---|---|"]  # fmt: skip
    for s in COLORS:
        g = grid_cases(cases, s)
        r = [c for c in g if c["reachable"]]
        u = [c for c in r if c["usable"]]
        q = lambda k: (np.median([c[k] for c in r]), np.percentile([c[k] for c in r], 95))
        sp, sr, pp, pr = (q(k) for k in ("settled_plate_pos_mm", "settled_plate_rot_deg", "peak_plate_pos_mm", "peak_plate_rot_deg"))
        lines.append(
            f"| {LABELS[s]} | {len(r)}/{len(g)} | {len(u)}/{len(r)} ({100 * len(u) / max(len(r), 1):.0f}%) | "
            f"{sp[0]:.2f} / {sp[1]:.2f} | {sr[0]:.2f} / {sr[1]:.2f} | {pp[0]:.2f} / {pp[1]:.2f} | {pr[0]:.2f} / {pr[1]:.2f} | "
            f"{max(c['final_plate_pos_mm'] for c in r):.2f} |"
        )
    lines += ["", "## Orientation study (workspace center, 20 Hz)", "",
              "| orientation | strategy | reachable | settled pos (mm) | settled rot (deg) | peak pos (mm) | peak rot (deg) | usable |",
              "|---|---|---|---|---|---|---|---|"]  # fmt: skip
    for c in sorted((c for c in cases if c["study"] == "orientation" and c["strategy"] != "none"),
                    key=lambda c: ("xyz".index(c["rot_axis"]), c["rot_deg"], c["strategy"])):  # fmt: skip
        lines.append(f"| {c['rot_axis']} {c['rot_deg']:+.0f}° | {c['strategy']} | {c['reachable']} | {c['settled_plate_pos_mm']:.2f} | "
                     f"{c['settled_plate_rot_deg']:.2f} | {c['peak_plate_pos_mm']:.2f} | {c['peak_plate_rot_deg']:.2f} | {'✓' if c['usable'] else '✗'} |")  # fmt: skip
    lines += ["", "## Control rate (workspace center)", "",
              "| strategy | rate | peak pos (mm) | peak rot (deg) | settled pos (mm) | settled rot (deg) | back at start (mm) |",
              "|---|---|---|---|---|---|---|"]  # fmt: skip
    for c in sorted((c for c in cases if c["study"] == "center"), key=lambda c: (list(COLORS).index(c["strategy"]), c["control_freq"])):
        lines.append(f"| {c['strategy']} | {c['control_freq']:.0f} Hz | {c['peak_plate_pos_mm']:.2f} | {c['peak_plate_rot_deg']:.2f} | "
                     f"{c['settled_plate_pos_mm']:.2f} | {c['settled_plate_rot_deg']:.2f} | {c['final_plate_pos_mm']:.3f} |")  # fmt: skip
    path = os.path.join(RESULTS, "summary.md")
    open(path, "w").write("\n".join(lines) + "\n")
    print("wrote nullspace/results/summary.md")
    print("\n".join(lines))


def main():
    cases = load_cases()
    fig_timeseries()
    fig_zoom()
    fig_maps(cases, "settled_plate_pos_mm", "settled plate position error (mm)", "workspace_settled_pos.png", 0.1, 50, SETTLED_POS_MM)
    fig_maps(cases, "settled_plate_rot_deg", "settled plate orientation error (deg)", "workspace_settled_rot.png", 0.02, 15, SETTLED_ROT_DEG)
    fig_maps(cases, "peak_plate_pos_mm", "peak (transient) plate position error (mm)", "workspace_peak_pos.png", 0.5, 60, None)
    fig_cdf(cases)
    fig_3d(cases)
    fig_orientation(cases)
    fig_rate(cases)
    summary(cases)


if __name__ == "__main__":
    import sys

    sys.path.insert(0, os.path.dirname(HERE))
    main()
