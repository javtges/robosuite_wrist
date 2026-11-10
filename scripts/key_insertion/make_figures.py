"""
Plots of the KeyInsertion oracle results (run evaluate.py first):

    cd scripts && python -m key_insertion.make_figures

figures/force_profiles.png    push force on the arm's flange vs key depth and vs time, and the wrist's deflection,
                              for the stiff / home / soft wrist, without and with a perception error
figures/success_maps.png      outcome and peak force over the perception-error grids, per wrist
figures/compliance_sweep.png  success rate and forces vs the wrist's leg stiffness
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
from key_insertion.evaluate import PROFILE_ERROR, RESULTS  # noqa: E402
from key_insertion.oracle import KeyOracle  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
FIGS = os.path.join(HERE, "figures")

# Reference palette (dataviz skill), as in nullspace/make_figures.py. The home wrist is the neutral reference.
SURFACE, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
WRISTS = ("stiff", "home", "soft")
COLORS = {"stiff": "#eb6834", "home": "#898781", "soft": "#2a78d6"}
LABELS = {"stiff": "stiff (wrist contracted)", "home": "home", "soft": "soft (wrist extended)"}
# Status colors (outcomes), never reused for series
GOOD, WARNING, CRITICAL = "#0ca30c", "#e6a100", "#d03b3b"
OUTCOMES = {"seated": GOOD, "stuck": WARNING, "jammed": CRITICAL}

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans"],
    "font.size": 10, "text.color": INK, "axes.labelcolor": INK2, "axes.edgecolor": AXIS,
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False, "lines.linewidth": 2.0, "legend.frameon": False,
    "axes.titlesize": 11, "axes.titleweight": "bold", "axes.titlelocation": "left",
})  # fmt: skip


def load_rows():
    rows = []
    for r in csv.DictReader(open(os.path.join(RESULTS, "episodes.csv"))):
        if r["result"] == "error":
            continue
        for k, v in r.items():
            if v in ("True", "False"):
                r[k] = v == "True"
            elif k not in ("study", "wrist", "result", "error_case", "error") and v != "":
                r[k] = float(v)
        rows.append(r)
    return rows


def save(fig, name):
    os.makedirs(FIGS, exist_ok=True)
    path = os.path.join(FIGS, name)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {os.path.relpath(path, os.path.dirname(HERE))}")


def env_geometry():
    """The KeyInsertion defaults the figures annotate (chamfer, plunger zone, plunger force)"""
    import inspect

    from robosuite.environments.manipulation.key_insertion import KeyInsertion

    d = {k: v.default for k, v in inspect.signature(KeyInsertion.__init__).parameters.items() if v.default is not v.empty}
    return d


def fig_profiles():
    P = np.load(os.path.join(RESULTS, "profiles.npz"))
    g = env_geometry()
    D, travel, ch = g["slot_depth"] * 1e3, g["plunger_travel"] * 1e3, g["chamfer"] * 1e3
    dx, dy, dyaw = PROFILE_ERROR
    fig, axes = plt.subplots(2, 3, figsize=(15, 8.2), constrained_layout=True)
    for row, (case, title) in enumerate(
        (("none", "no perception error"),
         ("offset", f"keyhole {dx * 1e3:g} mm and {np.degrees(dyaw):g}° off"))  # fmt: skip
    ):
        ax_d, ax_t, ax_w = axes[row]
        for ax in (ax_d,):
            ax.axvspan(0, ch, color=GRID, alpha=0.6, lw=0)
            ax.axvspan(D - travel, D, color=GRID, alpha=0.6, lw=0)
            ax.text(ch / 2, 1.0, "chamfer", transform=ax.get_xaxis_transform(), ha="center", va="bottom", color=MUTED, fontsize=8)
            ax.text(D - travel / 2, 1.0, "plunger", transform=ax.get_xaxis_transform(), ha="center", va="bottom", color=MUTED, fontsize=8)
            # The plunger's own spring: what pushing it down takes, with no friction
            ax.plot([D - travel, D], [g["seat_preload"], g["seat_force"]], ls="--", lw=1.2, color=INK2)
            ax.annotate("plunger spring", (D - travel / 2, (g["seat_preload"] + g["seat_force"]) / 2), xytext=(-75, 22),
                        textcoords="offset points", color=INK2, fontsize=8, arrowprops=dict(arrowstyle="-", color=AXIS))
        for w in WRISTS:
            k = f"{w}/{case}/"
            ph, t = P[k + "phase"], P[k + "t"]
            sel = ph >= KeyOracle.INSERT
            push = -P[k + "flange_force"][:, 2]  # flange frame +z points toward the key: pushing reads negative
            depth = P[k + "depth"] * 1e3
            t0 = t[sel][0]
            ins = sel & (ph == KeyOracle.INSERT)
            ax_d.plot(depth[ins], push[ins], color=COLORS[w], label=LABELS[w])
            ax_t.plot(t[sel] - t0, push[sel], color=COLORS[w], label=LABELS[w])
            defl = P[k + "wrist_deflection"]
            ax_w.plot(t[sel] - t0, np.linalg.norm(defl[sel, :2], axis=1) * 1e3, color=COLORS[w], label=f"{w}: lateral")
            ax_w.plot(t[sel] - t0, np.abs(defl[sel, 2]) * 1e3, color=COLORS[w], ls=":", lw=1.6, label=f"{w}: axial")
            # Where it ended
            if P[k + "success"].any():
                i = np.flatnonzero(P[k + "success"])[0]
                ax_t.plot(t[i] - t0, push[i], "o", ms=8, color=COLORS[w], mec=SURFACE, mew=2)
            else:
                ax_t.plot(t[sel][-1] - t0, push[sel][-1], "X", ms=10, color=COLORS[w], mec=SURFACE, mew=1.5)
                ax_d.plot(depth[ins][-1], push[ins][-1], "X", ms=10, color=COLORS[w], mec=SURFACE, mew=1.5)
        ax_t.axhline(40.0, color=AXIS, lw=1, ls=":")
        ax_t.text(0.0, 40.6, "oracle's force stop", color=MUTED, fontsize=8)
        ax_d.set(xlim=(-3, D + 1.5), xlabel="key tip depth below the lock's top face (mm)", ylabel="push on the arm's flange (N)")
        ax_t.set(xlabel="time since the insertion started (s)", ylabel="push on the arm's flange (N)")
        ax_w.set(xlabel="time since the insertion started (s)", ylabel="wrist deflection from its command (mm)")
        ax_d.set_title(f"{'a' if row == 0 else 'd'}.  force vs depth: {title}", loc="left")
        ax_t.set_title(f"{'b' if row == 0 else 'e'}.  force vs time (● seated, ✕ gave up)")
        ax_w.set_title(f"{'c' if row == 0 else 'f'}.  how far the wrist gives (solid lateral, dotted axial)")
        for ax in (ax_d, ax_t):
            ax.set_ylim(bottom=-1.5)
    axes[0, 0].legend(loc="upper left")
    fig.suptitle(
        "KeyInsertion: what the arm feels while the oracle pushes the key in (force-blind, 10 mm/s), per wrist stiffness",
        x=0.01, ha="left", fontsize=12, fontweight="bold",
    )
    save(fig, "force_profiles.png")


def fig_maps(rows):
    fig, axes = plt.subplots(2, 3, figsize=(14, 9.0), constrained_layout=True)
    cmap = ListedColormap([OUTCOMES["seated"], OUTCOMES["stuck"], OUTCOMES["jammed"]])
    code = {"seated": 0, "stuck": 1, "jammed": 2, "timeout": 1}
    for r_i, (study, ykey, ylabel) in enumerate(
        (("grid_yaw", "dyaw_deg", "yaw error (deg)"), ("grid_xy", "dy_mm", "error along the slot, y (mm)"))
    ):
        for c_i, w in enumerate(WRISTS):
            ax = axes[r_i, c_i]
            sel = [r for r in rows if r["study"] == study and r["wrist"] == w]
            xs = sorted({r["dx_mm"] for r in sel})
            ys = sorted({r[ykey] for r in sel})
            M = np.full((len(ys), len(xs)), np.nan)
            F = np.full_like(M, np.nan)
            for r in sel:
                i, j = ys.index(r[ykey]), xs.index(r["dx_mm"])
                M[i, j] = code[r["result"]]
                F[i, j] = r["peak_flange_N"]
            dxs, dys = np.diff(xs).mean() / 2, np.diff(ys).mean() / 2
            ax.imshow(M, cmap=cmap, vmin=-0.5, vmax=2.5, origin="lower", aspect="auto", alpha=0.85,
                      extent=(xs[0] - dxs, xs[-1] + dxs, ys[0] - dys, ys[-1] + dys))  # fmt: skip
            for i, y in enumerate(ys):
                for j, x in enumerate(xs):
                    ax.text(x, y, f"{F[i, j]:.0f}", ha="center", va="center", fontsize=7, color=SURFACE, fontweight="bold")
            ax.grid(False)
            n_ok = int(np.sum(M == 0))
            ax.set_title(f"{LABELS[w]}: {n_ok}/{M.size} seated")
            ax.set(xlabel="error across the slot, x (mm)", ylabel=ylabel if c_i == 0 else None)
    handles = [Patch(color=c, label=o) for o, c in OUTCOMES.items()]
    fig.legend(handles=handles, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.04))
    fig.suptitle(
        "Where the force-blind oracle still seats the key, given a perception error (cell text: peak flange force, N)",
        x=0.01, ha="left", fontsize=12, fontweight="bold",
    )
    save(fig, "success_maps.png")


def fig_sweep(rows):
    sel = [r for r in rows if r["study"] == "sweep"]
    zs = sorted({r["wrist_z"] for r in sel})
    k = np.array([next(r["leg_k"] for r in sel if r["wrist_z"] == z) for z in zs])
    rate = np.array([np.mean([r["success"] for r in sel if r["wrist_z"] == z]) for z in zs]) * 100
    nominal = [next(r for r in sel if r["wrist_z"] == z and r["dx_mm"] == 0 and r["dyaw_deg"] == 0) for z in zs]
    peak_ok = np.array([np.median([r["peak_flange_N"] for r in sel if r["wrist_z"] == z and r["success"]] or [np.nan]) for z in zs])
    peak_nom = np.array([r["peak_flange_N"] for r in nominal])
    push = np.array([r["hold_push_N"] for r in nominal])
    defl = np.array([r["max_wrist_axial_mm"] for r in nominal])
    n_err = len({(r["dx_mm"], r["dyaw_deg"]) for r in sel})

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), constrained_layout=True)
    ax = axes[0]
    ax.plot(k, rate, "o-", color=COLORS["soft"], ms=7, mec=SURFACE, mew=2)
    ax.set(xscale="log", ylim=(-5, 105), ylabel="seated (%)", xlabel="leg stiffness at the held pose (N/m)")
    ax.set_title(f"a.  success over {n_err} perception errors")
    ax = axes[1]
    ax.plot(k, peak_nom, "o-", color=INK2, ms=7, mec=SURFACE, mew=2, label="no error")
    ax.plot(k, peak_ok, "s--", color=COLORS["soft"], ms=6, mec=SURFACE, mew=1.5, label="with error (seated episodes, median)")
    ax.plot(k, push, "o:", color=MUTED, ms=5, label="no error, while seated (steady)")
    ax.set(xscale="log", ylim=(0, None), ylabel="push on the arm's flange (N)", xlabel="leg stiffness at the held pose (N/m)")
    ax.set_title("b.  peak force on the arm")
    ax.legend(loc="upper right", fontsize=8)
    ax = axes[2]
    ax.plot(k, defl, "o-", color=INK2, ms=7, mec=SURFACE, mew=2)
    ax.set(xscale="log", ylim=(0, None), ylabel="max axial wrist deflection (mm)", xlabel="leg stiffness at the held pose (N/m)")
    ax.set_title("c.  how far the wrist compresses (no error)")
    for ax in axes:
        for w, z in (("stiff", -1.0), ("home", 0.0), ("soft", 1.0)):
            if z in zs:
                ax.axvline(k[zs.index(z)], color=COLORS[w], lw=1, alpha=0.6)
                ax.text(k[zs.index(z)], 0.02, f" {w}", transform=ax.get_xaxis_transform(), color=COLORS[w], ha="left",
                        va="bottom", fontsize=8)  # fmt: skip
    fig.suptitle(
        "Wrist compliance vs the key insertion: wrist z swept from fully contracted (stiff) to fully extended (soft)",
        x=0.01, ha="left", fontsize=12, fontweight="bold",
    )
    save(fig, "compliance_sweep.png")


def main():
    rows = load_rows()
    fig_profiles()
    fig_maps(rows)
    fig_sweep(rows)


if __name__ == "__main__":
    main()
