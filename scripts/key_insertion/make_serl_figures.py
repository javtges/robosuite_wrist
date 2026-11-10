"""
Figures of the SERL task (KeyInsertionWrist-v0) with the scripted oracle:

    cd scripts && python -m key_insertion.make_serl_figures

figures/serl_oracle.png   left: the push on the arm's flange vs the key's depth, one oracle episode per mode, over the
                          latch's force profile; right: the oracle's peak push when it aims off the keyhole (x error by
                          yaw error), stiff vs variable, with the episodes that pass the force limit marked
results/serl_oracle.npz   the data behind it
"""

import multiprocessing as mp
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from key_insertion.make_figures import AXIS, CRITICAL, FIGS, HERE, INK, INK2, MUTED, SURFACE  # noqa: E402,F401

RESULTS = os.path.join(HERE, "results", "serl_oracle.npz")
MODES = ("stiff", "variable", "soft")
COLORS = {"stiff": "#eb6834", "variable": "#7d4fd1", "soft": "#2a78d6"}
LABELS = {"stiff": "stiff (wrist contracted)", "variable": "variable (soft in, stiff to push)", "soft": "soft (wrist extended)"}
BLUES = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95"]
DX = np.array([-1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5])  # mm
DYAW = np.array([-3.0, -1.5, 0.0, 1.5, 3.0])  # deg


def profile(mode):
    """One oracle episode (no noise): per policy step, depth (mm) and peak flange push (N); plus the wrist z"""
    from key_insertion.serl_env import KeyInsertionSERL, ScriptedPolicy

    env = KeyInsertionSERL(mode=mode, seed=0)
    policy = ScriptedPolicy(env, noise=0.0, seed=0)
    obs, _ = env.reset(seed=1)
    depth, push, z = [], [], []
    done = truncated = False
    while not (done or truncated):
        obs, reward, done, truncated, info = env.step(policy(obs))
        depth.append(env.env.insertion_depth() * 1e3)
        push.append(info["max_force"])
        z.append(env.z)
    env.close()
    return mode, np.array(depth), np.array(push), np.array(z), reward


def grid(mode):
    """Oracle peak push (N) and outcome over aim errors DX x DYAW"""
    from key_insertion.serl_env import KeyInsertionSERL, ScriptedPolicy

    env = KeyInsertionSERL(mode=mode, seed=0)
    peak = np.zeros((len(DX), len(DYAW)))
    seated = np.zeros_like(peak, dtype=bool)
    for i, dx in enumerate(DX):
        for j, dyaw in enumerate(DYAW):
            policy = ScriptedPolicy(env, noise=0.0, aim_error=(dx * 1e-3, 0.0, np.radians(dyaw)), seed=0)
            obs, _ = env.reset(seed=1)
            done = truncated = False
            while not (done or truncated):
                obs, reward, done, truncated, info = env.step(policy(obs))
                peak[i, j] = max(peak[i, j], info["max_force"])
            seated[i, j] = info["succeed"]
    env.close()
    return mode, peak, seated


def compute():
    with mp.get_context("spawn").Pool(5) as pool:
        profiles = pool.map_async(profile, MODES)
        grids = pool.map_async(grid, ("stiff", "variable"))
        profiles, grids = profiles.get(), grids.get()
    data = {}
    for mode, depth, push, z, reward in profiles:
        data.update({f"{mode}_depth": depth, f"{mode}_push": push, f"{mode}_z": z, f"{mode}_seated": reward})
    for mode, peak, seated in grids:
        data.update({f"{mode}_grid_peak": peak, f"{mode}_grid_seated": seated})
    os.makedirs(os.path.dirname(RESULTS), exist_ok=True)
    np.savez(RESULTS, dx=DX, dyaw=DYAW, **data)
    return dict(np.load(RESULTS))


def plot(d):
    from key_insertion.serl_env import FORCE_LIMIT, LOCK

    fig = plt.figure(figsize=(12.5, 4.6))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.35, 1, 1], wspace=0.32)

    ax = fig.add_subplot(gs[0])
    plunger_top = 30.0 - LOCK["plunger_travel"] * 1e3  # the plunger is met at the seat depth minus its travel
    latch = np.array(LOCK["latch_profile"])
    ax.plot(plunger_top + latch[:, 0] * 1e3, latch[:, 1], color=MUTED, lw=1.5, ls="--", zorder=1)
    ax.annotate("latch's own force", (plunger_top + 2.5, 40.0), (plunger_top - 13.5, 47.0), color=MUTED, fontsize=9,
                arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.8))  # fmt: skip
    ax.axhline(FORCE_LIMIT, color=CRITICAL, lw=1.0, ls=":")
    ax.text(0.5, FORCE_LIMIT + 1.5, f"force limit {FORCE_LIMIT:.0f} N (episode fails)", color=CRITICAL, fontsize=9)
    ax.axvspan(30.0, 35.0, color="#f0efec", zorder=0, lw=0)
    ax.text(30.3, 62, "below the\nseat: box\nlimit for the\ntarget", color=MUTED, fontsize=8, va="top")
    for mode in MODES:
        depth, push = d[f"{mode}_depth"], d[f"{mode}_push"]
        keep = depth > -1.0
        ax.plot(depth[keep], push[keep], color=COLORS[mode], label=LABELS[mode], zorder=3)
    ax.set_xlim(0, 35)
    ax.set_ylim(0, 90)
    ax.set_xlabel("key tip depth below the lock's top face (mm)")
    ax.set_ylabel("push on the arm's flange (N)")
    ax.set_title("a. Oracle insertions, aimed at the keyhole")
    ax.legend(loc="upper left", bbox_to_anchor=(0.0, 0.86), fontsize=9)
    soft_end = d["soft_depth"].max()
    ax.annotate(f"soft: out of travel\nat {d['soft_push'].max():.0f} N", (soft_end, d["soft_push"].max()),
                (soft_end - 14, 28), color=COLORS["soft"], fontsize=9,
                arrowprops=dict(arrowstyle="-", color=COLORS["soft"], lw=0.8))  # fmt: skip

    vmax = max(d["stiff_grid_peak"].max(), d["variable_grid_peak"].max())
    levels = [35, 42, 50, 60, 70, 80, max(90, np.ceil(vmax))]
    cmap = matplotlib.colors.ListedColormap(BLUES)
    norm = matplotlib.colors.BoundaryNorm(levels, cmap.N, clip=True)
    for k, mode in enumerate(("stiff", "variable")):
        ax = fig.add_subplot(gs[k + 1])
        peak, seated = d[f"{mode}_grid_peak"], d[f"{mode}_grid_seated"].astype(bool)
        im = ax.imshow(peak, cmap=cmap, norm=norm, aspect="auto", origin="lower")
        for i in range(len(DX)):
            for j in range(len(DYAW)):
                over = peak[i, j] > FORCE_LIMIT
                color = SURFACE if peak[i, j] > levels[3] else INK
                ax.text(j, i, f"{peak[i, j]:.0f}" + ("\n✕" if over else ""), ha="center", va="center", fontsize=8,
                        color=color, fontweight="bold" if over else "normal")  # fmt: skip
        ax.set_xticks(range(len(DYAW)), [f"{v:g}" for v in DYAW])
        ax.set_yticks(range(len(DX)), [f"{v:g}" for v in DX])
        ax.grid(False)
        ax.set_xlabel("aim error, yaw (deg)")
        if k == 0:
            ax.set_ylabel("aim error, x (mm)")
        n_over = int((peak > FORCE_LIMIT).sum())
        ax.set_title(f"{'b' if k == 0 else 'c'}. {mode}: peak push (N)\n{n_over} of {peak.size} over the limit (✕)")
    cb = fig.colorbar(im, ax=fig.axes[1:], fraction=0.03, pad=0.02)
    cb.outline.set_visible(False)
    cb.set_label("peak push (N)", color=INK2)
    os.makedirs(FIGS, exist_ok=True)
    path = os.path.join(FIGS, "serl_oracle.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"wrote {os.path.relpath(path, os.path.dirname(HERE))}")


def main():
    plot(compute() if "--cached" not in sys.argv else dict(np.load(RESULTS)))


if __name__ == "__main__":
    main()
