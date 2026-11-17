"""
The trained SERL policies (stiff and variable) inserting with an aim error that lasts into contact: each policy sees
the key relative to a keyhole shifted by (dx, dyaw) from the real one (KeyInsertionSERL.aim_error), as with a
mis-located lock, so it lines up off by that much and only contact can tell it otherwise. The grid is the oracle's in
make_serl_figures.py (x -1.5 ... 1.5 mm by yaw -3 ... 3 deg), each case from the same starts. Run in the SERL venv from
scripts/:

    python -m key_insertion.aim_error_eval                  # the final policies (runs/<mode>_seed0/final)
    python -m key_insertion.aim_error_eval --latest         # the learners' latest checkpoints (while training)
    python -m key_insertion.aim_error_eval --cached         # replot results/aim_error_policies.npz

figures/aim_error_policies.png   per policy: peak flange push over the grid; ✕ over the force limit, ○ not seated;
                                 scored with the 50 Hz force the reward sees (top) and the 2 ms physics-step peak (bottom)
results/aim_error_policies.npz   the data behind it
"""

import argparse
import multiprocessing as mp
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from key_insertion.make_figures import FIGS, HERE, INK, INK2, SURFACE  # noqa: E402
from key_insertion.make_serl_figures import BLUES, DX, DYAW  # noqa: E402

MODES = ("stiff", "variable")
RESET_SEEDS = (10000, 10001, 10002)  # starts per grid case (train_serl evaluates from seed + 10000 on)
# success / over / peak: as the reward and force limit see it (flange force sampled at 50 Hz); *_physics: the same
# with the peak over every physics step (2 ms), which also catches spikes between the 50 Hz samples
FIELDS = ("success", "seated", "over", "peak", "success_physics", "over_physics", "peak_physics", "steps", "z_mean", "z_contact")


def evaluate(args):
    mode, seed, which = args
    from key_insertion.make_ft_figure import load_policy
    from key_insertion.serl_env import KeyInsertionSERL

    env = KeyInsertionSERL(mode=mode, seed=seed)
    policy = load_policy(mode, seed, env, which)
    out = {f: np.zeros((len(DX), len(DYAW), len(RESET_SEEDS))) for f in FIELDS}
    for i, dx in enumerate(DX):
        for j, dyaw in enumerate(DYAW):
            env.aim_error = np.array([dx * 1e-3, 0.0, np.radians(dyaw)])
            for k, reset_seed in enumerate(RESET_SEEDS):
                obs, _ = env.reset(seed=reset_seed)
                done = truncated = False
                peak, peak_physics, zs, z_contact = 0.0, 0.0, [], np.nan
                while not (done or truncated):
                    obs, reward, done, truncated, info = env.step(policy(obs))
                    peak = max(peak, info["max_force"])
                    peak_physics = max(peak_physics, info["max_force_physics"])
                    zs.append(env.z)
                    if np.isnan(z_contact) and info["max_force"] > 5.0:
                        z_contact = env.z  # the wrist's z when the key first pushes on something
                over_physics = peak_physics > env.force_limit
                row = dict(success=info["succeed"], seated=info["seated"], over=info["over_force"], peak=peak,
                           success_physics=info["seated"] and not over_physics, over_physics=over_physics,
                           peak_physics=peak_physics, steps=env.steps, z_mean=np.mean(zs), z_contact=z_contact)  # fmt: skip
                for f in FIELDS:
                    out[f][i, j, k] = row[f]
    env.close()
    return mode, out


SCORINGS = (("", "flange force sampled at 50 Hz (what the reward and limit see)"),
            ("_physics", "peak over every 2 ms physics step (true)"))


def plot(d, path):
    from key_insertion.serl_env import FORCE_LIMIT

    fig, axes = plt.subplots(2, len(MODES), figsize=(10.5, 9.0), constrained_layout=True)
    peaks = {(m, k): d[f"{m}_peak{k}"].mean(axis=2) for m in MODES for k, _ in SCORINGS}
    levels = [35, 42, 50, 60, 70, 80, max(90, np.ceil(max(v.max() for v in peaks.values())))]
    cmap = matplotlib.colors.ListedColormap(BLUES)
    norm = matplotlib.colors.BoundaryNorm(levels, cmap.N, clip=True)
    n = len(RESET_SEEDS)
    for r, (key, label) in enumerate(SCORINGS):
        for c, mode in enumerate(MODES):
            ax, peak = axes[r, c], peaks[(mode, key)]
            success = d[f"{mode}_success{key}"].sum(axis=2)
            over = d[f"{mode}_over{key}"].sum(axis=2)
            im = ax.imshow(peak, cmap=cmap, norm=norm, aspect="auto", origin="lower")
            for i in range(len(DX)):
                for j in range(len(DYAW)):
                    marks = "✕" * int(over[i, j]) + "○" * int(n - success[i, j] - over[i, j])
                    color = SURFACE if peak[i, j] > levels[3] else INK
                    ax.text(j, i, f"{peak[i, j]:.0f}" + (f"\n{marks}" if marks else ""), ha="center", va="center",
                            fontsize=8, color=color, fontweight="bold" if marks else "normal")  # fmt: skip
            ax.set_xticks(range(len(DYAW)), [f"{v:g}" for v in DYAW])
            ax.set_yticks(range(len(DX)), [f"{v:g}" for v in DX])
            ax.grid(False)
            if r == 1:
                ax.set_xlabel("aim error, yaw (deg)")
            if c == 0:
                ax.set_ylabel(f"{label.split(' (')[0]}\naim error, x (mm)")
            total = success.size * n
            ax.set_title(f"{mode}: seated within {FORCE_LIMIT:.0f} N {int(success.sum())}/{total}, over {int(over.sum())}")
    cb = fig.colorbar(im, ax=axes, fraction=0.03, pad=0.02)
    cb.outline.set_visible(False)
    cb.set_label(f"peak push (N), mean of {n} starts", color=INK2)
    fig.suptitle("Trained SERL policies with an aim error into contact (✕ over the force limit, ○ not seated)\n"
                 "top: flange force sampled at 50 Hz, as the reward and limit see it; bottom: peak over every 2 ms physics step")  # fmt: skip
    os.makedirs(FIGS, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"wrote {os.path.relpath(path, os.path.dirname(HERE))}")


def table(d):
    print("| policy | force scored | seated within 80 N | over 80 N | not seated | median peak push, successes | median wrist z at first contact |")
    print("|---|---|---|---|---|---|---|")
    for mode in MODES:
        for key, label in SCORINGS:
            s, o = d[f"{mode}_success{key}"].astype(bool), d[f"{mode}_over{key}"].astype(bool)
            peaks = d[f"{mode}_peak{key}"][s]
            print(f"| {mode} | {'50 Hz' if not key else '2 ms'} | {s.sum()}/{s.size} | {o.sum()} | {(~s & ~o).sum()} | "
                  f"{np.median(peaks) if peaks.size else float('nan'):.1f} N | {np.nanmedian(d[f'{mode}_z_contact']):+.2f} |")  # fmt: skip


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--seed", type=int, default=0, help="training seed of the policies")
    p.add_argument("--latest", action="store_true", help="the learners' latest checkpoints instead of the final policies")
    p.add_argument("--cached", action="store_true", help="replot the saved results")
    args = p.parse_args()
    suffix = "_latest" if args.latest else ""
    results = os.path.join(HERE, "results", f"aim_error_policies{suffix}.npz")
    if args.cached:
        d = dict(np.load(results))
    else:
        which = "latest" if args.latest else "final"
        with mp.get_context("spawn").Pool(len(MODES)) as pool:
            outs = pool.map(evaluate, [(m, args.seed, which) for m in MODES])
        d = {f"{mode}_{f}": v for mode, out in outs for f, v in out.items()}
        os.makedirs(os.path.dirname(results), exist_ok=True)
        np.savez(results, dx=DX, dyaw=DYAW, reset_seeds=np.array(RESET_SEEDS), **d)
    plot(d, os.path.join(FIGS, f"aim_error_policies{suffix}.png"))
    table(d)


if __name__ == "__main__":
    main()
