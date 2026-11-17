"""
The force / torque at the arm's flange during an insertion, three ways, for each trained SERL policy (or the oracle):

    wrist   what the compliant wrist measures itself: its legs' spring + damper forces from their encoders
            (WristGripperModel.leg_wrench; the policies' F/T observation)
    arm     what the arm computes natively from its joint torques, carrying the wrist and key as a rigid payload
            (KeyInsertionSERL.arm_wrench, like a Franka's O_F_ext_hat_K)
    MuJoCo  the force the wrist actually puts on the flange (WristGripperModel.flange_wrench), for reference

All three in the flange frame, load compensated, sampled at every 50 Hz arm / wrist command update. Run in the SERL venv
from scripts/, after train_serl.py has saved runs/<mode>_seed<seed>/final/:

    python -m key_insertion.make_ft_figure                  # the trained policies, seed 0
    python -m key_insertion.make_ft_figure --oracle         # the scripted oracle instead (no training needed)
    python -m key_insertion.make_ft_figure --cached         # replot results/ft_comparison[_oracle].npz

figures/ft_comparison[_oracle].png   per mode (columns): key depth, then f_x, f_y, f_z, m_x, m_y, m_z over the episode
results/ft_comparison[_oracle].npz   the data behind it; the RMS differences are printed as a table
"""

import argparse
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from key_insertion.make_figures import FIGS, HERE, INK2, MUTED  # noqa: E402

MODES = ("stiff", "variable", "soft")
SOURCES = {"wrist": ("wrist (its legs)", "#2a78d6", 1.6, "-"), "arm": ("arm (joint torques)", "#eb6834", 1.6, "-"),
           "true": ("MuJoCo flange force", "#52514e", 1.0, "--")}  # fmt: skip
AXES = ["m_x", "m_y", "m_z", "f_x", "f_y", "f_z"]  # flange_wrench order: (m, f)
ROWS = [3, 4, 5, 0, 1, 2]  # plotted: forces first
RESET_SEED = 10000  # the first evaluation episode's start (train_serl evaluates with seed + 10000)


def load_policy(mode, seed, env, which="final"):
    """
    Deterministic action function of a trained policy: which="final" is runs/<mode>_seed<seed>/final (train_serl's
    actor saves it at the end), "latest" the learner's latest checkpoint (runs/<mode>_seed<seed>/checkpoints)
    """
    import jax
    from flax.training import checkpoints
    from serl_launcher.utils.launcher import make_sac_agent

    agent = make_sac_agent(seed=seed, sample_obs=env.observation_space.sample(), sample_action=env.action_space.sample())
    path = os.path.join(HERE, "runs", f"{mode}_seed{seed}", "final" if which == "final" else "checkpoints")
    state = checkpoints.restore_checkpoint(path, target=agent.state)
    assert state is not agent.state, f"no checkpoint in {path}"
    agent = agent.replace(state=state)
    return lambda obs: np.asarray(jax.device_get(agent.sample_actions(observations=jax.device_put(obs), argmax=True)))


def record(mode, oracle, seed):
    from key_insertion.serl_env import KeyInsertionSERL, ScriptedPolicy

    env = KeyInsertionSERL(mode=mode, seed=seed)
    policy = ScriptedPolicy(env, noise=0.0, seed=seed) if oracle else load_policy(mode, seed, env)
    rows = []

    def log():
        g, sim = env.gripper, env.env.sim
        rows.append(np.concatenate([g.leg_wrench(sim)[1], env.arm_wrench(), g.flange_wrench(sim)[1],
                                    [env.env.insertion_depth(), env.z]]))  # fmt: skip

    env.on_substep = log
    obs, _ = env.reset(seed=RESET_SEED)
    done = truncated = False
    while not (done or truncated):
        obs, reward, done, truncated, info = env.step(policy(obs))
    env.close()
    a = np.array(rows)
    dt = 1.0 / (env.env.control_freq)
    return dict(t=np.arange(len(a)) * dt, wrist=a[:, :6], arm=a[:, 6:12], true=a[:, 12:18], depth=a[:, 18], z=a[:, 19],
                seated=info["succeed"], over=info["over_force"])  # fmt: skip


def rms(x):
    return float(np.sqrt(np.mean(np.square(x))))


def table(d):
    print("| mode | outcome | peak flange force | wrist - MuJoCo, force / moment (RMS, max) | arm - MuJoCo | wrist - arm |")
    print("|---|---|---|---|---|---|")
    for mode in MODES:
        w, a, t = d[f"{mode}_wrist"], d[f"{mode}_arm"], d[f"{mode}_true"]
        peak = np.linalg.norm(t[:, 3:], axis=1).max()

        def diff(x, y):
            e = x - y
            ef, em = np.linalg.norm(e[:, 3:], axis=1), np.linalg.norm(e[:, :3], axis=1)
            return f"{rms(ef):.2f} N ({ef.max():.1f}) / {rms(em):.3f} N·m ({em.max():.2f})"

        outcome = "seated" if d[f"{mode}_seated"] else ("over force" if d[f"{mode}_over"] else "not seated")
        print(f"| {mode} | {outcome} | {peak:.1f} N | {diff(w, t)} | {diff(a, t)} | {diff(w, a)} |")


def plot(d, path, title):
    fig, axes = plt.subplots(7, 3, figsize=(13, 15), sharex="col", gridspec_kw=dict(height_ratios=[0.7] + [1] * 6))
    for c, mode in enumerate(MODES):
        t = d[f"{mode}_t"]
        ax = axes[0, c]
        ax.plot(t, d[f"{mode}_depth"] * 1e3, color=INK2, lw=1.4)
        ax.axhline(0, color=MUTED, lw=0.8)
        z = d[f"{mode}_z"]
        outcome = "seated" if d[f"{mode}_seated"] else ("over force" if d[f"{mode}_over"] else "not seated")
        ax.set_title(f"{mode}: {outcome}" + (f" (wrist z {z.min():+.2f} to {z.max():+.2f})" if mode == "variable" else ""))
        if c == 0:
            ax.set_ylabel("key depth (mm)")
        for r, k in enumerate(ROWS, start=1):
            ax = axes[r, c]
            for src, (label, color, lw, ls) in SOURCES.items():
                ax.plot(t, d[f"{mode}_{src}"][:, k], color=color, lw=lw, ls=ls, label=label)
            if c == 0:
                ax.set_ylabel(f"${AXES[k][0]}_{AXES[k][2]}$ ({'N' if k >= 3 else 'N·m'})")
        axes[-1, c].set_xlabel("time (s)")
    axes[1, 0].legend(loc="lower left", fontsize=9, frameon=False)
    fig.suptitle(title, y=0.995)
    fig.tight_layout()
    os.makedirs(FIGS, exist_ok=True)
    fig.savefig(path, dpi=130, bbox_inches="tight")
    print(f"wrote {os.path.relpath(path, os.path.dirname(HERE))}")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--oracle", action="store_true", help="the scripted oracle instead of the trained policies")
    p.add_argument("--seed", type=int, default=0, help="training seed of the policies")
    p.add_argument("--cached", action="store_true", help="replot the saved results")
    args = p.parse_args()
    suffix = "_oracle" if args.oracle else ""
    results = os.path.join(HERE, "results", f"ft_comparison{suffix}.npz")
    if args.cached:
        d = dict(np.load(results))
    else:
        d = {}
        for mode in MODES:
            d.update({f"{mode}_{k}": v for k, v in record(mode, args.oracle, args.seed).items()})
        os.makedirs(os.path.dirname(results), exist_ok=True)
        np.savez(results, **d)
    who = "the scripted oracle" if args.oracle else f"the trained SERL policies (seed {args.seed})"
    plot(d, os.path.join(FIGS, f"ft_comparison{suffix}.png"), f"Flange force / torque during one insertion by {who}")
    table(d)


if __name__ == "__main__":
    main()
