"""
State-action entropy of the same evaluation trajectories the Gramian analysis used, for the rigid, most flexible and
variable wrist: the SERL key-in-lock rollouts (record_rollouts.py, SAC's sampled actions) and the KeyDoor scheduled
expert's episodes (door_gramian.py). numpy + scipy + matplotlib only; reads the saved data, runs no simulation.

What is estimated, per condition and per phase (and whole trajectory), in nats:
  H(x, u)     joint state-action entropy: the entropy of the distribution of visited (state, action) pairs
  H(x)        entropy of the visited states
  H(u)        entropy of the actions taken
for the full state and for the same blocks as the Gramian (tip pose, wrench; door: + door/cylinder angle), and
  H(pi)       key-in-lock only: the SAC policy's own action entropy, exact (policy_entropy.py), averaged over the
              visited observations: how random the action is once the policy has seen its observation. The door
              expert is deterministic, so its H(pi) is -inf by construction.

States and actions are in the Gramian analysis's normalized units (state: 1 cm, 0.1 rad, 5 cm/s, 2 N m, 20 N,
door/cylinder 0.1 rad; actions: per-step command changes of 2 mm and 1 deg, the SERL action units). Differential
entropy depends on units: rescaling one coordinate by s shifts H by ln s, so only same-unit comparisons mean anything.

Estimator: Kozachenko-Leonenko k-nearest-neighbour entropy (k = 4, Euclidean). Its bias grows with dimension
(Gaussian check: about +0 nats at 4-6 dims, +3 to +6 at 15-23 dims), so within each group (phase, depth bin, ...)
every condition is evaluated at the same sample size n and only same-block, same-group values are compared.
Not reported: H(u | x) from k-NN estimates. Differencing H(x, u) - H(x) gave H(u|x) > H(u) where state coordinates
are nearly constant (the wrench in free space), and the KSG mutual-information estimator read ~1 nat for a true 3.7 in
15 dims; for the SAC policy the exact H(pi) replaces it.
Uncertainty: D random draws, each keeping 70 % of the episodes (without replacement: a bootstrap would duplicate rows,
and duplicated rows break nearest-neighbour estimates) and then n rows of those; points are medians over draws and
bars the 5-95 % range. Each coordinate gets uniform dither of 1e-3 units (a resolution floor, e.g. 10 um, 20 mN), so a
coordinate that never changes (e.g. the door angle before the key engages) contributes ln(1e-3) per dimension instead
of -inf. Comparisons use the motion actions every condition has (key-in-lock: x, y, z, yaw; door: the arm's 6); the
variable wrist's z command is reported as an extra variant.

Run from robosuite_wrist/scripts/key_insertion/gramian with robosuite_env:
    python entropy.py [--rollouts data/rollouts] [--door data/door] [--out results] [--draws 40]
"""

import argparse
import glob
import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy.spatial import cKDTree  # noqa: E402
from scipy.spatial.transform import Rotation  # noqa: E402
from scipy.special import digamma, gammaln  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gramian_lib import COLORS, LABEL, MODES, PHASES, load  # noqa: E402

K_NN = 4
DITHER = 1e-3
KEEP_EPISODES = 0.7
MIN_ROWS = 40  # a condition with fewer rows in a group is left out of that group
MAX_N = 2000

QLABEL = {"H_x": "H(x)  states", "H_u": "H(u)  actions", "H_xu": "H(x, u)  state-action",
          "H_pi": "H(pi)  policy action entropy given its observation (exact)"}


def kl_entropy(Z, k=K_NN):
    """Kozachenko-Leonenko differential entropy (nats), Euclidean k-NN"""
    n, d = Z.shape
    eps = cKDTree(Z).query(Z, k + 1, workers=-1)[0][:, k]
    log_cd = d / 2 * np.log(np.pi) - gammaln(d / 2 + 1)
    return float(digamma(n) - digamma(k) + log_cd + d * np.mean(np.log(np.maximum(eps, 1e-300))))


def entropies(X, U, blocks, hpi=None):
    """{block: {quantity: H}} for one sample (already dithered); H_u and H_pi are the same for every block"""
    hu = kl_entropy(U)
    out = {}
    for b, idx in blocks.items():
        out[b] = dict(H_x=kl_entropy(X[:, idx]), H_u=hu, H_xu=kl_entropy(np.c_[X[:, idx], U]))
        if hpi is not None:
            out[b]["H_pi"] = float(np.mean(hpi))
    return out


def compare(groups, blocks, draws, rng):
    """
    groups: {group: {mode: (X, U, ep, H_pi per row or None)}}. For each group, every mode with enough rows is evaluated at one common n.
    Returns {group: {mode: {block: {quantity: [p5, median, p95]}}, "n": n}}
    """
    res = {}
    for g, per_mode in groups.items():
        ok = {m: v for m, v in per_mode.items() if len(v[0]) >= MIN_ROWS}
        if not ok:
            continue
        # common n: what a 70 % episode draw leaves for the smallest condition (taken as 0.6 of its rows)
        n = int(min(MAX_N, min(0.6 * len(v[0]) for v in ok.values())))
        if n < 25:
            continue
        res[g] = {"n": n}
        for m, (X, U, ep, hpi) in ok.items():
            eps_u = np.unique(ep)
            acc = {}
            for _ in range(draws):
                keep = rng.choice(eps_u, max(1, int(round(KEEP_EPISODES * len(eps_u)))), replace=False)
                rows = np.where(np.isin(ep, keep))[0]
                if len(rows) < n:
                    rows = np.arange(len(ep))  # few long episodes: fall back to row subsampling
                rows = rng.choice(rows, n, replace=False)
                Xs = X[rows] + rng.uniform(-DITHER / 2, DITHER / 2, X[rows].shape)
                Us = U[rows] + rng.uniform(-DITHER / 2, DITHER / 2, U[rows].shape)
                for b, qs in entropies(Xs, Us, blocks, None if hpi is None else hpi[rows]).items():
                    for q, v in qs.items():
                        acc.setdefault(b, {}).setdefault(q, []).append(v)
            res[g][m] = {b: {q: [float(np.percentile(v, 5)), float(np.median(v)), float(np.percentile(v, 95))]
                             for q, v in qs.items()} for b, qs in acc.items()}
    return res


def plot_by_group(res, groups, glabels, blocks, blabel, title, path):
    """rows: H(x_S, u), H(x_S); columns: state blocks, plus H(u) and (key-in-lock) H(pi) in the last column"""
    rows = ("H_xu", "H_x")
    cols = list(blocks) + ["actions"]
    fig, axes = plt.subplots(len(rows), len(cols), figsize=(4.1 * len(cols), 3.3 * len(rows)), constrained_layout=True)
    for r, q in enumerate(rows):
        for c, b in enumerate(cols):
            ax = axes[r, c]
            qq, bb = (("H_u", "H_pi")[r], cols[0]) if b == "actions" else (q, b)
            if not any(g in res and m in res[g] and qq in res[g][m][bb] for g in groups for m in MODES):
                ax.axis("off")
                continue
            for j, m in enumerate(MODES):
                xs, ys, lo, hi = [], [], [], []
                for gi, g in enumerate(groups):
                    if g in res and m in res[g]:
                        p5, med, p95 = res[g][m][bb][qq]
                        xs.append(gi + (j - 1) * 0.22)
                        ys.append(med)
                        lo.append(med - p5)
                        hi.append(p95 - med)
                ax.errorbar(xs, ys, yerr=[lo, hi], fmt="o", color=COLORS[m], label=LABEL[m], capsize=3, ms=5)
            ax.set_xticks(range(len(groups)), [glabels.get(g, g) for g in groups], fontsize=8)
            ax.set_title(QLABEL[qq] + ("" if b == "actions" else f"\n{blabel[b]}"), fontsize=9)
            ax.grid(alpha=0.3)
    axes[0, 0].set_ylabel("nats")
    axes[0, 0].legend(fontsize=7)
    ns = ", ".join(f"{glabels.get(g, g).replace(chr(10), ' ')} {res[g]['n']}" for g in groups if g in res)
    fig.suptitle(f"{title}\nk-NN estimates at equal sample size per group (n: {ns}); median and 5-95 % over episode draws",
                 fontsize=10)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_series(res, keys, xvals, xlabel, blocks0, title, path, vlines=()):
    qs = ("H_xu", "H_x", "H_u", "H_pi")
    fig, axes = plt.subplots(1, len(qs), figsize=(4.6 * len(qs), 4.0), constrained_layout=True)
    for ax, q in zip(axes, qs):
        for m in MODES:
            pts = [(xv, res[k][m][blocks0][q]) for k, xv in zip(keys, xvals)
                   if k in res and m in res[k] and q in res[k][m][blocks0]]
            if not pts:
                continue
            xv = np.array([p[0] for p in pts])
            v = np.array([p[1] for p in pts])
            ax.plot(xv, v[:, 1], "-o", color=COLORS[m], label=LABEL[m], ms=4)
            ax.fill_between(xv, v[:, 0], v[:, 2], color=COLORS[m], alpha=0.15)
        for xv, lab in vlines:
            ax.axvline(xv, color="0.6", lw=0.8, ls="--")
            ax.text(xv, ax.get_ylim()[1], lab, fontsize=7, rotation=90, va="top", ha="right", color="0.4")
        ax.set_xlabel(xlabel)
        ax.set_title(QLABEL[q], fontsize=9)
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("nats")
    axes[0].legend(fontsize=7)
    fig.suptitle(title, fontsize=10)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_outcome(res, block, title, path):
    qs = [q for q in ("H_xu", "H_x", "H_u", "H_pi") if any(q in v[block] for k in ("success", "failure") if k in res
                                                          for m, v in res[k].items() if m != "n")]
    fig, axes = plt.subplots(1, len(qs), figsize=(4.6 * len(qs), 3.8), constrained_layout=True)
    for ax, q in zip(axes, qs):
        for j, m in enumerate(MODES):
            for o, (key, mk) in enumerate((("success", "o"), ("failure", "s"))):
                if key in res and m in res[key] and q in res[key][m][block]:
                    p5, med, p95 = res[key][m][block][q]
                    x = j + (o - 0.5) * 0.3
                    ax.errorbar([x], [med], yerr=[[med - p5], [p95 - med]], fmt=mk, color=COLORS[m], capsize=3,
                                mfc=COLORS[m] if key == "success" else "none", ms=6,
                                label=key if j == 0 or (key == "success" and m == "variable") else None)
        ax.set_xticks(range(3), [LABEL[m].split(" (")[0] for m in MODES], fontsize=8)
        ax.set_title(QLABEL[q], fontsize=9)
        ax.grid(alpha=0.3)
    h, lab = axes[0].get_legend_handles_labels()
    seen = {}
    for hh, ll in zip(h, lab):
        seen.setdefault(ll, hh)
    axes[0].legend(seen.values(), [f"{k} (filled: succeeded)" if k == "success" else k for k in seen], fontsize=7)
    axes[0].set_ylabel("nats")
    n = {k: res[k]["n"] for k in ("success", "failure") if k in res}
    fig.suptitle(f"{title} (n per condition: {n})", fontsize=10)
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------------------------------------------
def key_in_lock(rollouts, out, draws, rng):
    blocks = {"all": list(range(15)), "pose": list(range(6)), "wrench": list(range(9, 15))}
    blabel = {"all": "all 15 states", "pose": "tip pose block (6)", "wrench": "wrench block (6)"}
    data = {m: load(rollouts, m) for m in MODES}
    for m, d in data.items():  # exact policy entropy per row (policy_entropy.py), matched on (episode, step)
        pe = np.load(os.path.join(rollouts, f"policy_entropy_{m}.npz"))
        idx = {(int(e), int(t)): i for i, (e, t) in enumerate(zip(pe["ep"], pe["t"]))}
        d["H_pi"] = pe["H_dim"][[idx[(int(e), int(t))] for e, t in zip(d["ep"], d["t"])]]
    succ = {}
    for m, d in data.items():
        by_id = dict(zip(d["ep_id"], d["ep_success"]))
        succ[m] = np.array([bool(by_id[e]) for e in d["ep"]])

    def sample(m, sel, n_inputs=4):
        d = data[m]
        return d["x"][sel][:, :15], d["u"][sel][:, :n_inputs], d["ep"][sel], d["H_pi"][sel][:, :n_inputs].sum(1)

    rep = {}
    # 1. whole trajectory and phases
    groups = {"whole": {m: sample(m, np.ones(len(d["ep"]), bool)) for m, d in data.items()}}
    for i, p in enumerate(PHASES):
        groups[p] = {m: sample(m, d["phase"] == i) for m, d in data.items()}
    rep["by_phase"] = compare(groups, blocks, draws, rng)
    plot_by_group(rep["by_phase"], ["whole", *PHASES], {"whole": "whole\ntrajectory"}, blocks, blabel,
                  "Key-in-lock (SERL policy, sampled actions): state-action entropy by phase",
                  os.path.join(out, "entropy_by_phase.png"))

    # 2. depth bins
    bins = np.r_[-0.012, -0.004, 0.0, 0.002, 0.005, 0.010, 0.015, 0.020, 0.025, 0.031]
    groups = {}
    for lo, hi in zip(bins[:-1], bins[1:]):
        groups[f"{lo * 1e3:.0f}..{hi * 1e3:.0f}"] = {m: sample(m, (d["depth"] >= lo) & (d["depth"] < hi))
                                                    for m, d in data.items()}
    rep["by_depth"] = compare(groups, blocks, draws, rng)
    plot_series(rep["by_depth"], list(groups), (bins[:-1] + bins[1:]) / 2 * 1e3, "key-tip depth (mm)", "all",
                "Key-in-lock: state-action entropy by depth (all 15 states, 4 motion actions; equal n per bin)",
                os.path.join(out, "entropy_vs_depth.png"), vlines=((0, "lock face"), (2, "chamfer end"), (20, "latch")))

    # 3. variable wrist: within contact phases, by the z command (terciles), with and without z as an action
    d = data["variable"]
    zb = {}
    for i, p in enumerate(PHASES[1:], start=1):
        s = d["phase"] == i
        edges = np.quantile(d["zcmd"][s], [0, 1 / 3, 2 / 3, 1])
        for t in range(3):
            sel = s & (d["zcmd"] >= edges[t]) & ((d["zcmd"] < edges[t + 1]) if t < 2 else (d["zcmd"] <= edges[t + 1]))
            zb[(p, t)] = (sel, float(np.median(d["zcmd"][sel])))
    vz = {}
    for (p, t), (sel, zmed) in zb.items():
        r = compare({"g": {"variable": sample("variable", sel)}}, blocks, draws, rng)
        r5 = compare({"g": {"variable": sample("variable", sel, 5)}}, blocks, draws, rng)
        if "g" in r:
            vz[f"{p}_t{t}"] = dict(phase=p, z_median=zmed, n=r["g"]["n"], u4=r["g"]["variable"], u5=r5["g"]["variable"])
    rep["variable_by_z"] = vz
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.0), constrained_layout=True)
    for ax, q in zip(axes, ("H_xu", "H_pi", "H_u")):
        for p, c in zip(PHASES[1:], ("#7c3aed", "#0891b2", "#b45309")):
            pts = sorted((v["z_median"], v["u4"]["all"][q]) for v in vz.values() if v["phase"] == p)
            if pts:
                z = [a for a, _ in pts]
                v = np.array([b for _, b in pts])
                ax.errorbar(z, v[:, 1], yerr=[v[:, 1] - v[:, 0], v[:, 2] - v[:, 1]], fmt="-o", color=c, label=p, capsize=3)
        ax.set_xlabel("wrist z command, tercile median (-1 stiff ... +1 soft)")
        ax.set_title("variable: " + QLABEL[q].split("  ")[0] + " (all 15 states, 4 motion actions)", fontsize=9)
        ax.grid(alpha=0.3)
    axes[0].legend(fontsize=8)
    axes[0].set_ylabel("nats")
    fig.suptitle("Variable wrist: entropy within each contact phase, split by its z command (each tercile at its own n)",
                 fontsize=10)
    fig.savefig(os.path.join(out, "variable_entropy_vs_z.png"), dpi=150)
    plt.close(fig)

    # 4. successful vs failed episodes, pooled; whole trajectory and contact phases only
    groups = {"success": {m: sample(m, succ[m]) for m in MODES}, "failure": {m: sample(m, ~succ[m]) for m in MODES}}
    rep["by_outcome"] = compare(groups, blocks, draws, rng)
    contact = {m: data[m]["phase"] >= 1 for m in MODES}
    groups = {"success": {m: sample(m, succ[m] & contact[m]) for m in MODES},
              "failure": {m: sample(m, ~succ[m] & contact[m]) for m in MODES}}
    rep["by_outcome_contact"] = compare(groups, blocks, draws, rng)
    plot_outcome(rep["by_outcome_contact"], "all", "Key-in-lock: entropy of seated vs not-seated episodes, contact "
                 "phases only (all 15 states, 4 motion actions)", os.path.join(out, "entropy_vs_outcome.png"))
    rep["episodes"] = {m: dict(n=int(len(np.unique(data[m]["ep"]))), seated=int(np.sum(data[m]["ep_success"])))
                       for m in MODES}
    return rep


# ---------------------------------------------------------------------------------------------------------------
def door(door_dir, out, draws, rng):
    from door_analyze import PHASES as DPHASES
    from door_analyze import door_phase, state17

    blocks = {"all": list(range(17)), "pose": list(range(6)), "wrench": list(range(9, 15)), "mech": [15, 16]}
    blabel = {"all": "all 17 states", "pose": "tip pose block (6)", "wrench": "wrench block (6)",
              "mech": "door + cylinder angle (2)"}
    data, results = {}, {}
    for m in MODES:
        Xs, Us, Ps, Es, Ss = [], [], [], [], []
        results[m] = []
        for f in sorted(glob.glob(os.path.join(door_dir, f"{m}_ep*.npz"))):
            d = dict(np.load(f))
            X, U = state17(d["X"]), d["U"]
            # action at step n: the change of the commanded pose from n-1 (2 mm, 1 deg units), and the wrist z command
            dp = np.diff(U[:, :3], axis=0) / 0.002
            dr = np.array([(Rotation.from_rotvec(b) * Rotation.from_rotvec(a).inv()).as_rotvec()
                           for a, b in zip(U[:-1, 3:6], U[1:, 3:6])]) / np.radians(1.0)
            u = np.c_[dp, dr, U[1:, 8]]
            ph = np.array([door_phase(p, -x) for p, x in zip(d["phase"][1:], d["X"][1:len(U), 2])])
            ok = ph >= 0
            Xs.append(X[1:len(U)][ok])
            Us.append(u[ok])
            Ps.append(ph[ok])
            Es.append(np.full(ok.sum(), int(d["k"])))
            Ss.append(np.full(ok.sum(), str(d["result"]) == "opened"))
            results[m].append(str(d["result"]))
        data[m] = dict(x=np.concatenate(Xs), u=np.concatenate(Us), phase=np.concatenate(Ps), ep=np.concatenate(Es),
                       success=np.concatenate(Ss))

    def sample(m, sel, n_inputs=6):
        d = data[m]
        return d["x"][sel], d["u"][sel][:, :n_inputs], d["ep"][sel], None

    rep = {"episodes": results}
    phases = [p for p in DPHASES if p != "push"]
    groups = {"whole": {m: sample(m, np.ones(len(d["ep"]), bool)) for m, d in data.items()}}
    for p in phases:
        groups[p] = {m: sample(m, d["phase"] == DPHASES.index(p)) for m, d in data.items()}
    rep["by_phase"] = compare(groups, blocks, draws, rng)
    plot_by_group(rep["by_phase"], ["whole", *phases], {"whole": "whole\nepisode"}, blocks, blabel,
                  "KeyDoor (scheduled expert, deterministic): state-action entropy by phase",
                  os.path.join(out, "door_entropy_by_phase.png"))
    # variable schedule with its wrist z command as a 7th action
    g7 = {"whole": {"variable": sample("variable", np.ones(len(data["variable"]["ep"]), bool), 7)}}
    for p in phases:
        g7[p] = {"variable": sample("variable", data["variable"]["phase"] == DPHASES.index(p), 7)}
    rep["variable_with_z_action"] = compare(g7, blocks, draws, rng)
    groups = {"success": {m: sample(m, d["success"]) for m, d in data.items()},
              "failure": {m: sample(m, ~d["success"]) for m, d in data.items()}}
    rep["by_outcome"] = compare(groups, blocks, draws, rng)
    plot_outcome(rep["by_outcome"], "all", "KeyDoor: entropy of opened vs failed episodes, whole episode (all 17 states, "
                 "6 arm actions)", os.path.join(out, "door_entropy_vs_outcome.png"))
    return rep


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser()
    ap.add_argument("--rollouts", default=os.path.join(here, "data", "rollouts"))
    ap.add_argument("--door", default=os.path.join(here, "data", "door"))
    ap.add_argument("--out", default=os.path.join(here, "results"))
    ap.add_argument("--draws", type=int, default=40)
    ap.add_argument("--only", choices=("key_in_lock", "door"), default=None)
    a = ap.parse_args()
    rng = np.random.default_rng(0)
    for name, fn, src in (("key_in_lock", key_in_lock, a.rollouts), ("door", door, a.door)):
        if a.only and a.only != name:
            continue
        o = os.path.join(a.out, name)
        os.makedirs(o, exist_ok=True)
        rep = fn(src, o, a.draws, rng)
        rep["settings"] = dict(k_nn=K_NN, dither=DITHER, keep_episodes=KEEP_EPISODES, draws=a.draws, max_n=MAX_N)
        with open(os.path.join(o, "entropy_report.json"), "w") as f:
            json.dump(rep, f, indent=1)
        print(name, "done", flush=True)


if __name__ == "__main__":
    main()
