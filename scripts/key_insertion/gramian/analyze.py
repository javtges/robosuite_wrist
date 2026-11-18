"""
Gramian analysis of the stochastic SERL rollouts (record_rollouts.py). numpy + matplotlib only.

    python analyze.py --rollouts <dir> --out <dir> [--impulse <dir>/impulse.npy]
"""

import argparse
import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gramian_lib import (COLORS, LABEL, MODES, NAMES, PHASES, design, gramian, heldout_r2, knn_models, load,  # noqa: E402
                         logdet, ltv_gramian, ridge_fit, summaries)

N = 10  # Gramian horizon, policy steps (1 s)
LAM = 1e-3
B_BOOT = 200
METRICS = ("logdet", "logdet_pose", "logdet_wrench")
MLABEL = {"logdet": "log det W (15 states)", "logdet_pose": "log det W, tip pose block (6)",
          "logdet_wrench": "log det W, wrench block (6)"}


def spearman(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    ok = ~(np.isnan(a) | np.isnan(b))
    if ok.sum() < 5:
        return np.nan, ok.sum()
    ra, rb = np.argsort(np.argsort(a[ok])), np.argsort(np.argsort(b[ok]))
    return float(np.corrcoef(ra, rb)[0, 1]), int(ok.sum())


def phase_models(d, mode, sel_eps=None, n_inputs=None):
    """{group: (A, B, n)} for each phase and 'all'"""
    out = {}
    eps_mask = np.ones(len(d["ep"]), bool) if sel_eps is None else None
    for g, name in [(-1, "all"), *enumerate(PHASES)]:
        if sel_eps is None:
            idx = np.where(eps_mask if g < 0 else d["phase_fit"] == g)[0]
        else:  # bootstrap: rows of the resampled episodes, with multiplicity
            idx = np.concatenate([np.where((d["ep"] == e) & ((d["phase_fit"] == g) if g >= 0 else True))[0] for e in sel_eps])
        if len(idx) < 60:
            continue
        x, u, xn = design(d, mode, idx, n_inputs)
        A, B, _ = ridge_fit(x, u, xn, LAM)
        out[name] = (A, B, len(idx))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rollouts", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--impulse", default=None)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    rng = np.random.default_rng(0)
    data = {m: load(a.rollouts, m) for m in MODES}
    report = {}

    # ---------- 0. the data -------------------------------------------------------------------------------------
    for m, d in data.items():
        n_ep = len(d["ep_id"])
        report.setdefault(m, {})["episodes"] = dict(
            n=n_ep, transitions=len(d["ep"]), success=int(d["ep_success"].sum()), over=int(d["ep_over"].sum()),
            per_phase={p: int((d["phase"] == i).sum()) for i, p in enumerate(PHASES)}, per_phase_fit={p: int((d["phase_fit"] == i).sum()) for i, p in enumerate(PHASES)},
            action_noise_std=list(np.round((d["u"] - d["u_mean"]).std(0), 3)))

    # ---------- 1. identifiability: held-out one-step R^2 ------------------------------------------------------------
    for m, d in data.items():
        r2_all, _ = heldout_r2(d, m, np.zeros(len(d["ep"]), int), LAM)
        r2_ph, cov = heldout_r2(d, m, d["phase_fit"], LAM)
        report[m]["heldout_r2_global"] = dict(zip(NAMES, np.round(r2_all[:15], 3)))
        report[m]["heldout_r2_phase"] = dict(zip(NAMES, np.round(r2_ph[:15], 3)))
        # regressor conditioning (input excitation independent of state): R^2 of u on x
        x, u, _ = design(d, m)
        X1 = np.c_[x, np.ones(len(x))]
        res = u - X1 @ np.linalg.lstsq(X1, u, rcond=None)[0]
        report[m]["u_explained_by_x_R2"] = list(np.round(1 - res.var(0) / u.var(0), 3))

    # ---------- 2. phase / global LTI Gramians with episode bootstrap -----------------------------------------------
    groups = ["all", *PHASES]
    table = {m: {} for m in MODES}
    boot = {m: {g: {k: [] for k in METRICS} for g in groups} for m in MODES}
    for m, d in data.items():
        for g, (A, B, n) in phase_models(d, m).items():
            W = gramian(A, B[:, :4], N)
            s = summaries(W)
            s["n"] = n
            s["rho_A"] = float(np.max(np.abs(np.linalg.eigvals(A))))
            if m == "variable":
                s5 = summaries(gramian(A, B, N))
                s.update({f"{k}_5in": v for k, v in s5.items()})
            table[m][g] = s
        eps = np.unique(d["ep"])
        for _ in range(B_BOOT):
            for g, (A, B, n) in phase_models(d, m, rng.choice(eps, len(eps))).items():
                s = summaries(gramian(A, B[:, :4], N))
                for k in METRICS:
                    boot[m][g][k].append(s[k])
    for m in MODES:
        for g in table[m]:
            for k in METRICS:
                v = np.array(boot[m][g][k])
                table[m][g][f"{k}_ci"] = [float(np.percentile(v, 5)), float(np.percentile(v, 95))] if len(v) > 20 else None
    report["lti_gramians"] = table

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4), constrained_layout=True)
    for ax, k in zip(axes, METRICS):
        for j, m in enumerate(MODES):
            xs, ys, lo, hi = [], [], [], []
            for gi, g in enumerate(groups):
                if g in table[m]:
                    xs.append(gi + (j - 1) * 0.22)
                    ys.append(table[m][g][k])
                    ci = table[m][g][f"{k}_ci"] or [np.nan, np.nan]
                    lo.append(ys[-1] - ci[0])
                    hi.append(ci[1] - ys[-1])
            ax.errorbar(xs, ys, yerr=[np.maximum(lo, 0), np.maximum(hi, 0)], fmt="o", color=COLORS[m], label=LABEL[m], capsize=3, ms=6)
        ax.set_xticks(range(len(groups)), ["whole\ntrajectory", *PHASES])
        ax.set_title(MLABEL[k] + f", N = {N} steps")
        ax.grid(alpha=0.3)
    axes[0].legend(fontsize=8)
    fig.suptitle("Local-LTI controllability Gramian per phase, from stochastic SERL policy rollouts (4 motion inputs; 90% episode-bootstrap CI)")
    fig.savefig(os.path.join(a.out, "gramian_by_phase.png"), dpi=150)
    plt.close(fig)

    # ---------- 3. LTV (kNN local models) along every episode ---------------------------------------------------------
    per_ep = {}
    for m, d in data.items():
        As, Bs = knn_models(d, m, k=200, lam=LAM)
        Bs4 = Bs[:, :, :4]
        win = np.full(len(d["ep"]), np.nan)
        win5 = np.full(len(d["ep"]), np.nan)
        win_pose = np.full(len(d["ep"]), np.nan)
        win_wr = np.full(len(d["ep"]), np.nan)
        rows = []
        for e in np.unique(d["ep"]):
            i = np.where(d["ep"] == e)[0]
            i = i[np.argsort(d["t"][i])]
            for s0 in range(len(i)):
                w = i[s0:s0 + N]
                if len(w) < N:
                    break
                W = ltv_gramian(As[w], Bs4[w])
                s = summaries(W)
                win[i[s0]], win_pose[i[s0]], win_wr[i[s0]] = s["logdet"], s["logdet_pose"], s["logdet_wrench"]
                if m == "variable":
                    win5[i[s0]] = summaries(ltv_gramian(As[w], Bs[w]))["logdet"]
            Wfull = ltv_gramian(As[i], Bs4[i])
            k_ep = int(np.where(d["ep_id"] == e)[0][0])
            ph = d["phase"][i]
            rows.append(dict(ep=int(e), kind=int(d["kind"][i[0]]), dx=float(d["dx"][i[0]]), dyaw=float(d["dyaw"][i[0]]),
                             success=bool(d["ep_success"][k_ep]), over=bool(d["ep_over"][k_ep]), steps=len(i),
                             peak=float(d["fphys"][i].max()), full_logdet=logdet(Wfull[:15, :15]),
                             mean_win=float(np.nanmean(win[i])),
                             **{f"mean_win_{p}": float(np.nanmean(win[i][ph == pi])) if np.any(~np.isnan(win[i][ph == pi])) else np.nan
                                for pi, p in enumerate(PHASES)},
                             mean_z=float(d["zcmd"][i].mean())))  # fmt: skip
        d["win"], d["win5"], d["win_pose"], d["win_wr"] = win, win5, win_pose, win_wr
        per_ep[m] = rows

    # correlations
    corr = {}
    for m in MODES:
        r = per_ep[m]
        c = {}
        for key in ("mean_win", "mean_win_search", "mean_win_slide", "full_logdet"):
            v = [x[key] for x in r]
            c[key] = dict(vs_success=spearman(v, [x["success"] for x in r]), vs_peak=spearman(v, [x["peak"] for x in r]),
                          vs_abs_dyaw=spearman(v, [abs(x["dyaw"]) for x in r]))
        corr[m] = c
    d = data["variable"]
    ok = ~np.isnan(d["win"])
    corr["variable_within"] = dict(
        win_logdet_vs_z=spearman(d["win"][ok], d["zcmd"][ok]),
        win_pose_vs_z=spearman(d["win_pose"][ok], d["zcmd"][ok]),
        win_wrench_vs_z=spearman(d["win_wr"][ok], d["zcmd"][ok]),
        **{f"win_logdet_vs_z_{p}": spearman(d["win"][ok & (d["phase"] == i)], d["zcmd"][ok & (d["phase"] == i)])
           for i, p in enumerate(PHASES)},
        gain_from_5th_input=float(np.nanmedian(d["win5"] - d["win"])))
    # across modes: per-episode mean windowed logdet, by phase
    across = {}
    for p in ["mean_win", *[f"mean_win_{q}" for q in PHASES], "full_logdet"]:
        across[p] = {m: [float(np.nanpercentile([x[p] for x in per_ep[m]], q)) if np.any(~np.isnan([x[p] for x in per_ep[m]])) else None
                         for q in (25, 50, 75)] for m in MODES}
    report["ltv_correlations"] = corr
    report["ltv_across_modes_quartiles"] = across
    with open(os.path.join(a.out, "per_episode.json"), "w") as f:
        json.dump(per_ep, f)

    # figure: windowed logdet vs depth
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4), constrained_layout=True)
    bins = np.r_[-0.012, -0.004, 0.0, 0.001, 0.002, 0.005, 0.010, 0.015, 0.020, 0.0225, 0.025, 0.031]
    for ax, key, title in zip(axes, ("win", "win_pose", "win_wr"), METRICS):
        for m in MODES:
            dd = data[m]
            v, dep = dd[key], dd["depth"]
            mid, med, q1, q3 = [], [], [], []
            for lo, hi in zip(bins[:-1], bins[1:]):
                s = (dep >= lo) & (dep < hi) & ~np.isnan(v)
                if s.sum() >= 15:
                    mid.append((lo + hi) / 2 * 1000)
                    med.append(np.median(v[s]))
                    q1.append(np.percentile(v[s], 25))
                    q3.append(np.percentile(v[s], 75))
            ax.plot(mid, med, "-o", color=COLORS[m], label=LABEL[m], ms=4)
            ax.fill_between(mid, q1, q3, color=COLORS[m], alpha=0.15)
        for xv, lab in ((0, "lock face"), (2, "chamfer end"), (20, "latch")):
            ax.axvline(xv, color="0.6", lw=0.8, ls="--")
            ax.text(xv, ax.get_ylim()[1], lab, fontsize=7, rotation=90, va="top", ha="right", color="0.4")
        ax.set_xlabel("key-tip depth at window start (mm)")
        ax.set_title(MLABEL[title] + f", {N}-step window")
        ax.grid(alpha=0.3)
    axes[0].legend(fontsize=8)
    fig.suptitle("Time-varying (kNN local-linear) controllability Gramian along the rollouts: median and IQR by depth")
    fig.savefig(os.path.join(a.out, "gramian_vs_depth.png"), dpi=150)
    plt.close(fig)

    # figure: variable mode, logdet vs z
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2), constrained_layout=True)
    for ax, key, title in zip(axes, ("win", "win_pose", "win_wr"), METRICS):
        for i, p in enumerate(PHASES):
            s = ok & (d["phase"] == i)
            ax.scatter(d["zcmd"][s], d[key][s], s=6, alpha=0.5, label=p)
        ax.set_xlabel("wrist z command at window start (-1 stiff ... +1 soft)")
        ax.set_title("variable: " + MLABEL[title])
        ax.grid(alpha=0.3)
    axes[0].legend(fontsize=8, markerscale=3)
    fig.savefig(os.path.join(a.out, "variable_logdet_vs_z.png"), dpi=150)
    plt.close(fig)

    # figure: per-episode outcome vs search-phase logdet
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.0), constrained_layout=True, sharey=True)
    for ax, m in zip(axes, MODES):
        r = per_ep[m]
        v = np.array([x["mean_win_search"] for x in r])
        pk = np.array([x["peak"] for x in r])
        sc = np.array([x["success"] for x in r])
        ax.scatter(v[sc], pk[sc], c=COLORS[m], s=18, label="seated")
        ax.scatter(v[~sc], pk[~sc], facecolors="none", edgecolors=COLORS[m], s=18, label="not seated / over")
        ax.axhline(80, color="0.5", ls="--", lw=0.8)
        rho = corr[m]["mean_win_search"]["vs_peak"][0]
        ax.set_title(f"{LABEL[m]}: Spearman(logdet, peak) = {rho:+.2f}")
        ax.set_xlabel("episode mean of windowed log det W in the search phase")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7)
    axes[0].set_ylabel("peak flange force, physics step (N)")
    fig.savefig(os.path.join(a.out, "episode_logdet_vs_outcome.png"), dpi=150)
    plt.close(fig)

    # ---------- 4. simulator impulse-response check -----------------------------------------------------------------
    if a.impulse and os.path.exists(a.impulse):
        imp = np.load(a.impulse, allow_pickle=True).item()
        val = {}
        fig, axes = plt.subplots(1, 3, figsize=(14, 4.2), constrained_layout=True)
        for ax, m in zip(axes, MODES):
            dd = data[m]
            As, Bs = knn_models(dd, m, k=200, lam=LAM)  # recomputed (cheap enough)
            rel, ld_sim, ld_mod, reps = [], [], [], []
            for r in imp.get(m, []):
                if r["kind"] == "replay":
                    reps.append(r["err"])
                    continue
                i = np.where((dd["ep"] == r["ep"]) & (dd["t"] == r["k"]))[0][0]
                j = np.where(dd["ep"] == r["ep"])[0]
                j = j[np.argsort(dd["t"][j])]
                pos = int(np.where(j == i)[0][0])
                Rs = r["R"][:, :15, :4]
                L = int(np.sum(~np.isnan(Rs[:, 0, 0])))
                L = min(L, len(j) - pos)
                if L < 2:
                    continue
                Phi = np.eye(As.shape[1])
                Rm = []
                for q in range(L):
                    if q > 0:
                        Phi = As[j[pos + q]] @ Phi
                    Rm.append((Phi @ Bs[i])[:15, :4])
                Rm = np.array(Rm)
                Rs = Rs[:L]
                rel.append(np.linalg.norm(Rm - Rs) / np.linalg.norm(Rs))
                Ws = np.einsum("qai,qbi->ab", Rs, Rs)
                Wm = np.einsum("qai,qbi->ab", Rm, Rm)
                ld_sim.append(logdet(Ws, list(range(6))))
                ld_mod.append(logdet(Wm, list(range(6))))
                ax.scatter(ld_sim[-1], ld_mod[-1], color=COLORS[m], s=20)
            lim = [min(ld_sim + ld_mod) - 1, max(ld_sim + ld_mod) + 1] if ld_sim else [0, 1]
            ax.plot(lim, lim, "k--", lw=0.8)
            ax.set_xlabel("sim finite difference: log det (pose block)")
            ax.set_ylabel("data-driven model")
            ax.set_title(f"{LABEL[m]}: median rel. impulse error {np.median(rel) if rel else np.nan:.2f}")
            ax.grid(alpha=0.3)
            val[m] = dict(replay_max_err=reps, rel_impulse_err_median=float(np.median(rel)) if rel else None,
                          rel_impulse_err=list(np.round(rel, 3)), pose_logdet_sim=ld_sim, pose_logdet_model=ld_mod,
                          spearman=spearman(ld_sim, ld_mod))
        fig.suptitle("Check: impulse responses (u_k -> x_{k+1..k+N}) from the simulator vs the data-driven LTV model")
        fig.savefig(os.path.join(a.out, "sim_check.png"), dpi=150)
        plt.close(fig)
        report["sim_check"] = val

    with open(os.path.join(a.out, "report.json"), "w") as f:
        json.dump(report, f, indent=1, default=float)
    print(json.dumps(report, indent=1, default=float))


if __name__ == "__main__":
    main()
