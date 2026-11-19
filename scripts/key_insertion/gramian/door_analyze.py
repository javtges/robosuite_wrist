"""
Door Gramians from door_gramian.py's simulator impulse responses, plus a data-only attempt from the same logs.

    python door_analyze.py --door <dir> --out <dir>
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
from scipy.spatial.transform import Rotation  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gramian_lib import COLORS, ridge_fit  # noqa: E402

LABEL = {"stiff": "rigid (fixed_stiff)", "soft": "most flexible (fixed_soft)", "variable": "variable (soft_insert_home_turn)"}
MODES = ("stiff", "soft", "variable")
PHASES = ("free", "search", "slide", "seat", "turn", "push")
ROWS = [0, 1, 2, 3, 4, 5, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22]  # of the 23 raw response rows
SCALE = np.r_[[0.01] * 3, [0.1] * 3, [0.05] * 3, [2.0] * 3, [20.0] * 3, [0.1, 0.1]]  # as the SERL obs, + door, cylinder
POSE, WRENCH, MECH = list(range(6)), list(range(9, 15)), [15, 16]
BLOCKS = {"pose": POSE, "wrench": WRENCH, "mech": MECH}
BLABEL = {"all": "log det W (17 states)", "pose": "tip pose block (6)", "wrench": "wrench block (6)",
          "mech": "door + cylinder angle block (2)"}
N_ARM = 6
STATE_NAMES = ["x", "y", "z", "rx", "ry", "rz", "vx", "vy", "vz", "mx", "my", "mz", "fx", "fy", "fz", "door", "cylinder"]
INPUT_NAMES = ["arm_x", "arm_y", "arm_z", "arm_rx", "arm_ry", "arm_rz", "wrist_z"]


def door_phase(oracle_phase, depth):  # = arm_impedance.door_study.door_phase, from the logged values
    if oracle_phase < 2:
        return 0
    if oracle_phase == 4:
        return 4
    if oracle_phase == 5:
        return 5
    if oracle_phase == 6:
        return -1  # done
    if oracle_phase == 3 or depth >= 0.024:
        return 3
    return 1 if depth < 0.003 else 2


def logdet(W, idx=None, floor=1e-12):
    if idx is not None:
        W = W[np.ix_(idx, idx)]
    return float(np.sum(np.log(np.maximum(np.linalg.eigvalsh((W + W.T) / 2), floor))))


def blocks(W):
    return dict(all=logdet(W), **{k: logdet(W, v) for k, v in BLOCKS.items()})


def state17(X):
    """logged raw states (T, 23) -> normalized 17-d with a local rotation vector (vs the episode's first frame)"""
    R0 = X[0, 3:12].reshape(3, 3)
    rot = np.array([Rotation.from_matrix(x[3:12].reshape(3, 3) @ R0.T).as_rotvec() for x in X])
    return np.c_[X[:, :3], rot, X[:, 12:23]] / SCALE


def inputs(U):
    """absolute actions -> input units: arm pos / 2 mm, arm rotvec / 1 deg, wrist z"""
    return np.c_[U[:, :3] / 0.002, U[:, 3:6] / np.radians(1.0), U[:, 8]]


def spearman(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    ok = ~(np.isnan(a) | np.isnan(b))
    if ok.sum() < 5:
        return [np.nan, int(ok.sum())]
    ra, rb = np.argsort(np.argsort(a[ok])), np.argsort(np.argsort(b[ok]))
    return [float(np.corrcoef(ra, rb)[0, 1]), int(ok.sum())]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--door", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    rep = {"episodes": {}, "by_phase": {}, "correlations": {}, "data_only": {}}
    win = {m: [] for m in MODES}  # rows: dict per window
    eps = {m: [] for m in MODES}
    mats = {m: dict(W=[], W7=[], ep=[], n=[], phase=[], z=[]) for m in MODES}  # saved to gramians.npz
    for m in MODES:
        for f in sorted(glob.glob(os.path.join(a.door, f"{m}_ep*.npz"))):
            d = dict(np.load(f))
            X, U, ph_o = d["X"], d["U"], d["phase"]
            depth = -X[:, 2]
            phase = np.array([door_phase(p, dd) for p, dd in zip(ph_o, depth[: len(ph_o)])])
            pulses, R = d["pulse_n"], d["R"]
            stride = int(np.diff(pulses).min()) if len(pulses) > 1 else 1
            N = R.shape[1]
            Rn = R[:, :, ROWS, :] / SCALE[None, None, :, None]  # (P, N, 17, 7), per input unit
            pos = {int(n): i for i, n in enumerate(pulses)}
            peak = float(np.linalg.norm(X[:, 18:21], axis=1).max())  # leg-wrench force
            peak_m = float(np.linalg.norm(X[:, 15:18], axis=1).max())
            ep = dict(k=int(d["k"]), result=str(d["result"]), friction=float(d["friction"]), dx=float(d["dx"]),
                      dyaw=float(d["dyaw"]), steps=len(U), peak_force=peak, peak_moment=peak_m, mismatch=float(d["mismatch"]))
            eps[m].append(ep)
            # windowed controllability Gramian of x_{n+N} w.r.t. u_n..u_{n+N-1}: stride-weighted sum over pulses
            for n in pulses:
                js = [j for j in range(n, n + N) if j in pos]
                if len(js) < N // stride:
                    continue
                W6 = np.zeros((17, 17))
                W7 = np.zeros((17, 17))
                for j in js:
                    r = Rn[pos[j], n + N - 1 - j]  # (17, 7)
                    W6 += stride * r[:, :N_ARM] @ r[:, :N_ARM].T
                    W7 += stride * r @ r.T
                b6, b7 = blocks(W6), blocks(W7)
                for key, v in dict(W=W6, W7=W7, ep=ep["k"], n=int(n), phase=int(phase[n]), z=float(U[n, 8])).items():
                    mats[m][key].append(v)
                win[m].append(dict(ep=ep["k"], n=int(n), phase=int(phase[n]), z=float(U[n, 8]), success=ep["result"] == "opened",
                                   **{f"ld_{k}": v for k, v in b6.items()}, **{f"ld7_{k}": v for k, v in b7.items()}))
        rep["episodes"][m] = eps[m]
    np.savez_compressed(os.path.join(a.out, "gramians.npz"), state_names=np.array(STATE_NAMES), phases=np.array(PHASES),
                        input_names=np.array(INPUT_NAMES),
                        **{f"{m}/{key}": np.array(v) for m in MODES for key, v in mats[m].items()})

    # ---- by phase: median and IQR over windows; whole trajectory = all windows
    for m in MODES:
        rep["by_phase"][m] = {}
        for pi, pname in [(-9, "whole"), *enumerate(PHASES)]:
            rows = [w for w in win[m] if (pi == -9 or w["phase"] == pi)]
            if len(rows) < 5:
                continue
            rep["by_phase"][m][pname] = dict(n_windows=len(rows), **{
                f"{pre}{k}": [float(np.percentile([w[f'{pre}{k}'] for w in rows], q)) for q in (25, 50, 75)]
                for pre in ("ld_", "ld7_") for k in ("all", "pose", "wrench", "mech")})

    # ---- correlations: episode-level (per phase median) vs outcome; variable: z within phase
    for m in MODES:
        c = {}
        for pname in ("search", "slide", "turn", "push"):
            pi = PHASES.index(pname)
            for k in ("pose", "wrench", "mech"):
                v, out, pk, dy, fr = [], [], [], [], []
                for ep in eps[m]:
                    rows = [w[f"ld_{k}"] for w in win[m] if w["ep"] == ep["k"] and w["phase"] == pi]
                    v.append(np.median(rows) if rows else np.nan)
                    out.append(ep["result"] == "opened")
                    pk.append(ep["peak_force"])
                    dy.append(abs(ep["dyaw"]))
                    fr.append(ep["friction"])
                c[f"{pname}_{k}"] = dict(vs_success=spearman(v, out), vs_peak=spearman(v, pk), vs_abs_dyaw=spearman(v, dy),
                                         vs_friction=spearman(v, fr))
        rep["correlations"][m] = c
    zc = {}
    for pi, pname in enumerate(PHASES):
        rows = [w for w in win["variable"] if w["phase"] == pi]
        zs = [w["z"] for w in rows]
        if len(rows) >= 5 and np.ptp(zs) > 1e-6:
            zc[pname] = {k: spearman(zs, [w[f"ld_{k}"] for w in rows]) for k in ("pose", "wrench", "mech")}
    rows = [w for w in win["variable"] if w["phase"] >= 0]
    zc["all_phases"] = {k: spearman([w["z"] for w in rows], [w[f"ld_{k}"] for w in rows]) for k in ("pose", "wrench", "mech")}
    zc["gain_from_z_input_median"] = {k: float(np.median([w[f"ld7_{k}"] - w[f"ld_{k}"] for w in win[mm]]))
                                      for k in ("all", "pose", "wrench", "mech") for mm in ["variable"]}
    rep["correlations"]["variable_z_within_phase"] = zc

    # ---- data-only attempt (Q1 for the door): identifiability, and a phase-pooled ridge Gramian vs the simulator
    for m in MODES:
        Xs, Us, Xn, Ph, Ep, Ns = [], [], [], [], [], []
        for f in sorted(glob.glob(os.path.join(a.door, f"{m}_ep*.npz"))):
            d = dict(np.load(f))
            x = state17(d["X"])
            u = inputs(d["U"])
            ph = np.array([door_phase(p, -dd) for p, dd in zip(d["phase"], d["X"][: len(d["phase"]), 2])])
            Xs.append(x[:-1]); Xn.append(x[1:]); Us.append(u); Ph.append(ph); Ep.append(np.full(len(u), int(d["k"]))); Ns.append(np.arange(len(u)))
        X, Xn, Uu, Ph, Ep, Ns = map(np.concatenate, (Xs, Xn, Us, Ph, Ep, Ns))
        X1 = np.c_[X, Ph[:, None] == np.arange(6), np.ones(len(X))]
        res = Uu - X1 @ np.linalg.lstsq(X1, Uu, rcond=None)[0]
        r2_u = np.where(Uu.var(0) > 0, 1 - res.var(0) / np.maximum(Uu.var(0), 1e-300), np.nan)  # nan: input held constant
        dd = {"u_explained_by_state_R2": list(np.round(r2_u, 4))}
        # compare per phase with the simulator's windows
        cmp = {}
        for pi, pname in enumerate(PHASES):
            s = Ph == pi
            if s.sum() < 60:
                continue
            A, B, _ = ridge_fit(X[s], Uu[s][:, :N_ARM], Xn[s], 1e-3)
            W = np.zeros((17, 17))
            Ak = np.eye(17)
            for _ in range(20):
                W += (Ak @ B) @ (Ak @ B).T
                Ak = A @ Ak
            sim = rep["by_phase"][m].get(pname)
            cmp[pname] = dict(data_pose=logdet(W, POSE), data_wrench=logdet(W, WRENCH), data_mech=logdet(W, MECH),
                              sim_pose=sim["ld_pose"][1] if sim else None, sim_wrench=sim["ld_wrench"][1] if sim else None,
                              sim_mech=sim["ld_mech"][1] if sim else None, rho_A=float(np.abs(np.linalg.eigvals(A)).max()))
        dd["phase_ridge_vs_sim"] = cmp
        rep["data_only"][m] = dd

    with open(os.path.join(a.out, "door_report.json"), "w") as f:
        json.dump(rep, f, indent=1, default=float)

    # ---- figures
    fig, axes = plt.subplots(1, 4, figsize=(19, 4.4), constrained_layout=True)
    groups = ["whole", *PHASES]
    for ax, k in zip(axes, ("all", "pose", "wrench", "mech")):
        for j, m in enumerate(MODES):
            xs, med, lo, hi = [], [], [], []
            for gi, g in enumerate(groups):
                s = rep["by_phase"][m].get(g)
                if s:
                    q = s[f"ld_{k}"]
                    xs.append(gi + (j - 1) * 0.22)
                    med.append(q[1])
                    lo.append(q[1] - q[0])
                    hi.append(q[2] - q[1])
            ax.errorbar(xs, med, yerr=[lo, hi], fmt="o", color=COLORS[m], label=LABEL[m], capsize=3, ms=6)
        ax.set_xticks(range(len(groups)), ["whole\nepisode", *PHASES])
        ax.set_title(BLABEL[k])
        ax.grid(alpha=0.3)
    axes[0].legend(fontsize=8)
    fig.suptitle("KeyDoor: simulator (empirical) controllability Gramian, 1 s windows, 6 arm inputs: median and IQR over windows (8 episodes per schedule)")
    fig.savefig(os.path.join(a.out, "door_gramian_by_phase.png"), dpi=150)
    plt.close(fig)

    # time traces for one episode per schedule (same plan index)
    k0 = min(e["k"] for e in eps["stiff"]) if eps["stiff"] else 0
    fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True, constrained_layout=True)
    for ax, k in zip(axes, ("pose", "wrench", "mech")):
        for m in MODES:
            rows = [w for w in win[m] if w["ep"] == k0]
            t = np.array([w["n"] for w in rows]) / 20.0
            ax.plot(t, [w[f"ld_{k}"] for w in rows], color=COLORS[m], label=LABEL[m], lw=1.4)
            # phase changes as ticks on the curve
            ph = np.array([w["phase"] for w in rows])
            for i in np.where(np.diff(ph) != 0)[0]:
                ax.annotate(PHASES[ph[i + 1]] if ph[i + 1] >= 0 else "done", (t[i + 1], rows[i + 1][f"ld_{k}"]), fontsize=7,
                            color=COLORS[m], xytext=(2, 4), textcoords="offset points")
                ax.plot(t[i + 1], rows[i + 1][f"ld_{k}"], "|", color=COLORS[m], ms=10)
        ax.set_ylabel(BLABEL[k])
        ax.grid(alpha=0.3)
    axes[0].legend(fontsize=8)
    axes[-1].set_xlabel("time from the expert's start (s), window start")
    e0 = {m: next((e for e in eps[m] if e["k"] == k0), None) for m in MODES}
    fig.suptitle(f"KeyDoor episode {k0} (lock {e0['stiff']['friction'] if e0['stiff'] else '?'} N m, dx {e0['stiff']['dx'] if e0['stiff'] else 0:+.2f} mm, "
                 f"dyaw {e0['stiff']['dyaw'] if e0['stiff'] else 0:+.1f} deg): windowed log det along the episode")
    fig.savefig(os.path.join(a.out, "door_gramian_vs_time.png"), dpi=150)
    plt.close(fig)

    print(json.dumps(rep, indent=1, default=float)[:20000])


if __name__ == "__main__":
    main()
