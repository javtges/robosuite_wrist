"""Data-driven (local-linear) controllability Gramians from logged SERL transitions. numpy only."""

import glob
import os

import numpy as np

# observation layout (serl_env._obs, normalized by OBS_SCALE): pos 0:3 (/1 cm), rot err 3:6 (/0.1 rad),
# vel 6:9 (/5 cm/s), moment 9:12 (/2 N m), force 12:15 (/20 N), deflection 15:21, z 21
STATE = list(range(15))  # pos, rot, vel, moment, force
NAMES = ["x", "y", "z", "rx", "ry", "rz", "vx", "vy", "vz", "mx", "my", "mz", "fx", "fy", "fz"]
POSE = list(range(6))
WRENCH = list(range(9, 15))
MODES = ("stiff", "soft", "variable")
COLORS = {"stiff": "#c2410c", "soft": "#2563eb", "variable": "#15803d"}
LABEL = {"stiff": "rigid (z = -1)", "soft": "most flexible (z = +1)", "variable": "variable (policy sets z)"}

# phases of the SERL lock (depth = tip below the lock's top face): chamfer 2 mm, plunger / latch met at 20 mm
PHASES = ("free", "search", "slide", "latch")


CONTACT_N = 1.0  # wrist F/T force magnitude that counts as touching the lock


def phase_of(depth, force_n=None):
    """free: above the lock face, not touching; search: touching the face or in the chamfer; slide; latch"""
    d = np.asarray(depth)
    # the tip is the key's lowest point: it cannot touch the lock from more than 3 mm above the face (this also drops
    # the rigid wrist's 4.6 N leg-wrench transient at steps 0-1 after reset)
    touching = np.zeros_like(d, bool) if force_n is None else (np.asarray(force_n) >= CONTACT_N) & (d >= -0.003)
    return np.select([(d < 0.0) & ~touching, d < 0.002, d < 0.020], [0, 1, 2], 3)


def load(rollout_dir, mode):
    parts = [dict(np.load(f)) for f in sorted(glob.glob(os.path.join(rollout_dir, f"{mode}_*.npz")))]
    d = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
    d["depth"] = d["ex"][:, 0]
    d["zcmd"] = d["ex"][:, 2]
    d["force_n"] = np.linalg.norm(d["x"][:, 12:15], axis=1) * 20.0  # obs force is /20 N
    d["phase"] = phase_of(d["depth"], d["force_n"])
    # phase-pooled fits use only transitions that start and end in the same phase (-1: crosses a boundary)
    d["phase_next"] = phase_of(d["exn"][:, 0], np.linalg.norm(d["xn"][:, 12:15], axis=1) * 20.0)
    d["phase_fit"] = np.where(d["phase"] == d["phase_next"], d["phase"], -1)
    return d


def design(d, mode, idx=None, n_inputs=None):
    """regressors / targets: x+ = A x + B u + c. variable: z is a 16th state (index 15) and u has 5 entries."""
    idx = slice(None) if idx is None else idx
    x, xn, u = d["x"][idx][:, STATE], d["xn"][idx][:, STATE], d["u"][idx]
    if mode == "variable":
        x = np.c_[x, d["x"][idx][:, 21]]
        xn = np.c_[xn, d["xn"][idx][:, 21]]
    if n_inputs is not None:
        u = u[:, :n_inputs]
    return x, u, xn


def ridge_fit(x, u, xn, lam=1e-3, w=None):
    """A, B, c by weighted ridge (regressors standardized, intercept unpenalized)"""
    n, m = x.shape[1], u.shape[1]
    Z = np.c_[x, u]
    w = np.ones(len(Z)) if w is None else w
    mu = (w[:, None] * Z).sum(0) / w.sum()
    sd = np.sqrt((w[:, None] * (Z - mu) ** 2).sum(0) / w.sum()) + 1e-9
    Zs = (Z - mu) / sd
    Y = xn - x  # fit the increment (A = I + ...) for conditioning
    ymu = (w[:, None] * Y).sum(0) / w.sum()
    W = w[:, None]
    G = Zs.T @ (W * Zs) + lam * w.sum() * np.eye(Z.shape[1])
    Th = np.linalg.solve(G, Zs.T @ (W * (Y - ymu))) / sd[:, None]  # (n+m, n)
    A = np.eye(n) + Th[:n].T
    B = Th[n:].T
    c = ymu - mu @ Th
    return A, B, c


def gramian(A, B, N):
    """finite-horizon discrete controllability Gramian sum_{k<N} A^k B B^T A^kT"""
    W = np.zeros((A.shape[0],) * 2)
    Ak = np.eye(A.shape[0])
    for _ in range(N):
        AB = Ak @ B
        W += AB @ AB.T
        Ak = A @ Ak
    return W


def ltv_gramian(As, Bs):
    """Gramian of x_T w.r.t. u_0..u_{T-1} for x_{k+1} = A_k x_k + B_k u_k"""
    n = As[0].shape[0]
    W = np.zeros((n, n))
    for A, B in zip(As, Bs):
        W = A @ W @ A.T + B @ B.T
    return W


def logdet(W, idx=None, floor=1e-12):
    if idx is not None:
        W = W[np.ix_(idx, idx)]
    ev = np.linalg.eigvalsh((W + W.T) / 2)
    return float(np.sum(np.log(np.maximum(ev, floor))))


def summaries(W):
    """scalar summaries of a Gramian over the 15 shared states (z excluded for variable)"""
    W15 = W[:15, :15]
    ev = np.linalg.eigvalsh((W15 + W15.T) / 2)
    return dict(logdet=logdet(W15), logdet_pose=logdet(W15, POSE), logdet_wrench=logdet(W15, WRENCH),
                trace=float(ev.sum()), lam_min=float(ev.min()), lam_max=float(ev.max()))


def knn_models(d, mode, k=200, lam=1e-3, n_inputs=None, feats=None):
    """A_t, B_t for every transition: ridge on its k nearest neighbours (same mode) in a task feature space"""
    x, u, xn = design(d, mode, n_inputs=n_inputs)
    F = np.c_[d["depth"] * 100, d["x"][:, [0, 1, 5]], np.linalg.norm(d["x"][:, 12:15], axis=1), d["x"][:, 21] * 2]
    F = (F - F.mean(0)) / (F.std(0) + 1e-9)
    As, Bs = [], []
    for i in range(len(F)):
        nb = np.argpartition(((F - F[i]) ** 2).sum(1), k)[:k]
        A, B, _ = ridge_fit(x[nb], u[nb], xn[nb], lam)
        As.append(A)
        Bs.append(B)
    return np.array(As), np.array(Bs)


def heldout_r2(d, mode, groups, lam=1e-3, n_inputs=None, folds=5, seed=0):
    """episode-wise CV one-step R^2 of the increment, per state, for a model fit per group label"""
    rng = np.random.default_rng(seed)
    eps = np.unique(d["ep"])
    fold = dict(zip(eps, rng.permutation(len(eps)) % folds))
    f = np.array([fold[e] for e in d["ep"]])
    x, u, xn = design(d, mode, n_inputs=n_inputs)
    pred = np.full_like(xn, np.nan)
    for g in np.unique(groups):
        for k in range(folds):
            tr, te = (groups == g) & (f != k), (groups == g) & (f == k)
            if tr.sum() < 40 or te.sum() == 0:
                continue
            A, B, c = ridge_fit(x[tr], u[tr], xn[tr], lam)
            pred[te] = x[te] @ A.T + u[te] @ B.T + c
    ok = ~np.isnan(pred[:, 0])
    dy, dp = (xn - x)[ok], (pred - x)[ok]
    return 1 - ((dy - dp) ** 2).sum(0) / ((dy - dy.mean(0)) ** 2).sum(0), ok.mean()
