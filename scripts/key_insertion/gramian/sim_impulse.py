"""
Simulator ground truth for the data-driven models: at chosen (episode, step k) of the recorded rollouts, replay the
episode exactly (same reset seed, aim error, recorded actions), perturb u_k by +-eps along each input, then apply the
recorded u_{k+1..k+N-1} open-loop. Returns impulse responses R[j, :, i] = (x_{k+1+j}(+) - x_{k+1+j}(-)) / (2 eps).
Run with the SERL venv (or robosuite_env) from robosuite_wrist/scripts.
"""

import argparse
import multiprocessing as mp
import os
import sys

import numpy as np

SCRIPTS = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # robosuite_wrist/scripts
sys.path.insert(0, SCRIPTS)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def true_obs(env):
    saved = env.aim_error.copy()
    env.aim_error = np.zeros(3)
    o = env._obs()
    env.aim_error = saved
    return o


def rollout(env, ep_meta, actions, k=None, du=None, N=0):
    env.aim_error = np.array([ep_meta["dx"] * 1e-3, 0.0, np.radians(ep_meta["dyaw"])])
    env.reset(seed=int(ep_meta["reset_seed"]))
    xs = [true_obs(env)]
    T = len(actions) if k is None else min(k + N, len(actions))
    for t in range(T):
        u = actions[t].astype(float).copy()
        if k is not None and t == k:
            u = u + du
        _, _, done, trunc, _ = env.step(u)
        xs.append(true_obs(env))
        if (done or trunc) and t >= (k if k is not None else 0):
            break
    return np.array(xs)


def job(args):
    mode, points, N, eps, out = args
    from key_insertion.serl_env import KeyInsertionSERL

    env = KeyInsertionSERL(mode=mode, seed=0)
    res = []
    for p in points:
        acts, meta, k = p["actions"], p["meta"], p["k"]
        if p.get("check_replay"):
            xs = rollout(env, meta, acts)
            err = np.abs(xs[1:len(p["x_rec"]) + 1] - p["x_rec"][: len(xs) - 1]).max()
            res.append(dict(kind="replay", ep=p["ep"], err=float(err)))
            continue
        m = acts.shape[1]
        R = np.full((N, 22, m), np.nan)
        for i in range(m):
            e = np.zeros(m)
            e[i] = eps
            xp = rollout(env, meta, acts, k, e, N)
            xm = rollout(env, meta, acts, k, -e, N)
            L = min(len(xp), len(xm)) - (k + 1)
            if L > 0:
                R[:L, :, i] = (xp[k + 1:k + 1 + L] - xm[k + 1:k + 1 + L]) / (2 * eps)
        res.append(dict(kind="impulse", ep=p["ep"], k=k, phase=p["phase"], R=R))
    env.close()
    return mode, res


def main():
    from gramian_lib import load

    ap = argparse.ArgumentParser()
    ap.add_argument("--rollouts", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--per_phase", type=int, default=3)
    ap.add_argument("--N", type=int, default=10)
    ap.add_argument("--eps", type=float, default=0.1)
    a = ap.parse_args()
    rng = np.random.default_rng(0)
    jobs = []
    for mode in ("stiff", "soft", "variable"):
        d = load(a.rollouts, mode)
        meta_of = {}
        for e in np.unique(d["ep"]):
            i = np.where(d["ep"] == e)[0]
            seed = None
            meta_of[e] = i
        # reset seeds are not stored per row; rebuild them from record_rollouts.plan
        from record_rollouts import plan
        pl = plan(30, 2)
        pts = []
        # replay determinism check on 2 episodes
        for e in rng.choice(np.unique(d["ep"]), 2, replace=False):
            i = meta_of[e]
            pts.append(dict(check_replay=True, ep=int(e), actions=d["u"][i], meta=pl[int(e)], k=None, x_rec=d["xn"][i]))
        for ph in range(4):
            cand = np.where(d["phase"] == ph)[0]
            for idx in rng.choice(cand, min(a.per_phase, len(cand)), replace=False) if len(cand) else []:
                e = d["ep"][idx]
                i = meta_of[e]
                pts.append(dict(ep=int(e), actions=d["u"][i], meta=pl[int(e)], k=int(d["t"][idx]), phase=ph))
        # split each mode's points over 3 workers
        for s in range(3):
            jobs.append((mode, pts[s::3], a.N, a.eps, a.out))
    os.makedirs(a.out, exist_ok=True)
    allres = {}
    with mp.get_context("spawn").Pool(9) as pool:
        for mode, res in pool.imap_unordered(job, jobs):
            allres.setdefault(mode, []).extend(res)
            print(mode, len(res), flush=True)
    np.save(os.path.join(a.out, "impulse.npy"), allres, allow_pickle=True)


if __name__ == "__main__":
    main()
