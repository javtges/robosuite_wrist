"""
Roll out the trained SERL policies (runs/<mode>_seed0/final) with STOCHASTIC actions (SAC's own sampled actions, as the
actor does while training), from (a) the nominal training starts and (b) the aim-error grid of aim_error_eval.py.
Saves every transition for Gramian estimation. Imports the project's modules; changes nothing in the repo.

Run with the SERL venv from robosuite_wrist/scripts:
    ~/serl_env/bin/python <this file> --out <dir>
"""

import argparse
import multiprocessing as mp
import os
import sys

import numpy as np

SCRIPTS = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # robosuite_wrist/scripts
sys.path.insert(0, SCRIPTS)

MODES = ("stiff", "variable", "soft")
DX = np.array([-1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5])  # mm, = make_serl_figures.DX
DYAW = np.array([-3.0, -1.5, 0.0, 1.5, 3.0])  # deg, = make_serl_figures.DYAW


def plan(n_nominal, n_aim_starts):
    eps = [dict(kind=0, dx=0.0, dyaw=0.0, reset_seed=30000 + i) for i in range(n_nominal)]
    s = 40000
    for dx in DX:
        for dyaw in DYAW:
            for _ in range(n_aim_starts):
                eps.append(dict(kind=1, dx=float(dx), dyaw=float(dyaw), reset_seed=s))
                s += 1
    return eps


def run(args):
    mode, episodes, shard, out = args
    import jax
    from flax.training import checkpoints
    from serl_launcher.utils.launcher import make_sac_agent

    from key_insertion.serl_env import KeyInsertionSERL

    env = KeyInsertionSERL(mode=mode, seed=0)
    agent = make_sac_agent(seed=0, sample_obs=env.observation_space.sample(), sample_action=env.action_space.sample())
    path = os.path.join(SCRIPTS, "key_insertion", "runs", f"{mode}_seed0", "final")
    state = checkpoints.restore_checkpoint(path, target=agent.state)
    assert state is not agent.state, path
    agent = agent.replace(state=state)

    def true_obs():  # observation in the TRUE lock frame (the policy's obs is in the believed, aim-error frame)
        saved = env.aim_error.copy()
        env.aim_error = np.zeros(3)
        o = env._obs()
        env.aim_error = saved
        return o

    def extras():
        return [env.env.insertion_depth(), env.env.plunger_depression(), env.z, *env.target]

    rows = {k: [] for k in ("ep", "t", "x", "xn", "obs", "u", "u_mean", "ex", "exn", "r", "f", "fphys", "kind", "dx", "dyaw")}
    summ = []
    for e, ep in episodes:
        env.aim_error = np.array([ep["dx"] * 1e-3, 0.0, np.radians(ep["dyaw"])])
        obs, _ = env.reset(seed=ep["reset_seed"])
        key = jax.random.PRNGKey(ep["reset_seed"])
        x, ex = true_obs(), extras()
        done = trunc = False
        while not (done or trunc):
            key, sub = jax.random.split(key)
            jo = jax.device_put(obs)
            u = np.asarray(jax.device_get(agent.sample_actions(observations=jo, seed=sub)))
            um = np.asarray(jax.device_get(agent.sample_actions(observations=jo, argmax=True)))
            t = env.steps
            obs_n, r, done, trunc, info = env.step(u)
            xn, exn = true_obs(), extras()
            for k, v in dict(ep=e, t=t, x=x, xn=xn, obs=obs, u=u, u_mean=um, ex=ex, exn=exn, r=r, f=info["max_force"],
                             fphys=info["max_force_physics"], kind=ep["kind"], dx=ep["dx"], dyaw=ep["dyaw"]).items():
                rows[k].append(v)
            obs, x, ex = obs_n, xn, exn
        summ.append((e, info["succeed"], info["seated"], info["over_force"], env.steps))
    env.close()
    d = {k: np.array(v) for k, v in rows.items()}
    s = np.array(summ, dtype=float)
    np.savez(os.path.join(out, f"{mode}_{shard}.npz"), **d, ep_id=s[:, 0], ep_success=s[:, 1], ep_seated=s[:, 2],
             ep_over=s[:, 3], ep_steps=s[:, 4])
    return mode, shard, int(s[:, 1].sum()), len(s)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    p.add_argument("--nominal", type=int, default=30)
    p.add_argument("--aim_starts", type=int, default=2)
    p.add_argument("--shards", type=int, default=3)
    a = p.parse_args()
    os.makedirs(a.out, exist_ok=True)
    eps = list(enumerate(plan(a.nominal, a.aim_starts)))
    jobs = [(m, eps[i:: a.shards], i, a.out) for m in MODES for i in range(a.shards)]
    with mp.get_context("spawn").Pool(min(len(jobs), 9)) as pool:
        for mode, shard, ok, n in pool.imap_unordered(run, jobs):
            print(f"{mode} shard {shard}: {ok}/{n} succeeded", flush=True)


if __name__ == "__main__":
    main()
