"""
Exact action entropy of the SERL policies at every logged observation of the rollouts (record_rollouts.py): the SAC
actor's distribution is a diagonal Gaussian squashed by tanh, pi(u | o) = tanh_* N(mu(o), sigma(o)), so its entropy
per action dimension is E[-log N(a; mu, sigma) + log(1 - tanh(a)^2)], a ~ N(mu, sigma), estimated here with M samples
per observation. Dimensions are independent, so any subset's entropy (e.g. the 4 motion actions without the variable
wrist's z) is the sum over its dimensions. Units: the policy's own action units (2 mm, 1 deg; z command 1), the same
units entropy.py uses for H(u).

Run with the SERL venv from robosuite_wrist/scripts/key_insertion/gramian:
    ~/serl_env/bin/python policy_entropy.py [--rollouts data/rollouts] [--M 256]
Writes <rollouts>/policy_entropy_<mode>.npz: ep, t, H_dim (rows, action_dim), mu, sigma (pre-tanh), logp_logged
(log pi of the logged sampled action, per dim; its mean is a second, independent estimate).
"""

import argparse
import glob
import os
import sys

import numpy as np

SCRIPTS = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # robosuite_wrist/scripts
sys.path.insert(0, SCRIPTS)


def log_prob_dims(u_pre, mu, sigma):
    """per-dimension log density of u = tanh(a) at pre-tanh a"""
    lg = -0.5 * ((u_pre - mu) / sigma) ** 2 - np.log(sigma) - 0.5 * np.log(2 * np.pi)
    return lg - (2 * (np.log(2) - u_pre - np.logaddexp(0, -2 * u_pre)))  # log(1 - tanh^2), stable


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser()
    ap.add_argument("--rollouts", default=os.path.join(here, "data", "rollouts"))
    ap.add_argument("--M", type=int, default=256)
    a = ap.parse_args()

    import jax
    from flax.training import checkpoints
    from serl_launcher.utils.launcher import make_sac_agent

    from key_insertion.serl_env import KeyInsertionSERL

    rng = np.random.default_rng(0)
    for mode in ("stiff", "soft", "variable"):
        parts = [dict(np.load(f)) for f in sorted(glob.glob(os.path.join(a.rollouts, f"{mode}_*.npz")))]
        d = {k: np.concatenate([p[k] for p in parts]) for k in ("ep", "t", "obs", "u", "u_mean")}
        env = KeyInsertionSERL(mode=mode, seed=0)
        agent = make_sac_agent(seed=0, sample_obs=env.observation_space.sample(),
                               sample_action=env.action_space.sample())
        path = os.path.join(SCRIPTS, "key_insertion", "runs", f"{mode}_seed0", "final")
        state = checkpoints.restore_checkpoint(path, target=agent.state)
        assert state is not agent.state, path
        agent = agent.replace(state=state)
        env.close()
        mus, sds = [], []
        for i in range(0, len(d["obs"]), 1024):
            dist = agent.forward_policy(jax.device_put(d["obs"][i:i + 1024]), train=False)
            mus.append(np.asarray(dist.distribution.loc))
            sds.append(np.asarray(dist.distribution.scale_diag))
        mu, sd = np.concatenate(mus).astype(float), np.concatenate(sds).astype(float)
        print(f"{mode}: max |tanh(mu) - logged argmax action| = {np.abs(np.tanh(mu) - d['u_mean']).max():.1e}")
        H = np.zeros_like(mu)
        for _ in range(a.M // 32):
            z = mu[None] + sd[None] * rng.standard_normal((32, *mu.shape))
            H += -log_prob_dims(z, mu[None], sd[None]).sum(0)
        H /= (a.M // 32) * 32
        u = np.clip(d["u"].astype(float), -1 + 1e-6, 1 - 1e-6)
        logp = log_prob_dims(np.arctanh(u), mu, sd)
        np.savez(os.path.join(a.rollouts, f"policy_entropy_{mode}.npz"), ep=d["ep"], t=d["t"], H_dim=H, mu=mu,
                 sigma=sd, logp_logged=logp)
        print(f"{mode}: {len(mu)} obs, mean H(pi) per dim {np.round(H.mean(0), 3)}, "
              f"from logged actions {np.round(-logp.mean(0), 3)}", flush=True)


if __name__ == "__main__":
    main()
