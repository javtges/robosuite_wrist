"""
Oracle "demonstrations" for SERL: rolls out ScriptedPolicy in KeyInsertionWrist-v0 and saves every transition in SERL's
demo format (a pickled list of dicts with observations, actions, next_observations, rewards, masks, dones; what
SERL's record_demo.py scripts write and its learners load with --demo_path).

    cd scripts
    python -m key_insertion.collect_demos --mode variable --n 20       # -> key_insertion/demos/variable_20.pkl
    python -m key_insertion.collect_demos --mode all --n 20            # stiff, soft and variable

Every rollout is kept, seated or not, so each policy gets the same demonstrator: in soft mode the oracle can't pass the
latch inside the safety box, and its demos are all failures (reward 0).
"""

import argparse
import os
import pickle
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from key_insertion.serl_env import FORCE_PENALTY, MODES, KeyInsertionSERL, ScriptedPolicy  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))


def collect(mode, n, seed=0, noise=0.1, force_penalty=FORCE_PENALTY):
    """Returns (transitions, per-episode summaries)"""
    env = KeyInsertionSERL(mode=mode, seed=seed, force_penalty=force_penalty)
    policy = ScriptedPolicy(env, noise=noise, seed=seed)
    transitions, episodes = [], []
    for ep in range(n):
        obs, _ = env.reset()
        done = truncated = False
        peak = 0.0
        while not (done or truncated):
            action = policy(obs)
            next_obs, reward, done, truncated, info = env.step(action)
            transitions.append(
                dict(
                    observations=obs,
                    actions=action,
                    next_observations=next_obs,
                    rewards=np.float32(reward),
                    masks=np.float32(1.0 - done),
                    dones=bool(done or truncated),
                )
            )
            obs, peak = next_obs, max(peak, info["max_force"])
        episodes.append(dict(seated=info["succeed"], steps=env.steps, peak_force=peak, over_force=info["over_force"]))
    env.close()
    return transitions, episodes


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mode", default="all", choices=[*MODES, "all"])
    p.add_argument("--n", type=int, default=20, help="episodes")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--noise", type=float, default=0.1, help="std of the oracle's action noise (motion only)")
    p.add_argument("--force_penalty", type=float, default=FORCE_PENALTY, help="weight of the env's force penalty (as train_serl)")
    p.add_argument("--out", default=os.path.join(HERE, "demos"), help="output directory")
    args = p.parse_args()

    os.makedirs(args.out, exist_ok=True)
    for mode in MODES if args.mode == "all" else [args.mode]:
        transitions, episodes = collect(mode, args.n, args.seed, args.noise, args.force_penalty)
        path = os.path.join(args.out, f"{mode}_{args.n}.pkl")
        with open(path, "wb") as f:
            pickle.dump(transitions, f)
        seated = sum(e["seated"] for e in episodes)
        peak = np.median([e["peak_force"] for e in episodes])
        steps = np.median([e["steps"] for e in episodes])
        print(
            f"{mode:8s}: {len(transitions)} transitions, {seated}/{args.n} seated, median {steps:.0f} steps, "
            f"median peak flange force {peak:.1f} N -> {os.path.relpath(path)}"
        )


if __name__ == "__main__":
    main()
