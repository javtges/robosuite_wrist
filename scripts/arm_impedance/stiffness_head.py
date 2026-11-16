"""
A stiffness head: a small network that outputs the wrist's z (stiffness command, +1 extended / soft, -1 contracted /
stiff) from the robot's observations, trained on the scheduled expert's door demonstrations (door_demos.py), then
evaluated closed-loop in the sim with its output driving the wrist through the same compensated switching the
schedules use.

    cd scripts
    # train (needs torch: the robomimic venv), 80 / 20 split by episode, both input sets
    ~/hsa-ws/eval_venv/bin/python -m arm_impedance.stiffness_head train ../demos/door_s*_lock*.hdf5 --out arm_impedance/heads
    # evaluate closed-loop (sim venv, numpy forward): the learned z replaces the schedule, stiff and normal locks
    python -m arm_impedance.stiffness_head eval arm_impedance/heads/head_proprio.npz --workers 4
    python -m arm_impedance.stiffness_head eval arm_impedance/heads/head_vision.npz --workers 4

Inputs (a window of the last K=5 control steps, 20 Hz):
    proprio   tcp pose (7), tool pose (7), flange wrench (6): what the robot has without a camera
    vision    proprio + the key tip's pose in the keyhole frame (7): what a camera would have to provide
The wrist's own state is deliberately left out (see FEATURES). Target: the expert's wrist z HORIZON steps ahead
(0.25 s), so the head must anticipate the switch. The expert's phase is not an input.

The head is a 2-layer MLP (256 units), MSE, Adam. Weights go to an .npz (layers, input normalisation, feature list)
so the sim side needs no torch. Closed-loop, the prediction is rounded to the nearest 0.5 and a switch starts when it
differs from the current command by more than 0.25; the switch itself (rate limit, plate compensation, pause) is
WristScheduler's.
"""

import argparse
import glob
import json
import os
import sys
from collections import deque

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

HERE = os.path.dirname(os.path.abspath(__file__))
K = 5
# No wrist state in the inputs: the measured plate pose tracks the commanded z, and a head given it copies the current
# stiffness instead of deciding it (closed-loop it then never initiates a switch: 0 switches in 30 episodes)
FEATURES = {
    "proprio": ["robot0_tcp_pos", "robot0_tcp_quat", "robot0_eef_pos", "robot0_eef_quat", "flange_wrench"],
    "vision": ["robot0_tcp_pos", "robot0_tcp_quat", "robot0_eef_pos", "robot0_eef_quat", "flange_wrench", "key_tip_to_lock_pos", "key_tip_to_lock_quat"],
}
Z_COL = 8  # the wrist z in the 12-dim action: arm (6), then wrist x, y, z, ...
HORIZON = 5  # the target is the expert's z this many steps ahead (0.25 s): the head has to anticipate the switch


# ---- data ------------------------------------------------------------------------------------------------------------


def episode_features(obs_group, keys):
    """(T, D) raw features of one demo, from its obs/ group"""
    return np.concatenate([np.atleast_2d(obs_group[k][()].reshape(len(obs_group[k]), -1)) for k in keys], axis=1).astype(np.float32)


def window(x, k=K):
    """(T, k*D): each row is the last k rows of x, the first rows padded by repeating row 0"""
    pad = np.repeat(x[:1], k - 1, axis=0)
    xp = np.concatenate([pad, x], axis=0)
    return np.concatenate([xp[i : i + len(x)] for i in range(k)], axis=1)


def load_demos(paths, keys):
    import h5py

    eps = []
    for p in paths:
        f = h5py.File(p, "r")
        for name in sorted(f["data"].keys(), key=lambda s: int(s.split("_")[1])):
            g = f["data"][name]
            if not bool(g.attrs["success"]):
                continue
            z = g["actions"][:, Z_COL].astype(np.float32)
            z_ahead = np.concatenate([z[HORIZON:], np.repeat(z[-1:], HORIZON)])
            eps.append(dict(x=window(episode_features(g["obs"], keys)), z=z_ahead, phase=g["teleop/stage"][()], file=os.path.basename(p), demo=name))
        f.close()
    return eps


# ---- numpy forward (shared by train-time eval and the sim) -----------------------------------------------------------


def forward_np(params, x):
    h = (x - params["mean"]) / params["std"]
    h = np.maximum(h @ params["W1"] + params["b1"], 0.0)
    h = np.maximum(h @ params["W2"] + params["b2"], 0.0)
    return (h @ params["W3"] + params["b3"])[..., 0]


def load_head(path):
    d = np.load(path, allow_pickle=True)
    params = {k: d[k] for k in ("mean", "std", "W1", "b1", "W2", "b2", "W3", "b3")}
    params["keys"] = list(d["keys"])
    params["k"] = int(d["k"])
    return params


# ---- train -----------------------------------------------------------------------------------------------------------


def train(paths, out_dir, variant, epochs=40, hidden=256, seed=0):
    import torch

    torch.manual_seed(seed)
    keys = FEATURES[variant]
    eps = load_demos(paths, keys)
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(eps))
    n_val = max(1, len(eps) // 5)
    val, tr = [eps[i] for i in order[:n_val]], [eps[i] for i in order[n_val:]]
    X = np.concatenate([e["x"] for e in tr]); Z = np.concatenate([e["z"] for e in tr])
    Xv = np.concatenate([e["x"] for e in val]); Zv = np.concatenate([e["z"] for e in val]); Pv = np.concatenate([e["phase"] for e in val])
    mean, std = X.mean(0), X.std(0) + 1e-6
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    net = torch.nn.Sequential(torch.nn.Linear(X.shape[1], hidden), torch.nn.ReLU(), torch.nn.Linear(hidden, hidden), torch.nn.ReLU(), torch.nn.Linear(hidden, 1)).to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-5)
    Xt = torch.tensor((X - mean) / std, device=dev); Zt = torch.tensor(Z, device=dev)
    Xvt = torch.tensor((Xv - mean) / std, device=dev)
    print(f"{variant}: {len(tr)} train / {len(val)} val episodes, {len(X)} / {len(Xv)} steps, {X.shape[1]} inputs, {dev}", flush=True)
    for ep in range(epochs):
        perm = torch.randperm(len(Xt), device=dev)
        net.train()
        for i in range(0, len(Xt), 512):
            idx = perm[i : i + 512]
            loss = torch.nn.functional.mse_loss(net(Xt[idx])[:, 0], Zt[idx])
            opt.zero_grad(); loss.backward(); opt.step()
        if ep % 10 == 9 or ep == epochs - 1:
            net.eval()
            with torch.no_grad():
                pv = net(Xvt)[:, 0].cpu().numpy()
            print(f"  epoch {ep + 1}: train mse {loss.item():.4f}, val MAE {np.abs(pv - Zv).mean():.3f}", flush=True)
    net.eval()
    with torch.no_grad():
        pv = net(Xvt)[:, 0].cpu().numpy()
    W = [l for l in net if isinstance(l, torch.nn.Linear)]
    params = dict(mean=mean.astype(np.float32), std=std.astype(np.float32), keys=np.array(keys), k=np.array(K),
                  W1=W[0].weight.T.detach().cpu().numpy(), b1=W[0].bias.detach().cpu().numpy(),
                  W2=W[1].weight.T.detach().cpu().numpy(), b2=W[1].bias.detach().cpu().numpy(),
                  W3=W[2].weight.T.detach().cpu().numpy(), b3=W[2].bias.detach().cpu().numpy())  # fmt: skip
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"head_{variant}.npz")
    np.savez(path, **params)
    # report: MAE per expert phase, and switch timing on the validation episodes
    names = ("approach", "align", "insert", "hold", "turn", "push", "done")
    per_phase = {names[p]: round(float(np.abs(pv - Zv)[Pv == p].mean()), 3) for p in np.unique(Pv)}
    timing = []
    for e in val:
        p = forward_np(load_head(path), e["x"]); z = e["z"]
        up_t = np.argmax(z > 0.5) if (z > 0.5).any() else -1; up_p = np.argmax(p > 0.5) if (p > 0.5).any() else -1
        dn_t = up_t + np.argmax(z[up_t:] < 0.5) if up_t >= 0 and (z[up_t:] < 0.5).any() else -1
        dn_p = up_p + np.argmax(p[up_p:] < 0.5) if up_p >= 0 and (p[up_p:] < 0.5).any() else -1
        timing.append((up_p - up_t, dn_p - dn_t))
    timing = np.array(timing)
    report = dict(variant=variant, inputs=int(X.shape[1]), train_episodes=len(tr), val_episodes=len(val), val_mae=round(float(np.abs(pv - Zv).mean()), 4), val_mae_per_phase=per_phase,
                  soften_timing_steps=dict(median=float(np.median(timing[:, 0])), p90_abs=float(np.percentile(np.abs(timing[:, 0]), 90))),
                  stiffen_timing_steps=dict(median=float(np.median(timing[:, 1])), p90_abs=float(np.percentile(np.abs(timing[:, 1]), 90))))
    with open(os.path.join(out_dir, f"head_{variant}_report.json"), "w") as f:
        json.dump(report, f, indent=1)
    print(json.dumps(report, indent=1))
    return path


# ---- closed-loop evaluation in the sim --------------------------------------------------------------------------------


class LearnedScheduler:
    """on_step hook: the head's z drives the wrist through WristScheduler's compensated switching"""

    def __init__(self, params, threshold=0.25, quantum=0.5):
        from arm_impedance.door_study import PHASES, SCHEDULES, door_phase
        from arm_impedance.wrist_study import WristScheduler

        self.params, self.threshold, self.quantum = params, threshold, quantum
        self.inner = WristScheduler("fixed_home", schedules=SCHEDULES, phase_fn=door_phase, order=PHASES)
        self.inner.current = "push"  # the phase table is never consulted: the head decides
        self.hist = deque(maxlen=int(params["k"]))
        self.trace = []
        self.pred = None

    def __call__(self, env, oracle, row):
        obs = env._get_observations()
        x = np.concatenate([np.ravel(obs[k]) for k in self.params["keys"]]).astype(np.float32)
        while len(self.hist) < self.hist.maxlen:
            self.hist.append(x)
        self.hist.append(x)
        feat = np.concatenate(list(self.hist))
        self.pred = float(np.clip(forward_np(self.params, feat[None])[0], -1.0, 1.0))
        target = float(np.clip(np.round(self.pred / self.quantum) * self.quantum, -1.0, 1.0))
        self.trace.append((row["t"], self.pred, target))
        if not self.inner.switching and abs(target - oracle.wrist_z) > self.threshold:
            self.inner._begin_switch(oracle, row, target)
        self.inner._advance(oracle, row)
        return True

    @property
    def n_switches(self):
        return self.inner.n_switches

    @property
    def switch_tool_mm(self):
        return self.inner.switch_tool_mm


_ENVS = {}


def _eval_case(case):
    from key_insertion.door_oracle import DoorOracle, make_door_env, run_door_episode

    fr = case["cylinder_friction"]
    if fr not in _ENVS:
        _ENVS[fr] = make_door_env(arm="hsa", cylinder_friction=fr, lock_x_range=(0, 0), lock_y_range=(0, 0), lock_yaw_range=(0, 0))
    env = _ENVS[fr]
    env.rng = np.random.default_rng(0)
    env.reset()
    params = load_head(case["head"])
    oracle = DoorOracle(env, wrist_z=0.0, perception_error=(case["dx_mm"] * 1e-3, 0.0, np.radians(case["dyaw_deg"])), max_force=np.inf)
    hook = LearnedScheduler(params)
    result, log = run_door_episode(env, oracle, max_seconds=40.0, on_step=hook, reset=False)
    ph = log["phase"]
    f = np.linalg.norm(log["flange_force"], axis=1)
    mom = np.linalg.norm(log["flange_moment"], axis=1)
    peak = lambda p, a: float(a[ph == p].max()) if (ph == p).any() else 0.0  # noqa: E731
    z = np.array([t for _, _, t in hook.trace])
    return dict(case, result=result, success=result == "opened", peak_insert_N=peak(DoorOracle.INSERT, f), peak_turn_N=peak(DoorOracle.TURN, f),
                peak_turn_Nm=peak(DoorOracle.TURN, mom), peak_push_N=peak(DoorOracle.PUSH, f), n_switches=hook.n_switches, switch_tool_mm=round(hook.switch_tool_mm, 2),
                z_insert=float(z[ph == DoorOracle.INSERT].mean()) if (ph == DoorOracle.INSERT).any() else float("nan"),
                z_turn=float(z[ph == DoorOracle.TURN].mean()) if (ph == DoorOracle.TURN).any() else float("nan"))  # fmt: skip


def evaluate(head, workers):
    import csv
    import multiprocessing as mp

    errors = [(dx, dyaw) for dx in (-1.5, 0.0, 1.5) for dyaw in (-4.0, -2.0, 0.0, 2.0, 4.0)]
    cases = [dict(head=head, cylinder_friction=fr, dx_mm=dx, dyaw_deg=dyaw) for fr in (0.15, 0.4) for dx, dyaw in errors]
    os.nice(10)
    rows = []
    with mp.get_context("spawn").Pool(workers) as pool:
        for i, r in enumerate(pool.imap_unordered(_eval_case, cases, chunksize=1)):
            rows.append(r)
            print(f"  {i + 1}/{len(cases)} lock {r['cylinder_friction']:g} dyaw {r['dyaw_deg']:+.0f}: {r['result']:12s} z insert {r['z_insert']:.2f} turn {r['z_turn']:.2f} switches {r['n_switches']} | turn {r['peak_turn_Nm']:.2f} N m push {r['peak_push_N']:.0f} N", flush=True)
    rows.sort(key=lambda r: (r["cylinder_friction"], r["dx_mm"], r["dyaw_deg"]))
    name = os.path.splitext(os.path.basename(head))[0]
    out = os.path.join(os.path.dirname(head), f"{name}_eval.csv")
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    print(f"\n{name}: closed-loop in the sim, 15 aiming errors per lock")
    for fr in (0.15, 0.4):
        sel = [r for r in rows if r["cylinder_friction"] == fr]
        print(f"  lock {fr:g} N m: opened {sum(r['success'] for r in sel)}/{len(sel)}, median peak turn {np.median([r['peak_turn_Nm'] for r in sel]):.2f} N m, push {np.median([r['peak_push_N'] for r in sel]):.0f} N, "
              f"z during insert {np.nanmedian([r['z_insert'] for r in sel]):.2f}, during turn {np.nanmedian([r['z_turn'] for r in sel]):.2f}, switches {np.median([r['n_switches'] for r in sel]):.0f}")
    print("wrote", out)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("train"); t.add_argument("demos", nargs="+"); t.add_argument("--out", default=os.path.join(HERE, "heads")); t.add_argument("--variants", nargs="+", default=list(FEATURES)); t.add_argument("--epochs", type=int, default=40)
    e = sub.add_parser("eval"); e.add_argument("head"); e.add_argument("--workers", type=int, default=4)
    args = p.parse_args()
    if args.cmd == "train":
        paths = sorted(sum((glob.glob(d) for d in args.demos), []))
        for v in args.variants:
            train(paths, args.out, v, epochs=args.epochs)
    else:
        evaluate(args.head, args.workers)


if __name__ == "__main__":
    main()
