"""
Empirical (simulator) controllability Gramians along the KeyDoor scheduled expert's episodes, for three wrist
schedules. Uses key_insertion.door_demos.record_episode unchanged (with an in-memory writer), so the expert runs exactly
as when it records demos. Changes nothing in the repo.

Pass 1 records an episode (states, actions, phases). Pass 2 re-runs the identical episode and, before the step at every
STRIDE-th index n, forks the process (an exact snapshot: physics, warm start, controllers) once per input direction and
sign: the child applies u_n +- eps e_i, then the recorded u_{n+1..n+N-1} open-loop, and returns x_{n+1..n+N}.
R[n][h, :, i] = (x_{n+h+1}(+) - x_{n+h+1}(-)) / (2 eps), h = 0..N-1, in input units.

Run with robosuite_env from robosuite_wrist/scripts:
    python <this file> --out <dir> [--episodes 8] [--N 20] [--stride 2]
"""

import argparse
import multiprocessing as mp
import os
import pickle
import sys

import numpy as np

SCRIPTS = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # robosuite_wrist/scripts
sys.path.insert(0, SCRIPTS)

SCHEDULES_USED = {"stiff": "fixed_stiff", "soft": "fixed_soft", "variable": "soft_insert_home_turn"}
# input units: arm position 2 mm, arm rotation 1 deg (as the SERL task's per-step limits), wrist z command 1 (of -1..1)
POS_UNIT, ROT_UNIT, Z_UNIT = 0.002, np.radians(1.0), 1.0
EPS = 0.5  # perturbation, in input units (1 mm, 0.5 deg); z uses EPS_Z
EPS_Z = 0.05
N_INPUTS = 7  # arm x, y, z, rx, ry, rz (base frame), wrist z


class MemWriter:
    """stands in for demo_writer.DemoWriter in record_episode"""

    def __init__(self):
        self.recording = False
        self.phases, self._attrs = [], {}

    def start(self, attrs):
        self.recording = True
        self.phases = []

    def add(self, state, action, reward, obs_before, obs, phase, *rest):
        self.phases.append(int(phase))

    def save(self, **kw):
        self.recording = False


def raw_state(env):
    """tip pos (3) and R (9) in the lock frame, tip velocity in the lock frame (3), wrist leg wrench (6: m, f),
    door angle, cylinder angle"""
    p, R = env.key_tip_in_lock()
    _, R_lock = env.lock_pose()
    vel = R_lock.T @ env.sim.data.get_site_xvelp(env.sim.model.site_id2name(env.tip_site_id))
    w = env.robots[0].gripper["right"].leg_wrench(env.sim)[1]
    return np.concatenate([p, R.ravel(), vel, w, [env.door_angle(), env.cylinder_angle()]])


def perturb(u, i, s):
    from scipy.spatial.transform import Rotation

    u = u.astype(float).copy()
    if i < 3:
        u[i] += s * EPS * POS_UNIT
    elif i < 6:
        e = np.zeros(3)
        e[i - 3] = s * EPS * ROT_UNIT
        u[3:6] = (Rotation.from_rotvec(e) * Rotation.from_rotvec(u[3:6])).as_rotvec()  # about the base axes
    else:
        u[8] += s * EPS_Z * Z_UNIT
    return u


def child_rollout(env, step, u0, future, N):
    xs = []
    step(u0)
    xs.append(raw_state(env))
    for u in future[: N - 1]:
        step(u)
        xs.append(raw_state(env))
    return np.array(xs)


def fork_call(fn):
    r, w = os.pipe()
    pid = os.fork()
    if pid == 0:
        os.close(r)
        try:
            out = pickle.dumps(fn())
        except Exception as ex:  # noqa: BLE001
            out = pickle.dumps(ex)
        with os.fdopen(w, "wb") as f:
            f.write(out)
        os._exit(0)
    os.close(w)
    with os.fdopen(r, "rb") as f:
        data = f.read()
    os.waitpid(pid, 0)
    res = pickle.loads(data)
    if isinstance(res, Exception):
        raise res
    return res


def run_one(args):
    mode, case, N, stride, out = args
    import robosuite as suite

    from hsa_robots import DEFAULT_ROBOT
    from key_insertion.door_demos import make_env_args, record_episode

    schedule = SCHEDULES_USED[mode]
    env_args = make_env_args(DEFAULT_ROBOT, 20.0, case["friction"])
    env = suite.make(env_args["env_name"], **env_args["env_kwargs"])
    real_step = env.step
    attrs = dict(cylinder_friction=case["friction"], plan_index=case["k"])

    # ---- pass 1: record
    w1 = MemWriter()
    X, U = [], []

    def step1(action):
        if w1.recording:
            X.append(raw_state(env))
            U.append(np.asarray(action, float).copy())
        return real_step(action)

    env.step = step1
    env.rng = np.random.default_rng(case["seed"])
    result, steps = record_episode(env, w1, schedule, case["dx"], case["dyaw"], attrs)
    X.append(raw_state(env))
    X, U = np.array(X), np.array(U)

    # ---- pass 2: identical episode, forking at every stride-th step
    w2 = MemWriter()
    count = [0]
    R = {}
    mismatch = [0.0]

    def step2(action):
        if w2.recording:
            n = count[0]
            if n < len(U):
                mismatch[0] = max(mismatch[0], float(np.abs(np.asarray(action) - U[n]).max()))
            if n % stride == 0 and n + N <= len(U):
                resp = np.full((N, 23, N_INPUTS), np.nan)
                for i in range(N_INPUTS):
                    eps = EPS_Z if i == 6 else EPS
                    xp = fork_call(lambda: child_rollout(env, real_step, perturb(U[n], i, +1), U[n + 1:], N))
                    xm = fork_call(lambda: child_rollout(env, real_step, perturb(U[n], i, -1), U[n + 1:], N))
                    from scipy.spatial.transform import Rotation

                    d = (xp - xm) / (2 * eps)
                    # rotation: relative rotation between the + and - runs, as a rotation vector in the lock frame
                    for h in range(N):
                        Rp, Rm = xp[h, 3:12].reshape(3, 3), xm[h, 3:12].reshape(3, 3)
                        d[h, 3:6] = Rotation.from_matrix(Rp @ Rm.T).as_rotvec() / (2 * eps)
                        d[h, 6:12] = np.nan
                    resp[:, :, i] = d
                R[n] = resp
            count[0] += 1
        return real_step(action)

    env.step = step2
    env.rng = np.random.default_rng(case["seed"])
    result2, steps2 = record_episode(env, w2, schedule, case["dx"], case["dyaw"], attrs)
    env.close()
    np.savez(os.path.join(out, f"{mode}_ep{case['k']}.npz"), X=X, U=U, phase=np.array(w1.phases),
             pulse_n=np.array(sorted(R)), R=np.array([R[n] for n in sorted(R)]), result=result, result2=result2,
             mismatch=mismatch[0], dx=case["dx"], dyaw=case["dyaw"], friction=case["friction"], k=case["k"])
    return mode, case["k"], result, result2, steps, mismatch[0], len(R)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--episodes", type=int, default=8)
    ap.add_argument("--N", type=int, default=20)
    ap.add_argument("--stride", type=int, default=2)
    ap.add_argument("--workers", type=int, default=9)
    ap.add_argument("--modes", nargs="+", default=list(SCHEDULES_USED))
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    rng = np.random.default_rng(0)  # door_demos' plan: same draws, same order
    fr = [0.15, 0.4]
    plan = [dict(k=k, friction=fr[k % 2], dx=float(rng.uniform(-1.5, 1.5)), dyaw=float(rng.uniform(-3.0, 3.0)),
                 seed=int(rng.integers(1 << 31))) for k in range(a.episodes)]
    jobs = [(m, c, a.N, a.stride, a.out) for c in plan for m in a.modes]
    with mp.get_context("spawn").Pool(a.workers) as pool:
        for mode, k, r1, r2, steps, mm, npulse in pool.imap_unordered(run_one, jobs):
            print(f"{mode:8s} ep {k}: {r1} / rerun {r2}, {steps} steps, action mismatch {mm:.2e}, {npulse} pulse points", flush=True)


if __name__ == "__main__":
    main()
