"""
KeyDoor as a residual-RL gym environment (gym 0.26 API, like key_insertion/serl_env.py): a frozen base (the diffusion
policy, or the scripted DoorOracle with a wrist schedule) acts every step, and the agent's action is a bounded residual
added to the base's 12-dim robosuite action. 20 Hz, one robosuite control step per env step.

    import visuomotor.residual.door_residual_env                     # registers DoorResidual-v0
    env = gym.make("DoorResidual-v0", structure="wrist_z", base="oracle")
    env = gym.make("DoorResidual-v0", structure="wrist_6d", base="dp", base_ckpt="../runs/dp/ckpt.pt")

    python -m visuomotor.residual.door_residual_env --structure arm_wrist_6d --steps 40   # plumbing check

Structures (agent action in [-1, 1]^n; the stroke, i.e. the wrist's normalized range, is the trust region):
    wrist_z        n = 1   wrist z:            z = clip(z_base + s_w[z] a, -1, 1)
    wrist_6d       n = 6   wrist x..rz:        w = clip(w_base + s_w * a, -1, 1)
    arm_wrist_z    n = 7   arm (6) + wrist z
    arm_wrist_6d   n = 12  arm (6) + wrist (6)
The arm residual is a tcp position delta (+-pos_scale m) and a rotation delta (+-rot_scale rad, rotation vector)
composed onto the base's absolute target, both in the arm base frame: p = p_base + s_p a[:3], R = exp(s_r a[3:6]) R_base.
A wrist z change also moves the plate (20 mm per unit z) and changes the stiffness by ~100x; with z_compensation the
arm's target is moved the other way along the tool axis by the commanded plate travel (a feedforward, as the
WristScheduler's switch does with the measured travel), so z acts mostly as a stiffness channel.

Exploration (exploration.py, set_sampler) perturbs the wrist part only: executed = clip(base + residual + perturbation),
then rate-limited if wrist_rate > 0.
info["residual_executed"] is the executed action mapped back into the agent's action space (clipped): what the replay
buffer stores. info["executed_action"] is the 12-dim robosuite action. Eval envs have no sampler.

Observation (flat float32, scaled to O(1) like serl_env):
    tcp pose relative to the base's current target, arm base frame: position (3, /1 cm), rot6d minus identity (6, /0.1)
    the wrist's leg-sensed wrench (6, common.WRENCH_SCALE: moment /2 N m, force /20 N), load compensated, flange frame
    wrist deflection from its command (6, /5 mm, /0.05 rad; leg encoders)
    wrist command in force (6, normalized)
    base action: tcp position (3, m, base frame), rot6d (6), wrist z (1)
    [privileged_obs] key tip in the lock frame (3, /1 cm), its rot6d (6), cylinder / unlock angle, door / open angle
    [phase_obs] one-hot of the task phase from the env's state (env_phase: free search slide seat turn push)

Reward: 1 on success (door opened, env._check_success()), minus force_penalty * (F / force_limit)^2 every step, F the
peak flange force over the step's physics steps (force_measure "step": at the step's end). Terminates on success or F > force_limit (then no success reward).
Truncated after max_steps; also when the oracle base gives up (end_on_base_failure: "cannot turn", "door stuck", ...).
With skip_approach the free-space approach is run before the episode starts with zero residual (oracle: until its
align phase; dp: until the key tip is within approach_height of the lock face), so episodes start near the lock.
"""

import argparse
import os
import sys

import numpy as np
from scipy.spatial.transform import Rotation

try:
    import gym
    from gym import spaces
except ImportError:  # the robosuite venv has gymnasium only
    import gymnasium as gym
    from gymnasium import spaces

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import visuomotor.residual.compat  # noqa: E402,F401  (before any env is built)
from visuomotor import common  # noqa: E402

STRUCTURES = {
    "wrist_z": dict(arm=False, wrist_axes=(2,)),
    "wrist_6d": dict(arm=False, wrist_axes=(0, 1, 2, 3, 4, 5)),
    "arm_wrist_z": dict(arm=True, wrist_axes=(2,)),
    "arm_wrist_6d": dict(arm=True, wrist_axes=(0, 1, 2, 3, 4, 5)),
}
WRIST_AXES = ("x", "y", "z", "rx", "ry", "rz")
ENV_PHASES = ("free", "search", "slide", "seat", "turn", "push")  # arm_impedance.door_study.PHASES
FORCE_LIMIT = 80.0  # N at the flange, as serl_env
FORCE_PENALTY = 0.25
DEFLECTION_SCALE = np.array([0.005] * 3 + [0.05] * 3)
I6 = np.array([1.0, 0, 0, 0, 1.0, 0])  # rot6d of the identity
APPROACH_HEIGHT = 0.025  # dp base, skip_approach: the approach ends with the key tip this close above the lock face (m)
MAX_SKIP_STEPS = 400


def env_phase(env):
    """Task phase from the env's state alone (works for any base): door_study's phases without the expert's memory"""
    if env.unlocked() or env._check_success():
        return "push"
    if env.key_seated():
        return "turn" if env.cylinder_angle() > np.radians(3.0) else "seat"
    depth = env.insertion_depth()
    if depth >= 0.024:
        return "seat"
    if depth >= 0.003:
        return "slide"
    p, _ = env.key_tip_in_lock()
    return "search" if depth > -0.003 and np.linalg.norm(p[:2]) < 0.01 else "free"


def tcp_pose_base(env):
    """(position, rotation matrix) of the arm's tcp in the arm base frame (the action's frame)"""
    robot = env.robots[0]
    site = robot.gripper["right"].important_sites["tcp"]
    p0, R0 = robot.composite_controller.get_controller_base_pose(controller_name="right")
    d = env.sim.data
    return R0.T @ (d.get_site_xpos(site) - p0), R0.T @ d.get_site_xmat(site)


# ---- bases -----------------------------------------------------------------------------------------------------------
class OracleBase:
    """DoorOracle + WristScheduler (a phase schedule or a static z), as visuomotor/gen_demos.py records the demos"""

    needs_cameras = False

    def __init__(self, schedule="soft_insert_home_turn", static_z=None, max_dx=1.5, max_dyaw=3.0):
        from arm_impedance.door_study import SCHEDULES

        assert static_z is not None or schedule in SCHEDULES, f"unknown schedule {schedule}; choose from {list(SCHEDULES)}"
        self.schedule, self.static_z, self.max_dx, self.max_dyaw = schedule, static_z, max_dx, max_dyaw
        self.oracle = self.hook = None

    def reset(self, env, rng):
        """After env.reset(): a fresh oracle with a random aiming error; its reset() settles the wrist for 0.5 s"""
        from arm_impedance.door_study import PHASES, SCHEDULES, door_phase
        from arm_impedance.wrist_study import WristScheduler
        from key_insertion.door_oracle import DoorOracle

        kw = dict(phase_fn=door_phase, order=PHASES)
        if self.static_z is not None:
            z0 = float(self.static_z)
            self.hook = WristScheduler("static", schedules={"static": {p: (z0, 1.0) for p in PHASES}}, **kw)
        else:
            z0 = SCHEDULES[self.schedule]["free"][0]
            self.hook = WristScheduler(self.schedule, schedules=SCHEDULES, **kw)
        dx, dyaw = rng.uniform(-self.max_dx, self.max_dx), rng.uniform(-self.max_dyaw, self.max_dyaw)
        self.oracle = DoorOracle(env, wrist_z=z0, perception_error=(dx * 1e-3, 0.0, np.radians(dyaw)), max_force=np.inf)
        self.oracle.reset()

    def act(self, env, raw_obs, wrist_cmd=None):
        return np.asarray(self.oracle.action(self.oracle.next_target()), dtype=float)

    def after_step(self, env):
        self.hook(env, self.oracle, self.oracle.log_row())

    def wrist_cmd0(self):
        return self.oracle.wrist_action.copy()

    def approached(self, env):
        return self.oracle.phase >= self.oracle.ALIGN

    def failed(self):
        return self.oracle.done and self.oracle.result not in (None, "opened")

    @property
    def phase(self):
        return self.oracle.PHASES[self.oracle.phase]


class DPBase:
    """The frozen diffusion policy (visuomotor.dp.policy.DiffusionPolicyRunner); its env is built from its config"""

    def __init__(self, ckpt, device="cpu"):
        from visuomotor.dp.policy import DiffusionPolicyRunner

        self.runner = DiffusionPolicyRunner(ckpt, device=device)
        self.cfg = self.runner.cfg
        self.needs_cameras = bool(self.runner.needs_cameras)
        self.wrist_cmd = np.zeros(6)

    def _runner_wrist_cmd(self):
        w = getattr(self.runner, "wrist_cmd", None)
        if w is None:  # before the first act(): the policy's static wrist, or home
            w = np.zeros(6)
            w[2] = common.POLICY_WRIST.get(self.cfg.get("wrist", "medium")) or 0.0
        return np.asarray(w, dtype=float)

    def reset(self, env, rng):
        """After env.reset(): reset the policy and settle the wrist for 0.5 s holding the arm, as the demos started"""
        self.runner.reset()
        self.wrist_cmd = self._runner_wrist_cmd()
        p, R = tcp_pose_base(env)
        hold = np.concatenate([p, Rotation.from_matrix(R).as_rotvec(), self.wrist_cmd])
        for _ in range(int(round(0.5 * env.control_freq))):
            env.step(hold)

    def act(self, env, raw_obs, wrist_cmd=None):
        # wrist_cmd is an observation: the command in force, i.e. the last EXECUTED wrist part (base + residual +
        # exploration), as the dataset's wrist_cmd was the executed one (policy.py's caller contract). With wrist_6d
        # the residual's x, y, rotations are nonzero there, which the policy never saw in its data.
        obs = common.obs_extras(env, raw_obs, self.wrist_cmd if wrist_cmd is None else wrist_cmd, -1)
        a = np.asarray(self.runner.act(obs), dtype=float)
        self.wrist_cmd = a[common.WRIST].copy()
        return a

    def after_step(self, env):
        pass

    def wrist_cmd0(self):
        return self.wrist_cmd.copy()

    def approached(self, env):
        p, _ = env.key_tip_in_lock()
        return p[2] < APPROACH_HEIGHT and np.linalg.norm(p[:2]) < 0.02

    def failed(self):
        return False

    @property
    def phase(self):
        return "dp"


# ---- env -------------------------------------------------------------------------------------------------------------
class DoorResidualEnv(gym.Env):
    """
    Args:
        structure (str): see STRUCTURES / the module docstring
        base (str): "oracle" or "dp"; base_ckpt: the dp checkpoint
        base_schedule / base_static_z: the oracle's wrist (a door_study schedule, or a static normalized z if not None)
        base_max_dx / base_max_dyaw: the oracle's aiming error, uniform +- (mm, deg), drawn per episode
        cylinder_friction (float or list): lock friction (N m); a list is cycled over episodes (one robosuite env each)
        wrist_scale (float or 6 floats): wrist residual scale per axis, normalized units (1 = the full range)
        pos_scale (m) / rot_scale (rad): arm residual bounds
        wrist_rate (float): if > 0, the executed wrist command moves at most this much per second (normalized units,
            every axis; the expert switches z at 1 /s), so a residual cannot jump the plate within one step
        z_compensation (bool): feedforward plate-travel compensation of wrist z changes on the arm's target
        privileged_obs / phase_obs (bool): extra observations (module docstring)
        force_limit (N), force_penalty: reward / termination (module docstring)
        force_measure (str): "physics" (peak over the step's physics steps, catches command-jump spikes) or "step" (the
            flange force at the end of the step, as a 20 Hz F/T reading would see it)
        max_steps (int): episode length in env steps (20 Hz)
        skip_approach (bool): run the free-space approach with zero residual before the episode starts
        end_on_base_failure (bool): truncate when the oracle base gives up
        seed (int or None)
    """

    metadata = {"render_modes": [], "render_fps": 20}

    def __init__(
        self, structure="wrist_z", base="oracle", base_ckpt=None, base_schedule="soft_insert_home_turn", base_static_z=None,
        base_max_dx=1.5, base_max_dyaw=3.0, cylinder_friction=0.15, wrist_scale=1.0, pos_scale=0.003,
        rot_scale=np.radians(2.0), wrist_rate=1.0, z_compensation=False, privileged_obs=False, phase_obs=False, force_limit=FORCE_LIMIT,
        force_penalty=FORCE_PENALTY, force_measure="physics", max_steps=300, skip_approach=True, end_on_base_failure=True, seed=None, device="cpu",
    ):  # fmt: skip
        assert structure in STRUCTURES, f"unknown structure {structure}; choose from {list(STRUCTURES)}"
        self.structure = structure
        self.arm = STRUCTURES[structure]["arm"]
        self.wrist_axes = np.array(STRUCTURES[structure]["wrist_axes"])
        self.wrist_scale = np.broadcast_to(np.asarray(wrist_scale, dtype=float), (6,)).copy()
        self.pos_scale, self.rot_scale = float(pos_scale), float(rot_scale)
        self.z_compensation = bool(z_compensation)
        self.wrist_rate = float(wrist_rate)
        self.privileged_obs, self.phase_obs = bool(privileged_obs), bool(phase_obs)
        self.force_limit, self.force_penalty = float(force_limit), float(force_penalty)
        assert force_measure in ("physics", "step"), force_measure
        self.force_measure = force_measure
        self.max_steps, self.skip_approach, self.end_on_base_failure = int(max_steps), bool(skip_approach), bool(end_on_base_failure)
        self.frictions = [float(f) for f in np.atleast_1d(cylinder_friction)]
        self.np_random = np.random.default_rng(seed)

        if base == "oracle":
            self.base = OracleBase(base_schedule, base_static_z, base_max_dx, base_max_dyaw)
            self.env_args, self.ee_camera = common.default_env_args(cameras=()), None
        elif base == "dp":
            assert base_ckpt, "base dp needs base_ckpt"
            self.base = DPBase(base_ckpt, device=device)
            self.env_args, self.ee_camera = self.base.cfg["env_args"], self.base.cfg.get("ee_camera")
        else:
            raise ValueError(f"unknown base {base}")
        self._envs = {}
        self.episodes = 0
        self._select_env(self.frictions[0])

        n = (6 if self.arm else 0) + len(self.wrist_axes)
        self.action_space = spaces.Box(-1.0, 1.0, (n,), dtype=np.float32)
        self.sampler = None  # exploration.WristSampler, training env only
        self.explore = True
        self._reset_state()
        self.observation_space = spaces.Box(-np.inf, np.inf, (len(self._obs()),), dtype=np.float32)

    # ---- setup -------------------------------------------------------------------------------------------------------
    def _select_env(self, friction):
        if friction not in self._envs:
            env = common.make_env(self.env_args, self.ee_camera, cameras=self.base.needs_cameras, cylinder_friction=friction)
            self._envs[friction] = env
            self._watch_physics(env)
        self.env = self._envs[friction]
        self.friction = friction
        self.robot = self.env.robots[0]
        self.gripper = self.robot.gripper["right"]
        self.wrist = self.gripper.wrist

    def _reset_state(self):
        p, R = tcp_pose_base(self.env)
        self.base_action = np.concatenate([p, Rotation.from_matrix(R).as_rotvec(), np.zeros(6)])
        self.wrist_cmd = np.zeros(6)  # in force (executed)
        self.steps, self._physics_peak, self.peak_force = 0, 0.0, 0.0

    def set_sampler(self, sampler):
        """exploration.WristSampler on the active wrist axes (None: no exploration)"""
        if sampler is not None:
            assert list(sampler.axes) == list(self.wrist_axes) or set(sampler.axes) <= set(self.wrist_axes)
        self.sampler = sampler

    def _watch_physics(self, env):
        """Wraps the sim's physics step to track the peak flange force over every physics step (as serl_env)"""
        sim = env.sim
        name = "step2" if env.lite_physics else "step"
        physics_step = getattr(sim, name)
        gripper = env.robots[0].gripper["right"]

        def stepped(*args, **kwargs):
            physics_step(*args, **kwargs)
            self._physics_peak = max(self._physics_peak, float(np.linalg.norm(gripper.flange_wrench(sim)[1][3:])))

        setattr(sim, name, stepped)

    # ---- state -------------------------------------------------------------------------------------------------------
    def leg_wrench(self):
        """(moment, force) the wrist senses with its legs, flange frame, load compensated"""
        return np.asarray(self.gripper.leg_wrench(self.env.sim)[1], dtype=float)

    def wrist_pose(self):
        """Measured top plate pose relative to home (m, rad), leg encoders"""
        d = self.env.sim.data
        return np.array([d.get_joint_qpos(self.gripper.correct_naming(f"wrist_top_{a}")) for a in WRIST_AXES])

    def wrist_deflection(self):
        return self.wrist_pose() - self.wrist.last_commanded_pose

    def wrist_R(self):
        """World rotation of the wrist base (the deflection's and the leg wrench's frame)"""
        b = self.env.sim.model._model.body(self.gripper.worldbody[0].get("name")).id
        return self.env.sim.data._data.xmat[b].reshape(3, 3)

    def _obs(self):
        env = self.env
        p, R = tcp_pose_base(env)
        R_t = Rotation.from_rotvec(self.base_action[3:6]).as_matrix()
        rel = [R_t.T @ (p - self.base_action[:3]) / 0.01, (common.rotvec_to_rot6d(Rotation.from_matrix(R_t.T @ R).as_rotvec()) - I6) / 0.1]
        obs = rel + [
            self.leg_wrench() / common.WRENCH_SCALE,
            self.wrist_deflection() / DEFLECTION_SCALE,
            self.wrist_cmd,
            self.base_action[:3],
            common.rotvec_to_rot6d(self.base_action[3:6]),
            self.base_action[common.WRIST_Z:common.WRIST_Z + 1],
        ]
        if self.privileged_obs:
            pk, Rk = env.key_tip_in_lock()
            obs += [pk / 0.01, common.rotvec_to_rot6d(Rotation.from_matrix(Rk).as_rotvec()),
                    [env.cylinder_angle() / env.unlock_angle, env.door_angle() / env.open_angle]]  # fmt: skip
        if self.phase_obs:
            obs.append(np.eye(len(ENV_PHASES))[ENV_PHASES.index(env_phase(env))])
        return np.concatenate([np.ravel(o) for o in obs]).astype(np.float32)

    # ---- action mapping ----------------------------------------------------------------------------------------------
    def compose(self, base_action, action, perturbation=None):
        """(base 12-dim robosuite action, agent action, wrist perturbation on the active axes) -> executed 12-dim"""
        a = np.clip(np.asarray(action, dtype=float), -1.0, 1.0)
        out = np.array(base_action, dtype=float)
        if self.arm:
            out[:3] = base_action[:3] + self.pos_scale * a[:3]
            R = Rotation.from_rotvec(self.rot_scale * a[3:6]) * Rotation.from_rotvec(base_action[3:6])
            out[3:6] = R.as_rotvec()
            a = a[6:]
        w = out[common.WRIST].copy()
        w[self.wrist_axes] += self.wrist_scale[self.wrist_axes] * a
        if perturbation is not None:
            w[self.wrist_axes] += perturbation
        out[common.WRIST] = np.clip(w, -1.0, 1.0)
        return out

    def decompose(self, base_action, executed):
        """Executed 12-dim (without the z compensation) -> the agent action that produces it, clipped to [-1, 1]"""
        parts = []
        if self.arm:
            parts.append((executed[:3] - base_action[:3]) / self.pos_scale)
            dR = Rotation.from_rotvec(executed[3:6]) * Rotation.from_rotvec(base_action[3:6]).inv()
            parts.append(dR.as_rotvec() / self.rot_scale)
        ax = self.wrist_axes
        parts.append((executed[common.WRIST][ax] - base_action[common.WRIST][ax]) / self.wrist_scale[ax])
        return np.clip(np.concatenate(parts), -1.0, 1.0).astype(np.float32)

    def _z_compensation(self, base_action, executed):
        """Arm target shift (base frame) that cancels the commanded plate travel of the z residual + perturbation"""
        dz = (executed[common.WRIST_Z] - base_action[common.WRIST_Z]) * self.wrist.config.z_range
        _, R0 = self.robot.composite_controller.get_controller_base_pose(controller_name="right")
        return R0.T @ self.wrist_R() @ np.array([0.0, 0.0, -dz])

    def _sampler_info(self, nominal_wrist):
        return dict(
            wrist_cmd_agent=nominal_wrist,          # base + residual, normalized (before the perturbation)
            wrist_cmd_base=self.base_action[common.WRIST].copy(),
            wrist_exec_prev=self.wrist_cmd.copy(),  # the command in force during the last step
            wrench=self.leg_wrench(),
            wrist_pose=self.wrist_pose() / self.wrist.action_scale,
        )

    # ---- gym ---------------------------------------------------------------------------------------------------------
    def reset(self, seed=None, options=None):
        if seed is not None:
            self.np_random = np.random.default_rng(seed)
        self._select_env(self.frictions[self.episodes % len(self.frictions)])
        self.episodes += 1
        env = self.env
        env.rng = np.random.default_rng(int(self.np_random.integers(1 << 31)))  # the door pose randomization
        self._raw = env.reset()
        self.base.reset(env, self.np_random)
        self._raw = env._get_observations(force_update=True)
        self._reset_state()
        self.base_action[common.WRIST] = self.base.wrist_cmd0()  # the wrist command the base settled with
        self.wrist_cmd = self.base_action[common.WRIST].copy()
        if self.skip_approach:
            for _ in range(MAX_SKIP_STEPS):
                if self.base.approached(env):
                    break
                self.base_action = self.base.act(env, self._raw, self.wrist_cmd)
                self.wrist_cmd = self.base_action[common.WRIST].copy()
                self._raw = env.step(self.base_action)[0]
                self.base.after_step(env)
        self.steps = 0
        if self.sampler is not None:
            self.sampler.reset(self, self.np_random)
        return self._obs(), {}

    def step(self, action):
        env = self.env
        base_action = self.base.act(env, self._raw, self.wrist_cmd)
        nominal = self.compose(base_action, action)
        pert = None
        if self.sampler is not None and self.explore:
            pert = np.asarray(self.sampler.sample(self, self._sampler_info(nominal[common.WRIST]), 1.0 / env.control_freq))
        executed = self.compose(base_action, action, pert)
        if self.wrist_rate > 0:
            step = self.wrist_rate / env.control_freq
            executed[common.WRIST] = self.wrist_cmd + np.clip(executed[common.WRIST] - self.wrist_cmd, -step, step)
        residual_executed = self.decompose(base_action, executed)
        sent = executed.copy()
        if self.z_compensation and 2 in self.wrist_axes:
            sent[:3] += self._z_compensation(base_action, executed)
        self.base_action = base_action
        self._physics_peak = 0.0
        self._raw = env.step(sent)[0]
        self.base.after_step(env)
        self.wrist_cmd = executed[common.WRIST].copy()
        self.steps += 1

        force = self._physics_peak if self.force_measure == "physics" else float(np.linalg.norm(self.gripper.flange_wrench(env.sim)[1][3:]))
        self.peak_force = max(self.peak_force, force)
        opened = bool(env._check_success())
        over = force > self.force_limit
        success = opened and not over
        reward = float(success) - self.force_penalty * (force / self.force_limit) ** 2
        terminated = opened or over
        base_failed = self.base.failed()
        truncated = (self.steps >= self.max_steps or (self.end_on_base_failure and base_failed)) and not terminated
        info = dict(
            success=success, opened=opened, over_force=over, max_force=force, peak_force=self.peak_force,
            executed_action=sent, residual_executed=residual_executed, base_action=base_action,
            perturbation=np.zeros(len(self.wrist_axes)) if pert is None else pert,
            wrist_z=float(executed[common.WRIST_Z]), phase=env_phase(env), base_phase=self.base.phase,
            base_failed=base_failed, friction=self.friction,
        )  # fmt: skip
        return self._obs(), reward, terminated, truncated, info

    def close(self):
        for env in self._envs.values():
            env.close()


try:
    gym.register(id="DoorResidual-v0", entry_point=DoorResidualEnv)
except Exception:  # already registered (re-import)
    pass


def main():
    """Plumbing check: a few random-residual steps per structure"""
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--structure", nargs="+", default=list(STRUCTURES), choices=list(STRUCTURES))
    p.add_argument("--base", default="oracle", choices=["oracle", "dp"])
    p.add_argument("--base_ckpt", default=None)
    p.add_argument("--steps", type=int, default=30)
    p.add_argument("--residual", type=float, default=0.3, help="uniform random residual amplitude")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    for s in args.structure:
        env = DoorResidualEnv(structure=s, base=args.base, base_ckpt=args.base_ckpt, seed=args.seed, privileged_obs=True, phase_obs=True)
        obs, _ = env.reset()
        rng = np.random.default_rng(args.seed)
        ret, info, clipped = 0.0, {}, 0
        for _ in range(args.steps):
            a = args.residual * rng.uniform(-1, 1, env.action_space.shape)
            obs, r, term, trunc, info = env.step(a)
            ret += r
            assert np.all(np.isfinite(obs)) and obs.shape == env.observation_space.shape
            clipped += not np.allclose(info["residual_executed"], a, atol=1e-6)  # the stroke clipped the wrist residual
            if term or trunc:
                break
        print(f"{s:13s} act {env.action_space.shape} obs {obs.shape} steps {env.steps} return {ret:+.3f} phase "
              f"{info['phase']}/{info['base_phase']} peak {info['peak_force']:.1f} N wrist z {info['wrist_z']:+.2f}, "
              f"{clipped} steps stroke-clipped")
        env.close()


if __name__ == "__main__":
    main()
