"""
KeyInsertion as a gym environment for SERL (https://github.com/rail-berkeley/serl): a state-based, sparse-reward,
10 Hz insertion of the rigid key into a fixed lock with a RAM-style snap latch, starting close to the keyhole.

    import key_insertion.serl_env                          # registers the ids below
    env = gym.make("KeyInsertionWrist-v0", mode="variable")

Modes (the three policies to compare):
    stiff     wrist fully contracted the whole episode (z = -1: 100 kN/m legs, ~400 kN/m axially)
    soft      wrist fully extended the whole episode (z = +1: 300 N/m legs, ~1.5 kN/m axially, 140 N/m laterally)
    variable  the policy also sets the wrist's z (= its stiffness, since leg stiffness follows commanded leg length)

Action (Box in [-1, 1]): (dx, dy, dz, dyaw) of the key tip's target, up to POS_SCALE / YAW_SCALE per step, in the lock
frame; "variable" adds a 5th entry: the wrist's z command, absolute (-1 stiff ... +1 soft), followed at most
STIFFNESS_RATE per step. The arm tracks the tip target through a diagonal Cartesian impedance (20 kN/m, 300 N*m/rad),
interpolated over the step's physics, with the tip's offset from the arm's tcp recomputed for the wrist's z, so
changing stiffness in free space does not move the key. Under load, a stiffness change also moves the target so the
wrist's spring force stays continuous (see _compensate): at fixed deflection, going from 1.5 to 400 kN/m would
otherwise multiply the force ~260 times.

The target is clipped to a box around the keyhole (SERL's franka_env does the same with abs_pose_limit_*): at most
BOX_BELOW below the seated depth. That box is what makes full compliance costly: the latch's 40 N breakaway needs ~26 mm
of soft-wrist compression, more than the box allows, while the stiff wrist, unless the key is lined up well, wedges it
in and needs more than FORCE_LIMIT.

Observation (flat Box, 22): key tip position in the lock frame (/1 cm), the tip's orientation error from the slot
(rotation vector, /0.1 rad), tip velocity (/5 cm/s), the wrist's F/T at the flange as the wrist itself measures it
(moment /2 N*m, force /20 N; its legs' spring forces from their encoders, WristGripperModel.leg_wrench), wrist
deflection from its commanded pose (6, /5 mm, /0.05 rad; leg encoders), wrist z command (1). Idealized sensing (no
noise).

Reward: 1 when the key is seated (latch passed, plunger >= 90% of its travel) without the flange force passing
force_limit, minus a force penalty on every step: force_penalty * (F / force_limit)^2, F the step's peak flange force
(the force the wrist puts on the arm, MuJoCo's own: flange_wrench, sampled at every arm / wrist command update, 50 Hz).
Seating ends the episode; so does passing force_limit (no success reward, also on the step that seats the key:
info["seated"] still says it went in). Truncated after max_steps.
"""

import os
import sys

import numpy as np
from scipy.spatial.transform import Rotation

try:
    import gym
    from gym import spaces
except ImportError:  # the robosuite venv has gymnasium only; the API is the same
    import gymnasium as gym
    from gymnasium import spaces

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from key_insertion.oracle import KeyOracle, make_env, make_T  # noqa: E402

# The lock: keyhole fixed at the table's center, with a RAM-style latch at the bottom of the slot. The plunger is met
# 20 mm down; its force climbs to a 40 N breakaway over 2.5 mm, snaps down to 8 N, then rises to 15 N at the seat
# (10 mm of travel, 30 mm down).
LOCK = dict(
    lock_x_range=(0, 0),
    lock_y_range=(0, 0),
    lock_yaw_range=(0, 0),
    plunger_travel=0.010,
    latch_profile=[(0.0, 5.0), (0.0025, 40.0), (0.0035, 8.0), (0.010, 15.0)],
    plunger_stop_timeconst=0.01,  # after the snap the key reaches the seat fast; resolve that impact over 5 timesteps
)
MODES = {"stiff": -1.0, "soft": 1.0, "variable": 0.0}  # wrist z at the start of an episode
POS_SCALE = 0.002  # m per step
YAW_SCALE = np.radians(1.0)  # rad per step
# Wrist z per step: 2 mm of platform travel per step, 20 mm/s (full range in 2 s). Under contact, a z change moves the
# stiff platform at once while the arm (~70 ms to follow) compensates, so faster changes bounce the key
STIFFNESS_RATE = 0.1
BOX_XY = 0.010  # target within this of the keyhole's axis (m)
BOX_ABOVE = 0.030  # ... at most this far above the lock's top face (m)
BOX_BELOW = 0.005  # ... at most this far below the seated depth (m)
BOX_YAW = np.radians(15.0)
FORCE_LIMIT = 80.0  # N at the flange (2x the latch's breakaway): ends the episode as a failure, like breaking the part
# Reward per step: -FORCE_PENALTY * (peak flange force / FORCE_LIMIT)^2. A step pushing the latch's 40 N costs 0.06, one
# at the limit 0.25, against 1 for seating: the push the latch needs stays worth it, harder ones cost more and more
FORCE_PENALTY = 0.25
COMPENSATION_DEADBAND = 1.0  # N per axis of flange force ignored by the stiffness-change compensation
OBS_SCALE = np.concatenate([[0.01] * 3, [0.1] * 3, [0.05] * 3, [2.0] * 3, [20.0] * 3, [0.005] * 3, [0.05] * 3, [1.0]])


class KeyInsertionSERL(gym.Env):
    """
    Args:
        mode (str): "stiff", "soft" or "variable" (see the module docstring)
        control_hz (float): policy rate
        substeps (int): arm / wrist command updates per policy step (the target is interpolated across them)
        max_steps (int): episode length in policy steps
        start_xy (float): the key tip starts uniformly within +-start_xy of the keyhole's axis, in x and y (m)
        start_height (2-tuple): ... this far above the lock's top face (m)
        start_yaw (float): ... with the blade within +-start_yaw of the slot (rad)
        force_limit (float): flange force (N) that ends the episode as a failure
        force_penalty (float): weight of the per-step force penalty (see the module docstring); 0: success reward only
        seed (int or None): for the start randomization
        render_mode (None or "rgb_array"): render() returns the "lockview" close-up
    """

    metadata = {"render_modes": ["rgb_array"], "render_fps": 10}

    def __init__(
        self,
        mode="variable",
        control_hz=10.0,
        substeps=5,
        max_steps=100,
        start_xy=0.005,
        start_height=(0.010, 0.020),
        start_yaw=np.radians(5.0),
        force_limit=FORCE_LIMIT,
        force_penalty=FORCE_PENALTY,
        seed=None,
        render_mode=None,
        **env_kwargs,
    ):
        assert mode in MODES, f"Unknown mode {mode}"
        self.mode, self.substeps, self.max_steps = mode, substeps, max_steps
        self.start_xy, self.start_height, self.start_yaw = start_xy, start_height, start_yaw
        self.force_limit, self.force_penalty = force_limit, force_penalty
        self.render_mode = render_mode
        self.env = make_env(
            control_freq=control_hz * substeps,
            arm="impedance",
            offscreen=render_mode is not None,
            initialization_noise=None,
            **{**LOCK, **env_kwargs},
        )
        self.robot = self.env.robots[0]
        self.gripper = self.robot.gripper["right"]
        self.wrist = self.gripper.wrist
        self.np_random = np.random.default_rng(seed)
        self.on_substep = None  # if set, called after every arm / wrist command update (e.g. to log at 50 Hz)
        # Where the policy believes the keyhole is, relative to where it is: (dx, dy, dyaw) in the lock frame (m, m, rad).
        # The tip's pose and velocity are observed relative to that believed keyhole, so a policy aims off by this much
        # all the way into contact, as with a mis-located lock (forces and the wrist's deflection stay true). 0: none
        self.aim_error = np.zeros(3)
        self._physics_peak = 0.0  # peak flange force over every physics step of the current policy step (_watch_physics)

        n = 5 if mode == "variable" else 4
        self.action_space = spaces.Box(-1.0, 1.0, (n,), dtype=np.float32)
        self.observation_space = spaces.Box(-np.inf, np.inf, (len(OBS_SCALE),), dtype=np.float32)
        self._calibrate()

    # ---- setup -------------------------------------------------------------------------------------------------------

    def _calibrate(self):
        """
        Once: brings the arm to just above the keyhole (the scripted oracle's approach) and keeps that joint
        configuration as the reset pose; measures the key tip's pose relative to the arm's tcp, in free space, with the
        wrist fully contracted and fully extended (the tip moves along the wrist's z in between, linearly in z).
        """
        env = self.env
        env.reset()
        oracle = KeyOracle(env, wrist_z=0.0, hover=np.mean(self.start_height))
        oracle.reset()
        while oracle.phase < KeyOracle.INSERT:
            env.step(oracle.action(oracle.next_target()))
        self.R_tip_in_lock = oracle.R_tip_in_lock
        self.robot.init_qpos = env.sim.data.qpos[self.robot._ref_joint_pos_indexes].copy()
        self.T_tip_tcp = {}
        for z in (-1.0, 1.0):
            env.reset()
            self._settle(z, 1.0)
            self.T_tip_tcp[z] = np.linalg.inv(oracle.site_T(self.gripper.important_sites["grip_site"])) @ oracle.site_T(
                self.gripper.important_sites["tcp"]
            )
        self._oracle = oracle  # for its frame helpers

    def _settle(self, z, seconds, tcp_target=None):
        if tcp_target is None:
            tcp_target = self._tcp_hold()
        for _ in range(int(round(seconds * self.env.control_freq))):
            self.env.step(self._robot_action(tcp_target, z))

    def _tcp_hold(self):
        """Base-frame [pos, rotvec] target that holds the arm's tcp where it is"""
        d = self.env.sim.data
        site = self.gripper.important_sites["tcp"]
        p0, R0 = self.robot.composite_controller.get_controller_base_pose(controller_name="right")
        return np.concatenate([R0.T @ (d.get_site_xpos(site) - p0), Rotation.from_matrix(R0.T @ d.get_site_xmat(site)).as_rotvec()])

    def _robot_action(self, tcp_target, z):
        return self.robot.create_action_vector({"right": tcp_target, "right_gripper": np.array([0, 0, z, 0, 0, 0.0])})

    # ---- frames ------------------------------------------------------------------------------------------------------

    def tip_T_tcp(self, z):
        """Key tip -> arm tcp in free space at wrist z command @z (linear between the two calibrated ends)"""
        a, b = self.T_tip_tcp[-1.0], self.T_tip_tcp[1.0]
        T = a.copy()
        T[:3, 3] = a[:3, 3] + (z + 1) / 2 * (b[:3, 3] - a[:3, 3])
        return T

    def tip_target_T(self):
        """World pose of the tip target"""
        p, R = self.env.lock_pose()
        R_tip = R @ Rotation.from_euler("z", self.target[3]).as_matrix() @ self.R_tip_in_lock
        return make_T(R_tip, p + R @ self.target[:3])

    def tcp_target(self, z):
        T_tcp = self.tip_target_T() @ self.tip_T_tcp(z)
        p0, R0 = self.robot.composite_controller.get_controller_base_pose(controller_name="right")
        return np.concatenate([R0.T @ (T_tcp[:3, 3] - p0), Rotation.from_matrix(R0.T @ T_tcp[:3, :3]).as_rotvec()])

    def clip_target(self, target):
        lo = [-BOX_XY, -BOX_XY, -(self.env.slot_depth + BOX_BELOW), -BOX_YAW]
        hi = [BOX_XY, BOX_XY, BOX_ABOVE, BOX_YAW]
        return np.clip(target, lo, hi)

    # ---- state -------------------------------------------------------------------------------------------------------

    def wrist_deflection(self):
        """Top plate pose minus its commanded pose (x, y, z, roll, pitch, yaw; wrist frame), from the leg encoders"""
        names = [f"wrist_top_{a}" for a in ("x", "y", "z", "rx", "ry", "rz")]
        d = self.env.sim.data
        pose = np.array([d.get_joint_qpos(self.gripper.correct_naming(n)) for n in names])
        return pose - self.wrist.last_commanded_pose

    def wrist_R(self):
        """Wrist base frame (world rotation), the frame of the deflection"""
        b = self.env.sim.model._model.body(self.gripper.worldbody[0].get("name")).id
        return self.env.sim.data._data.xmat[b].reshape(3, 3)

    def arm_wrench(self):
        """
        The wrench at the flange as the arm estimates it from its joint torques, the way a torque-sensing arm (e.g. a
        Franka's O_F_ext_hat_K) does: tau_ext = tau_model - tau_joints, with tau_model the arm's rigid-body dynamics
        (gravity, Coriolis, inertia at its measured q, dq, ddq) carrying the wrist and tool as a rigid payload (wrist
        joints at rest), mapped to the flange through the arm's Jacobian, F = (J^T)^+ tau_ext. Same convention as
        flange_wrench's external wrench: (m, f) on the arm at the flange, flange frame, payload weight compensated. What
        a rigid-payload model can't know (the wrist's internal motion) ends up in it. Idealized: exact model (with the
        rotors' armature and joint damping), torques, and accelerations, except the joints' Coulomb friction (0.1 N*m
        each, MuJoCo frictionloss), which a model can't predict at rest either: it reads as a few tenths of N / N*m.
        """
        m, d = self.env.sim.model._model, self.env.sim.data._data
        arm = np.asarray(self.robot._ref_joint_vel_indexes)
        if getattr(self, "_scratch", None) is None:
            import mujoco

            self._mujoco = mujoco
            self._scratch = mujoco.MjData(m)
            self._flange = m.body(self.gripper.worldbody[0].get("name")).id
            self._tau_model = np.zeros(m.nv)
        mj, s = self._mujoco, self._scratch
        # Rigid payload: the arm's own velocities and accelerations only (the kinematics are the last step's, as for
        # the actuator, passive and constraint forces, so everything is from the same instant)
        mj.mj_copyData(s, m, d)
        s.qvel[:] = 0.0
        s.qvel[arm] = d.qvel[arm]
        s.qacc[:] = 0.0
        s.qacc[arm] = d.qacc[arm]
        mj.mj_comVel(m, s)
        mj.mj_rne(m, s, 1, self._tau_model)
        tau_model = self._tau_model[arm] + m.dof_armature[arm] * d.qacc[arm]  # mj_rne leaves out the rotors' inertia
        # M ddq + bias = tau_actuators + tau_passive + J^T F_ext: what the joint torques don't explain is the flange's load
        tau_ext = tau_model - d.qfrc_actuator[arm] - d.qfrc_passive[arm]
        jacp, jacr = np.zeros((3, m.nv)), np.zeros((3, m.nv))
        mj.mj_jacBody(m, d, jacp, jacr, self._flange)
        J = np.vstack([jacr[:, arm], jacp[:, arm]])  # (m, f) at the flange's origin, world axes
        F = np.linalg.lstsq(J.T, tau_ext, rcond=None)[0]  # the wrench applied to the arm's flange
        R = d.xmat[self._flange].reshape(3, 3)
        return np.concatenate([R.T @ F[:3], R.T @ F[3:]])

    def _obs(self):
        env = self.env
        p, R = env.key_tip_in_lock()
        _, R_lock = env.lock_pose()
        vel = R_lock.T @ env.sim.data.get_site_xvelp(env.sim.model.site_id2name(env.tip_site_id))
        if np.any(self.aim_error):  # observed in the believed keyhole's frame
            Rz = Rotation.from_euler("z", self.aim_error[2]).as_matrix()
            p = Rz.T @ (p - np.array([self.aim_error[0], self.aim_error[1], 0.0]))
            R, vel = Rz.T @ R, Rz.T @ vel
        rot_err = Rotation.from_matrix(self.R_tip_in_lock.T @ R).as_rotvec()  # tip frame, vs the slot-aligned pose
        wrench = self.gripper.leg_wrench(env.sim)[1]
        obs = np.concatenate([p, rot_err, vel, wrench, self.wrist_deflection(), [self.z]])
        return (obs / OBS_SCALE).astype(np.float32)

    # ---- gym ---------------------------------------------------------------------------------------------------------

    def reset(self, seed=None, options=None):
        if seed is not None:
            self.np_random = np.random.default_rng(seed)
        self.env.reset()
        self.z = MODES[self.mode]
        rng = self.np_random
        self.target = np.array(
            [
                rng.uniform(-self.start_xy, self.start_xy),
                rng.uniform(-self.start_xy, self.start_xy),
                rng.uniform(*self.start_height),
                rng.uniform(-self.start_yaw, self.start_yaw),
            ]
        )
        # Drive the tip to its start pose and let the arm and wrist settle there
        self._settle(self.z, 0.6, self.tcp_target(self.z))
        self.steps = 0
        return self._obs(), {}

    def step(self, action):
        action = np.clip(np.asarray(action, dtype=float), -1.0, 1.0)
        start = self.target.copy()
        goal = self.clip_target(start + np.concatenate([action[:3] * POS_SCALE, [action[3] * YAW_SCALE]]))
        z0 = self.z
        z1 = z0 + np.clip(action[4] - z0, -STIFFNESS_RATE, STIFFNESS_RATE) if self.mode == "variable" else z0
        max_force = 0.0
        self._watch_physics()
        self._physics_peak = 0.0
        for k in range(1, self.substeps + 1):
            s = k / self.substeps
            z = z0 + s * (z1 - z0)
            if z != self.z:
                shift = self._compensate(self.z, z)
                start[:3] += shift
                goal[:3] += shift
                goal = self.clip_target(goal)
            self.z = z
            self.target = start + s * (goal - start)
            self.env.step(self._robot_action(self.tcp_target(z), z))
            max_force = max(max_force, np.linalg.norm(self.gripper.flange_wrench(self.env.sim)[1][3:]))
            if self.on_substep is not None:
                self.on_substep()
        self.target = goal
        self.steps += 1
        seated = bool(self.env._check_success())
        over = max_force > self.force_limit
        success = seated and not over  # seating by passing the force limit counts as breaking the part
        reward = float(success) - self.force_penalty * (max_force / self.force_limit) ** 2
        # max_force (what the reward and the force limit use) is sampled at the 50 Hz updates; max_force_physics is the
        # peak over every physics step, which also catches spikes shorter than 20 ms
        info = dict(succeed=success, seated=seated, max_force=max_force, max_force_physics=self._physics_peak, over_force=over,
                    plunger=self.env.plunger_depression(), wrist_z=self.z)  # fmt: skip
        return self._obs(), reward, seated or over, self.steps >= self.max_steps, info

    def _watch_physics(self):
        """Wraps the sim's physics step (once per sim) to track the peak flange force over every physics step"""
        sim = self.env.sim
        if getattr(sim, "_peak_watched", False):
            return
        name = "step2" if self.env.lite_physics else "step"  # robosuite's base env steps physics with one of them
        physics_step = getattr(sim, name)

        def stepped(*args, **kwargs):
            physics_step(*args, **kwargs)
            self._physics_peak = max(self._physics_peak, np.linalg.norm(self.gripper.flange_wrench(sim)[1][3:]))

        setattr(sim, name, stepped)
        sim._peak_watched = True

    def _compensate(self, z_old, z_new):
        """
        Lock-frame shift of the tip target that keeps the wrist's spring force continuous when its z command (and so
        its stiffness) goes from @z_old to @z_new. The wrist's own F/T (its legs, leg_wrench) reads the force f it carries; statically its
        legs are deflected by K_old^-1 f, and with the key held by contact, moving the arm by (K_old^-1 - K_new^-1) f
        leaves them deflected by K_new^-1 f, i.e. carrying the same f. The first COMPENSATION_DEADBAND newtons of each
        axis are ignored: in free space f isn't exactly 0 (the wrist's own parts accelerate when z changes), and the
        soft wrist's lateral compliance (~7 mm/N) would turn those few tenths of a newton into millimetres of drift.
        The tip's free-space offset from the tcp is handled by tip_T_tcp(z). (Not the leg encoders' deflection: that
        also holds the soft wrist's dynamic lag whenever it moves, which isn't a force to keep.)
        """
        def K(z):
            return self.wrist.platform_stiffness(np.array([0, 0, z * self.wrist.config.z_range, 0, 0, 0]))[:3, :3]

        f = self.gripper.leg_wrench(self.env.sim)[1][3:]  # wrist (flange) frame
        f = np.sign(f) * np.maximum(np.abs(f) - COMPENSATION_DEADBAND, 0.0)
        _, R_lock = self.env.lock_pose()
        return R_lock.T @ self.wrist_R() @ (np.linalg.solve(K(z_old), f) - np.linalg.solve(K(z_new), f))

    def render(self):
        return self.env.sim.render(camera_name="lockview", width=480, height=360)[::-1]

    def close(self):
        self.env.close()


class ScriptedPolicy:
    """
    The demonstrator: a simple oracle that knows the keyhole (it is fixed) and acts in the env's action space. It lines
    the tip target up with the keyhole (x, y, yaw), descends at ~10 mm/s once lined up, and pushes until seated, where
    the box stops it. In "variable" mode it keeps the wrist soft while the key enters (compliance through the chamfer
    lines it up with the slot), then, once the blade is stiffen_depth down the slot, re-aims at where the slot put the
    tip (its pose from the leg encoders), stiffens, and waits at wait_depth (above the plunger) until the wrist is
    stiff before pushing through the latch. Gaussian action noise on the motion (not the stiffness) makes the demos
    cover more than one path. @aim_error (dx, dy, dyaw; m, m, rad, lock frame) makes it aim off the keyhole, as an
    imprecise policy would.
    """

    def __init__(self, env, noise=0.1, stiffen_depth=0.004, wait_depth=0.016, descend=0.5, aim_error=(0, 0, 0), seed=None):
        self.env, self.noise, self.descend = env, noise, descend
        self.aim = np.asarray(aim_error, dtype=float)
        self.stiffen_depth, self.wait_depth = stiffen_depth, wait_depth
        self.rng = np.random.default_rng(seed)

    def __call__(self, obs):
        env = self.env.unwrapped if hasattr(self.env, "unwrapped") else self.env
        if env.steps == 0:
            self.goal = self.aim.copy()  # where it lines the tip target up: (x, y, yaw), lock frame
        if env.mode == "variable" and env.env.insertion_depth() > self.stiffen_depth and env.z > 0.95:
            # Still soft and in the slot: aim where the slot holds the tip
            p, R = env.env.key_tip_in_lock()
            M = R @ env.R_tip_in_lock.T
            self.goal = np.array([p[0], p[1], np.arctan2(M[1, 0], M[0, 0])])
        t = env.target - np.concatenate([self.goal[:2], [0], self.goal[2:]])
        a = np.zeros(env.action_space.shape)
        a[:2] = np.clip(-t[:2] / POS_SCALE, -1, 1)
        a[3] = np.clip(-t[3] / YAW_SCALE, -1, 1)
        lined_up = np.all(np.abs(t[:2]) < 0.0005) and abs(t[3]) < np.radians(0.5)
        a[2] = -self.descend if lined_up else 0.0
        a[:4] += self.rng.normal(0, self.noise, 4)
        if env.mode == "variable":
            depth = env.env.insertion_depth()
            a[4] = -1.0 if depth > self.stiffen_depth else 1.0
            if depth > self.wait_depth and env.z > -0.95:
                a[2] = 0.0
        return np.clip(a, -1, 1).astype(np.float32)


try:
    gym.register(id="KeyInsertionWrist-v0", entry_point=KeyInsertionSERL)
except Exception:  # already registered (re-import)
    pass
