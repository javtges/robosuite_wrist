import numpy as np
from scipy.spatial.transform import Rotation

from robosuite.controllers.parts.arm.osc import OperationalSpaceController
from robosuite.utils.control_utils import nullspace_torques, opspace_matrices
from robosuite.utils.pose_trajectory_interpolator import PoseTrajectoryInterpolator, pose_distance


def damped_pinv(m, lam):
    """
    Damped pseudo-inverse m+ = V diag(s / (s^2 + lam^2)) U^T, with m = U diag(s) V^T. For lam = 0 it is the plain
    pseudo-inverse; lam > 0 keeps m+ bounded near singularities at the price of a small residual m m+ != I.
    """
    U, s, Vt = np.linalg.svd(m, full_matrices=False)
    return Vt.T @ np.diag(s / (s**2 + lam**2)) @ U.T


def projector_nullspace_torques(J, q, dq, q_null, stiffness, joint1_stiffness, lam):
    """
    The nullspace torque of serl_franka_controllers / franka_streaming_impedance_controller (nullspaceTorque in
    cartesian_impedance_law.cpp):

        N   = I - J^T (J^T)+          (J^T)+ is the damped pseudo-inverse of J^T
        tau = N (k (q_null - q) - 2 sqrt(k) dq)

    Joint 1 gets its own tier: its position error is scaled by joint1_stiffness and its velocity by
    2 sqrt(joint1_stiffness), so the base rotation is held more firmly than the elbow.
    """
    qe = q_null - q
    dqe = np.array(dq, dtype=float)
    qe[0] *= joint1_stiffness
    dqe[0] *= 2.0 * np.sqrt(joint1_stiffness)
    N = np.eye(J.shape[1]) - J.T @ damped_pinv(J.T, lam)
    return N @ (stiffness * qe - 2.0 * np.sqrt(stiffness) * dqe)


def extrapolate_chunk(target, prev_target, n):
    """
    HSA-UMI's teleop chunk (hsa_ros2/teleop_math.extrapolate_chunk) for 7-vector poses: waypoint k is the target moved
    by k times the last tick's displacement (position linearly, orientation by composing the last tick's rotation k
    times). With no previous target the chunk is a plain hold. A chunk that merely repeats the target would hand the
    monotone interpolant a zero tangent, so the reference would decelerate to a stop every tick.
    """
    target = np.asarray(target, dtype=float)
    if n <= 1 or prev_target is None:
        return [target.copy() for _ in range(max(1, n))]
    prev = np.asarray(prev_target, dtype=float)
    delta_pos = target[:3] - prev[:3]
    rot_t = Rotation.from_quat(target[3:])
    delta_rot = rot_t * Rotation.from_quat(prev[3:]).inv()
    out = []
    for k in range(n):
        pose = np.empty(7)
        pose[:3] = target[:3] + k * delta_pos
        pose[3:] = ((delta_rot**k) * rot_t).as_quat() if k else target[3:]
        out.append(pose)
    return out


class CartesianImpedanceController(OperationalSpaceController):
    """
    Cartesian impedance control with a diagonal stiffness, like libfranka's cartesian_impedance_control example:

        F = K (x_goal - x) - D xdot,    tau = J^T F + gravity / coriolis compensation + nullspace torques

    Unlike OSC (F = Lambda (kp e - kd xdot)), the stiffness seen at the reference site is exactly diag(kp) (N/m for
    position, N*m/rad for orientation) in every configuration, with no coupling between axes through the arm's inertia
    Lambda. kp is the stiffness itself, so it is typically 1e3 - 1e4 rather than OSC's ~150. Only fixed impedance and
    position + orientation control are supported. Goal handling, input types and interpolation are OSC's.

    Two flavours, chosen by the config keys below (all optional; the defaults give the original sim-only controller):

    - Inertia-shaped damping (default, `damping` unset): D = damping_ratio (Lambda^1/2 K^1/2 + K^1/2 Lambda^1/2),
      critical for K given Lambda, so the closed loop doesn't ring along any direction.
    - The hardware law of HSA-UMI's franka_streaming_impedance_controller (a port of serl_franka_controllers):
        damping (6 or [trans, rot]):   constant diagonal D (N s/m, N m s/rad) instead of the Lambda-based one
        translational_clip (m), rotational_clip (rad):  the pose error is clipped before K, so the commanded force
                                       is bounded by K * clip (2000 N/m * 0.01 m = 20 N on the robot)
        ki (6 or [trans, rot]):        integral gain on the clipped error; the integral is accumulated per 1 ms as on
                                       the robot (scaled by the sim timestep) and clamped to 0.1 m / 0.3 rad
        nullspace:                     "inertial" (robosuite's M-weighted projector, kp 10 toward initial_joint) or
                                       "projector" (the robot's: I - J^T (J^T)+ with a damped pseudo-inverse, lambda
                                       `nullspace_lambda`, gains `nullspace_stiffness`, `joint1_nullspace_stiffness`)
        torque_rate_limit (N m / s):   per-joint slew limit on the task + nullspace torques (the robot's 1 N m per 1 ms
                                       step = 1000). Gravity / Coriolis compensation is left out of the limit, as
                                       libfranka adds gravity after the controller's torque command.

    The reference the spring is anchored to (`reference`):
        "hold" (default):  the goal of the latest action, constant for the whole policy step, as every robosuite
                           controller does.
        "chunks":          the robot's streaming reference. Each action becomes a chunk of `chunk_len` absolutely-timed
                           waypoints, due at now + (1 + k) / policy_freq: the target, then the target continued along
                           its last step (HSA-UMI's teleop extrapolation), spliced into a PoseTrajectoryInterpolator
                           (monotone cubic Hermite position, slerp orientation) that is evaluated at every sim step.
                           `max_pos_speed` / `max_rot_speed` stretch a segment the reference can't reach in time. A
                           target that jumps more than `chunk_hold_pos_step` / `chunk_hold_rot_step` in one step is a
                           re-seed, and its chunk is a hold (HSA-UMI's teleop does the same). `schedule_chunk()` lets a
                           policy supply its own waypoints instead for the next action. The interpolator and the slew
                           and integral states are controller state, not sim state: `controller_state()` /
                           `restore_controller_state()` save and restore them for exact replays.
    """

    def __init__(
        self,
        *args,
        damping=None,
        translational_clip=None,
        rotational_clip=None,
        ki=0.0,
        nullspace="inertial",
        nullspace_stiffness=0.2,
        joint1_nullspace_stiffness=100.0,
        nullspace_lambda=0.2,
        torque_rate_limit=None,
        reference="hold",
        chunk_len=6,
        max_pos_speed=np.inf,
        max_rot_speed=np.inf,
        chunk_hold_pos_step=0.05,
        chunk_hold_rot_step=0.5,
        **kwargs,
    ):
        assert reference in ("hold", "chunks"), f"unknown reference {reference!r}"
        assert nullspace in ("inertial", "projector"), f"unknown nullspace {nullspace!r}"
        # Before super().__init__: it may call reset_goal(), which this class overrides
        self.reference = reference
        self.chunk_len = int(chunk_len)
        self.max_pos_speed = float(max_pos_speed) if max_pos_speed is not None else np.inf
        self.max_rot_speed = float(max_rot_speed) if max_rot_speed is not None else np.inf
        self.chunk_hold_pos_step = float(chunk_hold_pos_step)
        self.chunk_hold_rot_step = float(chunk_hold_rot_step)
        self.interp = None
        self.last_waypoint_time = -np.inf
        self.prev_target = None
        self._pending_chunk = None
        self.integral_error = np.zeros(6)
        self.tau_prev = None
        super().__init__(*args, **kwargs)
        assert self.impedance_mode == "fixed", "CartesianImpedanceController only supports fixed impedance"
        assert self.use_ori, "CartesianImpedanceController controls position and orientation"
        if reference == "chunks":
            assert self.interpolator_pos is None and self.interpolator_ori is None, "chunks replace robosuite's interpolator"
        # OSC turns damping_ratio into kd = 2 sqrt(kp) damping_ratio; keep the ratio itself
        self.damping_ratio = np.mean(self.kd / (2 * np.sqrt(self.kp)))
        self.damping = None if damping is None else self._six(damping)
        self.error_clip = None
        if translational_clip is not None or rotational_clip is not None:
            t = np.inf if translational_clip is None else abs(float(translational_clip))
            r = np.inf if rotational_clip is None else abs(float(rotational_clip))
            self.error_clip = np.array([t] * 3 + [r] * 3)
        self.ki = self._six(ki)
        self.nullspace = nullspace
        self.nullspace_stiffness = float(nullspace_stiffness)
        self.joint1_nullspace_stiffness = float(joint1_nullspace_stiffness)
        self.nullspace_lambda = float(nullspace_lambda)
        self.torque_rate_limit = None if torque_rate_limit is None else float(torque_rate_limit)
        if self.reference == "chunks" and self.interp is None:
            self._seed_reference()

    @staticmethod
    def _six(v):
        """6-vector from a scalar, a [translation, rotation] pair, or 6 values"""
        v = np.atleast_1d(np.asarray(v, dtype=float))
        if v.size == 1:
            return np.repeat(v, 6)
        if v.size == 2:
            return np.array([v[0]] * 3 + [v[1]] * 3)
        assert v.size == 6, f"expected 1, 2 or 6 values, got {v.size}"
        return v

    # ---- the reference (chunks) --------------------------------------------------------------------------------------

    @property
    def now(self):
        """The controller's clock: sim time (s)"""
        return float(self.sim.data.time)

    def _goal_pose7(self):
        """The latest goal (goal_pos, goal_ori; in input_ref_frame) as [pos, quat xyzw]"""
        return np.concatenate([np.asarray(self.goal_pos, dtype=float), Rotation.from_matrix(self.goal_ori).as_quat()])

    def _measured_pose7(self):
        """The reference site's pose in input_ref_frame"""
        pos, R = np.asarray(self.ref_pos, dtype=float), np.asarray(self.ref_ori_mat, dtype=float)
        if self.input_ref_frame == "base" and self.origin_pos is not None:
            pos = self.origin_ori.T @ (pos - self.origin_pos)
            R = self.origin_ori.T @ R
        return np.concatenate([pos, Rotation.from_matrix(R).as_quat()])

    def _seed_reference(self):
        """A constant reference at the arm's current pose, as the robot does at activation"""
        self.interp = None
        if getattr(self, "ref_pos", None) is None or getattr(self, "sim", None) is None:
            return
        if self.input_ref_frame == "base" and getattr(self, "origin_pos", None) is None:
            return  # the robot hasn't set its base frame yet (construction); the first set_goal seeds instead
        self.interp = PoseTrajectoryInterpolator.constant(self.now, self._measured_pose7())
        self.last_waypoint_time = -np.inf
        self.prev_target = None
        self._pending_chunk = None

    def schedule_chunk(self, poses, times_from_now):
        """
        Waypoints for the next action: @poses (n x 7, [pos, quat xyzw] in input_ref_frame, or n x 6 [pos, rotvec])
        due at now + @times_from_now (s). Used instead of the extrapolated chunk by the next set_goal(), whose action
        should be the chunk's first waypoint (that is what gets recorded).
        """
        poses = np.atleast_2d(np.asarray(poses, dtype=float))
        if poses.shape[1] == 6:
            poses = np.concatenate([poses[:, :3], Rotation.from_rotvec(poses[:, 3:]).as_quat()], axis=1)
        self._pending_chunk = (poses, np.asarray(times_from_now, dtype=float))

    def reference_pose(self, t=None):
        """The reference the spring is anchored to at time @t (default now), as [pos, quat] in input_ref_frame"""
        if self.reference == "chunks" and self.interp is not None:
            return self.interp(self.now if t is None else t)
        return self._goal_pose7()

    def set_goal(self, action):
        super().set_goal(action)
        if self.reference != "chunks":
            return
        now = self.now
        if self.interp is None or now < self.interp.first_time - 1e-9:  # a reset moved the clock backwards
            self._seed_reference()
        target = self._goal_pose7()
        dt = 1.0 / self.control_freq
        if self._pending_chunk is not None:
            poses, offsets = self._pending_chunk
            self._pending_chunk = None
            waypoints = [(now + float(o), p) for o, p in zip(offsets, poses)]
        else:
            prev = self.prev_target
            if prev is not None:
                pos_step, rot_step = pose_distance(prev, target)
                if pos_step > self.chunk_hold_pos_step or rot_step > self.chunk_hold_rot_step:
                    prev = None  # a re-seeded target has no velocity: the chunk is a hold
            chunk = extrapolate_chunk(target, prev, self.chunk_len)
            waypoints = [(now + (1 + k) * dt, p) for k, p in enumerate(chunk)]
        for t, p in waypoints:
            if t <= now:
                continue
            self.interp = self.interp.schedule_waypoint(
                p, t, now, self.last_waypoint_time, self.max_pos_speed, self.max_rot_speed
            )
            self.last_waypoint_time = t
        self.prev_target = target

    def compute_errors(self):
        if self.reference != "chunks" or self.interp is None:
            return super().compute_errors()
        p = self.interp(self.now)
        saved = self.goal_pos, self.goal_ori
        self.goal_pos, self.goal_ori = p[:3], Rotation.from_quat(p[3:]).as_matrix()
        try:
            return super().compute_errors()
        finally:
            self.goal_pos, self.goal_ori = saved

    # ---- controller state (not sim state) ----------------------------------------------------------------------------

    def controller_state(self):
        """JSON-friendly snapshot of everything a replay must restore besides the sim state and initial_joint"""
        d = {
            "integral_error": self.integral_error.tolist(),
            "tau_prev": None if self.tau_prev is None else self.tau_prev.tolist(),
        }
        if self.reference == "chunks" and self.interp is not None:
            now = self.now
            d.update(
                times_from_now=[t - now for t in self.interp.times],
                poses=self.interp.poses.tolist(),
                last_waypoint_from_now=None if not np.isfinite(self.last_waypoint_time) else self.last_waypoint_time - now,
                prev_target=None if self.prev_target is None else self.prev_target.tolist(),
            )
        return d

    def restore_controller_state(self, d):
        self.integral_error = np.asarray(d.get("integral_error", np.zeros(6)), dtype=float)
        self.tau_prev = None if d.get("tau_prev") is None else np.asarray(d["tau_prev"], dtype=float)
        if self.reference == "chunks" and "times_from_now" in d:
            now = self.now
            self.interp = PoseTrajectoryInterpolator([now + t for t in d["times_from_now"]], d["poses"])
            lw = d.get("last_waypoint_from_now")
            self.last_waypoint_time = -np.inf if lw is None else now + lw
            self.prev_target = None if d.get("prev_target") is None else np.asarray(d["prev_target"], dtype=float)
            self._pending_chunk = None

    # ---- the law -----------------------------------------------------------------------------------------------------

    def damping_matrix(self, lambda_full):
        """D = damping_ratio * (Lambda^1/2 K^1/2 + K^1/2 Lambda^1/2), critical for K = diag(kp) given Lambda"""
        w, V = np.linalg.eigh(lambda_full)
        lambda_sqrt = V @ np.diag(np.sqrt(np.clip(w, 0.0, None))) @ V.T
        k_sqrt = np.diag(np.sqrt(self.kp))
        return self.damping_ratio * (lambda_sqrt @ k_sqrt + k_sqrt @ lambda_sqrt)

    def reset_goal(self, goal_update_mode="achieved"):
        super().reset_goal(goal_update_mode)
        self.integral_error = np.zeros(6)
        self.tau_prev = None
        if getattr(self, "reference", "hold") == "chunks":
            self._seed_reference()

    def run_controller(self):
        """
        Returns:
             np.array: Command torques
        """
        self.update()

        position_error, vel_pos_error, ori_error, vel_ori_error = self.compute_errors()
        error = np.concatenate([position_error, ori_error])
        if self.error_clip is not None:
            error = np.clip(error, -self.error_clip, self.error_clip)
        vel_error = np.concatenate([vel_pos_error, vel_ori_error])  # = -xdot (the base is fixed)

        # The robot accumulates the clipped error once per 1 ms control step, clamped; do the same per sim step
        self.integral_error = self.integral_error + error * (self.model_timestep * 1000.0)
        self.integral_error[:3] = np.clip(self.integral_error[:3], -0.1, 0.1)
        self.integral_error[3:] = np.clip(self.integral_error[3:], -0.3, 0.3)

        if self.damping is None:
            lambda_full, _, _, nullspace_matrix = opspace_matrices(self.mass_matrix, self.J_full, self.J_pos, self.J_ori)
            damping_term = self.damping_matrix(lambda_full) @ vel_error
        else:
            nullspace_matrix = None
            damping_term = self.damping * vel_error

        wrench = self.kp * error + damping_term + self.ki * self.integral_error
        tau = self.J_full.T @ wrench

        if self.nullspace == "projector":
            tau += projector_nullspace_torques(
                self.J_full,
                self.joint_pos,
                self.joint_vel,
                self.initial_joint,
                self.nullspace_stiffness,
                self.joint1_nullspace_stiffness,
                self.nullspace_lambda,
            )
        else:
            if nullspace_matrix is None:
                _, _, _, nullspace_matrix = opspace_matrices(self.mass_matrix, self.J_full, self.J_pos, self.J_ori)
            tau += nullspace_torques(self.mass_matrix, nullspace_matrix, self.initial_joint, self.joint_pos, self.joint_vel)

        if self.torque_rate_limit is not None:
            max_delta = self.torque_rate_limit * self.model_timestep
            if self.tau_prev is None:
                self.tau_prev = tau.copy()
            tau = self.tau_prev + np.clip(tau - self.tau_prev, -max_delta, max_delta)
            self.tau_prev = tau.copy()

        self.torques = tau + self.torque_compensation

        # Skip OSC's run_controller, run the base class's cleanup
        super(OperationalSpaceController, self).run_controller()

        return self.torques

    @property
    def name(self):
        return "CARTESIAN_IMPEDANCE"
