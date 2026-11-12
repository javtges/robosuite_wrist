import numpy as np

from robosuite.controllers.parts.arm.osc import OperationalSpaceController
from robosuite.utils.control_utils import nullspace_torques, opspace_matrices


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
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        assert self.impedance_mode == "fixed", "CartesianImpedanceController only supports fixed impedance"
        assert self.use_ori, "CartesianImpedanceController controls position and orientation"
        assert nullspace in ("inertial", "projector"), f"unknown nullspace {nullspace!r}"
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
        self.integral_error = np.zeros(6)
        self.tau_prev = None

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
