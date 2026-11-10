import numpy as np

from robosuite.controllers.parts.arm.osc import OperationalSpaceController
from robosuite.utils.control_utils import nullspace_torques, opspace_matrices


class CartesianImpedanceController(OperationalSpaceController):
    """
    Cartesian impedance control with a diagonal stiffness, like libfranka's cartesian_impedance_control example:

        F = K (x_goal - x) - D xdot,    tau = J^T F + gravity / coriolis compensation + nullspace torques

    Unlike OSC (F = Lambda (kp e - kd xdot)), the stiffness seen at the reference site is exactly diag(kp) (N/m for
    position, N*m/rad for orientation) in every configuration, with no coupling between axes through the arm's inertia
    Lambda. The damping is critical for that stiffness given Lambda, using the factorization design
    D = damping_ratio * (Lambda^1/2 K^1/2 + K^1/2 Lambda^1/2), so the closed loop doesn't ring along any direction.

    Takes the same arguments as OperationalSpaceController (goal handling, input types and interpolation are shared);
    kp is the stiffness itself, so it is typically 1e3 - 1e4 rather than OSC's ~150. Only fixed impedance and
    position + orientation control are supported.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert self.impedance_mode == "fixed", "CartesianImpedanceController only supports fixed impedance"
        assert self.use_ori, "CartesianImpedanceController controls position and orientation"
        # OSC turns damping_ratio into kd = 2 sqrt(kp) damping_ratio; keep the ratio itself
        self.damping_ratio = np.mean(self.kd / (2 * np.sqrt(self.kp)))

    def damping_matrix(self, lambda_full):
        """D = damping_ratio * (Lambda^1/2 K^1/2 + K^1/2 Lambda^1/2), critical for K = diag(kp) given Lambda"""
        w, V = np.linalg.eigh(lambda_full)
        lambda_sqrt = V @ np.diag(np.sqrt(np.clip(w, 0.0, None))) @ V.T
        k_sqrt = np.diag(np.sqrt(self.kp))
        return self.damping_ratio * (lambda_sqrt @ k_sqrt + k_sqrt @ lambda_sqrt)

    def run_controller(self):
        """
        Returns:
             np.array: Command torques
        """
        self.update()

        position_error, vel_pos_error, ori_error, vel_ori_error = self.compute_errors()
        lambda_full, _, _, nullspace_matrix = opspace_matrices(self.mass_matrix, self.J_full, self.J_pos, self.J_ori)

        wrench = self.kp * np.concatenate([position_error, ori_error]) + self.damping_matrix(lambda_full) @ np.concatenate(
            [vel_pos_error, vel_ori_error]
        )
        self.torques = self.J_full.T @ wrench + self.torque_compensation
        self.torques += nullspace_torques(
            self.mass_matrix, nullspace_matrix, self.initial_joint, self.joint_pos, self.joint_vel
        )

        # Skip OSC's run_controller, run the base class's cleanup
        super(OperationalSpaceController, self).run_controller()

        return self.torques

    @property
    def name(self):
        return "CARTESIAN_IMPEDANCE"
