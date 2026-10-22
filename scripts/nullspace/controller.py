"""
Null-space teleop for the Panda + Stewart wrist: the input moves the wrist, and the arm moves the opposite way, so
the wrist's top plate (and the gripper bolted to it) stays still in task space. See README.md for the derivation.

Frames (all 4x4 homogeneous transforms, T_A_B = pose of frame B in frame A):
    B   robot base (the OSC controller's reference frame)
    S   wrist_tcp site: the frame the arm's OSC tracks, fixed to the wrist base (so to the arm flange)
    W   wrist base body (bolted to the flange)
    P   wrist top plate body (the gripper is rigidly attached to it)

    T_B_P = T_B_S @ T_S_W @ T_W_P(p)

with p = [x, y, z, roll, pitch, yaw] the wrist pose relative to home (wrist base frame, R = Rz Ry Rx), exactly the
pose that StewartPlatformWrist commands.

Holding T_B_P at the anchor T_B_P0 gives the arm target for any wrist pose (feedforward):

    T_B_S*(p) = T_B_P0 @ inv(T_W_P(p)) @ inv(T_S_W)

and, given the arm's MEASURED tcp pose, the wrist pose that puts the plate back on the anchor (feedback):

    T_W_P(p_fb) = inv(T_S_W) @ inv(T_B_S_meas) @ T_B_P0
"""

import numpy as np
from scipy.spatial.transform import Rotation

STRATEGIES = ("feedforward", "feedback")


def make_T(R, p):
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = R, p
    return T


def inv_T(T):
    R, p = T[:3, :3], T[:3, 3]
    return make_T(R.T, -R.T @ p)


def pose6_to_T(pose6):
    """[pos, rotvec] (the OSC absolute action / teleop target) -> 4x4"""
    pose6 = np.asarray(pose6, dtype=float)
    return make_T(Rotation.from_rotvec(pose6[3:]).as_matrix(), pose6[:3])


def T_to_pose6(T):
    return np.concatenate([T[:3, 3], Rotation.from_matrix(T[:3, :3]).as_rotvec()])


def rpy_to_R(rpy):
    """R = Rz(yaw) Ry(pitch) Rx(roll), the wrist_top joint chain (rz, ry, rx)"""
    return Rotation.from_euler("ZYX", np.asarray(rpy)[::-1]).as_matrix()


def R_to_rpy(R):
    return Rotation.from_matrix(R).as_euler("ZYX")[::-1]


def pose_error(T_a, T_b):
    """(position error in m, rotation error in rad) between two poses"""
    dp = np.linalg.norm(T_a[:3, 3] - T_b[:3, 3])
    dR = Rotation.from_matrix(T_a[:3, :3].T @ T_b[:3, :3]).magnitude()
    return dp, dR


class NullspaceCompensator:
    """
    Args:
        home_pos (3-array): top plate frame at home, in the wrist base frame (StewartPlatformWrist.home_pos)
        T_S_W (4x4): wrist base frame in the tcp frame (constant)
        limits (6-array): wrist workspace half-ranges (m, rad); feedback commands are clipped to them
    """

    def __init__(self, home_pos, T_S_W, limits):
        self.home_pos = np.asarray(home_pos, dtype=float)
        self.T_S_W = np.asarray(T_S_W, dtype=float)
        self.T_W_S = inv_T(self.T_S_W)
        self.limits = np.asarray(limits, dtype=float)
        self.T_B_P0 = None

    @classmethod
    def from_sim(cls, sim, gripper, wrist):
        """Reads the constant tcp -> wrist base transform off a running sim"""
        d = sim.data
        base = gripper.worldbody[0].get("name")
        T_world_W = make_T(d.get_body_xmat(base), d.get_body_xpos(base))
        site = gripper.important_sites["tcp"]
        T_world_S = make_T(d.get_site_xmat(site), d.get_site_xpos(site))
        return cls(wrist.home_pos, inv_T(T_world_S) @ T_world_W, wrist.action_scale)

    # ---- kinematics --------------------------------------------------------------------------------------------------

    def T_W_P(self, p):
        """Top plate in the wrist base frame for wrist pose @p"""
        p = np.asarray(p, dtype=float)
        return make_T(rpy_to_R(p[3:]), self.home_pos + p[:3])

    def p_from_T_W_P(self, T):
        return np.concatenate([T[:3, 3] - self.home_pos, R_to_rpy(T[:3, :3])])

    def plate_pose(self, T_B_S, p):
        """T_B_P for an arm tcp pose and a wrist pose"""
        return T_B_S @ self.T_S_W @ self.T_W_P(p)

    # ---- controller --------------------------------------------------------------------------------------------------

    def anchor(self, tcp_target6, wrist_target):
        """Freezes the plate pose that the current (commanded) arm and wrist targets produce"""
        self.T_B_P0 = self.plate_pose(pose6_to_T(tcp_target6), wrist_target)

    def arm_target(self, wrist_target):
        """Feedforward: the arm tcp target [pos, rotvec] that keeps the plate on the anchor at wrist pose @wrist_target"""
        return T_to_pose6(self.T_B_P0 @ inv_T(self.T_W_P(wrist_target)) @ self.T_W_S)

    def wrist_command(self, tcp_measured6):
        """Feedback: the wrist pose that puts the plate on the anchor given where the arm's tcp actually is, clipped to the
        wrist's workspace. Equals the wrist target once the arm has caught up with arm_target()."""
        T = self.T_W_S @ inv_T(pose6_to_T(tcp_measured6)) @ self.T_B_P0
        return np.clip(self.p_from_T_W_P(T), -self.limits, self.limits)
