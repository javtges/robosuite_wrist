"""
Defines the interface a wrist must satisfy to be composed between an arm and a gripper
(see robosuite/models/grippers/wrist_gripper.py).

A wrist is anything that sits between the arm flange and the gripper. It is responsible for
(1) producing an MJCF file and (2) mapping normalized wrist actions to normalized actuator commands.
Everything else (prefixing, merging, robot / controller bookkeeping) is handled by the composition
layer, so a wrist implementation can be rewritten freely as long as it keeps this interface.
"""
import numpy as np


class WristModel:
    """
    Base class for wrists. All names are RAW (unprefixed) MJCF names.
    """

    def xml_string(self):
        """
        Returns:
            str: full MJCF (<mujoco>...</mujoco>) for the wrist. Must contain exactly one root body in <worldbody>,
                and the body named @self.mount_body is where the gripper gets attached.
        """
        raise NotImplementedError

    @property
    def mount_body(self):
        """
        Returns:
            str: name of the body the gripper is attached to
        """
        raise NotImplementedError

    @property
    def actuators(self):
        """
        Returns:
            list of str: wrist actuator names, in the order returned by @format_action
        """
        raise NotImplementedError

    @property
    def observed_joints(self):
        """
        Returns:
            list of str: wrist joints exposed to the robot (reset to @init_qpos and reported in gripper_qpos)
        """
        raise NotImplementedError

    @property
    def init_qpos(self):
        """
        Returns:
            np.array: initial qpos for @observed_joints
        """
        return np.zeros(len(self.observed_joints))

    @property
    def dof(self):
        """
        Returns:
            int: dimension of the wrist action
        """
        raise NotImplementedError

    def format_action(self, action):
        """
        Maps a normalized wrist action in [-1, 1]^dof to normalized actuator commands in [-1, 1]
        (one per @actuators entry; -1 / 1 correspond to the low / high end of each actuator's ctrlrange).
        """
        raise NotImplementedError

    def actuator_gains(self, ctrl):
        """
        Args:
            ctrl (np.array): current actuator commands (ctrlrange units), one per @actuators entry

        Returns:
            None if the gains are fixed, else (kp, damping): position gain of each actuator and damping of the joint
                it drives, applied before every sim step
        """
        return None

    def mocap_visuals_xml(self):
        """
        Returns:
            str: optional MJCF <body mocap="true"> elements (no physics) to add directly under <worldbody>, for parts
                that should be drawn but can't be attached rigidly to any wrist body
        """
        return ""

    def mocap_visual_poses(self, body_pose, joint_qpos):
        """
        Args:
            body_pose (fn): raw wrist body name -> (world pos, world quat wxyz)
            joint_qpos (fn): raw wrist joint name -> qpos

        Returns:
            dict: raw mocap body name -> (world pos, world quat wxyz), for each body in @mocap_visuals_xml
        """
        return {}

    @property
    def home_action(self):
        """
        Returns:
            np.array: wrist action that holds the wrist at its home configuration
        """
        return np.zeros(self.dof)
