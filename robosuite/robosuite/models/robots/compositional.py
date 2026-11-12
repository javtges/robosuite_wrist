import numpy as np

import robosuite.utils.transform_utils as T
from robosuite.models.robots import *
from robosuite.utils.mjcf_utils import array_to_string, find_elements, find_parent, string_to_array


class PandaOmron(Panda):
    @property
    def default_base(self):
        return "OmronMobileBase"

    @property
    def default_arms(self):
        return {"right": "Panda"}

    @property
    def init_qpos(self):
        return np.array([0, np.pi / 16.0 - 0.2, 0.00, -np.pi / 2.0 - np.pi / 3.0, 0.00, np.pi - 0.4, np.pi / 4])

    @property
    def init_torso_qpos(self):
        return np.array([0.2])

    @property
    def base_xpos_offset(self):
        return {
            "bins": (-0.6, -0.1, 0),
            "empty": (-0.6, 0, 0),
            "table": lambda table_length: (-0.16 - table_length / 2, 0, 0),
        }


class SpotWithArm(SpotArm):
    @property
    def default_base(self):
        return "Spot"

    @property
    def default_arms(self):
        return {"right": "SpotArm"}

    @property
    def init_qpos(self):
        return np.array([0.0, -2, 1.26, -0.335, 0.862, 0.0])

    @property
    def base_xpos_offset(self):
        return {
            "bins": (-1.05, -0.1, -0.22),
            "empty": (-1.1, 0, -0.22),
            "table": lambda table_length: (-0.5 - table_length / 2, 0.0, -0.22),
        }


class SpotWithArmFloating(SpotArm):
    def __init__(self, idn=0):
        super().__init__(idn=idn)

    @property
    def init_qpos(self):
        return np.array([0.0, -2, 1.26, -0.335, 0.862, 0.0])

    @property
    def default_base(self):
        return "SpotFloating"

    @property
    def default_arms(self):
        return {"right": "SpotArm"}

    @property
    def base_xpos_offset(self):
        return {
            "bins": (-0.7, -0.1, 0.0),
            "empty": (-0.6, 0, 0.0),
            "table": lambda table_length: (-0.5 - table_length / 2, 0.0, 0.0),
        }


class PandaDexRH(Panda):
    @property
    def default_gripper(self):
        return {"right": "InspireRightHand"}

    @property
    def gripper_mount_pos_offset(self):
        return {"right": [0.0, 0.0, 0.0]}

    @property
    def gripper_mount_quat_offset(self):
        return {"right": [-0.5, 0.5, 0.5, -0.5]}


class PandaDexLH(Panda):
    @property
    def default_gripper(self):
        return {"right": "InspireLeftHand"}

    @property
    def gripper_mount_pos_offset(self):
        return {"right": [0.0, 0.0, 0.0]}

    @property
    def gripper_mount_quat_offset(self):
        return {"right": [0.5, -0.5, 0.5, -0.5]}


class StewartAlohaMixin:
    """
    A Franka arm + 6-DoF Stewart platform wrist + ALOHA gripper (see models/grippers/wrist_gripper.py). Mixed into the
    arm model (PandaStewartAloha, FR3StewartAloha below). Use gripper_types="StewartAlohaGripperLockedWrist" to keep
    the wrist fixed at home (7-dim action).
    """

    @property
    def default_gripper(self):
        return {"right": "StewartAlohaGripper"}

    @property
    def init_qpos(self):
        # Puts grip_site at the same initial pose as the stock Panda's, despite the ~17 cm longer wrist + gripper
        return np.array([-0.0121, -0.3035, 0.0124, -2.6197, 0.0058, 2.4435, 0.7812])

    def add_gripper(self, gripper, arm_name=None):
        super().add_gripper(gripper, arm_name)
        if getattr(gripper, "wrist", None) is None:
            return
        # Re-parent the eye-in-hand camera from the arm flange to the wrist's top plate, keeping its pose relative
        # to the gripper, so that it moves with the gripper and isn't blocked by the wrist
        cam = find_elements(self.worldbody, tags="camera", attribs={"name": self.naming_prefix + "eye_in_hand"}, return_first=True)
        flange = find_parent(self.worldbody, cam)
        top = find_elements(self.worldbody, tags="body", attribs={"name": gripper.correct_naming(gripper.wrist.mount_body)}, return_first=True)
        flange.remove(cam)
        wrist_quat = T.convert_quat(string_to_array(gripper.worldbody[0].get("quat", "1 0 0 0")), to="xyzw")
        cam_quat = T.convert_quat(string_to_array(cam.get("quat", "1 0 0 0")), to="xyzw")
        cam.set("pos", array_to_string(T.quat2mat(wrist_quat).T @ string_to_array(cam.get("pos", "0 0 0"))))
        cam.set("quat", array_to_string(T.convert_quat(T.quat_multiply(T.quat_inverse(wrist_quat), cam_quat), to="wxyz")))
        top.append(cam)
        gripper.add_mocap_visuals(self)


class PandaStewartAloha(StewartAlohaMixin, Panda):
    """Panda (Franka Emika Robot, HSA-UMI's `fer`) + Stewart wrist + ALOHA gripper"""


class FR3StewartAloha(StewartAlohaMixin, FR3):
    """Franka Research 3 (HSA-UMI's `fr3`) + Stewart wrist + ALOHA gripper: the arm in use from 2026-11"""
