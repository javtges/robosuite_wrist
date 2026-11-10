"""
Composes a wrist (robosuite/models/wrists) and an existing gripper into a single GripperModel, so the
pair can be mounted on any arm exactly like a normal gripper.

Action layout of the composed gripper: [wrist action (wrist.dof), gripper action (gripper.dof)].
With lock_wrist=True the wrist is held at home and the action is just the gripper action, so the
robot's action space is identical to that of the same arm with the plain gripper.

Observed joints (gripper_qpos / gripper_qvel): [gripper joints, wrist.observed_joints].
"""
import dataclasses
import os
import tempfile
import xml.etree.ElementTree as ET

import modern_robotics as mr
import mujoco
import numpy as np

import robosuite.utils.transform_utils as T
from robosuite.models.base import MujocoXML
from robosuite.models.grippers.gripper_model import GripperModel
from robosuite.models.grippers.key_tool import KeyConfig, KeyTool
from robosuite.models.wrists import StewartPlatformConfig, StewartPlatformWrist
from robosuite.utils.mjcf_utils import add_prefix, array_to_string, find_elements


def _xml_to_tmpfile(xml_string):
    fd, path = tempfile.mkstemp(suffix=".xml", prefix="wrist_gripper_")
    with os.fdopen(fd, "w") as f:
        f.write(xml_string)
    return path


class WristGripperModel(GripperModel):
    """
    Args:
        wrist (WristModel): wrist to place between the arm flange and the gripper
        gripper (GripperModel): gripper instance (only its raw xml file and its properties are used)
        idn (int or str): Number or some other unique identification string for this gripper instance
        wrist_quat (4-array): (w, x, y, z) orientation of the wrist base relative to the arm's eef mount body
        gripper_pos (3-array): position of the gripper root body relative to the wrist mount body
        eef_pos (None or 3-array): if set, overrides the gripper's "eef" body (grip_site) position,
            relative to the gripper root body
        lock_wrist (bool): if True, the wrist is held at home and is not part of the action space
        massless_gripper (bool): if True, the gripper's bodies get no mass or inertia (e.g. a NullGripper standing in
            for "no gripper": it keeps the grip_site / eef frames the rest of robosuite expects, and adds no load)
        tcp (str): where the "wrist_tcp" site goes. It is fixed to the wrist base (so to the arm flange), and arms
            built by WristFixedBaseRobot track it with their controller instead of the grip site, so wrist motion
            adds to arm motion instead of being cancelled by it (like HSA-UMI's hsa_tcp). "plate": the wrist's mount
            plate at home. "grip": the grip site at home. Either way its orientation is the grip site's at home.
    """

    def __init__(
        self,
        wrist,
        gripper,
        idn=0,
        wrist_quat=(1, 0, 0, 0),
        gripper_pos=(0, 0, 0),
        eef_pos=None,
        lock_wrist=False,
        tcp="plate",
        massless_gripper=False,
    ):
        self.wrist = wrist
        self.gripper = gripper
        self.lock_wrist = lock_wrist

        # Unprefixed copies of both models (defaults resolved inline, asset paths made absolute)
        wrist_file = _xml_to_tmpfile(wrist.xml_string())
        try:
            wrist_xml = MujocoXML(wrist_file)
        finally:
            os.remove(wrist_file)
        gripper_xml = MujocoXML(gripper.file)
        if massless_gripper:
            for body in gripper_xml.worldbody.iter("body"):
                for inertial in body.findall("inertial"):
                    body.remove(inertial)  # no geoms with mass either (NullGripper has none), so the body weighs 0

        wrist_xml.worldbody[0].set("quat", array_to_string(wrist_quat))
        gripper_root = gripper_xml.worldbody[0]
        gripper_root.set("pos", array_to_string(gripper_pos))
        if eef_pos is not None:
            eef = find_elements(root=gripper_xml.worldbody, tags="body", attribs={"name": "eef"}, return_first=True)
            eef.set("pos", array_to_string(eef_pos))
        wrist_xml.merge(gripper_xml, merge_body=wrist.mount_body)
        wrist_xml.worldbody[0].append(self._tcp_site(wrist_xml, wrist.mount_body, gripper._important_sites["grip_site"], tcp))

        merged_file = _xml_to_tmpfile(ET.tostring(wrist_xml.root, encoding="unicode"))
        try:
            super().__init__(merged_file, idn=idn)
        finally:
            os.remove(merged_file)

        # Gripper-relative orientation of the eef, composed with the wrist mount (wrist is identity at home)
        self.rotation_offset = T.quat_multiply(gripper.rotation_offset, T.convert_quat(np.array(wrist_quat), to="xyzw"))

        # Only expose the gripper joints and the wrist's observed joints. The remaining (passive) wrist joints
        # are fully determined by the closed chain and are restored from qpos0 on every sim reset.
        self._joints = list(gripper._joints) + list(wrist.observed_joints)
        self._init_qpos = np.concatenate([[] if gripper.init_qpos is None else gripper.init_qpos, wrist.init_qpos])

        # Map each (wrist / gripper) actuator command into the merged actuator ordering
        self._wrist_act_idx = [self._actuators.index(a) for a in wrist.actuators]
        self._gripper_act_idx = [self._actuators.index(a) for a in gripper._actuators]

    @staticmethod
    def _tcp_site(xml, mount_body, grip_site, tcp):
        """<site> for the wrist base: at the mount body (@tcp="plate") or @grip_site ("grip") at home, @grip_site's
        orientation at home"""
        assert tcp in ("plate", "grip"), f"Unknown tcp {tcp}"
        model = mujoco.MjModel.from_xml_string(ET.tostring(xml.root, encoding="unicode"))
        data = mujoco.MjData(model)
        mujoco.mj_kinematics(model, data)  # qpos0 is the wrist's home
        base = model.body(xml.worldbody[0].get("name")).id
        R_base, p_base = data.xmat[base].reshape(3, 3), data.xpos[base]
        site = model.site(grip_site).id
        pos = data.xpos[model.body(mount_body).id] if tcp == "plate" else data.site_xpos[site]
        quat = T.convert_quat(T.mat2quat(R_base.T @ data.site_xmat[site].reshape(3, 3)), to="wxyz")
        return ET.Element(
            "site",
            name="wrist_tcp",
            pos=array_to_string(R_base.T @ (pos - p_base)),
            quat=array_to_string(quat),
            size="0.005",
            rgba="0 0 0 0",
        )

    def format_action(self, action):
        action = np.asarray(action, dtype=float)
        assert len(action) == self.dof, f"Expected action of size {self.dof}, got {len(action)}"
        wrist_action = self.wrist.home_action if self.lock_wrist else action[: self.wrist.dof]
        gripper_action = action[self.dof - self.gripper.dof :]
        cmd = np.zeros(len(self._actuators))
        cmd[self._wrist_act_idx] = self.wrist.format_action(wrist_action)
        cmd[self._gripper_act_idx] = self.gripper.format_action(gripper_action)
        return cmd

    def add_mocap_visuals(self, robot_model):
        """
        Adds the wrist's mocap visuals to @robot_model's worldbody, so they end up directly under the world's
        <worldbody> (which mocap bodies require) instead of under the arm's eef body.
        """
        visuals = ET.fromstring(f"<worldbody>{self.wrist.mocap_visuals_xml()}</worldbody>")
        add_prefix(root=visuals, prefix=self.naming_prefix)
        for body in visuals:
            robot_model.worldbody.append(body)

    def update_mocap_visuals(self, sim):
        """Moves the mocap visuals to match the current physics state. Call after every sim step / reset."""
        model, data = sim.model._model, sim.data._data
        if getattr(self, "_mocap_model", None) is not model:
            # Name -> index lookups are slow, and this runs every sim step: cache them per compiled model
            self._mocap_model = model
            self._body_ids, self._qpos_adrs, self._mocap_ids = {}, {}, {}
        body_id = lambda b: self._body_ids.setdefault(b, model.body(self.correct_naming(b)).id)
        qpos_adr = lambda j: self._qpos_adrs.setdefault(j, model.joint(self.correct_naming(j)).qposadr[0])
        poses = self.wrist.mocap_visual_poses(
            body_pose=lambda b: (data.xpos[body_id(b)], data.xquat[body_id(b)]),
            joint_qpos=lambda j: data.qpos[qpos_adr(j)],
        )
        for name, (pos, quat) in poses.items():
            mocap_id = self._mocap_ids.setdefault(name, model.body_mocapid[body_id(name)])
            data.mocap_pos[mocap_id] = pos
            data.mocap_quat[mocap_id] = quat

    def update_actuator_gains(self, sim):
        """Applies the wrist's command-dependent actuator gains, if any. Call after setting ctrl, before stepping."""
        model, data = sim.model._model, sim.data._data
        if getattr(self, "_gains_model", None) is not model:
            self._gains_model = model
            self._wrist_act_ids = np.array([model.actuator(self.correct_naming(a)).id for a in self.wrist.actuators])
            self._wrist_dof_ids = model.jnt_dofadr[model.actuator_trnid[self._wrist_act_ids, 0]]
            self._joint_dof_ids = {}
        gains = self.wrist.actuator_gains(data.ctrl[self._wrist_act_ids])
        if gains is None:
            return
        kp = gains["kp"]
        # Position actuator: force = kp * ctrl - kp * qpos
        model.actuator_gainprm[self._wrist_act_ids, 0] = kp
        model.actuator_biasprm[self._wrist_act_ids, 1] = -kp
        model.dof_damping[self._wrist_dof_ids] = gains["damping"]
        if "armature" in gains:
            model.dof_armature[self._wrist_dof_ids] = gains["armature"]
        for joint, damping in gains.get("joint_damping", {}).items():
            if joint not in self._joint_dof_ids:
                self._joint_dof_ids[joint] = model.jnt_dofadr[model.joint(self.correct_naming(joint)).id]
            model.dof_damping[self._joint_dof_ids[joint]] = damping

    def flange_wrench(self, sim):
        """
        The wrench the wrist and what it carries apply to the arm at its flange, as Modern Robotics wrenches
        F = (m, f): moment (N m) then force (N), at the flange, in the flange frame (the wrist base body; +z points
        from the flange toward the tool, so pressing the tool into something gives a negative f_z).

        Returns (total, external):
          total     MuJoCo's own interaction force between the wrist base body and the arm's body it is mounted on
                    (cfrc_int, from mj_rnePostConstraint): everything the wrist transmits, i.e. the legs' spring and
                    damper forces, contacts on the tool, gravity and inertia of the wrist and tool
          external  total minus the static weight of the wrist and tool (MuJoCo's subtree mass and center of mass),
                    which leaves what contact and motion add, as a real F/T sensor's load compensation does
        """
        m, d = sim.model._model, sim.data._data
        b = m.body(self.worldbody[0].get("name")).id
        mujoco.mj_rnePostConstraint(m, d)  # fills cfrc_int from the step's accelerations and constraint forces
        R, p = d.xmat[b].reshape(3, 3), d.xpos[b]
        # cfrc_int[b]: (moment, force) the parent body applies to b's subtree, world axes, about the tree root's subtree
        # center of mass. [Ad_T]^T moves a wrench to another frame (Modern Robotics §3.4): F_f = [Ad_T_cf]^T F_c.
        on_wrist = mr.Adjoint(mr.RpToTrans(R, p - d.subtree_com[m.body_rootid[b]])).T @ d.cfrc_int[b]
        total = -on_wrist  # what the wrist applies to the arm
        gravity = np.concatenate([np.zeros(3), m.body_subtreemass[b] * m.opt.gravity])  # at the subtree's center of mass
        weight = mr.Adjoint(mr.RpToTrans(R, p - d.subtree_com[b])).T @ gravity
        return total, total - weight

    @property
    def current_action(self):
        return self._current_action

    @current_action.setter
    def current_action(self, value):
        # The robot zeroes this on reset; forward that reset to the inner gripper's internal state
        self._current_action = value
        self.gripper.current_action = np.zeros(self.gripper.dof)

    @property
    def dof(self):
        return self.gripper.dof + (0 if self.lock_wrist else self.wrist.dof)

    @property
    def init_qpos(self):
        return self._init_qpos

    @property
    def speed(self):
        return self.gripper.speed

    @property
    def _important_sites(self):
        return {**self.gripper._important_sites, "tcp": "wrist_tcp"}

    @property
    def _important_geoms(self):
        return self.gripper._important_geoms

    @property
    def _important_sensors(self):
        return self.gripper._important_sensors


def _null_gripper(idn):
    from robosuite.models.grippers.null_gripper import NullGripper

    return NullGripper(idn=idn)


def _aloha_gripper(idn):
    # Imported lazily: robosuite_models imports robosuite, so it can't be imported while robosuite loads
    try:
        from robosuite_models.robots.grippers.aloha_gripper import AlohaGripper
    except ImportError as e:
        raise ImportError("StewartAlohaGripper requires robosuite_models (pip install robosuite-models)") from e
    return AlohaGripper(idn=idn)


class StewartAlohaGripper(WristGripperModel):
    """
    6-DoF Stewart platform wrist with the ALOHA parallel-jaw gripper on top.
    Action: [x, y, z, roll, pitch, yaw (platform pose, normalized), grip (-1 open, 1 close)].

    The wrist geometry / workspace / leg stiffness for this gripper and its variants come from
    @wrist_config. To change them, replace it before creating the env, e.g.
        StewartAlohaGripper.wrist_config = StewartPlatformConfig(leg_stiffness_extended=3000.0)
    """

    wrist_config = StewartPlatformConfig()
    control_mode = "pose"
    lock_wrist = False
    tcp = "plate"
    # False: no gripper on the wrist (a massless NullGripper keeps the grip_site frame, at the top plate). The action
    # is then just the wrist's 6 dims. Use gripper_types="StewartWristNoGripper".
    attach_gripper = True

    def __init__(self, idn=0):
        wrist = StewartPlatformWrist(dataclasses.replace(self.wrist_config, control_mode=self.control_mode))
        gripper, kwargs = self._tool(idn)
        super().__init__(
            wrist=wrist,
            gripper=gripper,
            idn=idn,
            # Same yaw as the Panda hand, so the fingers close along the same axis as the stock Panda gripper
            wrist_quat=(0.707107, 0, 0, -0.707107),
            lock_wrist=self.lock_wrist,
            tcp=self.tcp,
            **kwargs,
        )

    def _tool(self, idn):
        """(GripperModel, extra WristGripperModel kwargs) for what goes on the wrist's top plate"""
        if not self.attach_gripper:
            return _null_gripper(idn), dict(eef_pos=(0, 0, 0), massless_gripper=True)
        return _aloha_gripper(idn), dict(
            # The ALOHA gripper's lowest point (camera bracket) is 42 mm below its root frame
            gripper_pos=(0, 0, 0.042),
            # Move grip_site from the ALOHA palm (z=0.04) to between the fingerpads, matching robosuite's
            # grip_site convention (used by the arm controller and by task reward shaping)
            eef_pos=(0, 0, 0.10),
        )


class StewartAlohaGripperLegControl(StewartAlohaGripper):
    """
    Same embodiment, but the 6 wrist actions command leg lengths directly (offsets from home, normalized by the
    leg stroke). Action: [leg0 ... leg5, grip].
    """

    control_mode = "legs"


class StewartAlohaGripperLockedWrist(StewartAlohaGripper):
    """
    Same embodiment as StewartAlohaGripper, but the wrist is held at home and only the grip is commanded
    (1-dim gripper action, identical action space to a stock parallel-jaw gripper).
    """

    lock_wrist = True


class StewartWristNoGripper(StewartAlohaGripper):
    """
    The Stewart wrist with nothing on it: no gripper, no load on the top plate. Action: the 6 wrist dims only.
    For studying the wrist alone, e.g. without the gripper's weight deflecting it.
    """

    attach_gripper = False


class StewartWristKey(StewartAlohaGripper):
    """
    The Stewart wrist with a rigid key (models/grippers/key_tool.py) on its top plate instead of a gripper, for the
    KeyInsertion task. Action: the 6 wrist dims only. grip_site is at the key's tip.
    To change the key, replace @key_config before creating the env (KeyInsertion sizes its keyhole from it).
    """

    key_config = KeyConfig()

    def _tool(self, idn):
        return KeyTool(self.key_config, idn=idn), {}
