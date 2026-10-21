import numpy as np

import robosuite.utils.transform_utils as T
from robosuite.robots.fixed_base_robot import FixedBaseRobot
from robosuite.utils.observables import sensor


class WristFixedBaseRobot(FixedBaseRobot):
    """
    FixedBaseRobot whose gripper may carry a wrist with mocap visuals (e.g. the telescoping sleeves of the Stewart
    wrist's legs) and command-dependent actuator gains (e.g. length-dependent leg stiffness). Keeps the visuals in
    sync with the physics (they have no effect on the dynamics) and applies the gains before every sim step.

    If the gripper has a "tcp" important site (WristGripperModel's wrist_tcp, fixed to the arm flange), the arm
    controller tracks it instead of the grip site, and it is observed as <robot prefix>tcp_pos / tcp_quat (world
    frame, quat xyzw, like eef_pos / eef_quat_site). eef_pos / eef_quat stay at the true grip site.
    """

    def _load_arm_controllers(self):
        super()._load_arm_controllers()
        for arm in self.arms:
            tcp = self.gripper[arm].important_sites.get("tcp")
            if tcp is not None:
                self.part_controller_config[arm]["ref_name"] = tcp

    def _create_arm_sensors(self, arm, modality):
        sensors, names = super()._create_arm_sensors(arm, modality)
        tcp = self.gripper[arm].important_sites.get("tcp")
        if tcp is None:
            return sensors, names
        pf = f"{arm}_" if len(self.arms) > 1 else ""  # as in Robot._create_arm_sensors; the robot prefix comes later

        @sensor(modality=modality)
        def tcp_pos(obs_cache):
            return np.array(self.sim.data.get_site_xpos(tcp))

        @sensor(modality=modality)
        def tcp_quat(obs_cache):
            return T.mat2quat(self.sim.data.get_site_xmat(tcp))

        return sensors + [tcp_pos, tcp_quat], names + [f"{pf}tcp_pos", f"{pf}tcp_quat"]

    def reset(self, deterministic=False, rng=None):
        super().reset(deterministic=deterministic, rng=rng)
        self.sim.forward()
        self._update_mocap_visuals()
        self._update_actuator_gains()

    def control(self, action, policy_step=False):
        self._update_mocap_visuals()
        super().control(action, policy_step=policy_step)
        self._update_actuator_gains()

    def _update_mocap_visuals(self):
        for arm in self.arms:
            if hasattr(self.gripper[arm], "update_mocap_visuals"):
                self.gripper[arm].update_mocap_visuals(self.sim)

    def _update_actuator_gains(self):
        for arm in self.arms:
            if hasattr(self.gripper[arm], "update_actuator_gains"):
                self.gripper[arm].update_actuator_gains(self.sim)
