"""
KeyDoor: the key-in-lock task on a door. A 3 lb door panel hangs on a vertical hinge in a fixed frame; the lock
(KeyInsertion's chamfered slot and spring plunger) sits in a cylinder in the door with its keyhole facing the robot.
The task has three parts, in order:

    insert   the key goes into the slot and seats on the plunger (KeyInsertion, turned on its side)
    turn     with the key seated, turning it turns the cylinder. The cylinder is pinned (a strong spring toward 0)
             until the key is seated, like the pins of a real lock. A bolt in the door is coupled to the cylinder
             angle and retracts from the frame's strike as the cylinder turns. Past `unlock_angle` the door is unlocked
    push     the door swings open on its hinge when pushed (the bolt no longer meets the strike); success at
             `open_angle`. Pushing before it is unlocked drives the bolt into the strike and the door stays shut

Frames: the door frame body sits on the table in front of the robot. With no yaw the door panel lies in the world
x-z plane, its front face toward the robot (+y), the hinge at the panel's -x edge and the lock near its +x edge. The
keyhole frame (lock_site) has +z out of the door toward the robot, so the key travels -z into the lock exactly as in
KeyInsertion, and the slot's long side is vertical. The hinge's positive direction opens the door away from the robot.

Extra observations: door_angle, cylinder_angle, bolt_retraction, unlocked. Everything KeyInsertion exposes still
applies (lock_pos / quat follow the door).
"""

import xml.etree.ElementTree as ET

import numpy as np

import robosuite.utils.transform_utils as T
from robosuite.environments.manipulation.key_insertion import KeyInsertion, _look_at
from robosuite.models.arenas import TableArena
from robosuite.models.tasks import ManipulationTask
from robosuite.utils.observables import Observable, sensor

LB = 0.45359237  # kg


class KeyDoor(KeyInsertion):
    """
    Args (beyond KeyInsertion's; its lock_x/y/yaw ranges randomize the whole door frame's pose instead of the lock):
        door_mass (float): panel mass (kg); default 3 lb
        door_size (3-tuple): panel width (along the hinge-to-latch direction), thickness, height (m)
        door_y (float): frame position from the table center, toward the robot being +y (m)
        lock_height_above_table (float): keyhole height above the table top (m)
        hinge_damping, hinge_friction (float): the door hinge's viscous damping (N m s/rad) and dry friction (N m)
        cylinder_range (float): how far the cylinder turns (rad); the bolt is fully retracted at the end of it
        cylinder_friction (float): the cylinder's dry friction (N m), what the key feels turning an unlocked cylinder
        pin_stiffness (float): spring (N m/rad) holding the cylinder at 0 while the key is not seated
        unlock_angle (float): cylinder angle (rad) from which the bolt clears the strike
        bolt_length (float): how far the bolt protrudes into the strike when extended (m)
        open_angle (float): door angle (rad) that counts as opened
    """

    def __init__(
        self,
        robots,
        door_mass=3.0 * LB,
        door_size=(0.40, 0.05, 0.50),
        door_y=-0.30,
        lock_height_above_table=0.30,
        hinge_damping=0.5,
        hinge_friction=0.3,
        cylinder_range=np.pi / 2,
        cylinder_friction=0.15,
        pin_stiffness=50.0,
        unlock_angle=np.radians(75.0),
        bolt_length=0.012,
        open_angle=np.radians(15.0),
        lock_x_range=(-0.02, 0.02),
        lock_y_range=(-0.02, 0.02),
        lock_yaw_range=(-np.pi / 36, np.pi / 36),
        **kwargs,
    ):
        self.door_mass = float(door_mass)
        self.door_size = tuple(door_size)
        self.door_y = float(door_y)
        self.lock_height_above_table = float(lock_height_above_table)
        self.hinge_damping = float(hinge_damping)
        self.hinge_friction = float(hinge_friction)
        self.cylinder_range = float(cylinder_range)
        self.cylinder_friction = float(cylinder_friction)
        self.pin_stiffness = float(pin_stiffness)
        self.unlock_angle = float(unlock_angle)
        self.pin_release = np.radians(5.0)
        self.bolt_length = float(bolt_length)
        self.open_angle = float(open_angle)
        super().__init__(
            robots, lock_x_range=lock_x_range, lock_y_range=lock_y_range, lock_yaw_range=lock_yaw_range, **kwargs
        )

    # ---- model -------------------------------------------------------------------------------------------------------

    def _door_xml(self):
        """
        The frame (fixed): a full-height hinge post, a latch post that stops below the lock zone, a riser behind the
        door plane carrying the strike, and jamb stops above and below the lock zone. The zone around the keyhole is
        kept clear in front of the door because the wrist's 132 mm base follows the keyhole as the door swings and
        would otherwise hit the frame. Then the door (hinge), with the lock cylinder and the bolt in it.
        """
        W, Tk, H = self.door_size
        hinge_x = -W / 2
        lock_x = W - 0.06  # keyhole 6 cm from the latch edge (door body coordinates: hinge at 0, latch edge at W)
        zl = self.lock_height_above_table
        rgba_door = "0.55 0.38 0.22 1"
        rgba_frame = "0.35 0.35 0.38 1"
        solid = 'solref="0.004 1" solimp="0.99 0.9999 0.0002" priority="1" group="1"'

        # The lock cylinder: KeyInsertion's lock body, given a hinge about its own z (the keyhole axis), turned so its
        # z points out of the door's front face (+y of the door): Rx(-90 deg) maps local z -> +y, local y -> -z
        lock = ET.fromstring(self._lock_xml())
        lock.set("pos", f"{lock_x:.6g} {Tk / 2:.6g} {zl:.6g}")
        lock.set("quat", f"{np.cos(-np.pi / 4):.8g} {np.sin(-np.pi / 4):.8g} 0 0")
        joint = ET.Element("joint")
        joint.set("name", "lock_cylinder_joint")
        joint.set("type", "hinge")
        joint.set("axis", "0 0 1")
        joint.set("limited", "true")
        joint.set("range", f"0 {self.cylinder_range:.6g}")
        joint.set("damping", "0.02")
        joint.set("armature", "0.02")  # a real cylinder's inertia is tiny; this keeps the explicit Coulomb friction stable
        # cylinder_friction is applied in _pre_action as a Coulomb torque, not as joint frictionloss: MuJoCo's
        # frictionloss is a soft constraint scaled by the dof's inertia, and this 50 g cylinder is too light for it to
        # hold more than a fraction of the set value (0.8 N m set, ~0.1 N m held)
        joint.set("solreflimit", "0.002 1")
        joint.set("solimplimit", "0.99 0.9999 0.001")
        lock.insert(0, joint)
        inertial = ET.Element("inertial")  # the cylinder now moves: it needs its own mass (the lock geoms are massless walls)
        inertial.set("pos", "0 0 -0.02")
        inertial.set("mass", "0.05")
        inertial.set("diaginertia", "1e-5 1e-5 1e-5")
        lock.insert(0, inertial)
        lock_xml = ET.tostring(lock, encoding="unicode")

        # The bolt: slides along the door's x toward the latch edge, protruding bolt_length beyond it when extended
        bl, bh, bt = 0.03 + self.bolt_length, 0.012, 0.008  # bolt box: length, height, thickness
        bolt_x0 = W - 0.03 + bl / 2  # extended: 30 mm inside the door, bolt_length outside (door body coordinates)
        bolt = f"""
      <body name="lock_bolt" pos="{bolt_x0:.6g} 0 {zl - 0.03:.6g}">
        <inertial pos="0 0 0" mass="0.02" diaginertia="2e-6 2e-6 2e-6"/>
        <joint name="lock_bolt_joint" type="slide" axis="1 0 0" limited="true" range="{-self.bolt_length - 0.001:.6g} 0.001" damping="2"/>
        <geom name="lock_bolt" type="box" size="{bl / 2:.6g} {bt / 2:.6g} {bh / 2:.6g}" rgba="0.8 0.8 0.3 1" contype="2" conaffinity="2" condim="1" {solid}/>
      </body>"""

        # The panel, with a hole the lock block fills (its half-sizes are KeyInsertion's X, Y: 35 x 40 mm), so the key
        # meets the slot and its chamfer, not wood
        LX, LY = 0.035, 0.040
        pw = W - 0.002  # panel width (2 mm clearance at the latch edge)
        panel = "\n      ".join(
            f'<geom name="door_panel_{n}" type="box" pos="{(x0 + x1) / 2:.6g} 0 {(z0 + z1) / 2:.6g}" '
            f'size="{(x1 - x0) / 2:.6g} {Tk / 2:.6g} {(z1 - z0) / 2:.6g}" rgba="{rgba_door}" {solid}/>'
            for n, x0, x1, z0, z1 in (
                ("l", 0.0, lock_x - LX, 0.0, H),
                ("r", lock_x + LX, pw, 0.0, H),
                ("b", lock_x - LX, lock_x + LX, 0.0, zl - LY),
                ("t", lock_x - LX, lock_x + LX, zl + LY, H),
            )
        )
        door = f"""
    <body name="door" pos="{hinge_x:.6g} 0 0">
      <joint name="door_hinge" type="hinge" pos="0 0 0" axis="0 0 -1" limited="true" range="0 {np.radians(110):.6g}"
             damping="{self.hinge_damping:.6g}" frictionloss="{self.hinge_friction:.6g}" armature="0.002"/>
      <inertial pos="{W / 2:.6g} 0 {H / 2:.6g}" mass="{self.door_mass:.6g}"
                diaginertia="{self.door_mass * (Tk**2 + H**2) / 12:.6g} {self.door_mass * (W**2 + H**2) / 12:.6g} {self.door_mass * (W**2 + Tk**2) / 12:.6g}"/>
      {panel}
      {lock_xml}{bolt}
    </body>"""
        # frame body coordinates: the hinge at -W/2, the latch edge at +W/2
        strike_x = W / 2 + 0.003  # just past the latch edge
        frame = f"""<body name="door_frame" pos="0 0 0">
      <site name="door_frame_site" pos="0 0 0" size="0.002" rgba="0 0 1 0" group="1"/>
      <geom name="frame_hinge_post" type="box" pos="{hinge_x - 0.03:.6g} 0 {H / 2:.6g}" size="0.025 {Tk / 2:.6g} {H / 2:.6g}" rgba="{rgba_frame}" {solid}/>
      <geom name="frame_latch_post" type="box" pos="{W / 2 + 0.03:.6g} 0 {(zl - 0.08) / 2:.6g}" size="0.025 {Tk / 2:.6g} {(zl - 0.08) / 2:.6g}" rgba="{rgba_frame}" {solid}/>
      <geom name="frame_strike_riser" type="box" pos="{W / 2 + 0.03:.6g} {-bt / 2 - 0.0095:.6g} {(zl - 0.08 + zl - 0.018) / 2:.6g}" size="0.025 0.004 {(0.08 - 0.018) / 2:.6g}" rgba="{rgba_frame}" {solid}/>
      <geom name="frame_stop_b" type="box" pos="{W / 2 + 0.01:.6g} {Tk / 2 + 0.006:.6g} {(zl - 0.12) / 2:.6g}" size="0.02 0.005 {(zl - 0.12) / 2:.6g}" rgba="{rgba_frame}" {solid}/>
      <geom name="frame_stop_t" type="box" pos="{W / 2 + 0.01:.6g} {Tk / 2 + 0.006:.6g} {(zl + 0.12 + H) / 2:.6g}" size="0.02 0.005 {(H - zl - 0.12) / 2:.6g}" rgba="{rgba_frame}" {solid}/>
      <geom name="frame_strike" type="box" pos="{strike_x + self.bolt_length / 2 + 0.01:.6g} {-bt / 2 - 0.0055:.6g} {zl - 0.03:.6g}" size="{self.bolt_length / 2 + 0.01:.6g} 0.004 {bh:.6g}" rgba="0.5 0.5 0.55 1" contype="2" conaffinity="2" condim="1" {solid}/>
      {door}
    </body>"""
        return frame

    def _load_model(self):
        super(KeyInsertion, self)._load_model()
        xpos = self.robots[0].robot_model.base_xpos_offset["table"](self.table_full_size[0])
        self.robots[0].robot_model.set_base_xpos(xpos)
        arena = TableArena(table_full_size=self.table_full_size, table_friction=self.table_friction, table_offset=self.table_offset)
        arena.set_origin([0, 0, 0])

        frame = ET.fromstring(self._door_xml())
        self._frame_pos0 = self.table_offset + [0.0, self.door_y, 0.0]
        frame.set("pos", " ".join(f"{v:.6g}" for v in self._frame_pos0))
        arena.worldbody.append(frame)
        # The lock cylinder and the bolt live inside the door panel: no contact among them, or with the frame posts
        for a, b in (("door", "lock"), ("door", "lock_plunger"), ("lock", "lock_bolt"), ("door", "door_frame"), ("lock", "door_frame"), ("lock_plunger", "door_frame")):
            ex = ET.SubElement(arena.contact, "exclude")
            ex.set("body1", a)
            ex.set("body2", b)
        # bolt retraction = -(bolt_length) * cylinder angle / cylinder_range
        eq = ET.SubElement(arena.equality, "joint")
        eq.set("name", "lock_bolt_coupling")
        eq.set("joint1", "lock_bolt_joint")
        eq.set("joint2", "lock_cylinder_joint")
        eq.set("polycoef", f"0 {-self.bolt_length / self.cylinder_range:.8g} 0 0 0")
        eq.set("solref", "0.002 1")
        eq.set("solimp", "0.99 0.9999 0.001")

        W, Tk, H = self.door_size
        eye = self._frame_pos0 + [W / 2 - 0.06 + 0.15, 0.22, self.lock_height_above_table + 0.08]
        self._lockview_pos = eye
        arena.set_camera(
            camera_name="lockview",
            pos=eye,
            quat=T.convert_quat(T.mat2quat(_look_at(eye, self._frame_pos0 + [W / 2 - 0.06, Tk / 2, self.lock_height_above_table])), to="wxyz"),
        )
        self.model = ManipulationTask(mujoco_arena=arena, mujoco_robots=[r.robot_model for r in self.robots])

    def _setup_references(self):
        super()._setup_references()
        m = self.sim.model
        self.frame_body_id = m.body_name2id("door_frame")
        self.door_body_id = m.body_name2id("door")
        self.hinge_qpos_adr = m.get_joint_qpos_addr("door_hinge")
        self.cylinder_qpos_adr = m.get_joint_qpos_addr("lock_cylinder_joint")
        self.cylinder_dof_adr = m._model.jnt_dofadr[m.joint_name2id("lock_cylinder_joint")]
        self.bolt_qpos_adr = m.get_joint_qpos_addr("lock_bolt_joint")
        self.door_geom_ids = [i for i in range(m.ngeom) if (m.geom_id2name(i) or "").startswith(("door_", "frame_"))]

    # ---- state -------------------------------------------------------------------------------------------------------

    def door_angle(self):
        return float(self.sim.data.qpos[self.hinge_qpos_adr])

    def cylinder_angle(self):
        return float(self.sim.data.qpos[self.cylinder_qpos_adr])

    def bolt_retraction(self):
        return float(-self.sim.data.qpos[self.bolt_qpos_adr])

    def unlocked(self):
        return self.cylinder_angle() >= self.unlock_angle

    def _check_success(self):
        return self.door_angle() >= self.open_angle

    def key_seated(self):
        return self.plunger_depression() >= self.seat_fraction * self.plunger_travel

    def reward(self, action=None):
        """Sparse: 1 when the door is open. Shaped: KeyInsertion's for the insertion, then turning, then opening"""
        if self._check_success():
            reward = 1.0
        elif self.reward_shaping:
            if not self.key_seated():
                reward = 0.4 * super().reward() / (self.reward_scale or 1.0)
            elif not self.unlocked():
                reward = 0.4 + 0.3 * self.cylinder_angle() / self.unlock_angle
            else:
                reward = 0.7 + 0.3 * np.clip(self.door_angle() / self.open_angle, 0.0, 1.0)
        else:
            reward = 0.0
        if self.reward_scale is not None:
            reward *= self.reward_scale
        return reward

    def _setup_observables(self):
        observables = super()._setup_observables()
        if self.use_object_obs:
            modality = "object"

            @sensor(modality=modality)
            def door_angle(obs_cache):
                return np.array([self.door_angle()])

            @sensor(modality=modality)
            def cylinder_angle(obs_cache):
                return np.array([self.cylinder_angle()])

            @sensor(modality=modality)
            def bolt_retraction(obs_cache):
                return np.array([self.bolt_retraction()])

            @sensor(modality=modality)
            def unlocked(obs_cache):
                return np.array([float(self.unlocked())])

            for s in (door_angle, cylinder_angle, bolt_retraction, unlocked):
                observables[s.__name__] = Observable(name=s.__name__, sensor=s, sampling_rate=self.control_freq)
        return observables

    def _pre_action(self, action, policy_step=False):
        super()._pre_action(action, policy_step=policy_step)
        # The pins: at the rest angle the cylinder is held unless the key is seated. Once turned, the key is captive
        # and the cylinder free, as in a real lock (the pins only line up with the shear line at rest)
        q = self.cylinder_angle()
        pinned = (not self.key_seated()) and q < self.pin_release
        qd = float(self.sim.data.qvel[self.cylinder_dof_adr])
        friction = -self.cylinder_friction * np.tanh(qd / 0.2)  # Coulomb, smoothed over 0.2 rad/s (the key turns at 0.8)
        self.sim.data.qfrc_applied[self.cylinder_dof_adr] = (-self.pin_stiffness * q if pinned else 0.0) + friction

    def _reset_internal(self):
        super(KeyInsertion, self)._reset_internal()
        m = self.sim.model
        if not self.deterministic_reset:
            x = self.rng.uniform(*self.lock_x_range)
            y = self.rng.uniform(*self.lock_y_range)
            yaw = self.rng.uniform(*self.lock_yaw_range)
            m.body_pos[self.frame_body_id] = self._frame_pos0 + [x, y, 0.0]
            m.body_quat[self.frame_body_id] = [np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)]
            m.cam_pos[self.lockview_cam_id] = self._lockview_pos + [x, y, 0]
        for adr in (self.plunger_qpos_adr, self.hinge_qpos_adr, self.cylinder_qpos_adr, self.bolt_qpos_adr):
            self.sim.data.qpos[adr] = 0.0
        self.sim.data.qfrc_applied[self.plunger_dof_adr] = 0.0
        self.sim.data.qfrc_applied[self.cylinder_dof_adr] = 0.0
        self.sim.forward()
