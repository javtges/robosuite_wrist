"""
Top-down key-in-lock (or RAM-stick-in-slot) insertion with a key rigidly mounted on a compliant wrist.

The key (models/grippers/key_tool.py) sits on the Stewart wrist's top plate in place of a gripper (gripper
"StewartWristKey"), and the task is to push it straight down into a lock fixed on the table. The keyhole is a
rectangular slot that fits the key's blade with a small clearance, with a chamfered (45 deg) entrance that a compliant
wrist can slide the key along when it lands off-center or slightly rotated. At the bottom of the slot a spring-loaded
plunger (a pin tumbler / the RAM slot's latch springs) pushes back: the key has to compress it by plunger_travel to
seat, against seat_preload at first contact rising to seat_force when seated. So inserting takes a real, sustained push
that the wrist's legs carry and the arm feels at its flange, on top of friction and any jamming on the walls.

Lock frame ("lock_site", the keyhole center on the lock's top face): z up, out of the keyhole; the slot's long side
(the blade's width) along y. At lock yaw 0 that matches the key on the robot's initial pose. The lock's position and
yaw are randomized on every reset (lock_x_range, lock_y_range, lock_yaw_range), by moving its (fixed) body in the
model, so hard_reset=False keeps working.

Success: the key is seated, i.e. the plunger is pushed down by at least seat_fraction of its travel (only the key can
reach it, so the blade must be down the slot).

Snap latch (latch_profile): instead of the linear spring, the plunger pushes back with any force-vs-depression curve,
e.g. a RAM slot's latch, where the force climbs to a breakaway peak, drops once the latch snaps over, then rises again
into the seat. The force is applied to the plunger's joint on every physics substep as a function of its position,
which is how MuJoCo integrates a joint spring too, so the key feels it only through its contact with the plunger.
"""
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

import robosuite.utils.transform_utils as T
from robosuite.environments.manipulation.manipulation_env import ManipulationEnv
from robosuite.models.arenas import TableArena
from robosuite.models.tasks import ManipulationTask
from robosuite.utils.observables import Observable, sensor


class KeyInsertion(ManipulationEnv):
    """
    Args (beyond the usual robosuite single-arm env args; see Lift):
        clearance (float): keyhole minus blade, per dimension (m). 0.5 mm = 0.25 mm per side.
        chamfer (float): size of the 45 deg chamfer around the keyhole's entrance (m). 0 for a sharp edge.
        slot_depth (float): depth (below the lock's top face) at which the key is seated (m)
        plunger_travel (float): how far the key pushes the plunger down to seat (m)
        seat_preload (float): plunger force (N) when the key first reaches it
        seat_force (float): plunger force (N) when the key is seated
        seat_fraction (float): fraction of plunger_travel that counts as seated
        latch_profile (None or list of (depression m, force N)): if given, replaces the spring (seat_preload,
            seat_force): the plunger's force, piecewise linear in its depression, held constant past the end points
        plunger_stop_timeconst (float): time constant (s) of the plunger's end stops. A fast key arriving at the bottom
            stop (e.g. after a latch snaps) needs ~5 timesteps (0.01) for its impact not to depend on the timestep
        lock_friction (float): sliding friction of the lock's surfaces
        lock_x_range, lock_y_range (2-tuples): lock position offsets (m) from the table center, sampled on reset
        lock_yaw_range (2-tuple): lock yaw (rad), sampled on reset
        lock_alpha (float): opacity of the lock's walls when rendered (e.g. 0.3 to see the key inside); physics unchanged
    """

    def __init__(
        self,
        robots,
        env_configuration="default",
        controller_configs=None,
        gripper_types="default",
        base_types="default",
        initialization_noise="default",
        table_full_size=(0.8, 0.8, 0.05),
        table_friction=(1.0, 5e-3, 1e-4),
        clearance=0.0005,
        chamfer=0.002,
        slot_depth=0.030,
        plunger_travel=0.004,
        seat_preload=4.0,
        seat_force=10.0,
        seat_fraction=0.9,
        latch_profile=None,
        plunger_stop_timeconst=0.004,
        lock_friction=0.4,
        lock_x_range=(-0.05, 0.05),
        lock_y_range=(-0.08, 0.08),
        lock_yaw_range=(-np.pi / 12, np.pi / 12),
        lock_alpha=1.0,
        use_camera_obs=True,
        use_object_obs=True,
        reward_scale=1.0,
        reward_shaping=False,
        has_renderer=False,
        has_offscreen_renderer=True,
        render_camera="frontview",
        render_collision_mesh=False,
        render_visual_mesh=True,
        render_gpu_device_id=-1,
        control_freq=20,
        lite_physics=True,
        horizon=1000,
        ignore_done=False,
        hard_reset=True,
        camera_names="agentview",
        camera_heights=256,
        camera_widths=256,
        camera_depths=False,
        camera_segmentations=None,
        renderer="mjviewer",
        renderer_config=None,
        seed=None,
    ):
        self.table_full_size = table_full_size
        self.table_friction = table_friction
        self.table_offset = np.array((0, 0, 0.8))

        self.clearance = clearance
        self.chamfer = chamfer
        self.slot_depth = slot_depth
        self.plunger_travel = plunger_travel
        self.seat_preload = seat_preload
        self.seat_force = seat_force
        self.seat_fraction = seat_fraction
        self.latch_profile = None if latch_profile is None else np.array(latch_profile, dtype=float).reshape(-1, 2)
        self.lock_friction = lock_friction
        self.lock_x_range = lock_x_range
        self.lock_y_range = lock_y_range
        self.lock_yaw_range = lock_yaw_range
        self.lock_alpha = lock_alpha
        self.plunger_armature = 0.05  # kg
        self.plunger_damping = 20.0  # N s/m: about critical with the armature
        self.plunger_stop_timeconst = plunger_stop_timeconst
        self.lock_height = slot_depth + 0.015  # lock's top face above the table

        self.reward_scale = reward_scale
        self.reward_shaping = reward_shaping
        self.use_object_obs = use_object_obs

        # The task only makes sense with the key on the wrist
        if gripper_types == "default":
            gripper_types = "StewartWristKey"

        super().__init__(
            robots=robots,
            env_configuration=env_configuration,
            controller_configs=controller_configs,
            base_types=base_types,
            gripper_types=gripper_types,
            initialization_noise=initialization_noise,
            use_camera_obs=use_camera_obs,
            has_renderer=has_renderer,
            has_offscreen_renderer=has_offscreen_renderer,
            render_camera=render_camera,
            render_collision_mesh=render_collision_mesh,
            render_visual_mesh=render_visual_mesh,
            render_gpu_device_id=render_gpu_device_id,
            control_freq=control_freq,
            lite_physics=lite_physics,
            horizon=horizon,
            ignore_done=ignore_done,
            hard_reset=hard_reset,
            camera_names=camera_names,
            camera_heights=camera_heights,
            camera_widths=camera_widths,
            camera_depths=camera_depths,
            camera_segmentations=camera_segmentations,
            renderer=renderer,
            renderer_config=renderer_config,
            seed=seed,
        )

    # ---- model -------------------------------------------------------------------------------------------------------

    @property
    def key_config(self):
        return self.robots[0].gripper["right"].gripper.config

    def _lock_xml(self):
        """
        The lock: a block with a rectangular, chamfered slot down its middle and a spring-loaded plunger at the slot's
        bottom, built from boxes. Body origin = keyhole center on the top face, z up.
        """
        k = self.key_config
        b = (k.blade_thickness + self.clearance) / 2  # slot half-size along x
        a = (k.blade_width + self.clearance) / 2  # slot half-size along y
        ch, D, H = self.chamfer, self.slot_depth, self.lock_height
        X, Y = 0.035, 0.040  # block half-size
        hp = 0.006  # plunger height
        floor = D + hp  # cavity floor depth: the plunger's bottom rests on it when the key is seated
        rgba = f"0.60 0.63 0.68 {self.lock_alpha}"
        fr = f"{self.lock_friction} 0.005 0.0001"
        # Stiff contacts: the key and the wrist's top plate are only ~20 g, and MuJoCo's contact softness scales with
        # the inverse of the effective mass, so the default solref would let the key sink into the walls
        geom_kw = f'type="box" friction="{fr}" solref="0.004 1" solimp="0.99 0.9999 0.0002" priority="1" rgba="{rgba}" group="1"'
        geoms = []

        def box(name, x0, x1, y0, y1, z0, z1, **extra):
            size = ((x1 - x0) / 2, (y1 - y0) / 2, (z1 - z0) / 2)
            pos = ((x1 + x0) / 2, (y1 + y0) / 2, (z1 + z0) / 2)
            attrs = " ".join(f'{k}="{v}"' for k, v in extra.items())
            geoms.append(
                f'<geom name="lock_{name}" size="{size[0]:.6g} {size[1]:.6g} {size[2]:.6g}" '
                f'pos="{pos[0]:.6g} {pos[1]:.6g} {pos[2]:.6g}" {geom_kw} {attrs}/>'
            )

        # Base under the cavity
        box("base", -X, X, -Y, Y, -H, -floor)
        # Walls around the slot, from the cavity floor up to the bottom of the chamfer
        box("wall_px", b, X, -Y, Y, -floor, -ch)
        box("wall_nx", -X, -b, -Y, Y, -floor, -ch)
        box("wall_py", -b, b, a, Y, -floor, -ch)
        box("wall_ny", -b, b, -Y, -a, -floor, -ch)
        if ch > 0:
            # Top slab around the chamfered opening
            box("top_px", b + ch, X, -Y, Y, -ch, 0)
            box("top_nx", -X, -b - ch, -Y, Y, -ch, 0)
            box("top_py", -b - ch, b + ch, a + ch, Y, -ch, 0)
            box("top_ny", -b - ch, b + ch, -Y, -a - ch, -ch, 0)
            # 45 deg chamfers: boxes whose top face runs from the slot's edge at depth ch to the top face, ch outward
            t = 0.004  # chamfer box thickness, into the (already solid) material below its face
            s = ch * np.sqrt(2) / 2
            for name, axis, sign, half_len in (("px", 0, 1, a), ("nx", 0, -1, a), ("py", 1, 1, b), ("ny", 1, -1, b)):
                edge = (b if axis == 0 else a) + ch / 2  # face center, outward from the slot center
                n_in = np.array([sign, -1.0]) / np.sqrt(2)  # into the material, in the (outward, z) plane
                c = np.array([sign * edge, -ch / 2]) + n_in * t / 2
                if axis == 0:  # bevel along y: rotate the box about y, so its top face tilts down toward the slot
                    pos, size = (c[0], 0, c[1]), (s, half_len, t / 2)
                    ang = -sign * np.pi / 4
                    quat = (np.cos(ang / 2), 0, np.sin(ang / 2), 0)
                else:  # bevel along x: rotate about x
                    pos, size = (0, c[0], c[1]), (half_len, s, t / 2)
                    ang = sign * np.pi / 4
                    quat = (np.cos(ang / 2), np.sin(ang / 2), 0, 0)
                geoms.append(
                    f'<geom name="lock_chamfer_{name}" size="{size[0]:.6g} {size[1]:.6g} {size[2]:.6g}" '
                    f'pos="{pos[0]:.6g} {pos[1]:.6g} {pos[2]:.6g}" quat="{" ".join(f"{q:.6g}" for q in quat)}" {geom_kw}/>'
                )
        else:
            box("top_px", b, X, -Y, Y, -1e-4, 0)  # (keeps the name set fixed; a sliver at the top)

        # Plunger: rests (joint at 0, against its upper limit) with its top plunger_travel above the seated depth, held
        # up by a preloaded spring: force = stiffness * (springref - q) for q in [-travel, 0]. Its armature keeps the
        # explicitly integrated spring's natural frequency well below the timestep's (a bare 5 g plunger rings at ~80 Hz)
        # With a latch profile, the joint has no spring: _pre_action applies the profile's force instead
        k_p = 0.0 if self.latch_profile is not None else (self.seat_force - self.seat_preload) / self.plunger_travel
        q_ref = 0.0 if self.latch_profile is not None else self.seat_preload / k_p
        top0 = -(D - self.plunger_travel)
        pb, pa = b - 0.0005, a - 0.0005  # plunger half-sizes: clears the walls (it never touches them)
        plunger = f"""
      <body name="lock_plunger" pos="0 0 {top0 - hp / 2:.6g}">
        <inertial pos="0 0 0" mass="0.005" diaginertia="1e-7 1e-7 1e-7"/>
        <joint name="lock_plunger_joint" type="slide" axis="0 0 1" limited="true" range="{-self.plunger_travel:.6g} 0"
               stiffness="{k_p:.6g}" springref="{q_ref:.6g}" damping="{self.plunger_damping:.6g}" armature="{self.plunger_armature:.6g}" solreflimit="{self.plunger_stop_timeconst:.6g} 1" solimplimit="0.99 0.9999 0.0002"/>
        <geom name="lock_plunger" type="box" size="{pb:.6g} {pa:.6g} {hp / 2:.6g}" friction="{fr}"
              solref="0.004 1" solimp="0.99 0.9999 0.0002" priority="1" rgba="0.75 0.2 0.15 1" group="1"/>
      </body>"""
        return f"""<body name="lock" pos="0 0 0">
      <site name="lock_site" pos="0 0 0" size="0.002" rgba="0 1 0 0" group="1"/>
      {chr(10).join("      " + g for g in geoms)}{plunger}
    </body>"""

    def _load_model(self):
        super()._load_model()

        xpos = self.robots[0].robot_model.base_xpos_offset["table"](self.table_full_size[0])
        self.robots[0].robot_model.set_base_xpos(xpos)

        mujoco_arena = TableArena(
            table_full_size=self.table_full_size,
            table_friction=self.table_friction,
            table_offset=self.table_offset,
        )
        mujoco_arena.set_origin([0, 0, 0])

        lock = ET.fromstring(self._lock_xml())
        lock.set("pos", " ".join(map(str, self.table_offset + [0, 0, self.lock_height])))
        mujoco_arena.worldbody.append(lock)
        # Close-up of the lock (from the front-left, looking down at it), for images of the insertion
        eye = np.array([0.12, -0.14, 0.11])
        self._lockview_pos = self.table_offset + eye
        mujoco_arena.set_camera(
            camera_name="lockview",
            pos=self._lockview_pos,
            quat=T.convert_quat(T.mat2quat(_look_at(eye, [0, 0, self.lock_height - 0.01])), to="wxyz"),
        )

        self.model = ManipulationTask(
            mujoco_arena=mujoco_arena,
            mujoco_robots=[robot.robot_model for robot in self.robots],
        )

    def _setup_references(self):
        super()._setup_references()
        m = self.sim.model
        self.lock_body_id = m.body_name2id("lock")
        self.lock_site_id = m.site_name2id("lock_site")
        self.lockview_cam_id = m.camera_name2id("lockview")
        self.plunger_qpos_adr = m.get_joint_qpos_addr("lock_plunger_joint")
        self.plunger_dof_adr = m._model.jnt_dofadr[m.joint_name2id("lock_plunger_joint")]
        gripper = self.robots[0].gripper["right"]
        self.key_geom_ids = [m.geom_name2id(gripper.correct_naming(g)) for g in gripper.gripper._important_geoms["key"]]
        self.lock_geom_ids = [i for i in range(m.ngeom) if (m.geom_id2name(i) or "").startswith("lock_")]
        self.tip_site_id = m.site_name2id(gripper.important_sites["grip_site"])

    # ---- state -------------------------------------------------------------------------------------------------------

    def lock_pose(self):
        """(pos, 3x3 rotation) of the keyhole frame (lock_site), world"""
        return self.sim.data.site_xpos[self.lock_site_id].copy(), self.sim.data.site_xmat[self.lock_site_id].reshape(3, 3).copy()

    def key_tip_in_lock(self):
        """(pos, 3x3 rotation) of the key's tip frame (grip_site) in the keyhole frame"""
        p, R = self.lock_pose()
        tip = self.sim.data.site_xpos[self.tip_site_id]
        R_tip = self.sim.data.site_xmat[self.tip_site_id].reshape(3, 3)
        return R.T @ (tip - p), R.T @ R_tip

    def insertion_depth(self):
        """How far the key's tip is below the lock's top face (m); negative above it"""
        return -self.key_tip_in_lock()[0][2]

    def plunger_depression(self):
        """How far the key has pushed the plunger down (m), 0 to plunger_travel"""
        return float(-self.sim.data.qpos[self.plunger_qpos_adr])

    def plunger_force(self):
        """Force (N) with which the plunger pushes up (spring or latch profile; not counting its end stops)"""
        x = self.plunger_depression()
        if self.latch_profile is None:
            return self.seat_preload + (self.seat_force - self.seat_preload) * x / self.plunger_travel
        return float(np.interp(x, self.latch_profile[:, 0], self.latch_profile[:, 1]))

    def key_contact_wrench(self):
        """
        Total contact force and torque (N, N m; world frame, torque about the key's tip) that the lock applies to the
        key, summed over every key-lock contact with mj_contactForce
        """
        m, d = self.sim.model._model, self.sim.data._data
        tip = d.site_xpos[self.tip_site_id]
        f6, force, torque = np.zeros(6), np.zeros(3), np.zeros(3)
        key, lock = set(self.key_geom_ids), set(self.lock_geom_ids)
        for i in range(d.ncon):
            c = d.contact[i]
            g1, g2 = c.geom1, c.geom2
            if not ((g1 in key and g2 in lock) or (g2 in key and g1 in lock)):
                continue
            mujoco.mj_contactForce(m, d, i, f6)
            frame = c.frame.reshape(3, 3)  # rows: contact normal (from geom1 to geom2), then the two tangents
            f = frame.T @ f6[:3]  # force on geom2 from geom1, world
            if g1 in key:
                f = -f  # on the key
            force += f
            torque += np.cross(c.pos - tip, f)
        return force, torque

    def _check_success(self):
        return self.plunger_depression() >= self.seat_fraction * self.plunger_travel

    # ---- reward / observations ---------------------------------------------------------------------------------------

    def reward(self, action=None):
        """
        Sparse: 1 when the key is seated.
        Shaped (reward_shaping=True), in [0, 1]: 0.3 for the tip approaching the keyhole's axis, 0.1 for the blade's yaw
        matching the slot's, 0.4 for insertion depth down to the plunger, 0.2 for the plunger's depression; 1 on success.
        """
        if self._check_success():
            reward = 1.0
        elif self.reward_shaping:
            p, R = self.key_tip_in_lock()
            lateral = np.linalg.norm(p[:2])
            # Blade width axis vs the slot's long side (y), sign-free: the blade is symmetric
            yaw_err = np.arccos(np.clip(abs(R[1, 1]), 0.0, 1.0))
            depth = np.clip(self.insertion_depth() / (self.slot_depth - self.plunger_travel), 0.0, 1.0)
            above = max(-self.insertion_depth(), 0.0)
            reward = (
                0.3 * (1 - np.tanh(20.0 * (lateral + above)))
                + 0.1 * (1 - np.tanh(5.0 * yaw_err))
                + 0.4 * depth
                + 0.2 * self.plunger_depression() / self.plunger_travel
            )
        else:
            reward = 0.0
        if self.reward_scale is not None:
            reward *= self.reward_scale
        return reward

    def _setup_observables(self):
        observables = super()._setup_observables()
        if self.use_object_obs:
            modality = "object"
            gripper = self.robots[0].gripper["right"]

            @sensor(modality=modality)
            def lock_pos(obs_cache):
                return self.lock_pose()[0]

            @sensor(modality=modality)
            def lock_quat(obs_cache):
                return T.mat2quat(self.lock_pose()[1])

            @sensor(modality=modality)
            def key_tip_to_lock_pos(obs_cache):
                """Key tip in the keyhole frame"""
                return self.key_tip_in_lock()[0]

            @sensor(modality=modality)
            def key_tip_to_lock_quat(obs_cache):
                return T.mat2quat(self.key_tip_in_lock()[1])

            @sensor(modality=modality)
            def plunger_depression(obs_cache):
                return np.array([self.plunger_depression()])

            @sensor(modality=modality)
            def flange_wrench(obs_cache):
                """External wrench on the arm's flange (moment, force; flange frame), as an F/T sensor there reads it"""
                return gripper.flange_wrench(self.sim)[1]

            sensors = [lock_pos, lock_quat, key_tip_to_lock_pos, key_tip_to_lock_quat, plunger_depression, flange_wrench]
            for s in sensors:
                observables[s.__name__] = Observable(name=s.__name__, sensor=s, sampling_rate=self.control_freq)
        return observables

    def _pre_action(self, action, policy_step=False):
        super()._pre_action(action, policy_step=policy_step)
        if self.latch_profile is not None:
            self.sim.data.qfrc_applied[self.plunger_dof_adr] = self.plunger_force()

    def _reset_internal(self):
        super()._reset_internal()
        if not self.deterministic_reset:
            x = self.rng.uniform(*self.lock_x_range)
            y = self.rng.uniform(*self.lock_y_range)
            yaw = self.rng.uniform(*self.lock_yaw_range)
            m = self.sim.model
            m.body_pos[self.lock_body_id] = self.table_offset + [x, y, self.lock_height]
            m.body_quat[self.lock_body_id] = [np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)]
            m.cam_pos[self.lockview_cam_id] = self._lockview_pos + [x, y, 0]  # the close-up follows the lock
        self.sim.data.qpos[self.plunger_qpos_adr] = 0.0
        self.sim.data.qfrc_applied[self.plunger_dof_adr] = 0.0
        self.sim.forward()

    def visualize(self, vis_settings):
        super().visualize(vis_settings=vis_settings)


def _look_at(eye, target):
    """Rotation of a MuJoCo camera (looks along -z, y up) at @eye looking at @target, world z up"""
    eye, target = np.asarray(eye, float), np.asarray(target, float)
    back = eye - target
    back /= np.linalg.norm(back)
    right = np.cross([0, 0, 1.0], back)
    right /= np.linalg.norm(right)
    up = np.cross(back, right)
    return np.column_stack([right, up, back])
