"""
6-DoF Stewart platform (6-UPS) wrist, generated procedurally from a handful of geometric parameters.

MJCF structure (raw names):

    wrist_base                      base plate, rigidly attached to the arm flange
      wrist_leg{i}                  2 hinges (universal joint) at the base anchor
        wrist_leg{i}_rod            1 slide joint (actuated prismatic leg)
      wrist_top                     top plate; 6 passive joints (x, y, z, then yaw, pitch, roll) relative to home
        <gripper attaches here>

The loop is closed by one `connect` equality per leg (spherical joint at the top anchor), so the
platform has exactly 6 DoF, driven by 6 position actuators on the leg slides. All joints are 0 at
the home pose, so qpos0 is a consistent configuration of the closed chain.

Two control modes (config.control_mode), both 6-dim, normalized to [-1, 1], all-zero = home:
    "pose": absolute platform pose target relative to home,
            (x, y, z, roll, pitch, yaw) * (xy_range, xy_range, z_range, tilt_range, tilt_range, yaw_range),
            converted to leg lengths with the closed-form inverse kinematics. If the target needs more leg
            travel than available, it is pulled back toward home along the same direction until it fits.
    "legs": leg length offsets from home, * leg stroke. Leg commands whose resulting pose (forward kinematics)
            lies outside the workspace box are pulled back toward home until it is inside. Without this, large
            leg commands drive the platform through singular configurations and it flips.
"""
from dataclasses import dataclass
from typing import Optional

import numpy as np

from robosuite.models.wrists.wrist_model import WristModel


@dataclass
class StewartPlatformConfig:
    # Geometry (meters / degrees)
    base_anchor_radius: float = 0.060  # 120 mm diameter circle where the legs originate
    top_anchor_radius: float = 0.045  # 90 mm diameter circle where the legs terminate
    anchor_height: float = 0.080  # base anchors to top anchors, at home
    base_plate_radius: float = 0.066
    top_plate_radius: float = 0.051
    plate_thickness: float = 0.008
    # Visual-only spacer between the arm flange and the base plate
    spacer_height: float = 0.030
    spacer_radius: Optional[float] = None  # None = same as base_plate_radius
    base_anchor_half_angle: float = 7.5  # each anchor pair is centered on 0 / 120 / 240 deg
    top_anchor_half_angle: float = 7.5  # each anchor pair is centered on 60 / 180 / 300 deg
    leg_radius: float = 0.0055

    # Workspace spanned by the normalized action (+/- about home)
    xy_range: float = 0.010
    z_range: float = 0.020  # 40 mm total z stroke
    tilt_range: float = 10.0  # 20 deg total roll / pitch
    yaw_range: float = 10.0
    stroke_margin: float = 0.001  # extra leg travel beyond what the workspace needs
    # Leg travel (+/- about home). None = just enough for every corner of the workspace box above.
    leg_stroke: Optional[float] = None
    control_mode: str = "pose"  # "pose" or "legs"

    # Dynamics
    actuator_kp: float = 100000.0  # leg stiffness at home length (N/m). Above ~2e5, raise leg_armature proportionally.
    leg_damping: float = 500.0  # at home length
    # Leg stiffness k = actuator_kp * (L_home / L) ** leg_stiffness_exponent, where L is the commanded (unloaded) leg
    # length: 1 is an elastic member (k = EA / L), 0 is constant stiffness. Leg damping scales with sqrt(k), keeping
    # each leg's damping ratio fixed.
    leg_stiffness_exponent: float = 1.0
    leg_armature: float = 1.0  # reflected actuator inertia (kg); keeps the stiff legs stable at dt = 2 ms
    # Implicit damping on the top plate's (passive) translational / rotational joints. Damps the fast internal
    # modes of the closed chain, which otherwise get excited by the arm's operational-space controller.
    top_lin_damping: float = 400.0
    top_rot_damping: float = 4.0
    actuator_force_limit: float = 400.0
    base_mass: float = 0.25
    top_mass: float = 0.15
    leg_mass: float = 0.02
    # Loop-closure (leg-to-top-plate ball joint) constraint softness. Harder than MuJoCo's defaults, which let the
    # chain stretch by several mm during fast moves. solref time constant must stay >= 2 * timestep. solimp near 1 makes
    # the joints nearly rigid under static load, so the wrist's stiffness is that of the legs (platform_stiffness); at
    # 0.99 the joints add enough compliance in series to make the wrist ~33% softer.
    eq_solref: tuple = (0.004, 1.0)
    eq_solimp: tuple = (0.9999, 0.99999, 0.001)


def _rpy2mat(roll, pitch, yaw):
    cr, sr, cp, sp, cy, sy = np.cos(roll), np.sin(roll), np.cos(pitch), np.sin(pitch), np.cos(yaw), np.sin(yaw)
    rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    return rz @ ry @ rx


def _quat_z_to(v):
    """wxyz quaternion rotating +z onto unit vector @v"""
    z = np.array([0.0, 0.0, 1.0])
    axis = np.cross(z, v)
    s = np.linalg.norm(axis)
    if s < 1e-9:
        return np.array([1.0, 0, 0, 0])
    angle = np.arctan2(s, np.dot(z, v))
    return np.concatenate([[np.cos(angle / 2)], np.sin(angle / 2) * axis / s])


def _fmt(a):
    return " ".join(f"{x:.6g}" for x in np.atleast_1d(a))


class StewartPlatformWrist(WristModel):
    """
    Args:
        config (StewartPlatformConfig): geometry / workspace / dynamics parameters
    """

    def __init__(self, config=None):
        self.config = config if config is not None else StewartPlatformConfig()
        c = self.config

        # Anchor points: base anchors in the base frame (on the base plate's top face), top anchors in the
        # top-plate frame (on the top plate's bottom face; the top-plate frame sits on its top face).
        self.base_anchors, self.top_anchors = [], []
        for k in range(3):
            for sign in (1, -1):
                b = np.deg2rad(120 * k + sign * c.base_anchor_half_angle)
                p = np.deg2rad(120 * k + sign * (60 - c.top_anchor_half_angle))
                self.base_anchors.append([c.base_anchor_radius * np.cos(b), c.base_anchor_radius * np.sin(b), c.spacer_height + c.plate_thickness])
                self.top_anchors.append([c.top_anchor_radius * np.cos(p), c.top_anchor_radius * np.sin(p), -c.plate_thickness])
        self.base_anchors, self.top_anchors = np.array(self.base_anchors), np.array(self.top_anchors)
        self.home_pos = np.array([0.0, 0.0, c.spacer_height + 2 * c.plate_thickness + c.anchor_height])

        self.home_leg_vectors = self.home_pos + self.top_anchors - self.base_anchors
        self.home_leg_lengths = np.linalg.norm(self.home_leg_vectors, axis=1)
        self.action_scale = np.array(
            [c.xy_range, c.xy_range, c.z_range] + list(np.deg2rad([c.tilt_range, c.tilt_range, c.yaw_range]))
        )

        assert c.control_mode in ("pose", "legs"), f"Unknown control_mode {c.control_mode}"
        if c.leg_stroke is None:
            # Symmetric about home, large enough for every corner of the action box
            corners = np.array(np.meshgrid(*[[-1, 1]] * 6)).reshape(6, -1).T
            self.stroke = max(np.abs(self.leg_offsets(a * self.action_scale)).max() for a in corners) + c.stroke_margin
        else:
            self.stroke = c.leg_stroke

        self._home_jacobian_inv = np.linalg.inv(self.leg_jacobian(np.zeros(6)))

        # Pose sent to the legs on the last command (after clipping)
        self.last_commanded_pose = np.zeros(6)
        # Runtime multiplier on every leg's stiffness (e.g. a stiff / soft toggle while teleoperating). It changes model
        # parameters, not the sim state, so a replay of recorded states / actions has to set it the same way.
        self.stiffness_scale = 1.0

    def leg_lengths(self, pose):
        """
        Inverse kinematics.

        Args:
            pose (6-array): (x, y, z, roll, pitch, yaw) of the top plate relative to home (m, rad)

        Returns:
            6-array: leg lengths (anchor to anchor)
        """
        return np.linalg.norm(self.leg_vectors(pose), axis=1)

    def leg_vectors(self, pose):
        """Base anchor -> top anchor vectors (base frame) for @pose"""
        R = _rpy2mat(*pose[3:])
        return self.home_pos + pose[:3] + self.top_anchors @ R.T - self.base_anchors

    def leg_offsets(self, pose):
        """Leg lengths relative to home for @pose"""
        return self.leg_lengths(pose) - self.home_leg_lengths

    def leg_jacobian(self, pose):
        """d(leg lengths) / d(pose) at @pose (6x6, central differences)"""
        pose = np.asarray(pose, dtype=float)
        return np.array([(self.leg_lengths(pose + 1e-7 * e) - self.leg_lengths(pose - 1e-7 * e)) / 2e-7 for e in np.eye(6)]).T

    def leg_stiffness(self, leg_offsets):
        """Axial stiffness (N/m) of each leg when commanded to @leg_offsets from home"""
        c = self.config
        ratio = (self.home_leg_lengths / (self.home_leg_lengths + leg_offsets)) ** c.leg_stiffness_exponent
        return self.stiffness_scale * c.actuator_kp * ratio

    def platform_stiffness(self, pose):
        """
        Stiffness of the top plate held at @pose, K = J^T diag(k) J, with the legs commanded to @pose.
        Rows / columns are the pose coordinates (x, y, z, roll, pitch, yaw) of @leg_lengths, i.e. the wrist_top joints
        (N/m, N, N*m/rad): torques act on roll / pitch / yaw about their joint axes, which differ from the base axes
        once tilted. Leaves out the small geometric term from leg preload (e.g. the gripper's weight).
        """
        J = self.leg_jacobian(pose)
        return J.T @ np.diag(self.leg_stiffness(self.leg_offsets(pose))) @ J

    def actuator_gains(self, ctrl):
        c = self.config
        k = self.leg_stiffness(ctrl)
        return k, c.leg_damping * np.sqrt(k / c.actuator_kp)

    def forward_kinematics(self, leg_offsets, iters=100, tol=1e-9):
        """
        Pose (relative to home) that produces @leg_offsets: Newton's method from home, reusing the Jacobian at home
        (cheap, and converges over the whole workspace).

        Returns:
            6-array or None: pose, or None if it did not converge (e.g. far outside the workspace)
        """
        target = self.home_leg_lengths + leg_offsets
        pose = np.zeros(6)
        for _ in range(iters):
            err = self.leg_lengths(pose) - target
            if np.abs(err).max() < tol:
                return pose
            pose = pose - self._home_jacobian_inv @ err
            if np.any(np.abs(pose) > 10 * self.action_scale):
                return None
        return None

    def in_workspace(self, pose):
        return pose is not None and np.all(np.abs(pose) <= self.action_scale * (1 + 1e-6))

    @staticmethod
    def _pull_back(x, ok):
        """Returns @x if ok(x), otherwise a * x for the largest a in [0, 1] with ok(a * x) (by bisection)"""
        if ok(x):
            return x
        lo, hi = 0.0, 1.0
        for _ in range(20):
            mid = 0.5 * (lo + hi)
            lo, hi = (mid, hi) if ok(mid * x) else (lo, mid)
        return lo * x

    def format_action(self, action):
        action = np.clip(action, -1.0, 1.0)
        if self.config.control_mode == "legs":
            offsets = self._pull_back(action * self.stroke, lambda o: self.in_workspace(self.forward_kinematics(o)))
            self.last_commanded_pose = self.forward_kinematics(offsets)
        else:
            # Poses inside the box are always within leg travel unless leg_stroke was set smaller than needed
            self.last_commanded_pose = self._pull_back(
                action * self.action_scale, lambda p: np.all(np.abs(self.leg_offsets(p)) <= self.stroke)
            )
            offsets = self.leg_offsets(self.last_commanded_pose)
        return np.clip(offsets / self.stroke, -1.0, 1.0)

    @property
    def dof(self):
        return 6

    @property
    def mount_body(self):
        return "wrist_top"

    @property
    def actuators(self):
        return [f"wrist_leg{i}_act" for i in range(6)]

    @property
    def observed_joints(self):
        # Platform pose relative to home, then leg length offsets from home
        return [f"wrist_top_{a}" for a in ("x", "y", "z", "rx", "ry", "rz")] + [f"wrist_leg{i}_slide" for i in range(6)]

    def _leg_visual_extents(self, L):
        """
        Visual only (legs have no collision): the rod stays inside the housing over the full stroke. For long strokes
        the housing extends below the pivot, like a trunnion-mounted cylinder, into the spacer.

        Returns:
            3-tuple: housing bottom, housing top, rod start (along the leg axis, from the base pivot)
        """
        housing_top = L - self.stroke - 0.005
        rod_start = housing_top - self.stroke - 0.005
        return 0.0, L - self.stroke - 0.004, self.stroke + 0.002

    @property
    def spacer_radius(self):
        c = self.config
        return c.base_plate_radius if c.spacer_radius is None else c.spacer_radius

    def mocap_visuals_xml(self):
        # Middle stage of each telescoping leg: drawn at half the rod's travel, with no physics
        c = self.config
        return "".join(
            f'<body name="wrist_leg{i}_sleeve" mocap="true"><geom name="wrist_leg{i}_sleeve_vis" type="cylinder" '
            f'fromto="0 0 {self.stroke / 2:.6g} 0 0 {L - self.stroke / 2 - 0.004:.6g}" size="{0.75 * c.leg_radius}" '
            f'rgba="0.4 0.4 0.42 1" contype="0" conaffinity="0" group="1"/></body>'
            for i, L in enumerate(self.home_leg_lengths)
        )

    def mocap_visual_poses(self, body_pose, joint_qpos):
        poses = {}
        for i in range(6):
            pos, quat = body_pose(f"wrist_leg{i}")
            w, x, y, z = quat
            axis = np.array([2 * (x * z + w * y), 2 * (y * z - w * x), 1 - 2 * (x * x + y * y)])  # leg z axis in world
            poses[f"wrist_leg{i}_sleeve"] = (pos + axis * joint_qpos(f"wrist_leg{i}_slide") / 2, quat)
        return poses

    def xml_string(self):
        c = self.config
        t, Rb, Rt, h = c.plate_thickness, c.base_plate_radius, c.top_plate_radius, c.spacer_height
        vis = 'contype="0" conaffinity="0" group="1"'
        plate_I = lambda m, R: _fmt([m * (3 * R**2 + t**2) / 12] * 2 + [m * R**2 / 2])

        legs = []
        for i, (b, v, L) in enumerate(zip(self.base_anchors, self.home_leg_vectors, self.home_leg_lengths)):
            housing_bottom, housing_top, rod_start = self._leg_visual_extents(L)
            legs.append(
                f"""
      <body name="wrist_leg{i}" pos="{_fmt(b)}" quat="{_fmt(_quat_z_to(v / L))}">
        <inertial pos="0 0 {L / 4:.6g}" mass="{c.leg_mass}" diaginertia="1e-5 1e-5 1e-6"/>
        <joint name="wrist_leg{i}_u1" type="hinge" axis="1 0 0" damping="0.001" armature="1e-4"/>
        <joint name="wrist_leg{i}_u2" type="hinge" axis="0 1 0" damping="0.001" armature="1e-4"/>
        <geom name="wrist_leg{i}_housing" type="cylinder" fromto="0 0 {housing_bottom:.6g} 0 0 {housing_top:.6g}" size="{c.leg_radius}" rgba="0.2 0.2 0.22 1" {vis}/>
        <body name="wrist_leg{i}_rod">
          <inertial pos="0 0 {0.75 * L:.6g}" mass="{c.leg_mass / 2}" diaginertia="5e-6 5e-6 5e-7"/>
          <joint name="wrist_leg{i}_slide" type="slide" axis="0 0 1" range="{-self.stroke:.6g} {self.stroke:.6g}" limited="true" damping="{c.leg_damping}" armature="{c.leg_armature}"/>
          <geom name="wrist_leg{i}_rod_vis" type="capsule" fromto="0 0 {rod_start:.6g} 0 0 {L:.6g}" size="{0.5 * c.leg_radius}" rgba="0.15 0.7 0.25 1" {vis}/>
        </body>
      </body>"""
            )

        anchors = lambda pts, prefix: "".join(
            f'\n        <geom name="{prefix}{i}" type="sphere" pos="{_fmt(p)}" size="{c.leg_radius * 1.1}" rgba="0.1 0.1 0.1 1" {vis}/>'
            for i, p in enumerate(pts)
        )
        eqs = "".join(
            f'\n    <connect name="wrist_leg{i}_ball" body1="wrist_leg{i}_rod" body2="wrist_top" anchor="0 0 {L:.6g}" solref="{_fmt(c.eq_solref)}" solimp="{_fmt(c.eq_solimp)}"/>'
            for i, L in enumerate(self.home_leg_lengths)
        )
        acts = "".join(
            f'\n    <position name="wrist_leg{i}_act" joint="wrist_leg{i}_slide" kp="{c.actuator_kp}" ctrllimited="true" ctrlrange="{-self.stroke:.6g} {self.stroke:.6g}" forcelimited="true" forcerange="{-c.actuator_force_limit} {c.actuator_force_limit}"/>'
            for i in range(6)
        )
        top_joints = "".join(
            f'\n        <joint name="wrist_top_{n}" type="{typ}" axis="{ax}" armature="1e-4" '
            f'damping="{c.top_lin_damping if typ == "slide" else c.top_rot_damping}"/>'
            for n, typ, ax in [
                ("x", "slide", "1 0 0"),
                ("y", "slide", "0 1 0"),
                ("z", "slide", "0 0 1"),
                # z-y-x chain, so (rx, ry, rz) are exactly the (roll, pitch, yaw) used by leg_lengths()
                ("rz", "hinge", "0 0 1"),
                ("ry", "hinge", "0 1 0"),
                ("rx", "hinge", "1 0 0"),
            ]
        )

        return f"""<mujoco model="stewart_wrist">
  <worldbody>
    <body name="wrist_base">
      <inertial pos="0 0 {h + t / 2}" mass="{c.base_mass}" diaginertia="{plate_I(c.base_mass, Rb)}"/>
      <geom name="wrist_spacer_vis" type="cylinder" pos="0 0 {h / 2}" size="{self.spacer_radius:.6g} {h / 2}" rgba="0.35 0.36 0.38 1" {vis}/>
      <geom name="wrist_base_vis" type="cylinder" pos="0 0 {h + t / 2}" size="{Rb} {t / 2}" rgba="0.1 0.3 0.85 1" {vis}/>
      <geom name="wrist_base_col" type="cylinder" pos="0 0 {h + t / 2}" size="{Rb} {t / 2}" group="0"/>{anchors(self.base_anchors, "wrist_base_anchor")}{"".join(legs)}
      <body name="wrist_top" pos="{_fmt(self.home_pos)}">
        <inertial pos="0 0 {-t / 2}" mass="{c.top_mass}" diaginertia="{plate_I(c.top_mass, Rt)}"/>{top_joints}
        <geom name="wrist_top_vis" type="cylinder" pos="0 0 {-t / 2}" size="{Rt} {t / 2}" rgba="0.85 0.15 0.15 1" {vis}/>
        <geom name="wrist_top_col" type="cylinder" pos="0 0 {-t / 2}" size="{Rt} {t / 2}" group="0"/>{anchors(self.top_anchors, "wrist_top_anchor")}
      </body>
    </body>
  </worldbody>
  <equality>{eqs}
  </equality>
  <actuator>{acts}
  </actuator>
</mujoco>
"""
