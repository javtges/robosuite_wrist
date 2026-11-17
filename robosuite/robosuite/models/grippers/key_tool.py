"""
A rigid "key" (or RAM-stick-like blade) to mount on a wrist instead of a gripper: no joints, no actuators.

MJCF structure (raw names), +z pointing away from the mount (toward the table when the arm points down):

    key                 root body, at the mount face (the wrist's top plate)
      key_bow_*         the bow: a short block the blade is fixed to
      key_blade_*       the blade: a box blade_width x blade_thickness x blade_length
      eef               at the blade tip; carries grip_site, so the "eef" of this tool is the key's tip
      ee_cam            (optional) the end-effector camera, see EECameraConfig
      key_mount_vis     (optional) the printed mount as a visual mesh

The blade's width is along the key frame's x, its thickness along y (KeyInsertion's lock frame has the slot's long side
along y; at lock yaw 0 the robot's initial pose puts the key frame's x there). The bitting (the cut edge) is the
blade's +x edge: visual only, the collision blade stays a box and the lock is sized from it.

The end-effector camera is HSA-UMI's: an OmniVision OV9782 colour global-shutter sensor behind a wide-angle lens in an
Arducam USB module (ezBOX GS, 166 deg diagonal fisheye), run at 640x480, on a printed bracket beside the key looking
up the blade at the keyhole. It sits on the bitting side and sees the cut edge. Its placement comes from the CAD
assemblies (assets/grippers/ee_camera/camera_pose_cam25_top.json): lens 35.5 mm off the key axis on the blade-width
side, 54.1 mm behind the tip, optical axis tilted 25 deg toward the axis so it crosses the key axis 22 mm past the tip
(where the lock face is once the key is in). The rigid and the HSA mounts differ only in riser height; relative to the
key tip the camera is at the same place on both, so one definition serves both.
"""
import hashlib
import math
import os
import tempfile
from dataclasses import dataclass
from typing import Optional

import numpy as np

from robosuite.models.grippers.gripper_model import GripperModel
from robosuite.utils.mjcf_utils import xml_path_completion

EE_CAMERA_ASSETS = "grippers/ee_camera"


@dataclass
class KeyConfig:
    # Blade (m): the part that goes into the keyhole. Its cross-section (width x thickness) is what the keyhole fits.
    blade_width: float = 0.014
    blade_thickness: float = 0.004
    blade_length: float = 0.035
    # Bow (m): the block between the wrist's plate and the blade
    bow_width: float = 0.022
    bow_thickness: float = 0.010
    bow_length: float = 0.008
    mass: float = 0.010  # kg, the whole key (light: a few grams of brass / a RAM stick)
    friction: float = 0.4  # sliding friction against the lock (brass on steel); MuJoCo uses the larger of the pair

    @classmethod
    def measured(cls, **overrides):
        """
        The real key mount (CAD assemblies, camera_pose_cam25_top.json): the riser + jaws block stands 43.5 mm above
        the plate's top face, 27 mm along the blade's width by 14.2 mm across it, and the blade shows 36.5 mm beyond
        the jaws (rigid mount: 90 mm from the metal plate's top face to the tip, nuc/hsa_calib.py, jaws to 53.5 mm),
        so the tip is 80 mm above the plate. The blade's cross-section stays the sim's default: the lock is sized
        from it. The end-effector camera needs this geometry: with the default 43 mm key the camera, 54 mm behind
        the tip, would sit inside the wrist's top plate.
        """
        return cls(bow_width=0.027, bow_thickness=0.0142, bow_length=0.0435, blade_length=0.0365, **overrides)


@dataclass
class EECameraConfig:
    """
    The end-effector camera, placed relative to the key tip (all m / deg). Defaults are the CAD assemblies' numbers.

    The STL assembly frame (origin at the plate's bottom face, camera on +y) maps onto the key frame by a -90 deg yaw:
    the camera is on the blade-width side where the bitting is, the key frame's +x (the real bracket faces the cut
    edge). Whether the image is upright or rolled 180 deg is NOT in the CAD numbers: check against one real frame and
    set @roll_180.
    """

    name: str = "ee_cam"
    offset_from_axis: float = 0.0355  # lens centre to the key axis
    behind_tip: float = 0.05413  # lens centre behind the blade tip, along the key axis (38.87 - 93 mm, rigid CAD)
    tilt_deg: float = 25.0  # optical axis toward the key axis, from the key's +z (atan(0.4226 / 0.9063))
    side: int = +1  # +1: the camera on the key frame's +x, the bitting side (the real mount); -1: mirrored
    roll_180: bool = False  # True if the real image is upside down relative to this convention
    # Pinhole stand-in for the 166 deg (diagonal) fisheye: fovy that matches the fisheye's pixel scale at the image
    # centre (equidistant f = 400 px / 83 deg = 276 px for 640x480 -> fovy = 2 atan(240 / 276)). The corners of the
    # real image see more than this; the centre, where the key tip and the keyhole are, matches. The OV9782 is a
    # 1280x800 sensor: if the driver's 640x480 is a centre crop rather than a scale, the real field is narrower than
    # the lens's and this fovy is too wide. Replace it with a checkerboard calibration's focal length.
    fovy: float = 82.0
    width: int = 640
    height: int = 480
    # Visual mesh of the printed mount (bracket, jaws, camera board), None for the box bow only
    mount_mesh: Optional[str] = f"{EE_CAMERA_ASSETS}/hsa_key_cam25_top_assembly.stl"
    mount_mesh_plate_top: float = 0.005  # the STL's plate top face above its origin (5 mm top plate)

    @property
    def aim_past_tip(self):
        """Where the optical axis crosses the key axis, beyond the tip (m); 22 mm by the CAD"""
        return self.offset_from_axis / math.tan(math.radians(self.tilt_deg)) - self.behind_tip

    def pose_in_key_frame(self, tip_z):
        """
        (pos, quat wxyz) of the MuJoCo camera in the key body frame, for a blade tip at z = @tip_z. MuJoCo cameras
        look along their -z with +y up the image.
        """
        s, c = math.sin(math.radians(self.tilt_deg)), math.cos(math.radians(self.tilt_deg))
        pos = np.array([self.side * self.offset_from_axis, 0.0, tip_z - self.behind_tip])
        optical = np.array([-self.side * s, 0.0, c])  # toward the key axis and past the tip
        x_cam = np.array([self.side * c, 0.0, s])  # image right: the CAD's image_width_axis
        z_cam = -optical
        y_cam = np.cross(z_cam, x_cam)
        if self.roll_180:
            x_cam, y_cam = -x_cam, -y_cam
        R = np.column_stack([x_cam, y_cam, z_cam])
        return pos, _mat2quat_wxyz(R)


def _mat2quat_wxyz(R):
    """Rotation matrix -> unit quaternion (w, x, y, z), MuJoCo's order"""
    w = math.sqrt(max(0.0, 1.0 + R[0, 0] + R[1, 1] + R[2, 2])) / 2.0
    x = math.sqrt(max(0.0, 1.0 + R[0, 0] - R[1, 1] - R[2, 2])) / 2.0
    y = math.sqrt(max(0.0, 1.0 - R[0, 0] + R[1, 1] - R[2, 2])) / 2.0
    z = math.sqrt(max(0.0, 1.0 - R[0, 0] - R[1, 1] + R[2, 2])) / 2.0
    x = math.copysign(x, R[2, 1] - R[1, 2])
    y = math.copysign(y, R[0, 2] - R[2, 0])
    z = math.copysign(z, R[1, 0] - R[0, 1])
    q = np.array([w, x, y, z])
    return q / np.linalg.norm(q)


class KeyTool(GripperModel):
    """
    Args:
        config (KeyConfig): key geometry and mass
        camera (EECameraConfig or None): add the end-effector camera (and the mount's visual mesh)
        idn (int or str): Number or some other unique identification string for this gripper instance
    """

    def __init__(self, config=None, camera=None, idn=0):
        self.config = config or KeyConfig()
        self.camera = camera
        # Kept on disk (one file per key geometry): WristGripperModel re-reads the tool's raw xml from self.file
        xml = self.xml_string()
        path = os.path.join(tempfile.gettempdir(), f"key_tool_{hashlib.sha1(xml.encode()).hexdigest()[:12]}.xml")
        if not os.path.exists(path):
            with open(path, "w") as f:
                f.write(xml)
        super().__init__(path, idn=idn)

    @property
    def tip_z(self):
        """The blade tip above the mount face (m)"""
        return self.config.bow_length + self.config.blade_length

    def xml_string(self):
        c = self.config
        # Mass split by volume (one material)
        v_bow = c.bow_width * c.bow_thickness * c.bow_length
        v_blade = c.blade_width * c.blade_thickness * c.blade_length
        m_bow, m_blade = c.mass * v_bow / (v_bow + v_blade), c.mass * v_blade / (v_bow + v_blade)
        tip = self.tip_z
        fr = f"{c.friction} 0.005 0.0001"
        bow = f'size="{c.bow_width / 2} {c.bow_thickness / 2} {c.bow_length / 2}" pos="0 0 {c.bow_length / 2}"'
        blade = f'size="{c.blade_width / 2} {c.blade_thickness / 2} {c.blade_length / 2}" pos="0 0 {c.bow_length + c.blade_length / 2}"'
        assets, camera, mount = "", "", f'<geom name="key_bow_vis" type="box" {bow} contype="0" conaffinity="0" group="1" mass="0" rgba="0.80 0.65 0.25 1"/>'
        # The blade's look: the bitting cut into the +x edge (depth d, a fixed pattern of teeth), visual only
        d = min(0.0015, 0.15 * c.blade_width)
        heights = (0.9, 0.35, 0.7, 0.2, 0.55, 0.8, 0.3, 0.6)
        n = len(heights)
        tooth_len = c.blade_length / n
        blade_vis = [
            f'<geom name="key_blade_vis" type="box" size="{(c.blade_width - d) / 2:.6g} {c.blade_thickness / 2} {c.blade_length / 2}" '
            f'pos="{-d / 2:.6g} 0 {c.bow_length + c.blade_length / 2}" contype="0" conaffinity="0" group="1" mass="0" rgba="0.85 0.72 0.30 1"/>'
        ]
        for i, h in enumerate(heights):
            blade_vis.append(
                f'<geom name="key_tooth{i}_vis" type="box" size="{h * d / 2:.6g} {0.45 * c.blade_thickness:.6g} {0.46 * tooth_len:.6g}" '
                f'pos="{c.blade_width / 2 - d + h * d / 2:.6g} 0 {c.bow_length + (i + 0.5) * tooth_len:.6g}" '
                'contype="0" conaffinity="0" group="1" mass="0" rgba="0.85 0.72 0.30 1"/>'
            )
        blade_vis = "\n      ".join(blade_vis)
        if self.camera is not None:
            cam = self.camera
            pos, quat = cam.pose_in_key_frame(tip)
            camera = f'<camera name="{cam.name}" mode="fixed" pos="{_fmt(pos)}" quat="{_fmt(quat)}" fovy="{cam.fovy}"/>'
            if cam.mount_mesh:
                # The STL (mm, origin at the plate's bottom face, camera on +y) -> key frame (m, origin at the plate's
                # top face, camera on side * x): scale, drop by the plate's thickness, yaw -90 deg (or +90 for side -1)
                yaw = -cam.side * math.pi / 2
                mq = _fmt([math.cos(yaw / 2), 0, 0, math.sin(yaw / 2)])
                assets = f'<asset><mesh name="key_mount_mesh" file="{xml_path_completion(cam.mount_mesh)}" scale="0.001 0.001 0.001"/></asset>'
                mount = (
                    f'<geom name="key_mount_vis" type="mesh" mesh="key_mount_mesh" pos="0 0 {-cam.mount_mesh_plate_top}" quat="{mq}" '
                    'contype="0" conaffinity="0" group="1" mass="0" rgba="0.55 0.55 0.58 1"/>'
                )
        return f"""<mujoco model="key_tool">
  {assets}
  <worldbody>
    <body name="key" pos="0 0 0">
      <site name="ft_frame" pos="0 0 0" size="0.005" rgba="1 0 0 0" type="sphere" group="1"/>
      <geom name="key_bow_col" type="box" {bow} mass="{m_bow:.6g}" friction="{fr}" group="0" rgba="0.75 0.6 0.2 1"/>
      {mount}
      <geom name="key_blade_col" type="box" {blade} mass="{m_blade:.6g}" friction="{fr}" group="0" rgba="0.75 0.6 0.2 1"/>
      {blade_vis}
      {camera}
      <body name="eef" pos="0 0 {tip}" quat="0.707105 0 0 -0.707105">
        <site name="grip_site" pos="0 0 0" size="0.002" rgba="1 0 0 0.5" type="sphere" group="1"/>
        <site name="ee_x" pos="0.1 0 0" size="0.005 .1" quat="0.707105 0 0.707108 0" rgba="1 0 0 0" type="cylinder" group="1"/>
        <site name="ee_y" pos="0 0.1 0" size="0.005 .1" quat="0.707105 0.707108 0 0" rgba="0 1 0 0" type="cylinder" group="1"/>
        <site name="ee_z" pos="0 0 0.1" size="0.005 .1" quat="1 0 0 0" rgba="0 0 1 0" type="cylinder" group="1"/>
        <site name="grip_site_cylinder" pos="0 0 0" size="0.001 10" rgba="0 1 0 0" type="cylinder" group="1"/>
      </body>
    </body>
  </worldbody>
  <sensor>
    <force name="force_ee" site="ft_frame"/>
    <torque name="torque_ee" site="ft_frame"/>
  </sensor>
</mujoco>"""

    def format_action(self, action):
        return action

    @property
    def init_qpos(self):
        return None

    @property
    def _important_geoms(self):
        return {**super()._important_geoms, "key": ["key_bow_col", "key_blade_col"]}


def _fmt(a):
    return " ".join(f"{float(x):.6g}" for x in a)
