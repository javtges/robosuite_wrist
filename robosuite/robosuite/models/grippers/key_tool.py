"""
A rigid "key" (or RAM-stick-like blade) to mount on a wrist instead of a gripper: no joints, no actuators.

MJCF structure (raw names), +z pointing away from the mount (toward the table when the arm points down):

    key                 root body, at the mount face (the wrist's top plate)
      key_bow_*         the bow: a short block the blade is fixed to
      key_blade_*       the blade: a box blade_width x blade_thickness x blade_length
      eef               at the blade tip; carries grip_site, so the "eef" of this tool is the key's tip
"""
import hashlib
import os
import tempfile
from dataclasses import dataclass

from robosuite.models.grippers.gripper_model import GripperModel


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


class KeyTool(GripperModel):
    """
    Args:
        config (KeyConfig): key geometry and mass
        idn (int or str): Number or some other unique identification string for this gripper instance
    """

    def __init__(self, config=None, idn=0):
        self.config = config or KeyConfig()
        # Kept on disk (one file per key geometry): WristGripperModel re-reads the tool's raw xml from self.file
        xml = self.xml_string()
        path = os.path.join(tempfile.gettempdir(), f"key_tool_{hashlib.sha1(xml.encode()).hexdigest()[:12]}.xml")
        if not os.path.exists(path):
            with open(path, "w") as f:
                f.write(xml)
        super().__init__(path, idn=idn)

    def xml_string(self):
        c = self.config
        # Mass split by volume (one material)
        v_bow = c.bow_width * c.bow_thickness * c.bow_length
        v_blade = c.blade_width * c.blade_thickness * c.blade_length
        m_bow, m_blade = c.mass * v_bow / (v_bow + v_blade), c.mass * v_blade / (v_bow + v_blade)
        tip = c.bow_length + c.blade_length
        fr = f"{c.friction} 0.005 0.0001"
        bow = f'size="{c.bow_width / 2} {c.bow_thickness / 2} {c.bow_length / 2}" pos="0 0 {c.bow_length / 2}"'
        blade = f'size="{c.blade_width / 2} {c.blade_thickness / 2} {c.blade_length / 2}" pos="0 0 {c.bow_length + c.blade_length / 2}"'
        return f"""<mujoco model="key_tool">
  <worldbody>
    <body name="key" pos="0 0 0">
      <site name="ft_frame" pos="0 0 0" size="0.005" rgba="1 0 0 0" type="sphere" group="1"/>
      <geom name="key_bow_col" type="box" {bow} mass="{m_bow:.6g}" friction="{fr}" group="0" rgba="0.75 0.6 0.2 1"/>
      <geom name="key_bow_vis" type="box" {bow} contype="0" conaffinity="0" group="1" mass="0" rgba="0.80 0.65 0.25 1"/>
      <geom name="key_blade_col" type="box" {blade} mass="{m_blade:.6g}" friction="{fr}" group="0" rgba="0.75 0.6 0.2 1"/>
      <geom name="key_blade_vis" type="box" {blade} contype="0" conaffinity="0" group="1" mass="0" rgba="0.85 0.72 0.30 1"/>
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
