"""
Renders GIFs of the SpaceMouse teleop (teleop_wrist.py) into ../screenshots/, driving the real Teleop loop with a
simulated puck: raw readings in pyspacemouse's convention go through the same axis map (--axis-map hsa) and button
logic as the real SpaceMouse. A both-buttons press switches to the next mode (arm -> wrist -> nullspace); the gripper is
the g key.

    python make_teleop_gifs.py

Writes teleop_1_arm_6d.gif, teleop_2_wrist.gif, teleop_3_gripper.gif and teleop_4_arm_wrist_offset.gif. Each frame: a wide view (left) and a close-up that follows the wrist (right), with the mode, the puck
input and the measured tcp / wrist poses.
"""

import os
import tempfile
import types

import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.spatial.transform import Rotation

import teleop_wrist as tw
from teleop_math import SPACEMOUSE_TO_ROBOT, map_spacemouse_axes, rpy_to_matrix

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "screenshots")
W, H = 400, 300
FRAME_EVERY = 2  # 20 Hz control -> 10 fps GIF
AXES = ["+x", "+y", "+z", "+rot x", "+rot y", "+rot z"]


class SimulatedPuck:
    """Stands in for SpaceMouseSource: holds a raw pyspacemouse reading + buttons, mapped exactly like the real one"""

    exclusive = True
    name = "simulated spacemouse"

    def __init__(self):
        self.raw, self.buttons = np.zeros(6), [False, False]

    def push(self, robot_axes, left=False, right=False):
        """Set the raw reading that HSA's map turns into @robot_axes (robot base frame)"""
        robot_axes = np.asarray(robot_axes, dtype=float)
        self.raw = np.concatenate([SPACEMOUSE_TO_ROBOT.T @ robot_axes[:3], SPACEMOUSE_TO_ROBOT.T @ robot_axes[3:]])
        self.buttons = [left, right]

    def release(self):
        self.push(np.zeros(6))

    def read(self):
        return map_spacemouse_axes(self.raw, 0.0, "hsa"), list(self.buttons)

    def close(self):
        pass


def make_teleop(out, control_freq=20.0, no_gripper=False, arm_controller="osc", robot=None):
    args = types.SimpleNamespace(
        env="Lift", input="none", out=out, control_freq=control_freq, arm_controller=arm_controller, arm_gains_yaml=None,
        robot=robot or tw.DEFAULT_ROBOT,
        reference="chunks", chunk_len=6,
        max_pos_speed=0.20, max_rot_speed=0.15,
        wrist_pos_speed=0.04, wrist_rot_speed=0.35, axis_map="hsa", deadzone=0.0, print_axes=False,
        reseed_distance=0.15, nullspace_comp="plate", soft_scale=0.3, x_min=0.25, x_max=0.80, y_min=-0.40, y_max=0.40, z_min=None, z_max=0.60,
        floor_margin=0.0, plan=None, condition="", min_steps=10, no_render=True, no_realtime=True,
        no_terminal_keys=True, max_seconds=None, key_script="", no_gripper=no_gripper,
    )  # fmt: skip
    t = tw.Teleop(args)
    t.input = SimulatedPuck()
    return t


class Recorder:
    def __init__(self, teleop):
        self.t = teleop
        m = teleop.env.sim.model._model
        self.renderer = mujoco.Renderer(m, H, W)
        self.opt = mujoco.MjvOption()
        self.opt.geomgroup[0] = 0  # hide collision geoms
        # Over the robot's shoulder like the teleop viewer, aimed a little higher so the text banner doesn't hide the gripper
        self.wide = mujoco.MjvCamera()
        self.wide.lookat[:], self.wide.distance, self.wide.azimuth, self.wide.elevation = [0.0, 0.0, 1.02], 1.5, 35.0, -25.0
        self.close = mujoco.MjvCamera()
        self.close.distance, self.close.azimuth, self.close.elevation = 0.45, 60.0, -15.0
        self.font = ImageFont.load_default(size=15)
        self.small = ImageFont.load_default(size=13)
        self.frames = {}
        self.segment = None
        self.label = ""
        self.tick_count = 0
        self.ref_tcp = None

    def start(self, name):
        self.segment = name
        self.frames[name] = []
        self.ref_tcp = self.t.measured_tcp()

    def tcp_delta(self):
        now = self.t.measured_tcp()
        dpos = (now[:3] - self.ref_tcp[:3]) * 1e3
        drot = np.degrees((Rotation.from_rotvec(now[3:]) * Rotation.from_rotvec(self.ref_tcp[3:]).inv()).as_rotvec())
        return dpos, drot

    close_label = "close-up following the wrist"

    def close_lookat(self):
        """Where the close-up camera aims: just below the wrist base, so it follows the wrist"""
        t = self.t
        return t.env.sim.data.get_body_xpos(t.gripper.correct_naming(t.wrist_base_body)) + [0, 0, -0.07]

    def frame(self):
        t, d = self.t, self.t.env.sim.data._data
        self.close.lookat[:] = self.close_lookat()
        imgs = []
        for cam in (self.wide, self.close):
            self.renderer.update_scene(d, cam, self.opt)
            imgs.append(self.renderer.render())
        img = Image.fromarray(np.concatenate(imgs, axis=1))
        draw = ImageDraw.Draw(img)
        dpos, drot = self.tcp_delta()
        # Wrist offset from home, in the robot base frame (the wrist's own frame is rotated on the flange)
        wp = t.measured_wrist_pose()
        R_w = t.wrist_rotation()
        wpos = R_w @ wp[:3] * 1e3
        wrot = np.degrees(Rotation.from_matrix(R_w @ rpy_to_matrix(wp[3:]) @ R_w.T).as_rotvec())
        mode = tw.MODE_NAMES[t.mode].upper()
        buttons = t.input.buttons
        held = " + ".join(n for n, b in zip(("left btn (rotate)", "right btn (z)"), buttons) if b) or "no buttons"
        lines = [
            (self.font, f"mode: {mode}   puck: {self.label}   [{held}]   gripper: {'closed' if t.gripper_closed else 'open'}"),
            (self.small, f"arm tcp moved: {dpos[0]:+6.1f} {dpos[1]:+6.1f} {dpos[2]:+6.1f} mm   "
                         f"rot {drot[0]:+5.1f} {drot[1]:+5.1f} {drot[2]:+5.1f} deg  (since this clip started)"),
            (self.small, f"wrist offset from home: {wpos[0]:+6.1f} {wpos[1]:+6.1f} {wpos[2]:+6.1f} mm   "
                         f"rot {wrot[0]:+5.1f} {wrot[1]:+5.1f} {wrot[2]:+5.1f} deg  (x, y, z: robot base frame)"),
        ]  # fmt: skip
        draw.rectangle([0, 0, img.width, 64], fill=(0, 0, 0))
        y = 4
        for font, text in lines:
            draw.text((8, y), text, fill=(255, 255, 255), font=font)
            y += 20
        for x, text in ((0, "wide view"), (W, self.close_label)):
            box = draw.textbbox((x + 8, img.height - 20), text, font=self.small)
            draw.rectangle([box[0] - 4, box[1] - 3, box[2] + 4, box[3] + 3], fill=(0, 0, 0))
            draw.text((x + 8, img.height - 20), text, fill=(255, 255, 255), font=self.small)
        return img

    def run(self, seconds, label):
        self.label = label
        for _ in range(int(round(seconds * self.t.hz))):
            self.t.tick()
            self.tick_count += 1
            if self.tick_count % FRAME_EVERY == 0:
                self.frames[self.segment].append(self.frame())


def move(rec, axis, sign, seconds, hold=0.5):
    """Push one robot-frame axis with the buttons that HSA's modes need, then hold still"""
    puck = rec.t.input
    v = np.zeros(6)
    v[axis] = sign
    left = axis >= 3  # rotation needs the left button
    right = axis == 2  # z needs the right button
    puck.push(v, left=left, right=right)
    name = AXES[axis] if sign > 0 else AXES[axis].replace("+", "-")
    rec.run(seconds, f"push {name}")
    puck.release()
    rec.run(hold, "released")


def toggle(rec, mode):
    """Both puck buttons pressed together, then released, until the teleop is in @mode (arm -> wrist -> nullspace)"""
    puck = rec.t.input
    while rec.t.mode != mode:
        puck.push(np.zeros(6), left=True, right=True)
        rec.run(0.1, "both buttons: next mode")
        puck.release()
        rec.run(0.6, "released")


def six_d(rec, pos_s, rot_s):
    for axis in range(6):
        s = pos_s if axis < 3 else rot_s
        move(rec, axis, +1, s)
        move(rec, axis, -1, 2 * s)
        move(rec, axis, +1, s)


def save(frames, path):
    # One shared palette for the whole clip (per-frame palettes make the file several times larger)
    palette = frames[len(frames) // 2].quantize(colors=96, method=Image.Quantize.MEDIANCUT)
    frames = [f.quantize(palette=palette, dither=Image.Dither.NONE) for f in frames]
    frames[0].save(path, save_all=True, append_images=frames[1:], duration=1000 // (20 // FRAME_EVERY), loop=0, optimize=True)
    print(f"wrote {os.path.relpath(path, os.path.join(HERE, '..'))}: {len(frames)} frames, {os.path.getsize(path) / 1e6:.1f} MB")


def main():
    os.makedirs(OUT, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        t = make_teleop(os.path.join(tmp, "unused.hdf5"))
        rec = Recorder(t)

        # 1. Arm in 6D: each axis out, back past the start, and back again (0.10 m, 0.15 rad per push)
        rec.start("teleop_1_arm_6d")
        rec.run(0.5, "released")
        six_d(rec, pos_s=0.5, rot_s=1.0)

        # 2. Toggle to the wrist and move it in 6D; the arm's tcp stays put
        rec.start("teleop_2_wrist")
        toggle(rec, tw.WRIST)
        six_d(rec, pos_s=0.25, rot_s=0.5)

        # 3. Gripper: g closes, g opens
        rec.start("teleop_3_gripper")
        rec.run(0.5, "released")
        t.handle_key("g")
        rec.run(2.0, "(g key) gripper closing")
        t.handle_key("g")
        rec.run(2.0, "(g key) gripper opening")

        # 4. Leave the wrist off home (up, tilted, twisted), toggle back to the arm, move the arm in 6D:
        #    the wrist keeps its pose while the arm carries it
        rec.start("teleop_4_arm_wrist_offset")
        move(rec, 2, +1, 0.5)  # wrist z up (+20 mm, its limit)
        move(rec, 3, +1, 0.5)  # roll
        move(rec, 5, +1, 0.5)  # yaw
        toggle(rec, tw.ARM)  # wrist -> nullspace -> arm
        six_d(rec, pos_s=0.5, rot_s=1.0)
        t.close()

    for name, frames in rec.frames.items():
        save(frames, os.path.join(OUT, f"{name}.gif"))


if __name__ == "__main__":
    main()
