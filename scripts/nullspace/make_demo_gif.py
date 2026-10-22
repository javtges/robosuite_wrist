"""
GIF of the nullspace teleop mode: figures/nullspace_demo.gif.

    cd scripts && python -m nullspace.make_demo_gif

The same puck input (wrist x, z, roll, yaw to their limits and back) is shown twice: first in wrist mode (the arm holds
still, so the gripper moves with the wrist), then in nullspace mode with feedback compensation (the arm moves the
opposite way, so the gripper stays still). The right-hand close-up is fixed in the world at the gripper's starting
position, so a still gripper looks still while the flange and the wrist's base move around it.
"""

import os
import sys
import tempfile

import numpy as np
from PIL import ImageDraw

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import make_teleop_gifs as mg  # noqa: E402
import teleop_wrist as tw  # noqa: E402
from nullspace.controller import make_T, pose_error  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))


class NullspaceRecorder(mg.Recorder):
    """Recorder with a world-fixed close-up and a line for the top plate's motion"""

    close_label = "close-up fixed in the world"

    def start(self, name):
        super().start(name)
        d = self.t.env.sim.data
        top = self.t.gripper.correct_naming("wrist_top")
        self.P0 = make_T(d.get_body_xmat(top), d.get_body_xpos(top))
        self.fixed_lookat = d.get_body_xpos(top) + [0, 0, -0.02]
        self.close.distance = 0.55

    def frame(self):
        img = super().frame()
        d = self.t.env.sim.data
        top = self.t.gripper.correct_naming("wrist_top")
        e = pose_error(self.P0, make_T(d.get_body_xmat(top), d.get_body_xpos(top)))
        draw = ImageDraw.Draw(img)
        y = img.height - 46
        text = f"top plate + gripper moved: {e[0] * 1e3:5.2f} mm, {np.degrees(e[1]):5.2f} deg  (since this clip started)"
        box = draw.textbbox((8, y), text, font=self.font)
        draw.rectangle([0, box[1] - 4, box[2] + 8, box[3] + 4], fill=(0, 0, 0))
        draw.text((8, y), text, fill=(255, 220, 120), font=self.font)
        return img

    def close_lookat(self):
        return self.fixed_lookat  # fixed in the world, at the gripper's starting position


def sequence(rec):
    for axis, dur in ((0, 0.25), (2, 0.5), (3, 0.5), (5, 0.5)):
        mg.move(rec, axis, +1, dur, hold=0.6)
        mg.move(rec, axis, -1, 2 * dur, hold=0.6)
        mg.move(rec, axis, +1, dur, hold=0.6)


def main():
    with tempfile.TemporaryDirectory() as tmp:
        t = mg.make_teleop(os.path.join(tmp, "unused.hdf5"))
        t.args.nullspace_comp = "feedback"
        for _ in range(20):
            t.tick()
        rec = NullspaceRecorder(t)
        rec.start("nullspace_demo")
        mg.toggle(rec, tw.WRIST)
        rec.start("nullspace_demo")  # restart the clock and the plate reference after the toggle
        sequence(rec)
        wrist_frames = rec.frames["nullspace_demo"]
        mg.toggle(rec, tw.NULLSPACE)
        rec.start("nullspace_demo")
        sequence(rec)
        frames = wrist_frames + rec.frames["nullspace_demo"]
        t.close()
    os.makedirs(os.path.join(HERE, "figures"), exist_ok=True)
    mg.save(frames, os.path.join(HERE, "figures", "nullspace_demo.gif"))


if __name__ == "__main__":
    main()
