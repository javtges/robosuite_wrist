"""
HSA-UMI's end-effector camera in the sim: the OV9782 colour global-shutter module on the bracket beside the key, on
the bitting side, looking up the blade at the keyhole (robosuite.models.grippers.key_tool.EECameraConfig has the
geometry and where it comes from).

    cd scripts
    MUJOCO_GL=egl python -m key_insertion.ee_camera --out ../demos/ee_cam_check        # one insertion, frames + mp4
    python -m key_insertion.ee_camera --out ../demos/ee_cam_check --door                # the door task instead
    python -m key_insertion.ee_camera --check                                           # geometry only, no render

Use from code:

    from key_insertion.ee_camera import CAMERA, enable, frame, pixel
    enable()                                   # StewartWristKey gets the measured key mount and the camera
    env = make_env(arm="hsa", robot=DEFAULT_ROBOT, camera_names=[CAMERA])
    obs = env.reset(); img = frame(obs)        # 480 x 640 x 3 uint8, upright
    u, v = pixel(env, env.lock_pose()[0])      # where the keyhole centre lands in the image (col, row)

The image is a pinhole rendering with the fisheye's centre pixel scale (fovy 82 deg); the real lens sees 166 deg
diagonally, so the real frame's corners show more than the sim's. One thing the CAD cannot settle is whether the image
is upright or rolled 180 deg: hold a real frame next to a sim frame once and set EECameraConfig.roll_180.
"""

import argparse
import dataclasses
import os
import sys

if "--check" not in sys.argv:
    os.environ.setdefault("MUJOCO_GL", "egl")  # offscreen rendering, before mujoco is imported

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from hsa_robots import DEFAULT_ROBOT  # noqa: E402
from robosuite.models.grippers import EECameraConfig, KeyConfig, StewartWristKey  # noqa: E402
from robosuite.utils import camera_utils  # noqa: E402

CAMERA = "gripper0_right_ee_cam"  # the camera's name in the model, for the first robot


def enable(camera=None, key=None):
    """
    Put the measured key mount and the end-effector camera on StewartWristKey (class-level, so every env built
    afterwards has them). Returns (previous key_config, previous camera_config) for restore().
    """
    previous = (StewartWristKey.key_config, StewartWristKey.camera_config)
    StewartWristKey.key_config = key or KeyConfig.measured()
    StewartWristKey.camera_config = camera or EECameraConfig()
    return previous


def restore(previous):
    StewartWristKey.key_config, StewartWristKey.camera_config = previous


def frame(obs, camera=CAMERA):
    """The camera's image from an observation, upright (robosuite renders row 0 at the bottom)"""
    return obs[f"{camera}_image"][::-1]


def pixel(env, point_world, camera=CAMERA):
    """
    (col, row) of a world point in the upright image (row 0 at the top, as frame() returns it), or None if the point
    is behind the camera. Not clipped to the image: the keyhole is off-frame for most of a door approach.
    """
    h, w = env.camera_heights[env.camera_names.index(camera)], env.camera_widths[env.camera_names.index(camera)]
    T = camera_utils.get_camera_transform_matrix(env.sim, camera, h, w)  # OpenCV convention: x right, y down
    p = T @ np.append(np.asarray(point_world, dtype=float), 1.0)
    if p[2] <= 0:
        return None
    return float(p[0] / p[2]), float(p[1] / p[2])


def camera_geometry(env, camera=CAMERA):
    """
    Where the camera is relative to the key, measured in the built model (world frame -> key frame): lens offset
    from the key axis, lens behind the tip, and where the optical axis crosses the key axis past the tip (all m).
    """
    sim = env.sim
    cam_id = sim.model.camera_name2id(camera)
    gripper = env.robots[0].gripper["right"]
    tip = sim.data.get_site_xpos(gripper.important_sites["grip_site"])
    key_body = gripper.correct_naming("key")
    R_key = sim.data.get_body_xmat(key_body).reshape(3, 3)
    axis = R_key[:, 2]  # the key's +z, toward the tip
    lens = sim.data.cam_xpos[cam_id]
    optical = -sim.data.cam_xmat[cam_id].reshape(3, 3)[:, 2]  # MuJoCo cameras look along -z
    rel = lens - tip
    behind = -rel @ axis
    offset = np.linalg.norm(rel - (rel @ axis) * axis)
    # the optical axis as a line, where it comes closest to the key axis
    lateral = rel - (rel @ axis) * axis
    s = -(lateral @ optical) / (np.linalg.norm(optical - (optical @ axis) * axis) ** 2 + 1e-12)
    cross = lens + s * optical
    past_tip = (cross - tip) @ axis
    return dict(offset_from_axis=offset, behind_tip=behind, aim_past_tip=past_tip, lens=lens, optical=optical)


class FrameRecorder:
    """on_step hook: saves the camera frames of an episode (every @every control steps) and writes an mp4"""

    def __init__(self, out_dir, name, fps=10, every=2, camera=CAMERA):
        import cv2

        self.cv2, self.out_dir, self.name, self.every, self.camera = cv2, out_dir, name, every, camera
        self.frames, self.n = [], 0
        self.writer = cv2.VideoWriter(os.path.join(out_dir, f"{name}.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), fps, (640, 480))

    def __call__(self, env, oracle, row):
        self.n += 1
        if self.n % self.every:
            return True
        obs = env._get_observations(force_update=False)
        img = frame(obs, self.camera)
        h, w = img.shape[:2]
        if (h, w) != (480, 640):
            img = self.cv2.resize(img, (640, 480))
        bgr = self.cv2.cvtColor(np.ascontiguousarray(img), self.cv2.COLOR_RGB2BGR)
        self.cv2.putText(bgr, f"t {row['t']:5.2f} s  phase {oracle.phase}", (8, 20), self.cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        self.writer.write(bgr)
        self.frames.append((row["t"], oracle.phase, img))
        return True

    def close(self, sheet_phases=6):
        """Writes the mp4 and a contact sheet of the first frame of each phase"""
        self.writer.release()
        if not self.frames:
            return
        picks, seen = [], set()
        for t, phase, img in self.frames:
            if phase not in seen:
                seen.add(phase)
                picks.append((t, phase, img))
        picks = picks[:sheet_phases]
        tiles = []
        for t, phase, img in picks:
            bgr = self.cv2.cvtColor(np.ascontiguousarray(img), self.cv2.COLOR_RGB2BGR)
            self.cv2.putText(bgr, f"phase {phase}  t {t:.1f} s", (8, 20), self.cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            tiles.append(bgr)
        while len(tiles) % 3:
            tiles.append(np.zeros_like(tiles[0]))
        rows = [np.hstack(tiles[i : i + 3]) for i in range(0, len(tiles), 3)]
        self.cv2.imwrite(os.path.join(self.out_dir, f"{self.name}_phases.png"), np.vstack(rows))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", default="../demos/ee_cam_check")
    p.add_argument("--door", action="store_true", help="KeyDoor instead of KeyInsertion")
    p.add_argument("--wrist-z", type=float, default=1.0)
    p.add_argument("--dx", type=float, default=0.0, help="aiming error across the slot (mm)")
    p.add_argument("--dyaw", type=float, default=0.0, help="aiming error in yaw (deg)")
    p.add_argument("--side", type=int, default=1, choices=(1, -1), help="bracket on the bitting side (+x, the real mount) or mirrored")
    p.add_argument("--roll-180", action="store_true")
    p.add_argument("--check", action="store_true", help="print the camera geometry and exit (no rendering)")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    enable(camera=EECameraConfig(side=args.side, roll_180=args.roll_180))
    cams = None if args.check else [CAMERA]
    if args.door:
        from key_insertion.door_oracle import DoorOracle as Oracle, make_door_env, run_door_episode as run
        env = make_door_env(arm="hsa", seed=args.seed, camera_names=cams)
    else:
        from key_insertion.oracle import KeyOracle as Oracle, make_env, run_episode as run
        env = make_env(arm="hsa", robot=DEFAULT_ROBOT, seed=args.seed, camera_names=cams)
    env.reset()
    g = camera_geometry(env)
    cfg = StewartWristKey.camera_config
    print(
        f"camera {CAMERA}: lens {g['offset_from_axis'] * 1e3:.1f} mm off the key axis (CAD {cfg.offset_from_axis * 1e3:.1f}), "
        f"{g['behind_tip'] * 1e3:.1f} mm behind the tip (CAD {cfg.behind_tip * 1e3:.2f}), optical axis crosses the key axis "
        f"{g['aim_past_tip'] * 1e3:.1f} mm past the tip (CAD {cfg.aim_past_tip * 1e3:.1f}); fovy {cfg.fovy} deg, {cfg.width}x{cfg.height}"
    )
    print(f"key: tip {StewartWristKey.key_config.bow_length + StewartWristKey.key_config.blade_length:.4f} m above the plate")
    if args.check:
        return
    os.makedirs(args.out, exist_ok=True)
    oracle = Oracle(env, wrist_z=args.wrist_z, perception_error=(args.dx * 1e-3, 0.0, np.radians(args.dyaw)))
    tip_site = env.robots[0].gripper["right"].important_sites["grip_site"]
    tips, locks = [], []

    rec = FrameRecorder(args.out, "door" if args.door else "insertion")

    def on_step(env, oracle, row):
        tips.append(pixel(env, env.sim.data.get_site_xpos(tip_site)))
        locks.append(pixel(env, env.lock_pose()[0]))
        return rec(env, oracle, row)

    result, log = run(env, oracle, on_step=on_step)
    rec.close()
    tips_a = np.array([t for t in tips if t is not None])
    locks_a = np.array([l for l in locks if l is not None])
    print(f"result {result}, {len(tips)} steps")
    print(f"key tip in the image: ({tips_a[:, 0].mean():.1f}, {tips_a[:, 1].mean():.1f}) px, wander {tips_a.std(0).max():.2f} px (rigid to the camera: should be ~0)")
    print(f"keyhole centre: from ({locks_a[0, 0]:.0f}, {locks_a[0, 1]:.0f}) to ({locks_a[-1, 0]:.0f}, {locks_a[-1, 1]:.0f}) px")
    print(f"wrote {args.out}/{rec.name}.mp4 and {rec.name}_phases.png")


if __name__ == "__main__":
    main()
