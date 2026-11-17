"""
The end-effector camera on the key: geometry against the CAD numbers, rendering, and the key tip fixed in the image.

    cd scripts && MUJOCO_GL=egl python -m pytest key_insertion/test_ee_camera.py -q      # ~1 min
"""

import math
import os
import sys

os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from robosuite.models.grippers import EECameraConfig, KeyConfig  # noqa: E402

from key_insertion import ee_camera  # noqa: E402
from hsa_robots import DEFAULT_ROBOT  # noqa: E402
from key_insertion.oracle import KeyOracle, make_env, run_episode  # noqa: E402

FIXED = dict(lock_x_range=(0, 0), lock_y_range=(0, 0), lock_yaw_range=(0, 0))


@pytest.fixture(scope="module")
def env():
    previous = ee_camera.enable()
    env = make_env(arm="hsa", robot=DEFAULT_ROBOT, camera_names=[ee_camera.CAMERA], **FIXED)
    yield env
    env.close()
    ee_camera.restore(previous)


def test_config_numbers():
    """The CAD's 25 deg tilt, 35.5 mm offset and 54.1 mm setback put the optical axis on the key axis 22 mm past the tip"""
    cam = EECameraConfig()
    assert abs(cam.aim_past_tip - 0.022) < 1e-3
    assert abs(KeyConfig.measured().bow_length + KeyConfig.measured().blade_length - 0.080) < 1e-6
    pos, quat = cam.pose_in_key_frame(0.080)
    assert np.allclose(pos, [0.0355, 0.0, 0.080 - 0.05413])
    assert abs(np.linalg.norm(quat) - 1) < 1e-9
    # the mirrored bracket is the same camera on the other side
    pos_m, _ = EECameraConfig(side=-1).pose_in_key_frame(0.080)
    assert np.allclose(pos_m, [-0.0355, 0.0, pos[2]])


def test_camera_geometry_in_model(env):
    env.reset()
    g = ee_camera.camera_geometry(env)
    cam = EECameraConfig()
    assert abs(g["offset_from_axis"] - cam.offset_from_axis) < 5e-4
    assert abs(g["behind_tip"] - cam.behind_tip) < 5e-4
    assert abs(g["aim_past_tip"] - cam.aim_past_tip) < 1e-3
    # 25 deg between the optical axis and the key axis
    key = env.robots[0].gripper["right"].correct_naming("key")
    axis = env.sim.data.get_body_xmat(key).reshape(3, 3)[:, 2]
    assert abs(math.degrees(math.acos(np.clip(g["optical"] @ axis, -1, 1))) - cam.tilt_deg) < 0.5


def test_render_and_tip_fixed(env):
    """A real image comes out, and the key tip (rigid to the camera) stays on the same pixel through an insertion"""
    obs = env.reset()
    img = ee_camera.frame(obs)
    assert img.shape == (480, 640, 3) and img.dtype == np.uint8
    assert img.std() > 10, "blank frame"
    tip_site = env.robots[0].gripper["right"].important_sites["grip_site"]
    rows = []

    def on_step(env, oracle, row):
        rows.append((oracle.phase, ee_camera.pixel(env, env.sim.data.get_site_xpos(tip_site)), ee_camera.pixel(env, env.lock_pose()[0])))
        return True

    oracle = KeyOracle(env, wrist_z=1.0)
    result, log = run_episode(env, oracle, on_step=on_step, reset=False)
    assert result == "seated"
    tips = np.array([t for _, t, _ in rows])
    assert tips.std(0).max() < 1.0, "the tip moved in the image although the camera is rigid to it"
    assert (0 <= tips[:, 0]).all() and (tips[:, 0] < 640).all() and (0 <= tips[:, 1]).all() and (tips[:, 1] < 480).all()
    # The keyhole centre (the lock's top face) is in view from the start of the approach, and once the key is seated it
    # sits 30 mm back from the tip along the blade: on the blade's line in the image, between the jaws and the tip
    # (the camera, 35.5 mm off the +x side, sees the blade run left to right with the tip at column ~280).
    approach = [l for ph, _, l in rows if ph == KeyOracle.APPROACH]
    hold = [l for ph, _, l in rows if ph == KeyOracle.HOLD]
    assert approach and hold and all(l is not None for l in approach + hold)
    assert 0 <= approach[0][0] < 640 and 0 <= approach[0][1] < 480
    u_lock, v_lock = hold[-1]
    u_tip, v_tip = tips[-1]
    assert abs(v_lock - v_tip) < 5, "the seated keyhole left the blade's line"
    assert 100 < u_lock < u_tip - 80, "the seated keyhole should be ~125 px behind the tip"
