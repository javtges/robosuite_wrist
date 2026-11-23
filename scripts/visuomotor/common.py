"""
What the visuomotor pipeline's three parts (gen_demos.py, dp/, residual/) agree on: observation groups, action
layout, wrist modes, rotation helpers and how the door env is built. Imported under both venvs (robosuite_env,
Python 3.14, and the SERL venv, Python 3.10), so it must stay plain numpy + robosuite.

Dataset (one hdf5 per lock friction, written by demo_writer.DemoWriter, robomimic layout):
    data.attrs: env_args (robosuite kwargs), ee_camera (camera + key configs for ee_camera.enable), demo_config (json:
                wrist mode and its parameters, the generator's arguments)
    data/demo_i/actions          (N, 12)   robosuite action: tcp pos (3) + rotvec (3), arm base frame, absolute;
                                           wrist pose (6), normalized [-1, 1]: x, y, z, rx, ry, rz (z = index 8)
    data/demo_i/obs/<key>, next_obs/<key>  every KeyDoor observation, plus:
        gripper0_right_ee_cam_image (N, H, W, 3) uint8, upright     the wrist camera
        doorview_image              (N, H, W, 3) uint8, upright     the scene camera
        wrist_wrench                (N, 6) (moment N m, force N)    the wrist's own F/T from its legs
                                                                    (WristGripperModel.leg_wrench, load-compensated,
                                                                    flange frame): what the hardware wrist senses
        wrist_cmd                   (N, 6)                          the wrist command in force when the obs was
                                                                    taken (= the previous action's wrist part)
        phase                       (N,) int                        the expert's phase (DoorOracle.PHASES index)
"""

import json
import os
import sys

import numpy as np
from scipy.spatial.transform import Rotation

SCRIPTS = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if SCRIPTS not in sys.path:
    sys.path.insert(0, SCRIPTS)

# ---- observations ----------------------------------------------------------------------------------------------------
WRIST_CAM = "gripper0_right_ee_cam_image"
SCENE_CAM = "doorview_image"
# The four groups a policy can be given (train_dp.py --obs wrist_cam,scene_cam,wrench,robot). "robot" is the robot's
# own state and last command: tcp pose (position + 6D rotation, computed from robot0_tcp_quat) and the wrist command
# in force (the previous action's wrist part), i.e. what the robot knows about its own actions.
OBS_GROUPS = {
    "wrist_cam": [WRIST_CAM],
    "scene_cam": [SCENE_CAM],
    "wrench": ["wrist_wrench"],
    "robot": ["robot0_tcp_pos", "robot0_tcp_rot6d", "wrist_cmd"],
}
IMAGE_KEYS = (WRIST_CAM, SCENE_CAM)
# Scale the wrench to O(1) before normalization statistics (moment / 2 N m, force / 20 N; as serl_env)
WRENCH_SCALE = np.array([2.0, 2.0, 2.0, 20.0, 20.0, 20.0])

# ---- actions ---------------------------------------------------------------------------------------------------------
ENV_ACTION_DIM = 12
TCP_POS, TCP_ROT, WRIST = slice(0, 3), slice(3, 6), slice(6, 12)
WRIST_Z = 8  # index of the wrist's normalized z in the robosuite action
# Demo-time wrist behaviour (gen_demos.py --wrist-mode)
DEMO_WRIST_MODES = ("static", "znoise", "schedule")
# Policy-time wrist behaviour (train_dp.py --wrist): static at its stiffest, static at home, or the policy's own z
POLICY_WRIST = {"stiff": -1.0, "medium": 0.0, "z": None}
WRIST_Z_RATE = 1.0  # normalized z per second, the expert's switch rate (arm_impedance.wrist_study.RATE)


def rotvec_to_rot6d(rv):
    """(..., 3) rotation vectors -> (..., 6): the first two columns of the rotation matrix (Zhou et al. 2019)"""
    R = Rotation.from_rotvec(np.asarray(rv, dtype=float).reshape(-1, 3)).as_matrix()
    return R[:, :, :2].transpose(0, 2, 1).reshape(*np.shape(rv)[:-1], 6)


def quat_to_rot6d(q):
    """(..., 4) quaternions, robosuite's (x, y, z, w) -> (..., 6)"""
    R = Rotation.from_quat(np.asarray(q, dtype=float).reshape(-1, 4)).as_matrix()
    return R[:, :, :2].transpose(0, 2, 1).reshape(*np.shape(q)[:-1], 6)


def rot6d_to_matrix(r6):
    """(..., 6) -> (..., 3, 3), Gram-Schmidt"""
    r6 = np.asarray(r6, dtype=float).reshape(-1, 6)
    a, b = r6[:, :3], r6[:, 3:]
    x = a / np.linalg.norm(a, axis=1, keepdims=True)
    b = b - (x * b).sum(1, keepdims=True) * x
    y = b / np.linalg.norm(b, axis=1, keepdims=True)
    return np.stack([x, y, np.cross(x, y)], axis=2)


def rot6d_to_rotvec(r6):
    return Rotation.from_matrix(rot6d_to_matrix(r6)).as_rotvec().reshape(*np.shape(r6)[:-1], 3)


def policy_action_dim(wrist):
    """tcp pos (3) + rot6d (6), plus the wrist's z for --wrist z"""
    return 10 if POLICY_WRIST[wrist] is None else 9


def env_to_policy_action(a, wrist):
    """(N, 12) robosuite actions -> (N, 9 or 10) policy targets"""
    a = np.asarray(a, dtype=float)
    out = [a[:, TCP_POS], rotvec_to_rot6d(a[:, TCP_ROT])]
    if POLICY_WRIST[wrist] is None:
        out.append(a[:, WRIST_Z:WRIST_Z + 1])
    return np.concatenate(out, axis=1)


class WristCommand:
    """
    Turns policy outputs into the robosuite action's wrist part: a fixed z for the static modes, the policy's z
    rate-limited at the expert's 1 z/s for --wrist z. x, y and rotations stay 0 (the residual adds to them).
    """

    def __init__(self, wrist, control_freq=20.0, z0=None):
        self.fixed = POLICY_WRIST[wrist]
        self.max_step = WRIST_Z_RATE / control_freq
        self.z = self.fixed if self.fixed is not None else (0.0 if z0 is None else z0)

    def __call__(self, z_target=None):
        if self.fixed is None and z_target is not None:
            self.z += float(np.clip(z_target - self.z, -self.max_step, self.max_step))
        w = np.zeros(6)
        w[2] = self.z
        return w


def policy_to_env_action(p, wrist_cmd):
    """one policy action (9 or 10) + the 6 wrist command -> the 12-dim robosuite action"""
    p = np.asarray(p, dtype=float)
    return np.concatenate([p[:3], rot6d_to_rotvec(p[3:9]), wrist_cmd])


# ---- env ---------------------------------------------------------------------------------------------------------------
def enable_ee_camera(ee_camera_attrs):
    """Put the end-effector camera (and the key mount it was recorded with) on the wrist, from a dataset's
    data.attrs["ee_camera"] json (None: the default camera and measured key). Before any env is built."""
    from robosuite.models.grippers import EECameraConfig, KeyConfig

    from key_insertion import ee_camera

    if ee_camera_attrs is None:
        ee_camera.enable()
    else:
        cam = json.loads(ee_camera_attrs) if isinstance(ee_camera_attrs, str) else ee_camera_attrs
        ee_camera.enable(camera=EECameraConfig(**cam["camera"]), key=KeyConfig(**cam["key"]))


def make_env(env_args, ee_camera_attrs=None, cameras=True, **overrides):
    """
    KeyDoor exactly as a dataset was recorded (its env_args json or dict), the end-effector camera enabled.
    @cameras False: no rendering (state-only rollouts, e.g. SERL without images). @overrides: env_kwargs changes,
    e.g. cylinder_friction=0.4 to evaluate on the stiff lock.
    """
    import robosuite as suite

    env_args = json.loads(env_args) if isinstance(env_args, str) else env_args
    enable_ee_camera(ee_camera_attrs)
    kw = dict(env_args["env_kwargs"], **overrides)
    if not cameras:
        kw.update(use_camera_obs=False, has_offscreen_renderer=False)
    return suite.make(env_args["env_name"], **kw)


def default_env_args(cylinder_friction=0.15, cameras=("wrist", "doorview"), camera_size=(120, 160), robot=None, control_freq=20.0):
    """env_args as gen_demos.py writes them, for building an env without a dataset"""
    from hsa_robots import DEFAULT_ROBOT
    from key_insertion.door_demos import make_env_args

    return make_env_args(robot or DEFAULT_ROBOT, control_freq, cylinder_friction, list(cameras), list(camera_size))


def obs_extras(env, obs, wrist_cmd, phase=-1):
    """The keys the dataset adds to KeyDoor's observations (see the module docstring); images flipped upright"""
    out = {k: (v[::-1].copy() if k.endswith("_image") else v) for k, v in obs.items()}
    gripper = env.robots[0].gripper["right"]
    out["wrist_wrench"] = np.asarray(gripper.leg_wrench(env.sim)[1], dtype=np.float32)
    out["wrist_cmd"] = np.asarray(wrist_cmd, dtype=np.float32)
    out["phase"] = np.int64(phase)
    return out


def policy_obs(obs, keys):
    """One step's observation dict (obs_extras' output) -> the policy's inputs: images HWC uint8, low-dim float32"""
    out = {}
    for k in keys:
        if k == "robot0_tcp_rot6d":
            out[k] = quat_to_rot6d(obs["robot0_tcp_quat"]).astype(np.float32)
        elif k == "wrist_wrench":
            out[k] = (np.asarray(obs[k]) / WRENCH_SCALE).astype(np.float32)
        elif k in IMAGE_KEYS:
            out[k] = np.asarray(obs[k], dtype=np.uint8)
        else:
            out[k] = np.asarray(obs[k], dtype=np.float32)
    return out


def obs_keys(groups, wrist=None):
    """
    'wrist_cam,scene_cam,wrench,robot' (or a list) -> the observation keys, in a fixed order. With a static policy
    wrist (@wrist stiff / medium) the robot group leaves out wrist_cmd: at deployment it is a constant, so a policy
    trained on znoise / schedule data would see a z it never sees in training.
    """
    groups = groups.split(",") if isinstance(groups, str) else list(groups)
    unknown = [g for g in groups if g not in OBS_GROUPS]
    assert not unknown, f"unknown observation groups {unknown}; choose from {list(OBS_GROUPS)}"
    keys = [k for g in OBS_GROUPS for k in OBS_GROUPS[g] if g in groups]
    if wrist is not None and POLICY_WRIST[wrist] is not None:
        keys = [k for k in keys if k != "wrist_cmd"]
    return keys
