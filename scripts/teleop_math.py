"""
The pure arithmetic of SpaceMouse / keyboard teleoperation, ported from HSA-UMI's
ros2_ws/src/hsa_ros2/hsa_ros2/teleop_math.py and extended to the Stewart wrist.

Numpy in, numpy out: no robosuite, no device, so it is unit-testable (test_teleop.py).

Arm pose convention (as in HSA-UMI): a 6-vector [x, y, z, rx, ry, rz] in meters and a rotation vector, in the robot
base frame. That is also what the OSC_POSE controller takes with input_type="absolute", input_ref_frame="base".

Wrist pose convention (as StewartPlatformWrist): [x, y, z, roll, pitch, yaw] of the top plate relative to home, in
the wrist base frame (m, rad), with R = Rz(yaw) Ry(pitch) Rx(roll).
"""

import numpy as np
from scipy.spatial.transform import Rotation

# Maps SpaceMouse axes (pyspacemouse's x, y, z) onto the robot base frame: robot = [-z, x, y] of the device.
# Same matrix as HSA-UMI, applied to translation and rotation alike.
SPACEMOUSE_TO_ROBOT = np.array(
    [
        [0.0, 0.0, -1.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
    ]
)


# Alternative ("zup"): pyspacemouse's axes are z-up (x right, y forward, z up), while HSA-UMI's matrix is UMI's
# tx_zup_spnav, written for libspacenav's y-up axes. This one maps puck forward -> robot +x, left -> +y, up -> +z, for
# an operator standing behind the robot base and facing along +x; rotations use the same axes (pyspacemouse's pitch is
# about x and roll about y).
ZUP_TO_ROBOT = np.array(
    [
        [0.0, 1.0, 0.0],
        [-1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0],
    ]
)


def map_spacemouse_axes(raw, deadzone=0.0, mapping="hsa"):
    """
    Turn a raw SpaceMouse reading into normalized velocities in the robot base frame.

    Args:
        raw (6-array): [x, y, z, roll, pitch, yaw] as pyspacemouse reports them, each in [-1, 1]
        deadzone (float): readings with |value| below this are zeroed, so a resting puck sends nothing
        mapping (str): "hsa" (HSA-UMI's, exactly) or "zup" (see ZUP_TO_ROBOT)

    Returns:
        6-array: [vx, vy, vz, wx, wy, wz] in [-1, 1], robot base axes
    """
    raw = np.asarray(raw, dtype=np.float64).copy()
    raw[np.abs(raw) < deadzone] = 0.0
    if mapping == "hsa":
        return np.concatenate([SPACEMOUSE_TO_ROBOT @ raw[:3], SPACEMOUSE_TO_ROBOT @ raw[3:]])
    if mapping == "zup":
        return np.concatenate([ZUP_TO_ROBOT @ raw[:3], ZUP_TO_ROBOT @ raw[[4, 3, 5]]])
    raise ValueError(f"Unknown SpaceMouse mapping {mapping}")


def _step(axes, max_pos_speed, max_rot_speed, hz, rotation_mode, z_enabled, exclusive):
    """Per-tick displacement (dpos, drot) in the robot base frame, with HSA-UMI's button modes"""
    dpos = np.asarray(axes[:3], dtype=np.float64) * (max_pos_speed / hz)
    drot = np.asarray(axes[3:], dtype=np.float64) * (max_rot_speed / hz)
    if exclusive:
        if rotation_mode:
            dpos[:] = 0.0
        else:
            drot[:] = 0.0
    if not z_enabled:
        dpos[2] = 0.0
    return dpos, drot


def integrate_target(target, axes, *, max_pos_speed, max_rot_speed, hz, rotation_mode, z_enabled, exclusive=True):
    """
    Advance the commanded arm target pose by one control tick.

    The SpaceMouse is a velocity device: full deflection moves the target by max_speed / hz per tick.
    With @exclusive (the SpaceMouse), @rotation_mode (left button held) rotates the tool and locks translation;
    otherwise translation only. @z_enabled (right button held) allows vertical motion; otherwise z is locked so an
    accidental push does not drive the tool into the table. The keyboard passes exclusive=False, z_enabled=True:
    it has dedicated keys for every axis.

    Args:
        target (6-array): current [x, y, z, rx, ry, rz] target
        axes (6-array): normalized velocities in the robot base frame (output of map_spacemouse_axes)

    Returns:
        6-array: the new target (a new array; the input is not modified)
    """
    target = np.asarray(target, dtype=np.float64).copy()
    dpos, drot = _step(axes, max_pos_speed, max_rot_speed, hz, rotation_mode, z_enabled, exclusive)
    target[:3] += dpos
    # Compose the small rotation ON THE LEFT: rotate about the base-frame axes the puck names, rather than about the
    # tool's own axes
    delta = Rotation.from_euler("xyz", drot)
    target[3:] = (delta * Rotation.from_rotvec(target[3:])).as_rotvec()
    return target


def clamp_workspace(target, box):
    """
    Clip the target position into an axis-aligned box in the base frame.

    Args:
        box (dict): {"x_min", "x_max", "y_min", "y_max", "z_min", "z_max"} in meters
    """
    target = np.asarray(target, dtype=np.float64).copy()
    for i, a in enumerate("xyz"):
        target[i] = np.clip(target[i], box[f"{a}_min"], box[f"{a}_max"])
    return target


def rpy_to_matrix(rpy):
    """R = Rz(yaw) Ry(pitch) Rx(roll), the convention of StewartPlatformWrist"""
    return Rotation.from_euler("ZYX", np.asarray(rpy)[::-1]).as_matrix()


def matrix_to_rpy(R):
    return Rotation.from_matrix(R).as_euler("ZYX")[::-1]


def integrate_wrist_target(
    pose, axes, R_wrist, limits, *, max_pos_speed, max_rot_speed, hz, rotation_mode, z_enabled, exclusive=True
):
    """
    Advance the commanded wrist pose by one control tick, from the same normalized robot-frame velocities and button
    modes as integrate_target, so the puck moves the tool the same way whether it drives the arm or the wrist.

    Args:
        pose (6-array): current wrist target [x, y, z, roll, pitch, yaw], wrist base frame
        axes (6-array): normalized velocities in the robot base frame
        R_wrist (3x3): orientation of the wrist base frame in the robot base frame
        limits (6-array): workspace half-ranges (m, rad); each axis is clipped to +-limit

    Returns:
        6-array: the new wrist target
    """
    pose = np.asarray(pose, dtype=np.float64).copy()
    dpos, drot = _step(axes, max_pos_speed, max_rot_speed, hz, rotation_mode, z_enabled, exclusive)
    pose[:3] += R_wrist.T @ dpos
    # Rotation about the robot-base axes, expressed in the wrist base frame, composed on the left
    delta = R_wrist.T @ Rotation.from_euler("xyz", drot).as_matrix() @ R_wrist
    pose[3:] = matrix_to_rpy(delta @ rpy_to_matrix(pose[3:]))
    return np.clip(pose, -np.asarray(limits), np.asarray(limits))


def fake_sine_axes(t, period_s=6.0, amplitude=0.4):
    """
    Stand in for the SpaceMouse with a slow circle in the base x-y plane, no rotation. Amplitude is a fraction of
    max speed.
    """
    w = 2.0 * np.pi / period_s
    return np.array([amplitude * np.cos(w * t), amplitude * np.sin(w * t), 0.0, 0.0, 0.0, 0.0])
