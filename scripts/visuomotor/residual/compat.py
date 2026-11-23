"""
Compatibility shims for running the door env in the SERL venv (Python 3.10, SciPy 1.11). Import before building an env.

SciPy < 1.12's Rotation has no integer power, which the robot's arm law (cartesian_impedance.extrapolate_chunk, the
streaming "chunks" reference) uses. The patch replaces that function with the same math, delta^k as the rotation
vector scaled by k (equal for |delta| < pi, which a 50 ms tick always is). Robosuite itself is not modified.
"""

import numpy as np
from scipy.spatial.transform import Rotation


def _has_int_pow():
    try:
        Rotation.identity() ** 2
        return True
    except TypeError:
        return False


def extrapolate_chunk(target, prev_target, n):
    """cartesian_impedance.extrapolate_chunk without Rotation ** int"""
    target = np.asarray(target, dtype=float)
    if n <= 1 or prev_target is None:
        return [target.copy() for _ in range(max(1, n))]
    prev = np.asarray(prev_target, dtype=float)
    delta_pos = target[:3] - prev[:3]
    rot_t = Rotation.from_quat(target[3:])
    delta_rv = (rot_t * Rotation.from_quat(prev[3:]).inv()).as_rotvec()
    out = []
    for k in range(n):
        pose = np.empty(7)
        pose[:3] = target[:3] + k * delta_pos
        pose[3:] = (Rotation.from_rotvec(k * delta_rv) * rot_t).as_quat() if k else target[3:]
        out.append(pose)
    return out


def apply():
    if _has_int_pow():
        return
    from robosuite.controllers.parts.arm import cartesian_impedance

    cartesian_impedance.extrapolate_chunk = extrapolate_chunk


apply()
