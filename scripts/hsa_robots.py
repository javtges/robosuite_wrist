"""
Which arm carries the wrist, by HSA-UMI's robot keys: `fr3` (Franka Research 3) or `fer` (Franka Emika Robot, the
Panda). The sim robots are FR3StewartAloha and PandaStewartAloha (robosuite/models/robots/compositional.py); they share
the wrist, the gripper, the start pose and the action space, and differ in joint limits, torque limits and inertials.

The default follows HSA-UMI's HSA_ROBOT environment variable, else fr3 (the arm in use from 2026-11).
"""

import os

ROBOTS = {"fr3": "FR3StewartAloha", "fer": "PandaStewartAloha"}
DEFAULT_ROBOT = os.environ.get("HSA_ROBOT", "fr3")


def robot_name(key):
    """robosuite robot name for an HSA-UMI robot key (or a robosuite name, passed through)"""
    if key in ROBOTS:
        return ROBOTS[key]
    if key in ROBOTS.values():
        return key
    raise ValueError(f"robot must be one of {sorted(ROBOTS)} (or {sorted(ROBOTS.values())}), got {key!r}")


def robot_key(name):
    """HSA-UMI robot key for a robosuite robot name"""
    for k, v in ROBOTS.items():
        if name in (k, v):
            return k
    raise ValueError(f"unknown robot {name!r}")
