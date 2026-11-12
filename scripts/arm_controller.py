"""
Arm controller selection for the teleop and the scripts: robosuite's OSC_POSE, or the Cartesian impedance law that
HSA-UMI runs on the real arm (franka_streaming_impedance_controller, a port of serl_franka_controllers), with its gains.

    cfg = load_composite_controller_config(robot=ROBOT)
    apply_arm_controller(cfg, "impedance")                         # HSA gains below
    apply_arm_controller(cfg, "impedance", gains_yaml=PATH)        # gains read from HSA-UMI's hsa_controllers.yaml

The gains are the deployment values in HSA-UMI/nuc/config/hsa_controllers.yaml (which repeats and explains them); that
file is the source of truth, and test_arm_controller.py checks this copy against it when the HSA-UMI checkout is next
to this repo. Pass --arm-gains-yaml to read it directly instead.

What each number does (the law is tau = J^T (-K clip(e) - D xdot) + nullspace + coriolis):
    translational_stiffness / rotational_stiffness   K: N/m and N m/rad, the spring between the target and the tcp
    translational_damping / rotational_damping       D: constant, 2 sqrt(K) is critical (89 for 2000 N/m); rotation
                                                     is run under-damped on purpose
    translational_clip / rotational_clip             the error K sees is clipped here, so K * clip is the force ceiling
                                                     (2000 N/m * 0.01 m = 20 N)
    translational_ki / rotational_ki                 integral gain, off
    nullspace_stiffness / joint1_nullspace_stiffness pull toward the posture at activation, through a damped projector
"""

import re

HSA_GAINS_SOURCE = "HSA-UMI/nuc/config/hsa_controllers.yaml"

HSA_GAINS = dict(
    translational_stiffness=2000.0,
    translational_damping=89.0,
    rotational_stiffness=150.0,
    rotational_damping=7.0,
    translational_clip=0.01,
    rotational_clip=0.05,
    translational_ki=0.0,
    rotational_ki=0.0,
    nullspace_stiffness=1.0,
    joint1_nullspace_stiffness=10.0,
)

# franka_streaming_impedance_controller constants that aren't parameters
NULLSPACE_LAMBDA = 0.2  # dampedPseudoInverse's default
TORQUE_RATE_LIMIT = 1000.0  # kMaxTorqueRate = 1 N m per 1 ms step

ARM_CONTROLLERS = ("osc", "impedance")


def read_gains_yaml(path):
    """
    The scalar gains of a ros2_control parameter file (`key: value` lines under ros__parameters), as floats, for the
    keys in HSA_GAINS. No yaml dependency: the file is flat where it matters.
    """
    found = {}
    pattern = re.compile(r"^\s*(" + "|".join(HSA_GAINS) + r")\s*:\s*([-+0-9.eE]+)")
    with open(path) as f:
        for line in f:
            m = pattern.match(line)
            if m:
                found[m.group(1)] = float(m.group(2))
    missing = sorted(set(HSA_GAINS) - set(found))
    if missing:
        raise ValueError(f"{path}: no value for {missing}")
    return found


def impedance_config(gains=None):
    """CARTESIAN_IMPEDANCE part-controller keys for @gains (HSA_GAINS by default)"""
    g = dict(HSA_GAINS, **(gains or {}))
    return dict(
        type="CARTESIAN_IMPEDANCE",
        kp=[g["translational_stiffness"]] * 3 + [g["rotational_stiffness"]] * 3,
        damping_ratio=1.0,  # unused when `damping` is set; kept valid for the OSC base class
        damping=[g["translational_damping"], g["rotational_damping"]],
        translational_clip=g["translational_clip"],
        rotational_clip=g["rotational_clip"],
        ki=[g["translational_ki"], g["rotational_ki"]],
        nullspace="projector",
        nullspace_stiffness=g["nullspace_stiffness"],
        joint1_nullspace_stiffness=g["joint1_nullspace_stiffness"],
        nullspace_lambda=NULLSPACE_LAMBDA,
        torque_rate_limit=TORQUE_RATE_LIMIT,
    )


def apply_arm_controller(cfg, name, gains_yaml=None, part="right"):
    """
    Sets the arm part of a composite controller config to @name: "osc" leaves robosuite's OSC_POSE (kp 150, Lambda-
    shaped stiffness), "impedance" installs the HSA Cartesian impedance law with HSA_GAINS, or the gains of
    @gains_yaml. Returns cfg.
    """
    if name not in ARM_CONTROLLERS:
        raise ValueError(f"arm controller must be one of {ARM_CONTROLLERS}, got {name!r}")
    if name == "impedance":
        gains = read_gains_yaml(gains_yaml) if gains_yaml else None
        cfg["body_parts"][part].update(impedance_config(gains))
    return cfg


def describe(cfg, part="right"):
    """One line for the startup banner"""
    c = cfg["body_parts"][part]
    if c["type"] == "CARTESIAN_IMPEDANCE":
        k, d, clip = c["kp"], c["damping"], (c.get("translational_clip"), c.get("rotational_clip"))
        force = k[0] * clip[0] if clip[0] is not None else None
        return (
            f"arm: CARTESIAN_IMPEDANCE K ({k[0]:.0f} N/m, {k[3]:.0f} N m/rad) D ({d[0]:.0f}, {d[1]:.0f}) "
            f"clip ({clip[0]} m, {clip[1]} rad)" + (f" -> {force:.0f} N ceiling" if force else "")
        )
    return f"arm: {c['type']} kp {c['kp']} (stiffness kp * Lambda, no force ceiling)"
