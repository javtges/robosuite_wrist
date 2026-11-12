"""
The FR3 arm model and FR3StewartAloha.

    cd scripts && python -m pytest test_fr3.py -q
"""

import os
import re
import sys

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import hsa_robots  # noqa: E402

FR3_URDF = os.path.normpath(os.path.join(HERE, "..", "..", "HSA-UMI", "external", "franka_ros2", "franka_hardware", "test", "fr3.urdf"))


def _env(robot, **kw):
    import robosuite as suite

    return suite.make("Lift", robots=robot, has_renderer=False, has_offscreen_renderer=False, use_camera_obs=False, **kw)


def test_robot_keys():
    assert hsa_robots.robot_name("fr3") == "FR3StewartAloha" and hsa_robots.robot_name("fer") == "PandaStewartAloha"
    assert hsa_robots.robot_name("FR3StewartAloha") == "FR3StewartAloha"
    assert hsa_robots.robot_key("PandaStewartAloha") == "fer" and hsa_robots.robot_key("fr3") == "fr3"
    with pytest.raises(ValueError):
        hsa_robots.robot_name("ur5")


def test_fr3_limits_and_inertials_match_the_urdf():
    """Joint ranges, torque limits and link masses are franka_description's fr3.urdf (when HSA-UMI is alongside)"""
    if not os.path.exists(FR3_URDF):
        pytest.skip(f"{FR3_URDF} not found")
    import mujoco

    env = _env("FR3StewartAloha")
    m = env.sim.model._model
    urdf = open(FR3_URDF).read()
    limits = re.findall(r'<joint name="fr3_joint(\d)" type="revolute">.*?<limit effort="([\d.]+)" lower="([-\d.]+)" upper="([-\d.]+)"', urdf, re.S)
    assert len(limits) == 7
    for j, effort, lo, hi in limits:
        jid = m.joint(f"robot0_joint{j}").id
        assert m.jnt_range[jid] == pytest.approx([float(lo), float(hi)])
        aid = m.actuator(f"robot0_torq_j{j}").id
        assert m.actuator_ctrlrange[aid] == pytest.approx([-float(effort), float(effort)])
    masses = re.findall(r'<link name="fr3_link(\d)">.*?<mass value="([\d.]+)"', urdf, re.S)
    for i, mass in masses:
        assert m.body(f"robot0_link{i}").mass[0] == pytest.approx(float(mass), rel=1e-6)
    # the flange (right_hand) is 107 mm from link 7, as fr3_joint8
    hand = m.body("robot0_right_hand")
    assert hand.pos == pytest.approx([0, 0, 0.107])
    env.close()


def test_fr3_and_panda_share_the_wrist_and_the_start_pose():
    a, b = _env("FR3StewartAloha", initialization_noise=None), _env("PandaStewartAloha", initialization_noise=None)
    oa, ob = a.reset(), b.reset()
    assert a.action_dim == b.action_dim == 13
    # same kinematics and init_qpos, so the grip site starts within the 0.5 mm flange difference
    assert np.linalg.norm(oa["robot0_eef_pos"] - ob["robot0_eef_pos"]) < 1e-3
    assert oa["robot0_gripper_qpos"].shape == ob["robot0_gripper_qpos"].shape == (14,)
    for _ in range(20):
        oa, *_ = a.step(np.zeros(13))
    assert np.isfinite(oa["robot0_eef_pos"]).all()
    a.close()
    b.close()


def test_fr3_teleop_records_and_replays(tmp_path):
    import subprocess

    out = str(tmp_path / "fr3.hdf5")
    teleop = [
        sys.executable, os.path.join(HERE, "teleop_wrist.py"), "--env", "Lift", "--input", "fake_sine", "--no-render",
        "--no-realtime", "--no-terminal-keys", "--robot", "fr3", "--arm-controller", "impedance", "--out", out,
        "--key-script", "0.5:c,2:m,4:s,4.5:q",
    ]  # fmt: skip
    r = subprocess.run(teleop, cwd=HERE, capture_output=True, text=True, timeout=300)
    assert r.returncode == 0, r.stdout + r.stderr
    assert "robot: FR3StewartAloha (fr3)" in r.stdout and "nullspace (0.2, joint1 100.0)" in r.stdout
    r = subprocess.run([sys.executable, os.path.join(HERE, "validate_demos.py"), out], cwd=HERE, capture_output=True, text=True, timeout=300)
    assert r.returncode == 0 and "VALIDATION OK" in r.stdout and "FR3StewartAloha" in r.stdout, r.stdout + r.stderr
