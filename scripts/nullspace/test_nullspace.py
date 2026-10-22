"""
Tests for the nullspace teleop mode. Run from scripts/:  python -m pytest nullspace/test_nullspace.py -q
"""

import os
import sys
import tempfile
import warnings

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from nullspace.controller import NullspaceCompensator, T_to_pose6, make_T, pose6_to_T, pose_error  # noqa: E402

LIMITS = np.array([0.01, 0.01, 0.02] + [np.radians(10)] * 3)
RNG = np.random.default_rng(0)


def _compensator():
    T_S_W = make_T(Rotation.from_euler("xyz", [3, -2, 40], degrees=True).as_matrix(), [0.001, -0.002, -0.126])
    return NullspaceCompensator(home_pos=[0, 0, 0.126], T_S_W=T_S_W, limits=LIMITS)


def _random_pose6():
    return np.concatenate([RNG.uniform(-0.5, 0.5, 3), Rotation.random(random_state=RNG.integers(1e6)).as_rotvec()])


def test_arm_target_keeps_the_plate_on_the_anchor():
    ns = _compensator()
    ns.anchor(_random_pose6(), RNG.uniform(-1, 1, 6) * LIMITS)
    for _ in range(50):
        p = RNG.uniform(-1, 1, 6) * LIMITS
        e = pose_error(ns.plate_pose(pose6_to_T(ns.arm_target(p)), p), ns.T_B_P0)
        assert e[0] < 1e-12 and e[1] < 1e-9


def test_feedback_inverts_feedforward():
    """If the arm is exactly where feedforward put it, feedback commands exactly the wrist target"""
    ns = _compensator()
    ns.anchor(_random_pose6(), np.zeros(6))
    for _ in range(50):
        p = RNG.uniform(-1, 1, 6) * LIMITS
        assert np.allclose(ns.wrist_command(ns.arm_target(p)), p, atol=1e-10)


def test_feedback_cancels_arm_error_and_clips():
    ns = _compensator()
    ns.anchor(_random_pose6(), np.zeros(6))
    arm = pose6_to_T(ns.arm_target(np.zeros(6)))
    lagging = arm @ make_T(Rotation.from_euler("z", 2, degrees=True).as_matrix(), [0.003, -0.002, 0.001])
    p = ns.wrist_command(T_to_pose6(lagging))
    e = pose_error(ns.plate_pose(lagging, p), ns.T_B_P0)
    assert e[0] < 1e-12 and e[1] < 1e-9  # small arm error: fully cancelled
    far = arm @ make_T(np.eye(3), [0.05, 0, 0])  # 50 mm: beyond the wrist's range
    assert np.all(np.abs(ns.wrist_command(T_to_pose6(far))) <= LIMITS + 1e-12)


@pytest.fixture(scope="module")
def teleop():
    warnings.filterwarnings("ignore")
    import make_teleop_gifs as mg

    t = mg.make_teleop(os.path.join(tempfile.mkdtemp(), "unused.hdf5"))
    yield t
    t.close()


def test_plate_model_matches_sim(teleop):
    t = teleop
    ns, d = t.nullspace, t.env.sim.data
    base, top = t.gripper.worldbody[0].get("name"), t.gripper.correct_naming("wrist_top")
    for _ in range(3):
        t.wrist_target = RNG.uniform(-1, 1, 6) * LIMITS
        for _ in range(10):
            t.tick()
        t.env.sim.forward()  # after mj_step, body poses lag the integrated qpos by one step
        T_W = make_T(d.get_body_xmat(base), d.get_body_xpos(base))
        T_P = make_T(d.get_body_xmat(top), d.get_body_xpos(top))
        e = pose_error(np.linalg.inv(T_W) @ T_P, ns.T_W_P(t.measured_wrist_pose()))
        assert e[0] < 1e-9 and e[1] < 1e-9


def test_mode_cycle(teleop):
    import teleop_wrist as tw

    t = teleop
    t.mode = tw.ARM
    seen = []
    for _ in range(3):
        t.handle_key("m")
        seen.append(t.mode)
    assert seen == [tw.WRIST, tw.NULLSPACE, tw.ARM]


@pytest.mark.parametrize("comp, settled_mm, final_mm", [("feedback", 1.0, 0.1), ("feedforward", 2.0, 1.0)])
def test_plate_stays_put(teleop, comp, settled_mm, final_mm):
    """Closed loop through the real teleop: push the wrist to +x and +roll limits and back, plate barely moves"""
    import teleop_wrist as tw

    t = teleop
    t.args.nullspace_comp = comp
    t.mode = tw.ARM
    t.reset_scene()
    for _ in range(40):
        t.tick()
    while t.mode != tw.NULLSPACE:
        t.toggle_mode()
    top = t.gripper.correct_naming("wrist_top")
    d = t.env.sim.data
    plate = lambda: make_T(d.get_body_xmat(top), d.get_body_xpos(top))
    P0 = plate()
    for axis, dur in ((0, 0.25), (3, 0.5)):
        v = np.zeros(6)
        v[axis] = 1
        t.input.push(v, left=axis >= 3)
        for _ in range(int(dur * t.hz)):
            t.tick()
        t.input.release()
        for _ in range(30):
            t.tick()
        assert np.abs(t.measured_wrist_pose()[:6]).max() > 0.009  # the wrist really moved
        assert pose_error(P0, plate())[0] * 1e3 < settled_mm
        v[axis] = -1
        t.input.push(v, left=axis >= 3)
        for _ in range(int(dur * t.hz)):
            t.tick()
        t.input.release()
        for _ in range(30):
            t.tick()
    e = pose_error(P0, plate())
    assert e[0] * 1e3 < final_mm and np.degrees(e[1]) < 0.25
