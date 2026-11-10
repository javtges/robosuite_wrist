"""
Tests of the KeyInsertion env, the key tool and the oracle.

    cd scripts && python -m pytest key_insertion/test_key_insertion.py -q
"""

import os
import re
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from key_insertion.evaluate import FIXED_LOCK  # noqa: E402
from key_insertion.oracle import KeyOracle, make_env, run_episode  # noqa: E402


@pytest.fixture(scope="module")
def env():
    e = make_env(**FIXED_LOCK)
    yield e
    e.close()


def test_key_on_the_wrist(env):
    gripper = env.robots[0].gripper["right"]
    m = env.sim.model._model
    key = m.body(gripper.correct_naming("key")).id
    assert m.body_subtreemass[key] == pytest.approx(0.010, abs=1e-6)  # 10 g key, eef body massless
    assert env.action_dim == 12  # arm pose (6) + wrist (6), no grip
    assert gripper.gripper.dof == 0


def test_keyhole_fits_the_blade(env):
    """The slot (between the wall geoms) is the blade plus the clearance, in both directions"""
    k = env.key_config
    m = env.sim.model
    wall = lambda n: m.geom_name2id(n)  # noqa: E731
    pos, size = m.geom_pos, m.geom_size
    gap_x = (pos[wall("lock_wall_px")][0] - size[wall("lock_wall_px")][0]) - (pos[wall("lock_wall_nx")][0] + size[wall("lock_wall_nx")][0])
    gap_y = (pos[wall("lock_wall_py")][1] - size[wall("lock_wall_py")][1]) - (pos[wall("lock_wall_ny")][1] + size[wall("lock_wall_ny")][1])
    assert gap_x == pytest.approx(k.blade_thickness + env.clearance, abs=1e-6)
    assert gap_y == pytest.approx(k.blade_width + env.clearance, abs=1e-6)


@pytest.mark.parametrize("wrist_z", [-1.0, 0.0, 1.0])
def test_oracle_seats_the_key(env, wrist_z):
    env.rng = np.random.default_rng(0)
    result, log = run_episode(env, KeyOracle(env, wrist_z=wrist_z))
    assert result == "seated"
    assert env._check_success()
    assert log["depth"][-1] == pytest.approx(env.slot_depth, abs=0.0005)  # tip on the plunger at its stop
    # Seated and held still: the flange feels (about) what the lock pushes on the key, and at least the plunger's force
    hold = log["phase"] == KeyOracle.HOLD
    f_flange = np.linalg.norm(log["flange_force"][hold][-1])
    f_key = np.linalg.norm(log["key_force"][hold][-1])
    assert f_flange == pytest.approx(f_key, rel=0.1)
    assert f_key > 0.95 * env.seat_force


def test_softer_wrist_lower_force(env):
    """Pushing the same distance past the seat: the stiff wrist passes more force to the arm than the soft one"""
    peaks = {}
    for z in (-1.0, 1.0):
        env.rng = np.random.default_rng(0)
        _, log = run_episode(env, KeyOracle(env, wrist_z=z))
        peaks[z] = np.linalg.norm(log["flange_force"], axis=1).max()
    assert peaks[-1.0] > peaks[1.0] * 1.5


def test_lock_randomized_and_saved_in_the_model():
    e = make_env()
    try:
        poses = []
        for _ in range(3):
            e.reset()
            poses.append(e.lock_pose()[0].copy())
            # What demos store as model_file, so a replay puts the lock back where it was
            saved = re.search(r'<body name="lock" pos="([^"]+)"', e.sim.model.get_xml()).group(1)
            assert np.allclose(np.fromstring(saved, sep=" "), poses[-1], atol=1e-6)
        assert np.ptp(np.array(poses)[:, :2], axis=0).max() > 1e-3
    finally:
        e.close()
