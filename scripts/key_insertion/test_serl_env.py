"""
Tests of the SERL setup: the Cartesian impedance arm controller, the latch, and KeyInsertionWrist-v0 with its oracle.

    cd scripts && python -m pytest key_insertion/test_serl_env.py -q
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from robosuite import macros  # noqa: E402

import key_insertion.serl_env as se  # noqa: E402
from key_insertion.evaluate import FIXED_LOCK  # noqa: E402
from key_insertion.oracle import IMPEDANCE_KP, make_env  # noqa: E402


def hold_action(env, z=-1.0):
    """Holds the arm's tcp where it is, wrist at @z"""
    from scipy.spatial.transform import Rotation

    robot = env.robots[0]
    c = robot.part_controllers["right"]
    p = c.origin_ori.T @ (c.ref_pos - c.origin_pos)
    r = Rotation.from_matrix(c.origin_ori.T @ c.ref_ori_mat).as_rotvec()
    return robot.create_action_vector({"right": np.concatenate([p, r]), "right_gripper": np.array([0, 0, z, 0, 0, 0.0])})


def test_cartesian_impedance_is_diagonal():
    """10 N on the flange body moves the tcp 10 / 20 kN/m along that axis only (OSC: ~2 kN/m and coupled)"""
    env = make_env(arm="impedance", **FIXED_LOCK)
    env.reset()
    c = env.robots[0].part_controllers["right"]
    a = hold_action(env)
    for _ in range(20):
        env.step(a)
    d, m = env.sim.data._data, env.sim.model._model
    body = m.site_bodyid[m.site(c.ref_name).id]
    p0 = c.ref_pos.copy()
    C = np.zeros((3, 3))
    for i in range(3):
        d.xfrc_applied[body, i] = 10.0
        for _ in range(30):
            env.step(a)
        C[:, i] = (c.ref_pos - p0) / 10.0
        d.xfrc_applied[body, :3] = 0.0
        for _ in range(30):
            env.step(a)
    env.close()
    assert 1 / np.diag(C) == pytest.approx(IMPEDANCE_KP[:3], rel=0.1)
    off = np.abs(C - np.diag(np.diag(C))).max() / np.diag(C).min()
    assert off < 0.1


def test_latch_profile():
    env = make_env(arm="impedance", **FIXED_LOCK, **{k: v for k, v in se.LOCK.items() if not k.startswith("lock_")})
    env.reset()
    q = env.plunger_qpos_adr
    for x, f in se.LOCK["latch_profile"]:
        env.sim.data.qpos[q] = -x
        assert env.plunger_force() == pytest.approx(f)
    env.sim.data.qpos[q] = -0.003  # on the falling edge: 40 N at 2.5 mm down to 8 N at 3.5 mm
    assert env.plunger_force() == pytest.approx(24.0)
    env.close()
    # Without a profile: the linear spring, as before
    env = make_env(**FIXED_LOCK)
    env.reset()
    env.sim.data.qpos[env.plunger_qpos_adr] = -env.plunger_travel / 2
    assert env.plunger_force() == pytest.approx((env.seat_preload + env.seat_force) / 2)
    env.close()


@pytest.fixture(scope="module")
def variable_env():
    e = se.KeyInsertionSERL(mode="variable", seed=0)
    yield e
    e.close()


def test_spaces(variable_env):
    obs, _ = variable_env.reset(seed=0)
    assert variable_env.action_space.shape == (5,)
    assert obs.shape == variable_env.observation_space.shape == (22,)
    assert obs.dtype == np.float32 and np.all(np.isfinite(obs))


def test_stiffness_change_does_not_move_the_key(variable_env):
    """In free space, sweeping the wrist from stiff to soft and back leaves the key's tip where it was (the wrist moves
    it up to 40 mm, the arm cancels that)"""
    env = variable_env
    env.reset(seed=0)
    tip = lambda: env.env.key_tip_in_lock()[0]  # noqa: E731
    p0 = tip()
    worst = 0.0
    for z in (-1.0, 1.0, -1.0, 0.5, -0.5):
        for _ in range(25):
            env.step(np.array([0, 0, 0, 0, z]))
            worst = max(worst, np.abs(tip() - p0).max())
        assert env.z == pytest.approx(z)
        assert tip() == pytest.approx(p0, abs=5e-4)  # of the up to 40 mm the arm cancels
    assert worst < 3e-3  # while the soft wrist moves (it lags the arm a little)


def test_stiffening_under_load_keeps_the_force(variable_env):
    """Soft wrist pressing ~15 N into the latch, then made stiff without moving: the push stays about the same (without
    the compensation it climbs past the latch's 40 N and seats the key)"""
    env = variable_env
    env.reset(seed=0)
    push = lambda: -env.gripper.flange_wrench(env.env.sim)[1][5]  # noqa: E731
    for _ in range(30):
        t = env.target
        env.step(np.clip([-t[0] / se.POS_SCALE, -t[1] / se.POS_SCALE, 0, -t[3] / se.YAW_SCALE, 1.0], -1, 1))
    while push() < 15.0 and env.target[2] > -0.0349:
        env.step(np.array([0, 0, -0.3, 0, 1.0]))
    for _ in range(10):
        env.step(np.array([0, 0, 0, 0, 1.0]))
    before = push()
    peak = 0.0
    for _ in range(35):
        _, _, _, _, info = env.step(np.array([0, 0, 0, 0, -1.0]))
        peak = max(peak, info["max_force"])
    assert env.z == -1.0
    assert 10.0 < before < 20.0
    assert 0.5 * before < push() < 2.0 * before
    assert peak < 35.0
    assert not env.env._check_success()


def run_oracle(mode, seed=0, **policy_kwargs):
    """Returns (success, steps, peak flange force, last info)"""
    env = se.KeyInsertionSERL(mode=mode, seed=seed)
    policy = se.ScriptedPolicy(env, seed=seed, **policy_kwargs)
    obs, _ = env.reset()
    done = truncated = False
    peak = 0.0
    while not (done or truncated):
        obs, reward, done, truncated, info = env.step(policy(obs))
        peak = max(peak, info["max_force"])
    env.close()
    return float(info["succeed"]), env.steps, peak, info


@pytest.mark.parametrize("mode", ["stiff", "variable"])
def test_oracle_seats_the_key(mode):
    success, steps, peak, info = run_oracle(mode)
    assert success == 1.0 and info["succeed"]
    assert 35.0 < peak < se.FORCE_LIMIT  # the latch's 40 N breakaway, sampled at 50 Hz


def test_soft_wrist_cannot_pass_the_latch():
    """Inside the box, the soft wrist runs out of travel at ~19 N, below the latch's 40 N"""
    success, steps, peak, info = run_oracle("soft")
    assert success == 0.0 and steps == 100
    assert 15.0 < peak < 25.0
    assert 0.0005 < info["plunger"] < 0.0025


def test_aim_error_costs_force_only_without_compliance():
    """Aiming 1.5 mm and 3 deg off: the variable oracle lets the chamfer line the key up and seats it at the latch's
    force; the stiff wrist wedges it in and passes the force limit"""
    error = (0.0015, 0.0, np.radians(3.0))
    success, _, peak, _ = run_oracle("variable", noise=0.0, aim_error=error)
    assert success == 1.0 and peak < 50.0
    success, _, peak, info = run_oracle("stiff", noise=0.0, aim_error=error)
    assert success == 0.0 and info["over_force"] and peak > se.FORCE_LIMIT


@pytest.mark.parametrize("mode", ["stiff", "variable"])
def test_independent_of_timestep(mode):
    """The oracle's episode at 2 ms (the default) and 0.5 ms physics: same outcome, length and peak force"""
    results = {}
    default = macros.SIMULATION_TIMESTEP
    try:
        for dt in (0.002, 0.0005):
            macros.SIMULATION_TIMESTEP = dt
            results[dt] = run_oracle(mode)
    finally:
        macros.SIMULATION_TIMESTEP = default
    (r1, n1, f1, _), (r2, n2, f2, _) = results[0.002], results[0.0005]
    assert r1 == r2 == 1.0
    assert abs(n1 - n2) <= 2
    assert f1 == pytest.approx(f2, rel=0.15)


def test_reward(variable_env):
    """Per step: the success reward minus the force penalty on the step's peak flange force"""
    env = variable_env
    env.reset(seed=0)
    _, reward, _, _, info = env.step(np.zeros(5))
    assert not info["succeed"]
    assert reward == pytest.approx(-se.FORCE_PENALTY * (info["max_force"] / se.FORCE_LIMIT) ** 2)
    assert reward > -1e-3  # free space: next to no force
    assert info["max_force_physics"] >= info["max_force"] - 1e-9  # the physics-step peak includes the 50 Hz samples


@pytest.mark.parametrize("mode", ["stiff", "soft"])
def test_force_readings_agree_at_rest(mode):
    """10 N on the key, held: the wrist's own reading (its legs), the arm's (its joint torques) and MuJoCo's flange
    wrench agree; with no load all read ~0 (the arm's joints' Coulomb friction shows up as a few tenths)"""
    env = se.KeyInsertionSERL(mode=mode, seed=0)
    env.reset(seed=0)
    hold = np.zeros(env.action_space.shape)
    read = lambda: (env.gripper.leg_wrench(env.env.sim)[1], env.arm_wrench(), env.gripper.flange_wrench(env.env.sim)[1])  # noqa: E731
    for _ in range(10):
        env.step(hold)
    legs, arm, true = read()
    assert np.abs(legs).max() < 0.3 and np.abs(arm).max() < 0.6 and np.abs(true).max() < 1e-3
    m, d = env.env.sim.model._model, env.env.sim.data._data
    key = m.site_bodyid[env.env.tip_site_id]
    d.xfrc_applied[key, :3] = [0.5, -0.3, 10.0]  # mostly axial: the soft wrist is 140 N/m laterally
    for _ in range(20):
        env.step(hold)
    legs, arm, true = read()
    d.xfrc_applied[key] = 0.0
    env.close()
    assert true[5] == pytest.approx(-10.0, abs=0.3)  # pushing the tool up reads as -z in the flange frame
    assert legs == pytest.approx(true, abs=0.4)
    assert arm == pytest.approx(true, abs=0.6)
