"""
KeyDoor: the mechanism (pins, bolt coupling, strike) and the door oracle end to end.

    cd scripts && python -m pytest key_insertion/test_key_door.py -q      # ~1 min
"""

import os
import sys

import mujoco
import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from key_insertion.door_oracle import DoorOracle, make_door_env, run_door_episode  # noqa: E402

FIXED = dict(lock_x_range=(0, 0), lock_y_range=(0, 0), lock_yaw_range=(0, 0))


def test_mechanism_without_the_arm():
    """Locked: a 30 N push on the door does nothing and 0.3 N m on the cylinder barely moves it. Seated: the key's
    torque turns the cylinder, the bolt retracts, 10 N opens the door"""
    env = make_door_env(arm="hsa", **FIXED)
    env.reset()
    m, d = env.sim.model._model, env.sim.data._data
    cyl, hinge, door = env.cylinder_dof_adr, env.hinge_qpos_adr, env.door_body_id
    assert env.action_dim == 12 and not env.unlocked() and env.door_angle() == 0.0

    def run(n, tau=0.0, fy=0.0, seated=False):
        for _ in range(n):
            if seated:
                d.qpos[env.plunger_qpos_adr] = -env.plunger_travel * 0.95
            q = env.cylinder_angle()
            d.qfrc_applied[cyl] = (-env.pin_stiffness * q if (not env.key_seated() and q < env.pin_release) else 0.0) + tau
            d.xfrc_applied[door, 1] = fy
            mujoco.mj_step(m, d)
        d.xfrc_applied[door, :] = 0

    run(500, fy=-30.0)
    assert np.degrees(d.qpos[hinge]) < 0.5 and env.bolt_retraction() < 1e-3  # the strike contact gives a fraction of a degree
    run(500, tau=0.3)
    assert np.degrees(env.cylinder_angle()) < 1.0
    run(1000, tau=0.3, seated=True)
    assert env.unlocked() and env.bolt_retraction() > 0.9 * env.bolt_length
    run(1000, fy=-10.0)
    assert env._check_success()
    env.close()


@pytest.mark.parametrize("arm,wrist,dyaw", [("hsa", 0.0, 0.0), ("hsa", 1.0, 3.0), ("osc", 0.0, 0.0)])
def test_oracle_opens_the_door(arm, wrist, dyaw):
    env = make_door_env(arm=arm, **FIXED)
    oracle = DoorOracle(env, wrist_z=wrist, perception_error=(0, 0, np.radians(dyaw)), max_force=np.inf if arm == "hsa" else 40.0)
    result, log = run_door_episode(env, oracle)
    assert result == "opened", result
    phases = [p for p in dict.fromkeys(log["phase"].tolist()) if p != DoorOracle.DONE]
    assert phases == [DoorOracle.APPROACH, DoorOracle.ALIGN, DoorOracle.INSERT, DoorOracle.HOLD, DoorOracle.TURN, DoorOracle.PUSH], phases
    assert np.degrees(log["cylinder_angle"][-1]) > 85 and np.degrees(log["door_angle"][-1]) >= 15
    f = np.linalg.norm(log["flange_force"], axis=1)
    assert f[log["phase"] >= DoorOracle.INSERT].max() < (40.0 if arm == "hsa" else 60.0)
    env.close()


def test_stiff_wrist_with_yaw_error_cannot_insert():
    env = make_door_env(arm="hsa", **FIXED)
    oracle = DoorOracle(env, wrist_z=-1.0, perception_error=(0, 0, np.radians(3.0)), max_force=np.inf)
    result, log = run_door_episode(env, oracle)
    assert result == "stuck" and log["depth"][-1] < 0.005
    env.close()
