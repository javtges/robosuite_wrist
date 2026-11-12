"""
The arm controller switch (arm_controller.py) and the hardware flavour of CartesianImpedanceController.

    cd scripts && python -m pytest test_arm_controller.py -q
"""

import os
import subprocess
import sys

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import arm_controller as ac  # noqa: E402
from robosuite.controllers.parts.arm.cartesian_impedance import damped_pinv, projector_nullspace_torques  # noqa: E402

HSA_YAML = os.path.normpath(os.path.join(HERE, "..", "..", ac.HSA_GAINS_SOURCE))


def test_hsa_gains_match_the_deployment_yaml():
    """The copy in arm_controller.py is the robot's hsa_controllers.yaml (when the HSA-UMI checkout is alongside)"""
    if not os.path.exists(HSA_YAML):
        pytest.skip(f"{HSA_YAML} not found")
    assert ac.read_gains_yaml(HSA_YAML) == ac.HSA_GAINS


def test_apply_arm_controller():
    from robosuite.controllers import load_composite_controller_config

    cfg = load_composite_controller_config(robot="PandaStewartAloha")
    assert ac.apply_arm_controller(cfg, "osc")["body_parts"]["right"]["type"] == "OSC_POSE"
    c = ac.apply_arm_controller(cfg, "impedance")["body_parts"]["right"]
    assert c["type"] == "CARTESIAN_IMPEDANCE"
    assert c["kp"] == [2000.0] * 3 + [150.0] * 3 and c["damping"] == [89.0, 7.0]
    assert c["translational_clip"] == 0.01 and c["nullspace"] == "projector"
    assert c["reference"] == "chunks" and c["chunk_len"] == 6 and c["max_pos_speed"] == 1.0
    assert "20 N ceiling" in ac.describe(cfg) and "6-waypoint chunks" in ac.describe(cfg)
    assert ac.apply_arm_controller(cfg, "impedance", reference="hold")["body_parts"]["right"]["reference"] == "hold"
    with pytest.raises(ValueError):
        ac.apply_arm_controller(cfg, "pid")


def test_damped_pinv_and_projector():
    rng = np.random.default_rng(0)
    J = rng.normal(size=(6, 7))
    assert np.allclose(damped_pinv(J.T, 0.0), np.linalg.pinv(J.T))
    # The undamped projector annihilates J (every nullspace torque is invisible at the tool); the damped one leaks a
    # little, which is the robot's behaviour (hsa_controllers.yaml: "a share leaks through as a flange force")
    N0 = np.eye(7) - J.T @ damped_pinv(J.T, 0.0)
    N = np.eye(7) - J.T @ damped_pinv(J.T, 0.2)
    assert np.allclose(J @ N0, 0.0, atol=1e-9)
    assert 0.0 < np.linalg.norm(J @ N) < 0.1 * np.linalg.norm(J)
    # Joint 1 tier: with every joint 1 rad off, joint 1 is pulled joint1_stiffness times harder before projection
    q, dq = np.zeros(7), np.zeros(7)
    tau = projector_nullspace_torques(J, q, dq, np.ones(7), 1.0, 10.0, 0.0)
    assert np.allclose(tau, N0 @ np.array([10.0] + [1.0] * 6))


def _push_into_table(arm_controller, depth):
    """Drives the arm's target @depth below where the gripper touches the table; returns the external flange force (N)"""
    import mujoco
    import tempfile

    import make_teleop_gifs as mg

    with tempfile.TemporaryDirectory() as d:
        t = mg.make_teleop(os.path.join(d, "unused.hdf5"), arm_controller=arm_controller)
        try:
            m, data = t.env.sim.model._model, t.env.sim.data._data
            base = m.body(t.gripper.correct_naming(t.wrist_base_body)).id
            qadr = m.jnt_qposadr[m.body(t.env.cube_body_id).jntadr[0]]
            data.qpos[qadr : qadr + 3] = [-1.5, -1.5, 0.03]
            mujoco.mj_forward(m, data)
            for _ in range(40):
                t.tick()
            free = np.linalg.norm(t.flange_wrench()[1])

            def contact_on_gripper():  # world frame, the wrist's and gripper's contacts with everything else
                out = np.zeros(3)
                for i in range(data.ncon):
                    c = data.contact[i]
                    ours = [m.body_rootid[m.geom_bodyid[g]] == m.body_rootid[base] and m.geom_bodyid[g] >= base for g in (c.geom1, c.geom2)]
                    if ours[0] != ours[1]:
                        f = np.zeros(6)
                        mujoco.mj_contactForce(m, data, i, f)
                        out += (c.frame.reshape(3, 3).T @ f[:3]) * (1 if ours[1] else -1)
                return out

            t.box["z_min"] = -1.0
            while np.linalg.norm(contact_on_gripper()) < 0.5:
                t.target[2] -= 0.0005
                t.tick()
            t.target[2] -= depth
            for _ in range(40):
                t.tick()
            # The fingertips rock on the table (1 or 2 contacts from tick to tick) while the commanded wrench is steady,
            # so compare 2 s averages, not one instant
            f_world, contact = np.zeros(3), np.zeros(3)
            for _ in range(40):
                t.tick()
                f_world += data.xmat[base].reshape(3, 3) @ t.flange_wrench()[1][3:] / 40
                contact += contact_on_gripper() / 40
            return free, f_world, contact
        finally:
            t.close()


def test_force_ceiling_and_free_space():
    """20 mm below the surface the hardware law pushes with K * clip = 20 N, however far the target goes; OSC does not"""
    free, f_imp, contact = _push_into_table("impedance", 0.020)
    assert free < 0.5
    assert np.allclose(f_imp, contact, atol=1.0), (f_imp, contact)  # the readout is the table's push
    assert f_imp[2] == pytest.approx(20.0, abs=3.0), f_imp
    _, f_imp_deep, _ = _push_into_table("impedance", 0.060)
    assert f_imp_deep[2] == pytest.approx(20.0, abs=3.0), f_imp_deep
    _, f_osc, _ = _push_into_table("osc", 0.020)
    assert f_osc[2] > 40.0, f_osc


def test_teleop_impedance_records_and_replays(tmp_path):
    out = str(tmp_path / "imp.hdf5")
    teleop = [
        sys.executable, os.path.join(HERE, "teleop_wrist.py"), "--env", "Lift", "--input", "fake_sine", "--no-render",
        "--no-realtime", "--no-terminal-keys", "--arm-controller", "impedance", "--out", out,
        "--key-script", "0.5:c,2:m,3:g,5:s,5.5:q",
    ]  # fmt: skip
    r = subprocess.run(teleop, cwd=HERE, capture_output=True, text=True, timeout=300)
    assert r.returncode == 0, r.stdout + r.stderr
    assert "arm: CARTESIAN_IMPEDANCE" in r.stdout and "saved demo_0" in r.stdout
    r = subprocess.run([sys.executable, os.path.join(HERE, "validate_demos.py"), out], cwd=HERE, capture_output=True, text=True, timeout=300)
    assert r.returncode == 0 and "VALIDATION OK" in r.stdout, r.stdout + r.stderr
    assert "CARTESIAN_IMPEDANCE" in r.stdout


def _teleop(arm_controller):
    import tempfile

    import make_teleop_gifs as mg

    d = tempfile.mkdtemp()
    t = mg.make_teleop(os.path.join(d, "unused.hdf5"), arm_controller=arm_controller)
    for _ in range(40):
        t.tick()
    return t


def test_reference_moves_through_the_tick_and_holds_at_the_end():
    """The spring's anchor is interpolated every sim step: continuous at the tick start, on the target at the tick end"""
    t = _teleop("impedance")
    try:
        ctrl = t.robot.part_controllers["right"]
        dt = 1.0 / t.hz
        t0 = ctrl.now
        before = ctrl.reference_pose(t0)
        t.target[0] += 0.01
        t.tick()
        assert ctrl.interp.first_time <= t0 + 1e-9 and ctrl.last_waypoint_time == pytest.approx(t0 + 6 * dt)
        assert np.allclose(ctrl.reference_pose(t0)[:3], before[:3], atol=1e-9)  # no jump at the splice
        at_end = ctrl.reference_pose(t0 + dt)
        assert at_end[:3] == pytest.approx(t.target[:3], abs=1e-9)
        mid = ctrl.reference_pose(t0 + 0.5 * dt)
        frac = (mid[0] - before[0]) / (at_end[0] - before[0])
        assert 0.05 < frac < 0.95, frac
        # The chunk continues along the step (HSA-UMI's extrapolation): waypoint k lies k steps beyond the target
        assert ctrl.reference_pose(t0 + 6 * dt)[0] == pytest.approx(t.target[0] + 5 * 0.01, abs=1e-9)
        # The next tick's chunk supersedes everything pending after its own first waypoint
        t.target[0] += 0.01
        t.tick()
        t1 = t0 + dt
        assert ctrl.reference_pose(t1 + dt)[0] == pytest.approx(t.target[0], abs=1e-9)
        assert ctrl.reference_pose(t1 + 6 * dt)[0] == pytest.approx(t.target[0] + 5 * 0.01, abs=1e-9)
    finally:
        t.close()


def test_reseed_jump_is_a_hold():
    """A target that jumps more than chunk_hold_pos_step in one tick has no velocity: a 1-waypoint chunk"""
    t = _teleop("impedance")
    try:
        ctrl = t.robot.part_controllers["right"]
        dt = 1.0 / t.hz
        t.target[0] += 0.005
        t.tick()
        t0 = ctrl.now
        t.target[0] += 0.08  # under the teleop's 15 cm re-seed, over the controller's 5 cm hold step
        t.tick()
        # Still a 6-waypoint chunk (HSA-UMI's teleop publishes 6 identical poses for a hold), but no extrapolation:
        # every waypoint is the target itself, so the reference arrives and stays instead of running 5 steps past it
        assert ctrl.last_waypoint_time == pytest.approx(t0 + 6 * dt)
        for k in range(2, 11):
            assert ctrl.reference_pose(t0 + k * dt)[:3] == pytest.approx(t.target[:3], abs=1e-9), k
    finally:
        t.close()


def test_controller_state_round_trip():
    t = _teleop("impedance")
    try:
        ctrl = t.robot.part_controllers["right"]
        t.target[0] += 0.01
        t.tick()
        t.target[1] += 0.01
        t.tick()
        import json

        d = json.loads(json.dumps(ctrl.controller_state()))
        ref_before = [ctrl.reference_pose(ctrl.now + k * 0.01) for k in range(20)]
        ctrl.reset_goal()  # wipes the reference
        assert not np.allclose(ctrl.reference_pose(ctrl.now + 0.1), ref_before[10])
        ctrl.restore_controller_state(d)
        for k in range(20):
            assert np.allclose(ctrl.reference_pose(ctrl.now + k * 0.01), ref_before[k], atol=1e-9)
        assert np.allclose(ctrl.prev_target, d["prev_target"])
    finally:
        t.close()
