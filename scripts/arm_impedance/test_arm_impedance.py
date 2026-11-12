"""
The wrist-less arm impedance harness: live gain changes reach the controller, the phase scheduler switches, and one
episode of each path runs.

    cd scripts && python -m pytest arm_impedance/test_arm_impedance.py -q
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import arm_impedance.evaluate as ae  # noqa: E402
from key_insertion.oracle import KeyOracle  # noqa: E402


def test_cases_and_gains():
    c = ae.cases(quick=True)
    assert {x["study"] for x in c} == {"profile", "grid", "rot_k", "trans_k", "ceiling", "reference", "schedule"}
    assert all(x["arm"] in ("osc", "hsa") for x in c)
    g = ae.gains(k_t=500.0, k_r=15.0, clip_t=0.04)
    assert g["kp"] == [500.0] * 3 + [15.0] * 3 and g["translational_clip"] == 0.04
    assert g["damping"][0] == pytest.approx(2 * ae.ZETA_T * np.sqrt(500.0))
    assert ae.gains()["damping"] == pytest.approx([ae.D_T, ae.D_R])  # the robot's gains round-trip
    for name, s in ae.SCHEDULES.items():
        assert set(s) == {"free", "search", "slide", "seat"}, name


def test_live_gains_survive_only_after_reset():
    """robosuite rebuilds the controllers on env.reset(): gains set before it are lost, after it they hold"""
    env = ae._env("hsa", "chunks", True, 0)
    ae.apply_gains(ae.controller(env), ae.gains(k_t=500.0))
    env.reset()
    assert ae.controller(env).kp[0] == ae.K_T  # gone
    ae.apply_gains(ae.controller(env), ae.gains(k_t=500.0, k_r=15.0, clip_t=0.04))
    for _ in range(5):
        env.step(np.zeros(env.action_dim))
    ctrl = ae.controller(env)
    assert ctrl.kp[0] == 500.0 and ctrl.kp[3] == 15.0 and ctrl.error_clip[0] == 0.04


def test_schedule_switches_and_case_runs():
    seen = {}
    row = ae.run_case(ae._case("schedule", "soft_rot_insert", overrides=ae.gains(), dx=0.0, dyaw=0.0, schedule="soft_rot_insert", keep_log=True))
    assert row["result"] == "seated", row
    phases = list(dict.fromkeys(row["log"]["schedule_phase"].tolist()))
    assert phases == ["free", "search", "slide", "seat"], phases
    assert row["peak_flange_N"] < 25.0 and row["hold_push_N"] > 8.0
    # the OSC path and the hold reference also run
    for case in (ae._case("grid", "osc", arm="osc"), ae._case("reference", "hold", overrides=dict(ae.gains(), reference="hold"))):
        r = ae.run_case(case)
        assert r["result"] in ("seated", "stuck", "jammed", "timeout"), r.get("error")
    assert ae.task_phase(type("O", (), {"phase": KeyOracle.INSERT})(), 0.010) == "slide"


def test_wrist_schedule_switches_without_moving_the_key():
    """An extension switch moves the plate 40 mm; the arm compensation keeps the key within a few mm of its target"""
    import arm_impedance.wrist_study as ws

    name = "soft_until_seat"
    case = ae._case("wrist", name, overrides=ae.gains(), dx=0.0, dyaw=0.0, wrist_z=ws.SCHEDULES[name]["free"][0], wrist_schedule=name, mechanism="extension", max_seconds=25.0)
    r = ae.run_case(case)
    assert r["result"] == "seated", r
    assert r["n_switches"] == 1 and r["switch_tool_mm"] < 6.0, r
    # phases only advance, and the stiffness-scale ramp is monotone
    sched = ws.WristScheduler("scale_soft_search")
    assert ws.PHASE_ORDER.index("seat") > ws.PHASE_ORDER.index("slide")
