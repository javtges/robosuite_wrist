"""
Tests for the teleop stack. Run from this directory:  python -m pytest test_teleop.py -q
The last test drives the whole loop headless (fake input, scripted keys), then validates and replays the file.
"""

import os
import subprocess
import sys

import h5py
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from demo_writer import DemoWriter
from teleop_inputs import KeyboardSource, parse_spacemouse_report, parse_terminal_bytes
from teleop_math import (
    clamp_workspace,
    integrate_target,
    integrate_wrist_target,
    map_spacemouse_axes,
    rpy_to_matrix,
)

HERE = os.path.dirname(os.path.abspath(__file__))
KW = dict(max_pos_speed=0.2, max_rot_speed=0.15, hz=50.0)


# ---- axis mapping ------------------------------------------------------------------------------------------------------


def test_hsa_mapping_matches_hsa_umi():
    # HSA-UMI's test_teleop_math: robot = [-z, x, y] for translation and rotation alike
    axes = map_spacemouse_axes([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
    assert np.allclose(axes[:3], [-0.3, 0.1, 0.2])
    assert np.allclose(axes[3:], [-0.6, 0.4, 0.5])


def test_zup_mapping():
    # pyspacemouse: x right, y forward, z up; pitch about x, roll about y
    assert np.allclose(map_spacemouse_axes([0, 1, 0, 0, 0, 0], mapping="zup"), [1, 0, 0, 0, 0, 0])  # forward -> +x
    assert np.allclose(map_spacemouse_axes([1, 0, 0, 0, 0, 0], mapping="zup"), [0, -1, 0, 0, 0, 0])  # right -> -y
    assert np.allclose(map_spacemouse_axes([0, 0, 1, 0, 0, 0], mapping="zup"), [0, 0, 1, 0, 0, 0])  # up -> +z
    assert np.allclose(map_spacemouse_axes([0, 0, 0, 0, 1, 0], mapping="zup"), [0, 0, 0, 0, -1, 0])  # pitch = about x
    assert np.allclose(map_spacemouse_axes([0, 0, 0, 0, 0, 1], mapping="zup"), [0, 0, 0, 0, 0, 1])


def test_deadzone():
    assert np.allclose(map_spacemouse_axes([0.05, 0, 0, 0, 0, 0.5], deadzone=0.1)[:3], 0)


# ---- arm target --------------------------------------------------------------------------------------------------------


def test_button_modes():
    t0 = np.zeros(6)
    axes = np.array([1, 0, 1, 0, 0, 1.0])
    t = integrate_target(t0, axes, rotation_mode=False, z_enabled=False, **KW)
    assert np.allclose(t, [0.2 / 50, 0, 0, 0, 0, 0])  # translation only, z locked
    t = integrate_target(t0, axes, rotation_mode=False, z_enabled=True, **KW)
    assert np.isclose(t[2], 0.2 / 50)
    t = integrate_target(t0, axes, rotation_mode=True, z_enabled=True, **KW)
    assert np.allclose(t[:3], 0) and np.isclose(t[5], 0.15 / 50)  # rotation only
    t = integrate_target(t0, axes, rotation_mode=False, z_enabled=True, exclusive=False, **KW)  # keyboard: both
    assert np.isclose(t[0], 0.2 / 50) and np.isclose(t[5], 0.15 / 50)


def test_rotation_about_base_axes():
    R0 = Rotation.from_euler("xyz", [180, 0, 30], degrees=True)
    t = integrate_target(np.r_[0, 0, 0, R0.as_rotvec()], [0, 0, 0, 0, 0, 1.0], rotation_mode=True, z_enabled=True, **KW)
    expected = Rotation.from_euler("z", 0.15 / 50) * R0  # composed on the left
    assert np.allclose(Rotation.from_rotvec(t[3:]).as_matrix(), expected.as_matrix())


def test_clamp():
    box = dict(x_min=0, x_max=1, y_min=-1, y_max=1, z_min=0.1, z_max=0.5)
    assert np.allclose(clamp_workspace([2, -2, 0, 1, 2, 3], box), [1, -1, 0.1, 1, 2, 3])


# ---- wrist target ------------------------------------------------------------------------------------------------------


def test_wrist_moves_like_the_arm():
    """The same robot-frame input moves the tool along the same robot-frame direction in wrist mode"""
    R_w = Rotation.from_euler("xyz", [180, 3, -90], degrees=True).as_matrix()
    limits = np.ones(6)
    kw = dict(max_pos_speed=0.04, max_rot_speed=0.35, hz=20.0, rotation_mode=False, z_enabled=True, exclusive=False)
    for axis in range(3):
        p = integrate_wrist_target(np.zeros(6), np.eye(6)[axis], R_w, limits, **kw)
        assert np.allclose(R_w @ p[:3], np.eye(3)[axis] * 0.04 / 20)
        p = integrate_wrist_target(np.zeros(6), np.eye(6)[3 + axis], R_w, limits, **kw)
        expected = Rotation.from_rotvec(np.eye(3)[axis] * 0.35 / 20).as_matrix()
        assert np.allclose(R_w @ rpy_to_matrix(p[3:]) @ R_w.T, expected)


def test_wrist_limits():
    p = integrate_wrist_target(
        np.zeros(6), np.ones(6), np.eye(3), np.full(6, 1e-3), max_pos_speed=1, max_rot_speed=1, hz=1,
        rotation_mode=False, z_enabled=True, exclusive=False,
    )  # fmt: skip
    assert np.all(np.abs(p) <= 1e-3 + 1e-12)


# ---- SpaceMouse HID decoding -------------------------------------------------------------------------------------------


def _le(v):
    v &= 0xFFFF
    return [v & 0xFF, v >> 8]


def test_parse_split_rotation_device():
    raw, buttons = np.zeros(6), [False, False]
    parse_spacemouse_report([1] + _le(350) + _le(175) + _le(-350), raw, buttons, split_rotation=True)
    assert np.allclose(raw[:3], [1.0, -0.5, 1.0])  # pyspacemouse: y and z are negated
    parse_spacemouse_report([2] + _le(350) + _le(-175) + _le(350), raw, buttons, split_rotation=True)
    assert np.allclose(raw[3:], [0.5, -1.0, 1.0])  # [roll, pitch, yaw] = [-ch2[3:5], -ch2[1:3], +ch2[5:7]]
    parse_spacemouse_report([3, 0x02], raw, buttons, split_rotation=True)
    assert buttons == [False, True]
    parse_spacemouse_report([3, 0x03], raw, buttons, split_rotation=True)
    assert buttons == [True, True]


def test_parse_combined_device_and_clip():
    raw, buttons = np.zeros(6), [False, False]
    parse_spacemouse_report([1] + _le(700) + _le(0) + _le(0) + _le(350) + _le(0) + _le(-175), raw, buttons, False)
    assert np.allclose(raw, [1.0, 0, 0, 0, -1.0, -0.5])


# ---- keyboard / terminal -----------------------------------------------------------------------------------------------


def test_terminal_bytes():
    assert parse_terminal_bytes(b"cS \x7f\x1b[Ax\t") == ["c", "s", "space", "backspace", "x", "tab"]


def test_keyboard_hold_logic():
    class Key:  # stands in for pynput's KeyCode / Key
        def __init__(self, char=None, name=None):
            self.char, self.name = char, name

    kb = KeyboardSource.__new__(KeyboardSource)  # no listener: drive the callbacks directly
    import threading

    kb._held, kb._lock = set(), threading.Lock()
    kb._on_press(Key(char="I"))
    kb._on_press(Key(name="left"))
    kb._on_press(Key(char="c"))  # a command key: no motion
    assert np.allclose(kb.read()[0], [1, 0, 0, 0, 0, 1])
    kb._on_press(Key(char="k"))  # opposite key cancels
    kb._on_release(Key(name="left"))
    assert np.allclose(kb.read()[0], 0)


# ---- hdf5 writer -------------------------------------------------------------------------------------------------------


def _env_args(name="Lift"):
    return {"env_name": name, "type": 1, "env_version": "test", "env_kwargs": {"robots": ["R"]}}


def test_writer_roundtrip(tmp_path):
    path = str(tmp_path / "d.hdf5")
    w = DemoWriter(path, _env_args(), {"actuator_kp": 1.0})
    for n in (3, 5):
        w.start(dict(plan_row={"a": 1}, condition="c", input="t", model_file="<mujoco/>"))
        for i in range(n):
            w.add(np.zeros(4), np.ones(13), 0.0, {"o": np.zeros(2)}, {"o": np.ones(2)}, 0, i % 2, 1.0)
        assert w.save(success=False, success_any=False) is not None
    assert w.demo_names == ["demo_0", "demo_1"] and w.data.attrs["total"] == 8
    assert w.delete_last() == "demo_1" and w.data.attrs["total"] == 3
    w.close()
    w = DemoWriter(path, _env_args(), {})  # reopening appends
    w.start({})
    w.add(np.zeros(4), np.ones(13), 0.0, {"o": np.zeros(2)}, {"o": np.ones(2)}, 0, 0, 1.0)
    assert w.save(True, True) == "demo_1"
    w.close()
    with h5py.File(path, "r") as f:
        g = f["data/demo_0"]
        assert g["actions"].shape == (3, 13) and list(g["dones"][()]) == [0, 0, 1]
        assert g["obs/o"].shape == (3, 2) and g["next_obs/o"].shape == (3, 2)
        assert list(g["teleop/mode"][()]) == [0, 1, 0]
    with pytest.raises(ValueError):
        DemoWriter(path, _env_args("Stack"), {})


# ---- whole loop --------------------------------------------------------------------------------------------------------


def test_teleop_records_and_replays(tmp_path):
    out = str(tmp_path / "e2e.hdf5")
    teleop = [
        sys.executable, os.path.join(HERE, "teleop_wrist.py"), "--env", "Lift", "--input", "fake_sine", "--no-render",
        "--no-realtime", "--no-terminal-keys", "--out", out,
        "--key-script", "0.5:c,2:m,2.5:x,3:g,4:space,5:s,5.5:c,6:backspace,6.5:c,7:s,7.5:q",
    ]  # fmt: skip
    r = subprocess.run(teleop, cwd=HERE, capture_output=True, text=True, timeout=300)
    assert r.returncode == 0, r.stdout + r.stderr
    assert "saved demo_0" in r.stdout and "episode dropped" in r.stdout and "saved demo_1" in r.stdout
    r = subprocess.run([sys.executable, os.path.join(HERE, "validate_demos.py"), out], cwd=HERE, capture_output=True, text=True, timeout=300)
    assert r.returncode == 0 and "VALIDATION OK" in r.stdout, r.stdout + r.stderr
