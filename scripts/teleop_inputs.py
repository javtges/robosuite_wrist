"""
Teleop input sources and the terminal command keys.

Every motion source has one job: read() returns (axes, buttons), with axes a 6-vector of normalized velocities
[vx, vy, vz, wx, wy, wz] in [-1, 1] in the robot base frame, and buttons two bools. `exclusive` tells the target
integrator whether HSA-UMI's button modes apply (translation XOR rotation, z locked unless the right button is held).

    SpaceMouseSource  the puck, read with `hid` (the HID library robosuite's own SpaceMouse driver uses). Reports are
                      decoded exactly as pyspacemouse does (HSA-UMI's reader), then mapped like HSA-UMI.
    KeyboardSource    hold-to-move 6D keyboard (pynput: needs key-release events, so it listens globally).
    FakeSineSource    HSA-UMI's desk-test stand-in: a slow circle in the base x-y plane.
    NullSource        no motion.

TerminalKeys reads single-key commands (c, s, Backspace, ...) from the terminal, like HSA-UMI's teleop node. It needs
no permissions, and keys typed into the viewer window never reach it.
"""

import os
import select
import sys
import termios
import threading
import time
import tty

import numpy as np

from teleop_math import fake_sine_axes, map_spacemouse_axes

# ---------------------------------------------------------------------------------------------------------------------
# SpaceMouse
# ---------------------------------------------------------------------------------------------------------------------

SPACEMOUSE_VENDORS = (0x256F, 0x046D)  # 3Dconnexion, and Logitech for the older 3Dconnexion devices
# Devices that send rotation on its own report (id 2). The others send all six axes in report 1.
# From pyspacemouse 2.1.0's devices.toml.
SPLIT_ROTATION_PRODUCTS = {
    0xC626: "SpaceNavigator",
    0xC627: "SpaceExplorer",
    0xC641: "SpaceMouseUSB",
    0xC635: "SpaceMouseCompact",
    0xC62B: "SpaceMousePro",
    0xC625: "SpacePilot",
    0xC629: "SpacePilotPro",
}
COMBINED_PRODUCTS = {
    0xC633: "SpaceMouseEnterprise",
    0xC632: "SpaceMouseProWireless",
    0xC638: "SpaceMouseProWirelessBluetooth",
    0xC62E: "SpaceMouseWireless",
    0xC63A: "SpaceMouseWirelessNew",
    0xC652: "UniversalReceiver",
}
AXIS_SCALE = 350.0  # pyspacemouse's axis_scale for every device


def _int16(lo, hi):
    x = lo | (hi << 8)
    return x - 65536 if x >= 32768 else x


def parse_spacemouse_report(data, raw, buttons, split_rotation):
    """
    Decode one HID report into pyspacemouse's state, in place.

    Args:
        data (list of int): the report; data[0] is the report id
        raw (6-array): [x, y, z, roll, pitch, yaw], pyspacemouse's signs and order
        buttons (list of 2 bools): left, right (report 3, byte 1, bits 0 and 1)
        split_rotation (bool): rotation comes in report 2 (see SPLIT_ROTATION_PRODUCTS)
    """
    v = lambda i, sign: float(np.clip(sign * _int16(data[i], data[i + 1]) / AXIS_SCALE, -1.0, 1.0))
    if data[0] == 1 and len(data) >= 7:
        raw[0], raw[1], raw[2] = v(1, 1), v(3, -1), v(5, -1)
        if not split_rotation and len(data) >= 13:
            raw[4], raw[3], raw[5] = v(7, -1), v(9, -1), v(11, 1)  # pitch, roll, yaw
    elif data[0] == 2 and len(data) >= 7:
        raw[4], raw[3], raw[5] = v(1, -1), v(3, -1), v(5, 1)
    elif data[0] == 3 and len(data) >= 2:
        buttons[0], buttons[1] = bool(data[1] & 0x01), bool(data[1] & 0x02)


def find_spacemouse():
    """Returns the hid.enumerate() entry of the first SpaceMouse found, or None"""
    import hid

    candidates = [
        d
        for d in hid.enumerate()
        if d["vendor_id"] in SPACEMOUSE_VENDORS
        and (
            d["product_id"] in SPLIT_ROTATION_PRODUCTS
            or d["product_id"] in COMBINED_PRODUCTS
            or "3dconnexion" in (d.get("manufacturer_string") or "").lower()
        )
    ]
    # A device can expose several HID interfaces: prefer the multi-axis controller one (usage page 1, usage 8)
    candidates.sort(key=lambda d: (d.get("usage_page"), d.get("usage")) != (1, 8))
    return candidates[0] if candidates else None


class SpaceMouseSource:
    """
    Reads the SpaceMouse in a background thread and hands out the latest state. The device sends reports far faster
    than the control rate, so a thread drains it continuously and the control loop takes a snapshot (as HSA-UMI).
    """

    exclusive = True

    def __init__(self, deadzone=0.0, mapping="hsa"):
        import hid

        info = find_spacemouse()
        if info is None:
            raise RuntimeError(
                "No SpaceMouse found. Is it plugged in? On macOS, quit 3Dconnexion's driver (3DxWare / 3DconnexionHelper) "
                "if it holds the device. Use --input fake_sine or --input keyboard to run without one."
            )
        self.name = SPLIT_ROTATION_PRODUCTS.get(info["product_id"]) or COMBINED_PRODUCTS.get(info["product_id"]) or (
            info.get("product_string") or "unknown SpaceMouse"
        )
        self._split = info["product_id"] in SPLIT_ROTATION_PRODUCTS
        self._device = hid.device()
        self._device.open_path(info["path"])
        self._deadzone = deadzone
        self._mapping = mapping
        self._raw = np.zeros(6)
        self._buttons = [False, False]
        self._lock = threading.Lock()
        self._stop = False
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def _loop(self):
        while not self._stop:
            data = self._device.read(13, 10)
            if data:
                with self._lock:
                    parse_spacemouse_report(data, self._raw, self._buttons, self._split)

    def read(self):
        with self._lock:
            return map_spacemouse_axes(self._raw, self._deadzone, self._mapping), list(self._buttons)

    def read_raw(self):
        """pyspacemouse-convention [x, y, z, roll, pitch, yaw] and buttons, before any mapping (for --print-axes)"""
        with self._lock:
            return self._raw.copy(), list(self._buttons)

    def close(self):
        self._stop = True
        self._thread.join(timeout=0.5)
        self._device.close()


# ---------------------------------------------------------------------------------------------------------------------
# Keyboard (motion)
# ---------------------------------------------------------------------------------------------------------------------

# Key -> (axis, sign). Axes are the robot base frame: +x away from the robot base, +y to its left, +z up.
# None of these is a command key (see teleop_wrist.py), so the terminal ignores them.
KEYBOARD_MOTION = {
    "i": (0, 1), "k": (0, -1),  # x
    "j": (1, 1), "l": (1, -1),  # y
    "u": (2, 1), "o": (2, -1),  # z
    "[": (3, 1), "]": (3, -1),  # rotation about x
    "up": (4, 1), "down": (4, -1),  # rotation about y
    "left": (5, 1), "right": (5, -1),  # rotation about z
}  # fmt: skip


class KeyboardSource:
    """
    6D hold-to-move keyboard: each held key drives its axis at full speed. Uses pynput, which sees key releases but
    listens globally (any focused window). On macOS the terminal app needs Accessibility / Input Monitoring permission.
    """

    exclusive = False
    name = "keyboard"

    def __init__(self):
        from pynput import keyboard

        self._held = set()
        self._lock = threading.Lock()
        self._listener = keyboard.Listener(on_press=self._on_press, on_release=self._on_release)
        self._listener.start()
        self._listener.wait()
        if getattr(self._listener, "IS_TRUSTED", True) is False:
            print(
                "WARNING: this terminal is not trusted to read the keyboard, so keyboard motion will do nothing. "
                "System Settings > Privacy & Security: enable your terminal under Accessibility and Input Monitoring, "
                "then restart it."
            )

    @staticmethod
    def _name(key):
        char = getattr(key, "char", None)
        if char is not None:
            return char.lower()
        return getattr(key, "name", None)

    def _on_press(self, key):
        with self._lock:
            self._held.add(self._name(key))

    def _on_release(self, key):
        with self._lock:
            self._held.discard(self._name(key))

    def read(self):
        axes = np.zeros(6)
        with self._lock:
            for k in self._held:
                if k in KEYBOARD_MOTION:
                    axis, sign = KEYBOARD_MOTION[k]
                    axes[axis] += sign
        return np.clip(axes, -1.0, 1.0), [False, False]

    def close(self):
        self._listener.stop()


# ---------------------------------------------------------------------------------------------------------------------
# Fake / none
# ---------------------------------------------------------------------------------------------------------------------


class FakeSineSource:
    """HSA-UMI's desk-test stand-in: a slow circle in the base x-y plane, no rotation. The z button reads as held."""

    exclusive = True
    name = "fake_sine"

    def __init__(self, period_s=6.0, amplitude=0.4):
        self._t0 = time.monotonic()
        self.period_s, self.amplitude = period_s, amplitude
        self.clock = lambda: time.monotonic() - self._t0  # replaced by the teleop loop with sim time

    def read(self):
        return fake_sine_axes(self.clock(), self.period_s, self.amplitude), [False, True]

    def close(self):
        pass


class NullSource:
    exclusive = True
    name = "none"

    def read(self):
        return np.zeros(6), [False, False]

    def close(self):
        pass


# ---------------------------------------------------------------------------------------------------------------------
# Terminal command keys
# ---------------------------------------------------------------------------------------------------------------------


def parse_terminal_bytes(data):
    """Terminal input -> key names ("c", "backspace", "space", "tab", ...). Escape sequences (arrows) are dropped."""
    keys, i = [], 0
    while i < len(data):
        b = data[i : i + 1]
        if b == b"\x1b":  # ESC [ X: skip the whole sequence
            i += 3 if data[i + 1 : i + 2] == b"[" else 1
            continue
        key = {b"\x7f": "backspace", b"\x08": "backspace", b" ": "space", b"\t": "tab"}.get(b)
        keys.append(key if key is not None else b.decode(errors="ignore").lower())
        i += 1
    return keys


class TerminalKeys:
    """
    Single-key commands from the terminal (cbreak mode, no echo), queued for the control loop. Escape sequences
    (arrow keys) are dropped. Restores the terminal on close().
    """

    def __init__(self):
        self.keys = []
        self._lock = threading.Lock()
        self._fd = sys.stdin.fileno() if sys.stdin.isatty() else None
        if self._fd is None:
            print("WARNING: stdin is not a terminal, so command keys are disabled.")
            return
        self._saved = termios.tcgetattr(self._fd)
        tty.setcbreak(self._fd)
        attrs = termios.tcgetattr(self._fd)
        attrs[3] &= ~termios.ECHO
        termios.tcsetattr(self._fd, termios.TCSANOW, attrs)
        self._stop = False
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def _loop(self):
        while not self._stop:
            if not select.select([self._fd], [], [], 0.05)[0]:
                continue
            keys = parse_terminal_bytes(os.read(self._fd, 64))
            with self._lock:
                self.keys += keys

    def pop_all(self):
        with self._lock:
            keys, self.keys = self.keys, []
        return keys

    def close(self):
        if self._fd is None:
            return
        self._stop = True
        self._thread.join(timeout=0.5)
        termios.tcflush(self._fd, termios.TCIFLUSH)
        termios.tcsetattr(self._fd, termios.TCSADRAIN, self._saved)
