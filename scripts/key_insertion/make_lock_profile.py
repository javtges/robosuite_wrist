"""
The plunger force the key meets in the lock the visuomotor policies trained on (KeyDoor, defaults: linear plunger
spring, no latch). Values are read from KeyInsertion's defaults, which KeyDoor does not override.

    cd scripts && python -m key_insertion.make_lock_profile     # -> key_insertion/figures/lock_profile.png
"""

import inspect
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from key_insertion.make_figures import INK, MUTED, save  # noqa: E402  (also applies the repo's figure style)
from robosuite.environments.manipulation.key_insertion import KeyInsertion  # noqa: E402

d = {k: p.default for k, p in inspect.signature(KeyInsertion.__init__).parameters.items()}
assert d["latch_profile"] is None
D, travel = d["slot_depth"] * 1e3, d["plunger_travel"] * 1e3
f0, f1 = d["seat_preload"], d["seat_force"]
top, seat = D - travel, D - travel + d["seat_fraction"] * travel

fig, ax = plt.subplots(figsize=(6, 3.6))
ax.plot([top - 6, top, top, D, D], [0, 0, f0, f1, f1 + 4], color=INK, solid_capstyle="round")
ax.axvline(seat, color=MUTED, lw=1, ls="--")
ax.text(seat - 0.15, 1.0, f"seated ({d['seat_fraction']:.0%} travel)", color=MUTED, fontsize=8, ha="right")
ax.annotate(f"plunger met: {f0:g} N", (top, f0), (top - 5.5, f0 + 3), color=INK, fontsize=9,
            arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.8))  # fmt: skip
ax.annotate(f"seat: {f1:g} N, then hard stop", (D, f1), (D - 7.5, f1 + 3), color=INK, fontsize=9,
            arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.8))  # fmt: skip
ax.set(xlim=(top - 6, D + 1), ylim=(0, f1 + 5), xlabel="key depth below lock face (mm)", ylabel="plunger force (N)")
ax.set_title("Training lock (KeyDoor default): plunger force vs key depth")
save(fig, "lock_profile.png")
