"""
Measured wrist numbers and the interim ("interpolated") sim calibration built from them.

Sources (HSA-UMI/data):
  sweep     wrist/sysid/sysid_static_sweep_20261108-153328_points.csv: the all-six staircase, unloaded. Plate rise per
            HSA twist, every 30 deg, up and down (hysteresis <= 0.8 mm mid-range). Geometry only.
  demos     key_in_hole/wrist/v1/replay_buffer.zarr: 65k contact samples at ONE setting (all HSAs at 360 deg, fully
            extended), plate pose (baseplate camera, AprilTag frame) against the Franka's external wrench rotated into
            the tool frame. Load-deflection at that one level.

What is measured and what is interpolated:
  - z stroke: 18.9 mm at 360 deg (sweep). The sim's z_range becomes 0.019.
  - extended axial stiffness: the demos give ~2.5 kN/m from the 10-20 N slope and 5.6 kN/m from a linear fit, with
    almost no deflection below 10 N. AXIAL_EXTENDED = 3500 N/m is the compromise. The sim's strut geometry turns a
    leg stiffness into K_zz ~ 5 x k_leg at full extension, so leg_stiffness_extended = 700 N/m (was 300).
  - extended lateral stiffness: measured ~2.1 kN/m. The same leg stiffness gives only 330 N/m in the sim, because
    pin-jointed struts have K_xx / K_zz ~ 0.09 where the real HSAs (which bend, not just stretch) show ~0.5. The sim
    cannot match both with one leg stiffness; this preset matches AXIAL and is 6x too soft laterally. Expect the
    sim's yaw tolerance for the soft wrist to be optimistic.
  - contracted (0 deg) and the levels between: NOT measured. leg_stiffness_contracted stays at the mentor's 100 kN/m
    and the levels between follow the sim's log-linear law. Replace with the push test (0 / 120 / 240 / 360 deg).

Use: `small_eval --wrist-calib measured_v0`, or `evaluate.WRIST_OVERRIDES = WRIST_PRESETS["measured_v0"]` before
building envs. Results produced with it carry `wrist_calib=measured_v0` in the ledger.
"""

import numpy as np

# sweep 2026-11-08 15:33, all six legs, going up: HSA twist (deg) -> plate rise (mm). Going down differs by <= 0.8 mm.
SWEEP_TWIST_DEG = np.array([0, 30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330, 360], dtype=float)
SWEEP_RISE_MM = np.array([0.0, 1.38, 3.35, 5.60, 7.76, 9.85, 11.73, 13.39, 14.87, 16.23, 17.41, 18.51, 18.93])
SWEEP_HOLD_IQ_A = {0: 0.55, 60: 0.30, 120: 0.59, 180: 0.95, 240: 1.26, 300: 1.49, 360: 2.15}  # per leg, holding

# demos 2026-10-22, all HSAs at 360 deg, tool-frame fits (N/m)
AXIAL_EXTENDED = 3500.0  # 2500 (10-20 N slope) .. 5600 (linear fit)
LATERAL_EXTENDED = 2100.0  # not reproducible with the strut model at the axial-matched leg stiffness

Z_STROKE_M = SWEEP_RISE_MM[-1] * 1e-3  # 18.9 mm from 0 to 360 deg; the sim's z_range is half the stroke


def twist_to_normalized_z(twist_deg):
    """HSA twist (deg, 0..360) -> the sim's normalized wrist z in [-1, 1], through the measured sweep curve"""
    rise = np.interp(twist_deg, SWEEP_TWIST_DEG, SWEEP_RISE_MM)
    return 2.0 * rise / SWEEP_RISE_MM[-1] - 1.0


def normalized_z_to_twist(z):
    """the inverse: sim z in [-1, 1] -> HSA twist (deg)"""
    rise = (np.asarray(z, dtype=float) + 1.0) / 2.0 * SWEEP_RISE_MM[-1]
    return np.interp(rise, SWEEP_RISE_MM, SWEEP_TWIST_DEG)


WRIST_PRESETS = {
    # the mentor's defaults: nothing measured
    "default": {},
    # geometry from the sweep, extended axial stiffness from the demos, everything else interpolated / unchanged
    "measured_v0": dict(z_range=Z_STROKE_M / 2.0, leg_stiffness_extended=700.0),
}
