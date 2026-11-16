"""
Wrist stiffness from HSA-UMI's wrist demonstrations: the plate's deflection (baseplate camera, in the plate's rest
frame) regressed on the Franka's external wrench, both recorded per step in the replay buffer. The demos ran with
every HSA at 360 deg, so this is the fully extended stiffness and nothing else.

    ~/hsa-ws/eval_venv/bin/python -m arm_impedance.demo_stiffness ~/hsa-ws/HSA-UMI/data/key_in_hole/wrist/v1/replay_buffer.zarr

Steps:
  1. zero: subtract each episode's first second from the wrench and the plate pose (the Franka's estimate carries a
     bias of a few N; the plate rests at an offset)
  2. rotate the base-frame force into the tool frame with the recorded tool orientation (robot0_eef_rot_axis_angle),
     so "axial" is along the keyhole axis and "lateral" across it
  3. keep steps with the plate seen and |F| > 2 N, and fit displacement = a * force + b per axis; stiffness = 1 / a.
     The axial response is not linear, so the slope is also given per force bin.

Caveats: the plate frame is the AprilTag's, rotated in-plane by an unknown angle from the tool frame, so the lateral
x / y fits are mixed; the wrench estimate includes inertia and payload-model error. These are the numbers behind
wrist_calibration.AXIAL_EXTENDED / LATERAL_EXTENDED; the push test replaces them.
"""

import argparse

import numpy as np


def main():
    import zarr
    from scipy.spatial.transform import Rotation as R

    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("replay_buffer")
    p.add_argument("--min-force", type=float, default=2.0)
    args = p.parse_args()

    z = zarr.open(args.replay_buffer, mode="r")
    d = z["data"]
    ends = z["meta/episode_ends"][:]
    starts = np.concatenate([[0], ends[:-1]])
    plate = d["robot0_plate_pos"][:] * 1e3  # mm
    visible = d["robot0_plate_visible"][:, 0]
    wrench = d["robot0_ext_wrench"][:]
    rot = d["robot0_eef_rot_axis_angle"][:]
    ext = d["robot0_wrist_ext_deg"][:]
    print(f"{len(ends)} episodes, {len(plate)} steps; HSA twist median per axis {np.round(np.nanmedian(ext, 0), 0)} deg")

    # 1. zero per episode
    F = np.zeros_like(wrench)
    P = np.zeros_like(plate)
    for s, e in zip(starts, ends):
        F[s:e] = wrench[s:e] - np.median(wrench[s : s + 10], 0)
        P[s:e] = plate[s:e] - np.median(plate[s : s + 10], 0)
    # 2. force into the tool frame
    Rm = R.from_rotvec(rot).as_matrix()  # tool -> base
    Ft = np.einsum("nji,nj->ni", Rm, F[:, :3])
    # 3. fit
    ok = (visible == 1) & (np.linalg.norm(F[:, :3], axis=1) > args.min_force)
    print(f"steps in contact (|F| > {args.min_force:g} N, plate seen): {ok.sum()}")
    for i, j, name in ((0, 0, "lateral x"), (1, 1, "lateral y"), (2, 2, "axial z")):
        A = np.vstack([Ft[ok, j], np.ones(ok.sum())]).T
        slope, b = np.linalg.lstsq(A, P[ok, i], rcond=None)[0]
        r = np.corrcoef(P[ok, i], Ft[ok, j])[0, 1]
        resid = (P[ok, i] - (slope * Ft[ok, j] + b)).std()
        print(f"  {name:10s}: {slope:+.3f} mm/N -> {abs(1 / slope) * 1e3 if slope else float('nan'):6.0f} N/m  (r = {r:+.2f}, residual sd {resid:.2f} mm)")
    f, q = Ft[ok, 2], P[ok, 2]
    print("  axial by force bin (median plate z):")
    for lo, hi in ((-20, -10), (-10, -5), (-5, -2), (2, 5), (5, 10), (10, 20)):
        m = (f >= lo) & (f < hi)
        if m.sum() > 30:
            print(f"    fz in [{lo:+3d}, {hi:+3d}) N: n = {m.sum():5d}, plate z {np.median(q[m]):+6.2f} mm")


if __name__ == "__main__":
    main()
