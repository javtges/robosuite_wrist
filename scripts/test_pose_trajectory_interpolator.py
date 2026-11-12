"""
Port of franka_streaming_impedance_controller/test/test_pose_trajectory_interpolator.cpp: the sim's reference
interpolator must behave exactly like the robot's.

    cd scripts && python -m pytest test_pose_trajectory_interpolator.py -q
"""

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from robosuite.utils.pose_trajectory_interpolator import PoseTrajectoryInterpolator, pose_distance

TOL = 1e-9
INF = np.inf


def make_pose(x, y, z, yaw):
    return np.concatenate([[x, y, z], Rotation.from_euler("z", yaw).as_quat()])


def expect_pose(got, x, y, z, qz, qw):
    assert got[:3] == pytest.approx([x, y, z], abs=TOL)
    sign = -1.0 if got[6] < 0.0 else 1.0  # q and -q are the same rotation
    assert sign * got[5] == pytest.approx(qz, abs=TOL)
    assert sign * got[6] == pytest.approx(qw, abs=TOL)
    assert got[3] == pytest.approx(0.0, abs=TOL) and got[4] == pytest.approx(0.0, abs=TOL)


def two_waypoints():
    return PoseTrajectoryInterpolator([10.0, 11.0], [make_pose(0, 0, 0, 0), make_pose(1, 2, 3, np.pi / 2)])


def test_constant_trajectory_holds_its_seed():
    interp = PoseTrajectoryInterpolator.constant(10.0, make_pose(0.3, -0.1, 0.5, 0.25))
    for t in (0.0, 10.0, 1e6):
        expect_pose(interp(t), 0.3, -0.1, 0.5, np.sin(0.125), np.cos(0.125))


def test_matches_umi_lerp_and_slerp():
    interp = two_waypoints()
    expect_pose(interp(10.0), 0.0, 0.0, 0.0, 0.0, 1.0)
    expect_pose(interp(10.25), 0.25, 0.5, 0.75, 0.195090322016, 0.980785280403)
    expect_pose(interp(10.5), 0.5, 1.0, 1.5, 0.382683432365, 0.923879532511)
    expect_pose(interp(11.0), 1.0, 2.0, 3.0, 0.707106781187, 0.707106781187)


def test_clamps_rather_than_extrapolating():
    interp = two_waypoints()
    expect_pose(interp(9.5), 0.0, 0.0, 0.0, 0.0, 1.0)
    expect_pose(interp(11.5), 1.0, 2.0, 3.0, 0.707106781187, 0.707106781187)


def test_schedule_waypoint_splices_without_discontinuity():
    interp = two_waypoints()
    before = interp(10.5)
    spliced = interp.schedule_waypoint(make_pose(5, 0, 0, 0), 12.0, 10.5, 11.0, INF, INF)
    assert len(spliced) == 3
    assert spliced.first_time == pytest.approx(10.5, abs=TOL) and spliced.last_time == pytest.approx(12.0, abs=TOL)
    after = spliced(10.5)
    assert pose_distance(before, after) == pytest.approx((0.0, 0.0), abs=1e-7)
    expect_pose(spliced(11.0), 1.0, 2.0, 3.0, 0.707106781187, 0.707106781187)
    expect_pose(spliced(12.0), 5.0, 0.0, 0.0, 0.0, 1.0)
    # Knot B is now interior: tangent monotone((1,2,3), (4,-2,-3)) = (1,0,0), hence the Hermite value, not lerp's
    expect_pose(spliced(11.5), 2.625, 1.25, 1.875, 0.382683432365, 0.923879532511)


def test_velocity_is_continuous_across_a_splice():
    interp = two_waypoints()
    spliced = interp.schedule_waypoint(make_pose(5, 0, 0, 0), 12.0, 10.5, 11.0, INF, INF)
    assert np.linalg.norm(spliced.velocity_at(0) - interp.velocity_at(0)) == pytest.approx(0.0, abs=TOL)


def test_retroactive_smoothing_upgrades_the_newest_waypoints_tangent():
    interp = PoseTrajectoryInterpolator.constant(10.0, make_pose(0, 0, 0, 0))
    interp = interp.schedule_waypoint(make_pose(1, 0, 0, 0), 11.0, 10.0, 10.0, INF, INF)
    assert np.linalg.norm(interp.velocity_at(1) - [1.0, 0.0, 0.0]) == pytest.approx(0.0, abs=TOL)
    interp = interp.schedule_waypoint(make_pose(-1, 0, 0, 0), 12.0, 10.5, 11.0, INF, INF)
    assert len(interp) == 3
    assert np.linalg.norm(interp.velocity_at(1)) == pytest.approx(0.0, abs=TOL)


def test_does_not_overshoot_past_its_waypoints():
    interp = PoseTrajectoryInterpolator(
        [0.0, 1.0, 2.0, 3.0],
        [make_pose(0, 0, 0, 0), make_pose(1, 0, 0, 0), make_pose(0.5, 0, 0, 0), make_pose(1.5, 0, 0, 0)],
    )
    for t in np.arange(0.0, 3.0 + 1e-9, 0.01):
        x = interp(t)[0]
        assert -TOL <= x <= 1.5 + TOL, t


def test_waypoint_in_the_past_is_ignored():
    interp = two_waypoints()
    same = interp.schedule_waypoint(make_pose(9, 9, 9, 0), 10.4, 10.5, 11.0, INF, INF)
    assert len(same) == 2 and same.first_time == 10.0 and same.last_time == 11.0


def test_speed_limit_stretches_the_segment():
    interp = two_waypoints()
    limited = interp.schedule_waypoint(make_pose(11, 2, 3, np.pi / 2), 11.5, 10.5, 11.0, 1.0, INF)
    assert len(limited) == 3 and limited.last_time == pytest.approx(21.0, abs=TOL)
    expect_pose(limited(21.0), 11.0, 2.0, 3.0, 0.707106781187, 0.707106781187)


def test_pose_distance_splits_translation_from_rotation():
    pos, rot = pose_distance(make_pose(0, 0, 0, 0), make_pose(3, 4, 0, np.pi / 2))
    assert pos == pytest.approx(5.0, abs=TOL) and rot == pytest.approx(np.pi / 2, abs=1e-7)


def test_speed_limited_chunk_keeps_its_path_shape():
    t0 = 0.0
    interp = PoseTrajectoryInterpolator.constant(t0, make_pose(0, 0, 0, 0))
    last = t0
    for k in range(6):
        target = t0 + 0.5 + 0.1 * k
        interp = interp.schedule_waypoint(make_pose(1.5, 0.05 * k, 0, 0), target, t0, last, 1.0, INF)
        last = target
    assert interp(0.25)[0] == pytest.approx(0.25, abs=1e-3) and interp(0.25)[1] == pytest.approx(0.0, abs=1e-6)
    assert interp(0.50)[0] == pytest.approx(0.50, abs=1e-3) and interp(0.50)[1] == pytest.approx(0.0, abs=1e-6)
    end = interp(interp.last_time)
    assert end[0] == pytest.approx(1.5, abs=TOL) and end[1] == pytest.approx(0.25, abs=TOL)


def test_speed_limit_bounds_reference_speed_but_not_its_lag():
    max_speed = 1.0
    curr = last = 0.0
    interp = PoseTrajectoryInterpolator.constant(curr, make_pose(0, 0, 0, 0))
    for chunk in range(40):
        curr = 0.3 * chunk
        for k in range(16):
            target = curr + 0.1 * (k + 1)
            interp = interp.schedule_waypoint(make_pose(0.15 * (chunk * 3 + k + 1), 0, 0, 0), target, curr, last, max_speed, INF)
            last = target
    t = curr
    while t + 0.1 <= interp.last_time:
        travelled = np.linalg.norm(interp(t + 0.1)[:3] - interp(t)[:3])
        assert travelled <= max_speed * 0.1 + 1e-9, t
        t += 0.1
    assert len(interp) <= 20
    assert interp.last_time - curr > 5.0


def test_repeated_splices_do_not_grow_the_trajectory():
    interp = two_waypoints()
    for i in range(200):
        now = 10.5 + 0.1 * i
        interp = interp.schedule_waypoint(make_pose(0.01 * i, 0, 0, 0), now + 0.3, now, now + 0.2, INF, INF)
        assert len(interp) <= 8
    assert len(interp) > 1
