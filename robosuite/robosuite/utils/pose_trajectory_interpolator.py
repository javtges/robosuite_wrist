"""
Python port of HSA-UMI's franka_streaming_impedance_controller/pose_trajectory_interpolator.{hpp,cpp}, itself a port
of UMI's umi/common/pose_trajectory_interpolator.py: what turns a sparse, absolutely-timed action chunk into the
continuous reference a 1 kHz (here: every-sim-step) servo needs.

Poses are 7-vectors [x, y, z, qx, qy, qz, qw] (quaternion xyzw, as scipy and robosuite use). Times are absolute
seconds on whatever clock the caller uses; this module never reads a clock.

- Position: monotone cubic Hermite spline through the waypoints. The tangent at each interior waypoint is the mean of
  its two secants, sign-consistent with both and capped at the smaller magnitude, and zero at a local extremum. That
  makes the velocity continuous through the knots and keeps the spline from overshooting past its own waypoints. A
  bare two-waypoint trajectory degenerates to lerp.
- Orientation: slerp between waypoints (angular velocity still steps at the knots, as on the robot).
- `schedule_waypoint` splices a future waypoint into a trajectory already being consumed without a discontinuity at
  the current instant, discarding whatever was pending after `last_waypoint_time`, and stretches the new segment when
  reaching it on time would exceed `max_pos_speed` / `max_rot_speed`.
- `__call__` clamps to the endpoints: past the last waypoint the reference holds, it never extrapolates.
"""

import bisect

import numpy as np


def quat_normalize(q):
    q = np.asarray(q, dtype=float)
    n = np.linalg.norm(q)
    if n < 1e-12:
        raise ValueError("near-zero quaternion")
    return q / n


def quat_angular_distance(q0, q1):
    """Angle (rad) of the rotation taking q0 to q1, on the shorter arc"""
    d = abs(float(np.dot(q0, q1)))
    return 2.0 * np.arccos(min(1.0, d))


def quat_slerp(q0, q1, alpha):
    """Spherical linear interpolation on the shorter arc (Eigen's Quaternion::slerp)"""
    q0 = np.asarray(q0, dtype=float)
    q1 = np.asarray(q1, dtype=float)
    d = float(np.dot(q0, q1))
    if d < 0.0:
        q1 = -q1
        d = -d
    if d > 1.0 - 1e-9:
        return quat_normalize(q0 + alpha * (q1 - q0))
    theta = np.arccos(d)
    s = np.sin(theta)
    return (np.sin((1.0 - alpha) * theta) / s) * q0 + (np.sin(alpha * theta) / s) * q1


def pose_distance(a, b):
    """(translational distance m, angular distance rad) between two 7-vector poses"""
    return float(np.linalg.norm(b[:3] - a[:3])), quat_angular_distance(a[3:], b[3:])


def _monotone_tangent(secant_prev, secant_next):
    out = np.zeros(3)
    for i in range(3):
        a, b = secant_prev[i], secant_next[i]
        if a * b <= 0.0:
            continue
        avg = 0.5 * (a + b)
        bound = min(abs(a), abs(b))
        out[i] = np.copysign(min(abs(avg), bound), avg)
    return out


class PoseTrajectoryInterpolator:
    def __init__(self, times, poses):
        """
        Trajectory through @poses at @times (non-empty, equal length, non-decreasing). `constant(time, pose)` builds
        the one-waypoint trajectory that evaluates to `pose` at every instant.
        """
        times = [float(t) for t in np.atleast_1d(times)]
        poses = np.atleast_2d(np.asarray(poses, dtype=float))
        if not times or len(times) != len(poses) or poses.shape[1] != 7:
            raise ValueError("PoseTrajectoryInterpolator: need equal, non-empty times and (n, 7) poses")
        if any(t1 < t0 for t0, t1 in zip(times, times[1:])):
            raise ValueError("PoseTrajectoryInterpolator: times must be non-decreasing")
        self.times = times
        self.poses = poses.copy()
        self._compute_velocities()

    @classmethod
    def constant(cls, time, pose):
        return cls([time], [pose])

    @property
    def first_time(self):
        return self.times[0]

    @property
    def last_time(self):
        return self.times[-1]

    def __len__(self):
        return len(self.times)

    def velocity_at(self, i):
        """The velocity (m/s) the position spline was built to hit exactly at waypoint @i"""
        return self.velocities[i]

    def _compute_velocities(self):
        n = len(self.times)
        self.velocities = np.zeros((n, 3))
        if n < 2:
            return
        secants = np.zeros((n - 1, 3))
        for i in range(n - 1):
            span = self.times[i + 1] - self.times[i]
            if span > 0.0:
                secants[i] = (self.poses[i + 1, :3] - self.poses[i, :3]) / span
        self.velocities[0] = secants[0]
        self.velocities[-1] = secants[-1]
        for i in range(1, n - 1):
            self.velocities[i] = _monotone_tangent(secants[i - 1], secants[i])

    def __call__(self, t):
        """Interpolate at @t, clamped to the trajectory's own endpoints (never extrapolates)"""
        t = min(max(float(t), self.times[0]), self.times[-1])
        hi = bisect.bisect_left(self.times, t)  # first waypoint at or after t
        if hi == 0:
            return self.poses[0].copy()
        if hi >= len(self.times):
            return self.poses[-1].copy()
        lo = hi - 1
        span = self.times[hi] - self.times[lo]
        alpha = (t - self.times[lo]) / span if span > 0.0 else 1.0
        out = np.empty(7)
        if span > 0.0:
            s = alpha
            h00 = 2 * s**3 - 3 * s**2 + 1
            h10 = s**3 - 2 * s**2 + s
            h01 = -2 * s**3 + 3 * s**2
            h11 = s**3 - s**2
            out[:3] = (
                h00 * self.poses[lo, :3]
                + (h10 * span) * self.velocities[lo]
                + h01 * self.poses[hi, :3]
                + (h11 * span) * self.velocities[hi]
            )
        else:
            out[:3] = self.poses[hi, :3]
        out[3:] = quat_slerp(self.poses[lo, 3:], self.poses[hi, 3:], alpha)
        return out

    def trim(self, start_t, end_t):
        """The sub-trajectory spanning [start_t, end_t], with both endpoints materialised as waypoints"""
        times, poses = [], []

        def push(t):
            if not times or t > times[-1]:
                times.append(t)
                poses.append(self(t))

        push(start_t)
        for t in self.times:
            if start_t < t < end_t:
                push(t)
        push(end_t)
        return PoseTrajectoryInterpolator(times, poses)

    def schedule_waypoint(self, pose, time, curr_time, last_waypoint_time, max_pos_speed=np.inf, max_rot_speed=np.inf):
        """
        Splice @pose in as a waypoint at absolute instant @time. Everything after @last_waypoint_time is discarded, so
        a fresh chunk supersedes the tail of the previous one while the part already being executed is left alone. A
        waypoint at or before @curr_time is ignored. The segment is stretched if reaching the waypoint on time would
        exceed either speed limit, which bounds how far the reference can run away from the arm.
        """
        if time <= curr_time:
            return self
        # Ported verbatim from UMI: this min/max sequence is what establishes curr_time <= start_time <= end_time <= time
        start_time = max(curr_time, self.times[0])
        end_time = curr_time if time <= last_waypoint_time else max(last_waypoint_time, curr_time)
        end_time = min(end_time, time)
        start_time = min(start_time, end_time)

        trimmed = self.trim(start_time, end_time)
        end_pose = trimmed(end_time)
        pos_dist, rot_dist = pose_distance(pose, end_pose)
        duration = max(time - end_time, pos_dist / max_pos_speed, rot_dist / max_rot_speed)
        new_time = end_time + duration

        times, poses = list(trimmed.times), [p for p in trimmed.poses]
        while times and times[-1] >= new_time:
            times.pop()
            poses.pop()
        times.append(new_time)
        poses.append(np.asarray(pose, dtype=float))
        return PoseTrajectoryInterpolator(times, poses)
