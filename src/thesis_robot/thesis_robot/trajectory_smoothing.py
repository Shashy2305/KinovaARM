"""
Jerk-free re-timing of a planned joint path (pure numpy/scipy, no ROS).

Pilz PTP and the Cartesian time parameterisation give a trapezoidal speed profile: the acceleration jumps
from 0 to its limit at the start of a move and back at the end, which the arm shows as a jerk. Here the SAME
verified joint-space path is re-timed along a minimum-jerk curve u(t) = 10t^3 - 15t^4 + 6t^5 (t in 0..1), so
velocity AND acceleration are exactly zero at both ends and continuous in between.

The duration is the shortest one that keeps every joint inside vel_scale * vmax and acc_scale * amax. Only the
timing changes: the path is a cubic spline through the planner's waypoints, and if that spline strays more than
`max_dev` rad from the straight lines between them (which are what the safety check looked at) the straight lines
are used instead.
"""
import math

import numpy as np

PROFILES = ('min_jerk', 's_curve')

PEAK_U_DOT = 1.875          # max of u'(t) for the minimum-jerk curve, times 1/T
PEAK_U_DDOT = 5.7735        # max of |u''(t)|, times 1/T^2
MIN_JERK_PEAK_U_DOT_SQ = PEAK_U_DOT ** 2


class _CubicPath:
    """Natural cubic spline through (u_i, P_i), numpy only (the controller process cannot import scipy.interpolate:
    its user-site scipy does not match its numpy). Evaluates the value and the first two derivatives."""

    def __init__(self, u, P):
        self.u, self.P = np.asarray(u, float), np.asarray(P, float)
        n = len(self.u)
        h = np.diff(self.u)
        A = np.zeros((n, n))
        rhs = np.zeros_like(self.P)
        A[0, 0] = A[-1, -1] = 1.0                    # natural ends: second derivative 0
        for i in range(1, n - 1):
            A[i, i - 1], A[i, i], A[i, i + 1] = h[i - 1], 2 * (h[i - 1] + h[i]), h[i]
            rhs[i] = 6 * ((self.P[i + 1] - self.P[i]) / h[i] - (self.P[i] - self.P[i - 1]) / h[i - 1])
        self.M = np.linalg.solve(A, rhs)             # second derivatives at the knots

    def __call__(self, uu, nu=0):
        uu = np.atleast_1d(np.asarray(uu, float))
        i = np.clip(np.searchsorted(self.u, uu, side='right') - 1, 0, len(self.u) - 2)
        h = (self.u[i + 1] - self.u[i])[:, None]
        a, b = (self.u[i + 1] - uu)[:, None], (uu - self.u[i])[:, None]
        M0, M1, P0, P1 = self.M[i], self.M[i + 1], self.P[i], self.P[i + 1]
        if nu == 0:
            return M0 * a ** 3 / (6 * h) + M1 * b ** 3 / (6 * h) + (P0 / h - M0 * h / 6) * a + (P1 / h - M1 * h / 6) * b
        if nu == 1:
            return -M0 * a ** 2 / (2 * h) + M1 * b ** 2 / (2 * h) - (P0 / h - M0 * h / 6) + (P1 / h - M1 * h / 6)
        return M0 * a / h + M1 * b / h


class _LinearPath:
    """Piecewise-linear path through (u_i, P_i): the fallback, and the reference the spline is checked against."""

    def __init__(self, u, P):
        self.u, self.P = np.asarray(u), np.asarray(P)

    def __call__(self, uu, nu=0):
        uu = np.atleast_1d(uu)
        if nu == 0:
            return np.column_stack([np.interp(uu, self.u, self.P[:, j]) for j in range(self.P.shape[1])])
        if nu == 1:
            idx = np.clip(np.searchsorted(self.u, uu, side='right') - 1, 0, len(self.u) - 2)
            return (self.P[idx + 1] - self.P[idx]) / (self.u[idx + 1] - self.u[idx])[:, None]
        return np.zeros((len(uu), self.P.shape[1]))


def scurve_profile(dt, V, A, J):
    """Rest-to-rest jerk-limited S-curve u(t) from 0 to 1 with |u'| <= V, |u''| <= A, |u'''| <= J: the acceleration ramps up with jerk J, holds A,
    ramps down, the speed holds V, and the whole thing mirrors for the stop. Returns (t, u, ud, udd) sampled every ~dt seconds, ending exactly at u = 1.
    Unlike the minimum-jerk curve (whose peak speed is 1.875 x its mean) it reaches V and stays there, so the same limits give a much shorter move."""
    def accel_time(vp):
        """duration of the speed-up from 0 to vp (jerk J, acceleration at most A)"""
        return (vp / A + A / J) if vp * J >= A * A else 2.0 * math.sqrt(vp / J)
    vp = V
    if vp * accel_time(vp) >= 1.0:                       # too short to reach V: lower the peak speed until the two ramps just cover the distance
        lo, hi = 0.0, V
        for _ in range(80):
            mid = 0.5 * (lo + hi)
            lo, hi = (mid, hi) if mid * accel_time(mid) < 1.0 else (lo, mid)
        vp = 0.5 * (lo + hi)
    t_acc = accel_time(vp)
    cruise = max(0.0, (1.0 - vp * t_acc) / vp)
    if vp * J >= A * A:
        tj, tc = A / J, vp / A - A / J
    else:
        tj, tc = math.sqrt(vp / J), 0.0
    # the jerk pattern: +J, 0, -J, cruise, -J, 0, +J
    seg = [(tj, J), (tc, 0.0), (tj, -J), (cruise, 0.0), (tj, -J), (tc, 0.0), (tj, J)]
    T = sum(d for d, _ in seg)
    fine = 0.0005
    n = int(math.ceil(T / fine)) + 1
    t = np.linspace(0.0, T, n)
    jerk = np.zeros(n)
    t0 = 0.0
    for d, jv in seg:
        jerk[(t >= t0) & (t < t0 + d)] = jv
        t0 += d
    h = t[1] - t[0]
    udd = np.concatenate([[0.0], np.cumsum(0.5 * (jerk[1:] + jerk[:-1]) * h)])
    ud = np.concatenate([[0.0], np.cumsum(0.5 * (udd[1:] + udd[:-1]) * h)])
    u = np.concatenate([[0.0], np.cumsum(0.5 * (ud[1:] + ud[:-1]) * h)])
    scale = 1.0 / u[-1]                                  # absorb the (tiny) numerical error so that u ends at exactly 1
    u, ud, udd = u * scale, ud * scale, udd * scale
    m = int(math.ceil(T / dt)) + 1
    ts = np.linspace(0.0, T, m)
    return ts, np.interp(ts, t, u), np.interp(ts, t, ud), np.interp(ts, t, udd)


def _dedupe(P):
    keep = [0]
    for i in range(1, len(P)):
        if np.linalg.norm(P[i] - P[keep[-1]]) > 1e-6:
            keep.append(i)
    return P[keep]


def smooth(positions, vmax, amax, vel_scale=0.35, acc_scale=0.25, dt=0.04, min_duration=0.6, max_dev=0.01, profile='min_jerk', jerk_max=8.0):
    """positions: (N, J) planned waypoints (rad). Returns {'t', 'q', 'qd', 'qdd', 'duration', 'path'} sampled
    every ~dt seconds, or None when the path has no length. vmax/amax: per-joint limits (rad/s, rad/s^2).
    profile 's_curve': a jerk-limited S-curve (jerk_max rad/s^3 per joint) with the SAME speed and acceleration limits; if the path's curvature makes
    the conservative bounds infeasible it falls back to the minimum-jerk law."""
    P = _dedupe(np.asarray(positions, dtype=float))
    if len(P) < 2:
        return None
    vmax, amax = np.asarray(vmax, float), np.asarray(amax, float)
    seg = np.linalg.norm(np.diff(P, axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    u = s / s[-1]

    lin = _LinearPath(u, P)
    path, kind = lin, 'linear'
    if len(P) >= 3:
        spl = _CubicPath(u, P)
        grid = np.linspace(0.0, 1.0, 400)
        if np.max(np.abs(spl(grid) - lin(grid))) <= max_dev:
            path, kind = spl, 'spline'

    grid = np.linspace(0.0, 1.0, 400)
    d1, d2 = np.abs(path(grid, 1)), np.abs(path(grid, 2))
    A1, A2 = d1.max(axis=0), d2.max(axis=0)              # rad per unit of path progress
    if profile == 's_curve':
        # Joint speed, acceleration and jerk along the path q(u(t)) are bounded with the path's own derivative maxima (conservative: every maximum is taken
        # over the whole path and the terms are added), so ONE shared S-curve u(t) keeps every joint inside its limits, straight or curved.
        vlim, alim = vmax * vel_scale, amax * acc_scale
        live = A1 > 1e-12                                # joints that move at all
        dd = np.diff(path(grid, 2), axis=0) / np.diff(grid)[:, None]
        A3 = np.abs(dd).max(axis=0)                      # q''' per unit of path progress (piecewise constant for a cubic spline, 0 for a line)
        V = float(np.min(vlim[live] / A1[live]))         # path progress per second
        bad = live & (A2 > 1e-12)
        if bad.any():                                    # curvature: keep the centripetal part q'' u'^2 under half of each joint's acceleration limit
            V = min(V, float(np.min(np.sqrt(0.5 * alim[bad] / A2[bad]))))
        A = float(np.min((alim[live] - A2[live] * V ** 2) / A1[live]))
        J = float(np.min((jerk_max - A3[live] * V ** 3 - 3.0 * A2[live] * V * max(A, 0.0)) / A1[live]))
        if V > 0 and A > 0 and J > 0:
            ts_, uu, ud, udd = scurve_profile(dt, V, A, J)
            if ts_[-1] < min_duration:                   # never faster than min_duration: stretch the same shape in time
                k = ts_[-1] / min_duration
                ts_, ud, udd = ts_ / k, ud * k, udd * k * k
            q = path(uu)
            qd = path(uu, 1) * ud[:, None]
            qdd = path(uu, 2) * (ud ** 2)[:, None] + path(uu, 1) * udd[:, None]
            q[0], q[-1] = P[0], P[-1]
            qd[0] = qd[-1] = 0.0
            qdd[0] = qdd[-1] = 0.0
            return {'t': ts_, 'q': q, 'qd': qd, 'qdd': qdd, 'duration': float(ts_[-1]), 'path': kind, 'profile': 's_curve'}
    t_vel = PEAK_U_DOT * np.max(A1 / (vmax * vel_scale))
    t_acc = math.sqrt(np.max((MIN_JERK_PEAK_U_DOT_SQ * A2 + PEAK_U_DDOT * A1) / (amax * acc_scale)))
    T = max(t_vel, t_acc, min_duration)

    n = int(math.ceil(T / dt)) + 1
    tau = np.linspace(0.0, 1.0, n)
    uu = 10 * tau ** 3 - 15 * tau ** 4 + 6 * tau ** 5
    ud = (30 * tau ** 2 - 60 * tau ** 3 + 30 * tau ** 4) / T
    udd = (60 * tau - 180 * tau ** 2 + 120 * tau ** 3) / T ** 2
    q = path(uu)
    qd = path(uu, 1) * ud[:, None]
    qdd = path(uu, 2) * (ud ** 2)[:, None] + path(uu, 1) * udd[:, None]
    q[0], q[-1] = P[0], P[-1]                            # exact ends
    qd[0] = qd[-1] = 0.0
    qdd[0] = qdd[-1] = 0.0
    return {'t': tau * T, 'q': q, 'qd': qd, 'qdd': qdd, 'duration': T, 'path': kind, 'profile': 'min_jerk'}
