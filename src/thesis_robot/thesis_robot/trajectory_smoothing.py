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


def _dedupe(P):
    keep = [0]
    for i in range(1, len(P)):
        if np.linalg.norm(P[i] - P[keep[-1]]) > 1e-6:
            keep.append(i)
    return P[keep]


def smooth(positions, vmax, amax, vel_scale=0.35, acc_scale=0.25, dt=0.04, min_duration=0.6, max_dev=0.01):
    """positions: (N, J) planned waypoints (rad). Returns {'t', 'q', 'qd', 'qdd', 'duration', 'path'} sampled
    every ~dt seconds, or None when the path has no length. vmax/amax: per-joint limits (rad/s, rad/s^2)."""
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
    return {'t': tau * T, 'q': q, 'qd': qd, 'qdd': qdd, 'duration': T, 'path': kind}
