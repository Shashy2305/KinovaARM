import numpy as np

from thesis_robot import trajectory_smoothing as ts

VMAX = np.array([1.3963] * 4 + [1.2218] * 3)
AMAX = np.full(7, 8.6)


def straight(delta=0.8, n=2):
    a = np.zeros(7)
    b = np.array([delta, -delta / 2, 0.0, delta / 3, 0.0, 0.0, 0.0])
    return np.array([a + (b - a) * i / (n - 1) for i in range(n)])


def test_ends_are_exact_and_rest_at_both_ends():
    P = straight(n=5)
    r = ts.smooth(P, VMAX, AMAX)
    assert np.allclose(r['q'][0], P[0]) and np.allclose(r['q'][-1], P[-1])
    assert np.allclose(r['qd'][[0, -1]], 0) and np.allclose(r['qdd'][[0, -1]], 0)


def test_limits_are_respected():
    for delta in (0.05, 0.4, 1.5):
        r = ts.smooth(straight(delta), VMAX, AMAX, vel_scale=0.35, acc_scale=0.25)
        assert (np.abs(r['qd']) <= VMAX * 0.35 * 1.02).all()
        assert (np.abs(r['qdd']) <= AMAX * 0.25 * 1.02).all()


def test_no_acceleration_step_unlike_a_trapezoid():
    r = ts.smooth(straight(1.0), VMAX, AMAX)
    jerk = np.abs(np.diff(r['qdd'], axis=0) / np.diff(r['t'])[:, None]).max()
    assert jerk < 5 * AMAX[0] * 0.25 / (r['duration'] / 4)     # smooth: far below a step (a step is ~ amax/dt)
    assert np.abs(np.diff(r['qdd'], axis=0)).max() < 0.2        # per-sample change in acceleration stays small


def test_short_moves_take_at_least_the_minimum_duration():
    r = ts.smooth(straight(0.01), VMAX, AMAX, min_duration=0.6)
    assert r['duration'] >= 0.6


def test_faster_scale_is_shorter():
    slow = ts.smooth(straight(1.0), VMAX, AMAX, vel_scale=0.2)['duration']
    fast = ts.smooth(straight(1.0), VMAX, AMAX, vel_scale=0.5)['duration']
    assert fast < slow


def test_collinear_waypoints_follow_the_same_line():
    P = straight(0.9, n=6)
    r = ts.smooth(P, VMAX, AMAX)
    d = P[-1] - P[0]
    frac = (r['q'] - P[0]) @ d / (d @ d)
    perp = (r['q'] - P[0]) - frac[:, None] * d
    assert np.abs(perp).max() < 1e-6
    assert (np.diff(frac) >= -1e-9).all()                       # never backs up


def test_duplicate_points_and_no_motion():
    P = np.vstack([straight(0.5, 3), straight(0.5, 3)[-1:]])
    assert ts.smooth(P, VMAX, AMAX) is not None
    assert ts.smooth(np.zeros((4, 7)), VMAX, AMAX) is None


def test_a_spline_that_strays_falls_back_to_the_straight_lines():
    P = np.zeros((4, 7))
    P[1, 0], P[2, 0], P[3, 0] = 0.5, 0.0, 0.5            # zig-zag: a natural spline overshoots the polyline
    r = ts.smooth(P, VMAX, AMAX, max_dev=0.001)
    assert r['path'] == 'linear'


def test_cubic_path_interpolates_and_has_continuous_derivatives():
    u = np.array([0.0, 0.2, 0.5, 0.7, 1.0])
    P = np.column_stack([np.sin(u * 3), u ** 2, np.cos(u)])
    c = ts._CubicPath(u, P)
    assert np.allclose(c(u), P, atol=1e-9)
    eps = 1e-6
    for k in (0.2, 0.5, 0.7):
        for nu in (1, 2):
            assert np.allclose(c(k - eps, nu), c(k + eps, nu), atol=1e-3)


def test_a_higher_speed_scale_shortens_a_long_transit_only_when_asked():
    P = straight(1.4, n=6)
    slow = ts.smooth(P, VMAX, AMAX, vel_scale=0.30)['duration']
    fast = ts.smooth(P, VMAX, AMAX, vel_scale=0.45)['duration']
    assert fast < 0.8 * slow
    assert (np.abs(ts.smooth(P, VMAX, AMAX, vel_scale=0.45)['qd']) <= VMAX * 0.45 * 1.02).all()
