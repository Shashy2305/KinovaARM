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


# ── the jerk-limited S-curve profile (smooth_profile: s_curve) ─────────────────────────────────────────────────────────────
VMAX7 = [1.3963] * 4 + [1.2218] * 3
AMAX7 = [8.6] * 7


def _limits_hold(r, vlim, alim, jlim, tol=1.03):
    q, qd, qdd, t = r['q'], r['qd'], r['qdd'], r['t']
    dt = np.diff(t)[:, None]
    jerk = np.diff(qdd, axis=0) / dt
    return (np.all(np.abs(qd) <= tol * np.asarray(vlim)) and np.all(np.abs(qdd) <= tol * np.asarray(alim)) and np.all(np.abs(jerk) <= tol * jlim))


def test_the_s_curve_keeps_every_joint_inside_speed_acceleration_and_jerk_limits():
    rng = np.random.default_rng(5)
    for _ in range(20):
        start = rng.uniform(-1.5, 1.5, 7)
        goal = start + rng.uniform(-2.5, 2.5, 7)
        r = ts.smooth([start, goal], VMAX7, AMAX7, vel_scale=0.45, acc_scale=0.20, profile='s_curve', jerk_max=8.0)
        assert r['profile'] == 's_curve'
        assert _limits_hold(r, 0.45 * np.array(VMAX7), 0.20 * np.array(AMAX7), 8.0)


def test_the_s_curve_starts_and_stops_at_rest_and_ends_exactly_on_the_goal():
    start, goal = np.zeros(7), np.array([2.0, -1.0, 0.5, 1.5, -0.7, 0.9, 1.2])
    r = ts.smooth([start, goal], VMAX7, AMAX7, vel_scale=0.30, acc_scale=0.20, profile='s_curve')
    assert np.allclose(r['q'][0], start) and np.allclose(r['q'][-1], goal)
    assert np.allclose(r['qd'][0], 0) and np.allclose(r['qd'][-1], 0) and np.allclose(r['qdd'][0], 0) and np.allclose(r['qdd'][-1], 0)
    assert np.all(np.diff(r['q'][:, 0]) >= -1e-9)                # monotone along a straight path: no overshoot


def test_the_s_curve_is_much_shorter_than_minimum_jerk_under_the_same_limits():
    start, goal = np.zeros(7), np.array([2.15, 0.3, 0.2, 0.5, 0.1, 0.4, 0.3])        # a typical 2 rad hover swing
    a = ts.smooth([start, goal], VMAX7, AMAX7, vel_scale=0.45, acc_scale=0.20, profile='min_jerk')
    b = ts.smooth([start, goal], VMAX7, AMAX7, vel_scale=0.45, acc_scale=0.20, profile='s_curve', jerk_max=8.0)
    assert b['duration'] < 0.75 * a['duration']                  # measured 2026-10-20 on 24 moves: 4.45 s against 7.35 s (39% shorter)
    # and it is never slower than the cap allows: the critical joint's mean speed cannot exceed the speed limit
    assert 2.15 / b['duration'] <= 0.45 * VMAX7[0] * 1.001


def test_a_short_move_that_cannot_reach_the_speed_cap_still_respects_acceleration_and_jerk():
    r = ts.smooth([np.zeros(7), np.full(7, 0.05)], VMAX7, AMAX7, vel_scale=0.45, acc_scale=0.20, profile='s_curve', jerk_max=8.0, min_duration=0.0)
    assert _limits_hold(r, 0.45 * np.array(VMAX7), 0.20 * np.array(AMAX7), 8.0)
    assert np.allclose(r['q'][-1], 0.05)


def test_the_min_duration_floor_still_applies_to_the_s_curve():
    r = ts.smooth([np.zeros(7), np.full(7, 0.01)], VMAX7, AMAX7, vel_scale=0.45, acc_scale=0.20, profile='s_curve', min_duration=0.6)
    assert r['duration'] >= 0.6 - 1e-9


def test_a_ptp_style_path_of_many_collinear_waypoints_gets_the_s_curve():
    """What MoveIt/Pilz hands over: dozens of waypoints on one straight joint-space line (the smoother calls that a 'spline')."""
    a, b = np.zeros(7), np.array([2.15, 0.3, 0.2, 0.5, 0.1, 0.4, 0.3])
    P = [a + (b - a) * k / 40 for k in range(41)]
    mj = ts.smooth(P, VMAX7, AMAX7, vel_scale=0.45, acc_scale=0.20)
    sc = ts.smooth(P, VMAX7, AMAX7, vel_scale=0.45, acc_scale=0.20, profile='s_curve', jerk_max=8.0)
    assert sc['profile'] == 's_curve' and sc['duration'] < 0.75 * mj['duration']
    assert _limits_hold(sc, 0.45 * np.array(VMAX7), 0.20 * np.array(AMAX7), 8.0)


def test_a_curved_path_keeps_every_limit_whichever_profile_is_chosen():
    t = np.linspace(0, 1, 9)
    path = [np.array([0.8 * np.sin(2.5 * k), 0.6 * k, 0.4 * np.cos(2 * k), 0.0, 0.2 * k, 0.0, 0.0]) for k in t]
    r = ts.smooth(path, VMAX7, AMAX7, vel_scale=0.45, acc_scale=0.20, profile='s_curve', jerk_max=8.0)
    assert r['profile'] in ('s_curve', 'min_jerk')
    assert _limits_hold(r, 0.45 * np.array(VMAX7), 0.20 * np.array(AMAX7), 8.0, tol=1.05)
    assert np.allclose(r['q'][-1], path[-1]) and np.allclose(r['qd'][-1], 0)


def test_the_default_profile_is_unchanged_minimum_jerk():
    start, goal = np.zeros(7), np.array([1.0, 0.2, 0.0, 0.3, 0.0, 0.1, 0.0])
    r = ts.smooth([start, goal], VMAX7, AMAX7)
    assert r['profile'] == 'min_jerk'
