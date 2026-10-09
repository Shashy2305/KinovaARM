import math

import pytest

from thesis_robot import grasp_planner as gp
from thesis_robot import wrist_servo as ws


def capsule_profile(length=0.105, width=0.062, long_axis=30.0):
    """Width/contact profile of a stadium-shaped object per closing direction (analytic)."""
    prof, cont = {}, {}
    r = width / 2.0
    for a in range(0, 180, 15):
        rel = math.radians(a - long_axis)
        # support function of a stadium along direction u: 2*(half_len*|cos| + r)
        prof[a] = 2 * ((length / 2 - r) * abs(math.cos(rel)) + r)
        # flat contact exists only when the closing axis is across the straight sides
        across = abs(math.cos(rel))                                # 0 when closing across the short axis
        cont[a] = max(0.0, (length - width) * (1 - across)) + 0.008
    return prof, cont


def test_wrap_and_delta_convention_matches_the_existing_alignment():
    for orient in (-45, -20, 15, 40, 75):
        existing = ws.rotation_to_align(orient, 2.0)
        mine = gp.delta_for_phi(orient + 90.0)                      # closing across the short side (orient + 90)
        assert existing is not None
        assert mine == pytest.approx(existing, abs=1e-6)
    assert ws.rotation_to_align(-80, 2.0) is None                    # |turn| = 10 deg < MIN_TURN_DEG: not worth a turn
    assert gp.delta_for_phi(-80 + 90.0) == pytest.approx(-10.0)
    assert gp.wrap90(100) == -80 and gp.wrap90(-90) == 90 and gp.wrap90(0) == 0


def test_best_grasp_for_an_elongated_object_is_across_its_short_side():
    prof, cont = capsule_profile(long_axis=30.0)
    best = gp.best_grasp(prof, cont, long_axis_deg=30.0)
    assert gp.angle_diff(best['phi'], 30.0 + 90.0) <= 15            # nearest sampled direction to the short axis
    assert best['width_m'] < 0.075


def test_a_retry_chooses_a_different_entry_angle_not_the_same_one():
    prof, cont = capsule_profile(long_axis=30.0)
    first = gp.best_grasp(prof, cont, long_axis_deg=30.0)
    second = gp.best_grasp(prof, cont, long_axis_deg=30.0, tried_rho=[first['rho']])
    assert second is not None
    assert gp.angle_diff(second['rho'], first['rho']) >= gp.DEFAULT_MIN_SEP_DEG
    assert second['score'] <= first['score']


def test_tried_angles_are_remembered_relative_to_the_object_not_the_image():
    prof, cont = capsule_profile(long_axis=30.0)
    first = gp.best_grasp(prof, cont, long_axis_deg=30.0)
    # the wrist turned 45 deg between attempts: the object now appears rotated, its long axis is at 75
    prof2, cont2 = capsule_profile(long_axis=75.0)
    ranked = gp.rank_grasps(prof2, cont2, long_axis_deg=75.0, tried_rho=[first['rho']])
    assert all(gp.angle_diff(c['rho'], first['rho']) >= gp.DEFAULT_MIN_SEP_DEG for c in ranked)
    # and the same physical direction is not offered again
    assert not any(gp.angle_diff(gp.phi_of(first['rho'], 75.0), c['phi']) < 15 for c in ranked)


def test_too_wide_and_neighbour_blocked_directions_are_left_out():
    prof = {0: 0.14, 45: 0.09, 90: 0.08, 135: 0.10}
    cont = {0: 0.0, 45: 0.02, 90: 0.03, 135: 0.02}
    ranked = gp.rank_grasps(prof, cont, blocked=lambda phi: phi == 90)
    phis = [c['phi'] for c in ranked]
    assert 0 not in phis and 90 not in phis and 45 in phis and 135 in phis


def test_a_handle_in_the_way_lowers_a_direction():
    prof = {0: 0.08, 90: 0.08}
    cont = {0: 0.03, 90: 0.03}
    clear = gp.rank_grasps(prof, cont)
    with_handle = gp.rank_grasps(prof, cont, handle_bearing_deg=0.0)       # a handle on the image-x axis
    assert [c['phi'] for c in with_handle][0] == 90
    assert with_handle[-1]['score'] < clear[-1]['score']


def test_nothing_fits_returns_empty():
    assert gp.rank_grasps({0: 0.16, 90: 0.15}, {0: 0.05, 90: 0.05}) == []


def test_string_keys_from_json_work():
    prof, cont = capsule_profile()
    s_prof = {str(k): v for k, v in prof.items()}
    s_cont = {str(k): v for k, v in cont.items()}
    assert gp.best_grasp(s_prof, s_cont, long_axis_deg=30.0) is not None


def test_end_to_end_with_the_shape_analysis_of_a_synthetic_mouse():
    cv2 = pytest.importorskip('cv2')
    import numpy as np
    from thesis_robot import shape_analysis as sa
    from test_shape_analysis import background, capsule, draw_poly, box_of
    img = background()
    poly = capsule(240, 180, 140, 60, 30)
    draw_poly(img, poly)
    a = sa.analyse(img, box_of(poly), polygon=poly)
    prof, cont = gp.shape_to_metres(a, distance_m=1.3, fx=1300.0)
    best = gp.best_grasp(prof, cont, long_axis_deg=a['long_axis_deg'])
    assert gp.angle_diff(best['phi'], a['short_axis_deg']) <= 15
    assert 0.05 < best['width_m'] < 0.075                          # 60 px at 1.3 m / 1300 px = 60 mm
