import numpy as np
from scipy.spatial.transform import Rotation as R

from thesis_robot import wrist_servo as ws

K = np.array([[1297.7, 0, 620.9], [0, 1298.6, 238.3], [0, 0, 1]])


def camera_over(flange_xy, flange_z, yaw_deg=0.0):
    """Wrist camera pose for a flange pointing straight down at (x, y, z): the
    optical axis is the tool axis (down), mounted 56 mm off-axis (URDF camera_module)."""
    R_ee = R.from_euler('x', 180, degrees=True) * R.from_euler('z', yaw_deg, degrees=True)   # tool z points down
    R_cam = R_ee * R.from_euler('z', 180, degrees=True)         # camera_module rpy 0 0 pi
    t = np.array([flange_xy[0], flange_xy[1], flange_z]) + R_ee.apply([0, 0.05639, 0.01305])
    return R_cam.as_matrix(), t


def project(P, Rm, t):
    c = Rm.T @ (np.asarray(P) - t)
    return K[0, 0] * c[0] / c[2] + K[0, 2], K[1, 1] * c[1] / c[2] + K[1, 2]


def test_recovers_object_position_on_the_plane():
    Rm, t = camera_over((0.45, 0.14), 0.40)
    plane = ws.plane_height('cup', -0.0125)
    for obj in [(0.42, 0.12), (0.47, 0.17), (0.40, 0.10)]:
        u, v = project([obj[0], obj[1], plane], Rm, t)
        x, y = ws.pixel_to_plane_xy(u, v, K, Rm, t, plane)
        assert abs(x - obj[0]) < 1e-6 and abs(y - obj[1]) < 1e-6


def test_wrong_height_assumption_gives_a_small_error_when_looking_down():
    Rm, t = camera_over((0.42, 0.12), 0.40)
    u, v = project([0.45, 0.14, 0.04], Rm, t)                   # object top really at z=0.04
    x, y = ws.pixel_to_plane_xy(u, v, K, Rm, t, 0.0775)          # we assumed 0.0775
    assert np.hypot(x - 0.45, y - 0.14) < 0.012


def test_ray_that_misses_the_plane_is_rejected():
    Rm, t = camera_over((0.42, 0.12), 0.40)
    assert ws.pixel_to_plane_xy(620 + 1297 * 20, 238, K, Rm, t, 0.05) is None   # almost parallel to the table
    assert ws.pixel_to_plane_xy(620, 238, K, Rm, t, 0.9) is None          # plane behind the camera


def test_step_is_clipped_and_reports_the_true_distance():
    dx, dy, dist = ws.clipped_step((0.5, 0.2), (0.4, 0.2), 0.06)
    assert abs(dx - 0.06) < 1e-9 and dy == 0 and abs(dist - 0.10) < 1e-9
    dx, dy, dist = ws.clipped_step((0.41, 0.2), (0.4, 0.2), 0.06)
    assert abs(dx - 0.01) < 1e-9


def test_detection_choice():
    dets = [{'label': 'cup', 'confidence': 0.9, 'u': 100, 'v': 100},
            {'label': 'cup', 'confidence': 0.5, 'u': 640, 'v': 360},
            {'label': 'mouse', 'confidence': 0.99, 'u': 0, 'v': 0}]
    assert ws.pick_detection(dets, 'cup')['confidence'] == 0.9
    assert ws.pick_detection(dets, 'cup', expected_uv=(650, 350))['confidence'] == 0.5
    assert ws.pick_detection(dets, 'bottle') is None


def test_search_pattern_is_small_and_covers_all_sides():
    offs = ws.search_offsets(0.04)
    assert all(max(abs(a), abs(b)) <= 0.04 for a, b in offs)
    assert {(0.04, 0), (-0.04, 0), (0, 0.04), (0, -0.04)} <= set(offs)


def test_handle_makes_the_box_wider_than_tall():
    assert ws.needs_quarter_turn((447, 232, 990, 646))        # the mug seen from above, handle on the left
    assert not ws.needs_quarter_turn((300, 200, 700, 600))     # round object
    assert not ws.needs_quarter_turn((300, 200, 700, 800))     # long along y: already fine


def test_quarter_turn_stays_inside_the_joint_limit():
    import math
    assert abs(ws.quarter_turn_target(0.0, 2.7)) == math.pi / 2
    assert ws.quarter_turn_target(1.57, 2.7) == 1.57 - 3.141592653589793 / 2     # +90 would hit 3.14
    assert ws.quarter_turn_target(2.6, 2.7) < 2.6
    assert ws.quarter_turn_target(2.0, 0.3) is None


REACH, TOP, FLOOR = 0.215, -0.0125, 0.2525


def _heights(label, oz, tip_clear=0.05, **kw):
    return ws.pick_heights(oz, label, TOP, REACH, FLOOR, tip_clear, 0.05, **kw)


def test_mug_heights_are_unchanged_from_the_live_tested_values():
    grasp, hover, lift, _ = _heights('cup', 0.056, approach_z=0.39)
    assert abs(grasp - 0.271) < 0.002 and abs(hover - 0.391) < 0.002 and abs(lift - 0.391) < 0.002


def test_a_tall_bottle_is_hovered_over_with_the_fingertips_clear_of_its_top():
    grasp, hover, lift, _ = _heights('bottle', 0.129, approach_z=0.46)
    top = TOP + 0.22
    assert hover - REACH >= top + 0.07 - 1e-9                      # tips at least 7 cm above the cap
    assert hover <= ws.HOVER_MAX_Z
    assert abs((grasp - REACH) - (TOP + 0.3 * 0.22)) < 1e-9         # tips on the straight lower body, not the shoulder
    assert lift == min(hover, grasp + 0.15) and lift < hover        # carry only a little above the grasp


def test_a_mouse_is_grasped_low_only_when_the_clearance_is_lowered():
    g_default, *_ = _heights('mouse', 0.0)
    assert g_default == FLOOR                                         # unchanged: tips stay 5 cm up
    g_low, hover, lift, low_floor = _heights('mouse', 0.0, tip_clear=0.03)
    assert g_low < FLOOR and g_low >= low_floor - 1e-9


def test_a_bottle_the_detector_calls_a_cup_is_still_found_by_its_position():
    exp = (0.452, 0.140)
    cands = [({'label': 'cup', 'confidence': 0.60}, (0.455, 0.147)),          # the bottle cap seen from above
             ({'label': 'mouse', 'confidence': 0.85}, (0.455, -0.02)),        # the real mouse, 16 cm away
             ({'label': 'cup', 'confidence': 0.82}, (0.57, 0.02))]            # the mug
    d, xy = ws.match_detection(cands, 'bottle', exp)
    assert d['confidence'] == 0.60 and xy == (0.455, 0.147)


def test_nothing_near_the_expected_spot_is_not_a_match():
    cands = [({'label': 'cup', 'confidence': 0.9}, (0.57, 0.02)), ({'label': 'mouse', 'confidence': 0.9}, (0.455, -0.02))]
    assert ws.match_detection(cands, 'bottle', (0.452, 0.140)) is None
    assert ws.match_detection([], 'bottle', (0.452, 0.140)) is None
    assert ws.match_detection(cands, 'bottle', None) is None                # no expected spot and wrong class


def test_the_expected_class_wins_when_it_is_near_enough():
    cands = [({'label': 'bottle', 'confidence': 0.4}, (0.50, 0.19)),          # 7 cm off: too far for "any class" but the right class
             ({'label': 'cup', 'confidence': 0.9}, (0.452, 0.141))]
    assert ws.match_detection(cands, 'bottle', (0.452, 0.140))[0]['label'] == 'bottle'


def test_tall_bottle_real_measurement_needs_the_mid_height_plane():
    """Regression from the real arm (2026-10-05, plastic water bottle). Flange (0.414, 0.188, 0.492); the wrist
    camera sits 5.6 cm from the tool axis along base +x. YOLO's box centre was pixel (600, 540). The bottle's
    real axis, from its cap pixel (595, 610) on the cap plane, was (0.387, 0.178). Centring with the TOP plane
    left the fingers about 2 cm off sideways (2.7 cm measured with the finger pads); the mid-height plane is within 1 cm."""
    Rm = R.from_quat([0.7298, -0.68323, 0.0164, 0.01793]).as_matrix()
    t = np.array([0.414 + 0.0563, 0.188 + 0.0031, 0.492 - 0.0132])
    axis = np.array([0.387, 0.178])
    top = np.array(ws.pixel_to_plane_xy(600, 540, K, Rm, t, ws.plane_height('bottle', -0.0125)))
    mid = np.array(ws.pixel_to_plane_xy(600, 540, K, Rm, t, ws.center_plane_height('bottle', -0.0125)))
    assert np.linalg.norm(top - axis) > 0.018
    assert np.linalg.norm(mid - axis) < 0.012


def test_short_objects_keep_the_top_plane():
    assert ws.center_plane_height('cup', -0.0125) == ws.plane_height('cup', -0.0125)
    assert ws.center_plane_height('mouse', -0.0125) == ws.plane_height('mouse', -0.0125)


def test_a_wildly_wrong_scene_height_is_not_trusted():
    """2026-10-05: the scene said a cup was at z=0.132; the real mug centre is ~0.04. The arm grasped 8 cm
    above the mug and closed on air."""
    g_bad, *_ = _heights('cup', 0.132, approach_z=0.467)
    g_ok, *_ = _heights('cup', 0.056, approach_z=0.39)
    assert abs(g_bad - 0.2525) < 0.02 and abs(g_ok - 0.271) < 0.002       # the bad reading gives the same low grasp, not 0.347
    assert g_bad < 0.30


def test_a_plausible_scene_height_is_still_used():
    g, *_ = _heights('cup', 0.07)
    assert abs(g - (0.07 + REACH)) < 1e-9


def test_open_fingers_sweeping_a_neighbour_are_detected():
    centre = (0.51, 0.15)                                  # the mug
    bottle = ('bottle', 0.45, 0.24)                         # ~11 cm away, up and to the left
    along_y = ws.finger_sweep_blocker(centre, (0.0, 1.0), [bottle])    # closing axis along +y points at it
    along_x = ws.finger_sweep_blocker(centre, (1.0, 0.0), [bottle])
    assert along_y == 'bottle' and along_x is None
    assert ws.finger_sweep_blocker(centre, (0.0, 1.0), [('cup', 0.51, 0.15)]) is None      # the target itself
    assert ws.finger_sweep_blocker(centre, (0.0, 1.0), [('mouse', 0.30, 0.15)]) is None    # far away


def test_a_path_that_only_moves_away_from_a_neighbour_is_allowed():
    bottle = [('bottle', 0.45, 0.38)]
    start, end = (0.43, 0.28), (0.526, 0.227)              # the 2026-10-05 refusal: mug 10 cm from the bottle, moving away
    assert ws.carry_path_blocker(start, end, bottle) is None
    assert ws.carry_path_blocker(start, (0.45, 0.33), bottle) == 'bottle'   # heading at it is still refused
    assert ws.carry_path_blocker((0.43, 0.30), (0.43, 0.30), [('cup', 0.43, 0.30)]) is None   # the carried object


def _rect(cx, cy, w, h, angle_deg):
    a = np.radians(angle_deg)
    R_ = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
    return [tuple(np.array([cx, cy]) + R_ @ np.array(p)) for p in ((-w / 2, -h / 2), (w / 2, -h / 2), (w / 2, h / 2), (-w / 2, h / 2))]


def test_polygon_orientation_recovers_angle_and_elongation():
    for ang in (0, 30, 45, 75, -30, -60):
        a, e = ws.polygon_orientation(_rect(600, 400, 200, 100, ang))      # long side along `ang`
        assert abs(((a - ang + 90) % 180) - 90) < 1.0, (ang, a)
        assert abs(e - 2.0) < 0.05
    a, e = ws.polygon_orientation(_rect(600, 400, 150, 150, 20))            # square: no preferred axis
    assert e < 1.05
    a, e = ws.polygon_orientation(_rect(600, 400, 100, 200, 0))             # long side along image y
    assert abs(abs(a) - 90) < 1.0


def test_rotation_to_align_puts_the_long_side_vertical():
    assert abs(ws.rotation_to_align(0.0, 2.0)) == 90.0                  # long along x: turn a quarter
    assert abs(ws.rotation_to_align(45.0, 1.8) - 45.0) < 1e-9          # the 45 degree mouse: its box looked square
    assert abs(ws.rotation_to_align(-45.0, 1.8) + 45.0) < 1e-9
    assert ws.rotation_to_align(88.0, 2.0) is None and ws.rotation_to_align(-85.0, 2.0) is None   # already vertical
    assert ws.rotation_to_align(30.0, 1.1) is None                      # round: nothing to align
    assert ws.rotation_to_align(None, None) is None
    for ang in range(-89, 91, 7):                                       # after the turn the long axis is vertical (mod 180)
        d = ws.rotation_to_align(float(ang), 2.0)
        if d is not None:
            assert abs(d) <= 90.0 and abs(((ang + d) % 180) - 90) < 1e-6
