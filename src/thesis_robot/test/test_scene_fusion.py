"""Scene graph fusion (2026-10-28): the published position must not flip between the cameras' slightly different readings, a moved object is followed at once, bad-depth
sources do not set heights, and the wrist camera only counts while it looks at the table."""
import math
from types import SimpleNamespace

import pytest

pytest.importorskip('rclpy')
tf2_ros = pytest.importorskip('tf2_ros')
from geometry_msgs.msg import TransformStamped  # noqa: E402

from thesis_robot import scene_graph_node as sgn  # noqa: E402
from thesis_robot.scene_graph_node import SceneGraphNode  # noqa: E402


def world(x=0.40, y=0.10, z=0.045):
    return SimpleNamespace(scene={'cup_00': {'label': 'cup', 'x': x, 'y': y, 'z': z, 'confidence': 0.9, 'stale': False}}, _obs={})


def feed(w, readings, t0=100.0, dt=0.04):
    """readings: [(source, x, y, z, conf)] in arrival order; returns the published (x, y, z) after each."""
    out, t = [], t0
    for src, x, y, z, conf in readings:
        fx, fy, fz, _c = SceneGraphNode._fuse_position(w, 'cup_00', x, y, z, conf, t, src)
        w.scene['cup_00'].update(x=fx, y=fy, z=fz)
        out.append((fx, fy, fz))
        t += dt
    return out


def test_two_cameras_that_disagree_by_2_cm_give_a_steady_estimate_not_a_flip():
    w = world(0.41, 0.09)
    rs1, rs2 = ('realsense', 0.411, 0.100, 0.047, 0.71), ('realsense2', 0.410, 0.080, 0.045, 0.93)
    out = feed(w, [rs1, rs2] * 40)
    ys = [o[1] for o in out[20:]]
    assert max(ys) - min(ys) < 0.003                                         # the old code swung by ~1.7 cm on every message
    assert 0.080 < sum(ys) / len(ys) < 0.100                                 # between the two, nearer the more confident one
    assert abs(sum(ys) / len(ys) - 0.0888) < 0.004


def test_the_old_behaviour_would_have_flipped(monkeypatch):
    # documents the failure: alpha ~ confidence meant the estimate followed the last reading
    est, ys = 0.09, []
    for src_y, conf in [(0.100, 0.71), (0.080, 0.93)] * 20:
        a = min(0.85, max(0.15, conf))
        est = a * src_y + (1 - a) * est
        ys.append(est)
    assert max(ys[10:]) - min(ys[10:]) > 0.01


def test_a_moved_object_is_followed_at_once():
    w = world(0.488, 0.199)
    out = feed(w, [('realsense2', 0.41, 0.08, 0.045, 0.93)])
    assert out[0][:2] == pytest.approx((0.41, 0.08), abs=0.001)             # 14 cm away: no slow drift over 25 s


def test_heights_come_from_the_good_depth_cameras_not_rs3_or_the_wrist():
    w = world(0.41, 0.09, 0.045)
    out = feed(w, [('realsense2', 0.41, 0.09, 0.045, 0.9), ('realsense3', 0.41, 0.09, -0.19, 0.9), ('wrist', 0.41, 0.09, -0.02, 0.8)] * 30)
    assert out[-1][2] == pytest.approx(0.045, abs=0.003)


def test_an_object_only_a_poor_depth_camera_sees_still_gets_its_height():
    w = world(0.41, 0.09, 0.045)
    out = feed(w, [('realsense3', 0.41, 0.09, 0.03, 0.9)] * 60)
    assert out[-1][2] == pytest.approx(0.03, abs=0.003)


def test_a_camera_that_stopped_reporting_stops_counting_after_two_seconds():
    w = world(0.41, 0.09)
    feed(w, [('realsense', 0.43, 0.12, 0.045, 0.9)], t0=100.0)
    out = feed(w, [('realsense2', 0.41, 0.08, 0.045, 0.9)] * 60, t0=110.0)    # 10 s later: RS1's old reading has expired
    assert out[-1][:2] == pytest.approx((0.41, 0.08), abs=0.002)


def test_a_new_or_stale_track_takes_the_reading_as_it_is():
    w = world()
    w.scene['cup_00']['stale'] = True
    fx, fy, fz, conf = SceneGraphNode._fuse_position(w, 'cup_00', 0.5, 0.2, 0.05, 0.8, 5.0, 'realsense')
    assert (fx, fy, fz, conf) == (0.5, 0.2, 0.05, 0.8)
    assert 'cup_09' not in w.scene
    assert SceneGraphNode._fuse_position(SimpleNamespace(scene={}, _obs={}), 'cup_09', 0.1, 0.2, 0.0, 0.7, 5.0, None)[:2] == (0.1, 0.2)


def tf_pointing(axis_z_down):
    """A buffer whose 'wrist_cam' optical axis (frame z) points down (axis_z_down=True) or forward."""
    buf = tf2_ros.Buffer()
    m = TransformStamped()
    m.header.frame_id, m.child_frame_id = 'base_link', 'wrist_cam'
    m.transform.translation.x = 0.4
    if axis_z_down:
        m.transform.rotation.x = 1.0                                           # 180 deg about x: frame z -> world -z
    else:
        m.transform.rotation.x = math.sqrt(0.5)                                # 90 deg about x: frame z -> world -y (horizontal)
        m.transform.rotation.w = math.sqrt(0.5)
    if axis_z_down:
        m.transform.rotation.w = 0.0
    buf.set_transform_static(m, 'test')
    return SimpleNamespace(tf_buffer=buf)


def test_the_wrist_camera_counts_only_when_it_looks_down_at_something_not_too_close():
    det = {'frame_id': 'wrist_cam', 'cz_3d': 0.45}
    assert SceneGraphNode._wrist_usable(tf_pointing(True), det) is True
    assert SceneGraphNode._wrist_usable(tf_pointing(False), det) is False                # looking at the room
    assert SceneGraphNode._wrist_usable(tf_pointing(True), dict(det, cz_3d=0.15)) is False   # too close: its depth is empty / wrong
    assert SceneGraphNode._wrist_usable(tf_pointing(True), {'frame_id': 'nope', 'cz_3d': 0.5}) is False


def test_constants_are_sane():
    assert 0.0 < sgn.SMOOTH_ALPHA < 1.0 and sgn.JUMP_M > 0.03 and sgn.SOURCE_WINDOW_S >= 1.0
    assert sgn.SOURCE_WEIGHT['realsense3'] < sgn.SOURCE_WEIGHT['realsense']


# ---- surface point -> object centre (2026-10-29) ---------------------------------------------------------------------------------------------------------
def tf_at_position(x, y, z=0.2):
    buf = tf2_ros.Buffer()
    m = TransformStamped()
    m.header.frame_id, m.child_frame_id = 'base_link', 'rs_cam'
    m.transform.translation.x, m.transform.translation.y, m.transform.translation.z = x, y, z
    m.transform.rotation.w = 1.0
    buf.set_transform_static(m, 'test')
    return SimpleNamespace(tf_buffer=buf)


def test_a_static_camera_point_moves_away_from_the_camera_by_the_object_radius():
    cam = tf_at_position(0.25, -1.19)                                              # RS2-like: looks along +y
    x, y = SceneGraphNode._to_object_centre(cam, {'frame_id': 'rs_cam'}, 'cup', 0.404, 0.063)
    from thesis_robot import wrist_servo as ws
    r = ws.OBJECT_RADIUS_M['cup']
    assert math.hypot(x - 0.404, y - 0.063) == pytest.approx(r, abs=1e-6)
    assert y > 0.063 and (y - 0.063) > abs(x - 0.404)                              # mostly along +y, away from the camera
    # a bigger object moves further; an unknown label uses the default radius
    xb, yb = SceneGraphNode._to_object_centre(cam, {'frame_id': 'rs_cam'}, 'bowl', 0.404, 0.063)
    assert math.hypot(xb - 0.404, yb - 0.063) > math.hypot(x - 0.404, y - 0.063)


def test_no_shift_when_the_camera_position_is_unknown_or_the_point_is_at_the_camera():
    cam = tf_at_position(0.25, -1.19)
    assert SceneGraphNode._to_object_centre(cam, {'frame_id': 'nope'}, 'cup', 0.4, 0.06) == (0.4, 0.06)
    assert SceneGraphNode._to_object_centre(tf_at_position(0.4, 0.06), {'frame_id': 'rs_cam'}, 'cup', 0.4, 0.06) == (0.4, 0.06)


def test_the_wrist_camera_is_not_in_the_static_sources():
    assert 'wrist' not in sgn.STATIC_SOURCES and 'realsense2' in sgn.STATIC_SOURCES


# ---- the colour vote ---------------------------------------------------------------------------------------------------------------------------------------
def test_a_clear_colour_from_one_camera_beats_black_from_another():
    vote = SceneGraphNode._colour_from_votes
    assert vote({'blue': 0.4, 'black': 1.0}) == 'blue'                 # RS1 sees blue, RS2 (underexposed) says black, RS2 reports more often
    assert vote({'blue': 0.4, 'black': 2.0}) == 'blue'                 # even when RS2 reports 2.5x as often (live 2026-10-30: 14 Hz black against 6 Hz blue)
    assert vote({'black': 2.0, 'blue': 0.2}) == 'black'                # a stray blue frame is not 15 % of the weight
    assert vote({'white': 1.0, 'grey': 0.5}) == 'white'
    assert vote({'grey': 0.8, 'black': 1.0}) == 'grey'                 # among neutrals black counts 0.6
    assert vote({'white': 0.9}) == 'white' and vote({}) is None


def test_the_vote_keeps_a_stable_colour_for_a_track():
    node = SimpleNamespace(_colour_votes={})
    node._colour_from_votes = SceneGraphNode._colour_from_votes
    out = []
    for i in range(60):
        out.append(SceneGraphNode._vote_colour(node, 'cup_00', 'blue' if i % 2 == 0 else 'black', 0.94 if i % 2 == 0 else 0.82))
    assert set(out[10:]) == {'blue'}


def blob_world():
    return SimpleNamespace(scene={
        'cup_00': {'label': 'cup', 'x': 0.40, 'y': 0.10, 'z': 0.045, 'confidence': 0.9, 'stale': False},
        'red_obj_00': {'label': 'red_obj', 'x': 0.42, 'y': 0.11, 'z': 0.04, 'confidence': 0.6, 'stale': False},
        'blue_obj_00': {'label': 'blue_obj', 'x': 0.55, 'y': -0.20, 'z': 0.04, 'confidence': 0.6, 'stale': False},
    })


def test_a_colour_blob_on_a_recognised_object_is_dropped_from_the_scene_and_from_new_detections():
    w = blob_world()
    new = [('red_obj', 0.41, 0.09, 0.04, 0.7), ('green_obj', 0.50, 0.30, 0.04, 0.7), ('cup', 0.40, 0.10, 0.045, 0.9)]
    kept = SceneGraphNode._drop_blobs_on_known_objects(w, new)
    assert [k[0] for k in kept] == ['green_obj', 'cup']                     # the red blob sits on the cup; the green one is a thing of its own
    assert 'red_obj_00' not in w.scene                                      # the old ghost track is gone too
    assert 'blue_obj_00' in w.scene                                         # a blob away from any recognised object stays


def test_a_blob_is_not_dropped_for_a_stale_neighbour():
    w = blob_world()
    w.scene['cup_00']['stale'] = True
    SceneGraphNode._drop_blobs_on_known_objects(w, [])
    assert 'red_obj_00' in w.scene


def agreement_world():
    w = SimpleNamespace(_pair_gaps={}, _obs={})
    w.AGREEMENT_WINDOW_S, w.AGREEMENT_MIN_N = SceneGraphNode.AGREEMENT_WINDOW_S, SceneGraphNode.AGREEMENT_MIN_N
    return w


def test_a_pair_of_static_cameras_that_are_8_cm_apart_shows_in_the_agreement_summary():
    w = agreement_world()
    for i in range(30):
        t = 100.0 + i * 0.5
        w._obs['cup_00'] = {'realsense': (t, 0.40, 0.10, 0.04, 0.8), 'realsense2': (t, 0.48, 0.10, 0.04, 0.9), 'wrist': (t, 0.45, 0.10, 0.04, 0.9)}
        SceneGraphNode._record_agreement(w, 'cup_00', t)
    out = SceneGraphNode.agreement_summary(w, 116.0)
    assert list(out) == ['realsense|realsense2']                             # the wrist is not a static camera: it never makes a pair
    assert out['realsense|realsense2']['median_m'] == pytest.approx(0.08, abs=0.001)


def test_too_few_samples_give_no_summary_and_old_ones_expire():
    w = agreement_world()
    for i in range(5):
        w._obs['cup_00'] = {'realsense': (100.0, 0.40, 0.10, 0.04, 0.8), 'realsense2': (100.0, 0.48, 0.10, 0.04, 0.9)}
        SceneGraphNode._record_agreement(w, 'cup_00', 100.0 + i)
    assert SceneGraphNode.agreement_summary(w, 106.0) == {}


def test_a_detection_far_taller_than_its_class_is_implausible_but_a_normal_one_is_not():
    top = -0.0125
    assert sgn.implausibly_tall('cup', 0.1555, top)                      # the red coffee machine read as a cup (top 17 cm above the table)
    assert sgn.implausibly_tall('cup', 0.358, top)                       # the gripper read as a cup
    assert not sgn.implausibly_tall('cup', 0.078, top)                   # a real 9 cm cup
    assert not sgn.implausibly_tall('cup', 0.1, top)                     # a 11 cm mug with depth noise
    assert not sgn.implausibly_tall('mouse', 0.0233, top)
    assert sgn.implausibly_tall('mouse', 0.0697, top)
    assert not sgn.implausibly_tall('bottle', 0.2, top)
    assert not sgn.implausibly_tall('laptop', 0.5, top)                  # a class without a known height is never judged
    assert not sgn.implausibly_tall('cup', None, top) and not sgn.implausibly_tall('cup', 0.5, None)


def test_fixed_obstacles_parse_and_bad_entries_are_skipped():
    out = sgn.parse_fixed_obstacles({'obstacles': [{'name': 'coffee machine', 'x': 0.4, 'y': 0.68, 'radius': 0.14, 'height': 0.3}, {'name': 'broken', 'x': 'a'}, {'x': 0.1, 'y': 0.2}]})
    assert [o['name'] for o in out] == ['coffee machine', 'large object']
    assert out[0]['radius'] == 0.14 and out[1]['height'] == 0.30
    assert sgn.parse_fixed_obstacles(None) == [] and sgn.parse_fixed_obstacles({'obstacles': None}) == []


def test_a_detection_inside_a_fixed_obstacle_footprint_is_recognised():
    w = SimpleNamespace(_fixed_obstacles=lambda: [{'name': 'coffee machine', 'x': 0.40, 'y': 0.68, 'radius': 0.14, 'height': 0.3}])
    assert SceneGraphNode._in_fixed_obstacle(w, 0.39, 0.55)             # the front face read 13 cm short
    assert not SceneGraphNode._in_fixed_obstacle(w, 0.56, 0.32)         # the white cup


# ---- colour vote, 2026-10-30: a white cup behind a blue one was read blue --------------------------------------------------------------------------------------
def vote_world():
    node = SimpleNamespace(_colour_votes={})
    node._colour_from_votes = SceneGraphNode._colour_from_votes
    node.COLOUR_SOURCE_TTL_S = SceneGraphNode.COLOUR_SOURCE_TTL_S
    return node


def test_a_few_wrong_blue_readings_do_not_turn_a_white_cup_blue():
    vote = SceneGraphNode._colour_from_votes
    assert vote({'white': 1.0, 'blue': 0.5}) == 'white'
    assert vote({'white': 1.0, 'blue': 0.4, 'grey': 0.3}) == 'white'
    assert vote({'blue': 1.0, 'white': 0.4}) == 'blue'                  # a real blue cup with a few white glints stays blue
    assert vote({'blue': 0.3, 'black': 1.0}) == 'blue'                  # the dark-teal rule against black is unchanged


def test_a_camera_that_reports_more_often_does_not_outvote_the_others():
    node = vote_world()
    out = None
    for i in range(80):                                                  # RS2 reports 14 times for every 6 of RS1; RS2 is wrong (the cup is hidden behind a blue one)
        out = SceneGraphNode._vote_colour(node, 'cup_00', 'blue', 0.9, 'realsense2')
        if i % 3 == 0:
            out = SceneGraphNode._vote_colour(node, 'cup_00', 'white', 0.9, 'realsense')
    assert out == 'white'                                               # RS1 weight 1.0 against RS2 0.6, one vote each


def test_a_camera_that_goes_quiet_drops_out_of_the_vote():
    node = vote_world()
    for _ in range(30):
        SceneGraphNode._vote_colour(node, 'cup_00', 'blue', 0.9, 'realsense2')
    for _ in range(5):
        SceneGraphNode._vote_colour(node, 'cup_00', 'white', 0.9, 'realsense')
    node._colour_votes['cup_00']['realsense2']['t'] -= 60                # RS2 has said nothing for a minute (it only saw 'unknown', or lost the cup)
    out = [SceneGraphNode._vote_colour(node, 'cup_00', None, 0.9, 'realsense2') for _ in range(12)]
    assert out[-1] == 'white'


def test_the_published_colour_changes_only_after_the_new_one_keeps_winning():
    node = SimpleNamespace(_colour_votes={}, _colour_pub={})
    hold = lambda w: SceneGraphNode._hold_colour(node, 'mouse_00', w)  # noqa: E731
    assert [hold('white') for _ in range(3)] == ['white'] * 3            # warm-up: the raw winner is published ...
    assert [hold('grey') for _ in range(12)][-1] == 'grey'               # ... so a lone first frame cannot lock a wrong colour in
    assert [hold('blue') for _ in range(5)] == ['grey'] * 5              # settled: a short bluish flash changes nothing
    assert hold('grey') == 'grey'
    assert [hold('blue') for _ in range(9)] == ['grey'] * 9              # the count starts again after grey won once
    assert hold('blue') == 'blue'                                        # the 10th in a row switches
    assert hold(None) == 'blue'                                          # no evidence keeps the current colour
