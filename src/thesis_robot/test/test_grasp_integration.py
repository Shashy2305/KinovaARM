"""Decision logic of ArmControllerNode._align_with_candidates / _record_grasp_axis with a stub node:
real shape analysis of a synthetic mouse in, wrist rotation requests out. No arm, no ROS graph."""
import math
import types

import numpy as np
import pytest

rclpy = pytest.importorskip('rclpy')
cv2 = pytest.importorskip('cv2')
from test_shape_analysis import background, box_of, capsule, draw_poly  # noqa: E402
from thesis_robot import arm_controller_node as acn  # noqa: E402
from thesis_robot import grasp_planner as gp  # noqa: E402
from thesis_robot import shape_analysis as sa  # noqa: E402
from thesis_robot import wrist_servo as ws  # noqa: E402

FX = 1300.0
CAM_Z = 1.33                      # a synthetic 60 px mouse is 60 mm wide at ~1.3 m with fx 1300 (the geometry, not realism, matters)


class Param:
    def __init__(self, v):
        self.value = v


class FakeLogger:
    def info(self, *a, **k): pass
    warn = error = info


def make_node(shape_det, obstacles=(), tried=(), params=None):
    p = {'grasp_candidates': True, 'grasp_retry_new_angle': True, 'use_shape_width': False,
         'sweep_half_span_m': ws.FINGER_HALF_SPAN_M, 'sweep_margin_m': ws.SWEEP_MARGIN_M, 'sweep_same_object_m': 0.06}
    p.update(params or {})
    n = types.SimpleNamespace()
    n.get_parameter = lambda k: Param(p[k])
    n.get_logger = lambda: FakeLogger()
    n._wrist_K = [[FX, 0, 640], [0, FX, 360], [0, 0, 1]]
    n._wrist_dets = (0.0, {'frame_id': 'cam', 'stamp': 0.0})
    n._tf_pose = lambda target, source, stamp=None: (np.eye(3) * np.array([1, -1, -1]), [0.4, 0.0, CAM_Z])   # looking down
    n._wrist_match = lambda label, xy, timeout=2.0, after=None: shape_det
    n._scene_obstacles = lambda exclude_id=None, min_height=0.0, min_conf=0.65: list(obstacles)
    n._pick_ctx = {'object_id': 'mouse_00', 'tip_over_table': 0.0}
    n._tried_rho = list(tried)
    n._pick_attempt = 2 if tried else 1
    n._yaw_sign = 1.0
    n.events, n.rotations = [], []
    n._log_event = lambda name, **kw: n.events.append((name, kw))
    n._log_note = lambda **kw: None
    n._current_joint_vector = lambda: np.zeros(7)
    n._rotate_wrist = lambda dq: (n.rotations.append(dq) or True)
    n._refuse = lambda msg: False
    n.dry_run, n.moveit2 = False, object()
    n._shape_now = lambda label, xy, timeout=2.0: acn.ArmControllerNode._shape_now(n, label, xy, timeout)
    return n


@pytest.fixture(autouse=True)
def table(monkeypatch):
    monkeypatch.setattr(acn.sg, 'load_geometry', lambda *a, **k: ({'table_top_z': -0.0125}, None))


def mouse_det(long_axis_deg=30.0):
    img = background()
    poly = capsule(240, 180, 140, 60, long_axis_deg)
    draw_poly(img, poly)
    return {'label': 'mouse', 'shape': sa.analyse(img, box_of(poly), polygon=poly)}


def test_first_attempt_closes_across_the_short_side():
    det = mouse_det(30.0)
    n = make_node(det)
    r = acn.ArmControllerNode._align_with_candidates(n, 'mouse', (0.4, 0.0), det)
    assert r is True and n.rotations
    # the chosen direction is (near) the short axis: a turn that puts it between the fingers
    short = det['shape']['short_axis_deg']
    expected = gp.delta_for_phi(short)
    assert abs(math.degrees(n.rotations[0]) - expected) <= 16
    assert n.events[0][0] == 'grasp_candidates' and n.events[0][1]['n'] > 0


def test_retry_uses_a_different_direction_than_the_one_tried():
    det = mouse_det(30.0)
    first = make_node(det)
    acn.ArmControllerNode._align_with_candidates(first, 'mouse', (0.4, 0.0), det)
    chosen_phi = [e for e in first.events if e[0] == 'grasp_candidates'][0][1]['top'][0]['phi']
    rho_first = gp.rho_of(chosen_phi, det['shape']['long_axis_deg'])
    second = make_node(det, tried=[rho_first])
    acn.ArmControllerNode._align_with_candidates(second, 'mouse', (0.4, 0.0), det)
    top = [e for e in second.events if e[0] == 'grasp_candidates'][0][1]['top'][0]
    assert gp.angle_diff(gp.rho_of(top['phi'], det['shape']['long_axis_deg']), rho_first) >= gp.DEFAULT_MIN_SEP_DEG


def test_a_neighbour_on_the_short_axis_makes_it_pick_another_direction():
    det = mouse_det(0.0)                       # long axis along image x; short axis (closing) along image y
    # camera looks down with image x -> base x, image y -> base -y (the fake TF): a bowl beside the mouse along base y
    bowl = [('bowl', 0.4, 0.15)]
    free = make_node(det)
    blocked = make_node(det, obstacles=bowl)
    acn.ArmControllerNode._align_with_candidates(free, 'mouse', (0.4, 0.0), det)
    acn.ArmControllerNode._align_with_candidates(blocked, 'mouse', (0.4, 0.0), det)
    top_free = [e for e in free.events if e[0] == 'grasp_candidates'][0][1]['top'][0]['phi']
    top_blocked = [e for e in blocked.events if e[0] == 'grasp_candidates'][0][1]['top'][0]['phi']
    assert gp.angle_diff(top_free, 90.0) <= 15
    assert gp.angle_diff(top_blocked, 90.0) > 15


def test_no_shape_data_falls_back_to_the_old_rule():
    n = make_node({'label': 'mouse'})
    assert acn.ArmControllerNode._align_with_candidates(n, 'mouse', (0.4, 0.0), {'label': 'mouse'}) is None


def test_record_grasp_axis_remembers_the_direction_relative_to_the_object():
    det = mouse_det(30.0)
    n = make_node(det)
    assert acn.ArmControllerNode._record_grasp_axis(n, 'mouse', (0.4, 0.0)) is True
    assert len(n._tried_rho) == 1
    assert n._tried_rho[0] == pytest.approx(gp.rho_of(0.0, det['shape']['long_axis_deg']))
    ev = [e for e in n.events if e[0] == 'shape'][0][1]
    assert 40 < ev['width_mm'] < 140 and ev['cls'] in ('elongated', 'box')


def test_use_candidates_only_on_a_retry_unless_forced():
    n = make_node({}, params={'grasp_candidates': False}, tried=[10.0])
    assert acn.ArmControllerNode._use_candidates(n) is True               # a retry with a tried direction
    n2 = make_node({}, params={'grasp_candidates': False})
    assert acn.ArmControllerNode._use_candidates(n2) is False             # first attempt, flag off
    n3 = make_node({}, params={'grasp_candidates': True})
    assert acn.ArmControllerNode._use_candidates(n3) is True
    n4 = make_node({}, params={'grasp_candidates': False, 'grasp_retry_new_angle': False}, tried=[10.0])
    assert acn.ArmControllerNode._use_candidates(n4) is False


def test_unknown_obstacles_join_the_checks_only_when_enabled_and_confirmed():
    import time as _t
    n = types.SimpleNamespace()
    params = {'use_unknown_obstacles': False, 'unknown_obstacles_min_cameras': 2}
    n.get_parameter = lambda k: Param(params[k])
    n.latest_scene = {}
    n._unknown = (_t.monotonic(), [
        {'x': 0.4, 'y': 0.1, 'height': 0.12, 'confirmed': True},
        {'x': 0.2, 'y': -0.2, 'height': 0.05, 'confirmed': False}])
    n._unknown_list = lambda confirmed_only=False: acn.ArmControllerNode._unknown_list(n, confirmed_only)
    f = acn.ArmControllerNode._scene_obstacles
    assert f(n) == []                                                     # flag off: nothing joins
    params['use_unknown_obstacles'] = True
    assert f(n) == [('unknown object', 0.4, 0.1)]                         # on: only the camera-confirmed one
    assert f(n, min_height=0.2) == []                                     # lower than the carried object's underside
    params['unknown_obstacles_min_cameras'] = 1
    assert len(f(n)) == 2                                                 # one camera is enough when asked
    n._unknown = (_t.monotonic() - 10.0, n._unknown[1])
    assert f(n) == []                                                     # a silent guard (10 s old) adds nothing
    near = acn.ArmControllerNode._unknown_near(types.SimpleNamespace(_unknown_list=lambda confirmed_only=False: [
        {'x': 0.4, 'y': 0.1, 'height': 0.12, 'confirmed': True}]), (0.35, 0.1), 0.3)
    assert near[0]['confirmed'] is True and near[0]['d'] == pytest.approx(0.05, abs=1e-3)
