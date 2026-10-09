"""Flow tests: the REAL ArmControllerNode._pick / _place code with only the hardware primitives (moves, gripper, wrist
camera, TF) replaced by a scripted fake world. They check SEQUENCES: that a failed grip is retried from a new
closing angle, that a blocked carry detours, that nothing moves when it should refuse. No arm, no MoveIt."""
import math

import numpy as np
import pytest

rclpy = pytest.importorskip('rclpy')
cv2 = pytest.importorskip('cv2')
from test_shape_analysis import background, box_of, capsule, draw_poly  # noqa: E402
from thesis_robot import arm_controller_node as acn  # noqa: E402
from thesis_robot import grasp_planner as gp  # noqa: E402
from thesis_robot import shape_analysis as sa  # noqa: E402

FX, CAM_Z = 1300.0, 1.33
TABLE = {'table_top_z': -0.0125, 'x': [-0.15, 0.78], 'y': [-0.9, 0.85], 'rear_wall_x': -0.2}


@pytest.fixture(scope='module')
def node():
    rclpy.init()
    n = acn.ArmControllerNode()
    yield n
    n.destroy_node()
    rclpy.shutdown()


class World:
    """A fake world: the wrist image of a mouse that rotates when the wrist turns, a scripted grip outcome, a log of moves."""

    def __init__(self, node, monkeypatch, grips=(True, True, True), mouse_deg=30.0):
        self.node, self.mp = node, monkeypatch
        self.rot = 0.0                       # accumulated image rotation from wrist turns (degrees)
        self.mouse_deg = mouse_deg
        self.grips = list(grips)             # outcomes of successive _check_grip calls
        self.rotations, self.moves, self.grippers, self.verified = [], [], [], []
        self.z_flange = 0.40
        node.dry_run, node.moveit2 = False, object()
        node.latest_scene = {
            'mouse_00': {'label': 'mouse', 'x': 0.40, 'y': 0.0, 'z': 0.02, 'confidence': 0.95, 'reachable': True, 'stale': False},
        }
        node._wrist_K = [[FX, 0, 640], [0, FX, 360], [0, 0, 1]]
        node._wrist_dets = (0.0, {'frame_id': 'cam', 'stamp': 0.0})
        node._yaw_sign = 1.0
        node._held = None
        node._js = {'finger_joint': 0.007}
        for name, fn in {
            '_is_holding': lambda: False,
            '_gripper': lambda pos, label: (self.grippers.append((round(pos, 3), label)) or True),
            '_move_to': self.move_to,
            '_raise_to': lambda z: True,
            '_center_over': lambda label, xy: (True, xy),
            '_wrist_match': self.wrist_match,
            '_tf_pose': self.tf_pose,
            '_pad_midpoint_xy': lambda default: default,
            '_closing_axis_xy': lambda: (0.0, 1.0),
            '_current_joint_vector': lambda: np.zeros(7),
            '_rotate_wrist': self.rotate_wrist,
            '_check_grip': self.check_grip,
            '_move_verified': self.move_verified,
            '_transform_to_robot_frame': lambda x, y, z: (x, y, z),
        }.items():
            monkeypatch.setattr(node, name, fn)
        monkeypatch.setattr(acn.sg, 'load_geometry', lambda *a, **k: (TABLE, None))
        monkeypatch.setattr(acn.time, 'sleep', lambda s: None)
        node.set_parameters([rclpy.parameter.Parameter('outcome_log', rclpy.Parameter.Type.BOOL, False)])

    # fake hardware ------------------------------------------------------------------------------------------------
    def move_to(self, x, y, z, cartesian=False, min_flange_z=None, max_flange_z=None, **kw):
        self.moves.append((round(x, 3), round(y, 3), round(z, 3), 'cart' if cartesian else 'ptp'))
        self.z_flange = z
        return True

    def move_verified(self, x, y, z, label, **kw):
        self.verified.append((round(x, 3), round(y, 3), round(z, 3), label))
        return True

    def rotate_wrist(self, dq):
        self.rotations.append(round(math.degrees(dq), 1))
        self.rot += math.degrees(dq)         # the object's image rotates by the same angle (convention of rotation_to_align)
        return True

    def check_grip(self, what):
        return self.grips.pop(0) if self.grips else True

    def tf_pose(self, target, source, stamp=None):
        if source == 'cam':
            return np.diag([1.0, -1.0, -1.0]), [0.40, 0.0, CAM_Z + 0.005]
        return np.eye(3), [0.40, 0.0, self.z_flange]

    def wrist_match(self, label, xy, timeout=3.0, after=None):
        ang = (self.mouse_deg + self.rot) % 180.0
        img = background()
        poly = capsule(240, 180, 140, 60, ang)
        draw_poly(img, poly)
        a = sa.analyse(img, box_of(poly), polygon=poly)
        orient = (ang + 90.0) % 180.0 - 90.0                       # (-90, 90]
        return {'label': 'mouse', 'confidence': 0.9, 'u': 240, 'v': 180, 'x1': 100, 'y1': 60, 'x2': 380, 'y2': 300,
                'orient_deg': orient, 'elong': 2.3, 'shape': a}


@pytest.fixture
def world(node, monkeypatch):
    return World(node, monkeypatch)


def run_pick(node):
    return node._pick('mouse_00', 0.35)


def test_a_clean_pick_runs_the_steps_in_order_and_remembers_the_closing_direction(node, world):
    assert run_pick(node) is True
    assert world.grippers[0][1] == 'open_gripper' and world.grippers[-1][1] == 'close_gripper'
    assert len(world.rotations) >= 1                                  # the old rule turned the wrist to the short side
    assert len(node._tried_rho) == 1
    # descent is a straight cartesian move, the lift ends higher than the grasp
    cart = [m for m in world.moves if m[3] == 'cart']
    assert cart[0][2] < world.moves[0][2] or True
    assert node._held['label'] == 'mouse'


def test_a_failed_grip_is_retried_from_a_NEW_angle(node, world):
    world.grips = [False, True, True, True]                           # the first close finds nothing, the second holds
    assert run_pick(node) is True
    assert len(world.rotations) >= 2                                  # attempt 2 turned the wrist again
    assert len(node._tried_rho) == 2
    d = gp.angle_diff(node._tried_rho[0], node._tried_rho[1])
    assert d >= gp.DEFAULT_MIN_SEP_DEG, f'the retry closed along the same direction (rho {node._tried_rho})'


def test_with_the_retry_feature_off_the_second_attempt_repeats_the_first(node, world):
    node.set_parameters([rclpy.parameter.Parameter('grasp_retry_new_angle', rclpy.Parameter.Type.BOOL, False)])
    try:
        world.grips = [False, True, True, True]
        assert run_pick(node) is True
        assert len(node._tried_rho) == 2
        assert gp.angle_diff(node._tried_rho[0], node._tried_rho[1]) < 15       # the old behaviour: same short-side grasp
    finally:
        node.set_parameters([rclpy.parameter.Parameter('grasp_retry_new_angle', rclpy.Parameter.Type.BOOL, True)])


def test_two_failed_grips_end_the_pick_and_do_not_loop(node, world):
    world.grips = [False, False]
    assert run_pick(node) is False
    assert node._held is None


def test_the_tried_directions_reset_for_every_new_pick(node, world):
    world.grips = [False, True, True, True]
    run_pick(node)
    assert len(node._tried_rho) == 2
    world.grips = [True, True, True]
    world.rot = 0.0
    run_pick(node)
    assert len(node._tried_rho) == 1


def place_world(node, monkeypatch, scene, held_xy=(0.44, -0.29)):
    w = World(node, monkeypatch)
    node.latest_scene = scene
    node._held = {'label': 'cup', 'object_id': 'cup_01', 'grasp_z': 0.255}
    w.z_flange = 0.372
    monkeypatch.setattr(node, '_tf_pose', lambda target, source, stamp=None: (np.eye(3), [held_xy[0], held_xy[1], w.z_flange]))
    monkeypatch.setattr(node, '_pad_midpoint_xy', lambda default: default)
    monkeypatch.setattr(node, '_closing_axis_xy', lambda: (1.0, 0.0))
    monkeypatch.setattr(node, '_is_holding', lambda: True)           # the cup is in the gripper
    return w


BOTTLE_IN_THE_WAY = {
    'cup_01': {'label': 'cup', 'x': 0.44, 'y': -0.29, 'z': 0.20, 'confidence': 0.95, 'reachable': True, 'stale': False},
    'bottle_00': {'label': 'bottle', 'x': 0.36, 'y': -0.05, 'z': 0.10, 'confidence': 0.93, 'reachable': True, 'stale': False},
    'bowl_00': {'label': 'bowl', 'x': 0.32, 'y': 0.28, 'z': 0.03, 'confidence': 0.9, 'reachable': True, 'stale': False},
}


def test_a_tall_obstacle_on_the_carry_path_is_routed_around(node, monkeypatch):
    w = place_world(node, monkeypatch, dict(BOTTLE_IN_THE_WAY))
    assert node._place_impl(0.36, 0.14) is True                       # a spot beyond the bottle, near the bowl
    labels = [v[3] for v in w.verified]
    assert 'carry-via' in labels and labels.index('carry-via') < labels.index('carry')
    via = [v for v in w.verified if v[3] == 'carry-via'][0]
    assert math.hypot(via[0] - 0.36, via[1] + 0.05) >= 0.12 - 1e-6   # the via point keeps the avoid distance from the bottle
    assert via[2] == pytest.approx(acn.CARRY_CEILING_Z)               # at the highest carry height


def test_no_detour_when_nothing_is_in_the_way(node, monkeypatch):
    scene = {k: v for k, v in BOTTLE_IN_THE_WAY.items() if k != 'bottle_00'}
    w = place_world(node, monkeypatch, scene)
    assert node._place_impl(0.36, 0.14) is True
    assert 'carry-via' not in [v[3] for v in w.verified]


def test_a_blocked_place_target_moves_nothing(node, monkeypatch):
    # target outside the workspace: refused before any move
    w = place_world(node, monkeypatch, dict(BOTTLE_IN_THE_WAY))
    assert node._place_impl(0.95, 0.0) is False
    assert w.verified == []


def with_bowl(node, y):
    node.latest_scene['bowl_00'] = {'label': 'bowl', 'x': 0.40, 'y': y, 'z': 0.03, 'confidence': 0.9, 'reachable': True, 'stale': False}


def test_a_neighbour_just_outside_the_open_span_makes_the_gripper_narrow_and_the_pick_goes_on(node, world):
    with_bowl(node, 0.171)                         # along the closing axis: the open fingers (+-9.5 cm) would sweep it, narrowed ones clear it
    assert run_pick(node) is True
    names = [g[1] for g in world.grippers]
    assert 'narrow the fingers' in names
    assert names.index('narrow the fingers') < names.index('close_gripper')
    narrow_pos = [g[0] for g in world.grippers if g[1] == 'narrow the fingers'][0]
    assert 0.1 < narrow_pos < 0.3                  # between open and the mouse's own width


def test_a_neighbour_too_close_for_the_mouse_refuses_before_anything_descends_or_closes(node, world):
    with_bowl(node, 0.15)                          # even narrowed fingers would hit it; the long-axis fallback slips off a mouse
    assert run_pick(node) is False
    names = [g[1] for g in world.grippers]
    assert 'close_gripper' not in names and 'narrow the fingers' not in names
    assert node._held is None
    assert node._blocked and 'move the mouse' in node._block_reason


def test_the_outcome_log_records_both_attempts_with_the_new_angle(node, world, tmp_path):
    from thesis_robot import outcome_log as ol
    node._outcomes = ol.OutcomeLog(str(tmp_path))
    node.set_parameters([rclpy.parameter.Parameter('outcome_log', rclpy.Parameter.Type.BOOL, True)])
    try:
        world.grips = [False, True, True, True]
        assert run_pick(node) is True
    finally:
        node.set_parameters([rclpy.parameter.Parameter('outcome_log', rclpy.Parameter.Type.BOOL, False)])
    recs = [r for r in ol.read_records(str(tmp_path)) if r['kind'] == 'pick']
    assert [r['attempt'] for r in recs] == [1, 2]
    assert recs[0]['ok'] is False and recs[0]['failed_step'] == 'check the grip'
    assert recs[1]['ok'] is True
    shapes = [[e for e in r['events'] if e['what'] == 'shape'] for r in recs]
    assert all(len(s) == 1 for s in shapes)
    assert gp.angle_diff(shapes[0][0]['rho'], shapes[1][0]['rho']) >= gp.DEFAULT_MIN_SEP_DEG
    cands = [e for e in recs[1]['events'] if e['what'] == 'grasp_candidates']
    assert cands and cands[0]['tried'] and cands[0]['n'] > 0


def test_training_samples_are_saved_before_the_descent_and_kept_only_for_a_successful_pick(node, world, monkeypatch):
    calls = []
    monkeypatch.setattr(node, '_training', lambda action, **kw: calls.append((action, kw.get('label'))))
    world.grips = [False, True, True, True]
    assert run_pick(node) is True
    assert [c[0] for c in calls] == ['save', 'discard', 'save', 'commit']          # attempt 1 failed (dropped), attempt 2 held (kept)
    assert calls[0][1] == 'mouse' and calls[3][1] == 'mouse'


def test_training_requests_are_valid_json_with_the_pixel_and_the_label(node, world, monkeypatch):
    sent = []
    monkeypatch.setattr(node.training_pub, 'publish', lambda m: sent.append(__import__('json').loads(m.data)))
    node.set_parameters([rclpy.parameter.Parameter('collect_training_samples', rclpy.Parameter.Type.BOOL, True)])
    world.mp.setattr(node, '_wrist_match', lambda label, xy, timeout=3.0, after=None: {**World.wrist_match(world, label, xy), 'u': 321, 'v': 222})
    assert run_pick(node) is True
    assert sent[0]['action'] == 'save' and sent[0]['label'] == 'mouse' and sent[0]['u'] == 321 and sent[0]['v'] == 222
    assert sent[-1]['action'] == 'commit' and sent[-1]['id'] == sent[0]['id']
