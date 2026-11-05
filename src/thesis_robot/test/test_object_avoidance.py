"""2026-11-05: a side swing with a turned wrist swept the wrist through the mug and knocked it 7 cm. The swing and the descent beside the object now refuse paths whose links pass through objects."""
import math

import pytest

from thesis_robot import arm_controller_node as acn
from thesis_robot import safety_geometry as sg

from test_pick_place_flow import TABLE, node, run_cup_pick, side_world, tilt_world, world  # noqa: F401

CUP = ('cup', 0.44, 0.04, 0.045, TABLE['table_top_z'] + 0.09)


def at(link, x, y, z):
    pos = {name: (0.2, -0.5, 0.6) for name in sg.GUARDED_LINKS}                  # everything else far away and high
    pos[link] = (x, y, z)
    return pos


def test_a_link_low_beside_the_object_is_refused_and_one_high_over_it_or_far_away_is_not(node):
    clear = node._links_clear_of_objects
    ok, why = clear(at('right_inner_finger_pad', 0.44, 0.10, 0.07), [CUP])                     # 6 cm from its axis, below its top: it would knock it over
    assert not ok and 'right_inner_finger_pad' in why and 'knock' in why
    assert clear(at('right_inner_finger_pad', 0.44, 0.10, 0.25), [CUP])[0] is True              # the same place, 25 cm up: it passes over
    assert clear(at('right_inner_finger_pad', 0.44, 0.22, 0.07), [CUP])[0] is True              # 18 cm away, low: fine (the pre pose stands 17 cm from the object)
    assert clear(at('forearm_link', 0.44, 0.04, 0.20), [CUP])[0] is True                        # over the object, above its top + 5 cm
    assert clear(at('forearm_link', 0.44, 0.04, 0.10), [CUP])[0] is False                       # ... but not 10 cm up


def test_the_scene_objects_become_cylinders_with_their_class_sizes_and_stale_ones_are_ignored(node):
    node.latest_scene = {'cup_1': {'label': 'cup', 'x': 0.44, 'y': 0.04, 'stale': False}, 'old': {'label': 'cup', 'x': 0.1, 'y': 0.1, 'stale': True},
                         'fixed_00': {'label': 'coffee machine', 'x': 0.485, 'y': 0.305, 'stale': False, 'size_x': 0.18, 'size_y': 0.225}}
    objs = node._objects_to_avoid()
    assert [o[0] for o in objs] == ['cup', 'coffee machine']
    cup = objs[0]
    assert cup[3] == pytest.approx(0.045) and cup[4] == pytest.approx(TABLE['table_top_z'] + 0.09)
    assert objs[1][3] == pytest.approx(0.1125)                                                  # half the machine's longer side


class Pt:
    def __init__(self, positions):
        self.positions = positions


class Traj:
    joint_names = list(acn.JOINT_NAMES)

    def __init__(self, n=3):
        self.points = [Pt([0.0] * 7) for _ in range(n)]


def test_the_path_check_looks_at_objects_only_inside_the_avoiding_block(node, monkeypatch):
    node.latest_scene = {'cup_1': {'label': 'cup', 'x': 0.44, 'y': 0.04, 'stale': False}}
    monkeypatch.setattr(node, '_fk_link_positions', lambda names, q: at('left_inner_finger_pad', 0.44, 0.09, 0.09))      # 5 cm from the cup, low, but above the table margins
    traj = Traj()
    assert node._check_trajectory(traj, TABLE)[0] is True                                       # outside the block: only the table is checked (as before)
    with node._avoiding_objects():
        ok, why = node._check_trajectory(traj, TABLE)
        assert not ok and 'cup' in why and 'waypoint' in why
    assert node._check_trajectory(traj, TABLE)[0] is True and node._avoid_scene is False        # and the flag is back off afterwards


def test_a_swing_refused_for_sweeping_an_object_makes_the_pick_go_home_and_try_a_new_branch(node, side_world, monkeypatch):
    real = node._guarded_move
    swings = {'n': 0}

    def guarded(geom, label, **kw):
        if 'swing' in label:
            swings['n'] += 1
            return False                                                                         # refused before any motion
        return real(geom, label, **kw)
    monkeypatch.setattr(node, '_guarded_move', guarded)
    side_world.guarded.clear()
    node._pick_side_az = 90.0
    try:
        assert run_cup_pick(node) is False
    finally:
        node._pick_side_az = None
    assert swings['n'] == 3 and side_world.guarded.count('go_home') == 2                           # three tries, each from home: a fresh hover, a fresh branch
