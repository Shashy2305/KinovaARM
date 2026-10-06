import pytest

pytest.importorskip('rclpy')
pytest.importorskip('ollama')

from thesis_robot import safety_geometry as sg  # noqa: E402
from thesis_robot.llm_planner_node import fix_move_to_heights, fix_pick_heights, fix_place_targets, normalize_place_here, find_free_spot, trim_pick_extras, validate_plan  # noqa: E402

GEOM = {'table_top_z': -0.06, 'x': [-0.15, 0.75], 'y': [-0.80, 0.85], 'rear_wall_x': -0.20}
SCENE = {'cup_00': {'label': 'cup', 'x': 0.159, 'y': 0.1327, 'z': -0.0897,
                    'reachable': True, 'stale': False}}


def _plan(z):
    return [{'action': 'move_to', 'x': 0.1586, 'y': 0.1327, 'z': z, 'speed': 0.2}]


@pytest.mark.parametrize('llm_z', [-0.091, 0.06, 0.12, 0.5])
def test_move_to_height_keeps_fingertips_above_table(llm_z):
    floor = sg.flange_floor_z(GEOM)
    plan = fix_move_to_heights(_plan(llm_z), SCENE, floor)
    z = plan[0]['z']
    assert z >= floor
    assert z - sg.TCP_REACH_M >= GEOM['table_top_z'] + sg.LINK_TABLE_CLEARANCE_M
    assert validate_plan(plan, SCENE, floor)[0]


def test_old_incident_target_would_now_be_rejected():
    # flange z=0.12 (the previous "safe" minimum) puts the pads below the table
    floor = sg.flange_floor_z(GEOM)
    ok, why, _ = validate_plan(_plan(0.12), SCENE, floor)
    assert not ok and 'floor' in why


def test_fallback_floor_without_geometry():
    floor = sg.flange_floor_z(None)
    plan = fix_move_to_heights(_plan(-0.091), SCENE, floor)
    assert plan[0]['z'] == floor


def test_pick_approach_height_is_derived_from_the_scene():
    floor = sg.flange_floor_z({'table_top_z': -0.0125})       # the real table: floor 0.2525
    scene = {'cup_00': {'label': 'cup', 'x': 0.4, 'y': 0.0, 'z': 0.05, 'reachable': True, 'stale': False}}
    plan = [{'action': 'pick', 'object_id': 'cup_00', 'approach_z': 0.25}]     # the LLM's value that used to be rejected
    assert not validate_plan([dict(plan[0])], scene, floor)[0]
    fixed = fix_pick_heights(plan, scene, floor)
    assert abs(fixed[0]['approach_z'] - (0.05 + sg.TCP_REACH_M + 0.12)) < 1e-6
    assert validate_plan(fixed, scene, floor)[0]


def test_pick_height_is_capped_for_tall_objects():
    floor = sg.flange_floor_z(GEOM)
    scene = {'b': {'label': 'bottle', 'x': 0.4, 'y': 0.0, 'z': 0.30, 'reachable': True, 'stale': False}}
    assert fix_pick_heights([{'action': 'pick', 'object_id': 'b', 'approach_z': 0.2}], scene, floor)[0]['approach_z'] == 0.5


def _llm_pick_plan():
    return [{'action': 'move_to', 'x': 0.4, 'y': 0.0, 'z': 0.37}, {'action': 'pick', 'object_id': 'cup_00', 'approach_z': 0.39},
            {'action': 'go_home'}, {'action': 'open_gripper'}]


def test_pick_up_does_not_drop_the_object():
    acts = [s['action'] for s in trim_pick_extras(_llm_pick_plan(), 'pick up the cup')]
    assert acts == ['pick']


def test_a_requested_release_or_return_is_kept():
    assert [s['action'] for s in trim_pick_extras(_llm_pick_plan(), 'pick up the cup and put it down')][-1] == 'open_gripper'
    acts = [s['action'] for s in trim_pick_extras(_llm_pick_plan(), 'grab the cup and go back home')]
    assert 'go_home' in acts and 'open_gripper' not in acts


def test_open_gripper_before_a_pick_is_untouched():
    plan = [{'action': 'open_gripper'}, {'action': 'pick', 'object_id': 'c', 'approach_z': 0.4}]
    assert trim_pick_extras(plan, 'pick up the cup') == plan


def test_stray_move_after_a_pick_is_removed():
    plan = [{'action': 'pick', 'object_id': 'c', 'approach_z': 0.4}, {'action': 'move_to', 'x': 0.26, 'y': -0.24, 'z': 0.33}]
    assert [x['action'] for x in trim_pick_extras(plan, 'pick up the cup')] == ['pick']
    assert [x['action'] for x in trim_pick_extras(plan, 'pick up the cup and put it next to the mouse')] == ['pick', 'move_to']


def test_move_to_before_a_pick_is_removed_because_pick_approaches_by_itself():
    plan = [{'action': 'move_to', 'x': 0.4035, 'y': 0.0279, 'z': 0.2779}, {'action': 'open_gripper'},
            {'action': 'pick', 'object_id': 'c', 'approach_z': 0.4}]
    assert [x['action'] for x in trim_pick_extras(plan, 'pick up the cup')] == ['open_gripper', 'pick']
    only_move = [{'action': 'move_to', 'x': 0.4, 'y': 0.0, 'z': 0.37}]
    assert trim_pick_extras(only_move, 'go near the cup') == only_move       # no pick: untouched


def _table():
    mk = lambda lbl, x, y: {'label': lbl, 'x': x, 'y': y, 'z': 0.04, 'reachable': True, 'stale': False}
    return {'cup_00': mk('cup', 0.385, -0.042), 'mouse_00': mk('mouse', 0.262, -0.237),
            'bowl_00': mk('bowl', 0.30, 0.20),
            'mouse_09': {'label': 'mouse', 'x': -0.3, 'y': 0.5, 'z': 0.0, 'reachable': False, 'stale': False}}   # side desk


def _dist(a, b):
    import math
    return math.hypot(a[0] - b[0], a[1] - b[1])


def test_place_next_to_an_object_finds_a_free_spot_near_it():
    scene = _table()
    plan = fix_place_targets([{'action': 'pick', 'object_id': 'cup_00'}, {'action': 'place', 'near': 'mouse_00'}], scene)
    st = plan[1]
    spot = (st['x'], st['y'])
    assert 'near' not in st and 'unplaceable' not in st
    assert 0.12 <= _dist(spot, (0.262, -0.237)) <= 0.21                      # next to the mouse
    assert _dist(spot, (0.30, 0.20)) >= 0.11                                  # clear of the bowl
    assert _dist(spot, (0.0, 0.0)) >= 0.25                                    # not under the robot base
    assert 0.13 <= spot[0] <= 0.57 and -0.32 <= spot[1] <= 0.32               # inside the workspace
    assert validate_plan(plan, scene, 0.2525)[0]


def test_place_near_accepts_a_label_and_ignores_unreachable_objects():
    scene = _table()
    plan = fix_place_targets([{'action': 'pick', 'object_id': 'cup_00'}, {'action': 'place', 'near': 'mouse'}], scene)
    assert plan[1]['x'] > 0 or plan[1]['x'] <= 0     # resolved (the side-desk mouse is not the target)
    assert _dist((plan[1]['x'], plan[1]['y']), (0.262, -0.237)) < 0.25


def test_place_at_a_taken_spot_is_moved_to_a_free_one():
    scene = _table()
    plan = fix_place_targets([{'action': 'pick', 'object_id': 'cup_00'}, {'action': 'place', 'x': 0.30, 'y': 0.20}], scene)
    assert _dist((plan[1]['x'], plan[1]['y']), (0.30, 0.20)) >= 0.11


def test_place_on_the_picked_objects_own_spot_is_fine():
    scene = _table()
    plan = fix_place_targets([{'action': 'pick', 'object_id': 'cup_00'}, {'action': 'place', 'x': 0.385, 'y': -0.042}], scene)
    assert (plan[1]['x'], plan[1]['y']) == (0.385, -0.042)


def test_place_with_unknown_or_missing_target_is_rejected():
    scene = _table()
    bad = fix_place_targets([{'action': 'pick', 'object_id': 'cup_00'}, {'action': 'place', 'near': 'giraffe'}], scene)
    ok, why, _ = validate_plan(bad, scene, 0.2525)
    assert not ok and 'cannot place' in why
    assert not validate_plan(fix_place_targets([{'action': 'place'}], scene), scene, 0.2525)[0]
    assert validate_plan([{'action': 'place', 'here': True}], scene, 0.2525)[0]


def test_no_free_spot_means_no_plan():
    crowded = {f'o{i}': {'label': 'bowl', 'x': 0.30 + 0.1 * (i % 3), 'y': -0.2 + 0.1 * (i // 3), 'z': 0.04,
                         'reachable': True, 'stale': False} for i in range(9)}
    assert find_free_spot((0.4, 0.0), crowded, (0.4, 0.0), min_clear=0.5) is None


def test_the_full_command_keeps_place_and_drops_the_padding():
    plan = [{'action': 'move_to', 'x': 0.4, 'y': 0.0, 'z': 0.37}, {'action': 'pick', 'object_id': 'cup_00', 'approach_z': 0.39},
            {'action': 'place', 'near': 'mouse_00'}, {'action': 'open_gripper'}, {'action': 'go_home'}]
    acts = [x['action'] for x in trim_pick_extras(plan, 'pick up the cup and put it next to the mouse')]
    assert acts == ['pick', 'place']


def test_move_to_between_pick_and_place_is_removed():
    plan = [{'action': 'pick', 'object_id': 'cup_00', 'approach_z': 0.43},
            {'action': 'move_to', 'x': 0.28, 'y': -0.235, 'z': 0.31},      # over the mouse: seen from the LLM
            {'action': 'place', 'x': 0.33, 'y': -0.13}]
    assert [x['action'] for x in trim_pick_extras(plan, 'pick up the cup and put it next to the mouse')] == ['pick', 'place']


def test_put_it_down_means_here_but_a_destination_is_respected():
    plan = lambda: [{'action': 'pick', 'object_id': 'c', 'approach_z': 0.4}, {'action': 'place', 'x': 0.36, 'y': 0.04}]
    assert normalize_place_here(plan(), 'pick up the cup and put it down')[1] == {'action': 'place', 'here': True}
    assert normalize_place_here(plan(), 'pick up the cup and set it down')[1].get('here') is True
    assert 'here' not in normalize_place_here(plan(), 'pick up the cup and put it down next to the mouse')[1]
    assert 'here' not in normalize_place_here(plan(), 'pick up the cup and put it down on the left')[1]
    assert 'here' not in normalize_place_here(plan(), 'pick up the cup and put it down at x 0.3 y 0.1')[1]
