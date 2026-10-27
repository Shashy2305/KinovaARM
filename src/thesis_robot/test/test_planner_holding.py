"""The planner knows what the gripper holds (2026-10-26: "put the cup back on the table" was planned as pick cup + place while the cup was already in the gripper; the controller refused the pick)."""
from thesis_robot.llm_planner_node import check_holding, drop_pick_of_held

SCENE = {'cup_00': {'label': 'cup', 'x': 0.32, 'y': -0.15}, 'mouse_00': {'label': 'mouse', 'x': 0.30, 'y': -0.25}}
HELD_CUP = {'label': 'cup', 'object_id': 'cup_00'}
PLAN = [{'action': 'pick', 'object_id': 'cup_00', 'approach_z': 0.39}, {'action': 'place', 'x': 0.45, 'y': -0.20}]


def test_a_pick_of_the_held_object_is_dropped_and_the_place_stays():
    out = drop_pick_of_held([dict(s) for s in PLAN], HELD_CUP, SCENE)
    assert [s['action'] for s in out] == ['place'] and out[0]['x'] == 0.45
    ok, why = check_holding(out, HELD_CUP, original_plan=PLAN)
    assert ok and why == ''


def test_matching_by_label_when_the_ids_differ():
    plan = [{'action': 'pick', 'object_id': 'cup_01'}, {'action': 'place', 'x': 0.4, 'y': 0.0}]
    scene = {'cup_01': {'label': 'cup'}}
    assert [s['action'] for s in drop_pick_of_held(plan, {'label': 'cup', 'object_id': 'cup_00'}, scene)] == ['place']


def test_nothing_changes_when_nothing_is_held():
    assert drop_pick_of_held([dict(s) for s in PLAN], None, SCENE) == PLAN
    assert check_holding(PLAN, None) == (True, '')


def test_picking_something_else_while_holding_is_refused_by_the_planner_with_a_clear_reason():
    plan = [{'action': 'pick', 'object_id': 'mouse_00'}, {'action': 'place', 'x': 0.4, 'y': 0.0}]
    out = drop_pick_of_held(plan, HELD_CUP, SCENE)
    assert out == plan                                                  # a different object: not dropped
    ok, why = check_holding(out, HELD_CUP, original_plan=plan)
    assert not ok and 'holding the cup' in why and 'put it down' in why


def test_a_command_that_was_only_a_pick_of_what_is_held_has_nothing_left():
    plan = [{'action': 'pick', 'object_id': 'cup_00'}]
    out = drop_pick_of_held(plan, HELD_CUP, SCENE)
    assert out == []
    ok, why = check_holding(out, HELD_CUP, original_plan=plan)
    assert not ok and 'already in the gripper' in why


def test_other_steps_are_untouched():
    plan = [{'action': 'move_to', 'x': 0.45, 'y': 0.0, 'z': 0.45}, {'action': 'place', 'here': True}]
    assert drop_pick_of_held([dict(s) for s in plan], HELD_CUP, SCENE) == plan
    assert check_holding(plan, HELD_CUP, original_plan=plan) == (True, '')


def test_go_home_is_refused_while_something_is_held_2026_10_26():
    ok, why = check_holding([{'action': 'go_home'}], HELD_CUP, original_plan=[{'action': 'go_home'}])
    assert not ok and 'put it down' in why and 'upside down' in why
    assert check_holding([{'action': 'go_home'}], None) == (True, '')                  # nothing held: going home is fine
    ok, _ = check_holding([{'action': 'place', 'here': True}, {'action': 'go_home'}], HELD_CUP)
    assert not ok                                                                       # even after a place in the same plan: the place comes first, then say go home
