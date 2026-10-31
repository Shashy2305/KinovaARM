import pytest

from thesis_robot import command_grammar as cg

SCENE = {
    'cup_01': {'label': 'cup', 'x': 0.40, 'y': -0.30, 'z': 0.04, 'reachable': True, 'stale': False, 'confidence': 0.95},
    'cup_02': {'label': 'cup', 'x': 0.30, 'y': 0.20, 'z': 0.04, 'reachable': True, 'stale': False, 'confidence': 0.80},
    'bowl_00': {'label': 'bowl', 'x': 0.32, 'y': 0.28, 'z': 0.02, 'reachable': True, 'stale': False, 'confidence': 0.90},
    'mouse_01': {'label': 'mouse', 'x': 0.37, 'y': -0.09, 'z': 0.02, 'reachable': True, 'stale': False, 'confidence': 0.93},
    'bottle_00': {'label': 'bottle', 'x': 0.25, 'y': -0.21, 'z': 0.07, 'reachable': True, 'stale': False, 'confidence': 0.93},
    'bowl_stale': {'label': 'bowl', 'x': 0.5, 'y': 0.0, 'z': 0.02, 'reachable': True, 'stale': True, 'confidence': 0.99},
}


@pytest.mark.parametrize('text,kind,a,b', [
    ('go home', 'go_home', None, None),
    ('Please go back home', 'go_home', None, None),
    ('return to home position', 'go_home', None, None),
    ('home', 'go_home', None, None),
    ('put it down', 'put_down', None, None),
    ('set it down.', 'put_down', None, None),
    ('release it', 'put_down', None, None),
    ('pick up the cup', 'pick', 'cup', None),
    ('grab the mug', 'pick', 'cup', None),
    ('Can you pick up the blue bottle?', 'pick', 'bottle', None),
    ('pick up the cup and put it next to the bowl', 'pick_place_near', 'cup', 'bowl'),
    ('grab the bottle and set it beside the mouse', 'pick_place_near', 'bottle', 'mouse'),
    ('pick up the mouse, then place it near the cup', 'pick_place_near', 'mouse', 'cup'),
    ('move the bowl next to the bottle', 'pick_place_near', 'bowl', 'bottle'),
    ('put the mouse beside the bowl', 'pick_place_near', 'mouse', 'bowl'),
    ('pick up the cup and put it aside', 'pick_aside', 'cup', None),
    ('move the mouse out of the way', 'pick_aside', 'mouse', None),
    ('go near the cup', 'go_near', 'cup', None),
    ('move over the bowl', 'go_near', 'bowl', None),
    ('look at the bottle', 'go_near', 'bottle', None),
])
def test_parse(text, kind, a, b):
    i = cg.parse(text)
    assert i is not None, text
    assert (i['kind'], i['a'], i['b']) == (kind, a, b)


@pytest.mark.parametrize('text', ['', 'tell me a joke', 'rotate the wrist by thirty degrees', 'pick up everything',
                                  'calibrate the camera and then go home maybe later'])
def test_unrecognised_commands_are_left_to_the_model(text):
    assert cg.parse(text) is None


def test_check_accepts_the_right_plan_and_names_the_mistake_in_a_wrong_one():
    i = cg.parse('pick up the cup and put it next to the bowl')
    good = [{'action': 'pick', 'object_id': 'cup_01'}, {'action': 'place', 'near': 'bowl_00'}]
    assert cg.check(i, good, SCENE) == (True, '')
    assert not cg.check(i, [{'action': 'move_to', 'x': 0.3, 'y': 0.2, 'z': 0.4}] * 2, SCENE)[0]
    wrong_obj = [{'action': 'pick', 'object_id': 'bowl_00'}, {'action': 'place', 'near': 'bowl_00'}]
    ok, why = cg.check(i, wrong_obj, SCENE)
    assert not ok and 'bowl' in why
    far = [{'action': 'pick', 'object_id': 'cup_01'}, {'action': 'place', 'x': 0.5, 'y': -0.3}]
    assert not cg.check(i, far, SCENE)[0]                                    # not near any bowl
    near_xy = [{'action': 'pick', 'object_id': 'cup_01'}, {'action': 'place', 'x': 0.33, 'y': 0.15}]
    assert cg.check(i, near_xy, SCENE)[0]                                    # within 0.30 of the bowl


def test_check_for_the_simple_kinds():
    assert cg.check(cg.parse('go home'), [{'action': 'go_home'}], SCENE)[0]
    assert not cg.check(cg.parse('go home'), [{'action': 'move_to', 'x': 0.3, 'y': 0, 'z': 0.4}], SCENE)[0]
    assert cg.check(cg.parse('put it down'), [{'action': 'place', 'here': True}], SCENE)[0]
    assert not cg.check(cg.parse('put it down'), [], SCENE)[0]
    assert cg.check(cg.parse('pick up the cup'), [{'action': 'pick', 'object_id': 'cup_01'}], SCENE)[0]
    assert cg.check(cg.parse('pick up the cup'), [{'action': 'pick', 'object_id': 'cup_01'}, {'action': 'go_home'}], SCENE)[0]   # trimmed later
    assert not cg.check(cg.parse('pick up the cup'), [{'action': 'pick', 'object_id': 'cup_01'}, {'action': 'place', 'here': True}], SCENE)[0]
    assert cg.check(cg.parse('go near the bowl'), [{'action': 'move_to', 'x': 0.32, 'y': 0.28, 'z': 0.3}], SCENE)[0]


def test_build_plan_round_trips_through_check_for_every_kind():
    for text in ('go home', 'put it down', 'pick up the cup', 'pick up the mouse and put it next to the bottle',
                 'pick up the cup and put it aside', 'go near the bowl'):
        i = cg.parse(text)
        plan = cg.build_plan(i, SCENE)
        assert plan is not None, text
        ok, why = cg.check(i, plan, SCENE)
        assert ok, f'{text}: {why}'


def test_build_plan_picks_the_most_confident_fresh_object_and_a_different_reference():
    plan = cg.build_plan(cg.parse('pick up the cup and put it next to the cup'), SCENE)
    assert plan[0]['object_id'] == 'cup_01' and plan[1]['near'] == 'cup_02'      # the other cup, never itself
    assert cg.build_plan(cg.parse('pick up the bowl'), SCENE)[0]['object_id'] == 'bowl_00'   # the stale one is ignored
    assert cg.build_plan(cg.parse('pick up the scissors'), SCENE) is None                     # not in the scene


@pytest.mark.parametrize('text,side', [
    ('pick up the cup and put it to the left of the bowl', 'left'),
    ('grab the mouse and place it on the right side of the bottle', 'right'),
    ('pick up the bottle and put it in front of the bowl', 'front'),
    ('pick up the cup and set it behind the mouse', 'behind'),
    ('move the mouse to the left of the cup', 'left'),
    ('put the bowl in back of the bottle', 'behind'),
])
def test_side_relations_parse(text, side):
    i = cg.parse(text)
    assert i is not None and i['kind'] == 'pick_place_side' and i['side'] == side, text


def test_side_check_and_build():
    i = cg.parse('pick up the cup and put it to the left of the bowl')
    plan = cg.build_plan(i, SCENE)
    assert plan[1] == {'action': 'place', 'near': 'bowl_00', 'side': 'left'}
    assert cg.check(i, plan, SCENE)[0]
    assert not cg.check(i, [plan[0], {'action': 'place', 'near': 'bowl_00'}], SCENE)[0]            # missing side
    assert not cg.check(i, [plan[0], {'action': 'place', 'near': 'bowl_00', 'side': 'right'}], SCENE)[0]
    # coordinates are accepted when they really lie on that side: the bowl is at (0.32, 0.28), left is -y
    assert cg.check(i, [plan[0], {'action': 'place', 'x': 0.32, 'y': 0.12}], SCENE)[0]
    assert not cg.check(i, [plan[0], {'action': 'place', 'x': 0.32, 'y': 0.44}], SCENE)[0]


def test_put_the_named_object_back_or_down_is_a_put_down_2026_10_26():
    """"put the cup back on the table" was not recognised: nothing verified the model's plan and a lone go_home was approved while the cup was still in the gripper."""
    for text in ('put the cup back on the table', 'put the cup down', 'put the mouse back', 'set the cup down on the table', 'place the cup back on the desk', 'put it back', 'put it down'):
        intent = cg.parse(text)
        assert intent is not None and intent['kind'] == 'put_down', text
    # a put with a destination is still a different intent, and unrelated commands are untouched
    assert cg.parse('put the cup next to the mouse')['kind'] == 'pick_place_near'
    assert cg.parse('pick up the cup')['kind'] == 'pick' and cg.parse('go home')['kind'] == 'go_home'
    # the verifier now rejects what happened live: a lone go_home for a put-back
    ok, why = cg.check(cg.parse('put the cup back on the table'), [{'action': 'go_home'}], {})
    assert not ok
    ok, why = cg.check(cg.parse('put the cup back on the table'), [{'action': 'place', 'here': True}], {})
    assert ok


# ---- a command about an object the arm cannot reach (2026-10-28: "pick up mouse and put it aside" -> "Plan is empty or not a list" after 3 slow retries) -----------------
def _scene(**objs):
    return {k: dict({'confidence': 0.9, 'stale': False, 'reachable': True, 'z': 0.02}, **v) for k, v in objs.items()}


def test_no_reason_when_a_reachable_object_of_that_label_exists():
    scene = _scene(mouse_00={'label': 'mouse', 'x': 0.4, 'y': 0.0})
    assert cg.no_reachable_reason(cg.parse('pick up the mouse'), scene) is None
    assert cg.no_reachable_reason(cg.parse('go home'), scene) is None                       # no object named


def test_an_object_outside_the_arms_range_is_explained_with_its_position():
    scene = _scene(mouse_00={'label': 'mouse', 'x': 0.56, 'y': -0.39, 'reachable': False}, cup_00={'label': 'cup', 'x': 0.4, 'y': 0.1})
    why = cg.no_reachable_reason(cg.parse('pick up mouse and put it aside'), scene)
    assert 'mouse_00' in why and '-0.39' in why and 'outside' in why and 'move it closer' in why


def test_a_stale_or_missing_object_is_explained_too():
    assert 'stale' in cg.no_reachable_reason(cg.parse('pick up the cup'), _scene(cup_00={'label': 'cup', 'x': 0.4, 'y': 0.1, 'stale': True, 'reachable': False}))
    assert 'no cup in the scene' in cg.no_reachable_reason(cg.parse('pick up the cup'), _scene(mouse_00={'label': 'mouse', 'x': 0.4, 'y': 0.0}))


# ---- colours and "near the robot base" (2026-10-29) ---------------------------------------------------------------------------------------------------------
def _cups():
    return {
        'cup_00': {'label': 'cup', 'x': 0.45, 'y': 0.04, 'z': 0.05, 'confidence': 0.9, 'color': 'blue', 'reachable': True, 'stale': False},
        'cup_01': {'label': 'cup', 'x': 0.30, 'y': 0.20, 'z': 0.05, 'confidence': 0.8, 'color': 'white', 'reachable': True, 'stale': False},
        'mouse_00': {'label': 'mouse', 'x': 0.37, 'y': -0.12, 'z': 0.02, 'confidence': 0.9, 'color': 'black', 'reachable': True, 'stale': False},
    }


def test_colour_words_are_parsed_into_the_intent():
    i = cg.parse('pick up the white cup')
    assert i['kind'] == 'pick' and i['a'] == 'cup' and i['a_color'] == 'white'
    assert cg.parse('grab the dark blue mug')['a_color'] == 'blue'
    assert cg.parse('pick up the teal cup')['a_color'] == 'blue' and cg.parse('pick up the silver mouse')['a_color'] == 'grey'
    assert cg.parse('pick up the cup')['a_color'] is None
    both = cg.parse('put the white cup next to the blue cup')
    assert both['kind'] == 'pick_place_near' and both['a_color'] == 'white' and both['b_color'] == 'blue'


def test_a_colour_request_picks_the_object_of_that_colour():
    scene = _cups()
    assert cg.best_object(scene, 'cup', colour='white') == 'cup_01'
    assert cg.best_object(scene, 'cup', colour='blue') == 'cup_00'
    assert cg.best_object(scene, 'cup', colour='red') is None
    assert cg.best_object(scene, 'cup') == 'cup_00'                                            # no colour asked: the most confident


def test_the_check_rejects_a_plan_that_picks_the_wrong_colour():
    scene = _cups()
    intent = cg.parse('pick up the white cup')
    ok, why = cg.check(intent, [{'action': 'pick', 'object_id': 'cup_00', 'approach_z': 0.35}], scene)
    assert not ok and 'blue' in why and 'white' in why
    assert cg.check(intent, [{'action': 'pick', 'object_id': 'cup_01', 'approach_z': 0.35}], scene)[0]
    assert cg.build_plan(intent, scene)[0]['object_id'] == 'cup_01'


def test_an_object_of_unknown_colour_never_matches_a_colour_request_and_the_reason_lists_what_is_seen():
    scene = _cups()
    scene['cup_00']['color'] = None
    scene['cup_01']['color'] = None
    why = cg.no_reachable_reason(cg.parse('pick up the white cup'), scene)
    assert 'no white cup' in why and 'cup_00' in why and 'cup_01' in why
    scene2 = _cups()
    assert cg.no_reachable_reason(cg.parse('pick up the red cup'), scene2).startswith('no red cup')
    assert cg.no_reachable_reason(cg.parse('pick up the white cup'), scene2) is None


def test_near_the_robot_base_is_its_own_intent_with_a_base_place():
    for text in ('pick up the white cup and put it near the robot base', 'move the white cup to the robot base', 'put the white cup next to the base'):
        i = cg.parse(text)
        assert i['kind'] == 'pick_place_base' and i['a_color'] == 'white', text
    plan = cg.build_plan(cg.parse('pick up the white cup and put it near the robot base'), _cups())
    assert plan[0]['object_id'] == 'cup_01' and plan[1] == {'action': 'place', 'near': 'base'}
    assert cg.check(cg.parse('pick up the white cup and put it near the robot base'), plan, _cups())[0]
    bad = [plan[0], {'action': 'place', 'x': 0.5, 'y': 0.3}]
    assert not cg.check(cg.parse('pick up the white cup and put it near the robot base'), bad, _cups())[0]


def test_next_to_a_coloured_object_checks_the_colour_of_the_reference():
    scene = _cups()
    intent = cg.parse('pick up the mouse and put it next to the white cup')
    assert cg.check(intent, [{'action': 'pick', 'object_id': 'mouse_00', 'approach_z': 0.35}, {'action': 'place', 'near': 'cup_01'}], scene)[0]
    assert not cg.check(intent, [{'action': 'pick', 'object_id': 'mouse_00', 'approach_z': 0.35}, {'action': 'place', 'near': 'cup_00'}], scene)[0]
    assert cg.build_plan(intent, scene)[1] == {'action': 'place', 'near': 'cup_01'}


# ---- corners (2026-10-31: "pick up mouse and keep it on left corner from the base" ran as a lone pick and the mouse stayed in the gripper) -----------------------
def test_a_corner_command_is_recognised_and_placed_at_that_corner():
    scene = {'mouse_00': {'label': 'mouse', 'x': 0.48, 'y': -0.22, 'z': 0.02, 'confidence': 0.9, 'reachable': True, 'stale': False, 'color': 'grey'}}
    for text, corner in (('pick up mouse and keep it on left corner from the base', ('left', 'far')),
                         ('pick up the mouse and put it in the far right corner of the table', ('right', 'far')),
                         ('pick up the mouse and leave it near left corner', ('left', 'near')),
                         ('pick up the mouse and put it in the right near corner', ('right', 'near')),
                         ('keep the mouse on the left corner', ('left', 'far'))):
        intent = cg.parse(text)
        assert intent and intent['kind'] == 'pick_place_corner' and intent['corner'] == corner, text
        plan = cg.build_plan(intent, scene)
        assert [s['action'] for s in plan] == ['pick', 'place']
        assert (plan[1]['x'], plan[1]['y']) == cg.CORNER_XY[corner]
        assert cg.check(intent, plan, scene) == (True, '')


def test_a_corner_plan_without_the_place_or_at_the_wrong_corner_is_rejected():
    scene = {'mouse_00': {'label': 'mouse', 'x': 0.48, 'y': -0.22, 'z': 0.02, 'confidence': 0.9, 'reachable': True, 'stale': False, 'color': 'grey'}}
    intent = cg.parse('pick up the mouse and keep it on the left corner')
    ok, why = cg.check(intent, [{'action': 'pick', 'object_id': 'mouse_00'}], scene)
    assert not ok and 'place' in why                                    # the lone pick the model produced live
    ok, why = cg.check(intent, [{'action': 'pick', 'object_id': 'mouse_00'}, {'action': 'place', 'x': 0.55, 'y': 0.38}], scene)
    assert not ok and 'corner' in why                                   # the right corner is not the left one


def test_the_corner_places_are_inside_the_planner_workspace():
    from thesis_robot import safety_geometry as sg
    for x, y in cg.CORNER_XY.values():
        assert sg.PLAN_X_RANGE[0] <= x <= sg.PLAN_X_RANGE[1] and sg.PLAN_Y_RANGE[0] <= y <= sg.PLAN_Y_RANGE[1]
