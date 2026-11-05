"""2026-11-03: the coffee routine takes a cup from anywhere (staging), checks where it put the cup, and learns its own placement error."""
import json

import pytest
import yaml

from thesis_robot import coffee_learn as cl
from thesis_robot import command_grammar as cg
from thesis_robot import locations as locs

from test_locations import COFFEE_SCENE, coffee_files, files  # noqa: F401  (fixtures)
from test_pick_place_flow import node  # noqa: F401


def cup(x, y, **kw):
    return dict({'label': 'cup', 'x': x, 'y': y, 'z': 0.06, 'confidence': 0.9, 'reachable': True, 'stale': False, 'color': 'blue'}, **kw)


# ---- the plan -------------------------------------------------------------------------------------------------------------------------------------------------
def test_a_cup_outside_the_window_gets_a_staging_step_and_is_picked_again_by_position(coffee_files):
    scene = {'cup_07': cup(0.30, -0.30)}
    intent = cg.parse('give me coffee')
    plan = cg.build_plan(intent, scene)
    assert [(s['action'], s.get('stage')) for s in plan] == [('go_home', None), ('stage_object', None), ('pick', None), ('go_to_location', 'put'), ('wait', None), ('go_to_location', 'in'),
                                                           ('go_to_location', 'take'), ('place', None), ('go_home', None), ('check_placed', None)]
    stage, pick = plan[1], plan[2]
    assert stage['object_id'] == 'cup_07' and (stage['x'], stage['y']) == (0.45, 0.0) and stage['tries'] == 3 and stage['tol_y'] == 0.02
    assert pick['near'] == [0.45, 0.0] and pick['label'] == 'cup' and pick['side_az'] == 90.0 and pick['object_id'] == 'cup_07'
    assert cg.check(intent, plan, scene) == (True, '')
    assert cg.no_reachable_reason(intent, scene) is None


def test_the_window_is_the_settings_not_a_constant(coffee_files):
    cfg = locs.coffee_config()
    assert cg.in_coffee_window(cup(0.47, 0.015), cfg) and not cg.in_coffee_window(cup(0.45, 0.03), cfg) and not cg.in_coffee_window(cup(0.50, 0.0), cfg)
    coffee_files.write_text(yaml.safe_dump({'serve_xy': [0.45, 0.0], 'window_y_m': 0.05, 'window_x_m': 0.08}))
    assert cg.in_coffee_window(cup(0.50, 0.04), locs.coffee_config())


def test_the_nearest_reachable_cup_is_the_one_used_and_an_unreachable_one_is_ignored(coffee_files):
    scene = {'cup_a': cup(0.08, -0.58, reachable=False), 'cup_b': cup(0.30, 0.20), 'cup_c': cup(0.55, 0.30)}
    plan = cg.build_plan(cg.parse('give me coffee'), scene)
    assert plan[1]['object_id'] == 'cup_b'


# ---- the learning ---------------------------------------------------------------------------------------------------------------------------------------------
def test_the_learned_offset_starts_at_zero_takes_the_first_run_in_full_then_averages(tmp_path):
    f = str(tmp_path / 'l.yaml')
    assert cl.offset('serve', f) == (0.0, 0.0) and cl.corrected('serve', (0.45, 0.0), f) == (0.45, 0.0)
    used, text = cl.update('serve', (0.45, 0.0), (0.449, -0.059), f)
    assert used and cl.offset('serve', f) == pytest.approx((-0.001, -0.059), abs=1e-4)
    assert cl.corrected('serve', (0.45, 0.0), f) == pytest.approx((0.451, 0.059), abs=1e-4)          # commanded 5.9 cm to the other side, so the cup lands on the target
    cl.update('serve', (0.451, 0.059), (0.451, 0.019), f)                                           # it landed 4 cm off to the same side: measured offset -0.04
    assert cl.offset('serve', f)[1] == pytest.approx(0.5 * -0.059 + 0.5 * -0.04, abs=1e-4)
    rec = yaml.safe_load(open(f))['serve']
    assert rec['n'] == 2 and len(rec['history']) == 2


def test_a_wild_measurement_is_not_used_and_the_correction_is_capped(tmp_path):
    f = str(tmp_path / 'l.yaml')
    used, text = cl.update('serve', (0.45, 0.0), (0.45, 0.30), f)                                    # a different cup, a bumped table
    assert not used and 'not used' in text and cl.offset('serve', f) == (0.0, 0.0)
    cl.update('serve', (0.45, 0.0), (0.45, -0.14), f)
    assert cl.offset('serve', f)[1] == pytest.approx(-0.08)                                          # never more than 8 cm
    assert cl.update('serve', (0.45, 0.0), None, f)[0] is False
    (tmp_path / 'l.yaml').write_text('this: [is: not valid')
    assert cl.offset('serve', f) == (0.0, 0.0)                                                       # a broken file is the same as none


# ---- the controller ---------------------------------------------------------------------------------------------------------------------------------------------
def stub_world(node, monkeypatch, cups_after, colour='blue'):
    """_pick / _place / _safe_home as fakes. `cups_after`: the cup position the scene shows after each place (a list; the last one repeats)."""
    node.dry_run, node.moveit2 = False, object()
    log = {'picks': [], 'places': []}
    seq = list(cups_after)
    monkeypatch.setattr(node, '_settle_scene', lambda: None)
    monkeypatch.setattr(node, '_safe_home', lambda: True)

    def pick(oid, approach_z=0.15):
        log['picks'].append((oid, node._pick_side_az, node._next_place_xy))
        return True

    def place(x=None, y=None, z=None, here=False):
        log['places'].append((x, y))
        px, py = seq.pop(0) if len(seq) > 1 else seq[0]
        node.latest_scene = {'cup_new': cup(px, py, color=colour)}
        return True
    monkeypatch.setattr(node, '_pick', pick)
    monkeypatch.setattr(node, '_place', place)
    return log


STAGE = {'action': 'stage_object', 'object_id': 'cup_07', 'label': 'cup', 'x': 0.45, 'y': 0.0, 'tol_x': 0.04, 'tol_y': 0.02, 'tries': 3, 'approach_z': 0.35}


def test_a_cup_already_in_the_window_is_left_alone(node, monkeypatch):
    log = stub_world(node, monkeypatch, [(0.45, 0.0)])
    node.latest_scene = {'cup_07': cup(0.46, 0.01)}
    assert node._dispatch(dict(STAGE)) is True and log['picks'] == [] and log['places'] == []


def test_a_cup_elsewhere_is_picked_anyhow_put_at_the_spot_and_checked_again(node, monkeypatch):
    log = stub_world(node, monkeypatch, [(0.452, 0.011)])
    node.latest_scene = {'cup_07': cup(0.30, -0.30)}
    assert node._dispatch(dict(STAGE)) is True
    assert log['picks'] == [('cup_07', None, (0.45, 0.0))]                                             # any grasp (no forced side az), and the carry is checked against the spot
    assert log['places'] == [(0.45, 0.0)]
    assert cl.offset('stage')[1] == pytest.approx(0.011, abs=1e-6)                                    # what it saw (1.1 cm off in y) is learned for the next staging


def test_staging_tries_again_when_the_cup_lands_outside_the_window_and_gives_up_after_the_tries(node, monkeypatch):
    log = stub_world(node, monkeypatch, [(0.45, 0.05), (0.45, 0.01)])
    node.latest_scene = {'cup_07': cup(0.30, -0.30)}
    assert node._dispatch(dict(STAGE)) is True and len(log['places']) == 2
    assert log['places'][1][1] < 0.0                                                                  # the second command is already corrected by the 5 cm the first one was off
    log2 = stub_world(node, monkeypatch, [(0.45, 0.06)])
    node.latest_scene = {'cup_07': cup(0.30, -0.30)}
    assert node._dispatch(dict(STAGE)) is False and len(log2['places']) == 3


def test_staging_stops_without_placing_when_the_pick_fails_or_there_is_no_cup(node, monkeypatch):
    log = stub_world(node, monkeypatch, [(0.45, 0.0)])
    monkeypatch.setattr(node, '_pick', lambda oid, approach_z=0.15: False)
    node.latest_scene = {'cup_07': cup(0.30, -0.30)}
    assert node._dispatch(dict(STAGE)) is False and log['places'] == []
    node.latest_scene = {}
    assert node._dispatch(dict(STAGE)) is False
    assert node._dispatch({'action': 'stage_object', 'label': 'cup'}) is False                         # no x / y


def test_staging_is_logged_only_in_a_dry_run(node):
    node.dry_run = True
    try:
        node.latest_scene = {'cup_07': cup(0.30, -0.30)}
        assert node._dispatch(dict(STAGE)) is True
    finally:
        node.dry_run = False


def test_the_pick_after_staging_finds_the_cup_by_where_it_is(node, monkeypatch):
    node.dry_run, node.moveit2 = False, object()
    seen = []
    monkeypatch.setattr(node, '_pick', lambda oid, approach_z=0.15: seen.append((oid, node._pick_side_az)) or True)
    node.latest_scene = {'cup_12': cup(0.452, 0.004), 'cup_99': cup(0.2, 0.3)}                        # cup_07 was seen again as cup_12
    step = {'action': 'pick', 'object_id': 'cup_07', 'label': 'cup', 'near': [0.45, 0.0], 'side_az': 90.0, 'approach_z': 0.35}
    assert node._dispatch(step) is True and seen == [('cup_12', 90.0)] and node._pick_side_az is None     # the cup found by position, picked from the side; the azimuth is cleared afterwards
    node.latest_scene = {'cup_99': cup(0.2, 0.3), 'cup_07': cup(0.2, 0.3)}                             # nothing near the spot: the id of the plan is used (the pick then fails or not on its own terms)
    seen.clear()
    assert node._dispatch(step) is True and seen == [('cup_07', 90.0)]


def test_a_learned_placement_offset_is_subtracted_from_the_place_command_and_remembered(node, monkeypatch):
    node.dry_run, node.moveit2 = False, object()
    cl.update('serve', (0.45, 0.0), (0.449, -0.059))
    placed = []
    monkeypatch.setattr(node, '_place', lambda x=None, y=None, z=None, here=False: placed.append((x, y)) or True)
    assert node._dispatch({'action': 'place', 'x': 0.45, 'y': 0.0, 'learn': 'serve'}) is True
    assert placed[0] == pytest.approx((0.451, 0.059), abs=1e-6)
    assert node._learn_commanded['serve'] == pytest.approx((0.451, 0.059), abs=1e-6)
    placed.clear()
    assert node._dispatch({'action': 'place', 'x': 0.30, 'y': 0.1}) is True and placed == [(0.30, 0.1)]      # a place without the key is never changed


def test_check_placed_learns_from_the_run_and_fails_with_an_alert_when_no_cup_is_there(node, monkeypatch):
    node.dry_run, node.moveit2 = False, object()
    monkeypatch.setattr(node, '_settle_scene', lambda: None)
    statuses = []
    monkeypatch.setattr(node, '_publish_pp_status', lambda t: statuses.append(t))
    node._learn_commanded['serve'] = (0.45, 0.0)
    node.latest_scene = {'cup_x': cup(0.449, -0.059)}
    step = {'action': 'check_placed', 'label': 'cup', 'x': 0.45, 'y': 0.0, 'learn': 'serve'}
    assert node._dispatch(step) is True and cl.offset('serve') == pytest.approx((-0.001, -0.059), abs=1e-4)
    node.latest_scene = {'cup_x': cup(0.2, 0.3)}                                                      # a cup somewhere else is not the one that was put down
    assert node._dispatch(step) is False and any('ALERT' in t for t in statuses)
    node.latest_scene = {}
    assert node._dispatch(step) is False
    node.dry_run = True
    try:
        assert node._dispatch(step) is True
    finally:
        node.dry_run = False


def test_the_whole_staged_coffee_plan_runs_through_the_executor_in_a_dry_run(node, coffee_files, monkeypatch):
    node.dry_run = True
    node.latest_scene = {'cup_07': cup(0.30, -0.30)}
    node._scene_time = __import__('time').monotonic()
    seen = []
    real = node._dispatch
    monkeypatch.setattr(node, '_dispatch', lambda step: (seen.append(step['action']) or real(step)))
    try:
        plan = cg.build_plan(cg.parse('give me coffee'), node.latest_scene)
        node.executing = True
        node._execute(json.dumps({'command': 'give me coffee', 'plan': plan, 'model': 'grammar'}))
    finally:
        node.dry_run = False
    assert seen[:3] == ['go_home', 'stage_object', 'pick'] and seen[-1] == 'check_placed' and len(seen) == 10


# ---- 2026-11-05: the professor's long plan: the white cup from where it stands, through the coffee machine, and back there ------------------------------------------------
WHITE_FAR = {'cup_00': cup(0.217, -0.485, color='white'), 'cup_02': cup(0.44, 0.03, color='blue')}


def test_named_cup_and_put_it_back_phrasings_are_recognised(coffee_files):
    for text, colour, back in (('give me coffee in the white cup and put it back where it was', 'white', True),
                               ('make coffee in the white cup then bring it back', 'white', True),
                               ('please give me a coffee in the white mug and return it to where it was', 'white', True),
                               ('use the white cup to make coffee and put it back', 'white', True),
                               ('give me coffee in the white cup', 'white', False),
                               ('get me a cup of coffee in the blue mug', 'blue', False),
                               ('give me coffee in a cup', None, False)):
        i = cg.parse(text)
        assert i and i['kind'] == 'coffee' and i['cup_colour'] == colour and i['return'] is back, text
    plain = cg.parse('give me coffee')
    assert plain['kind'] == 'coffee' and plain['cup_colour'] is None and plain['return'] is False
    assert cg.parse('pick up the white cup')['kind'] == 'pick'                                  # other commands unchanged


def test_the_white_cup_plan_brings_it_through_the_machine_and_back_to_where_it_stood(coffee_files):
    intent = cg.parse('give me coffee in the white cup and put it back where it was')
    plan = cg.build_plan(intent, WHITE_FAR)
    acts = [(s['action'], s.get('stage')) for s in plan]
    assert acts == [('go_home', None), ('stage_object', None), ('pick', None), ('go_to_location', 'put'), ('wait', None), ('go_to_location', 'in'), ('go_to_location', 'take'), ('place', None),
                    ('go_home', None), ('check_placed', None), ('stage_object', None), ('go_home', None), ('check_placed', None)]
    out, back = plan[1], plan[10]
    assert out['object_id'] == 'cup_00' and out['colour'] == 'white' and out['from'] == [0.217, -0.485] and (out['x'], out['y']) == (0.45, 0.0)       # to the serving spot
    assert plan[2]['near'] == [0.45, 0.0] and plan[2]['colour'] == 'white' and plan[2]['side_az'] == 90.0
    assert back['colour'] == 'white' and back['from'] == [0.45, 0.0] and (back['x'], back['y']) == (0.217, -0.485) and back['to_origin'] is True    # and back to where it stood
    assert plan[9]['learn'] == 'serve' and plan[9]['colour'] == 'white' and (plan[12]['x'], plan[12]['y']) == (0.217, -0.485) and 'learn' not in plan[12]
    assert cg.check(intent, plan, WHITE_FAR) == (True, '') and cg.no_reachable_reason(intent, WHITE_FAR) is None


def test_a_cup_already_at_the_serving_spot_needs_no_return_and_a_missing_colour_is_said_plainly(coffee_files):
    intent = cg.parse('give me coffee in the white cup and put it back')
    here = {'cup_00': cup(0.451, 0.01, color='white')}
    plan = cg.build_plan(intent, here)
    assert [s['action'] for s in plan].count('stage_object') == 0 and plan[-1]['action'] == 'check_placed'      # it stands at the spot: nothing to bring and nothing to return
    assert 'no white cup' in cg.no_reachable_reason(intent, {'cup_02': cup(0.44, 0.03, color='blue')})
    # the plain command still takes the nearest cup of any colour, as before
    plain = cg.build_plan(cg.parse('give me coffee'), WHITE_FAR)
    assert plain[1]['object_id'] == 'cup_02' and 'colour' not in plain[1] and not any(s.get('to_origin') for s in plain)


def test_staging_picks_the_named_colour_from_where_it_is_not_the_nearest_cup_to_the_spot(node, monkeypatch):
    log = stub_world(node, monkeypatch, [(0.45, 0.0)], colour='white')
    node.latest_scene = {'cup_00': cup(0.217, -0.485, color='white'), 'cup_02': cup(0.47, 0.06, color='blue')}          # the blue one is nearer to the serving spot
    step = dict(STAGE, object_id='cup_00', colour='white', **{'from': [0.217, -0.485]})
    assert node._dispatch(step) is True
    assert log['picks'] == [('cup_00', None, (0.45, 0.0))] and log['places'] == [(0.45, 0.0)]                          # the white one was picked
    assert node.latest_scene['cup_new']['color'] == 'white'


def test_the_return_trip_looks_for_the_cup_at_the_serving_spot_and_ends_when_it_stands_at_the_origin(node, monkeypatch):
    node.dry_run, node.moveit2 = False, object()
    monkeypatch.setattr(node, '_settle_scene', lambda: None)
    monkeypatch.setattr(node, '_safe_home', lambda: True)
    node.latest_scene = {'cup_00': cup(0.451, 0.005, color='white'), 'cup_02': cup(0.22, -0.47, color='blue')}     # the white cup at the serving spot; a blue cup stands NEAR the origin
    picked = []
    monkeypatch.setattr(node, '_pick', lambda oid, approach_z=0.15: picked.append(oid) or True)

    def place(x=None, y=None, z=None, here=False):
        node.latest_scene = {'cup_07': cup(x + 0.004, y - 0.006, color='white'), 'cup_02': cup(0.22, -0.47, color='blue')}
        return True
    monkeypatch.setattr(node, '_place', place)
    step = {'action': 'stage_object', 'object_id': 'cup_00', 'label': 'cup', 'colour': 'white', 'from': [0.45, 0.0], 'x': 0.217, 'y': -0.485, 'tol_x': 0.04, 'tol_y': 0.04, 'tries': 3, 'approach_z': 0.35,
            'to_origin': True}
    assert node._dispatch(step) is True
    assert picked == ['cup_00']                                                                                          # the white cup, although the blue one is closer to the origin


def test_check_placed_looks_for_the_named_colour_only(node, monkeypatch):
    node.dry_run, node.moveit2 = False, object()
    monkeypatch.setattr(node, '_settle_scene', lambda: None)
    statuses = []
    monkeypatch.setattr(node, '_publish_pp_status', lambda t: statuses.append(t))
    node.latest_scene = {'cup_x': cup(0.22, -0.48, color='blue')}                                                       # a blue cup stands there, the white one is not
    step = {'action': 'check_placed', 'label': 'cup', 'colour': 'white', 'x': 0.217, 'y': -0.485}
    assert node._dispatch(step) is False and any('ALERT' in t for t in statuses)
    node.latest_scene = {'cup_y': cup(0.22, -0.48, color='white')}
    assert node._dispatch(step) is True


def test_the_whole_white_cup_plan_runs_through_the_executor_in_a_dry_run(node, coffee_files, monkeypatch):
    node.dry_run = True
    node.latest_scene = dict(WHITE_FAR)
    node._scene_time = __import__('time').monotonic()
    seen = []
    real = node._dispatch
    monkeypatch.setattr(node, '_dispatch', lambda step: (seen.append(step['action']) or real(step)))
    try:
        plan = cg.build_plan(cg.parse('give me coffee in the white cup and put it back where it was'), node.latest_scene)
        node.executing = True
        node._execute(json.dumps({'command': 'white cup', 'plan': plan, 'model': 'grammar'}))
    finally:
        node.dry_run = False
    assert len(seen) == 13 and seen.count('stage_object') == 2 and seen.count('check_placed') == 2
