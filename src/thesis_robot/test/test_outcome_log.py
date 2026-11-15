import json
import os

from thesis_robot import outcome_log as ol
from thesis_robot import outcome_report as rep


def test_append_and_read_back(tmp_path):
    log = ol.OutcomeLog(str(tmp_path))
    log.append({'kind': 'pick', 'label': 'mouse', 'ok': True})
    log.append({'kind': 'place', 'ok': False, 'failed_step': 'lower'})
    recs = ol.read_records(str(tmp_path))
    assert [r['kind'] for r in recs] == ['pick', 'place'] and all('ts' in r for r in recs)


def test_unreadable_lines_are_skipped(tmp_path):
    (tmp_path / 'outcomes-2026-10.jsonl').write_text('{"kind":"pick","ts":1}\nnot json\n{"kind":"place","ts":2}\n')
    assert len(ol.read_records(str(tmp_path))) == 2


def test_attempt_collects_notes_events_steps_and_grip():
    a = ol.Attempt('pick', label='mouse', live=True)
    a.note(strategy='narrowed')
    a.event('align', turn_deg=-77)
    a.step_time('descend', 2.34)
    a.grip_reading('check the grip', 0.49)
    rec = a.finish(False, 'check the grip', 'nothing in the gripper')
    assert rec['kind'] == 'pick' and rec['strategy'] == 'narrowed' and rec['ok'] is False
    assert rec['steps']['descend'] == 2.34 and rec['grip'][0]['finger'] == 0.49
    assert rec['events'][0]['what'] == 'align'
    json.dumps(rec)                                  # serialisable


def test_numpy_values_are_serialised(tmp_path):
    import numpy as np
    ol.OutcomeLog(str(tmp_path)).append({'x': np.float64(1.5), 'v': np.array([1, 2])})
    assert os.listdir(str(tmp_path))


def test_neighbours_sorted_nearest_first():
    n = ol.neighbours([('bowl', 0.4, 0.0), ('cup', 0.1, 0.0)], (0.0, 0.0), k=2)
    assert [x['label'] for x in n] == ['cup', 'bowl'] and n[0]['dist_m'] == 0.1


def _pick(label, strategy, ok, near=0.3, step='check the grip'):
    return {'kind': 'pick', 'live': True, 'label': label, 'strategy': strategy, 'ok': ok,
            'failed_step': None if ok else step, 'neighbours': [{'dist_m': near}],
            'grip': [{'at': 'x', 'finger': 0.49}] if ok else []}


def test_summary_and_lessons():
    recs = [_pick('mouse', 'turned_90', False), _pick('mouse', 'turned_90', False),
            _pick('mouse', 'full_open', True), _pick('mouse', 'full_open', True), _pick('cup', 'full_open', True),
            {'kind': 'pick', 'live': False, 'label': 'cup', 'ok': False, 'strategy': 'full_open'}]
    s = rep.summarize(recs)
    assert s['picks'] == {'n': 5, 'ok': 3}                       # the dry-run record is ignored
    assert s['by_label']['mouse'] == (2, 4)
    assert any('mouse' in t and 'turned_90' in t and 'failed 2/2' in t for t in s['lessons'])
    assert s['grip_median']['mouse'] == 0.49
    assert 'Lessons' in rep.format_summary(s)


def test_empty_log_summarises_without_error():
    assert 'picks' in rep.format_summary(rep.summarize([]))


def test_append_adds_the_schema_fields_but_never_overrides_them(tmp_path):
    from thesis_robot import code_version as cv
    log = ol.OutcomeLog(str(tmp_path))
    rec = log.append({'kind': 'pick'})
    assert rec['schema_version'] == cv.SCHEMA_VERSION and rec['session_id'] == cv.SESSION_ID
    assert rec['grip_check_version'] == cv.GRIP_CHECK_VERSION
    rec2 = log.append({'kind': 'pick', 'schema_version': 99, 'session_id': 'custom', 'grip_check_version': 7})
    assert (rec2['schema_version'], rec2['session_id'], rec2['grip_check_version']) == (99, 'custom', 7)


def test_pool_for_scene_is_stable_and_empty_scene_id_is_not_assigned():
    assert ol.pool_for_scene('') == ''
    a = ol.pool_for_scene('layout_12')
    assert a == ol.pool_for_scene('layout_12')                   # same scene -> same pool, every time
    assert a in ('train_pool', 'test_pool')
    pools = {ol.pool_for_scene(f'layout_{i}') for i in range(50)}
    assert pools == {'train_pool', 'test_pool'}                  # both actually occur, not a constant


def test_categorize_failure_matches_the_task7_taxonomy():
    assert ol.categorize_failure('pick', 'check the grip', '') == ('parameter-related', 'high')
    assert ol.categorize_failure('pick', 'hover', '') == ('planning-or-kinematics', 'high')
    assert ol.categorize_failure('pick', 'turn the fingers away from any handle', '') == ('planning-or-kinematics', 'low')
    assert ol.categorize_failure('pick', 'check the open fingers are clear of its neighbours',
                                  'pick: the open fingers would hit the cup next to the bowl') == ('parameter-related', 'high')
    assert ol.categorize_failure('place', 'carry', 'move_to (0.1,0.1,0.1) unsafe: x') == ('planning-or-kinematics', 'high')
    assert ol.categorize_failure('pick', 'something new', 'no idea what this is') == (None, None)


def test_pick_finish_carries_pre_action_features_and_a_failure_category():
    a = ol.Attempt('pick', object_id='cup_00', label='cup', xy=(0.4, 0.0), z=0.05, attempt=1,
                   neighbours=[], grasp_z=0.27, hover_z=0.39, tip_clearance_m=0.03, scene_conf=0.9,
                   unknown_near=[], live=True)
    a.note(method='top', side_az=None, strategy='full_open')     # decided mid-pick: must NOT leak into pre_action_features
    rec = a.finish(False, 'check the grip', '')
    assert rec['pre_action_features'] == {
        'object_id': 'cup_00', 'label': 'cup', 'xy': (0.4, 0.0), 'z': 0.05, 'attempt': 1, 'neighbours': [],
        'grasp_z': 0.27, 'hover_z': 0.39, 'tip_clearance_m': 0.03, 'scene_conf': 0.9, 'unknown_near': []}
    assert 'method' not in rec['pre_action_features'] and 'strategy' not in rec['pre_action_features']
    assert (rec['failure_category'], rec['failure_category_confidence']) == ('parameter-related', 'high')


def test_an_ok_pick_and_an_uncategorized_failure_carry_no_failure_category():
    ok_rec = ol.Attempt('pick', label='cup', live=True).finish(True)
    assert 'failure_category' not in ok_rec
    unknown_rec = ol.Attempt('pick', label='cup', live=True).finish(False, 'brand_new_step', 'brand new reason')
    assert 'failure_category' not in unknown_rec


def test_command_finish_has_no_pre_action_features():
    rec = ol.Attempt('command', command='go home', live=True).finish(True)
    assert 'pre_action_features' not in rec
