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
