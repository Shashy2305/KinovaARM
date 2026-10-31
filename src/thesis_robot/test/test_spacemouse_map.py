"""SpaceMouse -> jog mapping (dashboard/backend/app/spacemouse_map.py): the shaping that decides what a push on the cap may do to the arm."""
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', 'dashboard', 'backend'))
from app import spacemouse_map as sm  # noqa: E402


def test_a_resting_cap_sends_nothing():
    assert sm.to_jog_vector([0.05, -0.08, 0.1, 0.0, 0.0, 0.11]) == [0.0] * 6


def test_the_dead_zone_edge_is_continuous_and_full_push_is_one():
    assert sm.shape(0.12) == 0.0 and sm.shape(0.121) < 0.001
    assert sm.shape(1.0) == pytest.approx(1.0) and sm.shape(-1.0) == pytest.approx(-1.0)
    assert 0 < sm.shape(0.3) < sm.shape(0.6) < 1


def test_the_curve_keeps_small_pushes_small():
    assert sm.shape(0.4, expo=2.0) < 0.25


def test_translation_or_rotation_never_both():
    v = sm.to_jog_vector([0.0, 0.8, 0.0, 0.3, 0.0, 0.0])                  # push (dev y) with a little twist
    assert any(v[:3]) and v[3:] == [0.0, 0.0, 0.0]
    v = sm.to_jog_vector([0.2, 0.0, 0.0, 0.0, 0.9, 0.0])                  # mostly a tilt
    assert v[:3] == [0.0, 0.0, 0.0] and any(v[3:])


def test_the_mapping_picks_axis_and_sign():
    m = dict(sm.DEFAULT_MAP)
    m['x'] = (1, -1)
    v = sm.to_jog_vector([0.0, 1.0, 0.0, 0.0, 0.0, 0.0], m)
    assert v[0] == pytest.approx(-1.0) and v[1:] == [0.0] * 5


def test_dominant_keeps_only_the_strongest_axis_of_the_group():
    m = {'x': (0, 1), 'y': (1, 1), 'z': (2, 1), 'rx': (3, 1), 'ry': (4, 1), 'rz': (5, 1)}
    v = sm.to_jog_vector([0.9, 0.5, 0.3, 0, 0, 0], m, dominant=True)
    assert v[0] > 0 and v[1] == 0 and v[2] == 0


def test_every_output_stays_inside_minus_one_to_one():
    v = sm.to_jog_vector([5, -5, 5, 0, 0, 0])                            # (the device layer clips; the mapping must not exceed 1 either)
    assert all(-1.0 <= c <= 1.0 for c in v)


def test_deadman_modes():
    assert sm.deadman_ok({0x100: True}) and sm.deadman_ok({0x101: True})
    assert not sm.deadman_ok({}) and not sm.deadman_ok({0x100: False, 0x101: False})
    assert sm.deadman_ok({0x100: True}, 'left') and not sm.deadman_ok({0x101: True}, 'left')
    assert sm.deadman_ok({0x101: True}, 'right') and not sm.deadman_ok({0x100: True}, 'right')


def test_a_saved_mapping_round_trips_and_a_broken_one_falls_back(tmp_path):
    p = str(tmp_path / 'map.json')
    m = {'x': (2, 1), 'y': (0, -1), 'z': (1, 1), 'rx': (3, 1), 'ry': (4, -1), 'rz': (5, 1)}
    sm.save_map(m, p)
    assert sm.load_map(p) == m
    (tmp_path / 'bad.json').write_text('{"x": [9, 1]}')
    assert sm.load_map(str(tmp_path / 'bad.json')) == sm.DEFAULT_MAP
    assert sm.load_map(str(tmp_path / 'missing.json')) == sm.DEFAULT_MAP
    dup = {k: (0, 1) for k in sm.AXIS_NAMES}
    sm.save_map(dup, p)
    assert sm.load_map(p) == sm.DEFAULT_MAP                               # two jog axes on one device axis is not a mapping


def test_the_probe_result_must_be_complete_and_unique():
    ok = {'x': (1, -1), 'y': (0, 1), 'z': (2, -1), 'rx': (4, 1), 'ry': (3, -1), 'rz': (5, -1)}
    assert sm.mapping_from_probe(ok) == ok
    with pytest.raises(ValueError):
        sm.mapping_from_probe({**ok, 'rz': None})
    with pytest.raises(ValueError):
        sm.mapping_from_probe({**ok, 'rz': (0, 1)})
