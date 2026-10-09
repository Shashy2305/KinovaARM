import pytest

from thesis_robot import grasp_planner as gp
from thesis_robot import grasp_policy as gpol
from thesis_robot import outcome_report as rep


def pick(label, rho, ok, live=True):
    return {'kind': 'pick', 'live': live, 'label': label, 'ok': ok, 'events': [{'what': 'shape', 'rho': rho}]}


def test_fold_and_bins():
    assert gpol.fold(100) == 80 and gpol.fold(10) == 10 and gpol.fold(170) == 10
    assert gpol.bin_of(5) == 0 and gpol.bin_of(45) == 1 and gpol.bin_of(90) == 2 and gpol.bin_of(175) == 0


def test_stats_use_only_live_picks_with_a_shape_event():
    recs = [pick('mouse', 90, True), pick('mouse', 90, False), pick('mouse', 90, True, live=False),
            {'kind': 'pick', 'live': True, 'label': 'mouse', 'ok': True, 'events': []},
            {'kind': 'place', 'live': True, 'label': 'mouse', 'ok': True}]
    pol = gpol.GraspPolicy(recs)
    assert pol.counts('mouse', 90) == (1, 2)


def test_a_failing_direction_sinks_and_a_working_one_rises():
    recs = [pick('mouse', 5, False) for _ in range(4)] + [pick('mouse', 90, True) for _ in range(4)]
    pol = gpol.GraspPolicy(recs)
    assert pol.prior('mouse', 5) < 0.45 < pol.prior('mouse', 90)
    assert pol.prior('mouse', 90) > 0.8


def test_an_untried_direction_is_not_ruled_out():
    pol = gpol.GraspPolicy([pick('mouse', 90, True)] * 5)
    assert pol.prior('mouse', 45) == pytest.approx(0.8)            # optimism for the unknown: 0.5 + 0.3 explore bonus
    assert gpol.GraspPolicy([]).prior('cup', 10) == pytest.approx(0.8)
    assert pol.prior('mouse', 90) > pol.prior('mouse', 45)         # but a proven direction still ranks above it


def test_exploration_bonus_shrinks_with_evidence():
    few = gpol.GraspPolicy([pick('cup', 90, False)])
    many = gpol.GraspPolicy([pick('cup', 90, False)] * 20)
    assert few.prior('cup', 90) > many.prior('cup', 90)


def test_lessons_are_plain_language_and_need_two_trials():
    recs = [pick('mouse', 5, False)] * 2 + [pick('mouse', 90, True)] * 3 + [pick('cup', 90, True)]
    text = ' | '.join(gpol.GraspPolicy(recs).lessons())
    assert 'mouse along its length has failed 2/2' in text and 'mouse across its short side has worked 3/3' in text
    assert 'cup' not in text                                       # one trial is not a lesson


def test_the_prior_reorders_candidates_but_cannot_resurrect_an_excluded_one():
    prof = {0: 0.07, 90: 0.07}
    cont = {0: 0.04, 90: 0.04}                                     # identical shape scores
    pol = gpol.GraspPolicy([pick('thing', 90, False)] * 4 + [pick('thing', 0, True)] * 4)
    prior = lambda rho: pol.prior('thing', rho)                    # noqa: E731
    ranked = gp.rank_grasps(prof, cont, long_axis_deg=0.0, prior=prior)
    assert ranked[0]['phi'] == 0                                   # rho 0 works, rho 90 failed: closing along 0 comes first
    blocked_best = gp.rank_grasps(prof, cont, long_axis_deg=0.0, prior=prior, blocked=lambda phi: phi == 0)
    assert [c['phi'] for c in blocked_best] == [90]                # the hard constraint still wins


def test_report_includes_the_grasp_lessons():
    recs = [pick('mouse', 5, False)] * 2 + [pick('mouse', 90, True)] * 2
    s = rep.summarize(recs)
    assert any('mouse' in t for t in s['grasp_lessons'])
    assert 'Lessons' in rep.format_summary(s)
