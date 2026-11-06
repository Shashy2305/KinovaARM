"""2026-11-05: every new outcome record carries the code tag (commit, dirty flag, branch, settings hash)."""
import json

from thesis_robot import code_version as cv
from thesis_robot import outcome_log as ol


def test_the_tag_names_the_commit_the_branch_and_the_settings(tmp_path, monkeypatch):
    t = cv.tag(force=True)
    assert set(t) == {'git', 'dirty', 'branch', 'cfg'} and len(t['cfg']) == 8
    assert t['git'] != 'unknown' and 4 <= len(t['git']) <= 12 and isinstance(t['dirty'], bool)
    # the settings hash follows the files
    a, b = tmp_path / 'a.yaml', tmp_path / 'b.yaml'
    a.write_text('x: 1')
    h1 = cv.config_hash((str(a), str(b)))
    a.write_text('x: 2')
    assert cv.config_hash((str(a), str(b))) != h1 and cv.config_hash((str(tmp_path / 'none1'), str(tmp_path / 'none2'))) == cv.config_hash((str(tmp_path / 'none3'), str(tmp_path / 'none4')))


def test_without_git_the_tag_says_unknown_and_never_raises(monkeypatch):
    monkeypatch.setattr(cv, 'repo_root', lambda start=None: None)
    t = cv.tag(force=True)
    assert t['git'] == 'unknown' and t['dirty'] is None
    monkeypatch.setattr(cv, 'repo_root', lambda start=None: '/nonexistent/path')
    assert cv.tag(force=True)['git'] == 'unknown'
    cv.tag(force=True)


def test_every_record_written_gets_the_tag_and_an_explicit_one_is_kept(tmp_path):
    log = ol.OutcomeLog(str(tmp_path))
    rec = log.append({'kind': 'pick', 'ok': True})
    assert rec['code']['git'] and 'cfg' in rec['code']
    kept = log.append({'kind': 'place', 'code': {'git': 'abc1234'}})
    assert kept['code'] == {'git': 'abc1234'}
    lines = [json.loads(l) for f in tmp_path.iterdir() for l in open(f)]
    assert all('code' in r for r in lines) and len(lines) == 2
