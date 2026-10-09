import importlib
import os
import sys

BACKEND = os.path.join(os.path.dirname(__file__), '..', '..', '..', 'dashboard', 'backend')


def load_config(tmp_path, monkeypatch, disabled_text):
    (tmp_path / 'config').mkdir()
    (tmp_path / 'config' / 'disabled_cameras.txt').write_text(disabled_text)
    monkeypatch.setenv('SHASHPROJECT_REPO_ROOT', str(tmp_path))
    monkeypatch.syspath_prepend(BACKEND)
    sys.modules.pop('app.config', None)
    return importlib.import_module('app.config')


def test_a_disabled_camera_takes_its_processes_out_of_the_bring_up(tmp_path, monkeypatch):
    cfg = load_config(tmp_path, monkeypatch, '# parked\noakd\n')
    assert cfg.disabled_camera_procs() == {'oakd_driver', 'oakd_tf_broadcaster', 'object_detection'}


def test_nothing_disabled_means_nothing_skipped(tmp_path, monkeypatch):
    cfg = load_config(tmp_path, monkeypatch, '# oakd is back\n')
    assert cfg.disabled_camera_procs() == set()


def test_a_missing_file_skips_nothing(tmp_path, monkeypatch):
    monkeypatch.setenv('SHASHPROJECT_REPO_ROOT', str(tmp_path))
    monkeypatch.syspath_prepend(BACKEND)
    sys.modules.pop('app.config', None)
    cfg = importlib.import_module('app.config')
    assert cfg.disabled_camera_procs() == set()
