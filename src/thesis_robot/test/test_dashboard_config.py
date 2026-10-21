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


# ── the third RealSense (replaced the OAK-D, 2026-10-20): every place that lists cameras must know it ───────────────────────
import os as _os
import re as _re

_REPO = _os.path.abspath(_os.path.join(_os.path.dirname(__file__), '..', '..', '..'))


def _read(rel):
    with open(_os.path.join(_REPO, rel)) as f:
        return f.read()


def test_the_third_realsense_is_wired_everywhere_the_second_one_is():
    from app import config
    assert config.CAMERA_TOPICS['realsense3']['image'] == '/global_camera_3/global_camera_3/color/image_raw'
    assert config.CAMERA_SERIALS['RS3'] == '207522071578'
    for proc in ('realsense3_tf_broadcaster', 'realsense3_detection'):
        assert proc in config.PROCESSES
    assert 'global_camera_3_color_optical_frame' in config.PROCESSES['realsense3_tf_broadcaster']['cmd']
    order = [p for p, _ in config.FULL_BRINGUP_ORDER]
    assert 'realsense2_tf_broadcaster' in order and 'realsense3_detection' in order
    # the module the dashboard starts exists and publishes the topic the scene graph listens to
    src = _read('src/thesis_robot/thesis_robot/realsense3_detection.py')
    assert '/detections_realsense3' in src and "'realsense3'" in src and '/global_camera_3/global_camera_3/color/image_raw' in src
    assert 'realsense2' not in src and 'global_camera_2' not in src
    assert '/detections_realsense3' in _read('src/thesis_robot/thesis_robot/scene_graph_node.py')
    assert "'realsense3'" in _read('src/thesis_robot/thesis_robot/camera_watchdog.py')
    assert "'realsense3'" in _read('src/thesis_robot/thesis_robot/obstacle_guard_node.py')
    assert 'realsense3_detection' in _read('src/thesis_robot/setup.py')
    assert "'realsense3'" in _read('dashboard/backend/app/routers/calibration.py')
    for tsx in ('MultiCameraView', 'ArmCalibration', 'CalibrationWizard', 'WorkspaceBoundary', 'TopStatusBar'):
        assert 'realsense3' in _read(f'dashboard/frontend/src/components/{tsx}.tsx'), tsx


def test_the_launch_file_pins_all_three_serials_and_both_copies_are_identical():
    rel = 'robot_bringup/ros2_kortex/kortex_moveit_config/kinova_gen3_7dof_robotiq_2f_140_moveit_config/launch/cameras.launch.py'
    launch = _read(rel)
    for serial in ('938422070760', '215322071290', '207522071578'):
        assert launch.count(serial) == 1, serial
    assert 'global_camera_3' in launch and 'realsense3_node,' in launch
    assert _re.search(r'"launch_oak_camera",\s*default_value="false"', launch)
    ws_copy = '/mnt/ros_workspace/ros2_kortex_ws/src/ros2_kortex/kortex_moveit_config/kinova_gen3_7dof_robotiq_2f_140_moveit_config/launch/cameras.launch.py'
    if _os.path.exists(ws_copy):
        with open(ws_copy) as f:
            assert f.read() == launch, 'the workspace copy of cameras.launch.py differs from the repo copy'


def test_the_full_bring_up_skips_the_third_camera_until_it_has_a_calibration_file(tmp_path, monkeypatch):
    from app import config
    assert set(config.NEEDS_CALIBRATION_FILE) == {'realsense3_tf_broadcaster', 'realsense3_detection'}
    assert all(p in config.PROCESSES for p in config.NEEDS_CALIBRATION_FILE)


def test_a_static_camera_without_a_calibration_file_is_never_used(tmp_path, monkeypatch):
    """2026-10-20: a removed trial calibration left a stale static transform in the long-running nodes and drew a phantom obstacle."""
    import importlib
    ccf = importlib.import_module('thesis_robot.camera_calibration_files')
    monkeypatch.setitem(ccf.CALIBRATION_FILES, 'realsense3', str(tmp_path / 'realsense3_calibration.yaml'))
    assert ccf.has_calibration('realsense3') is False
    (tmp_path / 'realsense3_calibration.yaml').write_text('x: 1\n')
    assert ccf.has_calibration('realsense3') is True
    assert ccf.has_calibration('wrist') is True                      # calibrated by the arm's kinematics, needs no file
    for module in ('obstacle_guard_node', 'scene_graph_node'):
        assert 'has_calibration' in _read(f'src/thesis_robot/thesis_robot/{module}.py'), module
    assert 'has_calibration' in _read('dashboard/backend/app/ros_bridge.py')
