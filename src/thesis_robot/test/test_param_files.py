"""The parameter files in config/ must stay in step with the nodes: every key declared, same type, and (as shipped) the
code's default. Otherwise a typo in a file would silently do nothing, or a type mismatch would stop the node starting."""
import os

import pytest
import yaml

rclpy = pytest.importorskip('rclpy')
CONFIG = os.path.join(os.path.dirname(__file__), '..', '..', '..', 'config')


def load(name, node_name):
    with open(os.path.join(CONFIG, name)) as f:
        return yaml.safe_load(f)[node_name]['ros__parameters']


@pytest.fixture(scope='module')
def ros():
    rclpy.init()
    yield
    rclpy.shutdown()


def check(node, params):
    declared = {p: node.get_parameter(p).value for p in node._parameters}
    for key, val in params.items():
        assert key in declared, f'{key} is in the parameter file but the node does not declare it'
        assert type(declared[key]) is type(val), f'{key}: file has {type(val).__name__}, node expects {type(declared[key]).__name__}'
        assert declared[key] == val, f'{key}: the shipped file says {val!r}, the code default is {declared[key]!r}'


def test_arm_controller_file_matches_the_node(ros):
    from thesis_robot.arm_controller_node import ArmControllerNode
    n = ArmControllerNode()
    try:
        check(n, load('arm_controller_params.yaml', 'arm_controller'))
    finally:
        n.destroy_node()


def test_planner_file_matches_the_node(ros):
    from thesis_robot.llm_planner_node import LLMPlannerNode
    n = LLMPlannerNode()
    try:
        check(n, load('llm_planner_params.yaml', 'llm_planner'))
    finally:
        n.destroy_node()


def test_the_dashboard_builds_start_commands_that_load_the_files(monkeypatch):
    import sys
    monkeypatch.setenv('SHASHPROJECT_REPO_ROOT', os.path.abspath(os.path.join(CONFIG, '..')))
    monkeypatch.syspath_prepend(os.path.join(CONFIG, '..', 'dashboard', 'backend'))
    sys.modules.pop('app.config', None)
    import importlib
    cfg = importlib.import_module('app.config')
    assert '--params-file' in cfg.PROCESSES['arm_controller']['cmd'] and 'arm_controller_params.yaml' in cfg.PROCESSES['arm_controller']['cmd']
    assert cfg.PROCESSES['arm_controller']['cmd'].endswith('-p dry_run:=true')           # dry_run still always wins, last
    assert 'llm_planner_params.yaml' in cfg.PROCESSES['llm_planner_node']['cmd']
    assert cfg.ros_args('no_such_node') == ''
