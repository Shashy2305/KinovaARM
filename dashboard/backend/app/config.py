"""
Single source of truth for every process the dashboard can start/stop/watch.
Deliberately mirrors README.md's "Running" section commands exactly — the
dashboard runs the same commands already proven by hand, not new launch
semantics, so there's one place "how does this pipeline start" can drift
out of sync, not two.

Each entry's `signature` is the substring process_manager.py greps `ps aux`
for to detect "is this already running" BEFORE starting it — the guard
against the exact two-OAK-D-drivers class of bug hit during development.
"""
import os

# Overridable via env so this same config works unmodified on any machine
# this repo gets checked out on (different machines keep this repo and the
# ros2_kortex workspace at different absolute paths -- e.g. the lab
# laptop uses the defaults below, REAL-1 sets SHASHPROJECT_REPO_ROOT /
# SHASHPROJECT_WORKSPACE_ROOT to its /mnt/ros_workspace/... paths).
REPO_ROOT = os.environ.get('SHASHPROJECT_REPO_ROOT', os.path.expanduser('~/Shashproject'))
WORKSPACE_ROOT = os.environ.get('SHASHPROJECT_WORKSPACE_ROOT', os.path.expanduser('~/workspace/ros2_kortex_ws'))
LOG_DIR = os.path.expanduser('~/.ros/dashboard_logs')
ROBOT_IP = os.environ.get('KINOVA_ROBOT_IP', '192.168.1.10')
# Only set where heavy Python deps (ultralytics, depthai, pyrealsense2, ...)
# live in a dedicated venv rather than being installed directly -- empty
# string is a no-op so this stays harmless where there's no venv.
VENV_ACTIVATE = os.environ.get('SHASHPROJECT_VENV_ACTIVATE', '')

ROS_ENV_CMD = (
    'source /opt/ros/humble/setup.bash && '
    f'source {WORKSPACE_ROOT}/install/setup.bash && '
    f'source {REPO_ROOT}/install/setup.bash && '
    + (f'source {VENV_ACTIVATE} && ' if VENV_ACTIVATE else '')
    + 'export ROS_DOMAIN_ID=42 && '
)

CAMERA_TOPICS = {
    'oakd':       {'image': '/global_camera/color/image_raw'},
    'realsense':  {'image': '/global_camera/global_camera/color/image_raw'},
    'realsense2': {'image': '/global_camera_2/global_camera_2/color/image_raw'},
    'wrist':      {'image': '/camera/color/image_raw'},
}

# category groups the Node Control grid in the UI; hardware_affecting drives
# the "are you sure" styling (not a hard block) for anything that moves the
# arm or talks to a physical device.
def params_file_arg(node_name):
    """'--params-file <repo>/config/<node_name>_params.yaml ' if that file exists, else ''. The dashboard restarts the arm controller
    (Go Live, Go Dry Run), which resets every `ros2 param set`; a parameters file makes a choice survive the restart."""
    path = os.path.join(REPO_ROOT, 'config', f'{node_name}_params.yaml')
    return f'--params-file {path} ' if os.path.exists(path) else ''


def ros_args(node_name, *extra):
    """' --ros-args <params file> <extra...>' for `ros2 run`, or '' when there is nothing to pass."""
    parts = (params_file_arg(node_name) + ' '.join(extra)).strip()
    return f' --ros-args {parts}' if parts else ''


# RealSense serials, to name which camera another user's launch command occupies (hardware_holders)
CAMERA_SERIALS = {'RS1': '938422070760', 'RS2': '215322071290'}

PROCESSES = {
    'robot_bringup': {
        'label': 'Robot bringup (robot.launch.py)',
        'category': 'robot',
        'hardware_affecting': True,
        'cmd': (
            f'{ROS_ENV_CMD} cd {WORKSPACE_ROOT} && '
            f'ros2 launch kinova_gen3_7dof_robotiq_2f_140_moveit_config '
            f'robot.launch.py robot_ip:={ROBOT_IP}'
        ),
        'signature': 'kinova_gen3_7dof_robotiq_2f_140_moveit_config robot.launch.py',
    },
    'cameras_bringup': {
        'label': 'Cameras bringup (composite — see process_manager.start_cameras)',
        'category': 'cameras',
        'hardware_affecting': True,
        'cmd': None,  # composite start handled in process_manager.py
        'signature': 'kinova_gen3_7dof_robotiq_2f_140_moveit_config cameras.launch.py',
    },
    'oakd_driver': {
        'label': 'OAK-D driver (this repo, 640x400)',
        'category': 'cameras',
        'hardware_affecting': True,
        'cmd': (
            f'{ROS_ENV_CMD} cd {REPO_ROOT} && '
            'python3 drivers/oak_camera_node.py --ros-args '
            '-p image_width:=640 -p image_height:=400'
        ),
        'signature': 'drivers/oak_camera_node.py',
        # Any OTHER process matching this signature is a conflict, not just
        # "already running" -- e.g. the stale external copy in
        # ~/workspace/ros2_kortex_ws/oak_camera_node.py. See
        # process_manager.py's duplicate guard.
        'conflict_signatures': ['oak_camera_node.py'],
    },
    'oakd_tf_broadcaster': {
        'label': 'OAK-D TF broadcaster',
        'category': 'cameras',
        'hardware_affecting': False,
        'cmd': (
            f'{ROS_ENV_CMD} ros2 run thesis_robot camera_tf_broadcaster --ros-args '
            '-r __node:=oakd_tf_broadcaster '
            '-p calibration_file:=~/.ros/oakd_calibration.yaml '
            '-p parent_frame:=base_link -p child_frame:=global_camera_link'
        ),
        'signature': 'oakd_tf_broadcaster',
    },
    'realsense_tf_broadcaster': {
        'label': 'RealSense TF broadcaster',
        'category': 'cameras',
        'hardware_affecting': False,
        'cmd': (
            f'{ROS_ENV_CMD} ros2 run thesis_robot camera_tf_broadcaster --ros-args '
            '-r __node:=realsense_tf_broadcaster '
            '-p calibration_file:=~/.ros/realsense_calibration.yaml '
            '-p parent_frame:=base_link -p child_frame:=global_camera_color_optical_frame'
        ),
        'signature': 'realsense_tf_broadcaster',
    },
    'realsense2_tf_broadcaster': {
        'label': 'RealSense #2 TF broadcaster',
        'category': 'cameras',
        'hardware_affecting': False,
        # NOT in FULL_BRINGUP_ORDER: starting this before
        # ~/.ros/realsense2_calibration.yaml exists will make the node
        # refuse to start (see static_tf_broadcaster.py's fallback fix) --
        # calibrate this camera via the Calibration tab first.
        'cmd': (
            f'{ROS_ENV_CMD} ros2 run thesis_robot camera_tf_broadcaster --ros-args '
            '-r __node:=realsense2_tf_broadcaster '
            '-p calibration_file:=~/.ros/realsense2_calibration.yaml '
            '-p parent_frame:=base_link -p child_frame:=global_camera_2_color_optical_frame'
        ),
        'signature': 'realsense2_tf_broadcaster',
    },
    'camera_watchdog': {
        'label': 'Camera watchdog',
        'category': 'cameras',
        'hardware_affecting': False,
        'cmd': f'{ROS_ENV_CMD} ros2 run thesis_robot camera_watchdog',
        'signature': 'thesis_robot camera_watchdog',
    },
    'object_detection': {
        'label': 'Object detection (OAK-D)',
        'category': 'perception',
        'hardware_affecting': False,
        'cmd': f'{ROS_ENV_CMD} ros2 run thesis_robot object_detection',
        'signature': 'thesis_robot/lib/thesis_robot/object_detection',
    },
    'realsense_detection': {
        'label': 'RealSense detection',
        'category': 'perception',
        'hardware_affecting': False,
        'cmd': f'{ROS_ENV_CMD} ros2 run thesis_robot realsense_detection',
        'signature': 'thesis_robot/lib/thesis_robot/realsense_detection',
    },
    'realsense2_detection': {
        'label': 'RealSense #2 detection',
        'category': 'perception',
        'hardware_affecting': False,
        # NOT in FULL_BRINGUP_ORDER yet -- see realsense2_tf_broadcaster.
        'cmd': f'{ROS_ENV_CMD} ros2 run thesis_robot realsense2_detection',
        # Careful: must not also match 'realsense_detection' as a substring
        # (process_manager.py's duplicate-guard greps ps aux for this).
        'signature': 'thesis_robot/lib/thesis_robot/realsense2_detection',
    },
    'wrist_detection': {
        'label': 'Wrist camera detection',
        'category': 'perception',
        'hardware_affecting': False,
        'cmd': f'{ROS_ENV_CMD} ros2 run thesis_robot wrist_detection',
        'signature': 'thesis_robot/lib/thesis_robot/wrist_detection',
    },
    'obstacle_guard': {
        'label': 'Unknown-obstacle guard (depth)',
        'category': 'perception',
        'hardware_affecting': False,
        # run as a module so it works without a rebuild; publishes /unknown_obstacles (the arm only uses it when
        # arm_controller's use_unknown_obstacles is on)
        'cmd': f'{ROS_ENV_CMD} python3 -m thesis_robot.obstacle_guard_node',
        'signature': 'thesis_robot.obstacle_guard_node',
    },
    'scene_graph_node': {
        'label': 'Scene graph (fusion)',
        'category': 'perception',
        'hardware_affecting': False,
        'cmd': f'{ROS_ENV_CMD} ros2 run thesis_robot scene_graph_node',
        'signature': 'thesis_robot/lib/thesis_robot/scene_graph_node',
    },
    'llm_planner_node': {
        'label': 'LLM planner',
        'category': 'planning',
        'hardware_affecting': False,
        'cmd': f'{ROS_ENV_CMD} ros2 run thesis_robot llm_planner_node{ros_args("llm_planner")}',
        'signature': 'thesis_robot/lib/thesis_robot/llm_planner_node',
    },
    'audio_node': {
        'label': 'Voice input (microphone + Whisper)',
        'category': 'audio',
        'hardware_affecting': False,
        # dashboard mode: idle until the mic button / hands-free toggle is used;
        # transcripts are shown for confirmation, never auto-sent to the arm.
        'cmd': f'{ROS_ENV_CMD} ros2 run thesis_robot audio_node --ros-args -p mode:=dashboard',
        'signature': 'thesis_robot/lib/thesis_robot/audio_node',
    },
    'arm_controller': {
        'label': 'Arm controller',
        'category': 'control',
        'hardware_affecting': True,
        # dry_run always true on start regardless of how it's later toggled
        # in the UI -- see routers/nodes.py's go-live endpoint.
        'cmd': f'{ROS_ENV_CMD} ros2 run thesis_robot arm_controller{ros_args("arm_controller", "-p dry_run:=true")}',
        'signature': 'thesis_robot/lib/thesis_robot/arm_controller',
    },
}

# The order "Full Bring-Up" starts things in, with a short settle delay
# (seconds) after each before moving on. Robot/cameras bringup are left
# out deliberately -- those need a human confirming robot_ip/E-stop/etc,
# see the dashboard's bring-up checklist copy.
# A camera listed in config/disabled_cameras.txt (the OAK-D, parked because its USB link is marginal) must not have its
# driver, TF broadcaster or detector started: they only burn CPU and USB for a stream nobody uses (the OAK detection
# node alone took ~45% of a core while the watchdog showed the camera as disabled).
PROCS_OF_CAMERA = {
    'oakd': ('oakd_driver', 'oakd_tf_broadcaster', 'object_detection'),
}


def disabled_camera_procs():
    """Process ids that belong to cameras listed in config/disabled_cameras.txt."""
    path = os.path.join(REPO_ROOT, 'config', 'disabled_cameras.txt')
    try:
        with open(path) as f:
            names = {ln.strip() for ln in f if ln.strip() and not ln.strip().startswith('#')}
    except OSError:
        return set()
    out = set()
    for n in names:
        out.update(PROCS_OF_CAMERA.get(n, ()))
    return out


FULL_BRINGUP_ORDER = [
    ('oakd_tf_broadcaster', 1.0),
    ('realsense_tf_broadcaster', 1.0),
    ('camera_watchdog', 1.0),
    ('object_detection', 2.0),
    ('realsense_detection', 2.0),
    ('wrist_detection', 2.0),
    ('scene_graph_node', 1.0),
    ('obstacle_guard', 1.0),
    ('llm_planner_node', 1.0),
    ('audio_node', 3.0),
    ('arm_controller', 1.0),
]
