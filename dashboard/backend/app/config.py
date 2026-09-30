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

REPO_ROOT = os.path.expanduser('~/Shashproject')
WORKSPACE_ROOT = os.path.expanduser('~/workspace/ros2_kortex_ws')
LOG_DIR = os.path.expanduser('~/.ros/dashboard_logs')
ROBOT_IP = os.environ.get('KINOVA_ROBOT_IP', '192.168.1.10')

ROS_ENV_CMD = (
    'source /opt/ros/humble/setup.bash && '
    f'source {WORKSPACE_ROOT}/install/setup.bash && '
    f'source {REPO_ROOT}/install/setup.bash && '
    'export ROS_DOMAIN_ID=42 && '
)

CAMERA_TOPICS = {
    'oakd':      {'image': '/global_camera/color/image_raw'},
    'realsense': {'image': '/global_camera/global_camera/color/image_raw'},
    'wrist':     {'image': '/camera/color/image_raw'},
}

# category groups the Node Control grid in the UI; hardware_affecting drives
# the "are you sure" styling (not a hard block) for anything that moves the
# arm or talks to a physical device.
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
    'wrist_detection': {
        'label': 'Wrist camera detection',
        'category': 'perception',
        'hardware_affecting': False,
        'cmd': f'{ROS_ENV_CMD} ros2 run thesis_robot wrist_detection',
        'signature': 'thesis_robot/lib/thesis_robot/wrist_detection',
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
        'cmd': f'{ROS_ENV_CMD} ros2 run thesis_robot llm_planner_node',
        'signature': 'thesis_robot/lib/thesis_robot/llm_planner_node',
    },
    'arm_controller': {
        'label': 'Arm controller',
        'category': 'control',
        'hardware_affecting': True,
        # dry_run always true on start regardless of how it's later toggled
        # in the UI -- see routers/nodes.py's go-live endpoint.
        'cmd': f'{ROS_ENV_CMD} ros2 run thesis_robot arm_controller --ros-args -p dry_run:=true',
        'signature': 'thesis_robot/lib/thesis_robot/arm_controller',
    },
}

# The order "Full Bring-Up" starts things in, with a short settle delay
# (seconds) after each before moving on. Robot/cameras bringup are left
# out deliberately -- those need a human confirming robot_ip/E-stop/etc,
# see the dashboard's bring-up checklist copy.
FULL_BRINGUP_ORDER = [
    ('oakd_tf_broadcaster', 1.0),
    ('realsense_tf_broadcaster', 1.0),
    ('camera_watchdog', 1.0),
    ('object_detection', 2.0),
    ('realsense_detection', 2.0),
    ('wrist_detection', 2.0),
    ('scene_graph_node', 1.0),
    ('llm_planner_node', 1.0),
    ('arm_controller', 1.0),
]
