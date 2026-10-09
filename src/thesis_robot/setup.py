from setuptools import find_packages, setup

package_name = 'thesis_robot'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/models', ['../../models/yolov8m.pt']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='lab',
    maintainer_email='akhiljoshi255@gmail.com',
    description='Audio-Visual Spatial Control for Kinova Gen3 7-DOF',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'object_detection     = thesis_robot.object_detection:main',
            'joint_state_remapper = thesis_robot.joint_state_remapper:main',
            'scene_graph_node     = thesis_robot.scene_graph_node:main',
            'realsense_detection = thesis_robot.realsense_detection:main',
            'realsense2_detection = thesis_robot.realsense2_detection:main',
            'wrist_detection      = thesis_robot.wrist_detection:main',
            'camera_watchdog      = thesis_robot.camera_watchdog:main',
            'llm_planner_node     = thesis_robot.llm_planner_node:main',
            'audio_node           = thesis_robot.audio_node:main',
            'arm_controller       = thesis_robot.arm_controller_node:main',
            'camera_tf_broadcaster = thesis_robot.static_tf_broadcaster:main',
            'obstacle_guard       = thesis_robot.obstacle_guard_node:main',
        ],
    },
)
