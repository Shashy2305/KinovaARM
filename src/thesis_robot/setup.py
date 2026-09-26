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
            # Senior nodes
            'yolo_detector        = thesis_robot.yolo_detector:main',
            'object_detection     = thesis_robot.object_detection:main',
            'depth_3d_node        = thesis_robot.depth_3d_node:main',
            'pick_and_place       = thesis_robot.pick_and_place:main',
            'joint_state_remapper = thesis_robot.joint_state_remapper:main',
            'grasp_detector       = thesis_robot.grasp_detector:main',
            'bottle_segmentation  = thesis_robot.bottle_segmentation:main',
            'bottle_filter        = thesis_robot.bottle_filter:main',
            'scene_graph_node     = thesis_robot.scene_graph_node:main',
            'fusion_node         = thesis_robot.fusion_node:main',
            'realsense_detection = thesis_robot.realsense_detection:main',
            'llm_planner_node     = thesis_robot.llm_planner_node:main',
            'audio_node           = thesis_robot.audio_node:main',
            'arm_controller       = thesis_robot.arm_controller_node:main',
            'camera_tf_broadcaster = thesis_robot.static_tf_broadcaster:main',
            # Your nodes (files don't exist yet — we create them next)
            # 'scene_graph_node  = thesis_robot.scene_graph_node:main',
            # 'llm_planner_node  = thesis_robot.llm_planner_node:main',
            # 'audio_node        = thesis_robot.audio_node:main',
            # 'arm_controller    = thesis_robot.arm_controller:main',
        ],
    },
)
