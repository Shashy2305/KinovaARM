# Copyright (c) 2023 PickNik, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

import os

from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch.actions import (
    ExecuteProcess,
    DeclareLaunchArgument,
    OpaqueFunction,
    RegisterEventHandler,
)
from launch.event_handlers import OnProcessExit
from launch.conditions import IfCondition, UnlessCondition
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
from moveit_configs_utils import MoveItConfigsBuilder


def launch_setup(context, *args, **kwargs):
    robot_ip                    = LaunchConfiguration("robot_ip")
    use_fake_hardware           = LaunchConfiguration("use_fake_hardware")
    gripper_max_velocity        = LaunchConfiguration("gripper_max_velocity")
    gripper_max_force           = LaunchConfiguration("gripper_max_force")
    launch_rviz                 = LaunchConfiguration("launch_rviz")
    use_sim_time                = LaunchConfiguration("use_sim_time")
    use_internal_bus_gripper_comm = LaunchConfiguration("use_internal_bus_gripper_comm")

    launch_arguments = {
        "robot_ip": robot_ip,
        "use_fake_hardware": use_fake_hardware,
        "gripper": "robotiq_2f_140",
        "vision": "false",
        "gripper_joint_name": "finger_joint",
        "dof": "7",
        "gripper_max_velocity": gripper_max_velocity,
        "gripper_max_force": gripper_max_force,
        "use_internal_bus_gripper_comm": use_internal_bus_gripper_comm,
    }

    moveit_config = (
        MoveItConfigsBuilder(
            "gen3",
            package_name="kinova_gen3_7dof_robotiq_2f_140_moveit_config",
        )
        .robot_description(mappings=launch_arguments)
        .trajectory_execution(file_path="config/moveit_controllers.yaml")
        .planning_scene_monitor(
            publish_robot_description=True,
            publish_robot_description_semantic=True,
        )
        .sensors_3d(file_path="config/sensors_3d.yaml")
        .planning_pipelines(pipelines=["pilz_industrial_motion_planner"])
        .to_moveit_configs()
    )

    moveit_config.moveit_cpp.update(
        {"use_sim_time": use_sim_time.perform(context) == "true"}
    )

    ros2_controllers_path = os.path.join(
        get_package_share_directory("kinova_gen3_7dof_robotiq_2f_140_moveit_config"),
        "config", "ros2_controllers.yaml",
    )

    ros2_control_node = Node(
        package="controller_manager",
        executable="ros2_control_node",
        parameters=[ros2_controllers_path],
        remappings=[("/controller_manager/robot_description", "/robot_description")],
        output="both",
    )

    robot_state_publisher = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        name="robot_state_publisher",
        output="both",
        parameters=[moveit_config.robot_description],
    )

    joint_state_broadcaster_spawner = Node(
        package="controller_manager",
        executable="spawner",
        arguments=["joint_state_broadcaster", "--controller-manager", "/controller_manager"],
    )

    robot_traj_controller_spawner = Node(
        package="controller_manager",
        executable="spawner",
        arguments=["joint_trajectory_controller", "-c", "/controller_manager"],
    )

    robot_pos_controller_spawner = Node(
        package="controller_manager",
        executable="spawner",
        arguments=["twist_controller", "--inactive", "-c", "/controller_manager"],
    )

    robot_hand_controller_spawner = Node(
        package="controller_manager",
        executable="spawner",
        arguments=["robotiq_gripper_controller", "-c", "/controller_manager"],
    )

    fault_controller_spawner = Node(
        package="controller_manager",
        executable="spawner",
        arguments=["fault_controller", "-c", "/controller_manager"],
        condition=UnlessCondition(use_fake_hardware),
    )

    move_group_node = Node(
        additional_env={"AMENT_PREFIX_PATH": "/home/lab/workspace/ros2_kortex_ws/install/moveit_ros_perception:" + os.environ.get("AMENT_PREFIX_PATH", "")},
        package="moveit_ros_move_group",
        executable="move_group",
        output="screen",
        parameters=[moveit_config.to_dict()],
    )

    static_tf = Node(
        package="tf2_ros",
        executable="static_transform_publisher",
        name="static_transform_publisher",
        output="log",
        arguments=["--frame-id", "world", "--child-frame-id", "base_link"],
    )

    calibration_tf = Node(
        package="tf2_ros",
        executable="static_transform_publisher",
        name="calibration_tf_publisher",
        output="log",
        arguments=[
            "--x",  "0.03986",
            "--y",  "0.35130",
            "--z",  "0.24388",
            "--qx", "0.20970",
            "--qy", "-0.16902",
            "--qz", "0.01846",
            "--qw", "0.96287",
            "--frame-id",       "base_link",
            "--child-frame-id", "global_camera_link",
        ],
    )

    optical_tf = Node(
        package="tf2_ros",
        executable="static_transform_publisher",
        name="optical_tf_publisher",
        output="log",
        arguments=[
            "--x", "0", "--y", "0", "--z", "0",
            "--roll", "-1.5708", "--pitch", "0", "--yaw", "-1.5708",
            "--frame-id",       "global_camera_link",
            "--child-frame-id", "global_camera_optical_link",
        ],
    )

    rviz_config_file = (
        get_package_share_directory("kinova_gen3_7dof_robotiq_2f_140_moveit_config")
        + "/config/moveit.rviz"
    )

    rviz_node = Node(
        package="rviz2",
        condition=IfCondition(launch_rviz),
        executable="rviz2",
        name="rviz2_moveit",
        output="log",
        arguments=["-d", rviz_config_file],
        parameters=[
            moveit_config.robot_description,
            moveit_config.robot_description_semantic,
            moveit_config.robot_description_kinematics,
            moveit_config.planning_pipelines,
            moveit_config.joint_limits,
        ],
    )

    delay_rviz_after_joint_state_broadcaster_spawner = RegisterEventHandler(
        event_handler=OnProcessExit(
            target_action=joint_state_broadcaster_spawner,
            on_exit=[rviz_node],
        ),
        condition=IfCondition(launch_rviz),
    )
    nodes_to_start = [
        ros2_control_node,
        robot_state_publisher,
        joint_state_broadcaster_spawner,
        robot_traj_controller_spawner,
        robot_pos_controller_spawner,
        robot_hand_controller_spawner,
        fault_controller_spawner,
        move_group_node,
        static_tf,
        calibration_tf,
        optical_tf,
        delay_rviz_after_joint_state_broadcaster_spawner,
    ]

    return nodes_to_start


def generate_launch_description():
    declared_arguments = [
        DeclareLaunchArgument("robot_ip", description="IP address by which the robot can be reached."),
        DeclareLaunchArgument("use_fake_hardware", default_value="false"),
        DeclareLaunchArgument("gripper_max_velocity", default_value="100.0"),
        DeclareLaunchArgument("gripper_max_force", default_value="100.0"),
        DeclareLaunchArgument("use_internal_bus_gripper_comm", default_value="true"),
        DeclareLaunchArgument("use_external_cable", default_value="false"),
        DeclareLaunchArgument("use_sim_time", default_value="false"),
        DeclareLaunchArgument("launch_rviz", default_value="true"),
    ]

    return LaunchDescription(
        declared_arguments + [OpaqueFunction(function=launch_setup)]
    )
