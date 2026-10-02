# Copyright (c) 2023 PickNik, Inc.
#
# Licensed under the Apache License, Version 2.0

# NETWORK TIP: Connect robot via a dedicated ethernet NIC.
# Do NOT share the robot's 192.168.1.x subnet with Wi-Fi or other devices.
# The wrist camera RTSP stream also runs over the same IP, so launch
# cameras.launch.py only AFTER robot.launch.py is fully stable
# ("You can start planning now!" appears in move_group output).

# ── FRAME REFERENCE ──────────────────────────────────────────────────────────
# world → base_link (static, identity)
# base_link → global_camera_color_optical_frame
#   x=0.99 y=-0.13 z=0.77
#   qx=0.6220 qy=0.6099 qz=-0.3475 qw=-0.3469
#   ⚠ STALE — redo easy_handeye2 calibration
# end_effector_link → pen_tip
#   xyz="0 0 -0.15452"  (154.52 mm down tool axis, EE -Z direction)
# end_effector_link → camera_color_frame
#   x=-0.0494305 y=0.049587 z=0.00395126
#   qx=0.200804 qy=0.290464 qz=0.442318 qw=0.824417
# base_link → global_camera_link (OAK-D)
#   x=0.48 y=0.72 z=1.0
#   qx=-0.341494 qy=-0.888985 qz=0.299234 qw=0.059552
# ─────────────────────────────────────────────────────────────────────────────

import os

from launch import LaunchDescription
from launch.substitutions import LaunchConfiguration
from launch.actions import (
    DeclareLaunchArgument,
    ExecuteProcess,
    OpaqueFunction,
    RegisterEventHandler,
)
from launch.event_handlers import OnProcessExit
from launch.conditions import IfCondition, UnlessCondition
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
from moveit_configs_utils import MoveItConfigsBuilder


def launch_setup(context, *args, **kwargs):
    robot_ip                      = LaunchConfiguration("robot_ip")
    use_fake_hardware             = LaunchConfiguration("use_fake_hardware")
    gripper                       = LaunchConfiguration("gripper")
    gripper_max_velocity          = LaunchConfiguration("gripper_max_velocity")
    gripper_max_force             = LaunchConfiguration("gripper_max_force")
    launch_rviz                   = LaunchConfiguration("launch_rviz")
    use_sim_time                  = LaunchConfiguration("use_sim_time")
    use_internal_bus_gripper_comm = LaunchConfiguration("use_internal_bus_gripper_comm")

    launch_arguments = {
        "robot_ip":                        robot_ip,
        "use_fake_hardware":               use_fake_hardware,
        "gripper":                         gripper,
        "vision":                          "true",
        "gripper_joint_name":              "finger_joint",
        "dof":                             "7",
        "gripper_max_velocity":            gripper_max_velocity,
        "gripper_max_force":               gripper_max_force,
        "use_internal_bus_gripper_comm":   use_internal_bus_gripper_comm,
        # Keys must match xacro:arg names in gen3.xacro (include _ms suffix).
        "session_inactivity_timeout_ms":    "120000",
        "connection_inactivity_timeout_ms": "10000",
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
        .planning_pipelines(pipelines=["ompl", "pilz_industrial_motion_planner"])
        .to_moveit_configs()
    )

    # Octomap/sensors_3d removed — perception-driven pick-and-place uses
    # YOLO+GPD grasp poses directly, not Octomap obstacle avoidance.
    moveit_config.sensors_3d = {}

    moveit_config.moveit_cpp.update(
        {"use_sim_time": use_sim_time.perform(context) == "true"}
    )

    # ── ROS2 control ─────────────────────────────────────────────────────────
    ros2_controllers_path = os.path.join(
        get_package_share_directory(
            "kinova_gen3_7dof_robotiq_2f_140_moveit_config"
        ),
        "config",
        "ros2_controllers.yaml",
    )

    ros2_control_node = Node(
        package="controller_manager",
        executable="ros2_control_node",
        parameters=[ros2_controllers_path],
        remappings=[("/controller_manager/robot_description", "/robot_description")],
        output="both",
        # NOTE: previously forced RMW_IMPLEMENTATION=rmw_cyclonedds_cpp here
        # (for lower-latency intra-process transport, to help the 1kHz
        # ros2_control loop avoid BaseCyclicClient timeouts under load).
        # Removed: this made ros2_control_node the ONLY node in the whole
        # launch using CycloneDDS while every spawner/other node uses the
        # system default (rmw_fastrtps_cpp) -- that cross-vendor DDS mix is
        # what was deadlocking controller_manager's service layer (every
        # executor thread piling on one internal rclcpp mutex, confirmed via
        # gdb; reproduced identically on two different machines/builds,
        # ruling out hardware/build causes). If BaseCyclicClient timeouts
        # reappear under load, set RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
        # globally (for every node in this launch, not just this one) rather
        # than reintroducing a mismatch.
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

    fault_controller_spawner = Node(
        package="controller_manager",
        executable="spawner",
        arguments=["fault_controller", "-c", "/controller_manager"],
        condition=UnlessCondition(use_fake_hardware),
    )

    robotiq_gripper_controller_spawner = Node(
        package="controller_manager",
        executable="spawner",
        arguments=["robotiq_gripper_controller", "-c", "/controller_manager"],
    )

    # ── MoveIt2 (no OctoMap) ─────────────────────────────────────────────────
    move_group_node = Node(
        package="moveit_ros_move_group",
        executable="move_group",
        output="screen",
        parameters=[moveit_config.to_dict()],
    )

    # ── TF: world → base_link ────────────────────────────────────────────────
    static_tf = Node(
        package="tf2_ros",
        executable="static_transform_publisher",
        name="static_transform_publisher",
        output="log",
        arguments=["--frame-id", "world", "--child-frame-id", "base_link"],
    )

    # ── RViz ─────────────────────────────────────────────────────────────────
    rviz_config_file = (
        get_package_share_directory(
            "kinova_gen3_7dof_robotiq_2f_140_moveit_config"
        )
        + "/config/moveit.rviz"
        if os.path.exists(
            get_package_share_directory(
                "kinova_gen3_7dof_robotiq_2f_140_moveit_config"
            )
            + "/config/moveit.rviz"
        )
        else "/tmp/moveit_simple.rviz"
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

    # ── Serialize controller spawners ───────────────────────────────────────
    # Launching all 5 spawners at once (as this file used to) makes 5
    # concurrent clients hit controller_manager's services during its own
    # startup -- this reliably deadlocks ros2_control_node's
    # MultiThreadedExecutor on some machines/builds (every thread piles up
    # on the executor's internal wait-set mutex, confirmed via gdb; matches
    # ros-controls/ros2_control#265 upstream). Chaining each spawner off the
    # previous one's exit keeps controller_manager's service layer to at
    # most one (or two, for the last pair) concurrent client at a time.
    delay_traj_controller_after_joint_state_broadcaster = RegisterEventHandler(
        event_handler=OnProcessExit(
            target_action=joint_state_broadcaster_spawner,
            on_exit=[robot_traj_controller_spawner],
        )
    )

    delay_pos_controller_after_traj_controller = RegisterEventHandler(
        event_handler=OnProcessExit(
            target_action=robot_traj_controller_spawner,
            on_exit=[robot_pos_controller_spawner],
        )
    )

    # robotiq_gripper_controller_spawner (unconditional) comes before
    # fault_controller_spawner (conditional, skipped under fake hardware) in
    # the chain -- if it were last, a skipped fault_controller_spawner would
    # never fire the OnProcessExit that triggers it, since that event only
    # fires for a process that actually ran. Putting the always-runs spawner
    # first, and the maybe-skipped one last (nothing depends on its exit),
    # keeps full serialization correct in both use_fake_hardware states.
    delay_gripper_controller_after_pos_controller = RegisterEventHandler(
        event_handler=OnProcessExit(
            target_action=robot_pos_controller_spawner,
            on_exit=[robotiq_gripper_controller_spawner],
        )
    )

    delay_fault_controller_after_gripper_controller = RegisterEventHandler(
        event_handler=OnProcessExit(
            target_action=robotiq_gripper_controller_spawner,
            on_exit=[fault_controller_spawner],
        )
    )

    robot_keepalive = ExecuteProcess(
        cmd=["python3", "/home/lab/workspace/ros2_kortex_ws/robot_keepalive.py"],
        output="log",
    )

    return [
        ros2_control_node,
        robot_state_publisher,
        joint_state_broadcaster_spawner,
        move_group_node,
        static_tf,
        robot_keepalive,
        delay_rviz_after_joint_state_broadcaster_spawner,
        delay_traj_controller_after_joint_state_broadcaster,
        delay_pos_controller_after_traj_controller,
        delay_gripper_controller_after_pos_controller,
        delay_fault_controller_after_gripper_controller,
    ]


def generate_launch_description():
    declared_arguments = [
        DeclareLaunchArgument(
            "robot_ip",
            description="IP address by which the robot can be reached."),
        DeclareLaunchArgument(
            "use_fake_hardware",
            default_value="false",
            description="Start robot with fake hardware mirroring command to its states."),
        DeclareLaunchArgument(
            "gripper",
            default_value="robotiq_2f_140",
            description=(
                "Gripper description package under kortex_description/grippers "
                "to attach — this robot is shared between projects using "
                "different end effectors. Pass e.g. gripper:=thesis_ee to "
                "override; don't hardcode a value here again, it silently "
                "changes what the NEXT person who launches this file gets, "
                "not just you.")),
        DeclareLaunchArgument("gripper_max_velocity", default_value="100.0"),
        DeclareLaunchArgument("gripper_max_force",    default_value="100.0"),
        DeclareLaunchArgument(
            "use_internal_bus_gripper_comm",
            default_value="true"),
        DeclareLaunchArgument("use_sim_time",  default_value="false"),
        DeclareLaunchArgument("launch_rviz",   default_value="true"),
    ]

    return LaunchDescription(
        declared_arguments + [OpaqueFunction(function=launch_setup)]
    )
