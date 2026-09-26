import threading
import time
import utilities
import vision_aruco
from kortex_api.autogen.client_stubs.BaseClientRpc import BaseClient
from grasp_utils import compute_approach_and_grasp
from gripper_control import open_gripper, close_gripper
from config_loader import load_configs
import rclpy
from rclpy.action import ActionClient
from moveit_msgs.action import MoveGroup

cfg = load_configs()
approach_offset = float(cfg["task"]["grasp"]["approach_offset_m"])
grasp_offset    = float(cfg["task"]["grasp"]["grasp_offset_m"])
lift_offset     = float(cfg["task"]["grasp"]["lift_offset_m"])


def move_to_cartesian_pose(move_client, T):
    import numpy as np
    from scipy.spatial.transform import Rotation
    from moveit_msgs.msg import (MotionPlanRequest, Constraints,
                                  PositionConstraint, OrientationConstraint,
                                  PlanningOptions)
    from geometry_msgs.msg import PoseStamped
    from shape_msgs.msg import SolidPrimitive

    pos  = T[:3, 3]
    quat = Rotation.from_matrix(T[:3, :3]).as_quat()

    ps = PoseStamped()
    ps.header.frame_id = "base_link"
    ps.pose.position.x    = float(pos[0])
    ps.pose.position.y    = float(pos[1])
    ps.pose.position.z    = float(pos[2])
    ps.pose.orientation.x = float(quat[0])
    ps.pose.orientation.y = float(quat[1])
    ps.pose.orientation.z = float(quat[2])
    ps.pose.orientation.w = float(quat[3])

    pc = PositionConstraint()
    pc.header.frame_id = "base_link"
    pc.link_name = "end_effector_link"
    box = SolidPrimitive()
    box.type = SolidPrimitive.BOX
    box.dimensions = [0.01, 0.01, 0.01]
    pc.constraint_region.primitives.append(box)
    pc.constraint_region.primitive_poses.append(ps.pose)
    pc.weight = 1.0

    oc = OrientationConstraint()
    oc.header.frame_id = "base_link"
    oc.link_name = "end_effector_link"
    oc.orientation = ps.pose.orientation
    oc.absolute_x_axis_tolerance = 0.15
    oc.absolute_y_axis_tolerance = 0.15
    oc.absolute_z_axis_tolerance = 0.15
    oc.weight = 1.0

    c = Constraints()
    c.position_constraints.append(pc)
    c.orientation_constraints.append(oc)

    req = MotionPlanRequest()
    req.group_name = "manipulator"
    req.num_planning_attempts = 5
    req.allowed_planning_time = 10.0
    req.max_velocity_scaling_factor = 0.2
    req.max_acceleration_scaling_factor = 0.1
    req.goal_constraints.append(c)

    opts = PlanningOptions()
    opts.plan_only   = False
    opts.replan      = True
    opts.replan_attempts = 3

    goal = MoveGroup.Goal()
    goal.request         = req
    goal.planning_options = opts

    print("[INFO] Sending MoveGroup goal...")
    future = move_client.send_goal_async(goal)
    end = time.time() + 15.0
    while not future.done() and time.time() < end:
        time.sleep(0.05)
    gh = future.result()
    if gh is None or not gh.accepted:
        print("[ERROR] Goal rejected")
        return False

    res_future = gh.get_result_async()
    end = time.time() + 30.0
    while not res_future.done() and time.time() < end:
        time.sleep(0.05)
    result = res_future.result()
    success = result.result.error_code.val == 1
    print(f"[INFO] Move {'succeeded' if success else 'FAILED'} (code={result.result.error_code.val})")
    return success


def main():
    args = utilities.parseConnectionArguments()

    with utilities.DeviceConnection.createTcpConnection(args) as router:
        base = BaseClient(router)

        # Start vision (inits rclpy inside)
        vision_thread = threading.Thread(target=vision_aruco.vision_loop, daemon=True)
        vision_thread.start()

        # Wait for rclpy to be running inside vision_loop
        time.sleep(2.0)

        # Create a persistent node + spin it for MoveGroup discovery
        move_node = rclpy.create_node("pick_move_client")
        move_client = ActionClient(move_node, MoveGroup, "/move_group")
        executor = rclpy.executors.SingleThreadedExecutor()
        executor.add_node(move_node)
        exec_thread = threading.Thread(target=executor.spin, daemon=True)
        exec_thread.start()

        print("[INFO] Waiting for MoveGroup action server...")
        move_client.wait_for_server()
        print("[INFO] MoveGroup ready. Press SPACE in vision window to pick.")

        while not vision_aruco.stop_event.is_set():
            vision_aruco.pose_ready_event.wait(timeout=0.1)
            if vision_aruco.stop_event.is_set():
                break
            if vision_aruco.pose_ready_event.is_set():
                T_base_O = vision_aruco.latest_T_base_O.copy()
                vision_aruco.pose_ready_event.clear()

                T_base_A, _, T_base_L = compute_approach_and_grasp(
                    T_base_O,
                    approach_offset=approach_offset,
                    grasp_offset=grasp_offset,
                    lift_offset=lift_offset)

                open_gripper(base)
                time.sleep(1.0)

                if not move_to_cartesian_pose(move_client, T_base_A):
                    print("[ERROR] Failed to reach approach pose")
                    continue

                close_gripper(base)
                time.sleep(1.0)

                if not move_to_cartesian_pose(move_client, T_base_L):
                    print("[ERROR] Failed to lift")
                    continue

                print("[INFO] Pick complete!")

        vision_aruco.stop_event.set()
        vision_thread.join()

if __name__ == "__main__":
    main()
