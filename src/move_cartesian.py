import subprocess
import time
import numpy as np
from scipy.spatial.transform import Rotation


def move_to_cartesian_pose(base, T):
    pos  = T[:3, 3]
    quat = Rotation.from_matrix(T[:3, :3]).as_quat()

    goal = (
        "{request: {group_name: manipulator, "
        "num_planning_attempts: 5, allowed_planning_time: 10.0, "
        "max_velocity_scaling_factor: 0.2, "
        "max_acceleration_scaling_factor: 0.1, "
        "goal_constraints: [{position_constraints: [{header: {frame_id: base_link}, "
        "link_name: end_effector_link, "
        "constraint_region: {primitives: [{type: 1, dimensions: [0.01,0.01,0.01]}], "
        f"primitive_poses: [{{position: {{x: {pos[0]:.4f}, y: {pos[1]:.4f}, z: {pos[2]:.4f}}}, "
        f"orientation: {{x: {quat[0]:.4f}, y: {quat[1]:.4f}, z: {quat[2]:.4f}, w: {quat[3]:.4f}}}}}]}}, "
        "weight: 1.0}]}]}, "
        "planning_options: {plan_only: false, replan: true, replan_attempts: 3}}"
    )

    cmd = f"ros2 action send_goal /move_group moveit_msgs/action/MoveGroup "{goal}""
    print(f"[INFO] Executing Cartesian waypoint")
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=45)
    print(result.stdout[-300:] if result.stdout else "")
    if result.returncode == 0 and "SUCCEEDED" in result.stdout:
        print("[INFO] Move succeeded")
        return True
    print(f"[ERROR] Move failed: {result.stderr[-200:]}")
    return False
