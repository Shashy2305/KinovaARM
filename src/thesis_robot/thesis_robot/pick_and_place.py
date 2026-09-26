import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from geometry_msgs.msg import PoseArray, PoseStamped, Pose
from std_msgs.msg import String
from moveit_msgs.action import MoveGroup
from moveit_msgs.msg import (MotionPlanRequest, WorkspaceParameters,
                              Constraints, PositionConstraint,
                              OrientationConstraint, BoundingVolume)
from control_msgs.action import GripperCommand
from shape_msgs.msg import SolidPrimitive
import tf2_ros
import tf2_geometry_msgs

PLACE_POSITION = [0.42, 0.35, 0.15]
GRIPPER_OPEN   = 0.14
GRIPPER_CLOSED = 0.02

class PickAndPlace(Node):
    def __init__(self):
        super().__init__("pick_and_place")
        self.tf_buffer   = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        self.move_client    = ActionClient(self, MoveGroup, "/move_action")
        self.gripper_client = ActionClient(self, GripperCommand,
                              "/robotiq_gripper_controller/gripper_cmd")
        self.status_pub = self.create_publisher(String, "/pick_place_status", 10)
        self.grasp_sub  = self.create_subscription(
            PoseArray, "/grasp_poses", self.grasp_callback, 1)
        self.executing = False
        self.get_logger().info("Waiting for MoveGroup...")
        self.move_client.wait_for_server()
        self.get_logger().info("Pick and place ready")

    def publish_status(self, msg):
        s = String()
        s.data = msg
        self.status_pub.publish(s)
        self.get_logger().info(f"[STATUS] {msg}")

    def send_gripper(self, position):
        goal = GripperCommand.Goal()
        goal.command.position   = position
        goal.command.max_effort = 50.0
        f = self.gripper_client.send_goal_async(goal)
        rclpy.spin_until_future_complete(self, f, timeout_sec=5.0)
        rclpy.spin_until_future_complete(self, f.result().get_result_async(), timeout_sec=5.0)

    def make_pose(self, x, y, z, orientation):
        ps = PoseStamped()
        ps.header.frame_id  = "base_link"
        ps.pose.position.x  = x
        ps.pose.position.y  = y
        ps.pose.position.z  = z
        ps.pose.orientation = orientation
        return ps

    def move_to_pose(self, pose_stamped, label="pose"):
        self.publish_status(f"Planning to {label}")
        goal = MoveGroup.Goal()
        req  = MotionPlanRequest()
        req.group_name            = "manipulator"
        req.num_planning_attempts = 10
        req.allowed_planning_time = 10.0
        req.max_velocity_scaling_factor     = 0.2
        req.max_acceleration_scaling_factor = 0.2
        ws = WorkspaceParameters()
        ws.header.frame_id = "base_link"
        ws.min_corner.x = -1.0; ws.min_corner.y = -1.0; ws.min_corner.z = -0.1
        ws.max_corner.x =  1.0; ws.max_corner.y =  1.0; ws.max_corner.z =  1.5
        req.workspace_parameters = ws
        pc = PositionConstraint()
        pc.header.frame_id = "base_link"
        pc.link_name       = "end_effector_link"
        bv = BoundingVolume()
        sp = SolidPrimitive()
        sp.type = SolidPrimitive.SPHERE
        sp.dimensions = [0.01]
        bv.primitives.append(sp)
        bp = Pose()
        bp.position      = pose_stamped.pose.position
        bp.orientation.w = 1.0
        bv.primitive_poses.append(bp)
        pc.constraint_region = bv
        pc.weight = 1.0
        oc = OrientationConstraint()
        oc.header.frame_id   = "base_link"
        oc.link_name         = "end_effector_link"
        oc.orientation       = pose_stamped.pose.orientation
        oc.absolute_x_axis_tolerance = 0.3
        oc.absolute_y_axis_tolerance = 0.3
        oc.absolute_z_axis_tolerance = 0.3
        oc.weight = 1.0
        c = Constraints()
        c.position_constraints.append(pc)
        c.orientation_constraints.append(oc)
        req.goal_constraints.append(c)
        goal.request = req
        goal.planning_options.plan_only       = False
        goal.planning_options.replan          = True
        goal.planning_options.replan_attempts = 3
        f = self.move_client.send_goal_async(goal)
        rclpy.spin_until_future_complete(self, f, timeout_sec=15.0)
        gh = f.result()
        if not gh.accepted:
            self.get_logger().error(f"Goal rejected: {label}")
            return False
        rf = gh.get_result_async()
        rclpy.spin_until_future_complete(self, rf, timeout_sec=30.0)
        ec = rf.result().result.error_code.val
        if ec != 1:
            self.get_logger().error(f"Motion failed {label}: error_code={ec}")
            return False
        self.publish_status(f"Reached {label}")
        return True

    def grasp_callback(self, msg):
        if self.executing or not msg.poses:
            return
        self.executing = True
        try:
            grasp_cam        = PoseStamped()
            grasp_cam.header = msg.header
            grasp_cam.pose   = msg.poses[0]
            grasp_base = self.tf_buffer.transform(
                grasp_cam, "base_link",
                timeout=rclpy.duration.Duration(seconds=3.0))
            gx  = grasp_base.pose.position.x
            gy  = grasp_base.pose.position.y
            gz  = grasp_base.pose.position.z
            ori = grasp_base.pose.orientation
            self.publish_status(f"Grasp at base_link: ({gx:.3f},{gy:.3f},{gz:.3f})")
            self.publish_status("Opening gripper")
            self.send_gripper(GRIPPER_OPEN)
            if not self.move_to_pose(self.make_pose(gx, gy, gz+0.12, ori), "pre-grasp"):
                raise RuntimeError("pre-grasp failed")
            if not self.move_to_pose(self.make_pose(gx, gy, gz, ori), "grasp"):
                raise RuntimeError("grasp failed")
            self.publish_status("Closing gripper")
            self.send_gripper(GRIPPER_CLOSED)
            if not self.move_to_pose(self.make_pose(gx, gy, gz+0.15, ori), "lift"):
                raise RuntimeError("lift failed")
            px, py, pz = PLACE_POSITION
            if not self.move_to_pose(self.make_pose(px, py, pz+0.12, ori), "pre-place"):
                raise RuntimeError("pre-place failed")
            if not self.move_to_pose(self.make_pose(px, py, pz, ori), "place"):
                raise RuntimeError("place failed")
            self.publish_status("Opening gripper - releasing")
            self.send_gripper(GRIPPER_OPEN)
            self.move_to_pose(self.make_pose(px, py, pz+0.15, ori), "retreat")
            self.publish_status("COMPLETE")
        except Exception as e:
            self.get_logger().error(f"Failed: {e}")
            self.publish_status(f"FAILED: {e}")
        self.executing = False

def main():
    rclpy.init()
    rclpy.spin(PickAndPlace())

if __name__ == "__main__":
    main()
