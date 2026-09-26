#!/usr/bin/env python3
"""
Arm Controller Node — executes validated action plans on the Kinova Gen3.
Subscribes to:  /action_plan
Publishes to:   /arm_status, /pick_place_status
"""
import rclpy, json, math, time, threading
import tf2_ros
import tf2_geometry_msgs
from geometry_msgs.msg import PointStamped
from rclpy.node import Node
from std_msgs.msg import String

JOINT_NAMES = [
    "joint_1", "joint_2", "joint_3", "joint_4",
    "joint_5", "joint_6", "joint_7"
]
BASE_LINK   = "base_link"
EE_LINK     = "end_effector_link"
GROUP_NAME  = "manipulator"
HOME_JOINTS = [0.0, -0.35, 3.14, -2.27, 0.0, 0.96, 1.57]
GRIPPER_OPEN   = 0.14
GRIPPER_CLOSED = 0.02
CAMERA_FRAME   = "global_camera_link"


class ArmControllerNode(Node):

    def __init__(self):
        super().__init__('arm_controller')

        self.declare_parameter('dry_run', True)
        self.declare_parameter('speed',   0.20)

        self.dry_run   = self.get_parameter('dry_run').value
        self.speed     = self.get_parameter('speed').value
        self.executing = False
        self._lock     = threading.Lock()
        self.moveit2   = None

        # TF buffer for camera -> robot frame transform
        self.tf_buffer   = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        self.status_pub    = self.create_publisher(String, '/arm_status',        10)
        self.pp_status_pub = self.create_publisher(String, '/pick_place_status', 10)
        self.create_subscription(String, '/action_plan', self.plan_callback, 10)

        if not self.dry_run:
            self.create_timer(2.0, self._delayed_moveit_init)
        else:
            self.get_logger().warn('*** DRY RUN MODE — arm will NOT move ***')

        mode = 'DRY RUN' if self.dry_run else 'LIVE (initialising...)'
        self._publish_status(f'READY [{mode}]')
        self.get_logger().info(f'ArmControllerNode ready — mode={mode}')

    def _delayed_moveit_init(self):
        self.destroy_timer(list(self._timers)[0])
        try:
            from pymoveit2 import MoveIt2
            self.moveit2 = MoveIt2(
                node=self,
                joint_names=JOINT_NAMES,
                base_link_name=BASE_LINK,
                end_effector_name=EE_LINK,
                group_name=GROUP_NAME,
            )
            self.moveit2.max_velocity     = self.speed
            self.moveit2.max_acceleration = self.speed
            self.moveit2.planner_id       = "PTP"
            self.get_logger().info('MoveIt2 initialised successfully — LIVE mode active')
            self._publish_status('READY [LIVE]')
        except Exception as e:
            self.get_logger().error(f'MoveIt2 init failed: {e}')
            self.dry_run = True
            self._publish_status('READY [DRY RUN — MoveIt2 failed]')

    def plan_callback(self, msg):
        with self._lock:
            if self.executing:
                self.get_logger().warn('Already executing — ignoring')
                return
            self.executing = True
        threading.Thread(target=self._execute, args=(msg.data,), daemon=True).start()

    def _execute(self, plan_json):
        try:
            data    = json.loads(plan_json)
            plan    = data.get('plan', [])
            command = data.get('command', 'unknown')
            model   = data.get('model', 'unknown')

            self.get_logger().info(f'Executing: "{command}" ({len(plan)} steps)')
            self._publish_status(f'EXECUTING: {command}')

            for i, step in enumerate(plan):
                act = step.get('action', '')
                # Mark steps as pre-calibrated if they use x_robot/y_robot coords
                if 'x_robot' in str(step) or data.get('calibrated', False):
                    step['_pre_calibrated'] = True
                self.get_logger().info(f'Step {i+1}/{len(plan)}: {json.dumps(step)}')
                self._publish_pp_status(f'Step {i+1}/{len(plan)}: {act}')

                success = self._dispatch(step)
                if not success:
                    self.get_logger().error(f'Step {i+1} FAILED')
                    self._publish_status(f'FAILED at step {i+1}')
                    self._safe_home()
                    return
                time.sleep(0.3)

            self.get_logger().info('Plan executed successfully')
            self._publish_status('COMPLETE')

        except Exception as e:
            self.get_logger().error(f'Execution error: {e}')
            self._publish_status(f'ERROR: {str(e)[:80]}')
        finally:
            with self._lock:
                self.executing = False
            mode = 'DRY RUN' if self.dry_run else 'LIVE'
            self._publish_status(f'READY [{mode}]')

    def _dispatch(self, step):
        act = step.get('action', '')
        if act == 'move_to':
            return self._move_to(step['x'], step['y'], step['z'])
        elif act == 'pick':
            return self._pick(step['object_id'], step.get('approach_z', 0.15))
        elif act == 'place':
            return self._place(step['x'], step['y'], step['z'])
        elif act == 'open_gripper':
            return self._open_gripper()
        elif act == 'close_gripper':
            return self._close_gripper()
        elif act == 'go_home':
            return self._safe_home()
        elif act == 'null_space_adjust':
            return self._null_space_adjust()
        else:
            self.get_logger().warn(f'Unknown action: {act}')
            return True

    # ── TF TRANSFORM ─────────────────────────────────────────────────
    def _transform_to_robot_frame(self, x, y, z):
        """
        Plan coordinates carry no frame of their own. They are base_link by
        contract: scene_graph_node is the single place where camera-frame
        detections are converted (via TF) and it only publishes base_link.
        So no transform happens here — only a check that the numbers are real.
        """
        vals = []
        for name, v in (('x', x), ('y', y), ('z', z)):
            try:
                v = float(v)
            except (TypeError, ValueError):
                raise ValueError(f'{name}={v!r} is not a number')
            if not math.isfinite(v):
                raise ValueError(f'{name}={v} is not finite')
            vals.append(v)
        self.get_logger().info(
            f'Using base_link coords: ({vals[0]:.3f},{vals[1]:.3f},{vals[2]:.3f})'
        )
        return tuple(vals)

    # ── PRIMITIVES ───────────────────────────────────────────────────
    def _move_to(self, x, y, z, speed=None):
        try:
            rx, ry, rz = self._transform_to_robot_frame(x, y, z)
        except ValueError as e:
            self.get_logger().error(f'move_to rejected: {e}')
            return False
        # Safety clamp to workspace
        rx = max(0.10, min(0.55, rx))
        ry = max(-0.35, min(0.35, ry))
        rz = max(0.08, min(0.50, rz))

        if self.dry_run or self.moveit2 is None:
            self.get_logger().info(
                f'  [DRY RUN] move_to robot({rx:.3f},{ry:.3f},{rz:.3f})')
            time.sleep(0.5)
            return True
        try:
            self.moveit2.move_to_pose(
                position=[rx, ry, rz],
                quat_xyzw=[0.0, 0.707, 0.0, 0.707],
                cartesian=False
            )
            self.moveit2.wait_until_executed()
            return True
        except Exception as e:
            self.get_logger().error(f'move_to failed: {e}')
            return False

    def _pick(self, object_id, approach_z=0.15):
        if self.dry_run or self.moveit2 is None:
            self.get_logger().info(
                f'  [DRY RUN] pick {object_id} approach_z={approach_z:.3f}')
            time.sleep(1.0)
            return True
        try:
            self._open_gripper()
            time.sleep(0.5)
            self._close_gripper()
            return True
        except Exception as e:
            self.get_logger().error(f'pick failed: {e}')
            return False

    def _place(self, x, y, z):
        try:
            rx, ry, rz = self._transform_to_robot_frame(x, y, z)
        except ValueError as e:
            self.get_logger().error(f'place rejected: {e}')
            return False
        rx = max(0.10, min(0.55, rx))
        ry = max(-0.35, min(0.35, ry))
        rz = max(0.08, min(0.50, rz))

        if self.dry_run or self.moveit2 is None:
            self.get_logger().info(
                f'  [DRY RUN] place robot({rx:.3f},{ry:.3f},{rz:.3f})')
            time.sleep(0.5)
            return True
        try:
            self.moveit2.move_to_pose(
                position=[rx, ry, rz],
                quat_xyzw=[0.0, 0.707, 0.0, 0.707],
                cartesian=False
            )
            self.moveit2.wait_until_executed()
            self._open_gripper()
            return True
        except Exception as e:
            self.get_logger().error(f'place failed: {e}')
            return False

    def _open_gripper(self):
        if self.dry_run or self.moveit2 is None:
            self.get_logger().info('  [DRY RUN] open_gripper')
            time.sleep(0.3)
            return True
        try:
            from control_msgs.action import GripperCommand as GC
            from rclpy.action import ActionClient
            if not hasattr(self, '_gripper_client'):
                self._gripper_client = ActionClient(
                    self, GC, '/robotiq_gripper_controller/gripper_cmd')
            goal = GC.Goal()
            goal.command.position   = GRIPPER_OPEN
            goal.command.max_effort = 50.0
            self._gripper_client.send_goal_async(goal)
            time.sleep(1.0)
            return True
        except Exception as e:
            self.get_logger().warn(f'open_gripper skipped: {e}')
            return True

    def _close_gripper(self):
        if self.dry_run or self.moveit2 is None:
            self.get_logger().info('  [DRY RUN] close_gripper')
            time.sleep(0.3)
            return True
        try:
            from control_msgs.action import GripperCommand as GC
            from rclpy.action import ActionClient
            if not hasattr(self, '_gripper_client'):
                self._gripper_client = ActionClient(
                    self, GC, '/robotiq_gripper_controller/gripper_cmd')
            goal = GC.Goal()
            goal.command.position   = GRIPPER_CLOSED
            goal.command.max_effort = 50.0
            self._gripper_client.send_goal_async(goal)
            time.sleep(1.0)
            return True
        except Exception as e:
            self.get_logger().warn(f'close_gripper skipped: {e}')
            return True

    def _safe_home(self):
        if self.dry_run or self.moveit2 is None:
            self.get_logger().info('  [DRY RUN] go_home')
            time.sleep(1.0)
            return True
        # Retry up to 5 times waiting for zero velocity
        for attempt in range(5):
            try:
                self.moveit2.move_to_configuration(HOME_JOINTS)
                self.moveit2.wait_until_executed()
                return True
            except Exception as e:
                err = str(e)
                if 'non-zero' in err or 'INVALID_ROBOT_STATE' in err or attempt < 4:
                    self.get_logger().warn(
                        f'go_home attempt {attempt+1}/5 failed — waiting 2s: {err[:60]}')
                    time.sleep(2.0)
                else:
                    self.get_logger().error(f'go_home failed: {e}')
                    return False
        return False

    def _null_space_adjust(self):
        self.get_logger().info('  null_space_adjust — Week 7 implementation')
        return True

    def _publish_status(self, text):
        msg = String(); msg.data = text
        self.status_pub.publish(msg)

    def _publish_pp_status(self, text):
        msg = String(); msg.data = text
        self.pp_status_pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = ArmControllerNode()
    executor = rclpy.executors.MultiThreadedExecutor(2)
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
