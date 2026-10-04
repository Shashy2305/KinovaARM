#!/usr/bin/env python3
"""
Arm Controller Node — executes validated action plans on the Kinova Gen3.
Subscribes to:  /action_plan
                /scene_snapshot  (object positions for `pick`, base_link)
Publishes to:   /arm_status, /pick_place_status
"""
import rclpy, json, math, time, threading
import tf2_ros
import tf2_geometry_msgs
from geometry_msgs.msg import PointStamped
from rclpy.node import Node
from std_msgs.msg import String

from thesis_robot import safety_geometry as sg

JOINT_NAMES = [
    "joint_1", "joint_2", "joint_3", "joint_4",
    "joint_5", "joint_6", "joint_7"
]
BASE_LINK   = "base_link"
EE_LINK     = "end_effector_link"
GROUP_NAME  = "manipulator"
HOME_JOINTS = [0.0, -0.35, 3.14, -2.27, 0.0, 0.96, 1.57]
# Verified on hardware 2026-09-30 (TESTING.md Stage 3), now that
# robotiq_gripper_controller is actually spawned (it wasn't before — see
# robot_bringup/README.md): position 0.0 is fully open, 0.695 is fully
# closed. The previous 0.14/0.02 values were backwards — 0.14 mostly
# closes the gripper and 0.02 leaves it nearly fully open, so a pick
# sequence would never actually grip anything.
GRIPPER_OPEN   = 0.0
GRIPPER_CLOSED = 0.695
CAMERA_FRAME   = "global_camera_link"
# Verified on hardware 2026-09-30 (TESTING.md Stage 4): jogged the gripper
# to point straight down, read base_link -> end_effector_link. The old
# [0, 0.707, 0, 0.707] here was a generic reference value, never confirmed
# against this specific robot. The rotation matrix from that reading
# showed the EE's Z-axis as (0.002, -0.049, -0.999) in base_link — i.e.
# genuinely straight down — so this quaternion is a verified working
# grasp orientation, not just the one TESTING.md happened to expect (any
# rotation about the vertical axis also points straight down; this is one
# such rotation, specifically the one the gripper was actually jogged to).
GRASP_QUAT_XYZW = [0.773, 0.635, -0.015, 0.019]


class ArmControllerNode(Node):

    def __init__(self):
        super().__init__('arm_controller')

        self.declare_parameter('dry_run', True)
        self.declare_parameter('speed',   0.20)
        # pick geometry (metres)
        self.declare_parameter('grasp_z_offset',     0.0)   # added to object z for the grasp height
        self.declare_parameter('pregrasp_clearance', 0.10)  # min height of pre-grasp above grasp

        self.dry_run   = self.get_parameter('dry_run').value
        self.speed     = self.get_parameter('speed').value
        self.grasp_z_offset     = self.get_parameter('grasp_z_offset').value
        self.pregrasp_clearance = self.get_parameter('pregrasp_clearance').value
        self.executing = False
        self._lock     = threading.Lock()
        self.moveit2   = None
        self.latest_scene = {}   # object_id -> {x, y, z, ...} in base_link
        self._blocked = False    # set when a move was refused/aborted on safety grounds
        self._block_reason = ''

        # TF buffer for camera -> robot frame transform
        self.tf_buffer   = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        self.status_pub    = self.create_publisher(String, '/arm_status',        10)
        self.pp_status_pub = self.create_publisher(String, '/pick_place_status', 10)
        self.create_subscription(String, '/action_plan', self.plan_callback, 10)
        self.create_subscription(String, '/scene_snapshot', self._scene_callback, 10)
        from moveit_msgs.srv import ApplyPlanningScene, GetPositionFK
        self._apply_scene_client = self.create_client(ApplyPlanningScene, '/apply_planning_scene')
        self._fk_client = self.create_client(GetPositionFK, '/compute_fk')

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
            geom, err = sg.load_geometry()
            if geom is None:
                self.get_logger().error(
                    f'NO TABLE GEOMETRY ({err}) — live motion is BLOCKED until the '
                    f'table is recorded in the dashboard.')
                self._publish_status('READY [LIVE — BLOCKED: no table geometry]')
            else:
                try:
                    self._ensure_safety_scene(geom)
                except Exception as e:
                    self.get_logger().error(f'Could not apply table/wall to the scene yet: {e}')
                self.get_logger().info(
                    f'Table geometry loaded: top z={geom["table_top_z"]:.3f}, '
                    f'flange floor z={sg.flange_floor_z(geom):.3f}')
                self._publish_status('READY [LIVE]')
        except Exception as e:
            self.get_logger().error(f'MoveIt2 init failed: {e}')
            self.dry_run = True
            self._publish_status('READY [DRY RUN — MoveIt2 failed]')

    def _scene_callback(self, msg):
        try:
            scene = json.loads(msg.data)
        except json.JSONDecodeError:
            return
        if isinstance(scene, dict):
            self.latest_scene = scene

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
                    self.get_logger().error(
                        f'Step {i+1} FAILED — stopping; the arm is NOT moved '
                        f'automatically after a failure (use go_home when it is clear)')
                    why = f': {self._block_reason}' if self._blocked else ''
                    self._publish_status(f'FAILED at step {i+1}{why}'[:100])
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
            approach_z = step.get('approach_z')
            return self._pick(step.get('object_id'),
                              0.15 if approach_z is None else approach_z)
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
    def _move_to(self, x, y, z, speed=None, cartesian=False):
        try:
            rx, ry, rz = self._transform_to_robot_frame(x, y, z)
        except ValueError as e:
            self.get_logger().error(f'move_to rejected: {e}')
            return False
        # Workspace clamp. x upper bound was widened 0.55->0.60 on 2026-09-30
        # (calibrated cup at x=0.558 came in 7mm over the old limit).
        # z is the FLANGE (end_effector_link) height, and the finger pads
        # hang ~0.21 m below it when the gripper points down, so the floor
        # is derived from the recorded table height, NOT a fixed 0.08 (which
        # put the fingertips into the table: flange z=0.12 -> tips at ~-0.09).
        geom, _ = sg.load_geometry()
        rx = max(0.10, min(0.60, rx))
        ry = max(-0.35, min(0.35, ry))
        rz = max(sg.flange_floor_z(geom), min(0.50, rz))

        if self.dry_run or self.moveit2 is None:
            self.get_logger().info(
                f'  [DRY RUN] move_to robot({rx:.3f},{ry:.3f},{rz:.3f})'
                f'{" cartesian" if cartesian else ""}')
            time.sleep(0.5)
            return True
        return self._guarded_move(
            geom, f'move_to ({rx:.3f},{ry:.3f},{rz:.3f})',
            position=[rx, ry, rz], quat_xyzw=GRASP_QUAT_XYZW, cartesian=cartesian)

    # ── SAFETY: planning scene + trajectory check ────────────────────
    def _move_group_count(self):
        return sum(1 for name, _ns in self.get_node_names_and_namespaces() if name == 'move_group')

    def _ensure_safety_scene(self, geom):
        """(Re-)apply the table slab and rear wall to MoveIt's planning scene
        through the synchronous /apply_planning_scene service (it reports
        success, unlike publishing to a topic). Done before every plan
        because move_group forgets them when robot_bringup restarts. Raises
        if it cannot be confirmed, so callers fail closed."""
        from geometry_msgs.msg import Pose
        from moveit_msgs.msg import CollisionObject, PlanningScene
        from moveit_msgs.srv import ApplyPlanningScene
        from shape_msgs.msg import SolidPrimitive

        n = self._move_group_count()
        if n != 1:
            raise RuntimeError(
                f'{n} move_group nodes are running (need exactly 1) — a scene update '
                f'would only reach one of them; restart robot_bringup cleanly')
        objs = []
        for oid, size, pos in sg.collision_boxes(geom):
            prim = SolidPrimitive()
            prim.type = SolidPrimitive.BOX
            prim.dimensions = [float(v) for v in size]
            pose = Pose()
            pose.position.x, pose.position.y, pose.position.z = (float(v) for v in pos)
            pose.orientation.w = 1.0
            co = CollisionObject()
            co.id = oid
            co.header.frame_id = BASE_LINK
            co.operation = CollisionObject.ADD
            co.primitives = [prim]
            co.primitive_poses = [pose]
            objs.append(co)
        scene = PlanningScene()
        scene.is_diff = True
        scene.world.collision_objects = objs
        if not self._apply_scene_client.wait_for_service(timeout_sec=3.0):
            raise RuntimeError('/apply_planning_scene is not available')
        fut = self._apply_scene_client.call_async(ApplyPlanningScene.Request(scene=scene))
        t0 = time.time()
        while not fut.done() and time.time() - t0 < 5.0:
            time.sleep(0.05)
        if not fut.done() or not fut.result().success:
            raise RuntimeError('move_group did not accept the table/wall collision objects')

    def _refuse(self, reason):
        self._blocked = True
        self._block_reason = reason
        self.get_logger().error(f'MOTION REFUSED: {reason}')
        self._publish_status(f'BLOCKED: {reason}'[:90])
        return False

    def _fk_link_positions(self, joint_names, joint_positions):
        """base_link xyz of every guarded link for a joint configuration,
        via MoveIt's /compute_fk. Polls the future instead of spinning: the
        node already has an executor, and pymoveit2's own spin_once calls
        from a worker thread collide with it."""
        from moveit_msgs.srv import GetPositionFK
        req = GetPositionFK.Request()
        req.header.frame_id = BASE_LINK
        req.fk_link_names = list(sg.GUARDED_LINKS)
        req.robot_state.joint_state.name = list(joint_names)
        req.robot_state.joint_state.position = [float(v) for v in joint_positions]
        if not self._fk_client.wait_for_service(timeout_sec=3.0):
            raise RuntimeError('/compute_fk is not available')
        fut = self._fk_client.call_async(req)
        t0 = time.time()
        while not fut.done() and time.time() - t0 < 5.0:
            time.sleep(0.01)
        res = fut.result() if fut.done() else None
        if res is None or res.error_code.val != 1 or len(res.pose_stamped) != len(sg.GUARDED_LINKS):
            raise RuntimeError('forward kinematics failed')
        out = {}
        for link, ps in zip(sg.GUARDED_LINKS, res.pose_stamped):
            if ps.header.frame_id not in (BASE_LINK, 'world', ''):
                raise RuntimeError(f'FK returned frame {ps.header.frame_id!r}, expected {BASE_LINK}')
            out[link] = (ps.pose.position.x, ps.pose.position.y, ps.pose.position.z)
        return out

    def _check_trajectory(self, traj, geom):
        """FK every (sampled) waypoint and make sure the gripper/wrist links
        stay above the table and in front of the rear limit."""
        pts = traj.points
        if not pts:
            return False, 'planner returned an empty trajectory'
        for i in sg.sample_indices(len(pts)):
            try:
                positions = self._fk_link_positions(traj.joint_names, pts[i].positions)
            except Exception as e:
                return False, f'cannot verify the path ({e})'
            ok, why = sg.check_link_positions(positions, geom)
            if not ok:
                return False, f'{why} (waypoint {i + 1}/{len(pts)})'
        return True, 'ok'

    def _guarded_move(self, geom, label, **plan_kwargs):
        """Plan -> verify the whole path -> only then execute. Fails closed:
        no recorded table, planner failure, FK failure or any waypoint
        outside the limits means the arm does not move."""
        self._blocked = False
        if geom is None:
            return self._refuse('no table geometry recorded — record the table first')
        try:
            self._ensure_safety_scene(geom)
        except Exception as e:
            return self._refuse(f'planning scene not ready: {e}')
        try:
            traj = self.moveit2.plan(**plan_kwargs)
            if traj is None:
                self.get_logger().error(f'{label}: planning failed')
                return False
            ok, why = self._check_trajectory(traj, geom)
            if not ok:
                return self._refuse(f'{label} unsafe: {why}')
            self.moveit2.execute(traj)
            if self.moveit2.wait_until_executed() is False:
                self.get_logger().error(f'{label} not executed')
                return False
            return True
        except Exception as e:
            self.get_logger().error(f'{label} failed: {e}')
            return False

    def _lookup_object(self, object_id):
        """Latest base_link position of object_id from /scene_snapshot,
        as (x, y, z), or None if it cannot be grasped."""
        obj = self.latest_scene.get(object_id)
        if obj is None:
            self.get_logger().error(
                f'pick: {object_id!r} not in scene '
                f'(known: {sorted(self.latest_scene)})')
            return None
        if obj.get('z') is None:
            self.get_logger().error(f'pick: {object_id} has unknown height')
            return None
        if obj.get('stale'):
            self.get_logger().warn(
                f'pick: {object_id} is stale — using last seen position')
        return obj.get('x'), obj.get('y'), obj.get('z')

    def _pick(self, object_id, approach_z=0.15):
        """Open, move above the object, descend straight down, close,
        lift straight back up. Any failed step fails the pick."""
        pos = self._lookup_object(object_id)
        if pos is None:
            return False
        try:
            ox, oy, oz = (float(v) for v in pos)
            grasp_z = oz + self.grasp_z_offset
            pre_z   = max(float(approach_z), grasp_z + self.pregrasp_clearance)
        except (TypeError, ValueError) as e:
            self.get_logger().error(f'pick: bad position for {object_id}: {e}')
            return False

        self.get_logger().info(
            f'pick {object_id} at ({ox:.3f},{oy:.3f},{oz:.3f}) — '
            f'pre-grasp z={pre_z:.3f}, grasp z={grasp_z:.3f}')
        steps = [
            ('open gripper', lambda: self._open_gripper()),
            ('pre-grasp',    lambda: self._move_to(ox, oy, pre_z)),
            ('descend',      lambda: self._move_to(ox, oy, grasp_z, cartesian=True)),
            ('close gripper', lambda: self._close_gripper()),
            ('lift',         lambda: self._move_to(ox, oy, pre_z, cartesian=True)),
        ]
        for name, action in steps:
            self._publish_pp_status(f'pick {object_id}: {name}')
            if not action():
                self.get_logger().error(f'pick {object_id} failed at: {name}')
                return False
        return True

    def _place(self, x, y, z):
        try:
            rx, ry, rz = self._transform_to_robot_frame(x, y, z)
        except ValueError as e:
            self.get_logger().error(f'place rejected: {e}')
            return False
        geom, _ = sg.load_geometry()
        rx = max(0.10, min(0.60, rx))  # see _move_to's clamp
        ry = max(-0.35, min(0.35, ry))
        rz = max(sg.flange_floor_z(geom), min(0.50, rz))

        if self.dry_run or self.moveit2 is None:
            self.get_logger().info(
                f'  [DRY RUN] place robot({rx:.3f},{ry:.3f},{rz:.3f})')
            time.sleep(0.5)
            return True
        if not self._guarded_move(
                geom, f'place ({rx:.3f},{ry:.3f},{rz:.3f})',
                position=[rx, ry, rz], quat_xyzw=GRASP_QUAT_XYZW, cartesian=False):
            return False   # don't release the object somewhere else
        self._open_gripper()
        return True

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
        # Retry up to 5 times waiting for zero velocity -- but never retry
        # a move that was refused on safety grounds.
        geom, _ = sg.load_geometry()
        for attempt in range(5):
            try:
                if not self._guarded_move(
                        geom, 'go_home', joint_positions=HOME_JOINTS, joint_names=JOINT_NAMES):
                    if self._blocked:
                        return False
                    raise RuntimeError('home trajectory not executed')
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
