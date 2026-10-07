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
from sensor_msgs.msg import CameraInfo, JointState
from std_msgs.msg import String

from thesis_robot import motion_utils as mu
from thesis_robot import outcome_log as ol
from thesis_robot import safety_geometry as sg
from thesis_robot import trajectory_smoothing as ts
from thesis_robot import wrist_servo as ws

JOINT_NAMES = [
    "joint_1", "joint_2", "joint_3", "joint_4",
    "joint_5", "joint_6", "joint_7"
]
# Joint limits of the Gen3 (config/joint_limits.yaml): rad/s and rad/s^2.
JOINT_VMAX = [1.3963, 1.3963, 1.3963, 1.3963, 1.2218, 1.2218, 1.2218]
JOINT_AMAX = [8.6] * 7
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
CARRY_CEILING_Z = 0.50       # highest flange height a carry is raised to (the same ceiling the hover uses)
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
        # wrist-camera centering over the target before the descent
        self.declare_parameter('center_tol',      0.010)  # m, stop correcting inside this
        self.declare_parameter('center_max_step', 0.06)   # m, largest single correction
        self.declare_parameter('center_accept_m', 0.02)   # m, accepted after >= 3 corrections if the detection jitters
        self.declare_parameter('center_max_iter', 5)      # corrections before giving up
        self.declare_parameter('hover_above_m',   0.12)   # fingertips this far above the object centre while centering
        # How far the FINGERTIPS may come down to the table during a pick's straight descent (and a place
        # set-down) of a LOW object (<= 6 cm, e.g. a mouse). Default 3 cm: a parameter that defaulted to the old
        # 5 cm and had to be set after every restart made the mouse pick fail 2026-10-05 (the fingers closed
        # 1.5 cm above it). Tall objects and normal moves keep the 5 cm floor. The path check (pad frames
        # >= 5 cm above the table, i.e. tips >= 1.5 cm) applies in every case.
        self.declare_parameter('low_pick_tip_clearance_m', 0.03)
        # Neighbour checks (settable at run time with `ros2 param set`, no restart needed)
        self.declare_parameter('sweep_same_object_m', 0.06)   # a "neighbour" this close to the target is the target itself
        self.declare_parameter('sweep_half_span_m', ws.FINGER_HALF_SPAN_M)
        self.declare_parameter('sweep_margin_m', ws.SWEEP_MARGIN_M)
        self.declare_parameter('carry_avoid_m', 0.12)
        self.declare_parameter('require_wrist_center', True)   # pick refuses to descend if the wrist cannot see the object
        # try several tool yaws and keep the IK solution that moves the joints least
        self.declare_parameter('yaw_flex', True)
        self.declare_parameter('outcome_log', True)   # one JSON line per pick/place/command in ~/.ros/outcomes
        # Re-time every verified path along a minimum-jerk curve (zero speed AND acceleration at both ends, no
        # acceleration steps) instead of the planner's trapezoid. Settable at run time for A/B tests.
        self.declare_parameter('smooth_trajectories', True)
        self.declare_parameter('smooth_vel_scale', 0.30)    # peak joint speed as a fraction of the joint limit
        self.declare_parameter('smooth_acc_scale', 0.20)    # peak joint acceleration as a fraction of the limit
        self.declare_parameter('smooth_min_duration_s', 0.6)

        self.dry_run   = self.get_parameter('dry_run').value
        self.speed     = self.get_parameter('speed').value
        self._attempt = None
        self._outcomes = ol.OutcomeLog()
        self.grasp_z_offset     = self.get_parameter('grasp_z_offset').value
        self.pregrasp_clearance = self.get_parameter('pregrasp_clearance').value
        self.yaw_flex  = self.get_parameter('yaw_flex').value
        self.center_tol = self.get_parameter('center_tol').value
        self.center_accept_m = float(self.get_parameter('center_accept_m').value)
        self.center_max_step = self.get_parameter('center_max_step').value
        self.center_max_iter = self.get_parameter('center_max_iter').value
        self.hover_above_m = self.get_parameter('hover_above_m').value
        self.require_wrist_center = self.get_parameter('require_wrist_center').value
        self._held = None          # {'label', 'object_id', 'grasp_z'} while carrying something
        self._wrist_dets = None    # (monotonic receive time, {frame_id, stamp, detections})
        self._wrist_K = None
        self._tool_quat = list(GRASP_QUAT_XYZW)   # orientation of the last pose move; cartesian moves keep it
        self._js = None                            # latest /joint_states {name: position}
        self._js_time = 0.0
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
        self.create_subscription(JointState, '/joint_states', self._on_joint_state, 10)
        self.create_subscription(String, '/wrist_pixel_detections', self._on_wrist_dets, 5)
        self.create_subscription(CameraInfo, '/camera/color/camera_info', self._on_wrist_info, 1)
        from moveit_msgs.srv import ApplyPlanningScene, GetPositionFK, GetPositionIK
        self._apply_scene_client = self.create_client(ApplyPlanningScene, '/apply_planning_scene')
        self._fk_client = self.create_client(GetPositionFK, '/compute_fk')
        self._ik_client = self.create_client(GetPositionIK, '/compute_ik')

        if not self.dry_run:
            self._init_timer = self.create_timer(2.0, self._delayed_moveit_init)
        else:
            self.get_logger().warn('*** DRY RUN MODE — arm will NOT move ***')

        mode = 'DRY RUN' if self.dry_run else 'LIVE (initialising...)'
        self._publish_status(f'READY [{mode}]')
        self.create_timer(2.0, self._status_heartbeat)
        self.get_logger().info(f'ArmControllerNode ready — mode={mode}')

    def _delayed_moveit_init(self):
        self.destroy_timer(self._init_timer)      # one-shot (a status heartbeat timer also exists)
        try:
            from pymoveit2 import MoveIt2
            # pymoveit2's blocking calls (plan, wait_until_executed, ...) call
            # rclpy.spin_once() on the node they were given. Doing that on THIS
            # node while the executor also spins it crashes the executor
            # ("wait set index ... out of bounds"), after which every service
            # call that waits on the executor (FK/IK/scene) hangs. A private
            # helper node that nothing else spins avoids the collision.
            self._moveit_node = rclpy.create_node('arm_controller_moveit')
            self.moveit2 = MoveIt2(
                node=self._moveit_node,
                joint_names=JOINT_NAMES,
                base_link_name=BASE_LINK,
                end_effector_name=EE_LINK,
                group_name=GROUP_NAME,
            )
            self.moveit2.max_velocity     = self.speed
            self.moveit2.max_acceleration = self.speed
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
            t_cmd = time.monotonic()

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
                    self._log_command(command, model, plan, False, i + 1, self._block_reason if self._blocked else None, t_cmd)
                    return
                time.sleep(0.3)

            self.get_logger().info('Plan executed successfully')
            self._publish_status('COMPLETE')
            self._log_command(command, model, plan, True, None, None, t_cmd)

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
            return self._move_to(step['x'], step['y'], step['z'], cartesian=bool(step.get('cartesian', False)))
        elif act == 'pick':
            approach_z = step.get('approach_z')
            return self._pick(step.get('object_id'),
                              0.15 if approach_z is None else approach_z)
        elif act == 'rotate_wrist':
            return self._rotate_wrist(float(step.get('radians', math.pi / 2)))
        elif act == 'align_for_handle':
            label = step.get('label') or (self.latest_scene.get(step.get('object_id'), {}) or {}).get('label')
            return self._align_for_handle(label, self._expected_xy(step.get('object_id')))
        elif act == 'center_over':
            label = step.get('label') or (self.latest_scene.get(step.get('object_id'), {}) or {}).get('label')
            if not label:
                self.get_logger().error('center_over: needs a label or a known object_id')
                return False
            ok, _xy = self._center_over(label, self._expected_xy(step.get('object_id')))
            return ok
        elif act == 'place':
            return self._place(step.get('x'), step.get('y'), step.get('z'), bool(step.get('here')))
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
    def _move_to(self, x, y, z, speed=None, cartesian=False, min_flange_z=None, max_flange_z=None):
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
        rx = max(sg.PLAN_X_RANGE[0], min(sg.PLAN_X_RANGE[1], rx))
        ry = max(sg.PLAN_Y_RANGE[0], min(sg.PLAN_Y_RANGE[1], ry))
        floor_z = sg.flange_floor_z(geom)
        if min_flange_z is not None:
            floor_z = min(floor_z, min_flange_z)       # only ever LOWERS the floor, for a pick's straight descent
        ceiling = 0.50 if max_flange_z is None else max(0.50, max_flange_z)   # only a pick's hover over a tall object goes higher
        rz = max(floor_z, min(ceiling, rz))

        if self.dry_run or self.moveit2 is None:
            self.get_logger().info(
                f'  [DRY RUN] move_to robot({rx:.3f},{ry:.3f},{rz:.3f})'
                f'{" cartesian" if cartesian else ""}')
            time.sleep(0.5)
            return True
        return self._guarded_move(
            geom, f'move_to ({rx:.3f},{ry:.3f},{rz:.3f})',
            position=[rx, ry, rz], cartesian=cartesian)

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
        floors = None
        for i in sg.sample_indices(len(pts)):
            try:
                positions = self._fk_link_positions(traj.joint_names, pts[i].positions)
            except Exception as e:
                return False, f'cannot verify the path ({e})'
            if floors is None:
                floors = sg.start_floors(positions, geom)   # from the first waypoint: may leave a low pose, not go lower
            ok, why = sg.check_link_positions(positions, geom, floors)
            if not ok:
                return False, f'{why} (waypoint {i + 1}/{len(pts)})'
        return True, 'ok'

    # ── IK / planning ────────────────────────────────────────────────
    def _on_joint_state(self, msg):
        self._js = dict(zip(msg.name, msg.position))
        self._js_time = time.time()

    def _current_joint_vector(self):
        """Arm joint angles from OUR OWN /joint_states subscription (the
        executor keeps it fresh). pymoveit2's cached copy only updates while
        one of its blocking calls is spinning, so it can be arbitrarily old.
        Refuses stale data -- also catches a dead arm connection."""
        if self._js is None or time.time() - self._js_time > 1.0:
            raise RuntimeError('joint states are missing or stale — is the arm connected?')
        try:
            return [self._js[j] for j in JOINT_NAMES]
        except KeyError as e:
            raise RuntimeError(f'joint state is missing {e}')

    def _ik(self, seed, position, quat, timeout_s=0.15):
        """One IK solve (collision-aware, seeded from `seed`) via /compute_ik;
        polls the future for the same reason as _fk_link_positions."""
        from moveit_msgs.srv import GetPositionIK
        req = GetPositionIK.Request()
        r = req.ik_request
        r.group_name = GROUP_NAME
        r.ik_link_name = EE_LINK
        r.avoid_collisions = True
        r.robot_state.joint_state.name = list(JOINT_NAMES)
        r.robot_state.joint_state.position = [float(v) for v in seed]
        r.pose_stamped.header.frame_id = BASE_LINK
        p = r.pose_stamped.pose
        p.position.x, p.position.y, p.position.z = (float(v) for v in position)
        p.orientation.x, p.orientation.y, p.orientation.z, p.orientation.w = (float(v) for v in quat)
        r.timeout.sec = 0
        r.timeout.nanosec = int(timeout_s * 1e9)
        if not self._ik_client.wait_for_service(timeout_sec=3.0):
            raise RuntimeError('/compute_ik is not available')
        fut = self._ik_client.call_async(req)
        t0 = time.time()
        while not fut.done() and time.time() - t0 < timeout_s + 3.0:
            time.sleep(0.01)
        res = fut.result() if fut.done() else None
        if res is None or res.error_code.val != 1:
            return None
        sol = dict(zip(res.solution.joint_state.name, res.solution.joint_state.position))
        return [sol[j] for j in JOINT_NAMES]

    def _solve_goal_joints(self, position, quat, allow_yaw):
        """The IK solution (and tool orientation) that moves the joints the
        least from where the arm is now -- see motion_utils. The IK solver
        restarts randomly, so a single sweep occasionally misses the good
        solution; if the best one found is still too big, sweep again with a
        longer per-solve timeout before refusing."""
        seed = self._current_joint_vector()
        best = None
        for timeout_s in (0.15, 0.4):
            for yaw in (mu.yaw_candidates_deg() if allow_yaw else [0]):
                q = mu.rotate_about_tool_z(quat, yaw)
                for _ in range(2):
                    sol = self._ik(seed, position, q, timeout_s)
                    if sol is None:
                        continue
                    sol = mu.unwrap_to_seed(sol, seed)
                    if sol is None:
                        continue
                    d = mu.joint_deltas(sol, seed)
                    cand = (max(d), sum(d), sol, q)
                    if mu.better(cand, best):
                        best = cand
                if best is not None and best[0] <= mu.GOOD_ENOUGH_DELTA_RAD:
                    break
            if best is not None and best[0] <= mu.MAX_JOINT_DELTA_RAD:
                break
        if best is None:
            raise RuntimeError('no collision-free IK solution for that pose')
        if best[0] > mu.MAX_JOINT_DELTA_RAD:
            raise RuntimeError(
                f'that pose needs a {best[0]:.1f} rad single-joint reconfiguration '
                f'(limit {mu.MAX_JOINT_DELTA_RAD}) — go home first or pick a closer target')
        self.get_logger().info(
            f'IK: max joint change {best[0]:.2f} rad, total {best[1]:.2f} rad')
        return best[2], best[3]

    def _plan_joint_goal(self, goal, geom):
        """Pilz PTP first (joint-space, repeatable); the collision-aware OMPL
        planner only as a fallback. Every candidate path must pass the FK
        check before it is allowed to execute. Returns (trajectory, reason)."""
        attempts = [('pilz_industrial_motion_planner', 'PTP')] + [('', '')] * 3
        reason = 'planning failed'
        for pipeline, planner in attempts:
            self.moveit2.pipeline_id = pipeline
            self.moveit2.planner_id = planner
            traj = self.moveit2.plan(
                joint_positions=goal, joint_names=JOINT_NAMES,
                start_joint_state=self._current_joint_vector())
            if traj is None:
                continue
            ok, why = self._check_trajectory(traj, geom)
            if ok:
                return traj, 'ok'
            reason = why
            self.get_logger().warn(
                f'{pipeline or "OMPL"} path rejected by the safety check: {why}')
        return None, reason

    def _guarded_move(self, geom, label, position=None, joint_positions=None, cartesian=False):
        """Plan -> verify the whole path -> only then execute. Fails closed:
        no recorded table, planner/IK failure, FK failure or any waypoint
        outside the limits means the arm does not move."""
        self._blocked = False
        if geom is None:
            return self._refuse('no table geometry recorded — record the table first')
        try:
            self._ensure_safety_scene(geom)
        except Exception as e:
            return self._refuse(f'planning scene not ready: {e}')
        try:
            quat_used = None
            if joint_positions is not None:
                traj, why = self._plan_joint_goal(list(joint_positions), geom)
                quat_used = list(GRASP_QUAT_XYZW)
            elif cartesian:
                # keep the orientation the tool ACTUALLY has: a remembered one goes stale after a
                # wrist rotation and the 'straight' move would turn the wrist back (it did: 2026-10-05)
                traj = self.moveit2.plan(
                    position=position, quat_xyzw=self._actual_tool_quat(), cartesian=True,
                    start_joint_state=self._current_joint_vector())
                why = 'cartesian planning failed'
                if traj is not None:
                    ok, why = self._check_trajectory(traj, geom)
                    if not ok:
                        traj = None
                        return self._refuse(f'{label} unsafe: {why}')
            else:
                goal, quat_used = self._solve_goal_joints(
                    position, GRASP_QUAT_XYZW, self.yaw_flex)
                traj, why = self._plan_joint_goal(goal, geom)
            if traj is None:
                if 'safety' in why or 'below the table' in why or 'behind' in why \
                        or 'cannot verify' in why:
                    return self._refuse(f'{label} unsafe: {why}')
                self.get_logger().error(f'{label}: {why}')
                return False
            self.moveit2.execute(self._smoothed(traj, label))
            if self.moveit2.wait_until_executed() is False:
                self.get_logger().error(f'{label} not executed')
                return False
            time.sleep(0.3)
            self._tool_quat = self._actual_tool_quat(default=quat_used)
            return True
        except Exception as e:
            self.get_logger().error(f'{label} failed: {e}')
            return False

    # ── OUTCOME LOG (never allowed to affect a move) ─────────────────
    def _log_note(self, **kw):
        try:
            if self._attempt is not None:
                self._attempt.note(**kw)
        except Exception:
            pass

    def _log_event(self, name, **kw):
        try:
            if self._attempt is not None:
                self._attempt.event(name, **kw)
        except Exception:
            pass

    def _log_begin(self, kind, **fields):
        try:
            if not self.get_parameter('outcome_log').value:
                self._attempt = None
                return
            self._attempt = ol.Attempt(kind, live=not (self.dry_run or self.moveit2 is None), **fields)
        except Exception:
            self._attempt = None

    def _log_end(self, ok, failed_step=None, reason=None):
        try:
            att, self._attempt = self._attempt, None
            if att is not None:
                self._outcomes.append(att.finish(ok, failed_step, reason or (self._block_reason if self._blocked else None)))
        except Exception as e:
            self.get_logger().warn(f'outcome log: {e}')

    def _log_command(self, command, model, plan, ok, failed_step, reason, t0):
        try:
            if not self.get_parameter('outcome_log').value:
                return
            self._outcomes.append({
                'kind': 'command', 'live': not (self.dry_run or self.moveit2 is None), 'command': command,
                'model': model, 'plan': plan, 'ok': bool(ok), 'failed_step': failed_step, 'reason': reason,
                'seconds': round(time.monotonic() - t0, 1)})
        except Exception as e:
            self.get_logger().warn(f'outcome log: {e}')

    def _smoothed(self, traj, label):
        """The verified path `traj` re-timed along a minimum-jerk curve (see trajectory_smoothing), or `traj`
        itself if smoothing is off or fails: a smoothing problem must never stop a safe move."""
        if not self.get_parameter('smooth_trajectories').value:
            return traj
        try:
            from builtin_interfaces.msg import Duration
            from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
            idx = [list(traj.joint_names).index(j) for j in JOINT_NAMES]
            pos = [[p.positions[i] for i in idx] for p in traj.points]
            last = traj.points[-1].time_from_start
            old_t = last.sec + last.nanosec * 1e-9
            r = ts.smooth(pos, JOINT_VMAX, JOINT_AMAX,
                          vel_scale=float(self.get_parameter('smooth_vel_scale').value),
                          acc_scale=float(self.get_parameter('smooth_acc_scale').value),
                          min_duration=float(self.get_parameter('smooth_min_duration_s').value))
            if r is None:
                return traj
            out = JointTrajectory()
            out.header = traj.header
            out.joint_names = list(JOINT_NAMES)
            for k in range(len(r['t'])):
                pt = JointTrajectoryPoint()
                pt.positions = [float(v) for v in r['q'][k]]
                pt.velocities = [float(v) for v in r['qd'][k]]
                pt.accelerations = [float(v) for v in r['qdd'][k]]
                t = float(r['t'][k])
                pt.time_from_start = Duration(sec=int(t), nanosec=int((t % 1.0) * 1e9))
                out.points.append(pt)
            self.get_logger().info(
                f'{label}: smoothed {len(traj.points)} points / {old_t:.1f}s -> {len(out.points)} points / '
                f'{r["duration"]:.1f}s ({r["path"]})')
            return out
        except Exception as e:
            self.get_logger().warn(f'{label}: smoothing failed ({e}), using the planner timing')
            return traj

    def _actual_tool_quat(self, default=None):
        """[x, y, z, w] orientation of end_effector_link in base_link right now (TF);
        falls back to `default`, then to the last remembered orientation."""
        try:
            tf = self.tf_buffer.lookup_transform(BASE_LINK, EE_LINK, rclpy.time.Time())
            q = tf.transform.rotation
            return [q.x, q.y, q.z, q.w]
        except Exception as e:
            self.get_logger().warn(f'could not read the tool orientation from TF ({e})')
            return list(default) if default is not None else list(self._tool_quat)

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
        """Open, hover above the object, centre the gripper over it using the wrist camera, check the open
        fingers will not hit a neighbour, descend straight down, close, lift a little and check, lift. Any
        failed step fails the pick; if the wrist cannot see the object the arm does NOT descend. A grip that
        closes on air or loses the object on the lift is retried once (the object is looked up again)."""
        for attempt in (1, 2):
            pos = self._lookup_object(object_id)
            if pos is None:
                return False
            self._pick_attempt = attempt
            ok, failed = self._pick_once(object_id, pos, approach_z)
            if ok:
                return True
            if failed in ('check the grip', 'check it is still held', 'check the grip (3 cm up)') and attempt == 1:
                self.get_logger().warn(f'pick {object_id}: the grip failed ({failed}) — trying once more')
                self._publish_pp_status(f'pick {object_id}: retrying')
                time.sleep(1.0)
                continue
            return False
        return False

    def _pick_once(self, object_id, pos, approach_z):
        if not (self.dry_run or self.moveit2 is None) and self._is_holding():
            self._refuse('pick: the gripper is already holding something — put it down first '
                         '("put it down"); opening it now would drop it from the air')
            return False, 'already holding'
        try:
            ox, oy, oz = (float(v) for v in pos)
        except (TypeError, ValueError) as e:
            self.get_logger().error(f'pick: bad position for {object_id}: {e}')
            return False, 'position'
        geom, _ = sg.load_geometry()
        label = (self.latest_scene.get(object_id) or {}).get('label', 'object')
        # z is the FLANGE height: the fingertips hang TCP_REACH_M below it. Grasp with the
        # fingertips at the object's centre height, but never below the table floor.
        tip_clear = float(self.get_parameter('low_pick_tip_clearance_m').value)
        grasp_z, hover_z, lift_z, low_floor = ws.pick_heights(
            oz, label, geom['table_top_z'], sg.TCP_REACH_M, sg.flange_floor_z(geom), tip_clear,
            sg.TCP_CLEARANCE_M, approach_z=float(approach_z), grasp_offset=self.grasp_z_offset,
            hover_above=self.hover_above_m, pregrasp_clearance=self.pregrasp_clearance)
        self.get_logger().info(
            f'pick {object_id} ({label}) at ({ox:.3f},{oy:.3f},{oz:.3f}) — '
            f'hover z={hover_z:.3f}, grasp z={grasp_z:.3f}, lift z={lift_z:.3f}')

        centred = {'xy': (ox, oy)}
        tip_over_table = grasp_z - sg.TCP_REACH_M - geom['table_top_z']
        try:
            self._log_begin(
                'pick', object_id=object_id, label=label, xy=[round(ox, 3), round(oy, 3)], z=round(oz, 3),
                attempt=getattr(self, '_pick_attempt', 1), strategy='full_open',
                neighbours=ol.neighbours(self._scene_obstacles(exclude_id=object_id), (ox, oy)),
                grasp_z=round(grasp_z, 3), hover_z=round(hover_z, 3),
                scene_conf=round(float((self.latest_scene.get(object_id) or {}).get('confidence', 0) or 0), 2),
                tip_clearance_m=tip_clear)
        except Exception:
            pass

        def center():
            ok, xy = self._center_over(label, (ox, oy))
            if ok:
                centred['xy'] = xy
                return True
            if not self.require_wrist_center:
                self.get_logger().warn('pick: wrist centering failed, continuing on the scene position')
                return True
            return False

        def fingers_clear():
            """The open fingers span ~19 cm along the closing axis. If a neighbour is in that sweep, first try
            narrowing the opening to the object's width; then turn the wrist 90 degrees (and centre again); if it
            still is blocked, do not descend."""
            if self.dry_run or self.moveit2 is None:
                return True
            obstacles = self._scene_obstacles(exclude_id=object_id, min_height=max(0.0, tip_over_table))
            same = float(self.get_parameter('sweep_same_object_m').value)
            margin = float(self.get_parameter('sweep_margin_m').value)
            full_span = float(self.get_parameter('sweep_half_span_m').value)
            narrow = ws.preshape_for(label)
            for turned in (False, True):
                axis = self._closing_axis_xy()
                mid = self._pad_midpoint_xy(default=centred['xy'])

                def blocker_for(span):
                    return ws.finger_sweep_blocker(mid, axis, obstacles, same_object_m=same,
                                                   half_span=span, margin=margin) if axis else None
                blocker = blocker_for(full_span)
                if blocker is None:
                    return True
                if narrow is not None and blocker_for(min(full_span, narrow[1])) is None:
                    self.get_logger().info(
                        f'pick: narrowing the fingers to the {label} (finger position {narrow[0]:.2f}) '
                        f'so they clear the {blocker}')
                    self._log_note(strategy='narrowed' if not turned else 'turned_90+narrowed', blocker=blocker)
                    return self._gripper(narrow[0], 'narrow the fingers')
                if turned:
                    return self._refuse(f'pick: the open fingers would hit the {blocker} next to the {label}')
                if label in ws.LONG_AXIS_UNRELIABLE:
                    return self._refuse(
                        f'pick: the {blocker} is too close to grip the {label} across its short side, and gripping '
                        f'it along its length slips off (it is low and tapered) - move the {label} or the {blocker} '
                        f'about 5 cm apart')
                self.get_logger().info(f'pick: the open fingers would sweep the {blocker} — turning the wrist 90 degrees')
                self._log_note(strategy='turned_90', blocker=blocker)
                try:
                    q7 = self._current_joint_vector()[6]
                except RuntimeError as e:
                    return self._refuse(str(e))
                target = ws.quarter_turn_target(q7, mu.PLANNER_JOINT_LIMIT - 0.1)
                if target is None or not self._rotate_wrist(target - q7) or not center():
                    return False
            return False

        def descend():
            return self._move_to(centred['xy'][0], centred['xy'][1], grasp_z, cartesian=True, min_flange_z=low_floor)

        steps = [
            ('open gripper', lambda: self._open_gripper()),
            ('raise',        lambda: self._raise_to(hover_z)),
            ('hover',        lambda: self._move_to(ox, oy, hover_z, max_flange_z=ws.HOVER_MAX_Z)),
            ('turn the fingers away from any handle', lambda: self._align_for_handle(label, (ox, oy))),
            ('centre on the object (wrist camera)', center),
            ('check the open fingers are clear of its neighbours', fingers_clear),
            ('descend',      descend),
            ('close gripper', lambda: self._close_gripper()),
            ('check the grip', lambda: self._check_grip(object_id)),
            ('lift 3 cm',    lambda: self._move_to(centred['xy'][0], centred['xy'][1], grasp_z + 0.03, cartesian=True)),
            ('check the grip (3 cm up)', lambda: self._check_grip(object_id)),
            ('lift',         lambda: self._move_to(centred['xy'][0], centred['xy'][1], lift_z, cartesian=True,
                                                    max_flange_z=ws.HOVER_MAX_Z)),
            ('check it is still held', lambda: self._check_grip(object_id)),
        ]
        for name, action in steps:
            self._publish_pp_status(f'pick {object_id}: {name}')
            t_step = time.monotonic()
            ok_step = action()
            try:
                if self._attempt is not None:
                    self._attempt.step_time(name, time.monotonic() - t_step)
            except Exception:
                pass
            if not ok_step:
                self.get_logger().error(f'pick {object_id} failed at: {name}')
                self._log_end(False, name)
                return False, name
        self._held = {'label': label, 'object_id': object_id, 'grasp_z': grasp_z}
        self._log_end(True)
        return True, None

    def _closing_axis_xy(self):
        """Unit vector (x, y) along which the fingers close, from the two pad frames, or None."""
        try:
            _, a = self._tf_pose(BASE_LINK, 'left_inner_finger_pad')
            _, b = self._tf_pose(BASE_LINK, 'right_inner_finger_pad')
        except Exception:
            return None
        dx, dy = b[0] - a[0], b[1] - a[1]
        n = math.hypot(dx, dy)
        return None if n < 1e-6 else (dx / n, dy / n)

    # ── WRIST-CAMERA CENTERING ───────────────────────────────────────
    def _on_wrist_dets(self, msg):
        try:
            self._wrist_dets = (time.monotonic(), json.loads(msg.data))
        except json.JSONDecodeError:
            pass

    def _on_wrist_info(self, msg):
        if self._wrist_K is None:
            self._wrist_K = [list(msg.k[0:3]), list(msg.k[3:6]), list(msg.k[6:9])]

    def _expected_xy(self, object_id):
        obj = self.latest_scene.get(object_id) if object_id else None
        if obj and obj.get('x') is not None:
            return (float(obj['x']), float(obj['y']))
        return None

    def _tf_pose(self, target, source, stamp_s=None):
        """(R, t) of `source` in `target`, at stamp_s if the buffer has it, else the latest."""
        from scipy.spatial.transform import Rotation
        when = rclpy.time.Time()
        if stamp_s:
            when = rclpy.time.Time(seconds=int(stamp_s), nanoseconds=int((stamp_s % 1) * 1e9))
        try:
            tf = self.tf_buffer.lookup_transform(target, source, when)
        except Exception:
            tf = self.tf_buffer.lookup_transform(target, source, rclpy.time.Time())
        q, tr = tf.transform.rotation, tf.transform.translation
        return (Rotation.from_quat([q.x, q.y, q.z, q.w]).as_matrix(),
                [tr.x, tr.y, tr.z])

    def _wrist_xy_candidates(self, msg, R_bc, t_bc, plane_z):
        out = []
        for d in msg.get('detections', []):
            xy = ws.pixel_to_plane_xy(d['u'], d['v'], self._wrist_K, R_bc, t_bc, plane_z)
            if xy is not None:
                out.append((d, xy))
        return out

    def _wrist_target_xy(self, label, expected_xy, after, plane_z, timeout=3.0):
        """Where the wrist camera places `label` on the table plane: (x, y) in
        base_link, or None if it is not seen within `timeout`. Only frames received
        after `after` (monotonic) count, so we never act on a picture taken while moving."""
        t0 = time.monotonic()
        while time.monotonic() - t0 < timeout:
            item = self._wrist_dets
            if item and item[0] > after and self._wrist_K is not None:
                msg = item[1]
                try:
                    R_bc, t_bc = self._tf_pose(BASE_LINK, msg['frame_id'], msg.get('stamp'))
                except Exception as e:
                    self.get_logger().warn(f'wrist TF not available: {e}')
                    time.sleep(0.1)
                    continue
                got = ws.match_detection(self._wrist_xy_candidates(msg, R_bc, t_bc, plane_z), label, expected_xy)
                if got is not None:
                    d, xy = got
                    if d.get('label') != label:
                        self.get_logger().info(
                            f'wrist: the detector calls the {label} a "{d.get("label")}" from above '
                            f'({d["confidence"]:.0%}); it is at the expected spot, so using it')
                    return xy, d['confidence']
                after = max(after, item[0])           # nothing usable in this frame; wait for a newer one
            time.sleep(0.05)
        return None

    def _rotate_wrist(self, delta):
        """Turn joint 7 by `delta` radians (tool axis stays vertical, the fingers and the
        wrist camera turn together). A joint-space move, so it goes through the same
        plan -> FK check -> execute path as every other move."""
        geom, _ = sg.load_geometry()
        if self.dry_run or self.moveit2 is None:
            self.get_logger().info(f'  [DRY RUN] rotate wrist {math.degrees(delta):+.0f} deg')
            time.sleep(0.3)
            return True
        try:
            goal = list(self._current_joint_vector())
        except RuntimeError as e:
            return self._refuse(str(e))
        goal[6] += delta
        if abs(goal[6]) > mu.PLANNER_JOINT_LIMIT:
            return self._refuse(f'rotating the wrist by {math.degrees(delta):+.0f} deg would exceed the joint-7 limit')
        return self._guarded_move(geom, f'rotate wrist {math.degrees(delta):+.0f} deg', joint_positions=goal)

    def _wrist_match(self, label, expected_xy, timeout=3.0, after=None):
        """The fresh wrist detection (dict, with its pixel box and, when the detector has a segmentation
        model, 'orient_deg'/'elong') that is the `label` object: same matching as the centring (class or,
        failing that, position). Only frames received after `after` (monotonic, default: now) count."""
        geom, _ = sg.load_geometry()
        plane_z = ws.center_plane_height(label, geom['table_top_z']) if geom else None
        t0 = time.monotonic()
        after = t0 if after is None else after
        while time.monotonic() - t0 < timeout:
            item = self._wrist_dets
            if item and item[0] > after and self._wrist_K is not None and plane_z is not None:
                msg = item[1]
                try:
                    R_bc, t_bc = self._tf_pose(BASE_LINK, msg['frame_id'], msg.get('stamp'))
                except Exception:
                    time.sleep(0.1)
                    continue
                got = ws.match_detection(self._wrist_xy_candidates(msg, R_bc, t_bc, plane_z), label, expected_xy)
                if got is not None:
                    return got[0]
            time.sleep(0.05)
        return None

    def _wrist_bbox(self, label, expected_xy, timeout=3.0):
        d = self._wrist_match(label, expected_xy, timeout)
        return None if d is None else (d['x1'], d['y1'], d['x2'], d['y2'])

    def _align_for_handle(self, label, expected_xy):
        """Turn the wrist so the object's SHORT side lies between the fingers (a mug's handle, a mouse's length,
        a phone). With the segmentation model the object's orientation is measured, so any angle works (a mouse
        at 45 deg has a square box); otherwise the old rule is used: a box stretched along the closing axis means
        turn 90 degrees. The turn is checked by measuring again; if the object got worse, the direction was
        wrong and it is turned the other way (the direction is remembered)."""
        if self.dry_run or self.moveit2 is None:
            self.get_logger().info(f'  [DRY RUN] align the {label} with the fingers (wrist camera)')
            return True
        time.sleep(0.8)
        det = self._wrist_match(label, expected_xy)
        if det is None:
            self.get_logger().error(f'align: the wrist camera cannot see the {label}')
            return False
        if ws.OBJECT_HEIGHT_M.get(label, ws.DEFAULT_HEIGHT_M) >= ws.TALL_OBJECT_M:
            # A tall round object (a bottle) seen from above, off-axis, smears into an elongated silhouette that
            # points away from the image centre: its "axis" is perspective, not shape. Turning to it made the wrist
            # flip three times (2026-10-06). The fingers straddle a round body at any angle.
            self.get_logger().info(f'align: the {label} is tall and round — no wrist turn')
            return True
        if 'elong' not in det:
            return self._align_quarter_turn(label, (det['x1'], det['y1'], det['x2'], det['y2']))
        sign = getattr(self, '_yaw_sign', 1.0)
        for attempt in range(2):
            delta = ws.rotation_to_align(det.get('orient_deg'), det.get('elong'))
            if delta is None:
                self.get_logger().info(
                    f'align: {label} axis {det["orient_deg"]:+.0f} deg, elongation {det["elong"]:.2f} — fingers already across its short side')
                return True
            try:
                q7 = self._current_joint_vector()[6]
            except RuntimeError as e:
                return self._refuse(str(e))
            dq = sign * math.radians(delta)
            self._log_event('align', axis_deg=round(float(det['orient_deg']), 0), elong=round(float(det['elong']), 2),
                            turn_deg=round(delta, 0))
            lim = mu.PLANNER_JOINT_LIMIT - 0.1
            if abs(q7 + dq) > lim:                       # a parallel gripper is symmetric: 180 degrees is the same grasp
                alt = dq - math.copysign(math.pi, dq)
                if abs(q7 + alt) > lim:
                    self.get_logger().warn('align: no wrist turn fits inside the joint-7 limit — grasping as is')
                    return True
                dq = alt
            self.get_logger().info(
                f'align: {label} axis {det["orient_deg"]:+.0f} deg, elongation {det["elong"]:.2f} — '
                f'turning the wrist {math.degrees(dq):+.0f} deg to put its short side between the fingers')
            if not self._rotate_wrist(dq):
                return False
            time.sleep(0.8)
            new = self._wrist_match(label, expected_xy)
            if new is None or 'elong' not in new:
                return True                              # cannot re-measure; keep what we did
            after_delta = ws.rotation_to_align(new.get('orient_deg'), new.get('elong'))
            if after_delta is None or abs(after_delta) < 0.5 * abs(delta):
                return True
            self.get_logger().warn(f'align: the turn did not help ({delta:+.0f} -> {after_delta:+.0f} deg) — direction flipped')
            sign = -sign
            self._yaw_sign = sign
            det = new
        return True

    def _align_quarter_turn(self, label, box):
        """Fallback without orientation data: a box stretched along the fingers' closing axis -> turn 90 deg."""
        if not ws.needs_quarter_turn(box):
            self.get_logger().info(f'align: {label} box {box[2]-box[0]}x{box[3]-box[1]} px — fingers already clear of any handle')
            return True
        try:
            q7 = self._current_joint_vector()[6]
        except RuntimeError as e:
            return self._refuse(str(e))
        target = ws.quarter_turn_target(q7, mu.PLANNER_JOINT_LIMIT - 0.1)
        if target is None:
            return self._refuse('cannot turn the wrist 90 degrees inside the joint-7 limit')
        self.get_logger().info(
            f'align: {label} box {box[2]-box[0]}x{box[3]-box[1]} px is stretched along the closing axis '
            f'(handle) — turning the wrist {math.degrees(target - q7):+.0f} deg')
        return self._rotate_wrist(target - q7)

    def _raise_to(self, z):
        """Straight up at the current x/y to flange height z before moving sideways, so the open
        fingers never sweep through the height of a tall object on the way to the hover point."""
        if self.dry_run or self.moveit2 is None:
            self.get_logger().info(f'  [DRY RUN] raise straight up to z={z:.3f}')
            time.sleep(0.3)
            return True
        try:
            _, g = self._tf_pose(BASE_LINK, EE_LINK)
        except Exception as e:
            return self._refuse(f'cannot read the arm pose ({e})')
        if g[2] >= z - 0.02:
            return True
        return self._move_to(g[0], g[1], z, cartesian=True, max_flange_z=ws.HOVER_MAX_Z)

    def _pad_midpoint_xy(self, default):
        """(x, y) midway between the two finger pads in base_link. The tool is a few degrees off
        vertical, so this differs from the flange origin by up to ~1 cm."""
        try:
            _, a = self._tf_pose(BASE_LINK, 'left_inner_finger_pad')
            _, b = self._tf_pose(BASE_LINK, 'right_inner_finger_pad')
            return ((a[0] + b[0]) / 2.0, (a[1] + b[1]) / 2.0)
        except Exception as e:
            self.get_logger().warn(f'pad poses not available ({e}) — centring on the flange origin')
            return default

    def _center_over(self, label, expected_xy):
        """Move the gripper (keeping its height) until the wrist camera says the
        `label` object is under the tool axis. Returns (ok, (x, y)). Searches in a
        small pattern if the object is not visible; gives up after a bounded number
        of corrections or when the object cannot be found."""
        geom, _ = sg.load_geometry()
        if geom is None:
            self.get_logger().error('center_over: no table geometry recorded')
            return False, None
        plane_z = ws.center_plane_height(label, geom['table_top_z'])
        if self.dry_run or self.moveit2 is None:
            self.get_logger().info(f'  [DRY RUN] centre over {label} (wrist camera, plane z={plane_z:.3f})')
            time.sleep(0.3)
            return True, expected_xy
        if self._wrist_K is None:
            self.get_logger().error('center_over: no wrist camera_info — is the wrist driver running?')
            return False, None

        corrections, searches = 0, ws.search_offsets()
        search_origin = None
        while True:
            time.sleep(0.8)                                  # let the arm and the image settle
            seen = self._wrist_target_xy(label, expected_xy, time.monotonic(), plane_z)
            try:
                _, gpos = self._tf_pose(BASE_LINK, EE_LINK)
            except Exception as e:
                self.get_logger().error(f'center_over: no end-effector pose: {e}')
                return False, None
            gx, gy, gz = gpos                                # flange origin
            cx_, cy_ = self._pad_midpoint_xy(default=(gx, gy))  # where the fingers actually close
            if seen is None:
                if search_origin is None:
                    search_origin = (gx, gy)
                if not searches:
                    self.get_logger().error(f'center_over: the wrist camera cannot find the {label}')
                    return False, None
                ox, oy = searches.pop(0)
                self.get_logger().info(f'center_over: {label} not visible — searching ({ox:+.2f},{oy:+.2f})')
                self._publish_pp_status(f'searching for the {label}')
                if not self._move_to(search_origin[0] + ox, search_origin[1] + oy, gz, cartesian=True):
                    return False, None
                continue
            (tx, ty), conf = seen
            dx, dy, dist = ws.clipped_step((tx, ty), (cx_, cy_), self.center_max_step)
            self.get_logger().info(
                f'center_over: {label} ({conf:.0%}) at ({tx:.3f},{ty:.3f}); fingers at '
                f'({cx_:.3f},{cy_:.3f}); off by {dist * 100:.1f} cm')
            self._log_event('center', off_cm=round(dist * 100, 1), conf=round(float(conf), 2), corrections=corrections)
            if dist < self.center_tol:
                return True, (gx, gy)        # FLANGE xy that puts the fingers over the object (a straight descent keeps it)
            if corrections >= 3 and dist < self.center_accept_m:
                self.get_logger().info(f'center_over: {dist * 100:.1f} cm off after {corrections} corrections — close enough '
                                       f'(the fingers have 2.4+ cm of clearance each side)')
                return True, (gx, gy)
            if corrections >= self.center_max_iter:
                self.get_logger().error(f'center_over: still {dist * 100:.1f} cm off after {corrections} corrections')
                return False, None
            corrections += 1
            self._publish_pp_status(f'centring over the {label}: {dist * 100:.1f} cm off')
            if not self._move_to(gx + dx, gy + dy, gz, cartesian=True):
                return False, None

    CARRY_CLEARANCE_M = 0.12      # object bottom this far above the table while it is carried
    SET_DOWN_GAP_M = 0.004        # released this far above where it stood when it was picked
    TALL_OBJECT_M = 0.09          # objects at least this tall block a carry path (the carried object is ~12 cm up)
    CARRY_AVOID_M = 0.12          # keep the carry path this far (centre to centre) from tall objects

    def _check_grip(self, what):
        """After closing (and again after lifting): the fingers must have stopped on something.
        Fully closed means they closed on air; the gripper is opened again and the pick fails."""
        if self.dry_run or self.moveit2 is None:
            return True
        time.sleep(0.5)
        try:
            if self._attempt is not None:
                self._attempt.grip_reading(what, (self._js or {}).get('finger_joint'))
        except Exception:
            pass
        if self._is_holding():
            return True
        f = (self._js or {}).get('finger_joint')
        self.get_logger().error(
            f'pick {what}: nothing in the gripper (finger joint {f}) — opening it again and stopping')
        self._publish_pp_status(f'pick {what}: nothing in the gripper')
        self._open_gripper()
        return False

    def _is_holding(self):
        """True while the fingers are closed on something: partly closed, not fully (missed)."""
        try:
            f = self._js['finger_joint']
        except (TypeError, KeyError):
            return False
        return 0.08 < f < 0.65

    def _scene_obstacles(self, exclude_id=None, min_height=0.0, min_conf=0.65):
        """[(label, x, y)] of every real, non-stale object on the table whose top is above min_height
        (metres over the table), whether or not the arm could pick it."""
        out = []
        for oid, o in self.latest_scene.items():
            if oid == exclude_id or not isinstance(o, dict) or o.get('stale'):
                continue
            if float(o.get('confidence', 1.0)) < min_conf:
                continue                          # phantoms read 0.5-0.62; real objects on the table 0.9+
            try:
                x, y, z = float(o['x']), float(o['y']), float(o['z'])
            except (KeyError, TypeError, ValueError):
                continue
            if not (-0.05 <= x <= 0.85 and -0.95 <= y <= 0.95 and -0.03 <= z <= 0.32):
                continue
            if ws.OBJECT_HEIGHT_M.get(o.get('label'), ws.DEFAULT_HEIGHT_M) < min_height:
                continue
            out.append((o.get('label', oid), x, y))
        return out

    def _carry_path_blocked(self, start_xy, end_xy, min_height=None):
        """Label of a scene object whose top is above min_height (the underside of the carried object, less a
        3 cm margin; default TALL_OBJECT_M) that the carry path comes within CARRY_AVOID_M of (moving closer
        than it started), or None. See ws.carry_path_blocker."""
        h = self.TALL_OBJECT_M if min_height is None else min_height
        return ws.carry_path_blocker(start_xy, end_xy, self._scene_obstacles(min_height=h),
                                     avoid=float(self.get_parameter('carry_avoid_m').value))

    def _move_verified(self, x, y, z, label, xy_tol=0.015, z_tol=0.01, min_flange_z=None):
        """Straight-line move that must END where asked: a cartesian path that stops short would
        otherwise leave the following steps (descend, release) at the wrong place."""
        if not self._move_to(x, y, z, cartesian=True, min_flange_z=min_flange_z):
            return False
        time.sleep(0.3)
        _, pos = self._tf_pose(BASE_LINK, EE_LINK)
        if math.hypot(pos[0] - x, pos[1] - y) > xy_tol or abs(pos[2] - z) > z_tol:
            self.get_logger().error(
                f'place: after "{label}" the flange is at ({pos[0]:.3f},{pos[1]:.3f},{pos[2]:.3f}), '
                f'not ({x:.3f},{y:.3f},{z:.3f}) — stopping')
            return False
        return True

    def _place(self, x=None, y=None, z=None, here=False):
        """Logged wrapper around _place_impl: one outcome-log record per place."""
        held = dict(self._held) if self._held else {}
        self._log_begin('place', label=held.get('label'), object_id=held.get('object_id'),
                        requested={'x': x, 'y': y, 'z': z, 'here': bool(here)}, strategy='straight_carry')
        ok = False
        try:
            ok = self._place_impl(x, y, z, here)
            return ok
        finally:
            self._log_end(ok, None if ok else ((self._attempt.data.get('failed_in') if self._attempt else None) or 'place'), None)

    def _place_impl(self, x=None, y=None, z=None, here=False):
        """Put the held object down: carry it (object ~12 cm above the table) to the target, lower
        it to the height it stood at when it was picked, open the gripper, back straight up.
        Every move is a guarded cartesian move; any failure stops with the object still held."""
        geom, _ = sg.load_geometry()
        floor = sg.flange_floor_z(geom)
        if self.dry_run or self.moveit2 is None:
            where = 'here' if here or x is None else f'({float(x):.3f},{float(y):.3f})'
            self.get_logger().info(f'  [DRY RUN] place {where}')
            time.sleep(0.5)
            self._held = None
            return True
        if not self._is_holding():
            return self._refuse('place: the gripper is not holding anything')
        if self._held is None and z is None:
            return self._refuse('place: do not know how the object is held (no pick in this run) — '
                                'give a z, or open the gripper by hand')
        try:
            _, g = self._tf_pose(BASE_LINK, EE_LINK)
            pad = self._pad_midpoint_xy(default=(g[0], g[1]))
        except Exception as e:
            return self._refuse(f'place: cannot read the arm pose ({e})')
        off = (g[0] - pad[0], g[1] - pad[1])               # flange minus fingers: the object sits between the pads
        if here or x is None or y is None:
            tx, ty = pad
        else:
            try:
                tx, ty, _z = self._transform_to_robot_frame(x, y, 0.0)
            except ValueError as e:
                return self._refuse(f'place rejected: {e}')
        if not (sg.PLAN_X_RANGE[0] <= tx <= sg.PLAN_X_RANGE[1] and sg.PLAN_Y_RANGE[0] <= ty <= sg.PLAN_Y_RANGE[1]):
            return self._refuse(f'place: ({tx:.3f},{ty:.3f}) is outside the workspace')
        fx, fy = tx + off[0], ty + off[1]                  # flange target that puts the object on (tx, ty)
        grasp_z = self._held['grasp_z'] if self._held else float(z)
        # A low object was grasped below the normal fingertip floor (low_pick_tip_clearance_m); put it
        # down at the same height, not 2-3 cm above the table.
        tip_clear = float(self.get_parameter('low_pick_tip_clearance_m').value)
        low_floor = min(floor, geom['table_top_z'] + sg.TCP_REACH_M + tip_clear) if geom else floor
        set_z = max(low_floor, grasp_z + self.SET_DOWN_GAP_M)
        # Closed on the object the pad frames hang lower than they did when the fingers were open at the pick
        # (about 2 cm), so the pick's height can put them under the table margin (it refused by 1 mm with a mouse).
        try:
            pad_z = min(self._tf_pose(BASE_LINK, f)[1][2] for f in ('left_inner_finger_pad', 'right_inner_finger_pad'))
            set_z = max(set_z, geom['table_top_z'] + sg.LINK_TABLE_CLEARANCE_M + (g[2] - pad_z) + 0.003)
        except Exception:
            pass
        carry_z = min(0.50, max(grasp_z + self.CARRY_CLEARANCE_M, floor + 0.02))
        far = math.hypot(fx - g[0], fy - g[1]) > 0.02
        if far:
            # The carried object's underside is (carry_z - grasp_z) over the table. Obstacles lower than that clear
            # it; for a taller one raise the carry (up to the ceiling) before giving up.
            blocked = None
            for cz in sorted({carry_z, max(carry_z, min(0.44, CARRY_CEILING_Z)), CARRY_CEILING_Z}):
                if cz < carry_z:
                    continue
                blocked = self._carry_path_blocked((g[0], g[1]), (fx, fy), min_height=cz - grasp_z - 0.03)
                if not blocked:
                    if cz > carry_z + 0.005:
                        self.get_logger().info(
                            f'place: carrying higher (flange z={cz:.3f}) so the object clears what is on the way')
                    carry_z = cz
                    break
            if blocked:
                return self._refuse(f'place: the carry path passes within {self.CARRY_AVOID_M * 100:.0f} cm of a {blocked}')
        self.get_logger().info(
            f'place: object to ({tx:.3f},{ty:.3f}); flange carry z={carry_z:.3f}, set-down z={set_z:.3f}')
        try:
            self._log_note(target=[round(tx, 3), round(ty, 3)], carry_z=round(carry_z, 3), set_z=round(set_z, 3),
                           carried_far=bool(far),
                           neighbours=ol.neighbours(self._scene_obstacles(), (tx, ty)))
        except Exception:
            pass
        steps = []
        if g[2] < carry_z - 0.005:
            steps.append(('lift', lambda: self._move_verified(g[0], g[1], carry_z, 'lift')))
        if far:
            steps.append(('carry', lambda: self._move_verified(fx, fy, carry_z, 'carry')))
        held_xy = pad                                       # where the carried object is now (its stale scene entry is not an obstacle)

        def place_fingers_clear():
            """When the gripper opens to release, its fingers swing ~9.5 cm to each side of the object. If a
            neighbour of the target spot is in that sweep, turn the wrist 90 degrees (the object turns with it, in
            hand) and check again; if it still is, do not lower."""
            obstacles = [o for o in self._scene_obstacles(min_height=0.0)
                         if math.hypot(o[1] - held_xy[0], o[2] - held_xy[1]) > ws.SAME_OBJECT_M
                         and math.hypot(o[1] - tx, o[2] - ty) > ws.SAME_OBJECT_M]    # (the carried object is over the target)
            for turned in (False, True):
                axis = self._closing_axis_xy()
                blocker = ws.finger_sweep_blocker(
                    (tx, ty), axis, obstacles,
                    same_object_m=float(self.get_parameter('sweep_same_object_m').value),
                    half_span=float(self.get_parameter('sweep_half_span_m').value),
                    margin=float(self.get_parameter('sweep_margin_m').value)) if axis else None
                if blocker is None:
                    return True
                if turned:
                    return self._refuse(f'place: the opening fingers would hit the {blocker} next to the target spot')
                self.get_logger().info(f'place: the opening fingers would sweep the {blocker} — turning the wrist 90 degrees')
                try:
                    q7 = self._current_joint_vector()[6]
                except RuntimeError as e:
                    return self._refuse(str(e))
                target = ws.quarter_turn_target(q7, mu.PLANNER_JOINT_LIMIT - 0.1)
                if target is None or not self._rotate_wrist(target - q7):
                    return False
            return False

        steps.append(('check the opening fingers are clear of the neighbours', place_fingers_clear))
        steps.append(('lower', lambda: self._move_verified(fx, fy, set_z, 'lower', min_flange_z=low_floor)))
        for name, action in steps:
            self._publish_pp_status(f'place: {name}')
            if not action():
                self.get_logger().error(f'place failed at: {name} — still holding the object')
                self._log_note(failed_in=name)
                return False
        self._publish_pp_status('place: release')
        if not self._open_gripper():
            return False
        self._held = None
        self._publish_pp_status('place: retreat')
        return self._move_verified(fx, fy, carry_z, 'retreat')

    def _gripper(self, position, label):
        """Command the gripper and CHECK the outcome. The old code swallowed
        every error and returned True, so a pick with a dead/absent gripper
        controller carried on "grasping" thin air. Closing on an object is
        reported as 'stalled' by the controller and counts as success."""
        if self.dry_run or self.moveit2 is None:
            self.get_logger().info(f'  [DRY RUN] {label}')
            time.sleep(0.3)
            return True
        try:
            from control_msgs.action import GripperCommand as GC
            from rclpy.action import ActionClient
            if not hasattr(self, '_gripper_client'):
                self._gripper_client = ActionClient(
                    self, GC, '/robotiq_gripper_controller/gripper_cmd')
            if not self._gripper_client.wait_for_server(timeout_sec=3.0):
                self.get_logger().error(f'{label}: gripper action server not available')
                return False
            goal = GC.Goal()
            goal.command.position = float(position)
            goal.command.max_effort = 50.0
            fut = self._gripper_client.send_goal_async(goal)
            t0 = time.time()
            while not fut.done() and time.time() - t0 < 3.0:
                time.sleep(0.02)
            handle = fut.result() if fut.done() else None
            if handle is None or not handle.accepted:
                self.get_logger().error(f'{label}: gripper goal was not accepted')
                return False
            rfut = handle.get_result_async()
            t0 = time.time()
            while not rfut.done() and time.time() - t0 < 6.0:
                time.sleep(0.02)
            if not rfut.done():
                self.get_logger().error(f'{label}: gripper did not finish within 6 s')
                return False
            res = rfut.result().result
            if res.reached_goal or res.stalled:
                return True
            self.get_logger().error(f'{label}: gripper stopped short (position {res.position:.3f})')
            return False
        except Exception as e:
            self.get_logger().error(f'{label} failed: {e}')
            return False

    def _open_gripper(self):
        ok = self._gripper(GRIPPER_OPEN, 'open_gripper')
        if ok:
            self._held = None                 # whatever it held is released
        return ok

    def _close_gripper(self):
        return self._gripper(GRIPPER_CLOSED, 'close_gripper')

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
                if not self._guarded_move(geom, 'go_home', joint_positions=HOME_JOINTS):
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
        self._last_status = text
        msg = String(); msg.data = text
        self.status_pub.publish(msg)

    def _status_heartbeat(self):
        """Re-publish the current status: a status published once at startup is lost if the
        dashboard is not listening yet, and it then keeps showing the PREVIOUS run's
        'LIVE' for a controller that is in Dry Run."""
        text = getattr(self, '_last_status', None)
        if text:
            self.status_pub.publish(String(data=text))

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
