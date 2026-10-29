"""
Dashboard jog: move the arm by hand from the browser (virtual stick, hold buttons, or a gamepad plugged into the PC that shows the dashboard).

Why it goes through ROS: the Kinova allows ONE control session. Driving the arm with the Xbox pad over Kinova's own session while robot_bringup runs makes the driver fault and
freeze /joint_states (2026-10-22: eight calibration poses recorded the same arm position). The jog sends ordinary joint-trajectory points to the controller that robot_bringup already
owns, so there is no second session.

Safety, all enforced here:
  * OFF until enabled with a typed phrase; switches itself off after IDLE_DISABLE_S without commands, on any fault, or when the joint feedback stops following the commands
  * hold-to-move: every command asks for only HORIZON_S of motion, so the arm stops by itself ~0.2 s after the commands stop (browser closed, network cut, finger released)
  * speed caps, joint-limit margins, table / rear limits of safety_geometry on every step (jog_math), refused while a pick/place command is running or the driver reports a fault
The hardware E-stop stays the primary stop. The Stop button only holds position and switches the jog off.
"""
import os
import threading
import time

import numpy as np
from builtin_interfaces.msg import Duration
from rclpy.duration import Duration as RclDuration
from rclpy.node import Node
from std_msgs.msg import Bool, String
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

from thesis_robot import safety_geometry as sg

from . import jog_math as jm
from . import kinematics as kin
from .joint_feed import JointFeed

ENABLE_PHRASE = 'area clear, hands off'
IDLE_DISABLE_S = 120.0
STALE_JOINT_S = 0.5
MIN_TICK_S = 0.06                 # commands arriving faster than this are dropped (no flood of trajectories)
GAP_S = 0.4                       # a longer pause between commands ends a "continuous move"
FOLLOW_CHECK_S = 1.5
FOLLOW_MIN_FRACTION = 0.25        # the arm must have covered this share of what was commanded in the window
FOLLOW_MIN_RAD = 0.03
TOPIC = '/joint_trajectory_controller/joint_trajectory'
URDF_CACHE = os.path.expanduser('~/.ros/arm_model_cache.urdf')
URDF_RETRY_S = 15.0
GRIPPER_ACTION = '/robotiq_gripper_controller/gripper_cmd'
GRIPPER_OPEN, GRIPPER_CLOSED = 0.0, 0.695          # finger_joint values (same as arm_controller_node)
GRIPPER_MAX_EFFORT = 50.0                          # same as the pick's grip
GRIPPER_WAIT_S = 6.0


class Jogger(Node):
    def __init__(self, urdf_xml=None):
        super().__init__('dashboard_jog')
        self._lock = threading.RLock()
        self._urdf = urdf_xml
        self.k = None
        self._urdf_retry_at = 0.0
        self._urdf_error = ''
        self.enabled = False
        self.enabled_at = 0.0
        self.last_cmd_t = 0.0
        self.last_tick_t = 0.0
        self.message = 'jog is off'
        self.feed = JointFeed(self)
        self._fault = False
        self._fault_t = 0.0
        self._arm_status = ''
        self._win = None              # [t_start, q_start, expected displacement (7,)] of the current continuous move
        self._ref = None              # where the arm should be NOW (integrated command) and the velocity being commanded; None = not in a continuous move
        self._v = np.zeros(7)
        self.last_step = None
        self.pub = self.create_publisher(JointTrajectory, TOPIC, 5)
        self.create_subscription(Bool, '/fault_controller/internal_fault', self._on_fault, 5)
        self.create_subscription(String, '/arm_status', self._on_status, 10)

    # ── inputs ────────────────────────────────────────────────────────
    def _on_fault(self, msg):
        with self._lock:
            self._fault = bool(msg.data)
            self._fault_t = time.monotonic()

    def _on_status(self, msg):
        with self._lock:
            self._arm_status = msg.data

    def _kinematics(self):
        if self.k is None:
            if time.monotonic() < self._urdf_retry_at:
                raise RuntimeError(self._urdf_error)
            try:
                xml = self._urdf or self._fetch_urdf()
                self.k = kin.Kinematics(xml)
            except Exception as e:
                self._urdf_error = f'cannot load the arm model: {e}'
                self._urdf_retry_at = time.monotonic() + URDF_RETRY_S      # do not make every status poll wait for a model that is not there
                raise
        return self.k

    def _fetch_urdf(self):
        """The running model: the latched /robot_description topic first (robot_state_publisher's parameter service is not always discoverable), then its parameter, then the
        copy saved by an earlier successful run (the arm model does not change between runs)."""
        xml = self._urdf_from_topic() or self._urdf_from_param()
        if xml:
            try:
                os.makedirs(os.path.dirname(URDF_CACHE), exist_ok=True)
                with open(URDF_CACHE, 'w') as f:
                    f.write(xml)
            except OSError:
                pass
            return xml
        if os.path.exists(URDF_CACHE):
            self.get_logger().warn(f'using the saved arm model {URDF_CACHE}: robot_description is not available')
            return open(URDF_CACHE).read()
        raise RuntimeError('robot_description is not available (is robot bringup running?)')

    def _urdf_from_topic(self, timeout=3.0):
        from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
        from std_msgs.msg import String as Str
        got = []
        qos = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL, reliability=ReliabilityPolicy.RELIABLE)
        sub = self.create_subscription(Str, '/robot_description', lambda m: got.append(m.data), qos)
        t0 = time.time()
        while not got and time.time() - t0 < timeout:
            time.sleep(0.02)
        self.destroy_subscription(sub)
        return got[0] if got else None

    def _urdf_from_param(self):
        from rcl_interfaces.srv import GetParameters
        cli = self.create_client(GetParameters, '/robot_state_publisher/get_parameters')
        try:
            if not cli.wait_for_service(timeout_sec=2.0):
                return None
            fut = cli.call_async(GetParameters.Request(names=['robot_description']))
            t0 = time.time()
            while not fut.done() and time.time() - t0 < 3.0:
                time.sleep(0.02)
            return fut.result().values[0].string_value if fut.done() else None
        finally:
            self.destroy_client(cli)

    def _q(self):
        """(q (7,), gripper dict, age_s) or (None, None, age)."""
        js, age = self.feed.latest()
        if not all(j in js for j in kin.ARM_JOINTS):
            return None, None, age
        finger = js['finger_joint'] if 'finger_joint' in js else self.feed.finger()
        grip = {'finger_joint': finger} if finger is not None else None
        return np.array([js[j] for j in kin.ARM_JOINTS]), grip, age

    # ── state machine ─────────────────────────────────────────────────
    def preconditions(self):
        """None if the jog may run, else why not."""
        q, _g, age = self._q()
        if q is None or age > STALE_JOINT_S:
            return f'no live joint states (last {age:.1f} s ago): is robot bringup running?'
        with self._lock:
            fault, status = self._fault, self._arm_status
        if fault:
            return 'the driver reports a fault (/fault_controller/internal_fault): restart robot bringup first'
        if status.startswith('EXECUTING'):
            return f'the arm controller is busy ({status})'
        geom, err = sg.load_geometry()
        if geom is None:
            return f'no table geometry: {err}'
        return None

    def enable(self, phrase):
        if (phrase or '').strip().lower() != ENABLE_PHRASE:
            return {'ok': False, 'error': f'type exactly: {ENABLE_PHRASE}'}
        try:
            self._kinematics()
        except Exception as e:
            return {'ok': False, 'error': f'cannot load the arm model: {e}'}
        why = self.preconditions()
        if why:
            return {'ok': False, 'error': why}
        with self._lock:
            self.enabled = True
            self.enabled_at = self.last_cmd_t = time.monotonic()
            self._win = self._ref = None
            self.message = 'jog is ON: hold a button to move'
        return {'ok': True}

    def disable(self, why='jog switched off'):
        with self._lock:
            was = self.enabled
            self.enabled = False
            self._win = self._ref = None
            self.message = why
        return was

    def stop(self, why='stopped'):
        """Hold the arm where it is and switch the jog off."""
        q, _g, age = self._q()
        if q is not None and age < 2.0:
            self._publish(q, 0.3)
        self.disable(why)

    def gripper(self, action, position=None):
        """Open / close / move the gripper from the dashboard. Needs the jog to be ON (the typed phrase), no driver fault and an idle arm controller. Opening drops whatever is held,
        so the UI says so. Returns {'ok', 'message', 'finger'}."""
        with self._lock:
            enabled, fault, status = self.enabled, self._fault, self._arm_status
        if not enabled:
            return {'ok': False, 'message': 'switch the jog on first (type the phrase): the gripper moves the real hardware'}
        if fault:
            return {'ok': False, 'message': 'the driver reports a fault: restart robot bringup first'}
        if status.startswith('EXECUTING'):
            return {'ok': False, 'message': f'the arm controller is busy ({status})'}
        if action == 'open':
            target = GRIPPER_OPEN
        elif action == 'close':
            target = GRIPPER_CLOSED
        elif action == 'set':
            try:
                target = float(position)
            except (TypeError, ValueError):
                return {'ok': False, 'message': 'position must be a number'}
            if not np.isfinite(target):
                return {'ok': False, 'message': 'position must be a number'}
            target = float(np.clip(target, GRIPPER_OPEN, GRIPPER_CLOSED))
        else:
            return {'ok': False, 'message': "action must be 'open', 'close' or 'set'"}
        ok, msg = self._gripper_goal(target)
        js, _age = self.feed.latest()
        with self._lock:
            self.message = f'gripper: {msg}'
        return {'ok': ok, 'message': msg, 'finger': js.get('finger_joint')}

    def _gripper_goal(self, target):
        """Send one GripperCommand goal and wait for it (the Robotiq reports 'stalled' when it closes on an object, which counts as done)."""
        from control_msgs.action import GripperCommand as GC
        from rclpy.action import ActionClient
        if not hasattr(self, '_gripper_client'):
            self._gripper_client = ActionClient(self, GC, GRIPPER_ACTION)
        if not self._gripper_client.wait_for_server(timeout_sec=3.0):
            return False, 'the gripper controller is not available (is robot bringup running?)'
        goal = GC.Goal()
        goal.command.position = float(target)
        goal.command.max_effort = GRIPPER_MAX_EFFORT
        fut = self._gripper_client.send_goal_async(goal)
        t0 = time.time()
        while not fut.done() and time.time() - t0 < 3.0:
            time.sleep(0.02)
        handle = fut.result() if fut.done() else None
        if handle is None or not handle.accepted:
            return False, 'the gripper did not accept the command'
        rfut = handle.get_result_async()
        t0 = time.time()
        while not rfut.done() and time.time() - t0 < GRIPPER_WAIT_S:
            time.sleep(0.02)
        if not rfut.done():
            return False, f'the gripper did not finish within {GRIPPER_WAIT_S:.0f} s'
        res = rfut.result().result
        if res.reached_goal:
            return True, f'at {res.position:.2f}'
        if res.stalled:
            return True, f'closed on something (stopped at {res.position:.2f})'
        return False, f'stopped short at {res.position:.2f}'

    def _publish(self, q_target, t_s):
        """One waypoint to q_target in t_s seconds (used for 'hold here' on Stop / fault)."""
        self._send([t_s], [np.asarray(q_target, float)], None)

    def _send(self, times, positions, velocities, backdate_s=0.0):
        msg = JointTrajectory()
        msg.joint_names = list(kin.ARM_JOINTS)
        if backdate_s:
            msg.header.stamp = (self.get_clock().now() - RclDuration(seconds=backdate_s)).to_msg()
        for i, t_s in enumerate(times):
            p = JointTrajectoryPoint()
            p.positions = [float(v) for v in positions[i]]
            if velocities is not None:
                p.velocities = [float(v) for v in velocities[i]]
            sec = int(t_s)
            p.time_from_start = Duration(sec=sec, nanosec=int((t_s - sec) * 1e9))
            msg.points.append(p)
        self.pub.publish(msg)

    def command(self, cmd):
        """One tick of a hold. cmd: {'mode': 'joint', 'joint': 1-7, 'dir': +-1, 'speed': 0-1} or {'mode': 'cart', 'v': [6 in -1..1], 'speed', 'frame': 'base'|'tool'}.
        Returns the state dict (with 'moved': bool and 'blocked': reason or None)."""
        now = time.monotonic()
        with self._lock:
            if not self.enabled:
                return self.state(blocked='jog is off')
            if now - self.last_tick_t < MIN_TICK_S:
                return self.state()
            prev_tick = self.last_tick_t
            self.last_tick_t = now
            idle = now - self.last_cmd_t
            self.last_cmd_t = now
        why = self.preconditions()
        if why:
            self.stop(f'stopped: {why}')
            return self.state(blocked=why)
        k = self._kinematics()
        q, grip, _age = self._q()
        geom, _ = sg.load_geometry()
        try:
            if cmd.get('mode') == 'joint':
                j = int(cmd.get('joint', 0)) - 1
                if not 0 <= j < 7:
                    return self.state(blocked='joint must be 1-7')
                dq, bad = jm.joint_step(q, j, cmd.get('dir', 0), cmd.get('speed', 0.25)), None
            elif cmd.get('mode') == 'cart':
                v = [float(x) for x in cmd.get('v', [])]
                if len(v) != 6 or not all(np.isfinite(v)):
                    return self.state(blocked='v must be 6 numbers')
                frame = 'tool' if cmd.get('frame') == 'tool' else 'base'
                dq, bad = jm.cart_step(k, q, v, cmd.get('speed', 0.25), frame)
            else:
                return self.state(blocked='unknown mode')
        except Exception as e:                                # numeric failure: never move on a half-computed step
            return self.state(blocked=f'step failed: {e}')
        if bad:
            self._win = self._ref = None
            return self.state(blocked=bad)
        if not np.any(dq):
            self._win = self._ref = None
            return self.state()
        v_target = dq / jm.HORIZON_S
        dt = now - prev_tick
        continuing = self._ref is not None and dt <= jm.CONT_S
        if continuing:
            # the previous message's line goes on through now; the commanded velocity may change only gently
            q_ref = self._ref + self._v * dt
            v = jm.approach_velocity(self._v, v_target, dt)
        else:
            q_ref = q.copy()
            v = jm.approach_velocity(np.zeros(7), v_target, jm.NOMINAL_TICK_S)
        lead = np.maximum(np.abs(v_target), np.abs(self._v if continuing else 0.0)) * jm.LEAD_MAX_S + 1e-4
        q_ref = np.clip(q_ref, q - lead, q + lead)
        q_hold = q_ref + v * jm.HOLD_S
        q_stop = q_ref + jm.coast_displacement(v)
        for q_chk in (q_hold, q_stop):
            ok, reason = jm.check_pose(k, q, q_chk, geom, grip)
            if not ok:
                self._win = self._ref = None
                self.last_step = {'ok': False, 'reason': reason}
                return self.state(blocked=reason)
        # the joint feedback must follow: frozen or faulted feedback would let the next tick plan from a lie
        w = self._win
        if w is None or idle > GAP_S:
            self._win = w = [now, q.copy(), np.zeros(7)]
        w[2] = w[2] + v * (dt if continuing else MIN_TICK_S)
        if now - w[0] >= FOLLOW_CHECK_S:
            expected, got = np.linalg.norm(w[2]), np.linalg.norm(q - w[1])
            if expected > FOLLOW_MIN_RAD and got < FOLLOW_MIN_FRACTION * expected:
                self.stop('stopped: the arm is not following the commands (joint feedback frozen or the driver is faulted)')
                return self.state(blocked=self.message)
            self._win = [now, q.copy(), np.zeros(7)]
        times, pos, vel = jm.velocity_profile(q_ref, v)
        self._send(times, pos, vel, backdate_s=jm.BACK_S)
        self._ref, self._v = q_ref, v
        self.last_step = {'ok': True, 'v_deg_s': [round(float(np.degrees(x)), 2) for x in v]}
        with self._lock:
            self.message = 'moving'
        return self.state(moved=True)

    ROBOT_LINKS = ['base_link', 'shoulder_link', 'half_arm_1_link', 'half_arm_2_link', 'forearm_link', 'spherical_wrist_1_link', 'spherical_wrist_2_link', 'bracelet_link',
                   'end_effector_link', 'robotiq_140_base_link', 'left_outer_finger', 'right_outer_finger', 'left_inner_finger_pad', 'right_inner_finger_pad']

    def robot_pose(self):
        """What the 3D robot view draws: the base_link position of every link of the arm and gripper from forward kinematics of the live joints, plus the planner's workspace limits.
        {'ok': False, ...} when there are no live joint states."""
        q, grip, age = self._q()
        if q is None or age > 2.0:
            return {'ok': False, 'reason': 'no live joint states', 'age_s': round(age, 1) if age < 1e8 else None}
        try:
            k = self._kinematics()
            poses = k.pose_for(self.ROBOT_LINKS, q, grip)
        except Exception as e:
            return {'ok': False, 'reason': f'cannot compute the pose: {e}'}
        geom, _ = sg.load_geometry()
        return {'ok': True, 'age_s': round(age, 2), 'joints_deg': [round(float(np.degrees(v)), 1) for v in q],
                'finger': None if grip is None else round(float(grip['finger_joint']), 3),
                'links': {n: [round(float(v), 4) for v in T[:3, 3]] for n, T in poses.items()},
                'workspace': {'x': list(sg.PLAN_X_RANGE), 'y': list(sg.PLAN_Y_RANGE)},
                'table_top_z': None if not geom else geom['table_top_z']}

    def tick_idle(self):
        """Called by the router's housekeeping: switch off after a long idle."""
        with self._lock:
            if self.enabled and time.monotonic() - self.last_cmd_t > IDLE_DISABLE_S:
                self.enabled = False
                self.message = f'jog switched off after {int(IDLE_DISABLE_S)} s without commands'

    def state(self, moved=False, blocked=None):
        q, grip, age = self._q()
        out = {'enabled': self.enabled, 'message': self.message, 'moved': moved, 'blocked': blocked,
               'speed_caps': {'joint_deg_s': round(float(np.degrees(jm.JOINT_MAX_RAD_S)), 1), 'linear_cm_s': jm.LINEAR_MAX_M_S * 100,
                              'angular_deg_s': round(float(np.degrees(jm.ANGULAR_MAX_RAD_S)), 1)},
               'joint_age_s': round(age, 2) if age < 1e8 else None, 'phrase': ENABLE_PHRASE}
        if q is not None:
            out['joints_deg'] = [round(float(np.degrees(v)), 2) for v in q]
            try:
                k = self._kinematics()
                T = k.pose_for(['end_effector_link'], q)['end_effector_link']
                out['flange'] = [round(float(v), 4) for v in T[:3, 3]]
                zs = [p[2] for p in jm.guarded_positions(k, q, grip).values()]
                geom, _ = sg.load_geometry()
                if geom:
                    out['lowest_link_above_limit_cm'] = round(float((min(zs) - geom['table_top_z'] - sg.LINK_TABLE_CLEARANCE_M) * 100), 1)
            except Exception:
                pass
        if grip is not None:
            out['finger'] = round(float(grip['finger_joint']), 3)
        with self._lock:
            out['fault'] = self._fault
            out['arm_status'] = self._arm_status
        return out
