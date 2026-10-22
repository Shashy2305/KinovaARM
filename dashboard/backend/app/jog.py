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
import threading
import time

import numpy as np
from builtin_interfaces.msg import Duration
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Bool, String
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

from thesis_robot import safety_geometry as sg

from . import jog_math as jm
from . import kinematics as kin

ENABLE_PHRASE = 'area clear, hands off'
IDLE_DISABLE_S = 120.0
STALE_JOINT_S = 0.5
MIN_TICK_S = 0.06                 # commands arriving faster than this are dropped (no flood of trajectories)
GAP_S = 0.4                       # a longer pause between commands ends a "continuous move"
FOLLOW_CHECK_S = 1.5
FOLLOW_MIN_FRACTION = 0.25        # the arm must have covered this share of what was commanded in the window
FOLLOW_MIN_RAD = 0.03
TOPIC = '/joint_trajectory_controller/joint_trajectory'


class Jogger(Node):
    def __init__(self, urdf_xml=None):
        super().__init__('dashboard_jog')
        self._lock = threading.RLock()
        self._urdf = urdf_xml
        self.k = None
        self.enabled = False
        self.enabled_at = 0.0
        self.last_cmd_t = 0.0
        self.last_tick_t = 0.0
        self.message = 'jog is off'
        self._js = {}
        self._js_t = 0.0
        self._fault = False
        self._fault_t = 0.0
        self._arm_status = ''
        self._win = None              # [t_start, q_start, expected displacement (7,)] of the current continuous move
        self.last_step = None
        self.pub = self.create_publisher(JointTrajectory, TOPIC, 5)
        self.create_subscription(JointState, '/joint_states', self._on_js, 10)
        self.create_subscription(Bool, '/fault_controller/internal_fault', self._on_fault, 5)
        self.create_subscription(String, '/arm_status', self._on_status, 10)

    # ── inputs ────────────────────────────────────────────────────────
    def _on_js(self, msg):
        with self._lock:
            self._js = dict(zip(msg.name, msg.position))
            self._js_t = time.monotonic()

    def _on_fault(self, msg):
        with self._lock:
            self._fault = bool(msg.data)
            self._fault_t = time.monotonic()

    def _on_status(self, msg):
        with self._lock:
            self._arm_status = msg.data

    def _kinematics(self):
        if self.k is None:
            xml = self._urdf or self._fetch_urdf()
            self.k = kin.Kinematics(xml)
        return self.k

    def _fetch_urdf(self):
        from rcl_interfaces.srv import GetParameters
        cli = self.create_client(GetParameters, '/robot_state_publisher/get_parameters')
        if not cli.wait_for_service(timeout_sec=3.0):
            raise RuntimeError('/robot_state_publisher is not running (start robot bringup)')
        fut = cli.call_async(GetParameters.Request(names=['robot_description']))
        t0 = time.time()
        while not fut.done() and time.time() - t0 < 5.0:
            time.sleep(0.02)
        if not fut.done():
            raise RuntimeError('timed out reading robot_description')
        return fut.result().values[0].string_value

    def _q(self):
        """(q (7,), gripper dict, age_s) or (None, None, age)."""
        with self._lock:
            js, t = dict(self._js), self._js_t
        age = time.monotonic() - t if t else 1e9
        if not all(j in js for j in kin.ARM_JOINTS):
            return None, None, age
        grip = {'finger_joint': js['finger_joint']} if 'finger_joint' in js else None
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
            self._win = None
            self.message = 'jog is ON: hold a button to move'
        return {'ok': True}

    def disable(self, why='jog switched off'):
        with self._lock:
            was = self.enabled
            self.enabled = False
            self._win = None
            self.message = why
        return was

    def stop(self, why='stopped'):
        """Hold the arm where it is and switch the jog off."""
        q, _g, age = self._q()
        if q is not None and age < 2.0:
            self._publish(q, 0.3)
        self.disable(why)

    def _publish(self, q_target, t_s):
        msg = JointTrajectory()
        msg.joint_names = list(kin.ARM_JOINTS)
        p = JointTrajectoryPoint()
        p.positions = [float(v) for v in q_target]
        sec = int(t_s)
        p.time_from_start = Duration(sec=sec, nanosec=int((t_s - sec) * 1e9))
        msg.points = [p]
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
            self._win = None
            return self.state(blocked=bad)
        if not np.any(dq):
            self._win = None
            return self.state()
        q_new = q + dq
        ok, reason = jm.check_pose(k, q, q_new, geom, grip)
        if not ok:
            self._win = None
            self.last_step = {'ok': False, 'reason': reason}
            return self.state(blocked=reason)
        # the joint feedback must follow: frozen or faulted feedback would let the next tick plan from a lie
        w = self._win
        if w is None or idle > GAP_S:
            self._win = w = [now, q.copy(), np.zeros(7)]
        w[2] = w[2] + dq
        if now - w[0] >= FOLLOW_CHECK_S:
            expected, got = np.linalg.norm(w[2]), np.linalg.norm(q - w[1])
            if expected > FOLLOW_MIN_RAD and got < FOLLOW_MIN_FRACTION * expected:
                self.stop('stopped: the arm is not following the commands (joint feedback frozen or the driver is faulted)')
                return self.state(blocked=self.message)
            self._win = [now, q.copy(), np.zeros(7)]
        self._publish(q_new, jm.HORIZON_S)
        self.last_step = {'ok': True, 'dq_deg': [round(float(np.degrees(x)), 3) for x in dq]}
        with self._lock:
            self.message = 'moving'
        return self.state(moved=True)

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
        with self._lock:
            out['fault'] = self._fault
            out['arm_status'] = self._arm_status
        return out
