"""
Latest measured arm joint positions for the dashboard's nodes without paying for a 1 kHz topic.

/joint_states is published at 1 kHz. Decoding it in Python (jog and arm calibration did) kept the executor at 100 % and starved every other callback: /api/jog/state took 5 s, and later the camera
frames stopped reaching the dashboard altogether (2026-10-22). Even a raw (undecoded) subscription costs a Python callback per message. The joint trajectory controller republishes the SAME
measured positions (feedback) at ~90 Hz on its controller_state topic, so that is what is used here (the gripper joint is not part of it: guarded links are computed with the fingers open).
Only the newest raw message is kept; it is decoded when somebody asks, at most every MIN_DECODE_S.
"""
import time

from rclpy.serialization import deserialize_message
from control_msgs.msg import JointTrajectoryControllerState as ControllerState
from std_msgs.msg import Float32

MIN_DECODE_S = 0.01


class JointFeed:
    def __init__(self, node, topic='/joint_trajectory_controller/controller_state'):
        self._raw = None
        self._raw_t = 0.0
        self._decoded_raw = None
        self._decoded_at = 0.0
        self._positions = {}
        self._velocities = {}
        self._t = 0.0
        self._node = node
        self._finger = None
        self._finger_t = 0.0
        node.create_subscription(ControllerState, topic, self._on_raw, 10, raw=True)
        node.create_subscription(Float32, '/gripper_finger', self._on_finger, 5)

    def _on_raw(self, data):
        self._raw = data
        self._raw_t = time.monotonic()

    def inject(self, msg):
        """Tests: act as if this JointState had just arrived."""
        self._positions = dict(zip(msg.name, msg.position))
        self._velocities = dict(zip(msg.name, msg.velocity)) if len(msg.velocity) == len(msg.name) else {}
        self._t = time.monotonic()
        self._decoded_raw = self._raw

    def latest(self):
        """(positions {name: value}, age_s). age is 1e9 when nothing has arrived."""
        raw, raw_t = self._raw, self._raw_t
        now = time.monotonic()
        if raw is not None and raw is not self._decoded_raw and now - self._decoded_at >= MIN_DECODE_S:
            msg = deserialize_message(raw, ControllerState)
            self._positions = dict(zip(msg.joint_names, msg.feedback.positions))
            self._velocities = dict(zip(msg.joint_names, msg.feedback.velocities)) if len(msg.feedback.velocities) == len(msg.joint_names) else {}
            self._decoded_raw, self._decoded_at, self._t = raw, now, raw_t
        t = max(self._t, 0.0)
        return dict(self._positions), (now - t if t else 1e9)

    # ── the gripper finger (not part of controller_state) ─────────────
    def _on_finger(self, msg):
        self._finger = float(msg.data)
        self._finger_t = time.monotonic()

    def finger(self):
        """The finger_joint position (0 open .. 0.695 closed), or None if the arm controller has not reported it for 3 s. The arm controller republishes it at 5 Hz on /gripper_finger:
        subscribing to the 1 kHz /joint_states for it (or creating a subscription per sample, which stalled the executor on 2026-10-29) would cost far more."""
        return self._finger if time.monotonic() - self._finger_t < 3.0 else None
