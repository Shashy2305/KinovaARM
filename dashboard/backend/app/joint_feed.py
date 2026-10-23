"""
Latest /joint_states for the dashboard's nodes without paying for every message.

The driver publishes joint states at 1 kHz. Decoding each one in Python (two nodes did, jog and arm calibration) kept three executor threads at 100 % and starved every other request
(/api/jog/state took 5 s, 2026-10-22). This keeps only the newest RAW message (a reference assignment per message) and decodes it when somebody asks, at most every MIN_DECODE_S.
"""
import time

from rclpy.serialization import deserialize_message
from sensor_msgs.msg import JointState

MIN_DECODE_S = 0.01


class JointFeed:
    def __init__(self, node, topic='/joint_states'):
        self._raw = None
        self._raw_t = 0.0
        self._decoded_raw = None
        self._decoded_at = 0.0
        self._positions = {}
        self._velocities = {}
        self._t = 0.0
        node.create_subscription(JointState, topic, self._on_raw, 10, raw=True)

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
            msg = deserialize_message(raw, JointState)
            self._positions = dict(zip(msg.name, msg.position))
            self._velocities = dict(zip(msg.name, msg.velocity)) if len(msg.velocity) == len(msg.name) else {}
            self._decoded_raw, self._decoded_at, self._t = raw, now, raw_t
        t = max(self._t, 0.0)
        return dict(self._positions), (now - t if t else 1e9)
