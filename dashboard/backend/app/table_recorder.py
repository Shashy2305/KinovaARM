"""
Records the table height/footprint by hand-guiding the arm: the operator
touches the open gripper's finger tips to the table at several spots and
records the fingertip-pad poses (from the arm's own forward kinematics via
TF, so no depth-camera noise). The result becomes ~/.ros/table_geometry.yaml,
which arm_controller turns into MoveIt collision objects and trajectory
limits (see thesis_robot/safety_geometry.py).
"""
import threading

import rclpy
import tf2_ros
from rclpy.node import Node

from thesis_robot import safety_geometry as sg

BASE_FRAME = 'base_link'
PAD_FRAMES = ('left_inner_finger_pad', 'right_inner_finger_pad')
MAX_POSE_AGE_S = 1.5


class TableRecorder(Node):
    def __init__(self):
        super().__init__('table_recorder')
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        self.points = []   # [(x, y, z_pad)], z = the LOWER pad frame origin
        self._lock = threading.Lock()

    def current_pose(self):
        """(pose dict or None, message). Refuses stale or missing arm poses
        so a dead arm connection can never be recorded as a table point."""
        found = []
        for frame in PAD_FRAMES:
            try:
                t = self.tf_buffer.lookup_transform(BASE_FRAME, frame, rclpy.time.Time())
            except Exception:
                return None, (f'No arm pose: TF {BASE_FRAME} -> {frame} is unavailable. '
                              f'Is the arm connected and robot_bringup healthy?')
            age = (self.get_clock().now() - rclpy.time.Time.from_msg(t.header.stamp)).nanoseconds / 1e9
            if age > MAX_POSE_AGE_S:
                return None, f'Arm pose is stale ({age:.1f}s old) — the arm connection looks dead.'
            found.append((t.transform.translation.x, t.transform.translation.y, t.transform.translation.z))
        return {
            'x': sum(p[0] for p in found) / 2.0,
            'y': sum(p[1] for p in found) / 2.0,
            'z': min(p[2] for p in found),
        }, 'ok'

    def record(self):
        pose, msg = self.current_pose()
        if pose is None:
            return False, msg
        with self._lock:
            self.points.append((pose['x'], pose['y'], pose['z']))
            n = len(self.points)
        return True, (f'point {n}: x={pose["x"]:.3f} y={pose["y"]:.3f} pad z={pose["z"]:.3f}')

    def reset(self):
        with self._lock:
            self.points.clear()

    def snapshot(self):
        with self._lock:
            return [list(p) for p in self.points]

    def save(self):
        with self._lock:
            pts = list(self.points)
        geom, err = sg.geometry_from_points(pts)
        if geom is None:
            return False, err, None
        sg.save_geometry(geom)
        return True, (f'saved {sg.TABLE_GEOMETRY_FILE}: table top z={geom["table_top_z"]:.3f} '
                      f'(restart arm_controller or just send the next command — it re-reads this file)'), geom
