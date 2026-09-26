#!/usr/bin/env python3
"""
Static TF Broadcaster — the single publisher of the OAK-D calibration TF
base_link -> global_camera_link.

Reads the calibration from a YAML file (parameter `calibration_file`,
default ~/.ros/handeye_calibration_corrected.yaml). Accepted format — as
written by handeye_calibration.py:
  translation:   {x, y, z}
  rotation_quat: {x, y, z, w}     (or `rotation:` with the same keys)
  parent_frame / child_frame      (optional; must match this node's frames)

If the file does not exist, the last known OAK-D calibration (previously
hard-coded in oak_camera_node.py) is used. If the file exists but is
invalid, the node refuses to start rather than publish a wrong transform.
"""
import math, os
import rclpy, yaml
from rclpy.node import Node
from geometry_msgs.msg import TransformStamped
import tf2_ros

DEFAULT_CALIB_FILE = '~/.ros/handeye_calibration_corrected.yaml'

# Last known OAK-D calibration — identical to the T_BASE_CAM that
# oak_camera_node.py used to broadcast, so behaviour is unchanged until a
# calibration file is written.
FALLBACK_T = [0.48, 0.72, 1.0]
FALLBACK_Q = [-0.341494, -0.888985, 0.299234, 0.059552]


def load_calibration(path, parent_frame, child_frame):
    """Return (translation [x,y,z], unit quaternion [x,y,z,w]) from a
    calibration YAML. Raises ValueError if the file is unusable."""
    with open(path) as f:
        c = yaml.safe_load(f)
    if not isinstance(c, dict):
        raise ValueError('file is not a YAML mapping')

    for key, want in (('parent_frame', parent_frame), ('child_frame', child_frame)):
        if key in c and c[key] != want:
            raise ValueError(f'{key} is {c[key]!r}, expected {want!r}')

    tr = c.get('translation')
    ro = c.get('rotation_quat', c.get('rotation'))
    if not isinstance(tr, dict) or not isinstance(ro, dict):
        raise ValueError('needs `translation` and `rotation_quat` (or `rotation`)')
    try:
        t = [float(tr[k]) for k in 'xyz']
        q = [float(ro[k]) for k in 'xyzw']
    except (KeyError, TypeError, ValueError) as e:
        raise ValueError(f'bad translation/rotation values: {e}')
    if not all(math.isfinite(v) for v in t + q):
        raise ValueError('non-finite value')

    norm = math.sqrt(sum(v * v for v in q))
    if abs(norm - 1.0) > 0.01:
        raise ValueError(f'rotation quaternion norm is {norm:.4f}, not ~1')
    return t, [v / norm for v in q]


class StaticTFBroadcaster(Node):
    def __init__(self):
        super().__init__('camera_tf_broadcaster')
        self.declare_parameter('calibration_file', DEFAULT_CALIB_FILE)
        self.declare_parameter('parent_frame', 'base_link')
        self.declare_parameter('child_frame', 'global_camera_link')
        calib_path = os.path.expanduser(self.get_parameter('calibration_file').value)
        parent     = self.get_parameter('parent_frame').value
        child      = self.get_parameter('child_frame').value

        self.broadcaster = tf2_ros.StaticTransformBroadcaster(self)

        if not os.path.exists(calib_path):
            self.get_logger().warn(
                f'No calibration found at {calib_path} — using last known '
                f'OAK-D calibration. Run handeye_calibration.py to generate one.'
            )
            t, q = FALLBACK_T, FALLBACK_Q
        else:
            try:
                t, q = load_calibration(calib_path, parent, child)
            except (OSError, yaml.YAMLError, ValueError) as e:
                self.get_logger().fatal(f'Invalid calibration {calib_path}: {e}')
                raise SystemExit(1)
            self.get_logger().info(
                f'Loaded calibration {calib_path}: '
                f't=[{t[0]:.3f},{t[1]:.3f},{t[2]:.3f}]'
            )

        msg = TransformStamped()
        msg.header.stamp    = self.get_clock().now().to_msg()
        msg.header.frame_id = parent
        msg.child_frame_id  = child
        msg.transform.translation.x = t[0]
        msg.transform.translation.y = t[1]
        msg.transform.translation.z = t[2]
        msg.transform.rotation.x = q[0]
        msg.transform.rotation.y = q[1]
        msg.transform.rotation.z = q[2]
        msg.transform.rotation.w = q[3]

        self.broadcaster.sendTransform(msg)
        self.get_logger().info(f'Broadcasting {parent} -> {child} TF')

def main(args=None):
    rclpy.init(args=args)
    try:
        node = StaticTFBroadcaster()
    except SystemExit:
        rclpy.shutdown()
        raise
    rclpy.spin(node)

if __name__ == '__main__':
    main()
