#!/usr/bin/env python3
"""
Scene Graph Node — builds and maintains a live world model.

Subscribes to:
  /detections, /detections_side, /detections_wrist
                     (std_msgs/String — JSON detection lists, one topic
                     per camera: OAK-D, RealSense, Kinova wrist)

Publishes to:
  /scene_snapshot    (std_msgs/String — JSON world model for LLM)
                     All x/y/z are in base_link; z is null when unknown.

Multi-camera fusion happens here, not in a separate fusion_node: every
detection already arrives with either base_link coordinates or
camera-frame coordinates + frame_id, so it's normalized to base_link by
TF regardless of which of the (possibly 3, possibly heterogeneous)
cameras it came from — see _resolve_base_xyz. When two cameras report the
same object at nearly the same time, their positions are blended by an
exponential moving average weighted by confidence (_fuse_position)
instead of the last writer simply overwriting the others.

Your thesis contribution: structured spatial world model with
predicates (reachable, near, left_of) that ground the LLM's
understanding of the physical workspace.
"""
import os, rclpy, json, math, time, yaml
from rclpy.node import Node
from std_msgs.msg import String
from geometry_msgs.msg import PointStamped
import tf2_ros
import tf2_geometry_msgs  # noqa: F401 — registers PointStamped with tf2

# Every coordinate stored in the scene (and therefore every coordinate the
# LLM plans with and arm_controller executes) is in this frame.
BASE_FRAME = 'base_link'

# Robot workspace bounds (metres, in base_link frame) — fallback used only
# if ~/.ros/workspace_bounds.yaml doesn't exist. Prefer defining these with
# calibration/define_workspace_boundary.py (click the table's corners in a
# camera view) over hand-editing these numbers — see _load_workspace().
DEFAULT_WORKSPACE = {'x': (0.08, 0.60), 'y': (-0.40, 0.40), 'z': (-0.50, 2.00)}
WORKSPACE_BOUNDS_FILE = os.path.expanduser('~/.ros/workspace_bounds.yaml')
NEAR_THRESHOLD  = 0.12   # metres — objects closer than this are "near"
STALE_THRESHOLD = 60.0    # seconds — unseen objects get marked stale
SAME_OBJECT_DIST = 0.15  # metres — detections this close are the same object
FUSION_MIN_ALPHA = 0.15  # a single low-confidence reading moves the estimate at least this much
FUSION_MAX_ALPHA = 0.85  # a single high-confidence reading moves the estimate at most this much


def _load_workspace(logger):
    """Load WORKSPACE bounds from workspace_bounds.yaml if present (written
    by calibration/define_workspace_boundary.py), else fall back to
    DEFAULT_WORKSPACE."""
    if not os.path.exists(WORKSPACE_BOUNDS_FILE):
        logger.info(
            f'No {WORKSPACE_BOUNDS_FILE} — using DEFAULT_WORKSPACE. Run '
            f'calibration/define_workspace_boundary.py to define this by '
            f'clicking the table instead of hand-editing scene_graph_node.py.')
        return DEFAULT_WORKSPACE
    try:
        with open(WORKSPACE_BOUNDS_FILE) as f:
            data = yaml.safe_load(f)
        bounds = {axis: tuple(data[axis]) for axis in ('x', 'y', 'z')}
        logger.info(f'Loaded workspace bounds from {WORKSPACE_BOUNDS_FILE}: {bounds}')
        return bounds
    except (OSError, yaml.YAMLError, KeyError, TypeError) as e:
        logger.error(
            f'Invalid {WORKSPACE_BOUNDS_FILE} ({e}) — using DEFAULT_WORKSPACE instead.')
        return DEFAULT_WORKSPACE


class SceneGraphNode(Node):

    def __init__(self):
        super().__init__('scene_graph_node')
        self.workspace = _load_workspace(self.get_logger())

        # ── scene storage ────────────────────────────────────────────
        # { object_id: { label, x, y, z, confidence, last_seen, stale } }
        self.scene = {}

        # TF buffer for camera frame -> base_link
        self.tf_buffer   = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # ── subscribers ──────────────────────────────────────────────
        self.create_subscription(
            String, '/detections',
            self.detection_callback, 10)
        self.create_subscription(
            String, '/detections_side',
            self.detection_callback, 10)
        self.create_subscription(
            String, '/detections_wrist',
            self.detection_callback, 10)

        # ── publishers ───────────────────────────────────────────────
        self.snapshot_pub = self.create_publisher(
            String, '/scene_snapshot', 10)

        # ── publish scene snapshot at 5 Hz ───────────────────────────
        self.create_timer(0.2, self.publish_snapshot)

        # ── mark stale objects at 1 Hz ───────────────────────────────
        self.create_timer(1.0, self.update_staleness)

        self.get_logger().info('SceneGraphNode ready — listening on /detections')

    # ── DETECTION CALLBACK ───────────────────────────────────────────
    def detection_callback(self, msg):
        """Receive YOLO detections and update scene graph."""
        try:
            detections = json.loads(msg.data)
        except json.JSONDecodeError as e:
            self.get_logger().warn(f'Bad detection JSON: {e}')
            return

        if not isinstance(detections, list):
            # Handle single detection dict (object_detection.py format)
            detections = [detections]

        now = time.time()
        for det in detections:
            label = det.get('label') or det.get('class', 'unknown')
            conf  = float(det.get('confidence', 0.0))

            if conf < 0.45:
                continue  # skip low confidence

            pos = self._resolve_base_xyz(det, label)
            if pos is None:
                continue
            x, y, z = pos

            # Generate stable object ID: label + index
            obj_id = self._get_or_create_id(label, x, y)
            x, y, z, conf = self._fuse_position(obj_id, x, y, z, conf, now)

            self.scene[obj_id] = {
                'label':      label,
                'x':          round(x, 4),
                'y':          round(y, 4),
                'z':          round(z, 4) if z is not None else None,
                'confidence': round(conf, 3),
                'last_seen':  now,
                'stale':      False,
                'reachable':  self._is_reachable(x, y, z),
            }

        self.get_logger().info(
            f'Scene: {len(self.scene)} objects — '
            f'{[v["label"] for v in self.scene.values()]}'
        )

    # ── SNAPSHOT PUBLISHER ───────────────────────────────────────────
    def publish_snapshot(self):
        """Publish current scene as clean JSON for LLM consumption."""
        if not self.scene:
            return

        # Build snapshot — include spatial predicates
        snapshot = {}
        ids = list(self.scene.keys())

        for obj_id, obj in self.scene.items():
            entry = {
                'label':      obj['label'],
                'x':          obj['x'],
                'y':          obj['y'],
                'z':          obj['z'],
                'confidence': obj['confidence'],
                'reachable':  obj['reachable'],
                'stale':      obj['stale'],
            }

            # Add spatial relations to all other objects
            relations = []
            for other_id in ids:
                if other_id == obj_id:
                    continue
                other = self.scene[other_id]
                if self._is_near(obj, other):
                    relations.append(f"near_{other['label']}")
                if self._is_left_of(obj, other):
                    relations.append(f"left_of_{other['label']}")
                if self._is_right_of(obj, other):
                    relations.append(f"right_of_{other['label']}")

            if relations:
                entry['relations'] = relations

            snapshot[obj_id] = entry

        msg = String()
        msg.data = json.dumps(snapshot)
        self.snapshot_pub.publish(msg)

    # ── STALENESS UPDATE ─────────────────────────────────────────────
    def update_staleness(self):
        """Mark objects not seen recently as stale."""
        now = time.time()
        for obj_id, obj in self.scene.items():
            age = now - obj['last_seen']
            was_stale = obj['stale']
            obj['stale'] = age > STALE_THRESHOLD
            if obj['stale'] and not was_stale:
                self.get_logger().warn(
                    f"Object {obj_id} ({obj['label']}) went stale "
                    f"({age:.1f}s since last seen)"
                )

    # ── SPATIAL PREDICATES ───────────────────────────────────────────
    def _is_reachable(self, x, y, z):
        if z is None:
            return False  # height unknown — cannot claim it is graspable
        return (self.workspace['x'][0] <= x <= self.workspace['x'][1] and
                self.workspace['y'][0] <= y <= self.workspace['y'][1] and
                self.workspace['z'][0] <= z <= self.workspace['z'][1])

    def _is_near(self, a, b):
        dist = ((a['x']-b['x'])**2 + (a['y']-b['y'])**2)**0.5
        return dist < NEAR_THRESHOLD

    def _is_left_of(self, a, b):
        # "left" = more negative Y in robot base frame
        return (b['y'] - a['y']) > 0.08

    def _is_right_of(self, a, b):
        return (a['y'] - b['y']) > 0.08

    def _fuse_position(self, obj_id, x, y, z, conf, now):
        """Blend a new detection into the existing estimate for obj_id,
        instead of whichever camera happens to publish last overwriting
        the others. `alpha` (how much this single reading moves the
        estimate) scales with its confidence, so a confident detection
        from one camera can correct a shakier one from another without
        either source ever fully discarding the other's contribution."""
        existing = self.scene.get(obj_id)
        if existing is None or existing['stale']:
            return x, y, z, conf

        alpha = min(FUSION_MAX_ALPHA, max(FUSION_MIN_ALPHA, conf))
        fx = alpha * x + (1 - alpha) * existing['x']
        fy = alpha * y + (1 - alpha) * existing['y']
        if z is None:
            fz = existing['z']
        elif existing['z'] is None:
            fz = z
        else:
            fz = alpha * z + (1 - alpha) * existing['z']
        fconf = max(conf, existing['confidence'])
        return fx, fy, fz, fconf

    # ── HELPERS ──────────────────────────────────────────────────────
    def _resolve_base_xyz(self, det, label):
        """
        Return (x, y, z) of a detection in BASE_FRAME, or None to drop it.

        A detection must carry one of:
          x_robot / y_robot [/ z_robot]   already in base_link
                                          (z_robot may be null = height unknown)
          cx_3d / cy_3d / cz_3d + frame_id  camera-frame point, moved to
                                            base_link through TF
        Anything else is dropped: camera-frame numbers are never used as
        robot-frame numbers.
        """
        def num(key):
            v = det.get(key)
            if v is None:
                return None
            try:
                v = float(v)
            except (TypeError, ValueError):
                return None
            return v if math.isfinite(v) else None

        x, y = num('x_robot'), num('y_robot')
        if x is not None and y is not None:
            return x, y, num('z_robot')

        cx, cy, cz = num('cx_3d'), num('cy_3d'), num('cz_3d')
        frame_id = det.get('frame_id')
        if None in (cx, cy, cz) or not frame_id:
            self.get_logger().warn(
                f'Dropping {label}: no base_link coords and no '
                f'camera coords + frame_id',
                throttle_duration_sec=5.0)
            return None
        if cz <= 0.0:
            return None  # no valid depth at this pixel

        try:
            tf = self.tf_buffer.lookup_transform(
                BASE_FRAME, frame_id, rclpy.time.Time())
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException,
                tf2_ros.ExtrapolationException) as e:
            self.get_logger().warn(
                f'Dropping {label}: no TF {frame_id} -> {BASE_FRAME}: {e}',
                throttle_duration_sec=5.0)
            return None

        pt = PointStamped()
        pt.header.frame_id = frame_id
        pt.point.x, pt.point.y, pt.point.z = cx, cy, cz
        p = tf2_geometry_msgs.do_transform_point(pt, tf).point
        # plain floats: tf2 may return numpy scalars, which json can't encode
        return float(p.x), float(p.y), float(p.z)

    def _get_or_create_id(self, label, x, y):
        """
        Return existing ID if an object with this label is already
        tracked nearby, otherwise create a new one.
        Prevents duplicate entries for the same physical object.
        """
        for obj_id, obj in self.scene.items():
            if obj['label'] == label:
                dist = ((obj['x']-x)**2 + (obj['y']-y)**2)**0.5
                if dist < SAME_OBJECT_DIST:
                    return obj_id

        # New object — assign next index for this label
        count = sum(1 for oid in self.scene if oid.startswith(label))
        return f'{label}_{count:02d}'


def main(args=None):
    rclpy.init(args=args)
    node = SceneGraphNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
