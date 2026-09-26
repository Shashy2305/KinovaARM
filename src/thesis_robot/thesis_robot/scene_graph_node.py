#!/usr/bin/env python3
"""
Scene Graph Node — builds and maintains a live world model.

Subscribes to:
  /detections        (std_msgs/String — JSON list from yolo_detector)

Publishes to:
  /scene_snapshot    (std_msgs/String — JSON world model for LLM)

Your thesis contribution: structured spatial world model with
predicates (reachable, near, left_of) that ground the LLM's
understanding of the physical workspace.
"""
import rclpy, json, time
from rclpy.node import Node
from std_msgs.msg import String

# Robot workspace bounds (metres, in base_link frame)
WORKSPACE = {'x': (0.08, 0.60), 'y': (-0.40, 0.40), 'z': (-0.50, 2.00)}
NEAR_THRESHOLD  = 0.12   # metres — objects closer than this are "near"
STALE_THRESHOLD = 60.0    # seconds — unseen objects get marked stale

class SceneGraphNode(Node):

    def __init__(self):
        super().__init__('scene_graph_node')

        # ── scene storage ────────────────────────────────────────────
        # { object_id: { label, x, y, z, confidence, last_seen, stale } }
        self.scene = {}

        # ── subscribers ──────────────────────────────────────────────
        self.create_subscription(
            String, '/detections',
            self.detection_callback, 10)
        self.create_subscription(
            String, '/detections_side',
            self.detection_callback, 10)
        self.create_subscription(
            String, '/detections_fused',
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

            # Get 3D position — handle both formats from senior's nodes
            # object_detection.py publishes cx_3d/cy_3d/cz_3d
            # depth_3d_node.py will publish x_robot/y_robot/z_robot
            x = float(det.get('x_robot') or det.get('cx_3d') or 0.0)
            y = float(det.get('y_robot') or det.get('cy_3d') or 0.0)
            z_raw = float(det.get('z_robot') or det.get('cz_3d') or 0.0)
            # If z > 0.5m it's camera depth not table height — use 0.0
            z = 0.0 if z_raw > 0.5 else z_raw

            # Generate stable object ID: label + index
            obj_id = self._get_or_create_id(label, x, y)

            self.scene[obj_id] = {
                'label':      label,
                'x':          round(x, 4),
                'y':          round(y, 4),
                'z':          round(z, 4),
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
        return (WORKSPACE['x'][0] <= x <= WORKSPACE['x'][1] and
                WORKSPACE['y'][0] <= y <= WORKSPACE['y'][1] and
                WORKSPACE['z'][0] <= z <= WORKSPACE['z'][1])

    def _is_near(self, a, b):
        dist = ((a['x']-b['x'])**2 + (a['y']-b['y'])**2)**0.5
        return dist < NEAR_THRESHOLD

    def _is_left_of(self, a, b):
        # "left" = more negative Y in robot base frame
        return (b['y'] - a['y']) > 0.08

    def _is_right_of(self, a, b):
        return (a['y'] - b['y']) > 0.08

    # ── HELPERS ──────────────────────────────────────────────────────
    def _get_or_create_id(self, label, x, y):
        """
        Return existing ID if an object with this label is already
        tracked nearby, otherwise create a new one.
        Prevents duplicate entries for the same physical object.
        """
        for obj_id, obj in self.scene.items():
            if obj['label'] == label:
                dist = ((obj['x']-x)**2 + (obj['y']-y)**2)**0.5
                if dist < 0.15:   # same object if within 15cm
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
