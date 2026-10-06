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
from collections import defaultdict
from rclpy.node import Node
from std_msgs.msg import String
from geometry_msgs.msg import PointStamped
from scipy.optimize import linear_sum_assignment
import numpy as np
import tf2_ros
import tf2_geometry_msgs  # noqa: F401 — registers PointStamped with tf2

from thesis_robot import safety_geometry as sg

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
STALE_THRESHOLD = 20.0    # seconds — unseen objects get marked stale (was 60: ghosts of moved objects lingered)
GC_THRESHOLD = 60.0      # seconds — stale objects unseen this long are dropped (was 300)
                          # entirely, so a ghost track can never sit around
                          # indefinitely as a candidate for anything
SAME_OBJECT_DIST = 0.15  # metres — detections this close are the same object
# Kinova Gen3 7DOF max horizontal reach from base_link, plus slack -- same
# bound used by calibration/define_workspace_boundary.py's own plausibility
# check. A resolved position beyond this is virtually always a bad
# detection (common with the HSV color-fallback detector misfiring on
# background clutter, not a real object on the table), not a legitimate
# occlusion/tracking case -- reject it before it ever becomes a track.
MAX_PLAUSIBLE_REACH_M = 2.0
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


TABLE_REGION_MARGIN_M = 0.05    # keep detections this far outside the recorded table footprint
DUP_RADIUS_M = 0.06          # two detections of different labels this close are one object


class SceneGraphNode(Node):

    def __init__(self):
        super().__init__('scene_graph_node')
        self.workspace = _load_workspace(self.get_logger())
        self._table_cache = (None, None)

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
        self.create_subscription(
            String, '/detections_realsense2',
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
        """Receive YOLO detections and update scene graph.

        Resolves and matches the WHOLE batch from this message together
        (not one detection at a time) so that when two same-label objects
        are detected in a single frame, each gets optimally matched to its
        own existing track rather than both racing for whichever track a
        naive first-match-wins scan happens to find first. See
        _match_batch for why this -- and excluding stale tracks from
        matching -- is what actually fixes occlusion mix-ups."""
        try:
            detections = json.loads(msg.data)
        except json.JSONDecodeError as e:
            self.get_logger().warn(f'Bad detection JSON: {e}')
            return

        if not isinstance(detections, list):
            # Handle single detection dict (object_detection.py format)
            detections = [detections]

        now = time.time()
        resolved = []  # [(label, x, y, z, conf), ...]
        for det in detections:
            label = det.get('label') or det.get('class', 'unknown')
            conf  = float(det.get('confidence', 0.0))
            if conf < 0.45:
                continue  # skip low confidence
            pos = self._resolve_base_xyz(det, label)
            if pos is None:
                continue
            x, y, z = pos
            if not self._on_table_region(x, y, z):
                continue                      # the room: another desk, the people, the floor — not objects on the table
            if z is not None and self._on_robot((x, y, z)):
                continue                      # a detection of the robot's own arm/gripper, not an object
            resolved.append((label, x, y, z, conf))

        resolved = self._drop_relabelled_duplicates(resolved)

        if not resolved:
            return

        assignments = self._match_batch(resolved)
        for (label, x, y, z, conf), obj_id in zip(resolved, assignments):
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

        # Once per 10 s, not once per detection message: this line alone wrote ~660,000 lines
        # (and, over a few days, 7 GB of ROS logs) and filled the disk.
        self.get_logger().info(
            f'Scene: {len(self.scene)} objects — '
            f'{[v["label"] for v in self.scene.values()]}',
            throttle_duration_sec=10.0,
        )

    def _drop_relabelled_duplicates(self, resolved):
        """A mouse seen from above is "cup" to the detectors, a bottle "cup", a bowl "mouse": the same object
        shows up under two labels a few cm apart. Keep the more confident one. Only within one batch and
        against fresh objects, and only within DUP_RADIUS_M, which no two real objects can be (centre to centre)."""
        keep = []
        now = time.time()
        for det in resolved:
            label, x, y, z, conf = det
            dup = False
            for other in resolved:
                if other is det or other[0] == label:
                    continue
                if math.hypot(other[1] - x, other[2] - y) < DUP_RADIUS_M and other[4] > conf + 0.05:
                    dup = True
                    break
            if not dup:
                for o in self.scene.values():
                    if o['stale'] or now - o['last_seen'] > 2.0 or o['label'] == label:
                        continue
                    if math.hypot(o['x'] - x, o['y'] - y) < DUP_RADIUS_M and o['confidence'] > conf + 0.15:
                        dup = True
                        break
            if not dup:
                keep.append(det)
        return keep

    def _on_table_region(self, x, y, z):
        """True if (x, y, z) is above the recorded table, within a margin. The cameras see the whole lab: other
        desks, chairs and people behind the robot (x < 0) and beyond the table's far edge produced 30+ "objects"
        that were never there. Falls back to the workspace box if no table is recorded."""
        now = time.monotonic()
        cached = getattr(self, '_table_region_cache', None)
        if cached is None or now - cached[0] > 5.0:
            geom, _ = sg.load_geometry()
            self._table_region_cache = cached = (now, geom)
        geom = cached[1]
        if geom is not None:
            return sg.in_table_region(geom, x, y, z, TABLE_REGION_MARGIN_M)
        ws_ = self.workspace
        return ws_['x'][0] - 0.1 <= x <= ws_['x'][1] + 0.3 and ws_['y'][0] - 0.2 <= y <= ws_['y'][1] + 0.2

    # ── ROBOT SELF-DETECTION GUARD ───────────────────────────────────
    def _arm_body(self):
        """(chain, fingers): the arm's link origins and finger segments in base_link, from TF; cached for
        0.15 s (this runs for every detection). (None, None) if TF is not available."""
        now = time.monotonic()
        cached = getattr(self, '_arm_body_cache', None)
        if cached and now - cached[0] < 0.15:
            return cached[1], cached[2]
        try:
            def pos(frame):
                t = self.tf_buffer.lookup_transform(BASE_FRAME, frame, rclpy.time.Time()).transform.translation
                return (t.x, t.y, t.z)
            chain = [(0.0, 0.0, 0.0)] + [pos(f) for f in sg.ARM_CHAIN_FRAMES]
            flange = chain[-1]
            fingers = [(flange, pos(f)) for f in sg.FINGER_PAD_FRAMES]
        except Exception:
            self._arm_body_cache = (now, None, None)
            return None, None
        self._arm_body_cache = (now, chain, fingers)
        return chain, fingers

    def _on_robot(self, point):
        chain, fingers = self._arm_body()
        if chain is None:
            return False                      # no TF for the arm: cannot tell, keep the detection
        return sg.point_on_robot(point, chain, fingers)

    # ── SNAPSHOT PUBLISHER ───────────────────────────────────────────
    def publish_snapshot(self):
        """Publish current scene as clean JSON for LLM consumption. An empty
        scene is published too ({}), so consumers clear their copy instead of
        planning against the last non-empty snapshot forever."""
        table_top = self._table_top()
        snapshot = {}
        for obj in self.scene.values():
            obj['reachable'] = self._is_reachable(obj['x'], obj['y'], obj['z'])
        # relations are only meaningful between objects the planner can act on;
        # computing them against every ghost made the payload quadratic and
        # repeated the same "right_of_mouse" dozens of times.
        live = [oid for oid, o in self.scene.items() if o['reachable'] and not o['stale']]

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
            if table_top is not None and obj['z'] is not None \
                    and obj['z'] < table_top - sg.BELOW_TABLE_TOL_M:
                entry['below_table'] = True     # phantom: lower than the table it should be on

            if obj_id in live:
                relations = set()
                for other_id in live:
                    if other_id == obj_id:
                        continue
                    other = self.scene[other_id]
                    if self._is_near(obj, other):
                        relations.add(f"near_{other['label']}")
                    if self._is_left_of(obj, other):
                        relations.add(f"left_of_{other['label']}")
                    if self._is_right_of(obj, other):
                        relations.add(f"right_of_{other['label']}")
                if relations:
                    entry['relations'] = sorted(relations)

            snapshot[obj_id] = entry

        self.snapshot_pub.publish(String(data=json.dumps(snapshot)))

    # ── STALENESS UPDATE ─────────────────────────────────────────────
    def update_staleness(self):
        """Mark objects not seen recently as stale, and drop ones that
        have been stale for a long time entirely -- otherwise a ghost
        track from an object that was moved/removed while occluded would
        sit in self.scene forever, excluded from LLM prompts (stale) and
        from matching (see _match_batch) but still cluttering /scene_snapshot
        and still occupying that label's next-index slot."""
        now = time.time()
        to_drop = []
        for obj_id, obj in self.scene.items():
            age = now - obj['last_seen']
            if age > GC_THRESHOLD:
                to_drop.append(obj_id)
                continue
            was_stale = obj['stale']
            obj['stale'] = age > STALE_THRESHOLD
            if obj['stale'] and not was_stale:
                self.get_logger().warn(
                    f"Object {obj_id} ({obj['label']}) went stale "
                    f"({age:.1f}s since last seen)"
                )
        for obj_id in to_drop:
            self.get_logger().info(f'Dropping {obj_id} — unseen for {GC_THRESHOLD:.0f}s+')
            del self.scene[obj_id]

    # ── SPATIAL PREDICATES ───────────────────────────────────────────
    def _table_top(self):
        """Recorded table-top z (base_link), re-read only when the file changes."""
        try:
            mtime = os.path.getmtime(sg.TABLE_GEOMETRY_FILE)
        except OSError:
            self._table_cache = (None, None)
            return None
        if self._table_cache[0] != mtime:
            geom, _err = sg.load_geometry()
            self._table_cache = (mtime, geom['table_top_z'] if geom else None)
        return self._table_cache[1]

    def _is_reachable(self, x, y, z):
        return sg.is_reachable(x, y, z, self.workspace, self._table_top())

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
            return self._check_plausible(label, x, y, num('z_robot'))

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

        tf = self._lookup_tf(frame_id, num('stamp'))
        if tf is None:
            self.get_logger().warn(
                f'Dropping {label}: no TF {frame_id} -> {BASE_FRAME}',
                throttle_duration_sec=5.0)
            return None
        if tf is False:
            return None          # frame too old to trust

        pt = PointStamped()
        pt.header.frame_id = frame_id
        pt.point.x, pt.point.y, pt.point.z = cx, cy, cz
        p = tf2_geometry_msgs.do_transform_point(pt, tf).point
        # plain floats: tf2 may return numpy scalars, which json can't encode
        return self._check_plausible(label, float(p.x), float(p.y), float(p.z))

    def _lookup_tf(self, frame_id, stamp_s):
        """base_link <- frame_id at the time the image was TAKEN when that is
        trustworthy, else the latest transform. Matters for the wrist camera:
        it moves with the arm, and YOLO + queueing add 0.05-1 s of delay, so
        projecting with the LATEST arm pose smears an object across the table
        while the arm is moving. A stamp is only used if it is within 10 s of
        now (the OAK-D driver may stamp with its own clock); a frame older
        than 1.5 s is rejected (returns False). None = no TF at all."""
        when = rclpy.time.Time()
        if stamp_s is not None:
            age = time.time() - stamp_s
            if abs(age) < 10.0:
                if age > 1.5:
                    return False
                when = rclpy.time.Time(seconds=int(stamp_s), nanoseconds=int((stamp_s % 1) * 1e9))
        for t in ([when, rclpy.time.Time()] if when != rclpy.time.Time() else [when]):
            try:
                return self.tf_buffer.lookup_transform(BASE_FRAME, frame_id, t)
            except (tf2_ros.LookupException, tf2_ros.ConnectivityException,
                    tf2_ros.ExtrapolationException):
                continue
        return None

    def _check_plausible(self, label, x, y, z):
        """Reject a resolved position too far from base_link to be a real
        object on the table -- common with the HSV color-fallback
        detector misfiring on background clutter (seen in practice:
        garbage detections 10+ metres away from a bad/noisy depth read),
        not a legitimate detection that just happens to be far. Returns
        (x, y, z) unchanged if plausible, else None."""
        dist = (x ** 2 + y ** 2 + (z or 0.0) ** 2) ** 0.5
        if dist > MAX_PLAUSIBLE_REACH_M:
            self.get_logger().warn(
                f'Dropping {label}: resolved position is {dist:.2f}m from '
                f'base_link (> {MAX_PLAUSIBLE_REACH_M}m) — almost certainly '
                f'a bad detection, not a real object.',
                throttle_duration_sec=5.0)
            return None
        return x, y, z

    def _match_batch(self, resolved):
        """Assign each (label, x, y, z, conf) in `resolved` to a track ID,
        returning a list of IDs in the same order as `resolved`.

        Two things that make this different from "find the nearest
        same-label track and reuse its ID" (what this used to do, one
        detection at a time):

        1. Optimal assignment, not greedy. When a single frame has two
           detections of the same label (e.g. two cups), matching them
           one at a time in list order can let the first detection grab
           the track that was actually closer to the second one, swapping
           which physical cup each tracked ID refers to. Hungarian
           assignment (scipy linear_sum_assignment) finds the matching
           that minimizes TOTAL distance across the whole label group at
           once, which avoids that.

        2. Stale tracks are never match candidates. A track that hasn't
           been seen in up to STALE_THRESHOLD seconds has no business
           silently absorbing a fresh detection just because it happens
           to land within SAME_OBJECT_DIST of that track's last known (by
           now possibly stale/wrong) position -- if a different
           same-label object was set down nearby in the meantime, that
           silent merge is exactly the "confuses objects after occlusion"
           bug this replaces. A fresh detection near a stale track simply
           starts a new track instead; the stale one ages out via
           update_staleness's GC_THRESHOLD.
        """
        by_label = defaultdict(list)
        for i, (label, x, y, z, conf) in enumerate(resolved):
            by_label[label].append(i)

        assignments = [None] * len(resolved)
        for label, idxs in by_label.items():
            track_ids = [
                oid for oid, obj in self.scene.items()
                if obj['label'] == label and not obj['stale']
            ]
            if not track_ids:
                for i in idxs:
                    assignments[i] = self._new_id(label, assignments)
                continue

            cost = np.full((len(idxs), len(track_ids)), 1e6)
            for r, i in enumerate(idxs):
                _, x, y, _, _ = resolved[i]
                for c, oid in enumerate(track_ids):
                    obj = self.scene[oid]
                    dist = ((obj['x'] - x) ** 2 + (obj['y'] - y) ** 2) ** 0.5
                    if dist < SAME_OBJECT_DIST:
                        cost[r, c] = dist

            rows, cols = linear_sum_assignment(cost)
            matched_tracks = set()
            for r, c in zip(rows, cols):
                i = idxs[r]
                if cost[r, c] >= 1e6:
                    continue  # no track within SAME_OBJECT_DIST -- new object
                assignments[i] = track_ids[c]
                matched_tracks.add(track_ids[c])
            for i in idxs:
                if assignments[i] is None:
                    assignments[i] = self._new_id(label, assignments)

        return assignments

    def _new_id(self, label, assignments_so_far):
        """A fresh track ID for `label`, distinct from every existing
        track AND every ID already handed out earlier in this same
        batch (assignments_so_far may contain not-yet-committed IDs that
        aren't in self.scene yet)."""
        used = {oid for oid in self.scene if oid.startswith(f'{label}_')}
        used |= {oid for oid in assignments_so_far if oid and oid.startswith(f'{label}_')}
        count = 0
        while f'{label}_{count:02d}' in used:
            count += 1
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
