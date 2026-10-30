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
from sensor_msgs.msg import JointState
from geometry_msgs.msg import PointStamped
from scipy.optimize import linear_sum_assignment
import numpy as np
import tf2_ros
import tf2_geometry_msgs  # noqa: F401 — registers PointStamped with tf2

from thesis_robot import camera_calibration_files as ccf
from thesis_robot import safety_geometry as sg
from thesis_robot import wrist_servo as ws

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
# Per-source fusion (2026-10-28): with a high alpha the published position was simply the LAST reading, so it jumped between the cameras' slightly different values on every message
# ("the scene is dancing"). Now each camera's latest reading is kept per object and the estimate is their weighted mean, low-passed.
SOURCE_WEIGHT = {'realsense': 1.0, 'realsense2': 1.0, 'realsense3': 0.3, 'wrist': 0.4}     # RS3's depth is poor; the wrist's is unreliable close up
SOURCE_Z_TRUSTED = ('realsense', 'realsense2')                                           # heights come from these when any of them saw the object
SOURCE_WINDOW_S = 2.0         # a camera's reading counts this long
SMOOTH_ALPHA = 0.35           # how far the published estimate moves toward the fused one per message
JUMP_M = 0.06                 # a fused estimate this far from the published one is a moved object: follow at once
COLOUR_VOTE_DECAY = 0.97
STATIC_SOURCES = ('realsense', 'realsense2', 'realsense3', 'oakd')
SURFACE_CENTRE_FACTOR = 1.0   # 1.0 = move static-camera points away from the camera by the object's radius (surface -> centre); 0 = off
WRIST_MIN_DOWN = 0.7          # the wrist camera only places objects when its axis points at the table (z component of the axis <= -0.7) ...
WRIST_MIN_RANGE_M = 0.25      # ... and the object is at least this far away (its depth is empty / wrong closer)


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
SAME_LABEL_MERGE_BY_LABEL_M = {'bottle': 0.14, 'cup': 0.12}   # transparent/low-contrast objects: depth differs most between cameras
SAME_LABEL_MERGE_M = 0.10      # same-label tracks closer than this are one object in the snapshot
DUP_RADIUS_M = 0.06          # two detections of different labels this close are one object
BLOB_GHOST_M = 0.08           # a colour blob this close to a recognised object is that object seen again


class SceneGraphNode(Node):

    def __init__(self):
        super().__init__('scene_graph_node')
        self.workspace = _load_workspace(self.get_logger())
        self._table_cache = (None, None)

        # ── scene storage ────────────────────────────────────────────
        # { object_id: { label, x, y, z, confidence, last_seen, stale } }
        self._obs = {}            # object_id -> {source: (t, x, y, z, conf)}: each camera's latest reading
        self._pair_gaps = {}      # 'cam_a|cam_b' -> deque of (t, gap_m): how far apart two static cameras put the SAME object (a camera that was moved shows up here)
        self._colour_votes = {}   # object_id -> {colour name: weight}: what the cameras say its colour is (a cast makes one frame lie; the vote does not)
        self.scene = {}

        # TF buffer for camera frame -> base_link
        self.tf_buffer   = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        self._finger = None                      # gripper finger joint, to know when the gripper is carrying something
        self.create_subscription(JointState, '/joint_states', self._on_joint_state, 10)

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
        self.create_subscription(
            String, '/detections_realsense3',
            self.detection_callback, 10)

        # ── publishers ───────────────────────────────────────────────
        self.snapshot_pub = self.create_publisher(
            String, '/scene_snapshot', 10)

        self.agreement_pub = self.create_publisher(String, '/camera_agreement', 10)
        self.create_timer(5.0, self.publish_agreement)

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
        src_of = {}    # id(detection tuple) -> the camera that saw it
        col_of = {}    # id(detection tuple) -> the colour name the camera gave it
        for det in detections:
            label = det.get('label') or det.get('class', 'unknown')
            conf  = float(det.get('confidence', 0.0))
            if conf < 0.45:
                continue  # skip low confidence
            pos = self._resolve_base_xyz(det, label)
            if pos is None:
                continue
            x, y, z = pos
            if det.get('source') in STATIC_SOURCES:
                x, y = self._to_object_centre(det, label, x, y)
            if det.get('source') == 'wrist' and not self._wrist_usable(det):
                continue                      # the wrist camera is not looking at the table: its 3D point is not an object on it
            if not self._on_table_region(x, y, z):
                continue                      # the room: another desk, the people, the floor — not objects on the table
            if z is not None and self._on_robot((x, y, z)):
                continue                      # a detection of the robot's own arm/gripper, not an object
            tup = (label, x, y, z, conf)
            src_of[id(tup)] = str(det.get('source', ''))
            col_of[id(tup)] = det.get('color')
            resolved.append(tup)

        resolved = self._drop_relabelled_duplicates(resolved)
        resolved = self._drop_blobs_on_known_objects(resolved)

        if not resolved:
            return

        assignments = self._match_batch(resolved)
        for tup, obj_id in zip(resolved, assignments):
            label, x, y, z, conf = tup
            x, y, z, conf = self._fuse_position(obj_id, x, y, z, conf, now, src_of.get(id(tup)))
            self._record_agreement(obj_id, now)
            self.scene[obj_id] = {
                'label':      label,
                'x':          round(x, 4),
                'y':          round(y, 4),
                'z':          round(z, 4) if z is not None else None,
                'confidence': round(conf, 3),
                'last_seen':  now,
                'stale':      False,
                'reachable':  self._is_reachable(x, y, z),
                'color':      self._vote_colour(obj_id, col_of.get(id(tup)), conf),
            }

        # Once per 10 s, not once per detection message: this line alone wrote ~660,000 lines
        # (and, over a few days, 7 GB of ROS logs) and filled the disk.
        self.get_logger().info(
            f'Scene: {len(self.scene)} objects — '
            f'{[v["label"] for v in self.scene.values()]}',
            throttle_duration_sec=10.0,
        )

    AGREEMENT_WINDOW_S = 120.0
    AGREEMENT_MIN_N = 15

    def _record_agreement(self, obj_id, now):
        """Remember how far apart each pair of fresh static-camera readings of this object is. Two cameras that are both where they were calibrated agree within 2-4 cm; a camera that was bumped
        or moved by someone shifts every object it sees by centimetres, and the pair's median gap shows it long before a pick misses."""
        obs = self._obs.get(obj_id) or {}
        fresh = sorted(k for k, v in obs.items() if k in STATIC_SOURCES and now - v[0] <= SOURCE_WINDOW_S)
        for i, a in enumerate(fresh):
            for b in fresh[i + 1:]:
                gap = math.hypot(obs[a][1] - obs[b][1], obs[a][2] - obs[b][2])
                q = self._pair_gaps.setdefault(f'{a}|{b}', [])
                q.append((now, gap))
                while q and now - q[0][0] > self.AGREEMENT_WINDOW_S:
                    q.pop(0)

    def agreement_summary(self, now=None):
        """{'a|b': {'median_m', 'n'}} over the last AGREEMENT_WINDOW_S, only for pairs with enough samples."""
        now = time.time() if now is None else now
        out = {}
        for k, q in self._pair_gaps.items():
            vals = sorted(g for t, g in q if now - t <= self.AGREEMENT_WINDOW_S)
            if len(vals) >= self.AGREEMENT_MIN_N:
                out[k] = {'median_m': round(vals[len(vals) // 2], 4), 'n': len(vals)}
        return out

    def publish_agreement(self):
        self.agreement_pub.publish(String(data=json.dumps({'pairs': self.agreement_summary(), 'time': time.time()})))

    def _same_label_duplicates(self):
        """Ids of fresh tracks that are a duplicate of a more confident fresh track of the same label within
        SAME_LABEL_MERGE_M. A transparent bottle is read 5-9 cm apart by two cameras and tracked twice; a ghost
        track near a place target made the carry guard refuse. The more confident track is kept."""
        fresh = [(oid, o) for oid, o in self.scene.items() if not o['stale']]
        dups = set()
        for i, (a, oa) in enumerate(fresh):
            for b, ob in fresh[i + 1:]:
                if oa['label'] != ob['label'] or a in dups or b in dups:
                    continue
                if math.hypot(oa['x'] - ob['x'], oa['y'] - ob['y']) < SAME_LABEL_MERGE_BY_LABEL_M.get(oa['label'], SAME_LABEL_MERGE_M):
                    dups.add(a if oa['confidence'] < ob['confidence'] else b)
        return dups

    def _drop_blobs_on_known_objects(self, resolved):
        """A colour blob ('blue_obj', the HSV fallback detector) on top of a recognised object is that object seen a second time: it only adds a ghost to the scene (a 'red_obj' once sat where the
        white cup was). Blobs away from every recognised object are kept: they are the only way to see things YOLO has no class for. Existing ghost tracks are removed too."""
        def blob(label):
            return str(label).endswith('_obj')
        known = [(o[1], o[2]) for o in resolved if not blob(o[0])]
        known += [(o['x'], o['y']) for o in self.scene.values() if not blob(o['label']) and not o.get('stale')]
        def covered(x, y):
            return any(math.hypot(x - kx, y - ky) < BLOB_GHOST_M for kx, ky in known)
        for oid in [i for i, o in self.scene.items() if blob(o['label']) and covered(o['x'], o['y'])]:
            del self.scene[oid]
        return [o for o in resolved if not (blob(o[0]) and covered(o[1], o[2]))]

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

    def _on_joint_state(self, msg):
        try:
            self._finger = msg.position[list(msg.name).index('finger_joint')]
        except (ValueError, IndexError):
            pass

    def _in_held_object(self, point):
        f = self._finger
        if f is None or not (0.08 < f < 0.65):
            return False                      # not carrying anything (open, or closed on air)
        chain, fingers = self._arm_body()
        if not fingers:
            return False
        a, b = fingers[0][1], fingers[1][1]
        pad_mid = ((a[0] + b[0]) / 2, (a[1] + b[1]) / 2, (a[2] + b[2]) / 2)
        return sg.in_held_volume(point, pad_mid)

    def _on_robot(self, point):
        chain, fingers = self._arm_body()
        if chain is None:
            return False                      # no TF for the arm: cannot tell, keep the detection
        return sg.point_on_robot(point, chain, fingers) or self._in_held_object(point)

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
        dups = self._same_label_duplicates()

        for obj_id, obj in self.scene.items():
            if obj_id in dups:
                continue                      # a second track of an object already listed (two cameras, two depths)
            entry = {
                'label':      obj['label'],
                'x':          obj['x'],
                'y':          obj['y'],
                'z':          obj['z'],
                'confidence': obj['confidence'],
                'reachable':  obj['reachable'],
                'stale':      obj['stale'],
            }
            if obj.get('color') and obj['color'] != 'unknown':
                entry['color'] = obj['color']
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
            self._colour_votes.pop(obj_id, None)
            self._obs.pop(obj_id, None)

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

    def _fuse_position(self, obj_id, x, y, z, conf, now, source=None):
        """The published estimate of obj_id after this reading. Each camera's latest reading (within SOURCE_WINDOW_S) is kept per object; the estimate is their weighted mean
        (weight = confidence x SOURCE_WEIGHT; heights from the cameras whose depth is good when any saw it), then the published value moves SMOOTH_ALPHA of the way toward it
        (or jumps when it is more than JUMP_M away: the object was moved). So the value no longer flips between the cameras' slightly different readings message by message."""
        existing = self.scene.get(obj_id)
        src = source or 'unknown'
        if existing is None or existing['stale']:
            self._obs[obj_id] = {src: (now, x, y, z, conf)}
            return x, y, z, conf
        obs = self._obs.setdefault(obj_id, {})
        obs[src] = (now, x, y, z, conf)
        for k in [k for k, v in obs.items() if now - v[0] > SOURCE_WINDOW_S]:
            del obs[k]
        ws_ = {k: max(v[4], 0.05) * SOURCE_WEIGHT.get(k, 0.5) for k, v in obs.items()}
        tot = sum(ws_.values())
        mx = sum(ws_[k] * obs[k][1] for k in obs) / tot
        my = sum(ws_[k] * obs[k][2] for k in obs) / tot
        zs = {k: v for k, v in obs.items() if v[3] is not None}
        trusted = {k: v for k, v in zs.items() if k in SOURCE_Z_TRUSTED} or zs
        if trusted:
            zt = sum(ws_[k] for k in trusted)
            mz = sum(ws_[k] * trusted[k][3] for k in trusted) / zt
        else:
            mz = existing['z']
        if math.hypot(mx - existing['x'], my - existing['y']) > JUMP_M:
            fx, fy, fz = mx, my, (mz if mz is not None else existing['z'])
        else:
            a = SMOOTH_ALPHA
            fx = existing['x'] + a * (mx - existing['x'])
            fy = existing['y'] + a * (my - existing['y'])
            fz = mz if existing['z'] is None else (existing['z'] if mz is None else existing['z'] + a * (mz - existing['z']))
        return fx, fy, fz, max(conf, existing['confidence'])

    def _vote_colour(self, obj_id, colour, conf):
        """The colour of track obj_id: every detection votes with its confidence, older votes fade (x0.97 per detection), the winner is published. A single frame under a colour cast cannot flip it."""
        votes = self._colour_votes.setdefault(obj_id, {})
        for k in list(votes):
            votes[k] *= COLOUR_VOTE_DECAY
            if votes[k] < 1e-3:
                del votes[k]
        if colour and colour != 'unknown':
            votes[colour] = votes.get(colour, 0.0) + max(float(conf), 0.05)
        return self._colour_from_votes(votes)

    NEUTRAL_COLOURS = ('white', 'grey', 'black')

    @staticmethod
    def _colour_from_votes(votes):
        """The published colour from the weighted votes. A dark reading can HIDE a colour (RS2 saw a dark teal cup at brightness 12 and said black) but a clear colour cannot come from a black object, so
        a coloured name with 15 % of the weight beats the neutrals; among neutrals black counts 0.6. Otherwise the strongest wins."""
        if not votes:
            return None
        total = sum(votes.values())
        chroma = {k: w for k, w in votes.items() if k not in SceneGraphNode.NEUTRAL_COLOURS}
        if chroma:
            best = max(chroma, key=chroma.get)
            if chroma[best] >= 0.15 * total:
                return best
        neutral = {k: w * (0.6 if k == 'black' else 1.0) for k, w in votes.items() if k in SceneGraphNode.NEUTRAL_COLOURS}
        pool = neutral or votes
        return max(pool, key=pool.get)

    def _to_object_centre(self, det, label, x, y):
        """A static depth camera measures the NEAR surface of an object, but the arm needs its centre. Move the point horizontally away from the camera by the object's radius
        (2026-10-29: against the wrist camera, RS2 read cups, mice and a bowl 6-9 cm too close to itself = radius + 2 cm; its y was off by that, and RS1's x likewise). Height is left alone."""
        try:
            t = self.tf_buffer.lookup_transform(BASE_FRAME, det['frame_id'], rclpy.time.Time()).transform.translation
            dx, dy = x - t.x, y - t.y
            n = math.hypot(dx, dy)
            if n < 0.2:
                return x, y
            r = ws.OBJECT_RADIUS_M.get(label, ws.DEFAULT_RADIUS_M) * SURFACE_CENTRE_FACTOR
            return x + dx / n * r, y + dy / n * r
        except Exception:
            return x, y

    def _wrist_usable(self, det):
        """The wrist camera's readings are scene evidence only while it looks at the table (its axis points down) and the object is not too close for its depth."""
        try:
            if float(det.get('cz_3d', 0.0)) < WRIST_MIN_RANGE_M:
                return False
            t = self.tf_buffer.lookup_transform(BASE_FRAME, det['frame_id'], rclpy.time.Time()).transform.rotation
            axis_z = 1.0 - 2.0 * (t.x * t.x + t.y * t.y)            # z component of the camera's optical (z) axis in base_link
            return axis_z <= -WRIST_MIN_DOWN
        except Exception:
            return False

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
        if not ccf.has_calibration(str(det.get('source', ''))):
            return None          # a static camera without a calibration file must not place anything (a stale remembered transform would)

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
