#!/usr/bin/env python3
"""
Multi-camera fusion node.
Merges /detections (OAK-D overhead) + /detections_side (RealSense)
into /detections_fused with best 3D coordinates.
"""
import rclpy, json, threading, time
from rclpy.node import Node
from std_msgs.msg import String

DEPTH_MIN = 0.15
DEPTH_MAX = 2.5
MATCH_DIST = 0.15   # metres — detections within 15cm are same object
STALE_SEC  = 2.0    # drop detections older than 2s

class FusionNode(Node):
    def __init__(self):
        super().__init__('fusion_node')
        self._lock = threading.Lock()
        self._oak_dets  = []
        self._rs_dets   = []
        self._oak_time  = 0.0
        self._rs_time   = 0.0

        self.create_subscription(
            String, '/detections', self._oak_cb, 10)
        self.create_subscription(
            String, '/detections_side', self._rs_cb, 10)

        self.fused_pub = self.create_publisher(
            String, '/detections_fused', 10)

        self.create_timer(0.1, self._fuse)
        self.get_logger().info(
            'Fusion node ready — merging OAK-D + RealSense')

    def _oak_cb(self, msg):
        try:
            with self._lock:
                self._oak_dets = json.loads(msg.data)
                self._oak_time = time.time()
        except: pass

    def _rs_cb(self, msg):
        try:
            with self._lock:
                self._rs_dets = json.loads(msg.data)
                self._rs_time = time.time()
        except: pass

    def _depth_reliable(self, det):
        cz = det.get('cz_3d', 0)
        return DEPTH_MIN < cz < DEPTH_MAX

    def _score(self, det):
        """Higher = more reliable detection."""
        score = det.get('confidence', 0) * 10
        cz = det.get('cz_3d', 0)
        # Prefer detections where depth is close to table level
        if 0.5 < cz < 1.8:
            score += 5
        # Prefer RealSense for close objects (better depth alignment)
        if det.get('source') == 'realsense' and cz < 1.0:
            score += 3
        # Prefer OAK-D for far objects
        if det.get('source') == 'oakd' and cz > 1.0:
            score += 2
        return score

    def _same_object(self, a, b):
        """Check if two detections are the same object."""
        if a.get('label') != b.get('label'):
            return False
        # Compare camera-frame positions
        da = (a.get('cx_3d',0), a.get('cy_3d',0), a.get('cz_3d',0))
        db = (b.get('cx_3d',0), b.get('cy_3d',0), b.get('cz_3d',0))
        dist = ((da[0]-db[0])**2 + (da[1]-db[1])**2)**0.5
        return dist < MATCH_DIST

    def _fuse(self):
        now = time.time()
        with self._lock:
            oak = list(self._oak_dets) if now-self._oak_time < STALE_SEC else []
            rs  = list(self._rs_dets)  if now-self._rs_time  < STALE_SEC else []

        if not oak and not rs:
            return

        # Tag sources
        for d in oak: d['source'] = 'oakd'
        for d in rs:  d['source'] = 'realsense'

        fused = []
        used_rs = set()

        for oak_det in oak:
            if not self._depth_reliable(oak_det):
                # Find matching RealSense detection
                best_rs = None
                best_score = -1
                for i, rs_det in enumerate(rs):
                    if i in used_rs:
                        continue
                    if rs_det.get('label') == oak_det.get('label'):
                        s = self._score(rs_det)
                        if s > best_score:
                            best_score = s
                            best_rs = (i, rs_det)
                if best_rs:
                    used_rs.add(best_rs[0])
                    merged = oak_det.copy()
                    merged.update({
                        'cx_3d':   best_rs[1]['cx_3d'],
                        'cy_3d':   best_rs[1]['cy_3d'],
                        'cz_3d':   best_rs[1]['cz_3d'],
                        'x_robot': best_rs[1].get('x_robot', oak_det.get('x_robot',0)),
                        'y_robot': best_rs[1].get('y_robot', oak_det.get('y_robot',0)),
                        'z_robot': best_rs[1].get('z_robot', oak_det.get('z_robot',0)),
                        'source':  'fused_rs',
                    })
                    fused.append(merged)
                else:
                    fused.append(oak_det)
            else:
                # OAK-D depth is fine — check if RealSense has better reading
                best_rs = None
                best_score = self._score(oak_det)
                for i, rs_det in enumerate(rs):
                    if i in used_rs:
                        continue
                    if self._same_object(oak_det, rs_det):
                        s = self._score(rs_det)
                        if s > best_score:
                            best_score = s
                            best_rs = (i, rs_det)
                if best_rs:
                    used_rs.add(best_rs[0])
                    merged = oak_det.copy()
                    merged.update({
                        'cx_3d':   best_rs[1]['cx_3d'],
                        'cy_3d':   best_rs[1]['cy_3d'],
                        'cz_3d':   best_rs[1]['cz_3d'],
                        'source':  'fused_both',
                    })
                    fused.append(merged)
                else:
                    fused.append(oak_det)

        # Add RealSense-only detections not matched to OAK-D
        for i, rs_det in enumerate(rs):
            if i not in used_rs and self._depth_reliable(rs_det):
                fused.append(rs_det)

        if fused:
            self.fused_pub.publish(String(data=json.dumps(fused)))
            sources = [d.get('source','?') for d in fused]
            self.get_logger().info(
                f'Fused {len(fused)} objects: {sources}',
                throttle_duration_sec=2.0)

def main(args=None):
    rclpy.init(args=args)
    node = FusionNode()
    rclpy.spin(node)

if __name__ == '__main__':
    main()
