#!/usr/bin/env python3
"""
record_shared_objects.py -- log where RS1 and RS2 (or any two static cameras) put the SAME objects. READ-ONLY.

    python3 calibration/record_shared_objects.py --a realsense --b realsense2 [--secs 8] [--set-name set1]

For ~secs seconds it collects both cameras' detections (/detections_side = RS1, /detections_realsense2, /detections_realsense3), turns them into base_link through the CURRENT calibrations, takes the median
of every steady cluster per label, pairs clusters of the same label that lie within --max-pair-m of each other, and appends the pairs to ~/.ros/calibration_candidates/shared_object_pairs.jsonl for
calibration/fit_camera_to_reference.py. Put objects at clearly different places, then move them and record again.
"""
import argparse
import collections
import json
import os
import time

import numpy as np
import rclpy
import tf2_ros
from rclpy.node import Node
from scipy.spatial.transform import Rotation as R
from std_msgs.msg import String

TOPIC = {'realsense': '/detections_side', 'realsense2': '/detections_realsense2', 'realsense3': '/detections_realsense3'}
OUT = os.path.expanduser('~/.ros/calibration_candidates/shared_object_pairs.jsonl')
TABLE_Z = (-0.05, 0.20)          # the objects stand on the table: anything else is the arm / the room


def clusters(points, cell=0.08, min_n=8):
    """points: [(x, y, z, conf)] -> list of (median x, y, z, n) of steady clusters (cells of `cell` m)."""
    grid = collections.defaultdict(list)
    for p in points:
        grid[(round(p[0] / cell), round(p[1] / cell))].append(p)
    out = []
    for pts in grid.values():
        if len(pts) >= min_n:
            a = np.array(pts)
            out.append((float(np.median(a[:, 0])), float(np.median(a[:, 1])), float(np.median(a[:, 2])), len(pts), float(np.std(a[:, 0]) + np.std(a[:, 1]))))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--a', default='realsense')
    ap.add_argument('--b', default='realsense2')
    ap.add_argument('--secs', type=float, default=8.0)
    ap.add_argument('--max-pair-m', type=float, default=0.25)
    ap.add_argument('--set-name', default=time.strftime('%H%M%S'))
    ap.add_argument('--no-save', action='store_true')
    args = ap.parse_args()
    rclpy.init()
    n = Node('record_shared_objects')
    buf = tf2_ros.Buffer()
    tf2_ros.TransformListener(buf, n)
    rec = {args.a: collections.defaultdict(list), args.b: collections.defaultdict(list)}

    def cb(m, cam):
        for d in json.loads(m.data):
            try:
                x = buf.lookup_transform('base_link', d['frame_id'], rclpy.time.Time()).transform
                p = R.from_quat([x.rotation.x, x.rotation.y, x.rotation.z, x.rotation.w]).apply([d['cx_3d'], d['cy_3d'], d['cz_3d']]) + [x.translation.x, x.translation.y, x.translation.z]
            except Exception:
                continue
            if TABLE_Z[0] <= p[2] <= TABLE_Z[1] and 0.0 <= p[0] <= 0.9 and -0.7 <= p[1] <= 0.6 and d['confidence'] >= 0.45:
                rec[cam][d['label']].append((float(p[0]), float(p[1]), float(p[2]), float(d['confidence'])))
    for cam in (args.a, args.b):
        n.create_subscription(String, TOPIC[cam], lambda m, c=cam: cb(m, c), 10)
    t0 = time.time()
    while time.time() - t0 < args.secs:
        rclpy.spin_once(n, timeout_sec=0.05)
    pairs = []
    for label, pts_a in rec[args.a].items():
        ca, cb_ = clusters(pts_a), clusters(rec[args.b].get(label, []))
        cand = sorted(((np.hypot(a[0] - b[0], a[1] - b[1]), i, j) for i, a in enumerate(ca) for j, b in enumerate(cb_)), key=lambda t: t[0])
        used_a, used_b = set(), set()
        for dist, i, j in cand:                       # nearest first, every detection used once: two readings of one object are not two objects
            if dist <= args.max_pair_m and i not in used_a and j not in used_b:
                used_a.add(i)
                used_b.add(j)
                pairs.append({'label': label, args.a: [round(v, 4) for v in ca[i][:3]], args.b: [round(v, 4) for v in cb_[j][:3]]})
    print(f'{len(pairs)} shared object(s) between {args.a} and {args.b}:')
    for p in pairs:
        a, b = np.array(p[args.a]), np.array(p[args.b])
        print(f"  {p['label']:8s} {args.a} ({a[0]:.3f},{a[1]:.3f},{a[2]:.3f})  {args.b} ({b[0]:.3f},{b[1]:.3f},{b[2]:.3f})  difference ({100*(a[0]-b[0]):+.1f}, {100*(a[1]-b[1]):+.1f}, {100*(a[2]-b[2]):+.1f}) cm")
    if pairs and not args.no_save:
        os.makedirs(os.path.dirname(OUT), exist_ok=True)
        with open(OUT, 'a') as f:
            f.write(json.dumps({'set': args.set_name, 'iso': time.strftime('%Y-%m-%dT%H:%M:%S'), 'a': args.a, 'b': args.b, 'pairs': pairs}) + '\n')
        print('saved to', OUT)
    rclpy.shutdown()


if __name__ == '__main__':
    main()
