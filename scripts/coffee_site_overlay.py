#!/usr/bin/env python3
"""
Draw the recommended coffee-machine footprint (and cup spot, drop-off) onto a live photo from a calibrated static camera, so a person can see where to put it. READ-ONLY.

    source scripts/ros_env.sh; ROS_DOMAIN_ID=42 python3 scripts/coffee_site_overlay.py --camera realsense --out docs/coffee/overlay_rs1.png

Points in base_link are projected with the camera's installed calibration (~/.ros/<camera>_calibration.yaml) and its camera_info. The calibration is good to about 2-4 cm, so use the overlay to SEE the place and a
tape measure (docs/coffee/coffee_placement_plan.png) for the final position.
"""
import argparse
import math
import os
import time
import urllib.request

import cv2
import numpy as np
import rclpy
import yaml
from rclpy.node import Node
from scipy.spatial.transform import Rotation
from sensor_msgs.msg import CameraInfo

CAM = {'realsense': ('/global_camera/global_camera/color/camera_info', 'realsense'), 'realsense2': ('/global_camera_2/global_camera_2/color/camera_info', 'realsense2')}
MACH_W, MACH_D, MACH_H = 0.208, 0.335, 0.318
CUP_FROM_FRONT_M = 0.10
TABLE_Z = -0.0125


def footprint(cup, front_deg):
    f = np.array([math.cos(math.radians(front_deg)), math.sin(math.radians(front_deg))])
    s = np.array([-f[1], f[0]])
    fc = np.array(cup) + f * CUP_FROM_FRONT_M
    bc = fc - f * MACH_D
    return [fc + s * MACH_W / 2, fc - s * MACH_W / 2, bc - s * MACH_W / 2, bc + s * MACH_W / 2]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--camera', default='realsense', choices=list(CAM))
    ap.add_argument('--cup', default='0.40,0.20')
    ap.add_argument('--front-deg', type=float, default=-90.0, help='direction the machine front faces (-90 = toward the robot\'s right / the centre line)')
    ap.add_argument('--serve', default='0.40,-0.25')
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    topic, name = CAM[a.camera]
    rclpy.init()
    n = Node('coffee_overlay')
    st = {}
    n.create_subscription(CameraInfo, topic, lambda m: st.__setitem__('K', np.array(m.k).reshape(3, 3)), 1)
    t0 = time.time()
    while 'K' not in st and time.time() - t0 < 8:
        rclpy.spin_once(n, timeout_sec=0.2)
    rclpy.shutdown()
    K = st['K']
    img = cv2.imdecode(np.frombuffer(urllib.request.urlopen(f'http://localhost:8000/api/camera/{a.camera}/snapshot.jpg', timeout=8).read(), np.uint8), cv2.IMREAD_COLOR)
    cal = yaml.safe_load(open(os.path.expanduser(f'~/.ros/{name}_calibration.yaml')))
    q, t = cal['rotation_quat'], cal['translation']
    R = Rotation.from_quat([q['x'], q['y'], q['z'], q['w']]).as_matrix()
    tv = np.array([t['x'], t['y'], t['z']])
    sx = img.shape[1] / (2.0 * K[0][2])                              # the snapshot may be scaled relative to camera_info
    Ks = K.copy()
    Ks[0] *= sx
    Ks[1] *= sx

    def px(p):
        pc = R.T @ (np.asarray(p, float) - tv)
        if pc[2] <= 0.05:
            return None
        u = Ks[0][0] * pc[0] / pc[2] + Ks[0][2]
        v = Ks[1][1] * pc[1] / pc[2] + Ks[1][2]
        return int(round(u)), int(round(v))

    def poly(pts_xy, z, colour, thick=3, closed=True):
        P = [px([x, y, z]) for x, y in pts_xy]
        if None in P:
            return
        cv2.polylines(img, [np.array(P, np.int32)], closed, colour, thick, cv2.LINE_AA)

    cup = [float(v) for v in a.cup.split(',')]
    srv = [float(v) for v in a.serve.split(',')]
    fp = footprint(cup, a.front_deg)
    RED, GREEN, BLUE, YEL = (40, 40, 220), (60, 190, 60), (230, 130, 30), (0, 200, 240)
    ov = img.copy()
    P = [px([x, y, TABLE_Z]) for x, y in fp]
    if None not in P:
        cv2.fillPoly(ov, [np.array(P, np.int32)], (90, 90, 230))
        img[:] = cv2.addWeighted(ov, 0.35, img, 0.65, 0)
    poly(fp, TABLE_Z, RED, 4)                                         # footprint on the table
    poly(fp, TABLE_Z + MACH_H, RED, 2)                                # top of the machine
    for x, y in fp:                                                   # vertical edges
        p0, p1 = px([x, y, TABLE_Z]), px([x, y, TABLE_Z + MACH_H])
        if p0 and p1:
            cv2.line(img, p0, p1, RED, 2, cv2.LINE_AA)
    f = np.array([math.cos(math.radians(a.front_deg)), math.sin(math.radians(a.front_deg))])
    front_a, front_b = fp[0], fp[1]
    pa, pb = px([front_a[0], front_a[1], TABLE_Z]), px([front_b[0], front_b[1], TABLE_Z])
    if pa and pb:
        cv2.line(img, pa, pb, YEL, 7, cv2.LINE_AA)                    # the FRONT edge in yellow
        mid = ((pa[0] + pb[0]) // 2, (pa[1] + pb[1]) // 2)
        cv2.putText(img, 'FRONT (open side)', (mid[0] - 110, mid[1] + 34), cv2.FONT_HERSHEY_SIMPLEX, 0.8, YEL, 2, cv2.LINE_AA)
    ang = np.linspace(0, 2 * math.pi, 40)
    ring = [(cup[0] + 0.045 * math.cos(t_), cup[1] + 0.045 * math.sin(t_)) for t_ in ang]
    poly(ring, TABLE_Z, (255, 255, 255), 3)
    c = px([cup[0], cup[1], TABLE_Z])
    if c:
        cv2.putText(img, 'cup spot', (c[0] - 40, c[1] - 26), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    d = -f                                                            # the gripper comes in along -front
    start = np.array(cup) + f * 0.30
    pa2, pb2 = px([start[0], start[1], TABLE_Z + 0.07]), px([cup[0] + f[0] * 0.07, cup[1] + f[1] * 0.07, TABLE_Z + 0.07])
    if pa2 and pb2:
        cv2.arrowedLine(img, pa2, pb2, BLUE, 4, cv2.LINE_AA, tipLength=0.15)
        cv2.putText(img, 'gripper comes in here', (pa2[0] - 120, pa2[1] - 46), cv2.FONT_HERSHEY_SIMPLEX, 0.7, BLUE, 2, cv2.LINE_AA)
    ring2 = [(srv[0] + 0.06 * math.cos(t_), srv[1] + 0.06 * math.sin(t_)) for t_ in ang]
    poly(ring2, TABLE_Z, GREEN, 4)
    s = px([srv[0], srv[1], TABLE_Z])
    if s:
        cv2.putText(img, 'DROP-OFF', (s[0] - 55, s[1] + 62), cv2.FONT_HERSHEY_SIMPLEX, 0.8, GREEN, 2, cv2.LINE_AA)
    b = px([0.0, 0.0, TABLE_Z])
    if b:
        cv2.circle(img, b, 7, (255, 255, 255), -1)
        cv2.putText(img, 'robot base centre', (b[0] + 12, b[1] + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    for xr in (0.2, 0.3, 0.4, 0.5, 0.6):                              # distance ruler along the centre line
        p = px([xr, 0.0, TABLE_Z])
        if p:
            cv2.circle(img, p, 4, (255, 255, 255), -1)
            cv2.putText(img, f'{int(xr * 100)} cm', (p[0] + 8, p[1] + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(img, 'Coffee machine placement (outline from the camera calibration, about +-3 cm)', (20, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 3, cv2.LINE_AA)
    cv2.putText(img, 'Coffee machine placement (outline from the camera calibration, about +-3 cm)', (20, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (30, 30, 30), 1, cv2.LINE_AA)
    cv2.imwrite(a.out, img)
    print('wrote', a.out, img.shape)


if __name__ == '__main__':
    main()
