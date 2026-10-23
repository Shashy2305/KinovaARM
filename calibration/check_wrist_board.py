#!/usr/bin/env python3
"""
check_wrist_board.py -- is the wrist camera's pose (and the board size) right? READ-ONLY; only the wrist camera, the arm's TF and a board that does NOT move.

    source /mnt/ros_workspace/Shashproject/scripts/ros_env.sh
    python3 calibration/check_wrist_board.py [--n 8] [--timeout 420]

Put the ChArUco board flat on the table and keep it still. Move the arm (the dashboard jog) so the wrist camera sees the board from DIFFERENT places and angles (at least a few
cm / 10 degrees apart, and tilt it a little too). Each time the arm has been still for a moment and the board is seen, one snapshot is taken. For a fixed board the same point of it
must come out at the same place in base_link every time:

    board_origin_in_base_i  =  p_i  +  s * R_i * t_i          p_i, R_i: the wrist camera's pose in base_link (forward kinematics + the URDF camera mount)
                                                              t_i: the board's position in the camera, from the ChArUco corners (with the board size the CODE assumes)

  * the scatter of that point over the snapshots (mm) is the combined error of the wrist camera's mounting pose and its intrinsics;
  * s is fitted from the same data: s * (assumed square) = the board's TRUE square size, measured by nothing but the wrist camera and the arm. If it matches the ruler (39 mm)
    the board size and the distance scale are right; if not, the wrist intrinsics or the camera mount are off by that factor;
  * the board's ORIENTATION must also agree between snapshots (degrees).
"""
import argparse
import os
import sys
import time

import cv2
import numpy as np
import rclpy
import tf2_ros
from cv_bridge import CvBridge
from rclpy.node import Node
from scipy.spatial.transform import Rotation
from sensor_msgs.msg import CameraInfo, Image

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__))))
from markers.charuco_detector import SQUARE_M, detect_pose, make_board_and_detector  # noqa: E402

IMAGE_TOPIC = '/camera/color/image_raw'
INFO_TOPIC = '/camera/color/camera_info'
MIN_MOVE_M = 0.03
MIN_MOVE_DEG = 8.0
STILL_M, STILL_DEG = 0.0005, 0.1


def fit_scale_and_point(P, U):
    """Least squares for s and b in  b = p_i + s * u_i  over the snapshots. P: (n,3) camera positions in base, U: (n,3) = R_i t_i. Returns (s, b, residuals (n,3))."""
    n = len(P)
    A = np.zeros((3 * n, 4))
    y = np.zeros(3 * n)
    for i in range(n):
        A[3 * i:3 * i + 3, 0] = -U[i]
        A[3 * i:3 * i + 3, 1:] = np.eye(3)
        y[3 * i:3 * i + 3] = P[i]
    x, *_ = np.linalg.lstsq(A, y, rcond=None)
    s, b = x[0], x[1:]
    res = (A @ x - y).reshape(n, 3)
    return float(s), b, res


def analyse(snaps, assumed_square_m=SQUARE_M):
    """snaps: list of (p (3,), R (3,3) camera pose in base, rvec, tvec board in camera). Returns a dict of results."""
    P = np.array([s[0] for s in snaps])
    U = np.array([s[1] @ s[3].reshape(3) for s in snaps])
    out = {'n': len(snaps)}
    # with the size the code assumes (s = 1)
    pts1 = P + U
    out['scatter_mm_assumed'] = float(1000 * np.sqrt(np.mean(np.sum((pts1 - pts1.mean(axis=0)) ** 2, axis=1))))
    s, b, res = fit_scale_and_point(P, U)
    out['scale'] = s
    out['true_square_mm'] = s * assumed_square_m * 1000
    out['scatter_mm_fitted'] = float(1000 * np.sqrt(np.mean(np.sum(res ** 2, axis=1))))
    out['per_snapshot_mm'] = [round(float(1000 * np.linalg.norm(r)), 1) for r in res]
    out['board_origin_base'] = [round(float(v), 4) for v in b]
    # orientation of the board in base_link for each snapshot
    Rb = Rotation.concatenate([Rotation.from_matrix(sn[1] @ cv2.Rodrigues(sn[2])[0]) for sn in snaps])
    mean = Rb.mean()
    out['orientation_spread_deg'] = float(np.degrees(np.sqrt(np.mean([(mean.inv() * r).magnitude() ** 2 for r in Rb]))))
    return out


class Collector(Node):
    def __init__(self):
        super().__init__('check_wrist_board')
        self.buf = tf2_ros.Buffer()
        tf2_ros.TransformListener(self.buf, self)
        self.bridge = CvBridge()
        self.gray = self.K = self.dist = self.frame = None
        self.create_subscription(Image, IMAGE_TOPIC, self._img, 1)
        self.create_subscription(CameraInfo, INFO_TOPIC, self._info, 1)

    def _img(self, msg):
        self.gray = cv2.cvtColor(self.bridge.imgmsg_to_cv2(msg, 'bgr8'), cv2.COLOR_BGR2GRAY)
        self.frame = msg.header.frame_id

    def _info(self, msg):
        self.K = np.array(msg.k).reshape(3, 3)
        self.dist = np.array(msg.d)

    def cam_pose(self):
        t = self.buf.lookup_transform('base_link', self.frame, rclpy.time.Time()).transform
        return (np.array([t.translation.x, t.translation.y, t.translation.z]),
                Rotation.from_quat([t.rotation.x, t.rotation.y, t.rotation.z, t.rotation.w]).as_matrix())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, default=8)
    ap.add_argument('--timeout', type=float, default=420)
    args = ap.parse_args()
    rclpy.init()
    node = Collector()
    board, detector = make_board_and_detector()
    print(f'assumed square {SQUARE_M * 1000:.1f} mm; waiting for {args.n} snapshots from different arm poses (timeout {args.timeout:.0f} s)...', flush=True)
    snaps, last_pose, still_ref, still_since = [], None, None, None
    t_end = time.time() + args.timeout
    while time.time() < t_end and len(snaps) < args.n:
        rclpy.spin_once(node, timeout_sec=0.1)
        if node.gray is None or node.K is None:
            continue
        try:
            p, R = node.cam_pose()
        except Exception:
            continue
        if still_ref is None or np.linalg.norm(p - still_ref[0]) > STILL_M or np.degrees((Rotation.from_matrix(R) * Rotation.from_matrix(still_ref[1]).inv()).magnitude()) > STILL_DEG:
            still_ref, still_since = (p, R), time.time()
            continue
        if time.time() - still_since < 1.0:
            continue
        if last_pose is not None:
            moved = np.linalg.norm(p - last_pose[0])
            turned = np.degrees((Rotation.from_matrix(R) * Rotation.from_matrix(last_pose[1]).inv()).magnitude())
            if moved < MIN_MOVE_M and turned < MIN_MOVE_DEG:
                continue
        det = detect_pose(detector, board, node.gray, node.K, node.dist)
        if det is None:
            continue
        rvec, tvec = det
        snaps.append((p, R, rvec, tvec))
        last_pose = (p, R)
        print(f'  snapshot {len(snaps)}/{args.n}: camera at {np.round(p, 3).tolist()}, board {np.linalg.norm(tvec):.3f} m away', flush=True)
        if len(snaps) >= 3:
            r = analyse(snaps)
            print(f'     so far: fitted square {r["true_square_mm"]:.1f} mm, scatter {r["scatter_mm_fitted"]:.1f} mm (with the assumed size {r["scatter_mm_assumed"]:.1f} mm)', flush=True)
        still_since = time.time() + 2.0        # let the operator move on before the next one
    if len(snaps) < 3:
        print(f'only {len(snaps)} snapshots: not enough to judge (need >= 3, ideally 6+)')
        rclpy.shutdown()
        return
    r = analyse(snaps)
    print('\n==== wrist camera / board result ====')
    print(f'snapshots                : {r["n"]}')
    print(f'board square (code)      : {SQUARE_M * 1000:.1f} mm')
    print(f'board square (fitted)    : {r["true_square_mm"]:.1f} mm   (scale {r["scale"]:.3f}; compare with your ruler)')
    print(f'scatter of the board point, assumed size : {r["scatter_mm_assumed"]:.1f} mm')
    print(f'scatter of the board point, fitted size  : {r["scatter_mm_fitted"]:.1f} mm   per snapshot {r["per_snapshot_mm"]}')
    print(f'board orientation spread : {r["orientation_spread_deg"]:.2f} deg')
    print(f'board origin in base_link: {r["board_origin_base"]}')
    rclpy.shutdown()


if __name__ == '__main__':
    main()
