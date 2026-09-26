#!/usr/bin/env python3
"""
Hand-Eye Calibration  --  Eye-on-Base
Camera : OAK-D Pro Wide (fixed, overhead)
Robot  : Kinova Gen3 7DOF

Workflow (same as MathWorks example):
  1. Move arm so ChArUco board is visible (green border)
  2. Press SPACE to save image + EE pose  (one sample)
  3. Move arm to a NEW pose and repeat 15+ times
  4. Press C to detect board in all saved images and compute result
  5. Copy printed values into robot_launch_fixed.py calibration_tf block

Run order:
  Terminal 1: ros2 launch ... robot_launch_fixed.py robot_ip:=192.168.1.10
  Terminal 2: python3 oak_camera_node.py
  Terminal 3: python3 handeye_calibration.py
"""

import rclpy
from rclpy.node import Node
from rclpy.duration import Duration
import tf2_ros
import numpy as np
import cv2
import yaml, json, os, threading, time, datetime

from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge

# ── CONFIG ─────────────────────────────────────────────────────────────────────
IMAGE_TOPIC       = '/global_camera/color/image_raw'
CAMERA_INFO_TOPIC = '/global_camera/color/camera_info'
ROBOT_BASE        = 'base_link'
ROBOT_EE          = 'bracelet_link'

SQUARES_X   = 5
SQUARES_Y   = 7
SQUARE_M    = 0.040    # metres — must match your printed board
MARKER_M    = 0.020    # metres — must match your printed board
ARUCO_DICT  = cv2.aruco.DICT_6X6_250
MIN_CORNERS = 6

MIN_MOVE_CM  = 2.0     # refuse capture if arm moved less than this
MIN_MOVE_DEG = 5.0
# ──────────────────────────────────────────────────────────────────────────────


def _tf_to_Rt(tf):
    q  = tf.transform.rotation
    tr = tf.transform.translation
    qx, qy, qz, qw = q.x, q.y, q.z, q.w
    R = np.array([
        [1-2*(qy**2+qz**2), 2*(qx*qy-qz*qw),  2*(qx*qz+qy*qw)],
        [2*(qx*qy+qz*qw),  1-2*(qx**2+qz**2),  2*(qy*qz-qx*qw)],
        [2*(qx*qz-qy*qw),   2*(qy*qz+qx*qw),  1-2*(qx**2+qy**2)],
    ], dtype=np.float64)
    t = np.array([[tr.x], [tr.y], [tr.z]])
    return R, t


def _R_to_quat(R):
    tr = R[0,0]+R[1,1]+R[2,2]
    if tr > 0:
        s=0.5/np.sqrt(tr+1.0)
        return (R[2,1]-R[1,2])*s,(R[0,2]-R[2,0])*s,(R[1,0]-R[0,1])*s,0.25/s
    elif R[0,0]>R[1,1] and R[0,0]>R[2,2]:
        s=2.0*np.sqrt(1.0+R[0,0]-R[1,1]-R[2,2])
        return 0.25*s,(R[0,1]+R[1,0])/s,(R[0,2]+R[2,0])/s,(R[2,1]-R[1,2])/s
    elif R[1,1]>R[2,2]:
        s=2.0*np.sqrt(1.0+R[1,1]-R[0,0]-R[2,2])
        return (R[0,1]+R[1,0])/s,0.25*s,(R[1,2]+R[2,1])/s,(R[0,2]-R[2,0])/s
    else:
        s=2.0*np.sqrt(1.0+R[2,2]-R[0,0]-R[1,1])
        return (R[0,2]+R[2,0])/s,(R[1,2]+R[2,1])/s,0.25*s,(R[1,0]-R[0,1])/s


# ── ROS collector node ────────────────────────────────────────────────────────
class Collector(Node):
    def __init__(self):
        super().__init__('handeye_calibration')
        self.bridge     = CvBridge()
        self.tf_buf     = tf2_ros.Buffer()
        self.tf_listen  = tf2_ros.TransformListener(self.tf_buf, self)
        self._lock      = threading.Lock()
        self._img       = None
        self._K         = None
        self._dist      = None

        # ArUco setup (created once, reused every frame)
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(ARUCO_DICT)
        self.board      = cv2.aruco.CharucoBoard_create(
            SQUARES_X, SQUARES_Y, SQUARE_M, MARKER_M, self.aruco_dict)
        self.det_params = cv2.aruco.DetectorParameters_create()
        self.det_params.minMarkerPerimeterRate = 0.01

        # Collected data
        self.images    = []    # raw BGR frames
        self.ee_poses  = []    # (R_ee, t_ee) at capture time
        self.ee_xyz    = []    # for display only
        self._last_R   = None
        self._last_t   = None

        ts = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        self.session = os.path.expanduser(f'~/handeye_session_{ts}')
        os.makedirs(self.session, exist_ok=True)

        self.create_subscription(Image,      IMAGE_TOPIC,       self._cb_img,  10)
        self.create_subscription(CameraInfo, CAMERA_INFO_TOPIC, self._cb_info, 10)

    def _cb_img(self, msg):
        with self._lock:
            self._img = self.bridge.imgmsg_to_cv2(msg, 'bgr8')

    def _cb_info(self, msg):
        with self._lock:
            if self._K is None:
                self._K    = np.array(msg.k).reshape(3, 3)
                self._dist = np.array(msg.d)
                print(f'[Camera] intrinsics received  fx={self._K[0,0]:.1f}')

    def get_frame(self):
        with self._lock:
            img  = self._img.copy()  if self._img  is not None else None
            K    = self._K.copy()    if self._K    is not None else None
            dist = self._dist.copy() if self._dist is not None else None
        return img, K, dist

    def detect(self, gray, K, dist):
        corners, ids, _ = cv2.aruco.detectMarkers(
            gray, self.aruco_dict, parameters=self.det_params)
        if ids is None or len(ids) == 0:
            return None
        _, ch_c, ch_ids = cv2.aruco.interpolateCornersCharuco(
            corners, ids, gray, self.board, cameraMatrix=K, distCoeffs=dist)
        if ch_ids is None or len(ch_ids) < MIN_CORNERS:
            return None
        ok, rvec, tvec = cv2.aruco.estimatePoseCharucoBoard(
            ch_c, ch_ids, self.board, K, dist, None, None)
        if not ok:
            return None
        return ch_c, ch_ids, rvec, tvec

    def capture(self):
        """Save current image + EE pose. Returns (ok, message)."""
        img, K, dist = self.get_frame()
        if img is None: return False, 'No image yet'
        if K   is None: return False, 'No camera_info yet'

        try:
            tf = self.tf_buf.lookup_transform(
                ROBOT_BASE, ROBOT_EE, rclpy.time.Time(),
                timeout=Duration(seconds=2.0))
        except Exception as e:
            return False, f'Robot TF failed: {e}'

        R_ee, t_ee = _tf_to_Rt(tf)
        tr = tf.transform.translation

        # Refuse if arm has not moved enough since last sample
        if self._last_R is not None:
            dt_cm  = float(np.linalg.norm(t_ee - self._last_t)) * 100.0
            dR     = self._last_R.T @ R_ee
            da_deg = float(np.degrees(np.arccos(
                np.clip((np.trace(dR) - 1.0) / 2.0, -1.0, 1.0))))
            if dt_cm < MIN_MOVE_CM and da_deg < MIN_MOVE_DEG:
                return False, (
                    f'MOVE ARM FIRST -- only {dt_cm:.1f}cm / {da_deg:.1f}deg '
                    f'from last sample (need >{MIN_MOVE_CM:.0f}cm or >{MIN_MOVE_DEG:.0f}deg)')

        n = len(self.images) + 1
        self.images.append(img.copy())
        self.ee_poses.append((R_ee.copy(), t_ee.copy()))
        self.ee_xyz.append((float(tr.x), float(tr.y), float(tr.z)))
        self._last_R = R_ee.copy()
        self._last_t = t_ee.copy()

        cv2.imwrite(os.path.join(self.session, f'image_{n:03d}.jpg'), img)
        with open(os.path.join(self.session, f'pose_{n:03d}.json'), 'w') as f:
            json.dump({'ee_xyz': [float(tr.x), float(tr.y), float(tr.z)]}, f)

        return True, (f'Sample {n} saved  |  '
                      f'EE x={tr.x:.3f} y={tr.y:.3f} z={tr.z:.3f}')


# ── Solver (runs offline after all images collected) ──────────────────────────
def solve(node):
    n    = len(node.images)
    K    = node._K
    dist = node._dist
    if n < 4:
        return None, f'Need at least 4 samples (have {n})'

    print(f'\nDetecting ChArUco in {n} saved images...')

    R_b2g, t_b2g = [], []   # R_base2gripper (eye-on-base inverted EE pose)
    R_t2c, t_t2c = [], []   # R_board2cam    (from ChArUco detection)

    for i, (img, (R_ee, t_ee)) in enumerate(zip(node.images, node.ee_poses)):
        gray   = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        result = node.detect(gray, K, dist)
        if result is None:
            print(f'  image {i+1:03d}: detection failed -- skipping')
            continue
        ch_c, ch_ids, rvec, tvec = result
        R_board, _ = cv2.Rodrigues(rvec)

        # Eye-on-base: invert EE pose before passing to calibrateHandEye
        R_b2g.append(R_ee.T)
        t_b2g.append(-R_ee.T @ t_ee)
        R_t2c.append(R_board)
        t_t2c.append(tvec.reshape(3, 1))
        print(f'  image {i+1:03d}: OK  ({len(ch_ids)} corners)')

    n_valid = len(R_b2g)
    print(f'\n{n_valid}/{n} images valid')
    if n_valid < 4:
        return None, f'Only {n_valid} valid detections -- need at least 4'

    methods = {
        'TSAI':    cv2.CALIB_HAND_EYE_TSAI,
        'PARK':    cv2.CALIB_HAND_EYE_PARK,
        'HORAUD':  cv2.CALIB_HAND_EYE_HORAUD,
        'ANDREFF': cv2.CALIB_HAND_EYE_ANDREFF,
    }
    results, best_R, best_t, best_name = {}, None, None, None

    for name, method in methods.items():
        try:
            R_, t_ = cv2.calibrateHandEye(
                R_b2g, t_b2g, R_t2c, t_t2c, method=method)
            if np.any(np.isnan(t_)) or np.linalg.norm(t_) > 5.0:
                results[name] = 'diverged'; continue
            results[name] = {'x': float(t_[0,0]),
                             'y': float(t_[1,0]),
                             'z': float(t_[2,0])}
            if best_R is None:
                best_R, best_t, best_name = R_, t_, name
        except Exception as e:
            results[name] = str(e)

    if best_R is None:
        return None, 'All methods diverged -- collect more varied poses'

    tx, ty, tz = float(best_t[0,0]), float(best_t[1,0]), float(best_t[2,0])
    qx, qy, qz, qw = [float(v) for v in _R_to_quat(best_R)]

    out = {
        'calibration_type': 'eye_on_base',
        'parent_frame':     ROBOT_BASE,
        'child_frame':      'global_camera_link',
        'num_samples':      n_valid,
        'method_used':      best_name,
        'translation':      {'x': tx, 'y': ty, 'z': tz},
        'rotation_quat':    {'x': qx, 'y': qy, 'z': qz, 'w': qw},
        'rotation_matrix':  best_R.tolist(),
        'all_methods':      results,
    }
    ros_path = os.path.expanduser('~/.ros/handeye_calibration.yaml')
    os.makedirs(os.path.dirname(ros_path), exist_ok=True)
    for p in (ros_path, os.path.join(node.session, 'result.yaml')):
        with open(p, 'w') as f:
            yaml.dump(out, f, default_flow_style=False)

    summary = '\n'.join([
        '',
        f'===== RESULT  ({n_valid} samples, method={best_name}) =====',
        f'Translation : x={tx:.5f}  y={ty:.5f}  z={tz:.5f}',
        f'Quaternion  : x={qx:.5f}  y={qy:.5f}  z={qz:.5f}  w={qw:.5f}',
        '',
        'Copy into robot_launch_fixed.py  calibration_tf arguments:',
        f'  "--x","{tx:.6f}","--y","{ty:.6f}","--z","{tz:.6f}",',
        f'  "--qx","{qx:.6f}","--qy","{qy:.6f}","--qz","{qz:.6f}","--qw","{qw:.6f}"',
        '',
        'All methods:',
    ] + [
        f'  {k:8s}: x={v["x"]:.4f} y={v["y"]:.4f} z={v["z"]:.4f}' if isinstance(v, dict)
        else f'  {k:8s}: {v}'
        for k, v in results.items()
    ] + ['', f'Saved: {ros_path}', f'Session: {node.session}', ''])

    return out, summary


# ── GUI ────────────────────────────────────────────────────────────────────────
class CalibGUI:
    FEED_W  = 960
    FEED_H  = 540
    PANEL_W = 400
    WIN_W   = FEED_W + PANEL_W
    WIN_H   = FEED_H
    PAD     = 14
    BTN_H   = 52

    # BGR colors
    BG        = (28,  28,  28)
    PANEL_BG  = (20,  20,  20)
    GREEN     = (50,  210,  70)
    RED       = (50,   50, 220)
    AMBER     = (30,  170, 220)
    WHITE     = (235, 235, 235)
    GRAY      = (110, 110, 110)
    DARK      = (50,   50,  50)
    BORDER_OK = (40,  210,  70)
    BORDER_BAD= (40,   40, 200)
    BTN_CAP   = (40,  160,  80)
    BTN_COM   = (30,  110, 190)

    def __init__(self, node):
        self.node      = node
        self.running   = True
        self.status    = 'Waiting for camera...'
        self.status_ok = False
        self._detected   = False
        self._n_corners  = 0
        self._board_dist = 0.0

        cv2.namedWindow('Hand-Eye Calibration', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Hand-Eye Calibration', self.WIN_W, self.WIN_H)
        cv2.setMouseCallback('Hand-Eye Calibration', self._mouse_cb)

        px = self.FEED_W + self.PAD
        self._btn_cap = (px, self.WIN_H - 2*(self.BTN_H+self.PAD),
                         self.PANEL_W - 2*self.PAD, self.BTN_H)
        self._btn_com = (px, self.WIN_H - (self.BTN_H+self.PAD),
                         self.PANEL_W - 2*self.PAD, self.BTN_H)

    def _mouse_cb(self, event, x, y, *_):
        if event != cv2.EVENT_LBUTTONDOWN: return
        if self._hit(x, y, self._btn_cap): self._capture()
        elif self._hit(x, y, self._btn_com): self._compute()

    @staticmethod
    def _hit(x, y, btn):
        bx, by, bw, bh = btn
        return bx <= x <= bx+bw and by <= y <= by+bh

    def _capture(self):
        ok, msg = self.node.capture()
        self.status    = msg
        self.status_ok = ok
        print(f'[{"OK" if ok else "!!"}] {msg}')

    def _compute(self):
        n = len(self.node.images)
        if n < 4:
            self.status = f'Need at least 4 samples (have {n})'
            self.status_ok = False
            return
        self.status = 'Computing...'
        self.status_ok = False
        def _run():
            result, summary = solve(self.node)
            if result:
                t = result['translation']
                self.status    = f'Done!  x={t["x"]:.3f} y={t["y"]:.3f} z={t["z"]:.3f}'
                self.status_ok = True
            else:
                self.status    = summary.strip()
                self.status_ok = False
            print(summary)
        threading.Thread(target=_run, daemon=True).start()

    def _txt(self, c, text, x, y, scale=0.46, color=None, bold=False):
        if color is None: color = self.WHITE
        cv2.putText(c, text, (x, y), cv2.FONT_HERSHEY_DUPLEX,
                    scale, color, 2 if bold else 1, cv2.LINE_AA)

    def _btn(self, c, rect, label, color, enabled=True):
        bx, by, bw, bh = rect
        fc = color if enabled else self.DARK
        cv2.rectangle(c, (bx, by), (bx+bw, by+bh), fc, -1)
        cv2.rectangle(c, (bx, by), (bx+bw, by+bh),
                      self.WHITE if enabled else self.GRAY, 1)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_DUPLEX, 0.6, 2)
        cv2.putText(c, label, (bx+(bw-tw)//2, by+(bh+th)//2),
                    cv2.FONT_HERSHEY_DUPLEX, 0.6,
                    self.WHITE if enabled else self.GRAY, 2, cv2.LINE_AA)

    def _hline(self, c, y):
        cv2.line(c, (self.FEED_W+self.PAD, y),
                 (self.WIN_W-self.PAD, y), self.DARK, 1)

    def _draw_panel(self, canvas):
        px = self.FEED_W
        cv2.rectangle(canvas, (px, 0), (self.WIN_W, self.WIN_H), self.PANEL_BG, -1)
        cv2.line(canvas, (px, 0), (px, self.WIN_H), self.DARK, 2)

        x, y = px + self.PAD, 26

        # Title
        self._txt(canvas, 'HAND-EYE CALIBRATION', x, y, 0.52, self.AMBER, True)
        y += 6; self._hline(canvas, y+8); y += 24

        # Board detection badge
        if self._detected:
            bc = self.GREEN
            bt = f'BOARD DETECTED  ({self._n_corners} corners)'
        else:
            bc = self.RED
            bt = 'BOARD NOT DETECTED'
        bw2 = self.PANEL_W - 2*self.PAD
        cv2.rectangle(canvas, (x, y-16), (x+bw2, y+10), bc, -1)
        (tw, _), _ = cv2.getTextSize(bt, cv2.FONT_HERSHEY_DUPLEX, 0.46, 1)
        cv2.putText(canvas, bt, (x+(bw2-tw)//2, y),
                    cv2.FONT_HERSHEY_DUPLEX, 0.46, self.WHITE, 1, cv2.LINE_AA)
        y += 30

        if self._detected:
            self._txt(canvas, f'Board distance: {self._board_dist*100:.1f} cm',
                      x, y, 0.43, self.GRAY)
        y += 20; self._hline(canvas, y); y += 16

        # Sample count
        n = len(self.node.images)
        clr = self.GREEN if n >= 15 else (self.AMBER if n >= 4 else self.RED)
        self._txt(canvas, 'Samples collected:', x, y, 0.46, self.GRAY)
        self._txt(canvas, str(n), x+175, y, 0.7, clr, True)
        y += 20

        if n < 4:
            rec, rc = 'Need >=4 to compute', self.RED
        elif n < 15:
            rec, rc = f'{15-n} more recommended', self.AMBER
        else:
            rec, rc = 'Ready to compute!', self.GREEN
        self._txt(canvas, rec, x, y, 0.42, rc)
        y += 20; self._hline(canvas, y); y += 14

        # Recent EE positions
        self._txt(canvas, 'Recent EE positions (x  y  z):', x, y, 0.41, self.GRAY)
        y += 16
        for i, (ex, ey, ez) in enumerate(self.node.ee_xyz[-6:]):
            idx = max(0, n-6) + i + 1
            self._txt(canvas,
                      f'  #{idx:02d}  {ex:+.3f}  {ey:+.3f}  {ez:+.3f}',
                      x, y, 0.39, self.WHITE)
            y += 15

        y += 4; self._hline(canvas, y); y += 14

        # Status (word-wrapped)
        sc = self.GREEN if self.status_ok else (
             self.AMBER if 'omputing' in self.status else self.RED)
        words, buf, lines_out = self.status.split(), '', []
        for w in words:
            test = (buf+' '+w).strip()
            (tw2, _), _ = cv2.getTextSize(test, cv2.FONT_HERSHEY_DUPLEX, 0.42, 1)
            if tw2 > self.PANEL_W - 2*self.PAD:
                lines_out.append(buf); buf = w
            else:
                buf = test
        if buf: lines_out.append(buf)
        for ln in lines_out[:4]:
            self._txt(canvas, ln, x, y, 0.42, sc); y += 16

        # Session folder
        fy = self.WIN_H - 2*(self.BTN_H+self.PAD) - 28
        self._txt(canvas, 'Session folder:', x, fy, 0.38, self.GRAY)
        self._txt(canvas, os.path.basename(self.node.session),
                  x, fy+14, 0.36, self.DARK)

        # Buttons
        self._btn(canvas, self._btn_cap, 'CAPTURE  [SPACE]',
                  self.BTN_CAP, enabled=self._detected)
        self._btn(canvas, self._btn_com, 'COMPUTE & SAVE  [C]',
                  self.BTN_COM, enabled=n >= 4)

    def run(self):
        print('\n' + '='*56)
        print('  SPACE = capture one sample (move arm first each time)')
        print('  C     = compute and save result')
        print('  Q     = quit')
        print(f'  Session: {self.node.session}')
        print('='*56 + '\n')

        while self.running:
            img, K, dist = self.node.get_frame()
            canvas = np.full((self.WIN_H, self.WIN_W, 3), self.BG, dtype=np.uint8)

            if img is not None:
                scale  = min(self.FEED_W / img.shape[1], self.FEED_H / img.shape[0])
                fw, fh = int(img.shape[1]*scale), int(img.shape[0]*scale)
                feed   = cv2.resize(img, (fw, fh))

                result = None
                if K is not None:
                    Ks = K.copy(); Ks[0] *= scale; Ks[1] *= scale
                    gray   = cv2.cvtColor(feed, cv2.COLOR_BGR2GRAY)
                    result = self.node.detect(gray, Ks, dist)

                vis = feed.copy()
                if result is not None:
                    ch_c, ch_ids, rvec, tvec = result
                    cv2.aruco.drawDetectedCornersCharuco(vis, ch_c, ch_ids)
                    cv2.drawFrameAxes(vis, Ks, dist, rvec, tvec, SQUARE_M*2)
                    self._detected   = True
                    self._n_corners  = len(ch_ids)
                    self._board_dist = float(np.linalg.norm(tvec))
                    border_c = self.BORDER_OK
                else:
                    self._detected = False
                    border_c       = self.BORDER_BAD

                oy = (self.FEED_H - fh) // 2
                canvas[oy:oy+fh, 0:fw] = vis
                cv2.rectangle(canvas, (2, 2),
                              (self.FEED_W-3, self.FEED_H-3), border_c, 8)

                n_str = f'{len(self.node.images)} samples'
                cv2.putText(canvas, n_str, (14, 36),
                            cv2.FONT_HERSHEY_DUPLEX, 0.8, (0,0,0), 5, cv2.LINE_AA)
                cv2.putText(canvas, n_str, (14, 36),
                            cv2.FONT_HERSHEY_DUPLEX, 0.8, self.WHITE, 2, cv2.LINE_AA)

                hint = 'SPACE = capture' if self._detected else 'Move board into camera view'
                hc   = self.GREEN if self._detected else self.GRAY
                cv2.putText(canvas, hint, (14, self.FEED_H-14),
                            cv2.FONT_HERSHEY_DUPLEX, 0.52, (0,0,0), 4, cv2.LINE_AA)
                cv2.putText(canvas, hint, (14, self.FEED_H-14),
                            cv2.FONT_HERSHEY_DUPLEX, 0.52, hc, 1, cv2.LINE_AA)
            else:
                self._txt(canvas, 'Waiting for camera image...',
                          20, self.FEED_H//2, 0.7, self.GRAY)

            self._draw_panel(canvas)
            cv2.imshow('Hand-Eye Calibration', canvas)

            key = cv2.waitKey(30) & 0xFF
            if key == ord(' '):
                self._capture()
            elif key in (ord('c'), ord('C')):
                self._compute()
            elif key in (ord('q'), ord('Q'), 27):
                self.running = False

        cv2.destroyAllWindows()


# ── MAIN ───────────────────────────────────────────────────────────────────────
def main():
    rclpy.init()
    node = Collector()
    spin_thread = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    spin_thread.start()

    print('Waiting for camera', end='', flush=True)
    for _ in range(40):
        _, K, _ = node.get_frame()
        if K is not None: break
        print('.', end='', flush=True); time.sleep(0.5)
    print()

    CalibGUI(node).run()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
