#!/usr/bin/env python3
"""
define_workspace_boundary.py — click the table's corners in a live camera
feed instead of hand-editing scene_graph_node.py's WORKSPACE dict.

Click points on the table in the image (at least 2 — opposite corners of
the reachable area is enough, click more for an irregular table and this
still just takes their bounding box, since WORKSPACE is an axis-aligned
box, not a polygon). Each click is backprojected through that camera's
depth image and TF'd to base_link, exactly like object_detection.py's own
detections are. Press:
  c — compute the bounding box from clicked points and save
  r — reset (clear clicked points)
  q — quit without saving

Writes ~/.ros/workspace_bounds.yaml, which scene_graph_node.py loads at
startup if present (falling back to its hardcoded default otherwise — see
its WORKSPACE / _load_workspace_bounds).

Usage:
  python3 calibration/define_workspace_boundary.py
  python3 calibration/define_workspace_boundary.py --ros-args -p camera:=realsense
"""
import os
import sys
import threading
import time

import cv2
import numpy as np
import rclpy
import tf2_ros
import tf2_geometry_msgs  # noqa: F401 — registers PointStamped with tf2
import yaml
from cv_bridge import CvBridge
from geometry_msgs.msg import PointStamped
from rclpy.node import Node
from sensor_msgs.msg import CameraInfo, Image

CAMERAS = {
    'oakd': {
        'image': '/global_camera/color/image_raw',
        'depth': '/global_camera/depth/image_raw',
        'info': '/global_camera/color/camera_info',
    },
    'realsense': {
        'image': '/global_camera/global_camera/color/image_raw',
        'depth': '/global_camera/global_camera/aligned_depth_to_color/image_raw',
        'info': '/global_camera/global_camera/color/camera_info',
    },
    'wrist': {
        'image': '/camera/color/image_raw',
        'depth': '/camera/depth/image_raw',
        'info': '/camera/color/camera_info',
    },
}

OUT_FILE = os.path.expanduser('~/.ros/workspace_bounds.yaml')
Z_PAD_M = 0.30  # default z range is table height +/- this, since we only click x/y
# Kinova Gen3 7DOF max horizontal reach from base_link, plus a little
# slack. A click producing a point beyond this is almost certainly a bad
# click (wrong pixel, camera calibration error, background bleed into
# frame) rather than a real corner of the reachable area -- this exact
# failure mode shipped silently to workspace_bounds.yaml once already
# (OAK-D, clicked from too far away) and wasn't caught until a much later
# pipeline stage. Refuse to save rather than repeat that.
MAX_PLAUSIBLE_REACH_M = 0.95


class DefineWorkspaceBoundary(Node):
    def __init__(self, camera_name):
        # Node name includes camera_name so multiple instances (one per
        # camera) can coexist in the same rclpy context without colliding --
        # needed once something creates more than one of these at a time
        # (the dashboard does, to let you pick which camera to click on).
        super().__init__(f'define_workspace_boundary_{camera_name}')
        self.camera_name = camera_name
        topics = CAMERAS[camera_name]
        self.bridge = CvBridge()
        self.bgr = None
        self.depth = None
        self.K = None
        self.frame_id = None
        self.base_frame = 'base_link'

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        self.create_subscription(Image, topics['image'], self._on_image, 10)
        self.create_subscription(Image, topics['depth'], self._on_depth, 10)
        self.create_subscription(CameraInfo, topics['info'], self._on_info, 10)

        self.clicked_base_xyz = []   # [(x, y, z), ...] in base_link
        self.clicked_pixels = []     # [(u, v), ...] — same length/order, for on-screen markers
        self.last_click_status = ''

    def _on_image(self, msg):
        self.bgr = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        self.frame_id = msg.header.frame_id

    def _on_depth(self, msg):
        self.depth = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')

    def _on_info(self, msg):
        self.K = np.array(msg.k).reshape(3, 3)

    def _depth_at(self, u, v):
        if self.depth is None:
            return None
        h, w = self.depth.shape
        u = max(0, min(u, w - 1))
        v = max(0, min(v, h - 1))
        patch = self.depth[max(0, v - 5):min(h, v + 5), max(0, u - 5):min(w, u + 5)]
        valid = patch[patch > 0]
        if len(valid) == 0:
            return None
        med = float(np.median(valid))
        depth_m = med / 1000.0 if med > 10.0 else med  # 16UC1 mm vs already-metres
        return depth_m if depth_m > 0.05 else None

    def click(self, u, v):
        if self.K is None or self.depth is None or self.frame_id is None:
            self.last_click_status = 'not ready yet (waiting for image/depth/info)'
            return
        depth_m = self._depth_at(u, v)
        if depth_m is None:
            self.last_click_status = f'no valid depth at ({u},{v}) — click somewhere else'
            return

        fx, fy, cx, cy = self.K[0, 0], self.K[1, 1], self.K[0, 2], self.K[1, 2]
        cx3 = (u - cx) * depth_m / fx
        cy3 = (v - cy) * depth_m / fy
        cz3 = depth_m

        pt = PointStamped()
        pt.header.frame_id = self.frame_id
        pt.point.x, pt.point.y, pt.point.z = cx3, cy3, cz3
        try:
            tf = self.tf_buffer.lookup_transform(
                self.base_frame, self.frame_id, rclpy.time.Time(),
                timeout=rclpy.duration.Duration(seconds=1.0))
        except Exception as e:
            self.last_click_status = f'no TF {self.frame_id} -> {self.base_frame}: {e}'
            return
        p = tf2_geometry_msgs.do_transform_point(pt, tf).point
        self.clicked_base_xyz.append((float(p.x), float(p.y), float(p.z)))
        self.clicked_pixels.append((u, v))
        self.last_click_status = (
            f'point {len(self.clicked_base_xyz)} @ pixel ({u},{v}): '
            f'base_link=({p.x:.3f}, {p.y:.3f}, {p.z:.3f})')

    def compute_and_save(self):
        if len(self.clicked_base_xyz) < 2:
            self.last_click_status = 'need at least 2 points — click more corners'
            return False

        implausible = [
            (i, x, y, (x ** 2 + y ** 2) ** 0.5)
            for i, (x, y, _z) in enumerate(self.clicked_base_xyz, start=1)
            if (x ** 2 + y ** 2) ** 0.5 > MAX_PLAUSIBLE_REACH_M
        ]
        if implausible:
            worst = max(implausible, key=lambda t: t[3])
            self.last_click_status = (
                f'point {worst[0]} is {worst[3]:.2f}m from base_link — beyond the '
                f'arm\'s plausible reach ({MAX_PLAUSIBLE_REACH_M}m). Not saving. '
                f'Likely a misclick, or this camera is too far/uncertain for an '
                f'accurate click here — try a closer camera (wrist works best: '
                f'jog the arm right up to each corner) or reset and re-click.')
            return False

        xs = [p[0] for p in self.clicked_base_xyz]
        ys = [p[1] for p in self.clicked_base_xyz]
        zs = [p[2] for p in self.clicked_base_xyz]
        table_z = sum(zs) / len(zs)
        bounds = {
            'x': [round(min(xs), 4), round(max(xs), 4)],
            'y': [round(min(ys), 4), round(max(ys), 4)],
            'z': [round(table_z - Z_PAD_M, 4), round(table_z + 2.0, 4)],
            'note': (
                f'Computed from {len(self.clicked_base_xyz)} clicked points by '
                f'define_workspace_boundary.py. x/y are the bounding box of the '
                f'clicked table corners; z is table height +/-{Z_PAD_M}m below and '
                f'+2.0m above, since only x/y were clicked on the table surface.'),
        }
        os.makedirs(os.path.dirname(OUT_FILE), exist_ok=True)
        with open(OUT_FILE, 'w') as f:
            yaml.dump(bounds, f, default_flow_style=False)
        self.last_click_status = f'saved {OUT_FILE} — restart scene_graph_node to pick it up'
        return True


def main():
    rclpy.init()
    camera_name = 'oakd'
    for i, a in enumerate(sys.argv):
        if a == '-p' and i + 1 < len(sys.argv) and sys.argv[i + 1].startswith('camera:='):
            camera_name = sys.argv[i + 1].split(':=', 1)[1]
    if camera_name not in CAMERAS:
        print(f'Unknown camera {camera_name!r} — choose from {list(CAMERAS)}')
        sys.exit(1)

    node = DefineWorkspaceBoundary(camera_name)
    spin_thread = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    spin_thread.start()

    win = 'Define Workspace Boundary'
    # WINDOW_AUTOSIZE deliberately, not WINDOW_NORMAL: a resizable window
    # risks the mouse callback's (x,y) not being scaled back to the native
    # image's pixel coordinates correctly on every Qt/GTK backend, which
    # would silently register clicks at the wrong pixel. AUTOSIZE keeps the
    # window's pixels 1:1 with the image, so this can't happen.
    cv2.namedWindow(win, cv2.WINDOW_AUTOSIZE)

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            node.click(x, y)

    cv2.setMouseCallback(win, on_mouse)

    print(f'\nUsing camera: {camera_name}')
    print('Click the table corners that bound the reachable workspace.')
    print('  c = compute bounding box + save   r = reset   q = quit\n')

    try:
        while rclpy.ok():
            if node.bgr is None:
                time.sleep(0.05)
                continue
            canvas = node.bgr.copy()
            for i, (u, v) in enumerate(node.clicked_pixels):
                cv2.drawMarker(canvas, (u, v), (0, 0, 255),
                                markerType=cv2.MARKER_CROSS, markerSize=16, thickness=2)
                cv2.putText(canvas, str(i + 1), (u + 8, v - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            cv2.rectangle(canvas, (0, 0), (canvas.shape[1], 30), (20, 20, 20), -1)
            cv2.putText(canvas, f'points: {len(node.clicked_base_xyz)}  '
                                 f'[c]ompute+save  [r]eset  [q]uit',
                        (8, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
            cv2.rectangle(canvas, (0, canvas.shape[0] - 26), (canvas.shape[1], canvas.shape[0]),
                          (20, 20, 20), -1)
            cv2.putText(canvas, node.last_click_status, (8, canvas.shape[0] - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            cv2.imshow(win, canvas)

            key = cv2.waitKey(30) & 0xFF
            if key in (ord('q'), 27):
                break
            elif key == ord('r'):
                node.clicked_base_xyz.clear()
                node.clicked_pixels.clear()
                node.last_click_status = 'reset'
            elif key == ord('c'):
                node.compute_and_save()
    except KeyboardInterrupt:
        pass
    finally:
        cv2.destroyAllWindows()
        rclpy.shutdown()
        spin_thread.join(timeout=2.0)
        node.destroy_node()


if __name__ == '__main__':
    main()
