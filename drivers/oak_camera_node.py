#!/usr/bin/env python3
"""
oak_camera_node.py  —  OAK-D Pro Wide ROS2 driver  (depthai v2.x API)
=====================================================================
Single definitive OAK-D driver. Do NOT use depthai_ros_driver alongside this.

Publishes (matching perception_module.py exactly):
  /global_camera/color/image_raw       sensor_msgs/Image     BGR8  416×256
  /global_camera/color/camera_info     sensor_msgs/CameraInfo
  /global_camera/depth/image_raw       sensor_msgs/Image     16UC1 depth mm
  /global_camera/depth/camera_info     sensor_msgs/CameraInfo
  /global_camera/stereo/points         sensor_msgs/PointCloud2  XYZRGB (full scene)

TF frame: global_camera_link  (matches perception_module CAM_FRAME)

Hand-eye calibration (base_link → global_camera_link) is NOT published here.
The single publisher is thesis_robot's camera_tf_broadcaster, which reads the
calibration from ~/.ros/handeye_calibration_corrected.yaml:
  ros2 run thesis_robot camera_tf_broadcaster
(oak_launch.py starts it alongside this node.)

USB 2.0 workaround:
  export DEPTHAI_USB2_MODE=1   → RGB published only, depth/pointcloud skipped.

Run (standalone, from the repo root):
  source ~/workspace/ros2_kortex_ws/install/setup.bash
  python3 drivers/oak_camera_node.py

Run via launch file (from the repo root):
  ros2 launch drivers/oak_launch.py
"""

import os
import threading
import time

import numpy as np
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from sensor_msgs.msg import CameraInfo, Image, PointCloud2, PointField

import depthai as dai

# This node is written against the depthai v2.x API (ColorCamera/MonoCamera +
# XLinkOut + Device(pipeline)). depthai v3 renamed/removed those APIs, so a
# silent upgrade would make every pipeline attempt throw and retry forever
# instead of failing clearly. Fail fast and loud instead.
_DAI_MAJOR = int(dai.__version__.split(".")[0])
if _DAI_MAJOR != 2:
    raise RuntimeError(
        f"oak_camera_node.py requires depthai v2.x (found {dai.__version__}). "
        f"Run: pip install 'depthai==2.32.0.0'  "
        f"(or port this node to the v3 API if upgrading is required)."
    )

# ── Resolution ───────────────────────────────────────────────────────────────
# 416×256 reduces USB bandwidth by ~57% vs 640×400, preventing XLink errors
# when sharing a USB 2.1 hub with the RealSense D435I.
IMG_W, IMG_H = 416, 256
PC_STEP      = 1          # point-cloud subsampling (1=dense, 2=half, 4=fast)

# ── Topic names — SINGLE SOURCE OF TRUTH (must match perception_module.py) ───
TOPIC_RGB        = "/global_camera/color/image_raw"
TOPIC_DEPTH      = "/global_camera/depth/image_raw"
TOPIC_RGB_INFO   = "/global_camera/color/camera_info"
TOPIC_DEPTH_INFO = "/global_camera/depth/camera_info"
TOPIC_SCENE_PC   = "/global_camera/stereo/points"

# ── TF frame (must match perception_module.py) ────────────────────────────────
CAM_FRAME  = "global_camera_link"

# ── Intrinsics @ 416×256 (proportionally scaled from 640×400 EEPROM values) ──
# scale_x = 416/640 = 0.65,  scale_y = 256/400 = 0.64
FX = 277.0
FY = 272.6
CX = 204.9
CY = 131.4

# ── Retry config ─────────────────────────────────────────────────────────────
RETRY_N     = 10
RETRY_DELAY = 3.0


class OakCameraNode(Node):

    def __init__(self):
        super().__init__("oak_camera_node")
        self.bridge = CvBridge()

        # Publishers
        self.rgb_pub   = self.create_publisher(Image,       TOPIC_RGB,        10)
        self.rgb_info  = self.create_publisher(CameraInfo,  TOPIC_RGB_INFO,   10)
        self.dep_pub   = self.create_publisher(Image,       TOPIC_DEPTH,      10)
        self.dep_info  = self.create_publisher(CameraInfo,  TOPIC_DEPTH_INFO, 10)
        self.pc_pub    = self.create_publisher(PointCloud2, TOPIC_SCENE_PC,   10)

        usb2 = os.environ.get("DEPTHAI_USB2_MODE") == "1"
        self.get_logger().info(
            f"\nOAK-D camera node starting  ({IMG_W}×{IMG_H})\n"
            f"  RGB   → {TOPIC_RGB}\n"
            f"  Depth → {TOPIC_DEPTH}\n"
            f"  PCL   → {TOPIC_SCENE_PC}\n"
            f"  Frame → {CAM_FRAME}  (base_link TF comes from "
            f"thesis_robot camera_tf_broadcaster)\n"
            f"  USB2 mode: {'ON (depth disabled)' if usb2 else 'OFF'}"
        )

        threading.Thread(target=self._run, daemon=True).start()

    # ── CameraInfo ────────────────────────────────────────────────────────────
    def _camera_info(self, stamp, dist_coeffs=None):
        msg = CameraInfo()
        msg.header.stamp     = stamp
        msg.header.frame_id  = CAM_FRAME
        msg.width            = IMG_W
        msg.height           = IMG_H
        msg.distortion_model = "plumb_bob"
        msg.d  = dist_coeffs if dist_coeffs else [0.0] * 5
        msg.k  = [FX,  0.0, CX,  0.0, FY,  CY,  0.0, 0.0, 1.0]
        msg.r  = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
        msg.p  = [FX,  0.0, CX,  0.0, 0.0, FY,  CY,  0.0, 0.0, 0.0, 1.0, 0.0]
        return msg

    # ── Point cloud ───────────────────────────────────────────────────────────
    def _make_pointcloud(self, depth_mm: np.ndarray, rgb_bgr: np.ndarray, stamp):
        h, w = depth_mm.shape
        u = np.arange(0, w, PC_STEP)
        v = np.arange(0, h, PC_STEP)
        uu, vv = np.meshgrid(u, v)

        z = depth_mm[vv, uu].astype(np.float32) * 0.001  # mm → m
        valid = (z > 0.15) & (z < 3.0)
        z  = z[valid];  uu = uu[valid];  vv = vv[valid]
        x  = (uu - CX) * z / FX
        y  = (vv - CY) * z / FY

        b = rgb_bgr[vv, uu, 0].astype(np.uint32)
        g = rgb_bgr[vv, uu, 1].astype(np.uint32)
        r = rgb_bgr[vv, uu, 2].astype(np.uint32)
        rgb_packed = ((r << 16) | (g << 8) | b).view(np.float32)

        pts = np.column_stack([
            x.astype(np.float32), y.astype(np.float32),
            z.astype(np.float32), rgb_packed,
        ])
        msg = PointCloud2()
        msg.header.stamp    = stamp
        msg.header.frame_id = CAM_FRAME
        msg.height = 1;  msg.width = len(pts)
        msg.is_dense = False;  msg.is_bigendian = False
        msg.point_step = 16;  msg.row_step = 16 * len(pts)
        msg.fields = [
            PointField(name='x',   offset=0,  datatype=PointField.FLOAT32, count=1),
            PointField(name='y',   offset=4,  datatype=PointField.FLOAT32, count=1),
            PointField(name='z',   offset=8,  datatype=PointField.FLOAT32, count=1),
            PointField(name='rgb', offset=12, datatype=PointField.FLOAT32, count=1),
        ]
        msg.data = pts.tobytes()
        return msg

    # ── Build the DepthAI v2 pipeline ─────────────────────────────────────────
    def _build_pipeline(self, usb2: bool):
        """Construct a depthai v2.x pipeline (ColorCamera + MonoCamera stereo)."""
        pipeline = dai.Pipeline()

        # RGB camera → XLinkOut "rgb"
        cam_rgb = pipeline.create(dai.node.ColorCamera)
        cam_rgb.setBoardSocket(dai.CameraBoardSocket.CAM_A)
        cam_rgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
        cam_rgb.setInterleaved(False)
        cam_rgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)
        cam_rgb.setPreviewSize(IMG_W, IMG_H)
        cam_rgb.setPreviewKeepAspectRatio(False)
        xout_rgb = pipeline.create(dai.node.XLinkOut)
        xout_rgb.setStreamName("rgb")
        cam_rgb.preview.link(xout_rgb.input)

        # Stereo depth (skipped on USB 2.0) → XLinkOut "depth", aligned to RGB
        if not usb2:
            mono_l = pipeline.create(dai.node.MonoCamera)
            mono_l.setBoardSocket(dai.CameraBoardSocket.CAM_B)
            mono_l.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
            mono_r = pipeline.create(dai.node.MonoCamera)
            mono_r.setBoardSocket(dai.CameraBoardSocket.CAM_C)
            mono_r.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)

            stereo = pipeline.create(dai.node.StereoDepth)
            stereo.setDefaultProfilePreset(
                dai.node.StereoDepth.PresetMode.HIGH_DENSITY)
            stereo.setLeftRightCheck(True)
            stereo.setDepthAlign(dai.CameraBoardSocket.CAM_A)  # align depth to RGB
            stereo.setOutputSize(IMG_W, IMG_H)                 # match RGB resolution
            mono_l.out.link(stereo.left)
            mono_r.out.link(stereo.right)

            xout_depth = pipeline.create(dai.node.XLinkOut)
            xout_depth.setStreamName("depth")
            stereo.depth.link(xout_depth.input)

        return pipeline

    # ── Device open with retry ────────────────────────────────────────────────
    def _open_device(self, pipeline):
        """Open the device WITH the pipeline (v2 API) — starts streaming."""
        for attempt in range(1, RETRY_N + 1):
            try:
                return dai.Device(pipeline)
            except RuntimeError as e:
                busy = "ALREADY_IN_USE" in str(e) or "no available" in str(e).lower()
                if busy:
                    self.get_logger().warn(
                        f"OAK-D busy — attempt {attempt}/{RETRY_N}, "
                        f"retry in {int(RETRY_DELAY)}s  "
                        f"(free with: sudo fuser -k /dev/bus/usb/*/*)"
                    )
                    time.sleep(RETRY_DELAY)
                else:
                    self.get_logger().error(f"Cannot open OAK-D: {e}")
                    return None
        self.get_logger().error("OAK-D still busy after all retries.")
        return None

    # ── DepthAI pipeline ──────────────────────────────────────────────────────

    def _run(self):
        """Restart wrapper — restarts the pipeline on any crash."""
        while rclpy.ok():
            try:
                self._run_pipeline()
            except Exception as e:
                self.get_logger().error(
                    f"OAK-D pipeline crashed: {e} — restarting in 5s"
                )
                time.sleep(5.0)

    def _run_pipeline(self):
        usb2     = os.environ.get("DEPTHAI_USB2_MODE") == "1"
        pipeline = self._build_pipeline(usb2)

        device = self._open_device(pipeline)
        if device is None:
            return

        with device:
            # Read EEPROM distortion coefficients
            try:
                calib       = device.readCalibration()
                dist        = calib.getDistortionCoefficients(
                    dai.CameraBoardSocket.CAM_A)
                dist_coeffs = list(dist[:5]) if len(dist) >= 5 else list(dist)
            except Exception:
                dist_coeffs = [0.0] * 5

            self.get_logger().info(
                f"OAK-D connected  FX={FX:.2f} FY={FY:.2f} CX={CX:.2f} CY={CY:.2f}\n"
                f"  dist={[round(d, 4) for d in dist_coeffs]}"
            )

            rgb_q   = device.getOutputQueue("rgb", maxSize=4, blocking=False)
            depth_q = (device.getOutputQueue("depth", maxSize=4, blocking=False)
                       if not usb2 else None)

            mode = "RGB + Depth + PointCloud" if depth_q else "RGB only"
            self.get_logger().info(f"✅ OAK-D streaming [{mode}]  frame={CAM_FRAME}")

            latest_rgb    = None
            last_rgb_time = self.get_clock().now()

            while rclpy.ok() and not device.isClosed():
                now   = self.get_clock().now()
                stamp = now.to_msg()
                info  = self._camera_info(stamp, dist_coeffs)

                rgb_frame = rgb_q.tryGet()
                if rgb_frame is not None:
                    img = rgb_frame.getCvFrame()
                    latest_rgb = img  # always refresh for depth/PCL sync
                    # Rate-limit RGB publication to 30 Hz (33 ms minimum interval).
                    # tryGet() can return batched frames that spike the apparent Hz.
                    if (now - last_rgb_time).nanoseconds >= 33_000_000:
                        msg = self.bridge.cv2_to_imgmsg(img, "bgr8")
                        msg.header.stamp    = stamp
                        msg.header.frame_id = CAM_FRAME
                        self.rgb_pub.publish(msg)
                        self.rgb_info.publish(info)
                        last_rgb_time = now

                if depth_q is not None:
                    depth_frame = depth_q.tryGet()
                    if depth_frame is not None:
                        depth = depth_frame.getFrame().astype(np.uint16)
                        dmsg = self.bridge.cv2_to_imgmsg(depth, "16UC1")
                        dmsg.header.stamp    = stamp
                        dmsg.header.frame_id = CAM_FRAME
                        self.dep_pub.publish(dmsg)
                        self.dep_info.publish(info)
                        if latest_rgb is not None:
                            self.pc_pub.publish(
                                self._make_pointcloud(depth, latest_rgb, stamp))

                time.sleep(0.001)  # yield CPU; queues are non-blocking


def main():
    rclpy.init()
    node = OakCameraNode()
    rclpy.spin(node)
    rclpy.shutdown()


if __name__ == "__main__":
    main()
