import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, PointCloud2, PointField, CameraInfo
from cv_bridge import CvBridge
import depthai as dai
import numpy as np
import threading
import time


# ── Published topics ────────────────────────────────────────────────────────────
# /global_camera/color/image_raw        ← BGR8 image  (640×400)
# /global_camera/color/camera_info      ← intrinsics  (640×400)
# /global_camera/depth/image_raw        ← uint16 depth (640×400, mm)
# /global_camera/depth/camera_info      ← intrinsics  (640×400)  ← same as color
# /global_camera/stereo/points          ← XYZRGB PointCloud2 for MoveIt2 / RViz2
# frame_id: global_camera_link
# ────────────────────────────────────────────────────────────────────────────────

# ✅ FIX: Both RGB and depth use the SAME resolution so camera_info is consistent
#         and yolo_ros 3D XYZ localization works correctly.
IMG_W, IMG_H = 640, 400

PC_STEP = 2          # point-cloud subsampling (1=dense, 2=half, 4=fast)
FRAME_ID = "global_camera_link"

DEVICE_RETRY_ATTEMPTS = 10
DEVICE_RETRY_DELAY_S  = 3.0

# Fallback intrinsics (overwritten from EEPROM at startup)
FX, FY = 432.8, 432.3
CX, CY = 320.0, 200.0


class OakCameraNode(Node):
    def __init__(self):
        super().__init__("oak_camera_node")
        self.bridge = CvBridge()

        self.rgb_pub        = self.create_publisher(Image,       "/global_camera/color/image_raw",   10)
        self.depth_pub      = self.create_publisher(Image,       "/global_camera/depth/image_raw",   10)
        self.pc_pub         = self.create_publisher(PointCloud2, "/global_camera/stereo/points",     10)
        self.color_info_pub = self.create_publisher(CameraInfo,  "/global_camera/color/camera_info", 10)
        self.depth_info_pub = self.create_publisher(CameraInfo,  "/global_camera/depth/camera_info", 10)

        self._dist_coeffs = [0.0, 0.0, 0.0, 0.0, 0.0]

        self.get_logger().info("OAK-D Pro Wide node starting  (640×400 unified resolution)...")
        threading.Thread(target=self._run_pipeline, daemon=True).start()

    # ── camera_info (width/height always match actual image) ────────────────────
    def _make_camera_info(self, stamp):
        msg = CameraInfo()
        msg.header.stamp     = stamp
        msg.header.frame_id  = FRAME_ID
        msg.width            = IMG_W       # ✅ matches actual published image
        msg.height           = IMG_H
        msg.distortion_model = "plumb_bob"
        msg.d = self._dist_coeffs
        msg.k = [FX,  0.0, CX,
                 0.0, FY,  CY,
                 0.0, 0.0, 1.0]
        msg.r = [1.0, 0.0, 0.0,
                 0.0, 1.0, 0.0,
                 0.0, 0.0, 1.0]
        msg.p = [FX,  0.0, CX,  0.0,
                 0.0, FY,  CY,  0.0,
                 0.0, 0.0, 1.0, 0.0]
        return msg

    # ── XYZRGB point cloud ───────────────────────────────────────────────────────
    def _depth_to_pointcloud_rgb(self, depth_img, rgb_img, stamp):
        """
        depth_img : uint16 (mm),  IMG_W × IMG_H
        rgb_img   : uint8  BGR,   IMG_W × IMG_H  (same resolution — no mismatch)
        """
        h, w = depth_img.shape
        u_idx = np.arange(0, w, PC_STEP)
        v_idx = np.arange(0, h, PC_STEP)
        uu, vv = np.meshgrid(u_idx, v_idx)

        z = depth_img[vv, uu].astype(np.float32) * 0.001   # mm → m
        valid = (z > 0.15) & (z < 3.0)

        z  = z[valid];  uu = uu[valid];  vv = vv[valid]
        x  = (uu - CX) * z / FX
        y  = (vv - CY) * z / FY

        b = rgb_img[vv, uu, 0].astype(np.uint32)
        g = rgb_img[vv, uu, 1].astype(np.uint32)
        r = rgb_img[vv, uu, 2].astype(np.uint32)

        rgb_packed = (r << 16) | (g << 8) | b
        rgb_float  = rgb_packed.view(np.float32)

        pts = np.column_stack([
            x.astype(np.float32),
            y.astype(np.float32),
            z.astype(np.float32),
            rgb_float,
        ])

        msg = PointCloud2()
        msg.header.stamp    = stamp
        msg.header.frame_id = FRAME_ID
        msg.height          = 1
        msg.width           = len(pts)
        msg.is_dense        = False
        msg.is_bigendian    = False
        msg.point_step      = 16
        msg.row_step        = 16 * len(pts)
        msg.fields = [
            PointField(name='x',   offset=0,  datatype=PointField.FLOAT32, count=1),
            PointField(name='y',   offset=4,  datatype=PointField.FLOAT32, count=1),
            PointField(name='z',   offset=8,  datatype=PointField.FLOAT32, count=1),
            PointField(name='rgb', offset=12, datatype=PointField.FLOAT32, count=1),
        ]
        msg.data = pts.tobytes()
        return msg

    # ── DepthAI v3 pipeline ──────────────────────────────────────────────────────
    def _run_pipeline(self):
        global FX, FY, CX, CY

        device = None
        for attempt in range(1, DEVICE_RETRY_ATTEMPTS + 1):
            try:
                device = dai.Device()
                break
            except RuntimeError as e:
                if "ALREADY_IN_USE" in str(e):
                    self.get_logger().warn(
                        f"OAK-D busy — attempt {attempt}/{DEVICE_RETRY_ATTEMPTS}, "
                        f"retry in {int(DEVICE_RETRY_DELAY_S)}s  "
                        f"(free: sudo fuser -k /dev/bus/usb/*/*)"
                    )
                    time.sleep(DEVICE_RETRY_DELAY_S)
                else:
                    self.get_logger().error(f"Cannot open OAK-D: {e}")
                    return

        if device is None:
            self.get_logger().error("OAK-D still busy after all retries.")
            return

        # ── Load EEPROM intrinsics at 640×400 ───────────────────────────────────
        calib = device.readCalibration()
        intrinsics = calib.getCameraIntrinsics(
            dai.CameraBoardSocket.CAM_A, IMG_W, IMG_H   # ✅ correct resolution
        )
        FX = intrinsics[0][0]
        FY = intrinsics[1][1]
        CX = intrinsics[0][2]
        CY = intrinsics[1][2]
        dist = calib.getDistortionCoefficients(dai.CameraBoardSocket.CAM_A)
        self._dist_coeffs = list(dist[:5]) if len(dist) >= 5 else list(dist)

        self.get_logger().info(
            f"EEPROM intrinsics @ {IMG_W}×{IMG_H}: "
            f"FX={FX:.2f}  FY={FY:.2f}  CX={CX:.2f}  CY={CY:.2f}"
        )

        with dai.Pipeline(device) as pipeline:

            # RGB camera at 640×400 (same as depth)
            cam_rgb = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_A)
            rgb_out = cam_rgb.requestOutput(
                (IMG_W, IMG_H),
                type=dai.ImgFrame.Type.BGR888i,
                enableUndistortion=True,
            )

            # Stereo pair
            left  = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_B)
            right = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_C)
            left_out  = left.requestOutput((IMG_W, IMG_H), type=dai.ImgFrame.Type.NV12)
            right_out = right.requestOutput((IMG_W, IMG_H), type=dai.ImgFrame.Type.NV12)

            # StereoDepth aligned to RGB
            stereo = pipeline.create(dai.node.StereoDepth).build(
                left=left_out,
                right=right_out,
                presetMode=dai.node.StereoDepth.PresetMode.DEFAULT,
            )
            rgb_out.link(stereo.inputAlignTo)

            rgb_queue   = rgb_out.createOutputQueue(maxSize=4, blocking=False)
            depth_queue = stereo.depth.createOutputQueue(maxSize=4, blocking=False)

            pipeline.start()
            self.get_logger().info(
                "✅ OAK-D streaming: /global_camera/{color,depth,stereo}  "
                "frame=global_camera_link  res=640×400"
            )

            latest_rgb = None

            while pipeline.isRunning() and rclpy.ok():
                stamp = self.get_clock().now().to_msg()
                info  = self._make_camera_info(stamp)

                rgb_frame = rgb_queue.tryGet()
                if rgb_frame is not None:
                    img = rgb_frame.getCvFrame()
                    latest_rgb = img

                    msg = self.bridge.cv2_to_imgmsg(img, "bgr8")
                    msg.header.stamp    = stamp
                    msg.header.frame_id = FRAME_ID
                    self.rgb_pub.publish(msg)
                    self.color_info_pub.publish(info)

                depth_frame = depth_queue.tryGet()
                if depth_frame is not None:
                    depth = depth_frame.getCvFrame().astype(np.uint16)

                    dmsg = self.bridge.cv2_to_imgmsg(depth, "16UC1")
                    dmsg.header.stamp    = stamp
                    dmsg.header.frame_id = FRAME_ID
                    self.depth_pub.publish(dmsg)
                    self.depth_info_pub.publish(info)

                    if latest_rgb is not None:
                        self.pc_pub.publish(
                            self._depth_to_pointcloud_rgb(depth, latest_rgb, stamp)
                        )


def main():
    rclpy.init()
    node = OakCameraNode()
    rclpy.spin(node)
    rclpy.shutdown()


if __name__ == "__main__":
    main()
