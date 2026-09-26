import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from std_msgs.msg import String
from geometry_msgs.msg import PointStamped
from cv_bridge import CvBridge
import tf2_ros
import tf2_geometry_msgs
import json
import numpy as np

class Depth3DNode(Node):
    def __init__(self):
        super().__init__("depth_3d_node")
        self.bridge = CvBridge()
        self.depth_image = None

        # Camera intrinsics for OAK-D Pro Wide at 640x400
        self.fx = 457.97
        self.fy = 457.97
        self.cx = 322.96
        self.cy = 199.5
        self.depth_scale = 0.001  # mm to metres

        # TF buffer for camera → robot transform
        self.tf_buffer   = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        self.create_subscription(
            Image, "/global_camera/depth/image_raw", self.depth_cb, 10)
        self.create_subscription(
            String, "/detections", self.det_cb, 10)

        # Publish in camera frame (legacy)
        self.point_pub = self.create_publisher(
            PointStamped, "/bottle_position", 10)
        # Publish in robot frame (new — used by scene graph)
        self.robot_point_pub = self.create_publisher(
            PointStamped, "/object_position_robot_frame", 10)
        # Publish enriched detections with 3D robot-frame coords
        self.det3d_pub = self.create_publisher(
            String, "/detections_3d", 10)

        self.get_logger().info("Depth 3D node ready — publishing camera + robot frame")

    def depth_cb(self, msg):
        self.depth_image = self.bridge.imgmsg_to_cv2(msg, "16UC1")

    def det_cb(self, msg):
        if self.depth_image is None:
            return

        try:
            detections = json.loads(msg.data)
        except Exception:
            return

        if not isinstance(detections, list):
            detections = [detections]

        enriched = []
        depth_h, depth_w = self.depth_image.shape

        for det in detections:
            label = det.get('label') or det.get('class', 'unknown')
            conf  = float(det.get('confidence', 0.0))
            if conf < 0.45:
                continue

            # Get pixel centre — handle both formats
            if 'center' in det:
                cx_px, cy_px = det['center']
            elif 'cx' in det:
                cx_px, cy_px = det['cx'], det['cy']
            else:
                continue

            # Scale to depth resolution
            sx = depth_w / 640.0
            sy = depth_h / 480.0
            dx = int(float(cx_px) * sx)
            dy = int(float(cy_px) * sy)
            dx = max(0, min(dx, depth_w - 1))
            dy = max(0, min(dy, depth_h - 1))

            # Sample depth
            margin = 5
            region = self.depth_image[
                max(0, dy-margin):min(depth_h, dy+margin),
                max(0, dx-margin):min(depth_w, dx+margin)
            ]
            valid = region[region > 0]
            if len(valid) == 0:
                continue

            depth_mm = float(np.median(valid))
            depth_m  = depth_mm * self.depth_scale

            # Back-project to camera frame
            X_cam = (float(cx_px) - self.cx) * depth_m / self.fx
            Y_cam = (float(cy_px) - self.cy) * depth_m / self.fy
            Z_cam = depth_m

            # Publish in camera frame
            pt_cam = PointStamped()
            pt_cam.header.stamp    = self.get_clock().now().to_msg()
            pt_cam.header.frame_id = "global_camera_link"
            pt_cam.point.x = X_cam
            pt_cam.point.y = Y_cam
            pt_cam.point.z = Z_cam
            self.point_pub.publish(pt_cam)

            # Transform to robot base frame
            x_robot, y_robot, z_robot = X_cam, Y_cam, Z_cam  # fallback
            try:
                tf = self.tf_buffer.lookup_transform(
                    "base_link", "global_camera_link",
                    rclpy.time.Time(), timeout=rclpy.duration.Duration(seconds=0.1))
                pt_robot = tf2_geometry_msgs.do_transform_point(pt_cam, tf)
                x_robot = pt_robot.point.x
                y_robot = pt_robot.point.y
                z_robot = pt_robot.point.z

                self.robot_point_pub.publish(pt_robot)
                self.get_logger().info(
                    f'{label} robot frame: '
                    f'x={x_robot:.3f} y={y_robot:.3f} z={z_robot:.3f}'
                )
            except Exception as e:
                self.get_logger().warn(
                    f'TF camera→base_link failed: {e} '
                    f'(using camera frame coords)'
                )

            enriched.append({
                'label':      label,
                'confidence': conf,
                # camera frame
                'cx_3d': round(X_cam, 4),
                'cy_3d': round(Y_cam, 4),
                'cz_3d': round(Z_cam, 4),
                # robot frame
                'x_robot': round(x_robot, 4),
                'y_robot': round(y_robot, 4),
                'z_robot': round(z_robot, 4),
            })

        if enriched:
            out = String()
            out.data = json.dumps(enriched)
            self.det3d_pub.publish(out)


def main():
    rclpy.init()
    node = Depth3DNode()
    rclpy.spin(node)


if __name__ == "__main__":
    main()
