#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from aruco_interfaces.msg import ArucoMarkers
from sensor_msgs.msg import Image
from pymoveit2 import MoveIt2
from rclpy.callback_groups import ReentrantCallbackGroup
from cv_bridge import CvBridge
import cv2, threading, time, numpy as np
import tf2_ros, tf2_geometry_msgs

CAMERA_FRAME = "global_camera_color_optical_frame"
BASE_FRAME   = "base_link"
MARKER_IDS   = [0, 1, 2, 3]
FX,FY,CX,CY  = 920.947, 921.319, 639.946, 357.603

# Confirmed glass position (from Kinova web interface)
GLASS_X =  0.263
GLASS_Y = -0.064
GLASS_Z =  0.211
Z_ABOVE =  0.15

class DemoNode(Node):
    def __init__(self):
        super().__init__("demo_node")
        self.cbg    = ReentrantCallbackGroup()
        self.bridge = CvBridge()
        self.lock   = threading.Lock()
        self.latest_image = None
        self.glass_px     = None
        self.marker_px    = []
        self.ready        = threading.Event()

        self.tf_buffer   = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        self.moveit2 = MoveIt2(
            node=self,
            joint_names=[f"joint_{i}" for i in range(1,8)],
            base_link_name="base_link",
            end_effector_name="pen_tip",
            group_name="manipulator",
            callback_group=self.cbg,
        )
        self.moveit2.max_velocity     = 0.05
        self.moveit2.max_acceleration = 0.05

        self.create_subscription(Image,
            "/global_camera/global_camera/color/image_raw",
            self.image_cb, 1, callback_group=self.cbg)

        self.create_subscription(ArucoMarkers,
            "/aruco_markers", self.aruco_cb, 10,
            callback_group=self.cbg)

        threading.Thread(target=self.display_loop, daemon=True).start()
        self.get_logger().info("Waiting for ArUco markers...")

    def image_cb(self, msg):
        with self.lock:
            self.latest_image = msg

    def aruco_cb(self, msg):
        poses = {}
        for i, mid in enumerate(msg.marker_ids):
            if int(mid) in MARKER_IDS:
                poses[int(mid)] = msg.poses[i]
        if len(poses) < 2:
            return
        visible = list(poses.keys())
        px_list = []
        for m in visible:
            p = poses[m].position
            u = int(FX*(p.x/p.z)+CX)
            v = int(FY*(p.y/p.z)+CY)
            px_list.append((u,v))
        cx = int(np.mean([x[0] for x in px_list]))
        cy = int(np.mean([x[1] for x in px_list]))
        with self.lock:
            self.glass_px   = (cx, cy)
            self.marker_px  = px_list
        self.ready.set()

    def move_to(self, label, x, y, z):
        self.get_logger().info(f"Moving {label}: x={x:.3f} y={y:.3f} z={z:.3f}")
        ps = PoseStamped()
        ps.header.frame_id    = BASE_FRAME
        ps.header.stamp       = self.get_clock().now().to_msg()
        ps.pose.position.x    = x
        ps.pose.position.y    = y
        ps.pose.position.z    = z
        ps.pose.orientation.x = 1.0
        ps.pose.orientation.w = 0.0
        self.moveit2.move_to_pose(pose=ps.pose, frame_id=BASE_FRAME,
            cartesian=True, cartesian_max_step=0.005)
        ok = self.moveit2.wait_until_executed()
        self.get_logger().info(f"{label}: {'SUCCESS' if ok else 'FAILED'}")
        return ok

    def display_loop(self):
        cv2.namedWindow("DEMO — GLASS DETECTION", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("DEMO — GLASS DETECTION", 1280, 720)
        while rclpy.ok():
            with self.lock:
                img = self.latest_image
                gpx = self.glass_px
                cors = self.marker_px.copy()
            if img is None:
                cv2.waitKey(50); continue
            try:
                frame = self.bridge.imgmsg_to_cv2(img, "bgr8")
            except:
                continue
            # Green lines around markers
            if len(cors) >= 2:
                pts = np.array(cors, dtype=np.int32).reshape(-1,1,2)
                cv2.polylines(frame,[pts],True,(0,255,0),3)
            for c in cors:
                cv2.circle(frame, c, 8, (0,255,0), -1)
            # Glass centre
            if gpx:
                cv2.circle(frame, gpx, 50, (0,255,65), 2)
                cv2.circle(frame, gpx,  6, (0,255,65), -1)
                cv2.putText(frame, "GLASS CENTRE",
                    (gpx[0]+15, gpx[1]-15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,65), 2)
            # HUD
            cv2.rectangle(frame,(0,0),(350,120),(0,0,0),-1)
            cv2.putText(frame,"KINOVA GEN3 — GLASS INSERT DEMO",
                (10,25),cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,200,255),1)
            cv2.putText(frame,f"Glass: x={GLASS_X} y={GLASS_Y} z={GLASS_Z}",
                (10,55),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,255),1)
            cv2.putText(frame,f"Markers: {len(cors)}/4 detected",
                (10,80),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,215,0),1)
            cv2.putText(frame,"ArUco + Hand-Eye Calibration",
                (10,105),cv2.FONT_HERSHEY_SIMPLEX,0.4,(80,180,80),1)
            cv2.imshow("DEMO — GLASS DETECTION", frame)
            cv2.waitKey(50)
        cv2.destroyAllWindows()

def main():
    rclpy.init()
    node = DemoNode()
    executor = rclpy.executors.MultiThreadedExecutor()
    executor.add_node(node)
    spin_thread = threading.Thread(target=executor.spin, daemon=True)
    spin_thread.start()

    node.get_logger().info("Waiting for marker detection...")
    node.ready.wait()
    node.get_logger().info("Markers detected! Waiting 8s for MoveIt...")
    time.sleep(8.0)

    print("\n>>> Detection window shows glass centre.")
    print(">>> Press ENTER to move arm above glass, Ctrl+C to abort: ", end="", flush=True)
    try:
        input()
    except KeyboardInterrupt:
        rclpy.shutdown(); return

    if node.move_to("Above glass", GLASS_X, GLASS_Y, GLASS_Z + Z_ABOVE):
        print(">>> Above glass! Press ENTER to descend into glass: ", end="", flush=True)
        try:
            input()
        except KeyboardInterrupt:
            rclpy.shutdown(); return
        if node.move_to("Into glass", GLASS_X, GLASS_Y, GLASS_Z):
            node.get_logger().info("=== SUCCESS: Pen tip inside glass! ===")

    rclpy.shutdown()

if __name__ == "__main__":
    main()
