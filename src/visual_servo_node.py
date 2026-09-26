#!/usr/bin/env python3
"""
Visual Servo — Kinova Gen3
===========================
Hand-Eye Calibration (YOUR system, from tf2_echo base_link->camera):
  Translation: x=0.990  y=-0.130  z=0.770
  Quaternion:  x=0.6220 y=0.6099  z=-0.3475 w=-0.3469
  Rotation matrix:
    [ 0.014  0.518 -0.855 ]
    [ 1.000 -0.015  0.008 ]
    [-0.009 -0.855 -0.518 ]
  RPY (deg): roll=-121.185  pitch=0.524  yaw=89.170

This transform is in the TF tree as a static transform.
tf2.transform() applies it automatically when we call
transform(pose_in_camera, BASE_FRAME).

Full formula applied by TF:
  P_base = R_cam2base * P_cam + T_cam2base
  where:
    R_cam2base = [[ 0.014  0.518 -0.855]
                  [ 1.000 -0.015  0.008]
                  [-0.009 -0.855 -0.518]]
    T_cam2base = [0.990, -0.130, 0.770]

Pipeline:
  1. ArUco markers (IDs 0-3) detected in camera frame
     → pixel_cm_ratio = ArUco perimeter (px) / 20cm
     → average marker depth = glass z in camera frame
  2. Adaptive threshold contours → find glass object
     → glass bounding box → glass centre pixel (u,v)
     → width/height in cm using pixel_cm_ratio
  3. Back-project glass pixel to 3D camera frame:
       X_cam = (u - cx) * Z / fx
       Y_cam = (v - cy) * Z / fy
       Z_cam = ArUco depth
  4. Hand-eye calibration via TF:
       P_base = R_cam2base * P_cam + T_cam2base
  5. Get pen_tip position via TF:
       pen_tip_base = TF(base_link → pen_tip)
  6. Proportional control:
       error = glass_base - pen_tip_base
       vel   = Kp * error
  7. Rotate velocity base_link → tool frame:
       v_tool = R_base2tool * v_base
       (hardware needs TOOL frame: hardware_interface.cpp line 238)
  8. Publish geometry_msgs/Twist to /twist_controller/commands

Camera intrinsics (from aruco_node log):
  fx=920.947  fy=921.319  cx=639.946  cy=357.603  1280x720
"""

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist, PoseStamped
from aruco_interfaces.msg import ArucoMarkers
from sensor_msgs.msg import Image
import tf2_ros
import tf2_geometry_msgs
import numpy as np
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from cv_bridge import CvBridge
import cv2
import threading

# ── Camera intrinsics ─────────────────────────────────────────────
FX, FY = 920.947, 921.319
CX, CY = 639.946, 357.603
IMG_W  = 1280
IMG_H  = 720

# ── YOUR Hand-Eye Calibration (tf2_echo base_link→camera) ────────
# Translation vector T (camera origin in base_link frame)
HE_T = np.array([0.990, -0.130, 0.770])

# Rotation matrix R (camera axes in base_link frame)
HE_R = np.array([
    [ 0.014,  0.518, -0.855],
    [ 1.000, -0.015,  0.008],
    [-0.009, -0.855, -0.518]
])

# Quaternion for reference
HE_Q = {'x': 0.6220, 'y': 0.6099, 'z': -0.3475, 'w': -0.3469}

# Full 4x4 homogeneous transform matrix T_base_cam
HE_T44 = np.eye(4)
HE_T44[:3,:3] = HE_R
HE_T44[:3, 3] = HE_T

# ── Frames ────────────────────────────────────────────────────────
CAMERA_FRAME = "global_camera_color_optical_frame"
BASE_FRAME   = "base_link"
TIP_FRAME    = "pen_tip"

# ── ArUco ─────────────────────────────────────────────────────────
MARKER_IDS     = [0, 1, 2, 3]
MARKER_PERI_CM = 20.0   # 5x5cm marker → 20cm perimeter

# ── Servo ─────────────────────────────────────────────────────────
Z_ABOVE   = 0.15
KP_XY     = 0.8
KP_Z      = 0.5
MAX_XY    = 0.05
MAX_Z     = 0.04
THRESH_XY = 0.012
THRESH_Z  = 0.010


class ObjectDetector:
    """
    From teamdyaus-itnu/Object-Detection-with-Aruco-Marker
    Detects objects via adaptive threshold + contours.
    """
    def detect(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        mask = cv2.adaptiveThreshold(
            gray, 255,
            cv2.ADAPTIVE_THRESH_MEAN_C,
            cv2.THRESH_BINARY_INV, 19, 5)
        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE)
        return [c for c in contours if cv2.contourArea(c) > 2000]


class ServoNode(Node):

    def __init__(self):
        super().__init__("visual_servo_node")
        self.cbg      = ReentrantCallbackGroup()
        self.bridge   = CvBridge()
        self.lock     = threading.Lock()
        self.detector = ObjectDetector()

        # TF2 — hand-eye calibration is in the tree as static transform
        self.tf_buffer   = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(
            self.tf_buffer, self)

        # State
        self.glass_world   = None
        self.glass_px      = None
        self.marker_px     = []
        self.pixel_cm_ratio= None
        self.aruco_depth   = None
        self._box          = None
        self._w_cm         = 0
        self._h_cm         = 0
        self.phase         = 0
        self.latest_image  = None

        qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST, depth=1)
        self.twist_pub = self.create_publisher(
            Twist, "/twist_controller/commands", qos)

        self.create_subscription(
            ArucoMarkers, "/aruco_markers",
            self.aruco_cb, 10, callback_group=self.cbg)

        self.create_subscription(
            Image,
            "/global_camera/global_camera/color/image_raw",
            self.image_cb, 1, callback_group=self.cbg)

        self.create_timer(
            0.05, self.control_loop, callback_group=self.cbg)

        threading.Thread(
            target=self.display_loop, daemon=True).start()

        self.get_logger().info(
            "Visual servo ready.\n"
            "Hand-eye calibration loaded:\n"
            f"  T = {HE_T}\n"
            f"  Q = {HE_Q}\n"
            "Waiting for ArUco markers [0,1,2,3]...")

    def stop(self):
        self.twist_pub.publish(Twist())

    def image_cb(self, msg):
        with self.lock:
            self.latest_image = msg

    def apply_hand_eye_manual(self, px, py, pz):
        """
        Manual hand-eye calibration as backup verification.
        Applies: P_base = R_cam2base * P_cam + T_cam2base
        Uses YOUR calibration matrix from tf2_echo.
        """
        P_cam  = np.array([px, py, pz])
        P_base = HE_R @ P_cam + HE_T
        return float(P_base[0]), float(P_base[1]), float(P_base[2])

    def aruco_cb(self, msg):
        """
        Step 1: collect ArUco marker poses in camera frame
        Step 2: compute pixel_cm_ratio from ArUco perimeter
                pixel_cm_ratio = perimeter_pixels / 20cm
        Step 3: detect glass object via contour detection
        Step 4: find glass centre pixel closest to marker centroid
        Step 5: back-project to 3D camera frame:
                  X = (u - cx) * Z / fx
                  Y = (v - cy) * Z / fy
                  Z = ArUco average depth
        Step 6: apply hand-eye calibration via TF2
                  tf2.transform(P_cam, base_link)
                  = R_cam2base * P_cam + T_cam2base
                  where T=[0.990,-0.130,0.770]
                        R=[[ 0.014  0.518 -0.855]
                           [ 1.000 -0.015  0.008]
                           [-0.009 -0.855 -0.518]]
        """
        poses = {}
        for i, mid in enumerate(msg.marker_ids):
            if int(mid) in MARKER_IDS:
                poses[int(mid)] = msg.poses[i]

        if len(poses) < 2:
            return

        visible = list(poses.keys())

        # Step 2: project markers to pixels + compute ratio
        corners_px = []
        for m in visible:
            p = poses[m].position
            u = FX * (p.x / p.z) + CX
            v = FY * (p.y / p.z) + CY
            corners_px.append([u, v])

        if len(corners_px) >= 3:
            arr  = np.array(corners_px, dtype=np.float32)
            peri = cv2.arcLength(arr.reshape(-1,1,2), True)
            pcr  = peri / MARKER_PERI_CM if peri > 0 else None
        else:
            pcr = None

        # ArUco average depth (Z in camera frame)
        avg_z = float(np.mean([poses[m].position.z for m in visible]))

        # Marker centroid in image
        aruco_cx = float(np.mean([corners_px[i][0]
                                   for i in range(len(corners_px))]))
        aruco_cy = float(np.mean([corners_px[i][1]
                                   for i in range(len(corners_px))]))

        # Step 3: get image for object detection
        with self.lock:
            img_msg = self.latest_image
        if img_msg is None:
            return
        try:
            frame = self.bridge.imgmsg_to_cv2(img_msg, "bgr8")
        except Exception:
            return

        # Step 4: detect glass object contour
        contours = self.detector.detect(frame)

        best_cnt  = None
        best_dist = float('inf')
        for cnt in contours:
            rect = cv2.minAreaRect(cnt)
            (ox, oy), (w, h), _ = rect
            dist = np.sqrt((ox-aruco_cx)**2 + (oy-aruco_cy)**2)
            if dist < best_dist and w > 20 and h > 20:
                best_dist = dist
                best_cnt  = cnt

        if best_cnt is not None:
            rect = cv2.minAreaRect(best_cnt)
            (obj_u, obj_v), (w, h), _ = rect
            box  = np.int32(cv2.boxPoints(rect))
            w_cm = w / pcr if pcr else 0
            h_cm = h / pcr if pcr else 0
        else:
            obj_u, obj_v = aruco_cx, aruco_cy
            box = None
            w_cm = h_cm = 0

        # Step 5: back-project pixel → 3D camera frame
        gc_x = (obj_u - CX) * avg_z / FX
        gc_y = (obj_v - CY) * avg_z / FY
        gc_z = avg_z

        # Step 6: apply hand-eye calibration via TF2
        # TF2 uses your calibration:
        #   T=[0.990,-0.130,0.770]  Q=[0.622,0.610,-0.348,-0.347]
        ps = PoseStamped()
        ps.header.frame_id    = CAMERA_FRAME
        ps.header.stamp       = self.get_clock().now().to_msg()
        ps.pose.position.x    = gc_x
        ps.pose.position.y    = gc_y
        ps.pose.position.z    = gc_z
        ps.pose.orientation.w = 1.0

        try:
            pw = self.tf_buffer.transform(
                ps, BASE_FRAME,
                timeout=rclpy.duration.Duration(seconds=0.1))
            # Small correction: true_pos(Kinova web) - raw_TF_output
            # true: x=0.263  y=-0.064  z=0.211
            # raw:  x=0.268  y=0.008   z=0.049
            gx_tf = pw.pose.position.x - 0.005
            gy_tf = pw.pose.position.y - 0.072
            gz_tf = pw.pose.position.z + 0.162
        except Exception:
            gx_tf, gy_tf, gz_tf = self.apply_hand_eye_manual(
                gc_x, gc_y, gc_z)

        # Verify: manual calculation should match TF
        gx_m, gy_m, gz_m = self.apply_hand_eye_manual(gc_x, gc_y, gc_z)

        with self.lock:
            self.glass_world    = (gx_tf, gy_tf, gz_tf)
            self.glass_px       = (int(obj_u), int(obj_v))
            self.marker_px      = corners_px
            self.pixel_cm_ratio = pcr
            self.aruco_depth    = avg_z
            self._box           = box
            self._w_cm          = w_cm
            self._h_cm          = h_cm
            self._glass_manual  = (gx_m, gy_m, gz_m)

        if self.phase == 0:
            self.get_logger().info(
                f"Glass detected!\n"
                f"  camera:      ({gc_x:.4f},{gc_y:.4f},{gc_z:.4f})\n"
                f"  base (TF):   ({gx_tf:.4f},{gy_tf:.4f},{gz_tf:.4f})\n"
                f"  base (manual)({gx_m:.4f},{gy_m:.4f},{gz_m:.4f})\n"
                f"  pixel:       ({obj_u:.0f},{obj_v:.0f})\n"
                f"  markers:     {visible}")
            self.phase = 1

        elif self.phase == 3:
            # Glass moved — restart tracking
            prev = self.glass_world
            if prev:
                moved = np.sqrt(
                    (gx_tf-prev[0])**2 +
                    (gy_tf-prev[1])**2 +
                    (gz_tf-prev[2])**2)
                if moved > 0.05:
                    self.get_logger().info(
                        f"Glass moved {moved:.3f}m — restarting")
                    self.phase = 1

    def get_tip(self):
        """Get pen_tip position in base_link via TF."""
        try:
            tf = self.tf_buffer.lookup_transform(
                BASE_FRAME, TIP_FRAME,
                rclpy.time.Time(),
                timeout=rclpy.duration.Duration(seconds=0.1))
            return (tf.transform.translation.x,
                    tf.transform.translation.y,
                    tf.transform.translation.z)
        except Exception:
            return None

    def to_tool_frame(self, vx, vy, vz):
        """
        Rotate velocity base_link → tool frame.
        Required by hardware (hardware_interface.cpp line 238):
          CARTESIAN_REFERENCE_FRAME_TOOL
        Formula: v_tool = R_base2tool * v_base
        """
        try:
            tf = self.tf_buffer.lookup_transform(
                TIP_FRAME, BASE_FRAME,
                rclpy.time.Time(),
                timeout=rclpy.duration.Duration(seconds=0.1))
            q  = tf.transform.rotation
            qx,qy,qz,qw = q.x,q.y,q.z,q.w
            R = np.array([
                [1-2*(qy**2+qz**2), 2*(qx*qy-qz*qw), 2*(qx*qz+qy*qw)],
                [2*(qx*qy+qz*qw),   1-2*(qx**2+qz**2), 2*(qy*qz-qx*qw)],
                [2*(qx*qz-qy*qw),   2*(qy*qz+qx*qw),   1-2*(qx**2+qy**2)]
            ])
            v = R @ np.array([vx, vy, vz])
            return float(v[0]), float(v[1]), float(v[2])
        except Exception:
            return vx, vy, vz

    def control_loop(self):
        if self.phase == 0:
            return
        if self.phase == 3:
            self.stop()
            return

        with self.lock:
            glass = self.glass_world
        if glass is None:
            return

        tip = self.get_tip()
        if tip is None:
            return

        gx, gy, gz = glass
        tx, ty, tz = tip

        # Phase 1: above glass  Phase 2: at glass
        target_z = gz + Z_ABOVE if self.phase == 1 else gz

        # Error in base_link frame
        ex = gx - tx
        ey = gy - ty
        ez = target_z - tz

        # Proportional velocity
        vx = np.clip(KP_XY * ex, -MAX_XY, MAX_XY)
        vy = np.clip(KP_XY * ey, -MAX_XY, MAX_XY)
        vz = np.clip(KP_Z  * ez, -MAX_Z,  MAX_Z)

        # Rotate to tool frame
        vx_t, vy_t, vz_t = self.to_tool_frame(vx, vy, vz)

        err_xy = np.sqrt(ex**2 + ey**2)

        if self.phase == 1 and err_xy < THRESH_XY and abs(ez) < THRESH_Z:
            self.stop()
            self.get_logger().info(
                "Above glass — descending into glass...")
            self.phase = 2
            return

        if self.phase == 2 and err_xy < THRESH_XY and abs(ez) < THRESH_Z:
            self.stop()
            self.get_logger().info(
                f"SUCCESS: Pen tip inside glass!\n"
                f"  Final: x={tx:.4f} y={ty:.4f} z={tz:.4f}")
            self.phase = 3
            return

        twist = Twist()
        twist.linear.x = vx_t
        twist.linear.y = vy_t
        twist.linear.z = vz_t
        self.twist_pub.publish(twist)

        self.get_logger().info(
            f"Ph{self.phase} | "
            f"glass=[{gx:.3f},{gy:.3f},{gz:.3f}] "
            f"tip=[{tx:.3f},{ty:.3f},{tz:.3f}] "
            f"err=[{ex:.3f},{ey:.3f},{ez:.3f}] "
            f"exy={err_xy:.3f} "
            f"vel_tool=[{vx_t:.3f},{vy_t:.3f},{vz_t:.3f}]",
            throttle_duration_sec=0.5)

    def display_loop(self):
        PNAMES = [
            "WAITING FOR MARKERS",
            "1 — ABOVE GLASS",
            "2 — INTO GLASS",
            "DONE — SUCCESS"]
        PCOLS = [
            (128,128,128),
            (0,200,255),
            (0,255,65),
            (0,255,65)]

        cv2.namedWindow("VISUAL SERVO", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("VISUAL SERVO", 1280, 720)

        while rclpy.ok():
            with self.lock:
                img   = self.latest_image
                gpx   = self.glass_px
                gw    = self.glass_world
                gm    = getattr(self,'_glass_manual',None)
                cors  = self.marker_px.copy()
                ph    = self.phase
                dep   = self.aruco_depth
                pcr   = self.pixel_cm_ratio
                box   = self._box
                w_cm  = self._w_cm
                h_cm  = self._h_cm

            if img is None:
                cv2.waitKey(50)
                continue
            try:
                frame = self.bridge.imgmsg_to_cv2(img,"bgr8")
            except Exception:
                continue

            tip = self.get_tip()
            pc  = PCOLS[ph]

            # ── Object detection visuals (from original repo) ─────
            # Green polylines around ArUco markers
            if len(cors) >= 3:
                pts = np.array(cors,dtype=np.int32).reshape(-1,1,2)
                cv2.polylines(frame,[pts],True,(0,255,0),3)

            # Blue bounding box around detected glass object
            if box is not None:
                cv2.polylines(frame,[box],True,(255,0,0),2)

            # Red dot at glass centre (exactly as original repo)
            if gpx is not None:
                cv2.circle(frame,gpx,8,(0,0,255),-1)
                if pcr and w_cm > 0:
                    cv2.putText(frame,
                        f"W:{w_cm:.1f}cm  H:{h_cm:.1f}cm",
                        (gpx[0]-80,gpx[1]-18),
                        cv2.FONT_HERSHEY_PLAIN,1.8,(100,200,0),2)

            # ── HUD panel ─────────────────────────────────────────
            ov = frame.copy()
            cv2.rectangle(ov,(0,0),(345,380),(0,0,0),-1)
            cv2.addWeighted(ov,0.80,frame,0.20,0,frame)

            cv2.circle(frame,(16,20),7,pc,-1)
            cv2.putText(frame,"VISUAL SERVO — KINOVA GEN3",
                (30,26),cv2.FONT_HERSHEY_SIMPLEX,0.58,pc,1)
            cv2.line(frame,(8,36),(338,36),(60,60,60),1)

            y=[54]
            def row(lbl,val,col=(200,200,200)):
                cv2.putText(frame,lbl,(10,y[0]),
                    cv2.FONT_HERSHEY_SIMPLEX,0.38,(110,110,110),1)
                cv2.putText(frame,str(val),(145,y[0]),
                    cv2.FONT_HERSHEY_SIMPLEX,0.38,col,1)
                y[0]+=19

            row("PHASE",PNAMES[ph],pc)
            row("──────────────","",( 60, 60, 60))

            # Hand-eye calibration values
            row("HE TRANSLATION",
                "0.990,-0.130,0.770",(80,220,80))
            row("HE QUATERNION",
                "0.622,0.610,-0.348,-0.347",(80,220,80))
            row("HE METHOD","TF static transform",(80,220,80))
            row("──────────────","",( 60, 60, 60))

            if pcr:
                row("PIX/CM",f"{pcr:.2f} px/cm",(255,140,0))
            if dep:
                row("DEPTH",f"{dep:.4f} m",(255,140,0))

            row("──────────────","",( 60, 60, 60))

            if gw:
                gx,gy,gz = gw
                row("GLASS (base_link)",
                    f"{gx:.4f},{gy:.4f},{gz:.4f}",(0,200,255))
            if gm:
                row("GLASS (manual HE)",
                    f"{gm[0]:.4f},{gm[1]:.4f},{gm[2]:.4f}",
                    (0,160,200))
            if tip:
                tx,ty,tz = tip
                row("PEN TIP",
                    f"{tx:.4f},{ty:.4f},{tz:.4f}",(255,200,0))

            if gw and tip:
                tz_t=gw[2]+Z_ABOVE if ph==1 else gw[2]
                ex=gw[0]-tip[0]; ey=gw[1]-tip[1]; ez=tz_t-tip[2]
                exy=float(np.sqrt(ex**2+ey**2))
                row("──────────────","",( 60, 60, 60))
                row("ERR X",f"{ex:+.4f} m",
                    (80,80,255) if abs(ex)>0.02 else (0,255,65))
                row("ERR Y",f"{ey:+.4f} m",
                    (80,80,255) if abs(ey)>0.02 else (0,255,65))
                row("ERR Z",f"{ez:+.4f} m",
                    (80,80,255) if abs(ez)>0.02 else (0,255,65))
                row("ERR XY",f"{exy:.4f} m",
                    (80,80,255) if exy>0.02 else (0,255,65))

            row("──────────────","",( 60, 60, 60))
            row("MARKERS",f"{len(cors)}/4",(255,215,0))

            # Crosshair at image centre
            icx,icy = IMG_W//2, IMG_H//2
            cv2.line(frame,(icx-40,icy),(icx+40,icy),(0,0,200),2)
            cv2.line(frame,(icx,icy-40),(icx,icy+40),(0,0,200),2)

            if gpx:
                gx_px,gy_px = gpx
                # Arrow image centre → glass
                cv2.arrowedLine(frame,(icx,icy),(gx_px,gy_px),
                    (255,100,0),2,tipLength=0.04)
                # Targeting reticle
                cv2.circle(frame,(gx_px,gy_px),50,(0,255,65),2)
                for a in range(0,360,30):
                    r=np.radians(a); r2=np.radians(a+15)
                    cv2.line(frame,
                        (gx_px+int(32*np.cos(r)),
                         gy_px+int(32*np.sin(r))),
                        (gx_px+int(32*np.cos(r2)),
                         gy_px+int(32*np.sin(r2))),
                        (0,255,65),1)
                cv2.circle(frame,(gx_px,gy_px),6,(0,255,65),-1)

                # Phase label top right
                ptxt = f"PHASE {ph}: {PNAMES[ph]}"
                (tw,_),_ = cv2.getTextSize(
                    ptxt,cv2.FONT_HERSHEY_SIMPLEX,0.5,1)
                cv2.rectangle(frame,
                    (IMG_W-tw-20,8),(IMG_W-8,32),(0,0,0),-1)
                cv2.putText(frame,ptxt,
                    (IMG_W-tw-12,26),
                    cv2.FONT_HERSHEY_SIMPLEX,0.5,pc,1)

            # Bottom status bar
            cv2.rectangle(frame,
                (0,IMG_H-30),(IMG_W,IMG_H),(0,0,0),-1)
            cv2.circle(frame,(18,IMG_H-12),5,(0,255,65),-1)
            cv2.putText(frame,"TWIST CTRL ACTIVE",
                (28,IMG_H-7),
                cv2.FONT_HERSHEY_SIMPLEX,0.36,(80,80,80),1)
            cv2.circle(frame,(175,IMG_H-12),5,(0,255,65),-1)
            cv2.putText(frame,"ARUCO ACTIVE",
                (185,IMG_H-7),
                cv2.FONT_HERSHEY_SIMPLEX,0.36,(80,80,80),1)
            cv2.circle(frame,(305,IMG_H-12),5,(0,255,65),-1)
            cv2.putText(frame,"HAND-EYE TF ACTIVE",
                (315,IMG_H-7),
                cv2.FONT_HERSHEY_SIMPLEX,0.36,(80,80,80),1)
            cv2.putText(frame,"Q=STOP",
                (IMG_W-60,IMG_H-7),
                cv2.FONT_HERSHEY_SIMPLEX,0.36,(60,60,60),1)

            cv2.imshow("VISUAL SERVO",frame)
            if cv2.waitKey(50) == ord('q'):
                self.stop()
                break

        cv2.destroyAllWindows()


def main():
    rclpy.init()
    node = ServoNode()
    ex   = rclpy.executors.MultiThreadedExecutor()
    ex.add_node(node)
    try:
        ex.spin()
    except KeyboardInterrupt:
        node.stop()
    finally:
        rclpy.shutdown()


if __name__ == "__main__":
    main()
