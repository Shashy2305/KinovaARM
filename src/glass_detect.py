import cv2
import numpy as np
import pyrealsense2 as rs
from ultralytics import YOLO

# ── pixel_depth_to_robot_coord from BerkeleyAutomation/aruco_camera_calibration
def pixel_depth_to_robot_coord(pixel_x, pixel_y, depth, cam_to_robot_4x4, K):
    pixel_homog = np.array([pixel_x, pixel_y, 1.0]).reshape(3,1)
    pixel_homog = np.linalg.inv(K) @ pixel_homog
    pixel_homog = pixel_homog * depth
    pixel_homog = np.append(pixel_homog, 1.0).reshape(4,1)
    result = cam_to_robot_4x4 @ pixel_homog
    return result[:3, 0]

# ── YOLO ──────────────────────────────────────────────────────────
model = YOLO("/home/lab/runs/detect/glass_v1/weights/best.pt")
print("YOLO loaded:", model.names)

# ── ArUco marker layout ───────────────────────────────────────────
PHYSICAL_MARKERS = {
    0: {"coord": (0.2,  0.0),  "inner": 3},
    1: {"coord": (0.50, 0.0),  "inner": 0},
    2: {"coord": (0.2, -0.30), "inner": 2},
    3: {"coord": (0.5, -0.30), "inner": 1},
}

# ── Camera intrinsics (from test_aruco2.py) ───────────────────────
K = np.array(
    [[1.62926289e3, 0.0, 9.71234776e2],
     [0.0, 1.62748788e3, 5.30839748e2],
     [0.0, 0.0, 1.0]], dtype=np.float64)
dist_coeffs = np.array(
    [0.595836256, -7.01481761, -0.00206565224,
     0.0000525532496, 24.2404007], dtype=np.float32)

# ── cam_to_robot 4x4 matrix (from tf2_echo base_link->camera) ────
# R and T from your hand-eye calibration
R_cam2base = np.array([
    [ 0.014,  0.518, -0.855],
    [ 1.000, -0.015,  0.008],
    [-0.009, -0.855, -0.518]
])
T_cam2base = np.array([0.990, -0.130, 0.770])

# Build 4x4 homogeneous transform
cam_to_robot = np.eye(4)
cam_to_robot[:3,:3] = R_cam2base
cam_to_robot[:3, 3] = T_cam2base

MARKER_LENGTH_M = 0.05
CONF_THRESHOLD  = 0.1
SMOOTH_N        = 10   # average over N frames for stability

# Smoothing buffers
from collections import deque
buf_x = deque(maxlen=SMOOTH_N)
buf_y = deque(maxlen=SMOOTH_N)
buf_z = deque(maxlen=SMOOTH_N)
buf_cx = deque(maxlen=SMOOTH_N)
buf_cy = deque(maxlen=SMOOTH_N)
GLASS_W = 0.0313
GLASS_H = 0.0313

aruco_dict   = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
aruco_params = cv2.aruco.DetectorParameters_create()

# ── RealSense ─────────────────────────────────────────────────────
pipeline = rs.pipeline()
config   = rs.config()
config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
profile  = pipeline.start(config)
align    = rs.align(rs.stream.color)

def get_depth_at(depth_frame, cx, cy, r=15):
    h = depth_frame.get_height()
    w = depth_frame.get_width()
    vals = []
    for dy in range(-r, r):
        for dx in range(-r, r):
            nx, ny = cx+dx, cy+dy
            if 0 <= nx < w and 0 <= ny < h:
                d = depth_frame.get_distance(nx, ny)
                if d > 0:
                    vals.append(d)
    return float(np.median(vals)) if vals else 0.0

def draw_axes(frame, cx, cy, depth_m):
    """Draw TF axes at glass centre using actual 3D position."""
    tvec = np.array([[
        (cx - K[0,2]) * depth_m / K[0,0],
        (cy - K[1,2]) * depth_m / K[1,1],
        depth_m
    ]])
    rvec = np.zeros((1,3))
    cv2.drawFrameAxes(frame, K.astype(np.float32),
        dist_coeffs, rvec, tvec, 0.04, 3)

def draw_3d_box(frame, cx, cy, depth_m):
    """Project 3D bounding box onto image."""
    w, h, d = GLASS_W/2, GLASS_H/2, 0.04
    cam_x = (cx - K[0,2]) * depth_m / K[0,0]
    cam_y = (cy - K[1,2]) * depth_m / K[1,1]
    corners_3d = np.float32([
        [cam_x-w, cam_y-h, depth_m],
        [cam_x+w, cam_y-h, depth_m],
        [cam_x+w, cam_y+h, depth_m],
        [cam_x-w, cam_y+h, depth_m],
        [cam_x-w, cam_y-h, depth_m+d],
        [cam_x+w, cam_y-h, depth_m+d],
        [cam_x+w, cam_y+h, depth_m+d],
        [cam_x-w, cam_y+h, depth_m+d],
    ])
    pts2d, _ = cv2.projectPoints(
        corners_3d, np.zeros(3), np.zeros(3), K.astype(np.float32), dist_coeffs)
    pts = pts2d.reshape(-1,2).astype(int)
    # Bottom face
    for i,j in [(0,1),(1,2),(2,3),(3,0)]:
        cv2.line(frame,tuple(pts[i]),tuple(pts[j]),(0,165,255),2)
    # Top face
    for i,j in [(4,5),(5,6),(6,7),(7,4)]:
        cv2.line(frame,tuple(pts[i]),tuple(pts[j]),(0,200,255),2)
    # Verticals
    for i,j in [(0,4),(1,5),(2,6),(3,7)]:
        cv2.line(frame,tuple(pts[i]),tuple(pts[j]),(0,230,255),1)

print("Press Q to quit")

while True:
    frames  = pipeline.wait_for_frames()
    aligned = align.process(frames)
    color_frame = aligned.get_color_frame()
    depth_frame = aligned.get_depth_frame()
    if not color_frame or not depth_frame:
        continue

    frame = np.asanyarray(color_frame.get_data())
    gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    depth_colormap = cv2.applyColorMap(
        cv2.convertScaleAbs(
            np.asanyarray(depth_frame.get_data()), alpha=0.03),
        cv2.COLORMAP_JET)

    # ── ArUco ────────────────────────────────────────────────────
    corners, ids, _ = cv2.aruco.detectMarkers(
        gray, aruco_dict, parameters=aruco_params)
    aruco_depth = None
    if ids is not None and len(ids) >= 2:
        cv2.aruco.drawDetectedMarkers(frame, corners, ids)
        _, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
            corners, MARKER_LENGTH_M, K.astype(np.float32), dist_coeffs)
        flat = ids.flatten()
        ds = [tvecs[i][0][2] for i,m in enumerate(flat)
              if m in PHYSICAL_MARKERS]
        if ds:
            aruco_depth = float(np.mean(ds))

    # ── YOLO ─────────────────────────────────────────────────────
    results     = model.predict(frame, conf=CONF_THRESHOLD, verbose=False)
    glass_robot = None
    best_conf   = 0.0

    if results and len(results[0].boxes) > 0:
        for box in results[0].boxes:
            conf = float(box.conf[0])
            if conf > best_conf:
                best_conf = conf
                x1,y1,x2,y2 = box.xyxy[0].cpu().numpy()
                cx = int((x1+x2)/2)
                cy = int((y1+y2)/2)

                # Depth from RealSense depth camera
                depth_m = get_depth_at(depth_frame, cx, cy)
                depth_src = "RealSense"
                if depth_m == 0.0 and aruco_depth:
                    depth_m = aruco_depth
                    depth_src = "ArUco"

                if depth_m > 0:
                    # ── Convert pixel+depth → robot coords ───────
                    # Using BerkeleyAutomation pixel_depth_to_robot_coord
                    robot_pos = pixel_depth_to_robot_coord(
                        cx, cy, depth_m, cam_to_robot, K)

                    # Smooth position over N frames
                    buf_x.append(float(robot_pos[0]))
                    buf_y.append(float(robot_pos[1]))
                    buf_z.append(float(robot_pos[2]))
                    buf_cx.append(cx)
                    buf_cy.append(cy)
                    glass_robot = {
                        "x": float(np.mean(buf_x)),
                        "y": float(np.mean(buf_y)),
                        "z": float(np.mean(buf_z))
                    }
                    cx = int(np.mean(buf_cx))
                    cy = int(np.mean(buf_cy))

                    # ── Draw 3D bounding box ──────────────────────
                    draw_3d_box(frame, cx, cy, depth_m)

                    # ── Draw TF axes at glass centre ──────────────
                    draw_axes(frame, cx, cy, depth_m)

                    # ── Draw 2D box + label ───────────────────────
                    cv2.rectangle(frame,
                        (int(x1),int(y1)),(int(x2),int(y2)),
                        (0,165,255), 2)
                    cv2.putText(frame,
                        f"GLass {conf:.2f} {depth_m:.2f}m [{depth_src}]",
                        (int(x1), int(y1)-10),
                        cv2.FONT_HERSHEY_SIMPLEX,0.55,(0,165,255),2)

                    # ── Targeting reticle ─────────────────────────
                    cv2.circle(frame,(cx,cy),50,(0,255,65),2)
                    cv2.circle(frame,(cx,cy), 6,(0,255,65),-1)
                    for a in range(0,360,30):
                        r_=np.radians(a); r2=np.radians(a+15)
                        cv2.line(frame,
                            (cx+int(32*np.cos(r_)),cy+int(32*np.sin(r_))),
                            (cx+int(32*np.cos(r2)),cy+int(32*np.sin(r2))),
                            (0,255,65),1)

                    # ── Robot coordinates on image ────────────────
                    cv2.putText(frame,
                        f"robot: x={glass_robot['x']:.3f} "
                        f"y={glass_robot['y']:.3f} "
                        f"z={glass_robot['z']:.3f}m",
                        (int(x1), int(y2)+25),
                        cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,200,255),2)

                    print(f"GLass → robot: "
                          f"x={glass_robot['x']:.4f} "
                          f"y={glass_robot['y']:.4f} "
                          f"z={glass_robot['z']:.4f}  "
                          f"conf={conf:.2f}  depth={depth_m:.3f}m [{depth_src}]")

    # ── HUD ───────────────────────────────────────────────────────
    cv2.rectangle(frame,(0,0),(700,90),(0,0,0),-1)
    cv2.putText(frame,
        "GLASS 3D POSE — YOLO + RealSense Depth + Hand-Eye Calib",
        (10,22),cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,200,255),1)
    n = len(ids) if ids is not None else 0
    cv2.putText(frame,
        f"ArUco:{n}/4  "
        f"ArUcoDepth:{aruco_depth:.3f}m" if aruco_depth
        else f"ArUco:{n}/4  depth:--",
        (10,47),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,215,0),1)
    cv2.putText(frame,
        "TF: X=RED  Y=GREEN  Z=BLUE  |  3D box=CYAN",
        (10,68),cv2.FONT_HERSHEY_SIMPLEX,0.45,(200,200,200),1)
    if glass_robot:
        cv2.putText(frame,
            f"Robot: x={glass_robot['x']:.4f}  "
            f"y={glass_robot['y']:.4f}  "
            f"z={glass_robot['z']:.4f}",
            (10,88),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,255,65),1)
    else:
        cv2.putText(frame,"Glass not detected",
            (10,88),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,255),1)

    depth_resized = cv2.resize(depth_colormap,(640,360))
    frame_resized = cv2.resize(frame,(640,360))
    combined = np.hstack([frame_resized, depth_resized])
    cv2.imshow("Glass 3D Pose  |  Color + Depth", combined)

    if cv2.waitKey(1) == ord('q'):
        break

pipeline.stop()
cv2.destroyAllWindows()
