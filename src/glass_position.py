import cv2
import numpy as np
import pyrealsense2 as rs

# ── ArUco marker layout (from test_aruco2.py) ─────────────────────
PHYSICAL_MARKERS = {
    0: {"coord": (0.2,  0.0),  "inner": 3},
    1: {"coord": (0.50, 0.0),  "inner": 0},
    2: {"coord": (0.2, -0.30), "inner": 2},
    3: {"coord": (0.5, -0.30), "inner": 1},
}

# ── Camera calibration (from test_aruco2.py) ──────────────────────
camera_matrix = np.array(
    [[1.62926289e3, 0.0, 9.71234776e2],
     [0.0, 1.62748788e3, 5.30839748e2],
     [0.0, 0.0, 1.0]], dtype=np.float32)
dist_coeffs = np.array(
    [0.595836256, -7.01481761, -0.00206565224,
     0.0000525532496, 24.2404007], dtype=np.float32)

# ── Hand-Eye Calibration (from tf2_echo base_link -> camera) ──────
HE_R = np.array([
    [ 0.014,  0.518, -0.855],
    [ 1.000, -0.015,  0.008],
    [-0.009, -0.855, -0.518]
])
HE_T = np.array([0.990, -0.130, 0.770])

MARKER_LENGTH_M = 0.05
aruco_dict   = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
aruco_params = cv2.aruco.DetectorParameters()

pipeline = rs.pipeline()
config   = rs.config()
config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
pipeline.start(config)

def build_workspace_transform(corners, ids):
    if ids is None:
        return None
    src, dst = [], []
    flat = ids.flatten()
    for mid, spec in PHYSICAL_MARKERS.items():
        matches = np.where(flat == mid)[0]
        if matches.size == 0:
            return None
        src.append(corners[matches[0]][0][spec["inner"]])
        dst.append(spec["coord"])
    return cv2.getPerspectiveTransform(
        np.float32(src), np.float32(dst))

print("Press Q to quit")
glass_robot = None

while True:
    frames      = pipeline.wait_for_frames()
    color_frame = frames.get_color_frame()
    if not color_frame:
        continue

    frame = np.asanyarray(color_frame.get_data())
    gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    corners, ids, _ = cv2.aruco.detectMarkers(
        gray, aruco_dict, parameters=aruco_params)

    marker_centres_px = []
    depths            = []
    glass_robot       = None

    if ids is not None and len(ids) >= 2:
        cv2.aruco.drawDetectedMarkers(frame, corners, ids)

        rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
            corners, MARKER_LENGTH_M, camera_matrix, dist_coeffs)

        flat = ids.flatten()
        for i, mid in enumerate(flat):
            if mid in PHYSICAL_MARKERS:
                cx = int(corners[i][0][:, 0].mean())
                cy = int(corners[i][0][:, 1].mean())
                marker_centres_px.append((cx, cy))
                depths.append(tvecs[i][0][2])

                cv2.putText(frame,
                    f"ID{mid} Z={tvecs[i][0][2]:.2f}m",
                    (cx, cy - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)

        if len(marker_centres_px) >= 2:
            glass_px_x = int(np.mean([p[0] for p in marker_centres_px]))
            glass_px_y = int(np.mean([p[1] for p in marker_centres_px]))
            glass_depth = float(np.mean(depths))

            glass_cam_x = (glass_px_x - camera_matrix[0,2]) * glass_depth / camera_matrix[0,0]
            glass_cam_y = (glass_px_y - camera_matrix[1,2]) * glass_depth / camera_matrix[1,1]
            glass_cam_z = glass_depth

            P_cam   = np.array([glass_cam_x, glass_cam_y, glass_cam_z])
            P_robot = HE_R @ P_cam + HE_T

            glass_robot = {
                "x": float(P_robot[0]),
                "y": float(P_robot[1]),
                "z": float(P_robot[2])
            }

            cv2.circle(frame, (glass_px_x, glass_px_y), 50, (0,255,65), 2)
            cv2.circle(frame, (glass_px_x, glass_px_y),  6, (0,255,65), -1)

            for a in range(0, 360, 30):
                r  = np.radians(a)
                r2 = np.radians(a + 15)
                cv2.line(frame,
                    (glass_px_x + int(32*np.cos(r)),
                     glass_px_y + int(32*np.sin(r))),
                    (glass_px_x + int(32*np.cos(r2)),
                     glass_px_y + int(32*np.sin(r2))),
                    (0,255,65), 1)

            cv2.putText(frame, "GLASS",
                (glass_px_x + 15, glass_px_y - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,65), 2)

            cv2.putText(frame,
                f"cam:   x={glass_cam_x:.3f} y={glass_cam_y:.3f} z={glass_cam_z:.3f}m",
                (glass_px_x - 150, glass_px_y + 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,200,0), 1)

            cv2.putText(frame,
                f"robot: x={glass_robot['x']:.3f} y={glass_robot['y']:.3f} z={glass_robot['z']:.3f}m",
                (glass_px_x - 150, glass_px_y + 55),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,200,255), 2)

    cv2.rectangle(frame, (0,0), (500, 80), (0,0,0), -1)
    cv2.putText(frame, "GLASS POSITION DETECTOR",
        (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0,200,255), 1)
    cv2.putText(frame,
        f"Markers: {len(ids) if ids is not None else 0}/4 detected",
        (10, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,215,0), 1)
    if glass_robot:
        cv2.putText(frame,
            f"Robot: x={glass_robot['x']:.4f} y={glass_robot['y']:.4f} z={glass_robot['z']:.4f}",
            (10, 68), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,65), 1)

    cv2.imshow("Glass Position", frame)
    if cv2.waitKey(1) == ord('q'):
        break

pipeline.stop()
cv2.destroyAllWindows()

if glass_robot:
    print(f"\nFinal glass position in robot frame:")
    print(f"  X = {glass_robot['x']:.4f} m")
    print(f"  Y = {glass_robot['y']:.4f} m")
    print(f"  Z = {glass_robot['z']:.4f} m")
