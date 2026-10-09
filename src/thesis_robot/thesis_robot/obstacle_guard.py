"""
Unknown-obstacle detection from depth (numpy + OpenCV, no ROS).

The detectors only know the object classes they were trained on. A black box, a stapler, a coat or a hand on the table
is invisible to them, so the scene graph has no entry and the arm's carry and finger-sweep checks cannot avoid it.
This module looks at the DEPTH instead: everything that stands above the table is an obstacle, whatever it is.

Pipeline (per camera frame):
  depth image -> 3D points in base_link (pinhole model + the camera's extrinsic from TF)
  -> drop the robot's own body (arm links and fingers, vectorised point_on_robot)
  -> top-down height-above-table grid (1 cm cells, tallest point per cell)
  -> threshold + morphological opening (cv2) -> connected components (cv2.connectedComponentsWithStats)
  -> per blob: footprint, tallest height, centre, minAreaRect
  -> subtract blobs the SCENE already explains (a known object's footprint, with margin) -> unknown obstacles.

Flat things (the ChArUco board, a sheet of paper, tape) stay below the height threshold and are ignored.
"""
import math

import cv2
import numpy as np

CELL_M = 0.01
MIN_HEIGHT_M = 0.03           # lower than this is table clutter (paper, board, cables), not an obstacle
MAX_HEIGHT_M = 0.70
MIN_AREA_M2 = 0.0008          # 8 cm^2: smaller blobs are depth noise or a cable
MIN_DEPTH_M, MAX_DEPTH_M = 0.30, 4.0
ARM_BODY_RADIUS_M = 0.12      # a little more than safety_geometry's: depth sees the whole link, not its origin
FINGER_BODY_RADIUS_M = 0.11       # the 2F-140 fingers and knuckles extend ~9 cm either side of the pad frames
KNOWN_MARGIN_M = 0.035        # a blob this close to a known object's footprint belongs to it


def clean_depth(depth, jump_m=0.04):
    """Zero the pixels next to a sharp depth step. At the edge of the dark gripper or of a box the sensor interpolates
    between near and far and produces 'flying pixels' hanging in mid-air; backprojected they look like a floating
    obstacle. A step of `jump_m` between neighbouring pixels cannot be a surface (a table tilted 60 degrees seen at
    1 m changes by ~1 cm per pixel), so such pixels and their neighbours are dropped."""
    d = np.asarray(depth, np.float32)
    valid = d > 0
    gx = cv2.Sobel(d, cv2.CV_32F, 1, 0, ksize=3) / 8.0
    gy = cv2.Sobel(d, cv2.CV_32F, 0, 1, ksize=3) / 8.0
    bad = (np.hypot(gx, gy) > jump_m / 2.0) & valid
    bad = cv2.dilate(bad.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
    out = d.copy()
    out[bad] = 0
    return out


def backproject(depth, K, R_bc, t_bc, stride=3, clean=True):
    """Depth image (metres, float; zeros = invalid) -> (N, 3) points in base_link. K = 3x3 intrinsics,
    (R_bc, t_bc) = pose of the camera optical frame in base_link."""
    if clean:
        depth = clean_depth(depth)
    h, w = depth.shape
    v, u = np.mgrid[0:h:stride, 0:w:stride]
    z = depth[v, u]
    ok = (z > MIN_DEPTH_M) & (z < MAX_DEPTH_M)
    u, v, z = u[ok].astype(float), v[ok].astype(float), z[ok]
    cam = np.column_stack([(u - K[0][2]) * z / K[0][0], (v - K[1][2]) * z / K[1][1], z])
    return cam @ np.asarray(R_bc, float).T + np.asarray(t_bc, float)


def _dist_to_segments(P, a, b):
    """Distance from each point in P (N, 3) to the segment a-b."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    ab = b - a
    L2 = float(ab @ ab)
    t = np.zeros(len(P)) if L2 < 1e-12 else np.clip(((P - a) @ ab) / L2, 0.0, 1.0)
    return np.linalg.norm(P - (a + t[:, None] * ab), axis=1)


def remove_robot(points, chain, fingers):
    """Drop points on the arm or the fingers. chain: [(x, y, z)] base -> flange; fingers: [(flange_xyz, pad_xyz)]."""
    if chain is None or len(points) == 0:
        return points
    keep = np.ones(len(points), bool)
    for a, b in zip(chain, chain[1:]):
        keep &= _dist_to_segments(points, a, b) >= ARM_BODY_RADIUS_M
    for a, b in fingers or []:
        keep &= _dist_to_segments(points, a, b) >= FINGER_BODY_RADIUS_M
    return points[keep]


def height_grid(points, table_z, x_range, y_range, cell=CELL_M):
    """Top-down grid of the tallest point above the table per cell: (H [ny, nx] metres, 0 where nothing was seen,
    origin (x0, y0))."""
    x0, x1 = x_range
    y0, y1 = y_range
    nx, ny = int(round((x1 - x0) / cell)), int(round((y1 - y0) / cell))
    H = np.zeros((ny, nx), np.float32)
    if len(points) == 0:
        return H, (x0, y0)
    ix = np.floor((points[:, 0] - x0) / cell).astype(int)
    iy = np.floor((points[:, 1] - y0) / cell).astype(int)
    h = points[:, 2] - table_z
    ok = (ix >= 0) & (ix < nx) & (iy >= 0) & (iy < ny) & (h > 0.0) & (h < MAX_HEIGHT_M)
    # tallest point per cell: write the heights in ascending order so the last (tallest) write to a cell wins. This
    # is ~50x faster than np.maximum.at, which made the live node take most of a CPU.
    idx, hh = iy[ok] * nx + ix[ok], h[ok].astype(np.float32)
    order = np.argsort(hh, kind='stable')
    H.reshape(-1)[idx[order]] = hh[order]
    return H, (x0, y0)


def find_blobs(H, origin, cell=CELL_M, min_height=MIN_HEIGHT_M, min_area=MIN_AREA_M2):
    """Connected regions taller than min_height: [{'x', 'y', 'height', 'area_m2', 'w', 'h', 'angle'}] (metres,
    base_link; w x h from the minimum-area rectangle)."""
    binary = (H >= min_height).astype(np.uint8) * 255
    k = np.ones((3, 3), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, k)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, k, iterations=2)         # bridge depth holes inside one object
    n, labels, stats, cents = cv2.connectedComponentsWithStats(binary, connectivity=8)
    out = []
    for i in range(1, n):
        area = stats[i, cv2.CC_STAT_AREA] * cell * cell
        if area < min_area:
            continue
        ys, xs = np.nonzero(labels == i)
        (rcx, rcy), (rw, rh), ang = cv2.minAreaRect(np.column_stack([xs, ys]).astype(np.float32))
        out.append({'x': origin[0] + (cents[i][0] + 0.5) * cell, 'y': origin[1] + (cents[i][1] + 0.5) * cell,
                    'height': float(H[ys, xs].max()), 'area_m2': float(area),
                    'w': float(min(rw, rh) * cell), 'h': float(max(rw, rh) * cell), 'angle': float(ang)})
    return out


def explained_by_known(blob, known, margin=KNOWN_MARGIN_M):
    """True if the blob is (mostly) a known object: its centre is inside a known footprint (+margin), or the blob is no
    bigger than that footprint and lies within it. known: [(x, y, radius_m)]."""
    r_blob = 0.5 * max(blob['w'], blob['h'])
    for kx, ky, kr in known:
        d = math.hypot(blob['x'] - kx, blob['y'] - ky)
        if d <= kr + margin:
            return True
        if d + r_blob <= kr + 2 * margin:
            return True
    return False


def unknown_obstacles(points, table_z, x_range, y_range, known, chain=None, fingers=None, **kw):
    """The whole pipeline from base_link points. Returns [{'x','y','height','area_m2','w','h','angle'}]."""
    pts = remove_robot(points, chain, fingers)
    H, origin = height_grid(pts, table_z, x_range, y_range)
    blobs = find_blobs(H, origin, **kw)
    return [b for b in blobs if not explained_by_known(b, known)]


def merge_across_cameras(per_camera, radius=0.06):
    """Several cameras report the same obstacle: merge detections within `radius`, keeping the tallest reading and
    counting how many cameras saw it. per_camera: {camera: [obstacle dicts]} -> [obstacle dict + 'seen_by']."""
    merged = []
    for cam, obs in per_camera.items():                       # (a 'confirmed' flag is added below: seen by 2+ cameras)
        for o in obs:
            for m in merged:
                if math.hypot(o['x'] - m['x'], o['y'] - m['y']) <= radius:
                    m['seen_by'].append(cam)
                    if o['height'] > m['height']:
                        m.update({k: o[k] for k in ('height', 'area_m2', 'w', 'h', 'angle')})
                    m['x'], m['y'] = (m['x'] + o['x']) / 2, (m['y'] + o['y']) / 2
                    break
            else:
                merged.append({**o, 'seen_by': [cam]})
    for m in merged:
        m['confirmed'] = len(set(m['seen_by'])) >= 2
    return merged
