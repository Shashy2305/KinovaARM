"""
The fused point cloud of the 3D Fusion tab: pure numpy, no ROS (so it can be tested with synthetic cameras).

What the old view did: decode every camera's driver PointCloud2 (up to 920k points), keep a RANDOM 3000 of them per frame, no filtering. The picture
flickered (different random points every frame), was sparse, showed the whole room, and decoding cost most of a CPU core.

What this does instead, per camera:
  1. a FIXED pixel grid (the same pixels every frame, so nothing flickers) taken from the aligned depth image, about GRID_POINTS points,
  2. depth = median of the last few frames at those pixels (temporal smoothing; a pixel needs valid depth in most of them),
  3. range limits and a flying-pixel filter (a pixel whose depth jumps against its grid neighbours is an edge artefact, not a surface),
  4. back-projection with the camera's intrinsics, then the camera's calibration (base_link <- camera) and a crop to the workspace box.
Then all cameras are merged by a VOXEL grid (deterministic): one point per voxel with the mean position and colour, plus a bit mask of the cameras
that saw it. Colour modes: 'rgb' (the pictures), 'camera' (one colour per camera, overlap bright: a mis-calibrated camera shows as a separate layer),
'overlap' (how many cameras agree: 1 grey, 2 yellow, 3+ green). stats() says how well each pair of cameras overlaps and the median distance between them.
"""
import numpy as np

GRID_POINTS = 70000          # target number of grid points per camera
DEPTH_MIN_M, DEPTH_MAX_M = 0.30, 3.0
FLY_REL = 0.04               # a grid pixel whose depth differs from a neighbour's by more than 4% of its depth, on both sides, is a flying pixel
MIN_VALID_FRAMES = 2         # of the last frames, a pixel needs this many valid depths
VOXEL_M = 0.010
CROP_MIN = np.array([-0.45, -0.95, -0.12])
CROP_MAX = np.array([1.15, 0.95, 1.20])
# bit per camera in the voxel mask and its display colour for the 'camera' mode
CAMERA_BITS = {'realsense': 1, 'realsense2': 2, 'realsense3': 4, 'wrist': 8}
CAMERA_COLORS = {'realsense': (1.0, 0.35, 0.30), 'realsense2': (0.30, 0.90, 0.40), 'realsense3': (0.35, 0.55, 1.0), 'wrist': (1.0, 0.85, 0.30)}


def grid_stride(width, height, target=GRID_POINTS):
    return max(2, int(round(np.sqrt(width * height / float(target)))))


def sample_depth(depth_frames_mm, stride):
    """depth in metres on the fixed grid, shape (gh, gw): the median over the frames, NaN where fewer than MIN_VALID_FRAMES are valid.
    depth_frames_mm: list of 2D uint16 arrays (millimetres, 0 = no reading), all the same size."""
    h, w = depth_frames_mm[0].shape
    v = np.arange(stride // 2, h, stride)
    u = np.arange(stride // 2, w, stride)
    stack = np.stack([f[np.ix_(v, u)].astype(np.float32) / 1000.0 for f in depth_frames_mm])
    stack[stack <= 0] = np.nan
    valid = np.sum(~np.isnan(stack), axis=0)
    with np.errstate(all='ignore'):
        med = np.nanmedian(stack, axis=0)
    med[valid < min(MIN_VALID_FRAMES, len(depth_frames_mm))] = np.nan
    med[(med < DEPTH_MIN_M) | (med > DEPTH_MAX_M)] = np.nan
    return med, v, u


def remove_flying_pixels(z):
    """NaN out grid pixels whose depth jumps against BOTH horizontal or BOTH vertical neighbours (depth-edge ghosts between a surface and the background)."""
    out = z.copy()
    pad = np.pad(z, 1, mode='edge')
    up, down, left, right = pad[:-2, 1:-1], pad[2:, 1:-1], pad[1:-1, :-2], pad[1:-1, 2:]
    tol = FLY_REL * z
    with np.errstate(invalid='ignore'):
        bad_v = (np.abs(z - up) > tol) & (np.abs(z - down) > tol)
        bad_h = (np.abs(z - left) > tol) & (np.abs(z - right) > tol)
        # an isolated pixel also has NaN neighbours: those count as "different"
        iso = (np.isnan(up).astype(int) + np.isnan(down) + np.isnan(left) + np.isnan(right)) >= 3
    out[bad_v | bad_h | iso] = np.nan
    return out


def camera_points(depth_frames_mm, color_rgb, K, T_base_cam, stride=None):
    """(xyz in base_link (N,3) float32, rgb 0-1 (N,3) float32) of one camera, filtered and cropped to the workspace."""
    h, w = depth_frames_mm[0].shape
    stride = stride or grid_stride(w, h)
    z, v, u = sample_depth(depth_frames_mm, stride)
    z = remove_flying_pixels(z)
    uu, vv = np.meshgrid(u, v)
    ok = ~np.isnan(z)
    if not ok.any():
        return np.zeros((0, 3), np.float32), np.zeros((0, 3), np.float32)
    zc = z[ok]
    x = (uu[ok] - K[0, 2]) * zc / K[0, 0]
    y = (vv[ok] - K[1, 2]) * zc / K[1, 1]
    pc = np.column_stack([x, y, zc])
    pb = pc @ T_base_cam[:3, :3].T + T_base_cam[:3, 3]
    # colour at the grid pixels (the aligned depth image and the colour image share a size; if not, scale the pixel indices)
    ch, cw = color_rgb.shape[:2]
    cu = np.clip((uu[ok] * cw / w).astype(int), 0, cw - 1)
    cv = np.clip((vv[ok] * ch / h).astype(int), 0, ch - 1)
    rgb = color_rgb[cv, cu].astype(np.float32) / 255.0
    keep = np.all((pb >= CROP_MIN) & (pb <= CROP_MAX), axis=1)
    return pb[keep].astype(np.float32), rgb[keep]


def voxel_merge(clouds, voxel=VOXEL_M):
    """clouds: {camera: (xyz, rgb)}. Returns (xyz mean (M,3), rgb mean (M,3), mask (M,) int of camera bits). Deterministic: no randomness anywhere."""
    xs, cs, ms = [], [], []
    for cam, (p, c) in clouds.items():
        if len(p):
            xs.append(p)
            cs.append(c)
            ms.append(np.full(len(p), CAMERA_BITS.get(cam, 16), dtype=np.int32))
    if not xs:
        return np.zeros((0, 3), np.float32), np.zeros((0, 3), np.float32), np.zeros(0, np.int32)
    p, c, m = np.vstack(xs), np.vstack(cs), np.concatenate(ms)
    ijk = np.floor((p - CROP_MIN) / voxel).astype(np.int64)
    dims = np.ceil((CROP_MAX - CROP_MIN) / voxel).astype(np.int64) + 2
    key = (ijk[:, 0] * dims[1] + ijk[:, 1]) * dims[2] + ijk[:, 2]
    order = np.argsort(key, kind='stable')
    key_s = key[order]
    starts = np.flatnonzero(np.r_[True, key_s[1:] != key_s[:-1]])
    counts = np.diff(np.r_[starts, len(key_s)]).astype(np.float32)
    mean_p = np.add.reduceat(p[order], starts, axis=0) / counts[:, None]
    mean_c = np.add.reduceat(c[order], starts, axis=0) / counts[:, None]
    mask = np.bitwise_or.reduceat(m[order], starts)
    return mean_p.astype(np.float32), mean_c.astype(np.float32), mask


_POPCOUNT = np.array([bin(i).count('1') for i in range(32)])


def colorize(rgb, mask, mode):
    if mode == 'camera':
        out = np.zeros((len(mask), 3), np.float32)
        for cam, bit in CAMERA_BITS.items():
            sel = mask == bit
            out[sel] = CAMERA_COLORS[cam]
        multi = (mask & (mask - 1)) != 0                       # more than one camera bit
        out[multi] = (1.0, 1.0, 1.0)
        return out
    if mode == 'overlap':
        n = _POPCOUNT[np.minimum(mask, 31)]
        out = np.full((len(mask), 3), 0.35, np.float32)
        out[n == 2] = (1.0, 0.85, 0.2)
        out[n >= 3] = (0.2, 0.95, 0.35)
        return out
    return rgb


def pair_stats(clouds, max_nn_m=0.10, max_points=15000):
    """{ 'a|b': {'median_nn_cm', 'overlap_pct'} }: for the points of camera a inside the region camera b also covers, the median distance to b's nearest point
    (a few cm = the calibrations agree) and the share of those shared-region points that have a partner within 3 cm. Points of a with no b point within max_nn_m
    are outside b's view and ignored for the median."""
    from scipy.spatial import cKDTree
    names = [n for n, (p, _) in clouds.items() if len(p) > 200]
    out = {}
    trees = {}
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            pa, pb = clouds[a][0], clouds[b][0]
            sa = pa[:: max(1, len(pa) // max_points)]
            sb = pb[:: max(1, len(pb) // max_points)]
            tb = trees.setdefault(b, cKDTree(sb))
            ta = trees.setdefault(a, cKDTree(sa))
            d_ab, _ = tb.query(sa)
            d_ba, _ = ta.query(sb)
            d = np.concatenate([d_ab[d_ab < max_nn_m], d_ba[d_ba < max_nn_m]])
            if len(d) < 100:
                out[f'{a}|{b}'] = {'median_nn_cm': None, 'overlap_pct': 0.0, 'shared_points': int(len(d))}
                continue
            out[f'{a}|{b}'] = {'median_nn_cm': round(float(np.median(d)) * 100, 2), 'overlap_pct': round(100.0 * float((d < 0.03).mean()), 1),
                               'shared_points': int(len(d))}
    return out
