"""
Classical shape analysis of one detected object (OpenCV + numpy, no ROS).

The detectors give a class and a box (and, from the segmentation model, a mask). A parallel gripper also needs to
know HOW the object is shaped: how wide it is across each possible closing direction, how flat the sides it would
touch are, whether it has a handle, and which way its straight edges run. This module computes that from the image
with several complementary techniques and cross-checks them:

  contour          segmentation polygon -> mask; refined by GrabCut inside the box; Canny edges as a cross-check
  minAreaRect      length, width and angle of the tightest rotated rectangle
  approxPolyDP     how many corners the outline has (Douglas-Peucker): box-like, round, irregular
  convex hull      solidity (1.0 = convex), convexity defects (a mug handle is a deep defect)
  fitEllipse       axes and angle: a second opinion on elongation / orientation for round-ish objects
  Hough lines      dominant straight-edge directions inside the box (HoughLinesP on Canny edges)
  width profile    the extent of the outline projected on 12 directions = the span the fingers need per angle
  contact patch    how long the flat part of each side is at each angle: a flat side holds, a rounded end slips
                   (the mouse is gripped across its straight sides; along its length the pads only touch the tips)
  edge support     how well the mask boundary agrees with image edges (low = untrustworthy outline, e.g. glass)

Everything is in PIXELS; px_to_m converts with the camera distance to the object's top (camera looking down).
"""
import math

import cv2
import numpy as np

PROFILE_STEP_DEG = 15
PROFILE_ANGLES = tuple(range(0, 180, PROFILE_STEP_DEG))      # direction of the closing axis in the image
CONTACT_BAND_FRACTION = 0.06      # points within this fraction of the width of a supporting line count as touching it


# ── outlines ───────────────────────────────────────────────────────────────────────────────────────────────────────
def mask_from_polygon(poly, shape):
    """Filled uint8 mask (0/255) of a polygon [(x, y), ...] for an image of `shape` (h, w[, c])."""
    m = np.zeros(shape[:2], np.uint8)
    pts = np.asarray(poly, np.float32).reshape(-1, 1, 2).round().astype(np.int32)
    if len(pts) >= 3:
        cv2.fillPoly(m, [pts], 255)
    return m


def refine_mask_grabcut(bgr, box, init_mask=None, iterations=3, pad=12):
    """GrabCut inside the (padded) box. With an initial mask, its eroded core is 'sure foreground' and everything
    outside its dilation is 'sure background'; without one the box itself seeds the segmentation. Returns a 0/255
    mask the size of the image, or None if GrabCut fails or returns nothing sensible."""
    h, w = bgr.shape[:2]
    x1, y1, x2, y2 = [int(v) for v in box]
    x1, y1, x2, y2 = max(0, x1 - pad), max(0, y1 - pad), min(w - 1, x2 + pad), min(h - 1, y2 + pad)
    if x2 - x1 < 8 or y2 - y1 < 8:
        return None
    roi = bgr[y1:y2 + 1, x1:x2 + 1]
    gc = np.full(roi.shape[:2], cv2.GC_PR_BGD, np.uint8)
    if init_mask is not None:
        m = init_mask[y1:y2 + 1, x1:x2 + 1] > 0
        k = np.ones((5, 5), np.uint8)
        sure = cv2.erode(m.astype(np.uint8), k, iterations=2) > 0
        near = cv2.dilate(m.astype(np.uint8), k, iterations=3) > 0
        gc[m] = cv2.GC_PR_FGD
        gc[sure] = cv2.GC_FGD
        gc[~near] = cv2.GC_BGD
    else:
        gc[:] = cv2.GC_PR_BGD
        inner = (slice(pad // 2, roi.shape[0] - pad // 2), slice(pad // 2, roi.shape[1] - pad // 2))
        gc[inner] = cv2.GC_PR_FGD
    bg, fg = np.zeros((1, 65), np.float64), np.zeros((1, 65), np.float64)
    try:
        cv2.grabCut(roi, gc, None, bg, fg, iterations, cv2.GC_INIT_WITH_MASK)
    except cv2.error:
        return None
    out = np.zeros((h, w), np.uint8)
    sel = (gc == cv2.GC_FGD) | (gc == cv2.GC_PR_FGD)
    out[y1:y2 + 1, x1:x2 + 1][sel] = 255
    return out if out.any() else None


def largest_contour(mask):
    """Largest external contour of a 0/255 mask as an (N, 2) int array, or None."""
    cnts, _ = cv2.findContours((mask > 0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not cnts:
        return None
    c = max(cnts, key=cv2.contourArea)
    return None if cv2.contourArea(c) < 30 else c.reshape(-1, 2)


def edge_map(bgr, box=None, sigma=0.33):
    """Canny edges with thresholds from the median brightness (auto Canny), blurred first so wood grain and noise do
    not dominate. Restricted to the box when given."""
    g = cv2.GaussianBlur(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY), (5, 5), 0)
    v = float(np.median(g))
    e = cv2.Canny(g, int(max(0, (1 - sigma) * v)), int(min(255, (1 + sigma) * v)))
    if box is not None:
        keep = np.zeros_like(e)
        x1, y1, x2, y2 = [int(t) for t in box]
        keep[max(0, y1):y2 + 1, max(0, x1):x2 + 1] = 255
        e = cv2.bitwise_and(e, keep)
    return e


def edge_support(contour, bgr, box=None, percentile=70.0, radius=4):
    """Fraction of the outline that sits on a real image edge, in [0, 1]. The gradient magnitude (Sobel on a blurred
    grey image) is compared with the `percentile` of the gradient inside the box; an outline point counts when the
    strongest gradient within `radius` px of it passes that. A crisp object has most of its mask boundary on
    edges; a blurred mask, or a transparent bottle whose outline is not where the glass edge is, does not."""
    if contour is None or len(contour) < 5:
        return 0.0
    g = cv2.GaussianBlur(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY), (0, 0), 1.6).astype(np.float32)
    mag = cv2.magnitude(cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3), cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3))
    h, w = mag.shape
    if box is not None:
        x1, y1, x2, y2 = [int(t) for t in box]
        region = mag[max(0, y1):min(h, y2 + 1), max(0, x1):min(w, x2 + 1)]
    else:
        region = mag
    thr = float(np.percentile(region, percentile)) if region.size else 0.0
    near = cv2.dilate(mag, np.ones((2 * radius + 1, 2 * radius + 1), np.uint8))
    pts = np.asarray(contour)
    vals = near[np.clip(pts[:, 1], 0, h - 1), np.clip(pts[:, 0], 0, w - 1)]
    return float((vals >= thr).mean())


def edge_contour(bgr, box):
    """Contour found from edges alone (Canny -> close gaps -> fill -> largest blob around the box centre): the
    fallback when there is no mask, and the second opinion when there is."""
    e = edge_map(bgr, box)
    e = cv2.morphologyEx(e, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8), iterations=2)
    cnts, _ = cv2.findContours(e, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not cnts:
        return None
    cx, cy = (box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0
    best, best_a = None, 0.0
    for c in cnts:
        a = cv2.contourArea(c)
        if a > best_a and cv2.pointPolygonTest(c, (float(cx), float(cy)), False) >= 0:
            best, best_a = c, a
    return None if best is None or best_a < 100 else best.reshape(-1, 2)


# ── shape descriptors ──────────────────────────────────────────────────────────────────────────────────────────────
def width_profile(contour, angles=PROFILE_ANGLES):
    """Extent (px) of the outline along each direction in `angles` (degrees from image x, y down): the span the
    fingers must open to when they close along that direction. {angle: width_px}."""
    pts = np.asarray(contour, float)
    out = {}
    for a in angles:
        t = math.radians(a)
        proj = pts @ np.array([math.cos(t), math.sin(t)])
        out[a] = float(proj.max() - proj.min())
    return out


def contact_profile(contour, angles=PROFILE_ANGLES, band=CONTACT_BAND_FRACTION):
    """For each closing direction, the length (px) of outline that lies within a thin band of BOTH supporting lines,
    taking the shorter of the two sides. Long = both pads land on flat surface; short = pads touch only a tip or
    a curve (the object tends to squirt out)."""
    pts = np.asarray(contour, float)
    out = {}
    # one band for every direction, from the narrowest span: a band proportional to each direction's own width
    # would make the long axis look flatter than it is
    tol = max(1.5, band * min(width_profile(contour, angles).values()))
    for a in angles:
        t = math.radians(a)
        u = np.array([math.cos(t), math.sin(t)])           # closing axis
        v = np.array([-math.sin(t), math.cos(t)])          # along the contact surface
        pu, pv = pts @ u, pts @ v
        lo, hi = pv[pu <= pu.min() + tol], pv[pu >= pu.max() - tol]
        side = lambda s: float(s.max() - s.min()) if len(s) >= 2 else 0.0
        out[a] = min(side(lo), side(hi))
    return out


def convexity_defects(contour, min_depth_px=6.0):
    """Deep concavities of the outline: [(depth_px, (x, y))] largest first. A mug handle shows as a deep defect."""
    c = np.asarray(contour, np.int32).reshape(-1, 1, 2)
    if len(c) < 10:
        return []
    hull = cv2.convexHull(c, returnPoints=False)
    if hull is None or len(hull) < 4:
        return []
    try:
        d = cv2.convexityDefects(c, hull)
    except cv2.error:
        return []
    if d is None:
        return []
    out = [(float(depth) / 256.0, (int(c[f][0][0]), int(c[f][0][1]))) for _, _, f, depth in d[:, 0]
           if depth / 256.0 >= min_depth_px]
    return sorted(out, reverse=True)


def enclosed_holes(mask, min_area_px=60):
    """Holes inside the silhouette (background seen through the object): [(area_px, (cx, cy))] largest first. A closed
    mug handle seen from above is a hole next to the body; the external contour alone never shows it."""
    cnts, hier = cv2.findContours((mask > 0).astype(np.uint8), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    out = []
    if hier is None:
        return out
    for c, hh in zip(cnts, hier[0]):
        if hh[3] != -1:                                    # has a parent: it is a hole
            a = cv2.contourArea(c)
            if a >= min_area_px:
                m = cv2.moments(c)
                if m['m00'] > 0:
                    out.append((float(a), (m['m10'] / m['m00'], m['m01'] / m['m00'])))
    return sorted(out, reverse=True)


def hough_dominant_angles(edges, n=3, min_len=25, tol_deg=8.0):
    """Directions (degrees in [0, 180)) of the longest straight edges (HoughLinesP), merged within tol_deg, longest
    first: [(angle_deg, total_length_px)]."""
    lines = cv2.HoughLinesP(edges, 1, math.pi / 180, threshold=25, minLineLength=min_len, maxLineGap=6)
    if lines is None:
        return []
    bins = []                                              # [angle, total length]
    for x1, y1, x2, y2 in lines[:, 0]:
        ang = math.degrees(math.atan2(y2 - y1, x2 - x1)) % 180.0
        ln = math.hypot(x2 - x1, y2 - y1)
        for b in bins:
            d = abs(b[0] - ang)
            d = min(d, 180.0 - d)
            if d <= tol_deg:
                b[0] = (b[0] * b[1] + ang * ln) / (b[1] + ln) if abs(b[0] - ang) < 90 else b[0]
                b[1] += ln
                break
        else:
            bins.append([ang, ln])
    bins.sort(key=lambda b: -b[1])
    return [(float(a), float(l)) for a, l in bins[:n]]


def classify_shape(circularity, solidity, vertices, aspect, defects):
    """round / box / elongated / handled / irregular, from the descriptors."""
    if circularity > 0.82 and aspect < 1.2 and solidity > 0.9:
        return 'round'
    if solidity < 0.80:
        return 'irregular'
    if vertices <= 6 and solidity > 0.9 and aspect >= 1.3:
        return 'box'
    if aspect >= 1.3:
        return 'elongated'
    return 'compact'


def analyse(bgr, box, polygon=None, use_grabcut=True):
    """Full analysis of one object. `box` = (x1, y1, x2, y2) px; `polygon` = segmentation polygon or None.
    Returns a JSON-friendly dict, or None when no outline can be found. Keys:
      shape, rect {cx, cy, w, h, angle} (w <= h), area_px, solidity, circularity, vertices, aspect,
      ellipse {major, minor, angle} | None, profile {angle: width_px}, contact {angle: px}, short_axis_deg,
      long_axis_deg, hough [(angle, len)], handle {depth_px, x, y, bearing_deg} | None, edge_support, source."""
    h, w = bgr.shape[:2]
    mask, source = None, None
    if polygon is not None and len(polygon) >= 3:
        mask, source = mask_from_polygon(polygon, bgr.shape), 'mask'
        if use_grabcut:
            g = refine_mask_grabcut(bgr, box, mask)
            if g is not None and 0.6 < g.sum() / max(1, mask.sum()) < 1.6:
                mask, source = g, 'mask+grabcut'
    if mask is None and use_grabcut:
        g = refine_mask_grabcut(bgr, box)
        if g is not None:
            mask, source = g, 'grabcut'
    cnt = largest_contour(mask) if mask is not None else None
    if cnt is None:
        cnt, source = edge_contour(bgr, box), 'edges'
    if cnt is None or len(cnt) < 8:
        return None

    area = float(cv2.contourArea(cnt.reshape(-1, 1, 2)))
    peri = float(cv2.arcLength(cnt.reshape(-1, 1, 2), True))
    hull = cv2.convexHull(cnt.reshape(-1, 1, 2))
    hull_area = float(cv2.contourArea(hull))
    solidity = area / hull_area if hull_area > 1 else 1.0
    circularity = 4 * math.pi * area / (peri * peri) if peri > 0 else 0.0
    poly = cv2.approxPolyDP(cnt.reshape(-1, 1, 2), 0.02 * peri, True)
    (rcx, rcy), (rw, rh), rang = cv2.minAreaRect(cnt.reshape(-1, 1, 2))
    if rw > rh:                                            # normalise: w = short side, h = long side
        rw, rh, rang = rh, rw, rang + 90.0
    long_axis = rang + 90.0                                # direction of the long side, degrees
    long_axis %= 180.0
    short_axis = (long_axis + 90.0) % 180.0
    aspect = rh / max(rw, 1e-6)
    ellipse = None
    if len(cnt) >= 5:
        (ex, ey), (ea, eb), eang = cv2.fitEllipse(cnt.reshape(-1, 1, 2))
        ellipse = {'major': float(max(ea, eb)), 'minor': float(min(ea, eb)), 'angle': float(eang % 180.0)}
    defects = convexity_defects(cnt, min_depth_px=0.10 * max(rw, 1.0))
    handle = None
    holes = enclosed_holes(mask) if mask is not None else []
    if holes:                                              # a closed loop (handle) beside the body
        ha, (hx, hy) = holes[0]
        handle = {'depth_px': round(math.sqrt(ha), 1), 'x': round(hx), 'y': round(hy), 'kind': 'hole',
                  'bearing_deg': round(math.degrees(math.atan2(hy - rcy, hx - rcx)) % 360.0, 1)}
    elif defects:                                          # an open handle / a notch in the outline
        depth, (hx, hy) = defects[0]
        handle = {'depth_px': round(depth, 1), 'x': hx, 'y': hy, 'kind': 'notch',
                  'bearing_deg': round(math.degrees(math.atan2(hy - rcy, hx - rcx)) % 360.0, 1)}
    edges = edge_map(bgr, box)
    shape = classify_shape(circularity, solidity, len(poly), aspect, defects)
    if handle is not None and shape in ('round', 'compact'):
        shape = 'handled'
    return {
        'shape': shape, 'source': source,
        'rect': {'cx': round(rcx, 1), 'cy': round(rcy, 1), 'w': round(rw, 1), 'h': round(rh, 1), 'angle': round(rang % 180.0, 1)},
        'area_px': round(area, 1), 'solidity': round(solidity, 3), 'circularity': round(circularity, 3),
        'vertices': int(len(poly)), 'aspect': round(aspect, 2), 'ellipse': ellipse,
        'profile': {a: round(v, 1) for a, v in width_profile(cnt).items()},
        'contact': {a: round(v, 1) for a, v in contact_profile(cnt).items()},
        'short_axis_deg': round(short_axis, 1), 'long_axis_deg': round(long_axis, 1),
        'hough': [(round(a, 1), round(l, 1)) for a, l in hough_dominant_angles(edges)],
        'handle': handle, 'edge_support': round(edge_support(cnt, bgr, box), 3),
    }


# ── metric conversion ──────────────────────────────────────────────────────────────────────────────────────────────
def px_to_m(px, distance_m, fx):
    """Length in metres of `px` image pixels at `distance_m` from a camera of focal length fx (looking straight at
    the object's top, so the object plane is perpendicular to the optical axis)."""
    return px * distance_m / fx


def widths_m(profile_px, distance_m, fx):
    return {a: px_to_m(v, distance_m, fx) for a, v in profile_px.items()}
