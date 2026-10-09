import math

import numpy as np
import pytest

cv2 = pytest.importorskip('cv2')
from thesis_robot import shape_analysis as sa  # noqa: E402

BG = (70, 90, 150)          # a reddish wood-like BGR
OBJ = (200, 190, 170)


def background(h=360, w=480, seed=0):
    rng = np.random.default_rng(seed)
    img = np.empty((h, w, 3), np.uint8)
    img[:] = BG
    noise = rng.normal(0, 6, (h, w, 1))
    grain = 8 * np.sin(np.arange(w) / 3.0)[None, :, None]            # wood grain-ish streaks
    return np.clip(img + noise + grain, 0, 255).astype(np.uint8)


def capsule(cx, cy, length, width, angle_deg):
    """Polygon of a stadium (straight sides, semicircular ends); a mouse seen from above."""
    r = width / 2.0
    half = length / 2.0 - r
    pts = []
    for t in np.linspace(-90, 90, 24):
        pts.append((half + r * math.cos(math.radians(t)), r * math.sin(math.radians(t))))
    for t in np.linspace(90, 270, 24):
        pts.append((-half + r * math.cos(math.radians(t)), r * math.sin(math.radians(t))))
    a = math.radians(angle_deg)
    R = np.array([[math.cos(a), -math.sin(a)], [math.sin(a), math.cos(a)]])
    return (np.array(pts) @ R.T) + np.array([cx, cy])


def draw_poly(img, poly, color=OBJ):
    cv2.fillPoly(img, [poly.round().astype(np.int32)], color)


def box_of(poly, pad=6):
    return (poly[:, 0].min() - pad, poly[:, 1].min() - pad, poly[:, 0].max() + pad, poly[:, 1].max() + pad)


def test_capsule_rectangle_angle_and_sizes():
    img = background()
    poly = capsule(240, 180, 140, 60, 30)
    draw_poly(img, poly)
    a = sa.analyse(img, box_of(poly), polygon=poly)
    assert a is not None
    assert abs(a['rect']['w'] - 60) < 5 and abs(a['rect']['h'] - 140) < 7
    d = abs(a['long_axis_deg'] - 30)
    assert min(d, 180 - d) < 4
    assert a['shape'] in ('elongated', 'box')
    assert a['aspect'] > 2.0


def test_width_profile_is_smallest_across_the_short_side():
    img = background()
    poly = capsule(240, 180, 140, 60, 30)
    draw_poly(img, poly)
    a = sa.analyse(img, box_of(poly), polygon=poly)
    prof = a['profile']
    best = min(prof, key=prof.get)                                  # closing direction with the narrowest span
    short = a['short_axis_deg']
    assert min(abs(best - short), 180 - abs(best - short)) <= sa.PROFILE_STEP_DEG
    assert prof[best] < 70 and max(prof.values()) > 130


def test_flat_sides_beat_rounded_ends_for_contact():
    img = background()
    poly = capsule(240, 180, 140, 60, 0)                           # long axis along image x
    draw_poly(img, poly)
    a = sa.analyse(img, box_of(poly), polygon=poly)
    across_short = a['contact'][90]                                 # fingers close along y: touch the long flat sides
    along_long = a['contact'][0]                                    # fingers close along x: touch the round ends
    assert across_short > 3 * along_long


def test_circle_is_round_with_constant_profile():
    img = background()
    cv2.circle(img, (240, 180), 45, OBJ, -1)
    a = sa.analyse(img, (190, 130, 290, 230), polygon=None)
    assert a is not None and a['shape'] == 'round' and a['circularity'] > 0.85
    vals = list(a['profile'].values())
    assert max(vals) - min(vals) < 8


def test_mug_handle_is_found_on_the_right_side():
    img = background()
    cv2.circle(img, (220, 180), 48, OBJ, -1)
    cv2.ellipse(img, (240, 180), (62, 38), 0, -90, 90, OBJ, 10)    # a C-shaped handle on the right, attached to the body
    box = (160, 120, 320, 240)
    a = sa.analyse(img, box, polygon=None)
    assert a is not None and a['handle'] is not None
    b = a['handle']['bearing_deg']
    assert min(b, 360 - b) < 50                                     # roughly to the right of the centre (bearing ~0)


def test_hough_finds_the_edges_of_a_rotated_box():
    img = background()
    poly = capsule(240, 180, 150, 90, 20)
    box_poly = np.array([[-75, -45], [75, -45], [75, 45], [-75, 45]], float)
    a_ = math.radians(20)
    R = np.array([[math.cos(a_), -math.sin(a_)], [math.sin(a_), math.cos(a_)]])
    pts = box_poly @ R.T + np.array([240, 180])
    draw_poly(img, pts)
    edges = sa.edge_map(img, box_of(pts))
    angs = [a for a, _ in sa.hough_dominant_angles(edges, n=4)]
    def near(target):
        return any(min(abs(a - target), 180 - abs(a - target)) < 6 for a in angs)
    assert near(20) and near(110)


def test_edge_support_is_high_for_a_crisp_outline_and_low_for_a_bad_one():
    img = background()
    poly = capsule(240, 180, 140, 60, 30)
    draw_poly(img, poly)
    good = sa.largest_contour(sa.mask_from_polygon(poly, img.shape))
    shifted = ((good - np.array([240, 180])) * 0.5 + np.array([240, 180])).astype(int)   # an outline inside the object
    box = box_of(poly)
    assert sa.edge_support(good, img, box) > 0.8
    assert sa.edge_support(shifted, img, box) < 0.5


def test_grabcut_refines_a_loose_mask():
    img = background()
    poly = capsule(240, 180, 140, 60, 30)
    draw_poly(img, poly)
    loose = capsule(240, 180, 150, 72, 30)                           # a sloppy segmentation, a bit too big
    a = sa.analyse(img, box_of(loose), polygon=loose)
    assert abs(a['rect']['w'] - 60) < 6 and abs(a['rect']['h'] - 140) < 8


def test_no_object_returns_none():
    assert sa.analyse(background(), (100, 100, 160, 160), polygon=None, use_grabcut=False) is None


def test_px_to_m_scale():
    assert sa.px_to_m(130, 0.26, 1300.0) == pytest.approx(0.026)
