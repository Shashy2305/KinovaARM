"""The wrist detector's rotated fallback: boxes and masks found in the turned image must land back on the right pixels."""
import cv2
import numpy as np

from thesis_robot.wrist_detection import unrotate_cw90


def test_points_round_trip_through_a_clockwise_quarter_turn():
    H, W = 48, 80
    img = np.zeros((H, W), np.uint8)
    pts = [(5, 7), (70, 3), (40, 40), (0, 0), (W - 1, H - 1)]
    for x, y in pts:
        img[y, x] = 255
    turned = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    ys, xs = np.nonzero(turned)
    back = unrotate_cw90(np.stack([xs, ys], axis=1), H)
    assert {(int(round(x)), int(round(y))) for x, y in back} == set(pts)


def test_a_box_is_mapped_by_its_two_corners():
    H, W = 60, 100
    img = np.zeros((H, W), np.uint8)
    img[10:30, 20:70] = 255                                   # x 20..69, y 10..29
    turned = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    ys, xs = np.nonzero(turned)
    q = unrotate_cw90([(xs.min(), ys.min()), (xs.max(), ys.max())], H)
    assert (q[:, 0].min(), q[:, 1].min(), q[:, 0].max(), q[:, 1].max()) == (20, 10, 69, 29)


def test_an_empty_polygon_does_not_crash():
    assert unrotate_cw90(np.zeros((0, 2)), 50).shape == (0, 2)
