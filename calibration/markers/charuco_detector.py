#!/usr/bin/env python3
"""
Shared ChArUco board setup and pose detection, targeting the current
cv2.aruco API (cv2.aruco.CharucoDetector / CharucoBoard(size, ...)).

The legacy calls used elsewhere in this repo — CharucoBoard_create(),
DetectorParameters_create(), used by handeye_calibration.py and the older
scripts in this directory — were removed in OpenCV >= 4.7. This machine
runs OpenCV 4.13, where those calls raise AttributeError. New code should
import from here rather than copy the legacy pattern.

Board spec matches handeye_calibration.py's physical board (same printed
board can be reused): 5x7 squares, 40mm squares, 20mm markers, DICT_6X6_250.
"""
import cv2

SQUARES_X = 5
SQUARES_Y = 7
SQUARE_M = 0.040
MARKER_M = 0.020
ARUCO_DICT = cv2.aruco.DICT_6X6_250
MIN_CHARUCO_CORNERS = 6


def make_board_and_detector():
    dictionary = cv2.aruco.getPredefinedDictionary(ARUCO_DICT)
    board = cv2.aruco.CharucoBoard((SQUARES_X, SQUARES_Y), SQUARE_M, MARKER_M, dictionary)
    detector = cv2.aruco.CharucoDetector(
        board, cv2.aruco.CharucoParameters(), cv2.aruco.DetectorParameters())
    return board, detector


def detect_pose(detector, board, gray, K, dist):
    """Return (rvec, tvec): the board's pose in the camera frame, or None
    if the board isn't confidently visible in this image."""
    charuco_corners, charuco_ids, _, _ = detector.detectBoard(gray)
    if charuco_ids is None or len(charuco_ids) < MIN_CHARUCO_CORNERS:
        return None

    obj_points, img_points = board.matchImagePoints(charuco_corners, charuco_ids)
    if obj_points is None or len(obj_points) < MIN_CHARUCO_CORNERS:
        return None

    ok, rvec, tvec = cv2.solvePnP(obj_points, img_points, K, dist)
    if not ok:
        return None
    return rvec, tvec
