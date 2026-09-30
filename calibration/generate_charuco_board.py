#!/usr/bin/env python3
"""
generate_charuco_board.py — renders the ChArUco board used by
handeye_calibration.py, multi_camera_calibrate.py and multi_camera_view.py
(calibration/markers/charuco_detector.py) to a PNG sized for accurate
printing.

IMPORTANT: print at 100% / "actual size" — do NOT let the print dialog
"fit to page" or "scale to fit", or the squares won't be 40mm and every
pose the board gives will be wrong. After printing, measure one square
with a ruler to confirm it's 40mm before using the board.

Usage:
  python3 calibration/generate_charuco_board.py [--dpi 300] [--out board.png]
"""
import argparse
import os
import sys

import cv2

sys.path.insert(0, os.path.dirname(__file__))
from markers.charuco_detector import SQUARES_X, SQUARES_Y, SQUARE_M, make_board_and_detector  # noqa: E402

MM_PER_M = 1000.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dpi', type=int, default=300)
    ap.add_argument('--out', default=os.path.join(os.path.dirname(__file__), 'charuco_board.png'))
    args = ap.parse_args()

    px_per_mm = args.dpi / 25.4
    square_mm = SQUARE_M * MM_PER_M
    width_px = round(SQUARES_X * square_mm * px_per_mm)
    height_px = round(SQUARES_Y * square_mm * px_per_mm)

    board, _ = make_board_and_detector()
    img = board.generateImage((width_px, height_px), marginSize=0, borderBits=1)
    cv2.imwrite(args.out, img)

    print(f'Wrote {args.out}  ({width_px}x{height_px}px at {args.dpi} DPI)')
    print(f'Physical size once printed at 100%: '
          f'{SQUARES_X * square_mm:.0f}mm x {SQUARES_Y * square_mm:.0f}mm '
          f'({SQUARES_X} x {SQUARES_Y} squares, {square_mm:.0f}mm each)')
    print('Print at "actual size" / 100% — NOT "fit to page".')
    print('After printing, measure one square with a ruler: it must be '
          f'{square_mm:.0f}mm, or every calibration pose will be wrong.')


if __name__ == '__main__':
    main()
