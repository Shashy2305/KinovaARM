"""
Which static cameras have a calibration file, and may therefore be trusted to place things in base_link.

A static transform never expires in a running node: after a camera's calibration is removed (or a trial pose was installed and taken out again), the
obstacle guard and the scene graph kept using the last pose they had seen. On 2026-10-20 that drew a phantom 25 x 55 cm "unknown obstacle" on the scene map
from RealSense 3's depth. A camera is only used while its calibration file exists, whatever transform a node still remembers.
"""
import os

CALIBRATION_FILES = {
    'realsense': '~/.ros/realsense_calibration.yaml',
    'realsense2': '~/.ros/realsense2_calibration.yaml',
    'realsense3': '~/.ros/realsense3_calibration.yaml',
    'oakd': '~/.ros/oakd_calibration.yaml',
}


def has_calibration(camera):
    """True when `camera` has a calibration file, or is not a static camera that needs one (the wrist camera is calibrated by the arm)."""
    path = CALIBRATION_FILES.get(camera)
    return path is None or os.path.exists(os.path.expanduser(path))
