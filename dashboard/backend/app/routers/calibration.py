import os
import subprocess
import sys

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel

from .. import config, ros_bridge

router = APIRouter()

REPO_ROOT = os.path.expanduser('~/Shashproject')
BOARD_PNG = os.path.join(REPO_ROOT, 'calibration', 'charuco_board_dashboard.png')


@router.post('/calibration/generate_board')
def generate_board():
    result = subprocess.run(
        [sys.executable, os.path.join(REPO_ROOT, 'calibration', 'generate_charuco_board.py'),
         '--out', BOARD_PNG],
        capture_output=True, text=True, timeout=30,
    )
    if result.returncode != 0:
        raise HTTPException(500, f'generate_charuco_board.py failed: {result.stderr[-500:]}')
    return {'ok': True, 'message': result.stdout, 'download_url': '/api/calibration/board.png'}


@router.get('/calibration/board.png')
def board_png():
    if not os.path.exists(BOARD_PNG):
        raise HTTPException(404, 'Board not generated yet — call generate_board first.')
    return FileResponse(BOARD_PNG, media_type='image/png')


CALIBRATION_TARGETS = ('oakd', 'realsense', 'realsense2')


@router.get('/calibration/board_visible/{camera_name}')
def board_visible(camera_name: str):
    """Live status for the wizard: does the wrist camera AND camera_name
    both see the board right now? Non-blocking, safe to poll."""
    if camera_name not in CALIBRATION_TARGETS:
        raise HTTPException(404, f'{camera_name!r} is not a calibration target (only {CALIBRATION_TARGETS})')
    calibrator = ros_bridge.get_or_create_calibrator()
    return {
        'wrist_ready': calibrator.wrist.ready,
        'camera_ready': calibrator.cams.get(camera_name).ready if camera_name in calibrator.cams else False,
        'board_visible': calibrator.board_visible(camera_name),
    }


@router.post('/calibration/capture/{camera_name}')
def capture(camera_name: str):
    if camera_name not in CALIBRATION_TARGETS:
        raise HTTPException(404, f'{camera_name!r} is not a calibration target (only {CALIBRATION_TARGETS})')
    calibrator = ros_bridge.get_or_create_calibrator()
    ok, msg, result = calibrator.capture_camera(camera_name, timeout_sec=3.0)
    return {'ok': ok, 'message': msg, 'result': result}


BOUNDARY_CAMERAS = ('oakd', 'realsense', 'wrist')


class ClickRequest(BaseModel):
    camera: str
    u: int
    v: int


def _boundary_tool_or_404(camera: str):
    if camera not in BOUNDARY_CAMERAS:
        raise HTTPException(404, f'{camera!r} is not a valid camera — choose from {BOUNDARY_CAMERAS}')
    return ros_bridge.get_or_create_boundary_tool(camera)


@router.post('/workspace_boundary/click')
def boundary_click(req: ClickRequest):
    tool = _boundary_tool_or_404(req.camera)
    tool.click(req.u, req.v)
    return {
        'ok': True,
        'status': tool.last_click_status,
        'num_points': len(tool.clicked_base_xyz),
    }


@router.post('/workspace_boundary/reset')
def boundary_reset(camera: str = 'oakd'):
    tool = _boundary_tool_or_404(camera)
    tool.clicked_base_xyz.clear()
    tool.clicked_pixels.clear()
    tool.last_click_status = 'reset'
    return {'ok': True}


@router.post('/workspace_boundary/save')
def boundary_save(camera: str = 'oakd'):
    tool = _boundary_tool_or_404(camera)
    ok = tool.compute_and_save()
    return {'ok': ok, 'status': tool.last_click_status}


@router.get('/workspace_boundary/current')
def boundary_current():
    path = os.path.expanduser('~/.ros/workspace_bounds.yaml')
    if not os.path.exists(path):
        return {'exists': False}
    import yaml
    with open(path) as f:
        data = yaml.safe_load(f)
    return {'exists': True, **data}
