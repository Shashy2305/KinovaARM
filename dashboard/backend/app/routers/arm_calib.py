from typing import Literal

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from .. import ros_bridge

router = APIRouter()
Camera = Literal['oakd', 'realsense', 'realsense2']


class Click(BaseModel):
    camera: Camera
    landmark: str
    u: float
    v: float


def _meta(camera):
    frame_id, size = ros_bridge.get_bridge().get_image_meta(camera)
    if not frame_id:
        raise HTTPException(503, f'No image from {camera} yet')
    return frame_id, size


@router.get('/arm_calib/state')
def state(camera: Camera):
    frame_id, size = _meta(camera)
    return ros_bridge.get_or_create_arm_calib().state(camera, frame_id, size)


@router.post('/arm_calib/click')
def click(req: Click):
    ok, msg = ros_bridge.get_or_create_arm_calib().click(req.camera, req.landmark, req.u, req.v)
    if not ok:
        raise HTTPException(400, msg)
    return {'ok': True}


@router.post('/arm_calib/reset')
def reset(camera: Camera):
    ros_bridge.get_or_create_arm_calib().reset(camera)
    return {'ok': True}


@router.post('/arm_calib/solve')
def solve(camera: Camera):
    frame_id, _ = _meta(camera)
    return ros_bridge.get_or_create_arm_calib().solve(camera, frame_id)


@router.post('/arm_calib/save_candidate')
def save_candidate(camera: Camera):
    ok, msg, path = ros_bridge.get_or_create_arm_calib().save_candidate(camera)
    return {'ok': ok, 'message': msg, 'path': path}
