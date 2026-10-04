from fastapi import APIRouter

from thesis_robot import safety_geometry as sg

from .. import ros_bridge

router = APIRouter()


@router.get('/table_geometry/status')
def status():
    rec = ros_bridge.get_or_create_table_recorder()
    pose, msg = rec.current_pose()
    geom, err = sg.load_geometry()
    return {
        'pose': pose,
        'pose_message': msg,
        'points': rec.snapshot(),
        'saved': geom,
        'saved_message': None if geom else err,
        'flange_floor_z': sg.flange_floor_z(geom),
    }


@router.post('/table_geometry/record')
def record():
    ok, msg = ros_bridge.get_or_create_table_recorder().record()
    return {'ok': ok, 'message': msg}


@router.post('/table_geometry/reset')
def reset():
    ros_bridge.get_or_create_table_recorder().reset()
    return {'ok': True}


@router.post('/table_geometry/save')
def save():
    ok, msg, geom = ros_bridge.get_or_create_table_recorder().save()
    return {'ok': ok, 'message': msg, 'saved': geom}
