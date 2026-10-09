import os
import subprocess
import time

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from .. import config, process_manager

router = APIRouter()


@router.get('/nodes')
def list_nodes():
    return process_manager.manager.all_status()


@router.post('/nodes/{proc_id}/start')
def start_node(proc_id: str):
    if proc_id not in config.PROCESSES:
        raise HTTPException(404, f'Unknown process {proc_id!r}')
    if proc_id == 'cameras_bringup':
        steps = process_manager.manager.start_cameras()
        return {'ok': all(s['ok'] for s in steps), 'steps': steps}
    ok, msg = process_manager.manager.start(proc_id)
    return {'ok': ok, 'message': msg}


@router.post('/nodes/{proc_id}/stop')
def stop_node(proc_id: str):
    if proc_id not in config.PROCESSES:
        raise HTTPException(404, f'Unknown process {proc_id!r}')
    ok, msg = process_manager.manager.stop(proc_id)
    return {'ok': ok, 'message': msg}


@router.get('/nodes/{proc_id}/log')
def node_log(proc_id: str, lines: int = 200):
    if proc_id not in config.PROCESSES:
        raise HTTPException(404, f'Unknown process {proc_id!r}')
    return {'log': process_manager.manager.tail_log(proc_id, lines)}


@router.post('/bringup/full')
def full_bringup():
    """Runs config.FULL_BRINGUP_ORDER in sequence. Robot/camera bringup are
    deliberately NOT included -- those need a human confirming robot_ip and
    watching the E-stop, not a background button."""
    steps = []
    skip = config.disabled_camera_procs()
    for proc_id, settle_sec in config.FULL_BRINGUP_ORDER:
        if proc_id in skip:
            steps.append({'proc_id': proc_id, 'ok': True, 'message': 'skipped: its camera is listed in config/disabled_cameras.txt'})
            continue
        ok, msg = process_manager.manager.start(proc_id)
        steps.append({'proc_id': proc_id, 'ok': ok, 'message': msg})
        time.sleep(settle_sec)
    return {'ok': all(s['ok'] for s in steps), 'steps': steps}


class GoLiveRequest(BaseModel):
    confirm_phrase: str


@router.post('/arm_controller/go_live')
def arm_controller_go_live(req: GoLiveRequest):
    """Two-step confirmation by design (see plan) -- restarts arm_controller
    with dry_run:=false. Requires the exact phrase, not a toggle flip, since
    this dashboard is reachable off the lab PC."""
    if req.confirm_phrase != 'MAKE IT LIVE':
        raise HTTPException(
            400, 'confirm_phrase must be exactly "MAKE IT LIVE" — this is '
            'deliberately not a simple toggle, see dashboard/backend README.')

    process_manager.manager.stop('arm_controller')
    time.sleep(1.0)
    cmd = (
        f'{config.ROS_ENV_CMD} ros2 run thesis_robot arm_controller '
        f'--ros-args -p dry_run:=false -p speed:=0.10'
    )
    log_f = open(os.path.join(config.LOG_DIR, 'arm_controller.log'), 'w')
    popen = subprocess.Popen(
        ['bash', '-c', cmd], stdout=log_f, stderr=subprocess.STDOUT,
        preexec_fn=os.setsid)
    process_manager.manager._procs['arm_controller'].popen = popen
    return {'ok': True, 'message': f'arm_controller restarted LIVE (pid {popen.pid}), speed=0.10.'}


@router.post('/arm_controller/go_dry_run')
def arm_controller_go_dry_run():
    process_manager.manager.stop('arm_controller')
    time.sleep(1.0)
    ok, msg = process_manager.manager.start('arm_controller')
    return {'ok': ok, 'message': msg}
